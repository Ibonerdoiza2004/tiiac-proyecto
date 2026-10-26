#!/usr/bin/env python3
"""Evalúa con el protocolo común el experto, el agente aleatorio y cada BC y GAIL guardados.

Es el procedimiento reproducible de la evaluación final. Cada agente juega
evaluacion.episodios partidas con las semillas de evaluacion.semilla, las mismas
para todos, en el modo de evaluacion.determinista, y con la recompensa real del
entorno (evaluar_final):

    python scripts/evaluar.py --entorno cartpole --experto checkpoints/expertos/cartpole/<fecha> \\
        --bc checkpoints/bc/cartpole/<fecha> --gail checkpoints/gail/cartpole/<fecha>
    python scripts/evaluar.py --experto <dir> --gail <dir> --cantidades 4 --repeticiones 0

--experto, --bc y --gail son opcionales: se evalúa lo que se pase, y siempre el
agente aleatorio. Sin --cantidades ni --repeticiones se evalúan las de la
configuración (demostraciones.subconjuntos × experimento.repeticiones).

Antes de evaluar nada carga todos los modelos y comprueba que son del entorno
de la configuración y que los BC y GAIL se entrenaron con el mismo NPZ de
demostraciones (su huella). Guarda en un JSON nuevo (--salida, por defecto
checkpoints/evaluacion/<entorno>/<fecha>.json) el protocolo, la huella, las
rutas, el commit, las versiones, el hardware y, por agente, los retornos y
longitudes de cada partida. Imprime el retorno medio y su desviación de cada
uno, sin interpretarlos.

Códigos: 0 si todo va bien; 2 si la configuración, las rutas, los modelos o la
salida no son válidos, sin evaluar nada; 130 si se interrumpe con Ctrl+C (no
se guarda nada).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.training.bc import cargar_politica_bc  # noqa: E402
from src.training.evaluacion import config_evaluacion, evaluar_final  # noqa: E402
from src.training.gail import cargar_gail  # noqa: E402
from src.training.trpo import cargar_experto  # noqa: E402
from src.data.splits import huella_archivo  # noqa: E402
from src.utils.archivos import escribir_archivo_nuevo  # noqa: E402
from src.utils.configuracion import TipoEntorno, resolver_configuracion  # noqa: E402
from src.utils.procedencia import (a_json, avisar_estado_git, estado_git,  # noqa: E402
                                   ruta_registrada, versiones_y_hardware)
from src.utils.tracking import (abrir_run, comprobar_wandb, crear_grupo, registrar_resumen,  # noqa: E402
                                subir_artefacto, terminar_run, usar_artefacto)


def _cargar_agentes(cfg: dict, args: argparse.Namespace) -> tuple[list[dict[str, Any]],
                                                                  str | None]:
    """Carga los agentes que se van a evaluar y comprueba que encajan entre sí.

    Returns:
        Una entrada por agente (metodo, cantidad, repeticion, ruta, agente y
        estadísticas de normalización) y la huella del NPZ de los BC y GAIL (None
        si no hay ninguno).

    Raises:
        OSError: Si falta algún modelo o no se puede leer.
        ValueError: Si las cantidades o repeticiones no son válidas, algún modelo
            es de otro entorno o los BC y GAIL no se entrenaron con los mismos datos.
    """
    entorno = cfg["entorno"]["id"]
    cantidades = list(args.cantidades or cfg["demostraciones"]["subconjuntos"])
    total = cfg["experimento"]["repeticiones"]
    repeticiones = (list(args.repeticiones) if args.repeticiones is not None
                    else list(range(total)))
    if any(not 0 <= r < total for r in repeticiones):
        raise ValueError(f"repeticiones fuera de rango: {repeticiones}; van de 0 a {total - 1}")
    agentes: list[dict[str, Any]] = [{"metodo": "aleatorio", "cantidad": None,
                                      "repeticion": None, "ruta": None, "agente": "aleatorio",
                                      "estadisticas": None}]
    huellas: set[str] = set()

    def anotar(metodo: str, cantidad: int | None, repeticion: int | None, ruta: Path,
               agente: Any, estadisticas: dict[str, Any] | None, entorno_modelo: str) -> None:
        """Añade un agente cargado, después de comprobar su entorno."""
        if entorno_modelo != entorno:
            raise ValueError(f"{ruta} es de {entorno_modelo!r} y la configuración pide "
                             f"{entorno!r}")
        agentes.append({"metodo": metodo, "cantidad": cantidad, "repeticion": repeticion,
                        "ruta": ruta, "agente": agente, "estadisticas": estadisticas})

    if args.experto is not None:
        modelo, meta = cargar_experto(args.experto)
        anotar("experto", None, None, args.experto, modelo, meta["resumen"]["normalizacion"],
               meta["configuracion"]["entorno"]["id"])
    for cantidad in cantidades:
        for repeticion in repeticiones:
            if args.bc is not None:
                ruta = args.bc / f"cantidad_{cantidad}" / f"repeticion_{repeticion}.pt"
                politica, meta = cargar_politica_bc(ruta)
                huellas.add(meta["huella_dataset"])
                anotar("bc", cantidad, repeticion, ruta, politica, None, meta["entorno"])
            if args.gail is not None:
                ruta = args.gail / f"cantidad_{cantidad}" / f"repeticion_{repeticion}"
                agente, _, meta = cargar_gail(ruta)
                huellas.add(meta["datos"]["huella_dataset"])
                anotar("gail", cantidad, repeticion, ruta, agente,
                       meta["resumen"]["normalizacion"], meta["configuracion"]["entorno"]["id"])
    if len(huellas) > 1:
        raise ValueError("Los BC y GAIL no se entrenaron con las mismas demostraciones: "
                         f"huellas {sorted(huellas)}")
    return agentes, next(iter(huellas), None)


def main() -> int:
    """Carga los agentes, los evalúa con el protocolo común y guarda los resultados."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"),
                    help="configuración base")
    ap.add_argument("--entorno", choices=[e.value for e in TipoEntorno],
                    default=TipoEntorno.CARTPOLE.value, help="perfil de entorno")
    ap.add_argument("--experto", type=Path, default=None,
                    help="carpeta de entrenar_experto.py (modelo.zip y experto.json)")
    ap.add_argument("--bc", type=Path, default=None, help="carpeta de entrenar_bc.py")
    ap.add_argument("--gail", type=Path, default=None, help="carpeta de entrenar_gail.py")
    ap.add_argument("--cantidades", type=int, nargs="+", default=None,
                    help="tamaños de subconjunto (por defecto, demostraciones.subconjuntos)")
    ap.add_argument("--repeticiones", type=int, nargs="+", default=None,
                    help="repeticiones, de 0 a experimento.repeticiones − 1 (por defecto, todas)")
    ap.add_argument("--salida", type=Path, default=None,
                    help="archivo JSON nuevo (por defecto "
                         "checkpoints/evaluacion/<entorno>/<fecha>.json)")
    ap.add_argument("--sin-wandb", action="store_true", help="no registrar en W&B")
    ap.add_argument("--grupo", default=None, help="grupo W&B de esta ejecución completa")
    args = ap.parse_args()

    try:
        ajustes = {"wandb": {"modo": "disabled"}} if args.sin_wandb else None
        cfg = resolver_configuracion(args.config, args.entorno, ajustes)
        protocolo = config_evaluacion(cfg)
    except (OSError, ValueError) as error:
        print(f"error de configuración: {error}", file=sys.stderr)
        return 2
    salida = args.salida or (Path("checkpoints") / "evaluacion" / args.entorno
                             / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}.json")
    if salida.exists():
        print(f"error: {salida} ya existe; no se sobrescribe, elige otra --salida",
              file=sys.stderr)
        return 2
    try:
        agentes, huella = _cargar_agentes(cfg, args)
    except (OSError, ValueError) as error:
        print(f"error con los modelos: {error}", file=sys.stderr)
        return 2
    try:
        comprobar_wandb(cfg, exigir_sesion=True)
    except RuntimeError as error:
        print(f"error de W&B: {error}", file=sys.stderr)
        return 2

    # Un hilo, como al entrenar: el resultado no depende de cuántos núcleos haya.
    torch.set_num_threads(1)
    inicio = time.perf_counter()
    resultados = []
    grupo = args.grupo or crear_grupo()
    print(f"grupo W&B: {grupo}")
    run = abrir_run(cfg, grupo=grupo, job_type="comparacion",
                    etiquetas=[f"entorno:{cfg['entorno']['id']}", "metodo:comparacion"])
    try:
        for entrada in agentes:
            resultado = evaluar_final(entrada["agente"], cfg, entrada["estadisticas"])
            ruta = entrada["ruta"]
            resultados.append({"metodo": entrada["metodo"], "cantidad": entrada["cantidad"],
                               "repeticion": entrada["repeticion"],
                               "ruta": None if ruta is None else ruta_registrada(ruta),
                               **resultado})
            cantidad = "-" if entrada["cantidad"] is None else entrada["cantidad"]
            repeticion = "-" if entrada["repeticion"] is None else entrada["repeticion"]
            print(f"{entrada['metodo']:9} cantidad {cantidad:>3} repetición {repeticion:>2}: "
                  f"retorno {resultado['retorno_medio']:7.1f} ± "
                  f"{resultado['retorno_desviacion']:6.1f}", flush=True)
    except KeyboardInterrupt:
        terminar_run(run, KeyboardInterrupt())
        print("interrumpido: no se ha guardado nada", file=sys.stderr)
        return 130
    except BaseException as error:
        terminar_run(run, error)
        raise
    segundos = time.perf_counter() - inicio
    texto = json.dumps({"entorno": cfg["entorno"]["id"], "protocolo": protocolo,
                        "huella_dataset": huella, "segundos": segundos,
                        "codigo": estado_git(), "entorno_ejecucion": versiones_y_hardware(),
                        "resultados": resultados},
                       ensure_ascii=False, indent=2, allow_nan=False, default=a_json)
    try:
        escribir_archivo_nuevo(salida,
                               lambda temporal: temporal.write_text(texto, encoding="utf-8"))
        if run is not None:
            wandb = comprobar_wandb(cfg)
            if cfg.get("wandb", {}).get("modo") == "online":
                for entrada in agentes:
                    metodo = entrada["metodo"]
                    if metodo == "experto":
                        huella_modelo = huella_archivo(entrada["ruta"] / "modelo.zip")
                        nombre = (f"experto-{cfg['entorno']['id']}-"
                                  f"{huella_modelo[:12]}:latest")
                        usar_artefacto(run, nombre, "experto")
                    elif metodo in ("bc", "gail"):
                        ruta_modelo = (entrada["ruta"] if metodo == "bc" else
                                       entrada["ruta"] / "modelo.zip")
                        huella_modelo = huella_archivo(ruta_modelo)
                        nombre = (f"{metodo}-{cfg['entorno']['id']}-c{entrada['cantidad']}-"
                                  f"r{entrada['repeticion']}-{huella_modelo[:12]}:latest")
                        usar_artefacto(run, nombre, metodo)
            columnas = ["metodo", "cantidad", "repeticion", "retorno_medio",
                        "retorno_desviacion", "longitud_media"]
            tabla = wandb.Table(columns=columnas)
            episodios = wandb.Table(columns=["metodo", "cantidad", "repeticion",
                                             "episodio", "retorno", "longitud"])
            for fila in resultados:
                tabla.add_data(*(fila.get(k) for k in columnas))
                for indice, (retorno, longitud) in enumerate(zip(fila["retornos"],
                                                                  fila["longitudes"])):
                    episodios.add_data(fila["metodo"], fila["cantidad"], fila["repeticion"],
                                       indice, retorno, longitud)
            run.log({"comparacion": tabla, "episodios": episodios})
        registrar_resumen(run, {"entorno": cfg["entorno"]["id"],
                                "episodios": protocolo["episodios"],
                                "semilla": protocolo["semilla"], "segundos": segundos,
                                "huella_dataset": huella})
        subir_artefacto(run, f"comparacion-{cfg['entorno']['id']}", "comparacion",
                        [salida], {"grupo": grupo, "huella_dataset": huella})
    except BaseException as error:
        terminar_run(run, error)
        raise
    terminar_run(run)
    print(f"{len(resultados)} agentes evaluados con {protocolo['episodios']} partidas cada uno "
          f"en {segundos:.0f} s; resultados en {salida}")
    avisar_estado_git(estado_git())
    return 0


if __name__ == "__main__":
    sys.exit(main())
