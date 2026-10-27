#!/usr/bin/env python3
"""Genera las demostraciones del experto y sus splits, y verifica su huella.

Los datos NO van al repositorio. No se descargan: se graban con el experto TRPO
propio, que tampoco se versiona (scripts/entrenar_experto.py). Lo que va es este
script, que es el procedimiento reproducible para obtenerlos:

    python scripts/entrenar_experto.py --entorno cartpole
    python scripts/get_data.py --entorno cartpole --experto checkpoints/expertos/cartpole/<fecha>
    python scripts/get_data.py --entorno cartpole --experto <dir> --huella-esperada <sha256>

Escribe demostraciones.npz y splits.json en una carpeta nueva (por defecto
data/demostraciones/<entorno>/<fecha>) e imprime tres huellas SHA-256: la de
cada archivo y la de las partidas (huella_partidas, solo su contenido). Los
splits citan la del NPZ, que también depende de los metadatos (commit,
versiones). --huella-esperada compara la de las partidas, que con el mismo
experto, configuración y máquina no cambia aunque cambie el commit. En otra
máquina los números del experto pueden variar en los últimos decimales y la
huella no está garantizada.

Registra el run de tipo datos y sube el NPZ y sus splits como artefacto. Usa
--grupo para reunirlo con los runs del experto, BC, GAIL y comparación; usa
--sin-wandb para desactivar el seguimiento.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.data.demostraciones import generar_dataset  # noqa: E402
from src.training.trpo import cargar_politica_experto  # noqa: E402
from src.utils.archivos import comprobar_directorio_libre  # noqa: E402
from src.utils.configuracion import TipoEntorno, resolver_configuracion  # noqa: E402
from src.utils.procedencia import avisar_estado_git  # noqa: E402
from src.utils.tracking import (abrir_run, comprobar_wandb, crear_grupo, registrar_resumen,  # noqa: E402
                                nombre_run, subir_artefacto, terminar_run, usar_artefacto)


def main() -> int:
    """Graba las demostraciones con el experto, crea los splits y devuelve el código de salida.

    Códigos: 0 si todo va bien; 1 si la huella de las partidas no coincide con
    --huella-esperada (los archivos se conservan para compararlos); 2 si la configuración, el
    experto o la carpeta de salida no son válidos, sin grabar nada.
    """
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"),
                    help="configuración base")
    ap.add_argument("--entorno", choices=[e.value for e in TipoEntorno],
                    default=TipoEntorno.CARTPOLE.value, help="perfil de entorno")
    ap.add_argument("--experto", type=Path, required=True,
                    help="carpeta del experto (modelo.zip y experto.json)")
    ap.add_argument("--salida", type=Path, default=None,
                    help="carpeta nueva para los datos "
                         "(por defecto data/demostraciones/<entorno>/<fecha>)")
    ap.add_argument("--huella-esperada", default=None,
                    help="huella de las partidas congeladas, para verificar la regeneración")
    ap.add_argument("--sin-wandb", action="store_true", help="no registrar en W&B")
    ap.add_argument("--grupo", default=None, help="grupo W&B de esta ejecución completa")
    args = ap.parse_args()

    try:
        ajustes = {"wandb": {"modo": "disabled"}} if args.sin_wandb else None
        cfg = resolver_configuracion(args.config, args.entorno, ajustes)
    except (OSError, ValueError) as error:
        print(f"error de configuración: {error}", file=sys.stderr)
        return 2
    salida = args.salida or (Path("data") / "demostraciones" / args.entorno
                             / datetime.now().strftime("%Y%m%d-%H%M%S"))
    try:
        comprobar_directorio_libre(salida)
    except FileExistsError as error:
        print(f"error: {error}; elige otra --salida", file=sys.stderr)
        return 2
    try:
        politica, metadatos = cargar_politica_experto(args.experto, cfg)
    except (OSError, ValueError) as error:
        print(f"error con el experto: {error}", file=sys.stderr)
        return 2
    try:
        comprobar_wandb(cfg, exigir_sesion=True)
    except RuntimeError as error:
        print(f"error de W&B: {error}", file=sys.stderr)
        return 2

    grupo = args.grupo or crear_grupo()
    print(f"grupo W&B: {grupo}")
    run = abrir_run(cfg, grupo=grupo, job_type="datos", nombre=nombre_run(cfg, "datos"),
                    etiquetas=[f"entorno:{cfg['entorno']['id']}", "metodo:datos"])
    try:
        if cfg.get("wandb", {}).get("modo") == "online":
            huella_modelo = metadatos["experto"]["huella_modelo"]
            usar_artefacto(run,
                           f"experto-{cfg['entorno']['id']}-{huella_modelo[:12]}:latest",
                           "experto")
        inicio = time.perf_counter()
        resultado = generar_dataset(cfg, politica, salida, metadatos)
        segundos = time.perf_counter() - inicio
        if (args.huella_esperada is not None
                and args.huella_esperada != resultado["huella_partidas"]):
            terminar_run(run, ValueError("la huella de las partidas no coincide"))
            print(f"huella de demostraciones.npz: {resultado['huella_dataset']}")
            print(f"huella de splits.json: {resultado['huella_splits']}")
            print(f"huella de las partidas: {resultado['huella_partidas']}")
            print(f"error: la huella de las partidas no coincide con la esperada "
                  f"({args.huella_esperada}); son distintas de las congeladas", file=sys.stderr)
            return 1
        registrar_resumen(run, {"pasos": resultado["pasos"],
                                "semillas": resultado["semillas"],
                                "segundos": segundos,
                                "partidas": len(resultado["retornos"]),
                                "retorno_medio": sum(resultado["retornos"])
                                / len(resultado["retornos"]),
                                "huella_dataset": resultado["huella_dataset"],
                                "huella_splits": resultado["huella_splits"],
                                "huella_partidas": resultado["huella_partidas"]})
        subir_artefacto(run, f"demostraciones-{resultado['huella_dataset'][:12]}",
                        "demostraciones",
                        [resultado["demostraciones"], resultado["splits"]],
                        {k: resultado[k] for k in
                         ("huella_dataset", "huella_splits", "huella_partidas")})
    except BaseException as error:
        terminar_run(run, error)
        if isinstance(error, ValueError):
            print(f"error de configuración: {error}", file=sys.stderr)
            return 2
        raise
    terminar_run(run)
    retornos = resultado["retornos"]
    print(f"{len(retornos)} partidas y {resultado['pasos']} pasos guardados en {salida}")
    print(f"retorno medio: {sum(retornos) / len(retornos):.1f} "
          f"(mínimo {min(retornos):.1f}, máximo {max(retornos):.1f})")
    print(f"train: {resultado['train']}; validación: {resultado['validacion']}")
    print(f"huella de {resultado['demostraciones'].name}: {resultado['huella_dataset']}")
    print(f"huella de {resultado['splits'].name}: {resultado['huella_splits']}")
    print(f"huella de las partidas: {resultado['huella_partidas']}")
    avisar_estado_git(metadatos["procedencia"]["codigo"])
    if args.huella_esperada is not None:
        print("ok: las partidas coinciden con las congeladas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
