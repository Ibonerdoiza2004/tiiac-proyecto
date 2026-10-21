#!/usr/bin/env python3
"""Entrena GAIL con las demostraciones del experto y guarda cada entrenamiento.

Los modelos no se versionan: este script es el procedimiento reproducible para
regenerarlos. Usa los datos de scripts/get_data.py (demostraciones.npz y
splits.json en una carpeta) y entrena, para cada tamaño de subconjunto y cada
repetición, un GAIL con las semillas de esa repetición (las mismas que BC):

    python scripts/entrenar_gail.py --entorno cartpole --datos data/demostraciones/cartpole/<fecha>
    python scripts/entrenar_gail.py --datos <dir> --cantidades 1 4 --repeticiones 0
    python scripts/entrenar_gail.py --datos <dir> --salida checkpoints/gail/prueba --sin-wandb

Sin --cantidades ni --repeticiones entrena todas las de la configuración
(demostraciones.subconjuntos × experimento.repeticiones). Elegirlas permite
repartir el barrido en varias sesiones: cada entrenamiento depende solo de su
cantidad y su repetición, no de cuántos se lancen juntos.

Cada uno se guarda en <salida>/cantidad_<c>/repeticion_<r> (modelo.zip,
discriminador.pt y gail.json, con la procedencia de los datos). Imprime el
último retorno de validación y el tiempo medido de cada uno. W&B sigue la
configuración (wandb.modo), con un run por entrenamiento; --sin-wandb lo
desactiva. Cómo se agrupan y nombran los runs es provisional hasta el módulo
de seguimiento.
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.data.demostraciones import cargar_demostraciones  # noqa: E402
from src.data.entornos import crear_entorno  # noqa: E402
from src.data.splits import cargar_splits, huella_archivo  # noqa: E402
from src.training.gail import entrenar_gail, guardar_gail, procedencia_datos  # noqa: E402
from src.training.trpo import comprobar_directorio_libre, estado_git  # noqa: E402
from src.utils.configuracion import TipoEntorno, resolver_configuracion  # noqa: E402
from src.utils.tracking import abrir_run  # noqa: E402

_DEMOSTRACIONES = "demostraciones.npz"
_SPLITS = "splits.json"


def main() -> int:
    """Carga los datos, entrena y guarda cada GAIL elegido; devuelve el código de salida.

    Códigos: 0 si todo va bien; 2 si la configuración, los datos, las
    cantidades, las repeticiones o la carpeta de salida no son válidos, sin
    entrenar nada.
    """
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"),
                    help="configuración base")
    ap.add_argument("--entorno", choices=[e.value for e in TipoEntorno],
                    default=TipoEntorno.CARTPOLE.value, help="perfil de entorno")
    ap.add_argument("--datos", type=Path, required=True,
                    help=f"carpeta de get_data.py ({_DEMOSTRACIONES} y {_SPLITS})")
    ap.add_argument("--cantidades", type=int, nargs="+", default=None,
                    help="tamaños de subconjunto (por defecto, demostraciones.subconjuntos)")
    ap.add_argument("--repeticiones", type=int, nargs="+", default=None,
                    help="repeticiones, de 0 a experimento.repeticiones − 1 (por defecto, todas)")
    ap.add_argument("--salida", type=Path, default=None,
                    help="carpeta nueva para los modelos "
                         "(por defecto checkpoints/gail/<entorno>/<fecha>)")
    ap.add_argument("--sin-wandb", action="store_true", help="no registrar en W&B")
    args = ap.parse_args()

    ajustes = {"wandb": {"modo": "disabled"}} if args.sin_wandb else None
    try:
        cfg = resolver_configuracion(args.config, args.entorno, ajustes)
    except (OSError, ValueError) as error:
        print(f"error de configuración: {error}", file=sys.stderr)
        return 2
    salida = args.salida or (Path("checkpoints") / "gail" / args.entorno
                             / datetime.now().strftime("%Y%m%d-%H%M%S"))
    # Antes de entrenar: si la salida está ocupada, el entrenamiento se perdería.
    try:
        comprobar_directorio_libre(salida)
    except FileExistsError as error:
        print(f"error: {error}; elige otra --salida", file=sys.stderr)
        return 2

    ruta_demostraciones, ruta_splits = args.datos / _DEMOSTRACIONES, args.datos / _SPLITS
    try:
        entorno = crear_entorno(cfg, None)
        try:
            espacios = entorno.observation_space, entorno.action_space
        finally:
            entorno.close()
        trayectorias, _ = cargar_demostraciones(ruta_demostraciones, *espacios)
        splits = cargar_splits(ruta_splits, huella_archivo(ruta_demostraciones),
                               [t.id_episodio for t in trayectorias])
        datos = procedencia_datos(ruta_demostraciones, ruta_splits, trayectorias)
    except (OSError, ValueError) as error:
        print(f"error con los datos: {error}", file=sys.stderr)
        return 2

    cantidades = args.cantidades or sorted(splits.subconjuntos)
    total_repeticiones = cfg["experimento"]["repeticiones"]
    repeticiones = (args.repeticiones if args.repeticiones is not None
                    else list(range(total_repeticiones)))
    no_validas = [c for c in cantidades if c not in splits.subconjuntos]
    if no_validas or len(set(cantidades)) != len(cantidades):
        print(f"error: cantidades no válidas o repetidas: {cantidades}; disponibles: "
              f"{sorted(splits.subconjuntos)}", file=sys.stderr)
        return 2
    if (any(not 0 <= r < total_repeticiones for r in repeticiones)
            or len(set(repeticiones)) != len(repeticiones)):
        print(f"error: repeticiones no válidas o repetidas: {repeticiones}; van de 0 a "
              f"{total_repeticiones - 1}", file=sys.stderr)
        return 2

    inicio = time.perf_counter()
    for cantidad in cantidades:
        for repeticion in repeticiones:
            cfg_run = copy.deepcopy(cfg)
            cfg_run["experimento"]["nombre"] = (f"{cfg['experimento']['nombre']}-gail-"
                                                f"{args.entorno}-c{cantidad}-r{repeticion}")
            run = abrir_run(cfg_run)
            try:
                agente, discriminador, resumen = entrenar_gail(cfg, trayectorias, splits,
                                                               cantidad, repeticion, run)
                destino = salida / f"cantidad_{cantidad}" / f"repeticion_{repeticion}"
                guardar_gail(destino, agente, discriminador, cfg, resumen, datos)
            finally:
                if run is not None:
                    run.finish()
            ultima = resumen["validaciones"][-1]
            print(f"cantidad {cantidad}, repetición {repeticion}: retorno de validación "
                  f"{ultima['retorno_medio']:.1f} en la iteración {ultima['iteracion']}; "
                  f"{resumen['pasos']} pasos en {resumen['segundos']:.0f} s")
    print(f"{len(cantidades) * len(repeticiones)} entrenamientos guardados en {salida}; "
          f"tiempo total medido: {time.perf_counter() - inicio:.0f} s")
    codigo = estado_git()
    if codigo["commit"] is None:
        print("aviso: no se pudo leer el commit de git; gail.json no dice con qué código "
              "se entrenó", file=sys.stderr)
    elif codigo["cambios_sin_commit"]:
        print("aviso: había cambios sin commit; estos modelos no se pueden reproducir "
              "exactamente desde el repositorio", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
