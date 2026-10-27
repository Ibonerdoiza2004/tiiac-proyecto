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
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.training.flujo import cargar_agentes, evaluar_y_guardar  # noqa: E402
from src.training.evaluacion import config_evaluacion  # noqa: E402
from src.utils.configuracion import TipoEntorno, resolver_configuracion  # noqa: E402
from src.utils.tracking import comprobar_wandb, crear_grupo  # noqa: E402


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
    return cargar_agentes(cfg, args.experto, args.bc, args.gail,
                          args.cantidades, args.repeticiones)


def main() -> int:
    """Carga los agentes, los evalúa con el protocolo común y guarda los resultados.

    Returns:
        Código de salida 0 si se guardó la comparación, 2 si falló la validación
        y 130 si se interrumpió.

    Raises:
        Ninguna: los errores esperados se presentan al usuario como códigos de salida.
    """
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
    if args.salida is not None and args.salida.exists():
        print(f"error: {args.salida} ya existe; no se sobrescribe, elige otra --salida",
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
    salida = args.salida or (Path("checkpoints") / "evaluacion" / args.entorno
                             / f"{datetime.now():%Y%m%d-%H%M%S}.json")
    torch.set_num_threads(1)
    grupo = args.grupo or crear_grupo()
    print(f"grupo W&B: {grupo}")
    try:
        resultados = evaluar_y_guardar(cfg, agentes, huella, salida, grupo)
    except KeyboardInterrupt:
        print("interrumpido: no se ha guardado nada", file=sys.stderr)
        return 130
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    for resultado in resultados:
        cantidad = "-" if resultado["cantidad"] is None else resultado["cantidad"]
        repeticion = "-" if resultado["repeticion"] is None else resultado["repeticion"]
        print(f"{resultado['metodo']:9} cantidad {cantidad:>3} repetición {repeticion:>2}: "
              f"retorno {resultado['retorno_medio']:7.1f} ± "
              f"{resultado['retorno_desviacion']:6.1f}", flush=True)
    print(f"{len(resultados)} agentes evaluados con {protocolo['episodios']} partidas cada uno; "
          f"resultados en {salida}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
