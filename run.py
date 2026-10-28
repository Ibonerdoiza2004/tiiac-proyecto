#!/usr/bin/env python3
"""
Punto de entrada único del proyecto: el experimento completo de un entorno.

    python run.py --smoke                    # flujo real en miniatura, < 5 min en CPU
    python run.py --entorno cartpole         # experimento sobre los datos congelados
    python run.py --solo-evaluar --salida <carpeta de una ejecución anterior>

Orquesta, no implementa: las etapas viven en src/training/flujo.py.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from src.training.flujo import (  # noqa: E402
    comprobar_datos_congelados,
    ejecutar_experimento,
    entrenar_y_guardar_experto,
    evaluar_carpeta,
    generar_y_registrar_datos,
    imprimir_resultados,
    obtener_datos_congelados,
)
from src.utils.configuracion import TipoEntorno, resolver_configuracion  # noqa: E402
from src.utils.tracking import comprobar_wandb, crear_grupo  # noqa: E402


def main() -> int:
    """Ejecuta el experimento (o el smoke, o solo evaluación) y devuelve su código."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    ap.add_argument("--entorno", choices=[e.value for e in TipoEntorno],
                    default=TipoEntorno.CARTPOLE.value)
    ap.add_argument("--smoke", action="store_true", help="flujo real en miniatura")
    ap.add_argument("--solo-evaluar", action="store_true",
                    help="evalúa una ejecución anterior (--salida), sin entrenar")
    ap.add_argument("--salida", type=Path, default=None,
                    help="carpeta nueva (por defecto checkpoints/run/<entorno>/<fecha>)")
    ap.add_argument("--datos", type=Path, default=None, help="sustituye datos.demostraciones")
    ap.add_argument("--experto", type=Path, default=None, help="sustituye datos.experto")
    ap.add_argument("--descargar", action="store_true",
                    help="baja de W&B los datos y el experto congelados que falten")
    ap.add_argument("--sin-wandb", action="store_true", help="no registrar en W&B")
    ap.add_argument("--grupo", default=None, help="grupo W&B compartido con otros scripts")
    args = ap.parse_args()

    try:
        ajustes = (yaml.safe_load(Path("configs/smoke.yaml").read_text(encoding="utf-8"))
                   if args.smoke else {})
        if args.sin_wandb:
            ajustes["wandb"] = {"modo": "disabled"}
        cfg = resolver_configuracion(args.config, args.entorno, ajustes)
        comprobar_wandb(cfg, exigir_sesion=True)
    except (OSError, ValueError, RuntimeError, yaml.YAMLError) as error:
        print(f"error de configuración: {error}", file=sys.stderr)
        return 2
    grupo = args.grupo or crear_grupo()
    print(f"grupo W&B: {grupo}")
    torch.set_num_threads(1)

    try:
        if args.solo_evaluar:
            if args.salida is None:
                print("error: --solo-evaluar necesita --salida", file=sys.stderr)
                return 2
            experto = args.experto or Path(cfg["datos"]["experto"])
            resultado = evaluar_carpeta(cfg, args.salida, experto, grupo)
        elif args.smoke:
            with tempfile.TemporaryDirectory() as temporal:
                carpeta = Path(temporal)
                experto = carpeta / "experto"
                datos = carpeta / "datos"
                entrenar_y_guardar_experto(cfg, experto, grupo)
                generar_y_registrar_datos(cfg, experto, datos, grupo)
                resultado = ejecutar_experimento(cfg, datos, experto, carpeta / "run", grupo)
        else:
            datos = args.datos or Path(cfg["datos"]["demostraciones"])
            experto = args.experto or Path(cfg["datos"]["experto"])
            if args.descargar:
                obtener_datos_congelados(cfg, datos, experto)
            comprobar_datos_congelados(cfg, datos, experto)
            salida = args.salida or (Path("checkpoints") / "run" / args.entorno
                                     / datetime.now().strftime("%Y%m%d-%H%M%S"))
            resultado = ejecutar_experimento(cfg, datos, experto, salida, grupo)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrumpido", file=sys.stderr)
        return 130
    imprimir_resultados(resultado["resultados"])
    for fallo in resultado["fallos"]:
        print(f"falló {fallo}", file=sys.stderr)
    return 1 if resultado["fallos"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
