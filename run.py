#!/usr/bin/env python3
"""
Punto de entrada unico del proyecto.

    python run.py --smoke                        # pipeline entero en miniatura, < 5 min en CPU
    python run.py --config configs/default.yaml  # entrenamiento completo, < 20 min en T4
    python run.py --config configs/default.yaml --solo-evaluar

Este fichero es deliberadamente corto: orquesta, no implementa. La logica vive
en src/. Si run.py crece por encima de un centenar de lineas, algo tendria que
haber bajado a un modulo.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))

from src.utils.seed import fijar_seed          # noqa: E402
from src.utils.tracking import abrir_run       # noqa: E402
from src.data.loaders import construir_loaders  # noqa: E402
from src.models.factory import construir_modelo  # noqa: E402
from src.training.loop import entrenar, evaluar  # noqa: E402
from src.utils.configuracion import TipoEntorno, resolver_configuracion  # noqa: E402


def cargar_config(ruta: Path) -> dict:
    with open(ruta, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"),
                    help="fichero de configuracion")
    ap.add_argument("--smoke", action="store_true",
                    help="reduce la configuracion para comprobar el pipeline")
    ap.add_argument("--entorno", choices=[e.value for e in TipoEntorno],
                    default=TipoEntorno.CARTPOLE.value,
                    help="perfil de entorno que se combina con la configuracion base")
    ap.add_argument("--solo-evaluar", action="store_true",
                    help="carga el checkpoint y evalua, sin entrenar")
    args = ap.parse_args()

    try:
        ajustes = cargar_config(Path("configs/smoke.yaml")) if args.smoke else None
        cfg = resolver_configuracion(args.config, args.entorno, ajustes)
        if args.smoke:
            cfg["origen"]["archivos"].append("configs/smoke.yaml")
    except (OSError, ValueError) as error:
        print(f"error de configuración: {error}", file=sys.stderr)
        return 2

    # 1 · reproducibilidad, antes que nada
    fijar_seed(cfg["experimento"]["seed"])

    # 2 · seguimiento del experimento
    run = abrir_run(cfg)

    # 3 · datos
    train_loader, val_loader = construir_loaders(cfg)

    # 4 · modelo
    modelo = construir_modelo(cfg)

    # 5 · entrenar y evaluar
    if not args.solo_evaluar:
        entrenar(modelo, train_loader, val_loader, cfg, run)
    metricas = evaluar(modelo, val_loader, cfg)

    # 6 · el smoke test TIENE que imprimir una metrica: es la prueba de que el
    #     pipeline llega hasta el final. El checker lo comprueba.
    print("\n=== metricas ===")
    for k, v in metricas.items():
        print(f"{k}: {v:.4f}")

    if run is not None:
        run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
