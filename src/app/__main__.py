"""Arranca el dashboard: la API y el frontend compilado.

    python -m src.app                       # configs/default.yaml
    python -m src.app --config otra.yaml

La dirección y el puerto salen de la sección aplicativo de la configuración.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import uvicorn

from src.app.api import CONFIG_POR_DEFECTO, config_aplicativo, crear_app
from src.utils.configuracion import leer_configuracion_base

CODIGO_CONFIG_INVALIDA = 2


def main(argv: Sequence[str] | None = None) -> int:
    """Lee la configuración, crea la aplicación y la sirve con Uvicorn.

    Args:
        argv: Argumentos de la línea de comandos (por defecto, los del proceso).

    Returns:
        0 al parar el servidor y 2 si la configuración no es válida.
    """
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=CONFIG_POR_DEFECTO,
                    help="configuración base")
    args = ap.parse_args(argv)
    try:
        aplicativo = config_aplicativo(leer_configuracion_base(args.config))
        app = crear_app(args.config)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return CODIGO_CONFIG_INVALIDA
    uvicorn.run(app, host=aplicativo["host"], port=aplicativo["puerto"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
