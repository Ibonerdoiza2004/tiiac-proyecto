#!/usr/bin/env python3
"""Llena la biblioteca del dashboard desde W&B: congelados y BC y GAIL de entrega.

    python scripts/preparar_biblioteca.py                       # los cinco entornos
    python scripts/preparar_biblioteca.py --entorno cartpole --entorno reacher
    python scripts/preparar_biblioteca.py --sin-entrega         # solo los congelados

Necesita una sesión de W&B con acceso al proyecto (wandb login). Descarga en la
carpeta aplicativo.biblioteca de la configuración y comprueba las huellas de los
congelados, como run.py --descargar. Se puede repetir: lo que ya está no se vuelve
a descargar. Es lo mismo que hará el contenedor de Docker al arrancar.

Sale con 0 si todo fue bien, 1 si algún entorno falló (los demás siguen) y 2 si la
configuración no es válida.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.data.biblioteca import Biblioteca  # noqa: E402
from src.training.precarga import (artefactos_de_comparacion, precargar_congelados,  # noqa: E402
                                   precargar_entrega)
from src.utils.configuracion import (TipoEntorno, leer_configuracion_base,  # noqa: E402
                                     resolver_configuracion)

CODIGO_FALLO = 1
CODIGO_CONFIG = 2


def _argumentos(argv: Sequence[str] | None) -> argparse.Namespace:
    """Lee la línea de comandos."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    ap.add_argument("--entorno", action="append", choices=[t.value for t in TipoEntorno],
                    help="entorno que se precarga (se puede repetir; por defecto, todos)")
    ap.add_argument("--sin-entrega", action="store_true",
                    help="solo los congelados, sin los BC y GAIL de entrega")
    return ap.parse_args(argv)


def preparar(config: Path, entornos: Sequence[str], con_entrega: bool) -> int:
    """Precarga cada entorno; si uno falla, informa y sigue con los demás.

    Args:
        config: YAML base.
        entornos: Claves de los entornos.
        con_entrega: Si se añaden también los BC y GAIL de entrega.

    Returns:
        0 si todo fue bien y 1 si algún entorno falló.

    Raises:
        ValueError: Si la configuración base o dashboard.yaml no valen.
    """
    base = leer_configuracion_base(config)
    comparaciones = leer_configuracion_base(Path(config).parent / "dashboard.yaml").get(
        "precarga", {}).get("comparaciones", {})
    cfgs = {e: resolver_configuracion(config, e) for e in entornos}
    biblioteca = Biblioteca(Path(base["aplicativo"]["biblioteca"]),
                            {e: cfg["entorno"]["id"] for e, cfg in cfgs.items()})
    fallos = []
    for entorno, cfg in cfgs.items():
        try:
            precargar_congelados(cfg, biblioteca)
            if con_entrega:
                if entorno not in comparaciones:
                    print(f"{entorno}: no hay run de comparación en precarga.comparaciones")
                    continue
                precargar_entrega(cfg, biblioteca,
                                  artefactos_de_comparacion(cfg, comparaciones[entorno]))
        except Exception as error:  # noqa: BLE001 - un entorno que falla no para los demás
            print(f"error en {entorno}: {error or type(error).__name__}", file=sys.stderr)
            fallos.append(entorno)
    return CODIGO_FALLO if fallos else 0


def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada del script."""
    args = _argumentos(argv)
    try:
        return preparar(args.config, args.entorno or [t.value for t in TipoEntorno],
                        not args.sin_entrega)
    except (ValueError, KeyError) as error:
        print(f"error: configuración no válida: {error}", file=sys.stderr)
        return CODIGO_CONFIG


if __name__ == "__main__":
    raise SystemExit(main())
