#!/usr/bin/env python3
"""Entrena el experto TRPO de un entorno y lo guarda.

El experto no se versiona: este script es el procedimiento reproducible para
regenerarlo. Con la misma configuración y la misma semilla (experto.semilla) se
obtiene el mismo experto en la misma máquina.

    python scripts/entrenar_experto.py --entorno cartpole
    python scripts/entrenar_experto.py --entorno cartpole --salida checkpoints/expertos/prueba
    python scripts/entrenar_experto.py --entorno cartpole --sin-wandb

Imprime el resumen: retorno de la mejor evaluación, si alcanzó el objetivo,
pasos y segundos medidos. W&B sigue la configuración (wandb.modo); --sin-wandb
lo desactiva con la capa de ajustes, por ejemplo para medir tiempos sin cuenta.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.training.flujo import entrenar_y_guardar_experto  # noqa: E402
from src.utils.configuracion import TipoEntorno, resolver_configuracion  # noqa: E402
from src.utils.tracking import comprobar_wandb, crear_grupo  # noqa: E402


def main() -> int:
    """Resuelve la configuración, entrena el experto y lo guarda.

    Returns:
        Código de salida del script (0 correcto, 2 si configuración o ruta no válida).

    Raises:
        Ninguna: los errores esperados se presentan al usuario como códigos de salida.
    """
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"),
                    help="configuración base")
    ap.add_argument("--entorno", choices=[e.value for e in TipoEntorno],
                    default=TipoEntorno.CARTPOLE.value, help="perfil de entorno")
    ap.add_argument("--salida", type=Path, default=None,
                    help="directorio nuevo donde guardar el experto "
                         "(por defecto checkpoints/expertos/<entorno>/<fecha>)")
    ap.add_argument("--sin-wandb", action="store_true", help="no registrar en W&B")
    ap.add_argument("--grupo", default=None, help="grupo W&B compartido con el resto del flujo")
    args = ap.parse_args()

    ajustes = {"wandb": {"modo": "disabled"}} if args.sin_wandb else None
    try:
        cfg = resolver_configuracion(args.config, args.entorno, ajustes)
    except (OSError, ValueError) as error:
        print(f"error de configuración: {error}", file=sys.stderr)
        return 2
    salida = args.salida or (Path("checkpoints") / "expertos" / args.entorno
                             / datetime.now().strftime("%Y%m%d-%H%M%S"))
    try:
        comprobar_wandb(cfg, exigir_sesion=True)
    except RuntimeError as error:
        print(f"error de W&B: {error}", file=sys.stderr)
        return 2

    grupo = args.grupo or crear_grupo()
    print(f"grupo W&B: {grupo}")
    try:
        resumen = entrenar_y_guardar_experto(cfg, salida, grupo)
    except FileExistsError as error:
        print(f"error: {error}; elige otra --salida", file=sys.stderr)
        return 2

    estado = "alcanzado" if resumen["objetivo_alcanzado"] else "no alcanzado"
    print(f"experto de {resumen['entorno']} guardado en {salida}")
    print(f"mejor retorno: {resumen['mejor_retorno']:.1f} "
          f"(iteración {resumen['mejor_iteracion']}); objetivo {resumen['objetivo']} {estado}")
    print(f"pasos: {resumen['pasos']}; tiempo medido: {resumen['segundos']:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
