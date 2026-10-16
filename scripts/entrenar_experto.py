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

from src.training.trpo import (comprobar_directorio_libre, entrenar_experto,  # noqa: E402
                               estado_git, guardar_experto)
from src.utils.configuracion import TipoEntorno, resolver_configuracion  # noqa: E402
from src.utils.tracking import abrir_run  # noqa: E402


def main() -> int:
    """Resuelve la configuración, entrena el experto y lo guarda; devuelve el código de salida."""
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
    args = ap.parse_args()

    ajustes = {"wandb": {"modo": "disabled"}} if args.sin_wandb else None
    try:
        cfg = resolver_configuracion(args.config, args.entorno, ajustes)
    except (OSError, ValueError) as error:
        print(f"error de configuración: {error}", file=sys.stderr)
        return 2
    salida = args.salida or (Path("checkpoints") / "expertos" / args.entorno
                             / datetime.now().strftime("%Y%m%d-%H%M%S"))
    # Antes de entrenar: si la salida está ocupada, el entrenamiento se perdería.
    try:
        comprobar_directorio_libre(salida)
    except FileExistsError as error:
        print(f"error: {error}; elige otra --salida", file=sys.stderr)
        return 2

    run = abrir_run(cfg)
    try:
        modelo, resumen = entrenar_experto(cfg, run)
        guardar_experto(salida, modelo, cfg, resumen)
    finally:
        if run is not None:
            run.finish()

    estado = "alcanzado" if resumen["objetivo_alcanzado"] else "no alcanzado"
    print(f"experto de {resumen['entorno']} guardado en {salida}")
    print(f"mejor retorno: {resumen['mejor_retorno']:.1f} "
          f"(iteración {resumen['mejor_iteracion']}); objetivo {resumen['objetivo']} {estado}")
    print(f"pasos: {resumen['pasos']}; tiempo medido: {resumen['segundos']:.0f} s")
    codigo = estado_git()
    if codigo["commit"] is None:
        print("aviso: no se pudo leer el commit de git; experto.json no dice con qué código "
              "se entrenó", file=sys.stderr)
    elif codigo["cambios_sin_commit"]:
        print("aviso: había cambios sin commit; este experto no se puede reproducir "
              "exactamente desde el repositorio", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
