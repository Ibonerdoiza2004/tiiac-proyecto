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

from src.training.trpo import entrenar_experto, guardar_experto  # noqa: E402
from src.utils.archivos import comprobar_directorio_libre  # noqa: E402
from src.utils.configuracion import TipoEntorno, resolver_configuracion  # noqa: E402
from src.utils.procedencia import avisar_estado_git, estado_git  # noqa: E402
from src.utils.tracking import (abrir_run, comprobar_wandb, crear_grupo,  # noqa: E402
                                registrar_resumen, subir_artefacto, terminar_run)


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
    # Antes de entrenar: si la salida está ocupada, el entrenamiento se perdería.
    try:
        comprobar_directorio_libre(salida)
    except FileExistsError as error:
        print(f"error: {error}; elige otra --salida", file=sys.stderr)
        return 2
    try:
        comprobar_wandb(cfg, exigir_sesion=True)
    except RuntimeError as error:
        print(f"error de W&B: {error}", file=sys.stderr)
        return 2

    grupo = args.grupo or crear_grupo()
    print(f"grupo W&B: {grupo}")
    run = abrir_run(cfg, grupo=grupo, job_type="experto",
                    etiquetas=[f"entorno:{cfg['entorno']['id']}", "metodo:experto"])
    try:
        modelo, resumen = entrenar_experto(cfg, run)
        guardar_experto(salida, modelo, cfg, resumen)
        registrar_resumen(run, resumen)
        subir_artefacto(run, f"experto-{cfg['entorno']['id']}-{grupo}", "experto",
                        [salida / "modelo.zip", salida / "experto.json"],
                        {"entorno": cfg["entorno"]["id"], "grupo": grupo})
    except BaseException as error:
        # Si falla el entrenamiento o el guardado, W&B lo marca como fallido.
        terminar_run(run, error)
        raise
    terminar_run(run)

    estado = "alcanzado" if resumen["objetivo_alcanzado"] else "no alcanzado"
    print(f"experto de {resumen['entorno']} guardado en {salida}")
    print(f"mejor retorno: {resumen['mejor_retorno']:.1f} "
          f"(iteración {resumen['mejor_iteracion']}); objetivo {resumen['objetivo']} {estado}")
    print(f"pasos: {resumen['pasos']}; tiempo medido: {resumen['segundos']:.0f} s")
    avisar_estado_git(estado_git())
    return 0


if __name__ == "__main__":
    sys.exit(main())
