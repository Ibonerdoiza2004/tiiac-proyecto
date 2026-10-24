#!/usr/bin/env python3
"""Entrena Behavioral Cloning con las demostraciones del experto y guarda cada política.

Los modelos no se versionan: este script es el procedimiento reproducible para
regenerarlos. Usa los datos de scripts/get_data.py (demostraciones.npz y
splits.json en una carpeta) y entrena, para cada tamaño de subconjunto y cada
repetición, una política de BC con las semillas de esa repetición. Son las
mismas que usa GAIL, así que BC y GAIL de la misma repetición entrenan con los
mismos pares expertos:

    python scripts/entrenar_bc.py --entorno cartpole --datos data/demostraciones/cartpole/<fecha>
    python scripts/entrenar_bc.py --datos <dir> --cantidades 4 --repeticiones 0
    python scripts/entrenar_bc.py --datos <dir> --salida checkpoints/bc/prueba --sin-wandb

Sin --cantidades ni --repeticiones entrena todas las de la configuración
(demostraciones.subconjuntos × experimento.repeticiones), con
experimento.procesos a la vez y un hilo de PyTorch, como scripts/entrenar_gail.py:
el flujo y los códigos de salida son los de main_barrido (src/training/barrido.py).

Cada política se guarda en <salida>/cantidad_<c>/repeticion_<r>.pt con
guardar_politica_bc (arquitectura, pesos, normalización y procedencia: semillas,
huellas de los datos, commit y versiones). Imprime la mejor época, los pares de
train y el tiempo medido de cada una. W&B sigue la configuración (wandb.modo),
con un run por entrenamiento; --sin-wandb lo desactiva.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.training.barrido import MetodoBarrido, main_barrido  # noqa: E402
from src.training.bc import comprobar_config_barrido_bc, entrenar_y_guardar_bc  # noqa: E402


def _describir(resumen: dict[str, Any]) -> str:
    """Mejor época, pares de train y tiempo medido de un entrenamiento."""
    return (f"mejor época {resumen['mejor_epoca']} de {resumen['epocas']}; "
            f"{resumen['pares_train']} pares de train en {resumen['segundos']:.0f} s")


BC = MetodoBarrido(nombre="bc", comprobar_config=comprobar_config_barrido_bc,
                   tarea=entrenar_y_guardar_bc,
                   destino=lambda salida, c, r: salida / f"cantidad_{c}" / f"repeticion_{r}.pt",
                   describir=_describir)


def main() -> int:
    """Entrena y guarda cada BC elegido; devuelve el código de salida (main_barrido)."""
    return main_barrido(BC, __doc__)


if __name__ == "__main__":
    sys.exit(main())
