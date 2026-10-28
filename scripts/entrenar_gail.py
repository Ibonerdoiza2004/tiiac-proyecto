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

experimento.procesos de la configuración dice cuántos entrenamientos se lanzan
a la vez, cada uno en su proceso (1 = en serie). Siempre se entrena con un solo
hilo de PyTorch, también en serie: con varios hilos cambia el orden de las sumas
y el redondeo, y el modelo ya no sería el mismo. Así, en la misma máquina, los
modelos son idénticos en serie y en paralelo. Si uno falla, los demás siguen.

Cada uno se guarda en <salida>/cantidad_<c>/repeticion_<r> (modelo.zip,
discriminador.pt y gail.json, con la procedencia de los datos). Imprime el
último retorno de validación y el tiempo medido de cada uno y, mientras entrena,
una línea por validación con la iteración, el retorno, los aciertos del
discriminador y el tiempo que queda (estimado). W&B sigue la configuración
(wandb.modo), con un run por entrenamiento; --sin-wandb lo desactiva. Cómo se
agrupan y nombran los runs es provisional hasta el módulo de seguimiento.

El flujo (argumentos, comprobaciones, reparto en procesos y códigos de salida)
es el de main_barrido en src/training/barrido.py, común con scripts/entrenar_bc.py.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.training.barrido import MetodoBarrido, main_barrido  # noqa: E402
from src.training.gail import comprobar_config_barrido, entrenar_y_guardar_gail  # noqa: E402


def _describir(resumen: dict[str, Any]) -> str:
    """Último retorno de validación, pasos y tiempo medido de un entrenamiento."""
    ultima = resumen["validaciones"][-1]
    return (f"retorno de validación {ultima['retorno_medio']:.1f} en la iteración "
            f"{ultima['iteracion']}; {resumen['pasos']} pasos en {resumen['segundos']:.0f} s")


GAIL = MetodoBarrido(nombre="gail", comprobar_config=comprobar_config_barrido,
                     tarea=entrenar_y_guardar_gail,
                     destino=lambda salida, c, r: salida / f"cantidad_{c}" / f"repeticion_{r}",
                     describir=_describir)


def main() -> int:
    """Entrena y guarda cada GAIL elegido; devuelve el código de salida (main_barrido)."""
    return main_barrido(GAIL, __doc__)


if __name__ == "__main__":
    sys.exit(main())
