"""Semillas aleatorias, fijadas en UN SOLO SITIO.

Si te encuentras llamando a manual_seed en tres ficheros distintos, es que
esta funcion no se esta usando. El checker avisa de eso.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def fijar_seed(seed: int, determinista: bool = True) -> None:
    """Fija las semillas de las tres fuentes de aleatoriedad del proyecto.

    Args:
        seed: la semilla. Va en configs/, no aqui.
        determinista: fuerza los kernels deterministas de cuDNN. Cuesta algo de
            velocidad; a la escala de la regla 5/20 no se nota y a cambio dos
            ejecuciones dan el mismo numero.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if determinista:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def crear_generador(semilla: int) -> np.random.Generator:
    """Crea un generador aleatorio propio, independiente de la semilla global.

    Sirve para decisiones que deben repetirse igual aunque cambie el resto de la
    ejecución, como el reparto de demostraciones o el submuestreo de pares. Usar
    siempre esta función en lugar de crear generadores en otros módulos.

    Args:
        semilla: entero no negativo. Va en configs/, no aquí.

    Returns:
        Un numpy.random.Generator; la misma semilla produce la misma secuencia.

    Raises:
        ValueError: si la semilla no es un entero no negativo.
    """
    if (isinstance(semilla, (bool, np.bool_))
            or not isinstance(semilla, (int, np.integer)) or semilla < 0):
        raise ValueError(f"La semilla debe ser un entero no negativo, no {semilla!r}")
    return np.random.default_rng(int(semilla))
