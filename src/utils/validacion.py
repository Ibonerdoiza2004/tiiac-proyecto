"""Comprobaciones de valores numéricos que comparte todo el proyecto.

Un único sitio para decidir qué cuenta como entero o real válido, en la
configuración y en los argumentos de las funciones: enteros de Python o de
NumPy (nunca booleanos, aunque bool sea un subtipo de int) y reales finitos.
"""

from __future__ import annotations

import numpy as np


def es_entero(valor: object) -> bool:
    """Indica si valor es un entero de Python o de NumPy que no sea booleano."""
    return isinstance(valor, (int, np.integer)) and not isinstance(valor, (bool, np.bool_))


def es_entero_positivo(valor: object) -> bool:
    """Indica si valor es un entero (es_entero) mayor que cero."""
    # bool(): con enteros de NumPy, la comparación da np.bool_.
    return es_entero(valor) and bool(valor > 0)


def es_real_finito(valor: object) -> bool:
    """Indica si valor es un número real finito de Python o de NumPy que no sea booleano.

    Un entero de Python vale si cabe en un float (los más grandes, como 10**400, darían
    error al convertirlos); NumPy no admite enteros de Python fuera de int64, así que no
    se le pasan.
    """
    if isinstance(valor, (bool, np.bool_)):
        return False
    if isinstance(valor, int):
        try:
            float(valor)
        except OverflowError:
            return False
        return True
    return isinstance(valor, (float, np.integer, np.floating)) and bool(np.isfinite(valor))


def es_semilla(valor: object) -> bool:
    """Indica si valor es una semilla válida: un entero (es_entero) no negativo.

    Es la regla que aplica validar_semilla (seed.py), aquí sin importar PyTorch para
    que la API pueda comprobar la configuración.
    """
    return es_entero(valor) and bool(valor >= 0)
