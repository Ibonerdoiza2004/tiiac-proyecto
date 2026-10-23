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
    """Indica si valor es un número real finito de Python o de NumPy que no sea booleano."""
    return (isinstance(valor, (int, float, np.integer, np.floating))
            and not isinstance(valor, (bool, np.bool_)) and bool(np.isfinite(valor)))
