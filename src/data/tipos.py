"""Contenedores compartidos para las demostraciones del proyecto."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

ID_EPISODIO_MINIMO = int(np.iinfo(np.int64).min)
ID_EPISODIO_MAXIMO = int(np.iinfo(np.int64).max)


def es_id_episodio(valor: object) -> bool:
    """Indica si valor es un id_episodio válido: entero no booleano en el rango de int64.

    El rango permite guardar los IDs en arrays int64 y tensores sin perder precisión.
    """
    return (isinstance(valor, (int, np.integer)) and not isinstance(valor, (bool, np.bool_))
            and ID_EPISODIO_MINIMO <= int(valor) <= ID_EPISODIO_MAXIMO)


@dataclass(frozen=True, slots=True)
class Trayectoria:
    """Una partida completa con sus transiciones en orden.

    estados y siguientes_estados tienen forma [T, d_observacion]; acciones,
    [T] para discretas o [T, d_accion] para continuas. Los otros arrays tienen
    forma [T]. Las recompensas son las originales del entorno. El último paso
    termina o se trunca; se conservan ambos indicadores por separado.

    Este contenedor no normaliza ni copia arrays. La validación frente a los
    espacios del entorno la realiza validar_demostraciones, también al guardar.
    id_episodio es un entero en el rango de int64 (es_id_episodio) que identifica
    la partida completa, no un índice distinto por cada transición.
    """

    id_episodio: int
    estados: NDArray[np.generic]
    acciones: NDArray[np.generic]
    siguientes_estados: NDArray[np.generic]
    recompensas_originales: NDArray[np.generic]
    terminados: NDArray[np.bool_]
    truncados: NDArray[np.bool_]

    @property
    def numero_pasos(self) -> int:
        """Devuelve el número de transiciones almacenadas en la partida."""
        return len(self.estados)


@dataclass(frozen=True, slots=True)
class LoteTransiciones:
    """Pares (estado, acción) sueltos, de una o varias partidas.

    A diferencia de Trayectoria, los pares no tienen por qué ser consecutivos ni
    formar partidas completas. estados tiene forma [N, d_observacion]; acciones,
    [N] para discretas o [N, d_accion] para continuas. ids_episodio y pasos, de
    forma [N], indican de qué partida y de qué paso de ella sale cada par, de modo
    que se puede reconstruir qué pares de cada partida se usaron.

    Como Trayectoria, no normaliza ni copia arrays; quien lo crea garantiza la
    coherencia de las formas.
    """

    estados: NDArray[np.generic]
    acciones: NDArray[np.generic]
    ids_episodio: NDArray[np.integer]
    pasos: NDArray[np.integer]

    @property
    def numero_transiciones(self) -> int:
        """Devuelve el número de pares del lote."""
        return len(self.estados)
