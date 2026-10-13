"""Contenedores compartidos para las demostraciones del proyecto."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class Trayectoria:
    """Una partida completa con sus transiciones en orden.

    estados y siguientes_estados tienen forma [T, d_observacion]; acciones,
    [T] para discretas o [T, d_accion] para continuas. Los otros arrays tienen
    forma [T]. Las recompensas son las originales del entorno. El último paso
    termina o se trunca; se conservan ambos indicadores por separado.

    Este contenedor no normaliza ni copia arrays. La validación frente a los
    espacios del entorno la realiza validar_demostraciones, también al guardar.
    id_episodio es un entero que identifica la partida completa, no un índice
    distinto por cada transición.
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
