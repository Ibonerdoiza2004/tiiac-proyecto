"""Si se puede registrar en W&B desde el dashboard: hay sesión iniciada y se valida.

La API lo comprueba al arrancar (en segundo plano, para no retrasar el arranque) y
cuando se pide (``POST /api/wandb/comprobar``, por ejemplo después de hacer
``wandb login``), y lo dice en ``GET /api/estado``. Pedir un entrenamiento con W&B sin
sesión da error antes de encolarlo.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from src.utils.tracking import comprobar_wandb

# Devuelve None si W&B se puede usar o el motivo por el que no.
Comprobacion = Callable[[], "str | None"]


def comprobar_sesion_wandb() -> str | None:
    """Valida la sesión de W&B contra el servidor, sin pedir nada por la terminal.

    Returns:
        None si hay una sesión válida; si no, el motivo.
    """
    try:
        comprobar_wandb({"wandb": {"modo": "online"}}, exigir_sesion=True)
    except RuntimeError as error:
        return str(error)
    return None


class EstadoWandb:
    """Último resultado de comprobar W&B, seguro entre hilos.

    Si dos comprobaciones se solapan (la del arranque, lenta, y otra pedida después),
    se queda el resultado de la que empezó más tarde, aunque acabe antes.
    """

    def __init__(self, comprobar: Comprobacion = comprobar_sesion_wandb):
        """Prepara el estado sin comprobar todavía (``disponible`` es None).

        Args:
            comprobar: Función que comprueba W&B (los tests pasan una falsa).
        """
        self._comprobar = comprobar
        self._cerrojo = threading.Lock()
        self._disponible: bool | None = None
        self._motivo: str | None = None
        self._pedidas = 0  # comprobaciones empezadas
        self._guardada = 0  # número de la comprobación cuyo resultado está guardado

    def describir(self) -> dict[str, Any]:
        """``{"disponible", "motivo"}``; disponible es None mientras no se ha comprobado."""
        with self._cerrojo:
            return {"disponible": self._disponible, "motivo": self._motivo}

    def comprobar(self) -> dict[str, Any]:
        """Comprueba W&B ahora, guarda el resultado y lo devuelve.

        Un error inesperado de la comprobación cuenta como no disponible, con su motivo.
        Si mientras tanto empezó y acabó otra más reciente, su resultado no se pisa y
        se devuelve ese.
        """
        with self._cerrojo:
            self._pedidas += 1
            numero = self._pedidas
        try:
            motivo = self._comprobar()
        except Exception as error:  # noqa: BLE001 - cualquier fallo es «no disponible»
            motivo = f"no se pudo comprobar W&B: {error!r}"
        with self._cerrojo:
            if numero > self._guardada:
                self._disponible, self._motivo = motivo is None, motivo
                self._guardada = numero
            return {"disponible": self._disponible, "motivo": self._motivo}

    def exigir(self) -> None:
        """Comprueba que W&B se puede usar; si aún no se había comprobado, lo hace ahora.

        Raises:
            RuntimeError: Con el motivo si no hay sesión.
        """
        estado = self.describir()
        if estado["disponible"] is None:
            estado = self.comprobar()
        if not estado["disponible"]:
            raise RuntimeError(f"W&B no está disponible: {estado['motivo']}. Inicia sesión con "
                               "wandb login y vuelve a comprobarlo.")
