"""Configuración común de los tests."""

from __future__ import annotations

import pytest

from src.app import estado_wandb


@pytest.fixture(autouse=True)
def sin_red_de_wandb(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ningún test comprueba la sesión de W&B contra el servidor.

    La API la comprueba al arrancar; en los tests, la comprobación por defecto dice que
    no está disponible sin tocar la red. Los tests de W&B pasan su propia comprobación.
    """
    monkeypatch.setattr(estado_wandb, "comprobar_sesion_wandb",
                        lambda: "en los tests no se comprueba W&B")
