"""Configuración común de los tests."""

from __future__ import annotations

import os

import pytest

# Sin pantalla (el CI, una sesión SSH), MuJoCo intenta abrir una ventana con GLFW al
# dibujar y aborta el proceso entero: los tests dibujan con EGL de Mesa, que no necesita
# pantalla (en el CI se instalan sus librerías). Va antes de importar nada que cargue
# MuJoCo, y los procesos hijos del gestor lo heredan. Quien ya fije MUJOCO_GL manda.
if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from src.app import estado_wandb  # noqa: E402 - después de elegir cómo dibuja MuJoCo


@pytest.fixture(autouse=True)
def sin_red_de_wandb(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ningún test comprueba la sesión de W&B contra el servidor.

    La API la comprueba al arrancar; en los tests, la comprobación por defecto dice que
    no está disponible sin tocar la red. Los tests de W&B pasan su propia comprobación.
    """
    monkeypatch.setattr(estado_wandb, "comprobar_sesion_wandb",
                        lambda: "en los tests no se comprueba W&B")
