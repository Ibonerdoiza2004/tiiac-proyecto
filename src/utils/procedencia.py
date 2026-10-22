"""Metadatos reproducibles sobre el código, las versiones y el hardware."""

from __future__ import annotations

import platform
import subprocess
from pathlib import Path
from typing import Any

import gymnasium as gym
import sb3_contrib
import stable_baselines3
import torch


def estado_git() -> dict[str, Any]:
    """Devuelve el commit del repositorio y si había cambios sin commit.

    Si git no está disponible o esta carpeta no es un repositorio, ambos
    valores son None en vez de fallar.
    """
    raiz = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=raiz, capture_output=True,
                                text=True, check=True, timeout=10).stdout.strip()
        cambios = subprocess.run(["git", "status", "--porcelain"], cwd=raiz,
                                 capture_output=True, text=True, check=True,
                                 timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "cambios_sin_commit": None}
    return {"commit": commit, "cambios_sin_commit": bool(cambios)}


def versiones_y_hardware() -> dict[str, Any]:
    """Devuelve versiones de librerías y descripción del hardware."""
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "stable_baselines3": stable_baselines3.__version__,
        "sb3_contrib": sb3_contrib.__version__,
        "gymnasium": gym.__version__,
        "plataforma": platform.platform(),
        "procesador": platform.processor() or platform.machine(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def ruta_registrada(ruta: Path) -> str:
    """Devuelve una ruta relativa al directorio actual o solo su nombre."""
    try:
        return ruta.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return ruta.name
