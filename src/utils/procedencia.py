"""Metadatos reproducibles sobre el código, las versiones y el hardware."""

from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import sb3_contrib
import stable_baselines3
import torch

# Segundos que se espera a cada orden de git, para no bloquear el guardado si se cuelga.
ESPERA_GIT = 10


def estado_git() -> dict[str, Any]:
    """Devuelve el commit del repositorio y si había cambios sin commit.

    Si git no está disponible o esta carpeta no es un repositorio, ambos
    valores son None en vez de fallar.
    """
    raiz = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=raiz, capture_output=True,
                                text=True, check=True, timeout=ESPERA_GIT).stdout.strip()
        cambios = subprocess.run(["git", "status", "--porcelain"], cwd=raiz,
                                 capture_output=True, text=True, check=True,
                                 timeout=ESPERA_GIT).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "cambios_sin_commit": None}
    return {"commit": commit, "cambios_sin_commit": bool(cambios)}


def avisar_estado_git(codigo: dict[str, Any]) -> None:
    """Avisa por la salida de errores si el resultado no se puede reproducir desde git.

    Es el aviso con el que terminan los scripts que guardan resultados: si no se
    pudo leer el commit, o si había cambios sin commit al generarlos.

    Args:
        codigo: Lo que devuelve estado_git (o lo que se guardó de él).
    """
    if codigo["commit"] is None:
        print("aviso: no se pudo leer el commit de git; los metadatos guardados no dicen con "
              "qué código se generó el resultado", file=sys.stderr)
    elif codigo["cambios_sin_commit"]:
        print("aviso: había cambios sin commit; el resultado no se puede reproducir exactamente "
              "desde el repositorio", file=sys.stderr)


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


def a_json(valor: object) -> object:
    """Convierte escalares de NumPy a tipos de Python, para usarla como default de json.dumps.

    Raises:
        TypeError: Si el valor no se puede convertir, como hace json por defecto.
    """
    if isinstance(valor, (np.integer, np.floating, np.bool_)):
        return valor.item()
    raise TypeError(f"{type(valor).__name__} no se puede guardar en JSON")
