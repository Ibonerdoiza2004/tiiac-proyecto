"""Semillas aleatorias, fijadas en UN SOLO SITIO.

Si te encuentras llamando a manual_seed en tres ficheros distintos, es que
esta funcion no se esta usando. El checker avisa de eso.
"""

from __future__ import annotations

import os
import random

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.vec_env import VecEnv


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


def _validar_semilla_entorno(semilla: int) -> int:
    """Valida una semilla y la convierte al entero nativo de Python."""
    if (isinstance(semilla, (bool, np.bool_))
            or not isinstance(semilla, (int, np.integer)) or semilla < 0):
        raise ValueError(f"La semilla debe ser un entero no negativo, no {semilla!r}")
    return int(semilla)


def sembrar_entorno(env: gym.Env, semilla: int) -> None:
    """Inicializa el RNG del entorno y sus espacios de observación y acción.

    Se llama a ``reset(seed=...)`` una sola vez para inicializar el generador
    del entorno; los resets posteriores continúan su secuencia aleatoria.

    Args:
        env: entorno Gymnasium, posiblemente envuelto por ``Monitor``.
        semilla: semilla entera no negativa.

    Raises:
        TypeError: si ``env`` no es un entorno Gymnasium.
        ValueError: si la semilla no es un entero no negativo.
    """
    semilla_validada = _validar_semilla_entorno(semilla)
    if not isinstance(env, gym.Env):
        raise TypeError("env debe ser un entorno Gymnasium")
    env.reset(seed=semilla_validada)
    env.action_space.seed(semilla_validada)
    env.observation_space.seed(semilla_validada)


def sembrar_entorno_vectorizado(venv: VecEnv, semilla: int) -> None:
    """Siembra las instancias de un VecEnv con semillas consecutivas.

    Args:
        venv: vectorizador que expone sus entornos mediante ``envs``, como
            ``DummyVecEnv``.
        semilla: semilla base entera no negativa.

    Raises:
        TypeError: si ``venv`` no es un VecEnv compatible.
        ValueError: si la semilla no es un entero no negativo.
    """
    semilla_validada = _validar_semilla_entorno(semilla)
    if not isinstance(venv, VecEnv) or not hasattr(venv, "envs"):
        raise TypeError("venv debe ser un VecEnv que exponga sus entornos internos")
    for indice, env in enumerate(venv.envs):
        sembrar_entorno(env, semilla_validada + indice)
