"""Creación reproducible de entornos Gymnasium del proyecto."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv

from src.utils.seed import sembrar_entorno, sembrar_entorno_vectorizado


def crear_entorno(
    cfg: dict[str, Any],
    semilla: int | None,
    render_mode: str | None = None,
) -> gym.Env:
    """Crea el entorno de la configuración y, opcionalmente, lo siembra.

    Args:
        cfg: configuración resuelta que contiene ``entorno.id``.
        semilla: semilla no negativa, o ``None`` para no inicializarla.
        render_mode: modo de renderizado opcional soportado por Gymnasium.

    Returns:
        Entorno Gymnasium creado y listo para usar.

    Raises:
        ValueError: si falta el ID, Gymnasium no puede crear el entorno o la
            semilla no es válida.
    """
    try:
        entorno_id = cfg["entorno"]["id"]
    except (KeyError, TypeError) as error:
        raise ValueError("cfg debe incluir entorno.id") from error
    if not isinstance(entorno_id, str) or not entorno_id:
        raise ValueError("cfg['entorno']['id'] debe ser un ID no vacío")

    try:
        env = gym.make(entorno_id, render_mode=render_mode)
    except (gym.error.Error, TypeError, ValueError) as error:
        raise ValueError(f"No se pudo crear el entorno {entorno_id!r}: {error}") from error
    if semilla is not None:
        try:
            sembrar_entorno(env, semilla)
        except (TypeError, ValueError):
            env.close()
            raise
    return env


def crear_entorno_vectorizado(
    cfg: dict[str, Any], n_entornos: int, semilla: int,
) -> VecEnv:
    """Crea un DummyVecEnv con Monitor y una semilla distinta por instancia.

    El Monitor queda dentro del vectorizador para conservar las recompensas
    reales aunque un VecEnvWrapper exterior cambie las recompensas entregadas
    al algoritmo.

    Args:
        cfg: configuración resuelta que contiene ``entorno.id``.
        n_entornos: número de instancias, debe ser entero positivo.
        semilla: semilla base no negativa; cada instancia usa ``semilla + i``.

    Returns:
        ``DummyVecEnv`` listo para resetear y usar.

    Raises:
        ValueError: si el número de entornos o la semilla no son válidos.
    """
    if (isinstance(n_entornos, bool) or not isinstance(n_entornos, int)
            or n_entornos <= 0):
        raise ValueError("n_entornos debe ser un entero positivo")
    if isinstance(semilla, bool) or not isinstance(semilla, int) or semilla < 0:
        raise ValueError("semilla debe ser un entero no negativo")

    def crear_fabrica() -> Callable[[], gym.Env]:
        """Crea una fábrica aislada para una instancia del vectorizador."""
        def fabrica() -> gym.Env:
            return Monitor(crear_entorno(cfg, None))
        return fabrica

    venv = DummyVecEnv([crear_fabrica() for _ in range(n_entornos)])
    sembrar_entorno_vectorizado(venv, semilla)
    return venv
