"""Creación reproducible de entornos Gymnasium del proyecto."""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv

from src.utils.seed import sembrar_entorno, sembrar_entorno_vectorizado, validar_semilla
from src.utils.validacion import es_entero_positivo

# MuJoCo 3.16 declaró obsoleto settotalmass y avisa cada vez que compila un modelo que lo
# usa, como el HalfCheetah de Gymnasium 1.3.0. El atributo
# sigue escalando las masas: solo se calla ese aviso, la simulación no cambia.
AVISO_MUJOCO_IGNORADO = "compiler attribute 'settotalmass' is deprecated"


def filtrar_aviso_mujoco(mensaje: str) -> None:
    """Escribe en stderr un aviso de MuJoCo, salvo el de settotalmass.

    Args:
        mensaje: Texto del aviso, tal como lo pasa MuJoCo.
    """
    if AVISO_MUJOCO_IGNORADO not in mensaje:
        print(f"WARNING: {mensaje}", file=sys.stderr)


def _instalar_filtro_mujoco() -> None:
    """Pone filtrar_aviso_mujoco como gestor de avisos de MuJoCo si nadie ha puesto otro.

    Sin MuJoCo instalado no hace nada. Con el gestor propio MuJoCo deja de escribir
    MUJOCO_LOG.TXT.
    """
    try:
        import mujoco
    except ImportError:
        return
    if mujoco.get_mju_user_warning() is None:
        mujoco.set_mju_user_warning(filtrar_aviso_mujoco)


def crear_entorno(
    cfg: dict[str, Any],
    semilla: int | None,
    render_mode: str | None = None,
) -> gym.Env:
    """Crea el entorno de la configuración y, opcionalmente, lo siembra.

    Antes instala el filtro de avisos de MuJoCo (filtrar_aviso_mujoco).

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

    _instalar_filtro_mujoco()
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


def espacios_entorno(cfg: dict[str, Any]) -> tuple[gym.Space, gym.Space]:
    """Devuelve los espacios de observación y de acción del entorno de la configuración.

    Crea el entorno solo para leerlos y lo cierra.

    Raises:
        ValueError: Si no se puede crear el entorno (ver crear_entorno).
    """
    entorno = crear_entorno(cfg, None)
    try:
        return entorno.observation_space, entorno.action_space
    finally:
        entorno.close()


def crear_entorno_vectorizado(
    cfg: dict[str, Any], n_entornos: int, semilla: int,
) -> VecEnv:
    """Crea un DummyVecEnv con Monitor y una semilla independiente por instancia.

    El Monitor queda dentro del vectorizador para conservar las recompensas
    reales aunque un VecEnvWrapper exterior cambie las recompensas entregadas
    al algoritmo.

    Args:
        cfg: configuración resuelta que contiene ``entorno.id``.
        n_entornos: número de instancias, debe ser entero positivo.
        semilla: semilla base no negativa; las de cada instancia salen de
            ``semillas_independientes`` y no se solapan entre bases cercanas.

    Returns:
        ``DummyVecEnv`` listo para resetear y usar.

    Raises:
        ValueError: si el número de entornos o la semilla no son válidos.
    """
    if not es_entero_positivo(n_entornos):
        raise ValueError("n_entornos debe ser un entero positivo")
    # Se comprueba antes de crear los entornos, para no dejar ninguno abierto.
    validar_semilla(semilla)

    def crear_fabrica() -> Callable[[], gym.Env]:
        """Crea una fábrica aislada para una instancia del vectorizador."""
        def fabrica() -> gym.Env:
            """Crea una instancia sin sembrar, envuelta en Monitor; se siembra después."""
            return Monitor(crear_entorno(cfg, None))
        return fabrica

    venv = DummyVecEnv([crear_fabrica() for _ in range(int(n_entornos))])
    sembrar_entorno_vectorizado(venv, semilla)
    return venv
