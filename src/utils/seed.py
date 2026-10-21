"""Semillas aleatorias, fijadas en UN SOLO SITIO.

Si te encuentras llamando a manual_seed en tres ficheros distintos, es que
esta funcion no se esta usando. El checker avisa de eso.
"""

from __future__ import annotations

import os
import random
from collections.abc import Iterator
from contextlib import contextmanager

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


def validar_semilla(semilla: object) -> int:
    """Comprueba que semilla es un entero no negativo y la devuelve como int.

    Es la única comprobación de semillas del proyecto: la usan crear_generador,
    las funciones de siembra de entornos y quien necesite validar una semilla
    antes de crear objetos. Acepta enteros de Python y de NumPy, no booleanos.

    Raises:
        ValueError: Si no es un entero no negativo.
    """
    if (isinstance(semilla, (bool, np.bool_))
            or not isinstance(semilla, (int, np.integer)) or semilla < 0):
        raise ValueError(f"La semilla debe ser un entero no negativo, no {semilla!r}")
    return int(semilla)


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
    return np.random.default_rng(validar_semilla(semilla))


def semillas_independientes(semilla: int, cantidad: int) -> list[int]:
    """Deriva cantidad semillas independientes a partir de una semilla base.

    Usa numpy.random.SeedSequence(semilla).spawn(cantidad): cada hija depende
    de la base y de su posición, así que bases cercanas no comparten semillas
    (con semilla + i, la base 5 y la 6 compartirían una instancia).

    Args:
        semilla: Semilla base, entero no negativo. Va en configs/.
        cantidad: Número de semillas, entero positivo.

    Returns:
        Lista de cantidad enteros no negativos, la misma para la misma base.

    Raises:
        ValueError: Si la semilla o la cantidad no son válidas.
    """
    base = validar_semilla(semilla)
    if (isinstance(cantidad, (bool, np.bool_)) or not isinstance(cantidad, (int, np.integer))
            or cantidad <= 0):
        raise ValueError(f"cantidad debe ser un entero positivo, no {cantidad!r}")
    hijas = np.random.SeedSequence(base).spawn(int(cantidad))
    return [int(hija.generate_state(1)[0]) for hija in hijas]


USOS_REPETICION = ("modelo", "pares", "entornos", "evaluacion")


def semillas_repeticion(semilla: int, repeticion: int) -> dict[str, int]:
    """Devuelve las semillas de una repetición de BC y GAIL.

    BC y GAIL comparten las de cada repetición, así que la repetición r de los
    dos parte del mismo punto aleatorio. Cada uso tiene su semilla:

    - modelo: fijar_seed antes de crear la red (pesos iniciales, orden de los
      batches y acciones muestreadas al entrenar);
    - pares: crear_generador para elegir los pares de cada partida
      (submuestrear_pares) y los pares expertos de GAIL (muestrear_expertos);
    - entornos: los entornos de entrenamiento de GAIL;
    - evaluacion: las partidas de validación que se registran durante GAIL.

    Salen de numpy.random.SeedSequence(semilla).spawn(...): la repetición r es
    la hija r, y sus usos son nietos. Así cada repetición depende solo de su
    número (no de cuántas haya) y ninguna semilla se repite entre repeticiones
    ni entre usos.

    Args:
        semilla: Semilla base de las repeticiones. Va en configs/
            (experimento.semilla_repeticiones).
        repeticion: Número de repetición, entero no negativo.

    Returns:
        {uso: semilla} para cada uso de USOS_REPETICION.

    Raises:
        ValueError: Si la semilla o la repetición no son válidas.
    """
    base = validar_semilla(semilla)
    if (isinstance(repeticion, (bool, np.bool_)) or not isinstance(repeticion, (int, np.integer))
            or repeticion < 0):
        raise ValueError(f"repeticion debe ser un entero no negativo, no {repeticion!r}")
    hija = np.random.SeedSequence(base).spawn(int(repeticion) + 1)[int(repeticion)]
    return {uso: int(nieta.generate_state(1)[0])
            for uso, nieta in zip(USOS_REPETICION, hija.spawn(len(USOS_REPETICION)), strict=True)}


@contextmanager
def generador_torch_aislado(semilla: int) -> Iterator[None]:
    """Siembra el generador global de PyTorch solo dentro del bloque.

    Al salir, el generador vuelve exactamente al estado que tenía. Sirve para
    que algo aleatorio que usa el generador global sin aceptar uno propio (como
    muestrear acciones con predict de sb3 al evaluar) dependa solo de su
    semilla y no altere lo que venga después, por ejemplo el entrenamiento.
    Solo afecta a la CPU, donde corren los modelos del proyecto.

    Raises:
        ValueError: si la semilla no es un entero no negativo.
    """
    semilla_validada = validar_semilla(semilla)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(semilla_validada)
        yield


def sembrar_entorno(env: gym.Env, semilla: int) -> None:
    """Inicializa el RNG del entorno y sus espacios de observación y acción.

    Primero se siembran los espacios y después se llama una sola vez a
    reset(seed=...), para que ese reset ya use espacios sembrados aunque el
    entorno los muestree. Los resets posteriores continúan la secuencia: la
    primera partida que se juega usa el segundo estado inicial de la semilla.

    Args:
        env: entorno Gymnasium, posiblemente envuelto por ``Monitor``.
        semilla: semilla entera no negativa.

    Raises:
        TypeError: si ``env`` no es un entorno Gymnasium.
        ValueError: si la semilla no es un entero no negativo.
    """
    semilla_validada = validar_semilla(semilla)
    if not isinstance(env, gym.Env):
        raise TypeError("env debe ser un entorno Gymnasium")
    env.action_space.seed(semilla_validada)
    env.observation_space.seed(semilla_validada)
    env.reset(seed=semilla_validada)


def reiniciar_con_semilla(env: gym.Env, semilla: int) -> tuple[np.ndarray, dict]:
    """Empieza una partida con una semilla concreta y la devuelve.

    A diferencia de sembrar_entorno, no gasta ningún reset: la partida que
    devuelve es la primera de esa semilla. Sirve para evaluar cada episodio con
    su propia semilla, igual para todas las políticas. También siembra los
    espacios, por si el entorno los muestrea.

    Args:
        env: entorno Gymnasium, posiblemente envuelto por ``Monitor``.
        semilla: semilla entera no negativa.

    Returns:
        La observación inicial y el diccionario info de ``reset``.

    Raises:
        TypeError: si ``env`` no es un entorno Gymnasium.
        ValueError: si la semilla no es un entero no negativo.
    """
    semilla_validada = validar_semilla(semilla)
    if not isinstance(env, gym.Env):
        raise TypeError("env debe ser un entorno Gymnasium")
    env.action_space.seed(semilla_validada)
    env.observation_space.seed(semilla_validada)
    return env.reset(seed=semilla_validada)


def sembrar_entorno_vectorizado(venv: VecEnv, semilla: int) -> None:
    """Siembra cada instancia de un VecEnv con una semilla independiente.

    Las semillas salen de semillas_independientes(semilla, num_envs), así que
    dos vectorizadores con bases distintas no comparten instancias.

    Args:
        venv: vectorizador que expone sus entornos mediante ``envs``, como
            ``DummyVecEnv``.
        semilla: semilla base entera no negativa.

    Raises:
        TypeError: si ``venv`` no es un VecEnv compatible.
        ValueError: si la semilla no es un entero no negativo.
    """
    semilla_validada = validar_semilla(semilla)
    if not isinstance(venv, VecEnv) or not hasattr(venv, "envs"):
        raise TypeError("venv debe ser un VecEnv que exponga sus entornos internos")
    for env, semilla_instancia in zip(venv.envs, semillas_independientes(
            semilla_validada, len(venv.envs)), strict=True):
        sembrar_entorno(env, semilla_instancia)
