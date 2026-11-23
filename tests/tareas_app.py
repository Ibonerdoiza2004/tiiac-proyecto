"""Tareas de juguete para probar el gestor de trabajos (src/app/trabajos.py).

Se ejecutan en procesos hijos (spawn), que las importan por su ruta
"tareas_app:funcion". No importan NumPy ni PyTorch arriba para poder comprobar que
el proceso hijo fija los hilos antes de cargarlos.
"""

from __future__ import annotations

import os
import signal
import sys
import time
from typing import Any

PAUSA = 0.01


def sumar(argumentos: dict[str, Any], canal: Any) -> dict[str, Any]:
    """Manda un progreso por número y devuelve la suma."""
    for i, numero in enumerate(argumentos["numeros"]):
        canal.progreso({"paso": i, "valor": numero})
    return {"suma": sum(argumentos["numeros"])}


def contar_hasta_orden(argumentos: dict[str, Any], canal: Any) -> dict[str, Any]:
    """Cuenta pasos sin parar, con un progreso por paso, hasta recibir {"tipo": "finalizar"}."""
    recibidas = []
    lotes = []  # cuántas órdenes devolvió cada llamada que trajo alguna
    paso = 0
    while True:
        nuevas = canal.ordenes()
        if nuevas:
            lotes.append(len(nuevas))
        for orden in nuevas:
            recibidas.append(orden)
            if orden.get("tipo") == "finalizar":
                return {"pasos": paso, "ordenes": recibidas, "lotes": lotes}
        canal.progreso({"paso": paso})
        paso += 1
        time.sleep(PAUSA)


def esperar(argumentos: dict[str, Any], canal: Any) -> dict[str, Any]:
    """Espera los segundos pedidos y termina."""
    time.sleep(argumentos.get("segundos", 0.0))
    return {"esperado": argumentos.get("segundos", 0.0)}


def fallar(argumentos: dict[str, Any], canal: Any) -> dict[str, Any]:
    """Lanza un error de la tarea."""
    raise ValueError(argumentos.get("mensaje", "mal"))


def morir(argumentos: dict[str, Any], canal: Any) -> dict[str, Any]:
    """Muere sin enviar el resultado, como si el sistema lo matara por falta de memoria."""
    canal.progreso({"paso": 0})
    os.kill(os.getpid(), signal.SIGKILL)
    return {}


def no_devolver_diccionario(argumentos: dict[str, Any], canal: Any) -> Any:
    """Devuelve algo que no es un diccionario."""
    return [1, 2, 3]


def mandar_imagenes(argumentos: dict[str, Any], canal: Any) -> dict[str, Any]:
    """Manda imágenes numeradas y, después, un progreso."""
    for i in range(argumentos["cuantas"]):
        canal.imagen(f"imagen-{i}".encode())
    canal.progreso({"imagenes": argumentos["cuantas"]})
    return {"enviadas": argumentos["cuantas"]}


def hilos(argumentos: dict[str, Any], canal: Any) -> dict[str, Any]:
    """Informa de los hilos con los que corre el proceso hijo."""
    numpy_antes = "numpy" in sys.modules
    entorno = {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                              "OPENBLAS_NUM_THREADS")}
    import torch  # noqa: PLC0415 - aquí a propósito: después de fijar las variables

    return {"numpy_antes": numpy_antes, "entorno": entorno,
            "torch": torch.get_num_threads(),
            "training": sorted(m for m in sys.modules if m.startswith("src.training"))}


no_es_funcion = 3


def imagenes_hasta_orden(argumentos: dict[str, Any], canal: Any) -> dict[str, Any]:
    """Manda una imagen numerada por paso, sin parar, hasta recibir {"tipo": "finalizar"}."""
    paso = 0
    while True:
        if any(orden.get("tipo") == "finalizar" for orden in canal.ordenes()):
            return {"imagenes": paso}
        canal.imagen(f"imagen-{paso}".encode())
        paso += 1
        time.sleep(PAUSA)


def progreso_no_finito(argumentos: dict[str, Any], canal: Any) -> dict[str, Any]:
    """Manda medidas que JSON no admite (NaN e infinito) y termina."""
    canal.progreso({"nan": float("nan"), "infinito": float("inf"), "normal": 1.5})
    return {"perdida": float("nan")}


def eventos_propios(argumentos: dict[str, Any], canal: Any) -> dict[str, Any]:
    """Emite dos partidas terminadas, un progreso y espera a {"tipo": "finalizar"}."""
    for i in range(2):
        canal.evento("partida_terminada", {"partida": {"indice": i, "pasos": 10 + i,
                                                       "retorno": 1.5, "semilla": 7}})
    canal.progreso({"paso": 1})
    while not any(o.get("tipo") == "finalizar" for o in canal.ordenes()):
        time.sleep(PAUSA)
    canal.evento("partida_terminada", {"partida": {"indice": 2, "pasos": 3, "retorno": 0.0,
                                                   "semilla": 8}})
    return {"partidas": 3}


def evento_no_valido(argumentos: dict[str, Any], canal: Any) -> dict[str, Any]:
    """Intenta emitir un evento con un tipo o unos datos que no valen."""
    canal.evento(argumentos["tipo"], argumentos.get("datos"))
    return {}
