"""Checkpoints de los entrenamientos del dashboard, para reanudar uno cortado.

Cada entrenamiento con checkpoint tiene su carpeta, que elige quien lo lanza (dentro de
``<biblioteca>/.en_curso/``, CARPETA_EN_CURSO en src/data/biblioteca.py), con dos
archivos:

- ``argumentos.json``: el método y los argumentos con los que empezó, para listar los
  que se pueden reanudar y relanzarlos tal cual.
- ``checkpoint.pt``: el último estado guardado (el del bucle, el progreso mandado hasta
  entonces y la configuración resuelta), que se lee con ``weights_only=True``.

Mientras un entrenamiento la usa, la carpeta está reservada (en_uso, con el cerrojo
``.en_uso``): otro que quiera usarla a la vez, por ejemplo al pulsar dos veces
«Reanudar», se rechaza al momento (CarpetaOcupada) en lugar de pisarse con el primero.
El checkpoint se escribe en un temporal que se renombra al final, así que un corte a
mitad deja el anterior intacto. Al terminar bien, el trabajador borra la carpeta; si
falla o se corta, se queda para reanudar.
"""

from __future__ import annotations

import json
import os
import pickle
import shutil
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any

import torch

from src.utils.archivos import turno_exclusivo
from src.utils.procedencia import a_json

ARCHIVO_ARGUMENTOS = "argumentos.json"
ARCHIVO_CHECKPOINT = "checkpoint.pt"
ARCHIVO_EN_USO = ".en_uso"
FORMATO_CHECKPOINT = "tiiac-checkpoint"
VERSION_CHECKPOINT = 1
# Argumentos que definen el entrenamiento: para reanudar tienen que ser los mismos. Los
# demás (vista, W&B, calidad de la vista, rutas) pueden cambiar.
CLAVES_REANUDAR = ("entorno", "nombre", "ajustes", "semilla", "datos", "cantidad")
# Partes de la configuración resuelta que pueden cambiar al reanudar.
SECCIONES_LIBRES = ("wandb", "origen")


class CarpetaOcupada(ValueError):
    """Otro entrenamiento está usando la carpeta de checkpoints."""


@contextmanager
def en_uso(carpeta: Path) -> Iterator[None]:
    """Reserva la carpeta de un entrenamiento mientras dura el bloque, sin esperar.

    La crea si no existe. El cerrojo se suelta al salir del bloque, también si algo
    falla, y si el proceso muere.

    Raises:
        CarpetaOcupada: Si otro entrenamiento (otro proceso u otro hilo) la está usando.
    """
    carpeta = Path(carpeta)
    with ExitStack() as pila:
        try:
            pila.enter_context(turno_exclusivo(carpeta / ARCHIVO_EN_USO, esperar=False))
        except BlockingIOError:
            raise CarpetaOcupada(f"Ya hay un entrenamiento usando {carpeta}: espera a que "
                                 "termine") from None
        yield


def _a_json(argumentos: dict[str, Any]) -> Any:
    """Los argumentos tal como quedan en argumentos.json."""
    return json.loads(json.dumps(argumentos, default=a_json))


def preparar_en_curso(carpeta: Path, metodo: str,
                      argumentos: dict[str, Any]) -> dict[str, Any] | None:
    """Prepara la carpeta de un entrenamiento y devuelve su último checkpoint, si hay.

    Se llama con la carpeta reservada (en_uso). Si no tiene argumentos.json (es nueva,
    o se cortó justo al crearla), los escribe y empieza de cero. Si los tiene, comprueba
    que es del mismo método y con los mismos CLAVES_REANUDAR, y devuelve el checkpoint
    (leer_checkpoint), o None si se cortó antes de guardar el primero.

    Raises:
        ValueError: Si la carpeta es de otro entrenamiento, tiene un checkpoint sin sus
            argumentos o su contenido no vale.
    """
    carpeta = Path(carpeta)
    ruta = carpeta / ARCHIVO_ARGUMENTOS
    actuales = _a_json(argumentos)
    if not ruta.exists():
        if (carpeta / ARCHIVO_CHECKPOINT).exists():
            raise ValueError(f"{carpeta} tiene un checkpoint pero no sus argumentos: no se "
                             "sabe de qué entrenamiento es")
        carpeta.mkdir(parents=True, exist_ok=True)
        temporal = carpeta / f".{ARCHIVO_ARGUMENTOS}.tmp"
        temporal.write_text(json.dumps({"metodo": metodo, "argumentos": actuales},
                                       ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporal, ruta)
        return None
    try:
        guardados = json.loads(ruta.read_text(encoding="utf-8"))
        metodo_guardado, argumentos_guardados = guardados["metodo"], guardados["argumentos"]
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise ValueError(f"{carpeta} no tiene un {ARCHIVO_ARGUMENTOS} válido: {error}") from error
    if metodo_guardado != metodo:
        raise ValueError(f"{carpeta} es de un entrenamiento de {metodo_guardado}, no de {metodo}")
    distintas = [clave for clave in CLAVES_REANUDAR
                 if argumentos_guardados.get(clave) != actuales.get(clave)]
    if distintas:
        raise ValueError(f"No se puede reanudar con otros {distintas}: {carpeta} es de otro "
                         "entrenamiento")
    return leer_checkpoint(carpeta, metodo)


def escribir_checkpoint(carpeta: Path, metodo: str, contenido: dict[str, Any]) -> None:
    """Escribe el checkpoint de un entrenamiento, sustituyendo el anterior de golpe.

    Args:
        carpeta: Carpeta del entrenamiento (preparar_en_curso).
        metodo: experto, gail o bc.
        contenido: Lo que se guarda (tensores, números, textos, listas y diccionarios).
    """
    carpeta = Path(carpeta)
    temporal = carpeta / f".{ARCHIVO_CHECKPOINT}.tmp"
    torch.save({"formato": FORMATO_CHECKPOINT, "version": VERSION_CHECKPOINT,
                "metodo": metodo, **contenido}, temporal)
    os.replace(temporal, carpeta / ARCHIVO_CHECKPOINT)


def leer_checkpoint(carpeta: Path, metodo: str) -> dict[str, Any] | None:
    """Lee el checkpoint de la carpeta (None si todavía no hay ninguno).

    Raises:
        ValueError: Si está dañado, no es un checkpoint o es de otro método.
    """
    ruta = Path(carpeta) / ARCHIVO_CHECKPOINT
    if not ruta.is_file():
        return None
    try:
        contenido = torch.load(ruta, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, EOFError, pickle.UnpicklingError) as error:
        raise ValueError(f"Checkpoint dañado o no válido: {ruta}") from error
    if (not isinstance(contenido, dict) or contenido.get("formato") != FORMATO_CHECKPOINT
            or contenido.get("version") != VERSION_CHECKPOINT):
        raise ValueError(f"{ruta} no es un checkpoint de esta versión")
    if contenido.get("metodo") != metodo:
        raise ValueError(f"{ruta} es de {contenido.get('metodo')}, no de {metodo}")
    return contenido


def comprobar_misma_config(guardada: Any, actual: dict[str, Any]) -> None:
    """Comprueba que la configuración resuelta es la del checkpoint (salvo SECCIONES_LIBRES).

    Raises:
        ValueError: Si cambió algo, por ejemplo configs/ entre el corte y la reanudación.
    """
    if not isinstance(guardada, dict):
        raise ValueError("El checkpoint no tiene su configuración")
    claves = (set(guardada) | set(actual)) - set(SECCIONES_LIBRES)
    distintas = sorted(c for c in claves if _a_json({"v": guardada.get(c)})
                       != _a_json({"v": actual.get(c)}))
    if distintas:
        raise ValueError(f"La configuración cambió desde el checkpoint en {distintas}: no se "
                         "puede reanudar")


def borrar_en_curso(carpeta: Path) -> None:
    """Borra la carpeta de un entrenamiento terminado (nada si no existe)."""
    if Path(carpeta).exists():
        shutil.rmtree(carpeta)
