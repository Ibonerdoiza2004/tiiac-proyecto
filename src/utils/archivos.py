"""Escritura de resultados sin sobrescribir y sin dejar nada a medias.

Lo usan todos los guardados del proyecto (experto, demostraciones, BC, GAIL y
evaluación): comprueban antes de entrenar que la salida está libre y escriben
en un temporal que solo se renombra al final, así que un fallo o un Ctrl+C no
deja una carpeta ni un archivo incompletos.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Callable
from datetime import datetime
from pathlib import Path


def marca_de_tiempo() -> str:
    """Fecha y hora actuales como AAAAMMDD-HHMMSS: el nombre de las salidas por defecto."""
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def comprobar_directorio_libre(directorio: Path) -> Path:
    """Comprueba que se puede guardar en directorio sin sobrescribir nada.

    Vale si no existe o si es una carpeta vacía. Conviene llamarla antes de
    entrenar, para no perder un entrenamiento largo al guardar.

    Returns:
        El directorio como Path.

    Raises:
        FileExistsError: Si existe como archivo o es una carpeta con archivos.
    """
    directorio = Path(directorio)
    if directorio.exists() and (not directorio.is_dir() or any(directorio.iterdir())):
        raise FileExistsError(f"{directorio} ya existe con contenido; no se sobrescribe")
    return directorio


def escribir_carpeta_nueva(directorio: Path, escribir: Callable[[Path], None]) -> None:
    """Escribe una carpeta de resultados entera o nada: nunca la deja a medias.

    escribir(temporal) crea los archivos en una carpeta temporal junto a
    directorio, que después se renombra a directorio (en el mismo sistema de
    archivos, el cambio de nombre es inmediato). Si escribir falla, también por
    Ctrl+C, la temporal se borra y directorio queda como estaba, así que se puede
    volver a guardar ahí. Comprueba antes, como comprobar_directorio_libre, que
    directorio no existe o está vacía.

    Raises:
        FileExistsError: Si directorio existe con contenido, antes o al terminar.
        Lo que lance escribir.
    """
    directorio = comprobar_directorio_libre(directorio)
    directorio.parent.mkdir(parents=True, exist_ok=True)
    temporal = Path(tempfile.mkdtemp(prefix=f".{directorio.name}.", suffix=".tmp",
                                     dir=directorio.parent))
    try:
        escribir(temporal)
        # Una carpeta vacía ya creada se sustituye; con contenido, no.
        comprobar_directorio_libre(directorio)
        if directorio.exists():
            directorio.rmdir()
        temporal.rename(directorio)
    except BaseException:
        shutil.rmtree(temporal, ignore_errors=True)
        raise


def escribir_archivo_nuevo(ruta: Path, escribir: Callable[[Path], None]) -> None:
    """Escribe un archivo entero o nada, sin sobrescribir uno que ya exista.

    Como escribir_carpeta_nueva, pero para un solo archivo: escribir(temporal)
    lo crea en un temporal junto a ruta, que se renombra al final. Si escribir
    falla, también por Ctrl+C, el temporal se borra. Se crean las carpetas
    necesarias.

    Raises:
        FileExistsError: Si ruta ya existe, antes o al terminar.
        Lo que lance escribir.
    """
    ruta = Path(ruta)
    if ruta.exists():
        raise FileExistsError(f"{ruta} ya existe; no se sobrescribe")
    ruta.parent.mkdir(parents=True, exist_ok=True)
    descriptor, nombre = tempfile.mkstemp(prefix=f".{ruta.name}.", suffix=".tmp",
                                          dir=ruta.parent)
    os.close(descriptor)
    temporal = Path(nombre)
    try:
        escribir(temporal)
        if ruta.exists():
            raise FileExistsError(f"{ruta} ya existe; no se sobrescribe")
        temporal.rename(ruta)
    except BaseException:
        temporal.unlink(missing_ok=True)
        raise
