"""Selecciones de demostraciones: subconjuntos y combinaciones.

Lo que el dashboard hace con las demostraciones antes de entrenar: elegir partidas de
un conjunto, juntar varios conjuntos en uno y repartir sus partidas entre train y
validación. Siempre con partidas enteras (no se recortan: decisión del usuario), así
que cada partida va entera a train o a validación, y siempre se acaba en un NPZ con sus
splits, como en la E1.

Un subconjunto no copia datos: es un ``subconjunto.json`` en la biblioteca que dice de
qué conjunto sale (su id y la huella de su NPZ) y qué partidas toma (sus id_episodio).
Solo se saca de un conjunto, no de otro subconjunto. Si el conjunto falta o ha cambiado,
el subconjunto deja de ser válido.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import spaces

from src.data.biblioteca import ARCHIVO_SUBCONJUNTO, Biblioteca, agregar_nuevo, es_id
from src.data.demostraciones import (ARCHIVO_DEMOSTRACIONES, cargar_demostraciones,
                                     resumen_demostraciones)
from src.data.splits import comprobar_huella, huella_archivo
from src.data.tipos import Trayectoria, es_id_episodio

FORMATO_SUBCONJUNTO = "tiiac-subconjunto"
VERSION_SUBCONJUNTO = 1
CLAVES_SUBCONJUNTO = frozenset({"formato", "version", "conjunto", "huella_dataset", "partidas"})


def comprobar_subconjunto(datos: object) -> dict[str, Any]:
    """Comprueba la forma de un subconjunto.json, sin mirar su conjunto.

    Args:
        datos: ``{"formato", "version", "conjunto", "huella_dataset", "partidas"}``, con
            ``partidas`` una lista no vacía de id_episodio sin repetir.

    Returns:
        Los mismos datos.

    Raises:
        ValueError: Si falta o sobra algo o algún valor no vale.
    """
    if not isinstance(datos, Mapping) or set(datos) != CLAVES_SUBCONJUNTO:
        raise ValueError(f"Un subconjunto tiene exactamente {sorted(CLAVES_SUBCONJUNTO)}")
    if datos["formato"] != FORMATO_SUBCONJUNTO or datos["version"] != VERSION_SUBCONJUNTO:
        raise ValueError(f"Formato de subconjunto desconocido: {datos['formato']!r} "
                         f"versión {datos['version']!r}")
    if not es_id(datos["conjunto"]):
        raise ValueError(f"conjunto no es un id de la biblioteca: {datos['conjunto']!r}")
    comprobar_huella(datos["huella_dataset"])
    partidas = datos["partidas"]
    if not isinstance(partidas, list) or not partidas:
        raise ValueError("Un subconjunto toma al menos una partida")
    for indice, identificador in enumerate(partidas):
        if not es_id_episodio(identificador):
            raise ValueError(f"partidas[{indice}] no es un id de partida: {identificador!r}")
    if len(set(partidas)) != len(partidas):
        raise ValueError("Hay partidas repetidas en el subconjunto")
    return dict(datos)


def comprobar_contra_conjunto(datos: Mapping[str, Any], huella: str,
                              pasos_por_partida: Mapping[int, int]) -> None:
    """Comprueba que un subconjunto encaja con su conjunto tal como está ahora.

    Args:
        datos: Subconjunto ya comprobado (comprobar_subconjunto).
        huella: Huella actual del NPZ del conjunto.
        pasos_por_partida: Pasos de cada partida del conjunto.

    Raises:
        ValueError: Si el NPZ ha cambiado o falta alguna partida.
    """
    if datos["huella_dataset"] != huella:
        raise ValueError("el conjunto ha cambiado desde que se creó el subconjunto "
                         "(la huella de su NPZ no coincide)")
    faltan = [i for i in datos["partidas"] if i not in pasos_por_partida]
    if faltan:
        raise ValueError(f"las partidas {faltan} no están en el conjunto")


def _conjunto(biblioteca: Biblioteca, id_conjunto: str) -> dict[str, Any]:
    """Elemento de un conjunto válido de la biblioteca (no un subconjunto)."""
    try:
        elemento = biblioteca.obtener(id_conjunto)
    except KeyError:
        raise ValueError(f"No existe el conjunto {id_conjunto}") from None
    if elemento["tipo"] != "demostraciones" or elemento["detalles"].get("tipo") == "subconjunto":
        raise ValueError(f"{id_conjunto} no es un conjunto de demostraciones")
    if not elemento["valido"]:
        raise ValueError(f"El conjunto {id_conjunto} no es válido: {elemento['motivo']}")
    return elemento


def crear_subconjunto(biblioteca: Biblioteca, id_conjunto: str, partidas: Sequence[int],
                      nombre: str) -> dict[str, Any]:
    """Crea un subconjunto de un conjunto y lo añade a la biblioteca.

    Args:
        biblioteca: Índice de la biblioteca.
        id_conjunto: Conjunto del que sale (no otro subconjunto).
        partidas: id_episodio de las partidas elegidas, enteras.
        nombre: Nombre para el usuario.

    Returns:
        La ficha del subconjunto (origen «dashboard», la fuente del conjunto).

    Raises:
        ValueError: Si el conjunto no existe o no vale, o las partidas no son suyas.
    """
    conjunto = _conjunto(biblioteca, id_conjunto)
    npz = biblioteca.ruta(id_conjunto) / ARCHIVO_DEMOSTRACIONES
    huella = huella_archivo(npz)
    if isinstance(partidas, (str, bytes, Mapping)) or not isinstance(partidas, Sequence):
        raise ValueError("partidas debe ser una lista de id_episodio")
    datos = comprobar_subconjunto({
        "formato": FORMATO_SUBCONJUNTO, "version": VERSION_SUBCONJUNTO, "conjunto": id_conjunto,
        "huella_dataset": huella, "partidas": list(partidas)})
    # Como int y en orden: el mismo subconjunto da el mismo archivo.
    datos["partidas"] = sorted(int(i) for i in datos["partidas"])
    comprobar_contra_conjunto(datos, huella, resumen_demostraciones(npz)["pasos_por_partida"])
    texto = json.dumps(datos, ensure_ascii=False, indent=2)

    def escribir(carpeta: Path) -> None:
        """Escribe el subconjunto.json."""
        (carpeta / ARCHIVO_SUBCONJUNTO).write_text(texto, encoding="utf-8")

    return agregar_nuevo(biblioteca.raiz, conjunto["entorno"], "demostraciones", nombre,
                         "dashboard", escribir, fuente=conjunto["fuente"])


def leer_subconjunto(carpeta: Path) -> dict[str, Any]:
    """Lee y comprueba el subconjunto.json de una carpeta de la biblioteca.

    Raises:
        ValueError: Si no se puede leer o no tiene la forma de un subconjunto.
    """
    try:
        datos = json.loads((Path(carpeta) / ARCHIVO_SUBCONJUNTO).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"no se pudo leer {ARCHIVO_SUBCONJUNTO}: {error}") from error
    return comprobar_subconjunto(datos)


def cargar_subconjunto(biblioteca: Biblioteca, id_subconjunto: str,
                       espacio_observacion: spaces.Space, espacio_accion: spaces.Space,
                       ) -> tuple[list[Trayectoria], dict[str, Any]]:
    """Partidas de un subconjunto: las elegidas de su conjunto, enteras.

    Args:
        biblioteca: Índice de la biblioteca.
        id_subconjunto: Subconjunto válido de la biblioteca.
        espacio_observacion: Espacio de observación del entorno.
        espacio_accion: Espacio de acción del entorno.

    Returns:
        Las partidas, en el orden del subconjunto y con sus ids, y el contenido de su
        subconjunto.json (de qué conjunto salen).

    Raises:
        ValueError: Si no es un subconjunto válido o su conjunto ha cambiado.
    """
    try:
        elemento = biblioteca.obtener(id_subconjunto)
    except KeyError:
        raise ValueError(f"No existe el subconjunto {id_subconjunto}") from None
    if elemento["detalles"].get("tipo") != "subconjunto":
        raise ValueError(f"{id_subconjunto} no es un subconjunto")
    if not elemento["valido"]:
        raise ValueError(f"El subconjunto {id_subconjunto} no es válido: {elemento['motivo']}")
    datos = leer_subconjunto(biblioteca.ruta(id_subconjunto))
    npz = biblioteca.ruta(datos["conjunto"]) / ARCHIVO_DEMOSTRACIONES
    trayectorias, _ = cargar_demostraciones(npz, espacio_observacion, espacio_accion)
    por_id = {t.id_episodio: t for t in trayectorias}
    # Se vuelve a comprobar con lo que se ha cargado, por si el NPZ cambió entretanto.
    comprobar_contra_conjunto(datos, huella_archivo(npz),
                              {i: t.numero_pasos for i, t in por_id.items()})
    return [por_id[i] for i in datos["partidas"]], datos


# Campos de una partida que tienen que coincidir si sale dos veces del mismo NPZ.
_CAMPOS_PARTIDA = ("estados", "acciones", "siguientes_estados", "recompensas_originales",
                   "terminados", "truncados")


def combinar_demostraciones(origenes: Sequence[tuple[str, Sequence[Trayectoria]]],
                            entorno: str) -> tuple[list[Trayectoria], dict[str, Any]]:
    """Junta partidas de varios NPZ en una sola colección, sin repetir ninguna.

    Una partida se identifica por la huella de su NPZ y su id_episodio: si llega dos
    veces (por ejemplo, de un conjunto y de un subconjunto suyo), cuenta una vez, así
    que los pares repetidos cuentan una vez. Las partidas se ordenan por (huella,
    id original) y reciben ids nuevos de 0 a N − 1: con los mismos datos, en cualquier
    orden, el resultado es el mismo. Los arrays no se copian.

    Args:
        origenes: Pares (huella del NPZ, partidas que se toman de él); las partidas,
            enteras y con su id_episodio del NPZ.
        entorno: Id de Gymnasium del entorno de todas ellas.

    Returns:
        Las partidas con sus ids nuevos y los metadatos de un NPZ combinado
        (comprobar_metadatos): fuente «combinado», entorno y, de cada partida, de qué
        NPZ e id sale.

    Raises:
        ValueError: Si no hay ninguna partida, una huella o una partida no vale, o la
            misma partida de un NPZ llega con contenidos distintos.
    """
    if isinstance(origenes, (str, bytes)) or not isinstance(origenes, Sequence):
        raise ValueError("origenes debe ser una lista de (huella, partidas)")
    if not isinstance(entorno, str) or not entorno.strip():
        raise ValueError("entorno debe ser el id de Gymnasium del entorno")
    unicas: dict[tuple[str, int], Trayectoria] = {}
    for indice, origen in enumerate(origenes):
        if not isinstance(origen, (tuple, list)) or len(origen) != 2:
            raise ValueError(f"origenes[{indice}] debe ser (huella, partidas)")
        huella, partidas = origen
        comprobar_huella(huella)
        if isinstance(partidas, (str, bytes)) or not isinstance(partidas, Sequence):
            raise ValueError(f"origenes[{indice}]: partidas debe ser una lista de Trayectoria")
        for partida in partidas:
            if not isinstance(partida, Trayectoria) or not es_id_episodio(partida.id_episodio):
                raise ValueError(f"origenes[{indice}]: se esperaban objetos Trayectoria con "
                                 "id_episodio")
            clave = (huella, int(partida.id_episodio))
            anterior = unicas.setdefault(clave, partida)
            if anterior is not partida and not all(
                    np.array_equal(getattr(anterior, c), getattr(partida, c))
                    for c in _CAMPOS_PARTIDA):
                raise ValueError(f"La partida {clave[1]} del NPZ {huella[:12]} llega dos veces "
                                 "con contenidos distintos")
    if not unicas:
        raise ValueError("No hay ninguna partida que combinar")
    claves = sorted(unicas)
    trayectorias = [dataclasses.replace(unicas[clave], id_episodio=nuevo)
                    for nuevo, clave in enumerate(claves)]
    metadatos = {"fuente": "combinado", "entorno": entorno,
                 "partidas": [{"id_episodio": nuevo, "huella_dataset": huella,
                               "id_original": original}
                              for nuevo, (huella, original) in enumerate(claves)]}
    return trayectorias, metadatos

