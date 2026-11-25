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
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import spaces

from src.data.biblioteca import (ARCHIVO_FICHA, ARCHIVO_SUBCONJUNTO, Biblioteca, agregar_nuevo,
                                 es_id)
from src.data.demostraciones import (ARCHIVO_DEMOSTRACIONES, ARCHIVO_SPLITS,
                                     cargar_demostraciones, config_demostraciones,
                                     fuente_demostraciones, guardar_demostraciones,
                                     resumen_demostraciones, retornos_por_partida)
from src.data.entornos import espacios_entorno
from src.data.splits import (SplitsDemostraciones, cargar_splits, comprobar_huella,
                             guardar_splits, huella_archivo, validar_splits)
from src.data.tipos import Trayectoria, es_id_episodio
from src.utils.archivos import turno_exclusivo
from src.utils.seed import crear_generador, semillas_independientes, validar_semilla

FORMATO_SUBCONJUNTO = "tiiac-subconjunto"
VERSION_SUBCONJUNTO = 1
CLAVES_SUBCONJUNTO = frozenset({"formato", "version", "conjunto", "huella_dataset", "partidas"})
# Cerrojo de los combinados, en la carpeta de la biblioteca (oculto para el índice): dos
# entrenamientos que piden los mismos datos a la vez no crean dos combinados iguales.
ARCHIVO_CERROJO_COMBINADOS = ".combinados.lock"


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


def proporcion_validacion(cfg: Mapping[str, Any]) -> float:
    """Parte de las partidas que va a validación en un combinado: la del perfil.

    ``demostraciones.partidas_validacion / (partidas_train + partidas_validacion)``: 5 de
    15 en CartPole, MountainCar y Acrobot; 5 de 30 en Reacher y HalfCheetah.

    Raises:
        ValueError: Si la sección demostraciones no es válida (config_demostraciones).
    """
    dcfg = config_demostraciones(cfg)
    return dcfg["partidas_validacion"] / (dcfg["partidas_train"] + dcfg["partidas_validacion"])


def repartir_combinado(metadatos: Mapping[str, Any], huella_dataset: str, proporcion: float,
                       generador: np.random.Generator) -> SplitsDemostraciones:
    """Reparte las partidas de un combinado entre train y validación, por origen.

    Cada NPZ de origen aporta a validación, al azar, la proporción pedida de sus
    partidas, redondeada al entero más cercano (los medios hacia arriba): así la
    validación tiene de todos los datos elegidos. Hay al menos una partida en train y
    una en validación; si el redondeo deja alguno vacío, se completa con una del
    origen con más partidas. Cada partida va entera a un lado. ``train`` sale
    barajado y los subconjuntos son sus N primeras partidas, de 1 a todas, así que
    cualquier cantidad se entrena como en la E1.

    Args:
        metadatos: Metadatos del combinado (combinar_demostraciones): de qué NPZ sale
            cada partida.
        huella_dataset: Huella del NPZ combinado ya guardado.
        proporcion: Parte que va a validación, entre 0 y 1 (proporcion_validacion).
        generador: Generador de src.utils.seed.crear_generador.

    Returns:
        Los splits, validados.

    Raises:
        ValueError: Si los metadatos no son de un combinado, hay menos de dos partidas
            o la proporción o el generador no valen.
    """
    # True y False quedan fuera por el rango (valen 1 y 0).
    if not isinstance(proporcion, (int, float)) or not 0 < proporcion < 1:
        raise ValueError(f"proporcion debe estar entre 0 y 1, sin incluirlos, no {proporcion!r}")
    if not isinstance(generador, np.random.Generator):
        raise ValueError("generador debe ser un numpy.random.Generator de crear_generador")
    if not isinstance(metadatos, Mapping) or metadatos.get("fuente") != "combinado":
        raise ValueError("Solo se reparten así los combinados (fuente «combinado»)")
    grupos: dict[str, list[int]] = {}
    for partida in metadatos.get("partidas", []):
        grupos.setdefault(partida["huella_dataset"], []).append(int(partida["id_episodio"]))
    if sum(len(ids) for ids in grupos.values()) < 2:
        raise ValueError("Hacen falta al menos dos partidas: una de train y una de validación")

    # Por huella y por id, para que el resultado no dependa del orden de llegada. Se
    # barajan posiciones, no ids (como crear_splits).
    porciones: list[tuple[list[int], list[int]]] = []
    for huella in sorted(grupos):
        ids = sorted(grupos[huella])
        barajados = [ids[k] for k in generador.permutation(len(ids))]
        cuantas = math.floor(len(ids) * proporcion + 0.5)
        porciones.append((barajados[cuantas:], barajados[:cuantas]))
    # El origen con más partidas (el primero por huella, si empatan) completa lo vacío.
    tamanos = [len(t) + len(v) for t, v in porciones]
    train_mayor, validacion_mayor = porciones[tamanos.index(max(tamanos))]
    if not any(v for _, v in porciones):
        validacion_mayor.append(train_mayor.pop())
    elif not any(t for t, _ in porciones):
        train_mayor.append(validacion_mayor.pop())

    train_ordenado = sorted(i for t, _ in porciones for i in t)
    train = tuple(train_ordenado[k] for k in generador.permutation(len(train_ordenado)))
    splits = SplitsDemostraciones(
        train=train, validacion=tuple(sorted(i for _, v in porciones for i in v)),
        subconjuntos={n: train[:n] for n in range(1, len(train) + 1)},
        huella_dataset=huella_dataset)
    validar_splits(splits, [i for ids in grupos.values() for i in ids])
    return splits


# --- datos de una petición de entrenamiento --------------------------------------------

def _elemento_datos(biblioteca: Biblioteca, id_elemento: object, entorno: str) -> dict[str, Any]:
    """Un conjunto o subconjunto válido de demostraciones del entorno."""
    if not isinstance(id_elemento, str):
        raise ValueError(f"Los datos se indican por su id de la biblioteca, no {id_elemento!r}")
    try:
        elemento = biblioteca.obtener(id_elemento)
    except KeyError:
        raise ValueError(f"No existen las demostraciones {id_elemento}") from None
    if elemento["tipo"] != "demostraciones":
        raise ValueError(f"{id_elemento} no son demostraciones")
    if elemento["entorno"] != entorno:
        raise ValueError(f"{id_elemento} son demostraciones de {elemento['entorno']}, no de "
                         f"{entorno}")
    if not elemento["valido"]:
        raise ValueError(f"Las demostraciones {id_elemento} no son válidas: {elemento['motivo']}")
    return elemento


def _comprobar_ids(ids: object) -> list[str]:
    """Lista no vacía de ids sin repetir."""
    if isinstance(ids, (str, bytes, Mapping)) or not isinstance(ids, Sequence) or not ids:
        raise ValueError("Hay que elegir al menos un conjunto de demostraciones")
    if len(set(ids)) != len(ids):
        raise ValueError("Hay datos repetidos en la selección")
    return list(ids)


def clave_seleccion(metadatos: Mapping[str, Any], proporcion: float, semilla: int) -> str:
    """Huella de unos datos combinados y de su reparto: con la misma, el mismo combinado.

    Depende solo de qué partidas hay (huella del NPZ de origen e id) y de con qué se
    reparten (proporción y semilla), no del orden en que se eligieron.
    """
    contenido = {"partidas": sorted([p["huella_dataset"], int(p["id_original"])]
                                    for p in metadatos["partidas"]),
                 "proporcion": proporcion, "semilla": int(semilla)}
    return hashlib.sha256(json.dumps(contenido, sort_keys=True).encode("utf-8")).hexdigest()


def _combinado_existente(biblioteca: Biblioteca, entorno: str, clave: str) -> str | None:
    """Id de un combinado válido con sus splits y esa clave, si ya está en la biblioteca."""
    for elemento in biblioteca.listar(entorno, ("demostraciones",)):
        if not (elemento["valido"] and elemento["fuente"] == "combinado"
                and elemento["detalles"].get("splits")):
            continue
        ficha = json.loads((biblioteca.ruta(elemento["id"]) / ARCHIVO_FICHA)
                           .read_text(encoding="utf-8"))
        if ficha.get("seleccion") == clave:
            return elemento["id"]
    return None


def identidades_originales(metadatos: Mapping[str, Any], huella: str,
                           ids: Sequence[int]) -> dict[int, tuple[str, int]]:
    """De qué NPZ y partida originales sale cada partida de unas demostraciones.

    Una partida de un combinado es una copia de otra: su identidad es la de su origen
    (sus metadatos dicen la huella y el id originales). Las de cualquier otro NPZ son
    originales: su huella y su id. Así, una partida elegida por dos caminos (un conjunto
    y un combinado que la incluye, o un subconjunto de ese combinado) cuenta una vez y
    nunca acaba a la vez en train y en validación.

    Args:
        metadatos: Metadatos del NPZ.
        huella: Huella del NPZ.
        ids: id_episodio de las partidas que se toman de él.

    Returns:
        (huella original, id original) de cada id.
    """
    if fuente_demostraciones(metadatos) == "combinado":
        origen = {p["id_episodio"]: (p["huella_dataset"], int(p["id_original"]))
                  for p in metadatos["partidas"]}
        return {int(i): origen[i] for i in ids}
    return {int(i): (huella, int(i)) for i in ids}


def _por_origen(partidas: Sequence[Trayectoria], identidades: Mapping[int, tuple[str, int]],
                ) -> list[tuple[str, list[Trayectoria]]]:
    """Partidas agrupadas por su NPZ original, con su id original."""
    grupos: dict[str, list[Trayectoria]] = {}
    for partida in partidas:
        huella, original = identidades[partida.id_episodio]
        grupos.setdefault(huella, []).append(dataclasses.replace(partida, id_episodio=original))
    return list(grupos.items())


def _npz_de(biblioteca: Biblioteca, elemento: Mapping[str, Any]) -> tuple[Path, list[int] | None]:
    """NPZ del que salen unas demostraciones y las partidas que toman (None: todas)."""
    carpeta = biblioteca.ruta(elemento["id"])
    if elemento["detalles"].get("tipo") == "subconjunto":
        datos = leer_subconjunto(carpeta)
        return biblioteca.ruta(datos["conjunto"]) / ARCHIVO_DEMOSTRACIONES, datos["partidas"]
    return carpeta / ARCHIVO_DEMOSTRACIONES, None


def _origenes(biblioteca: Biblioteca, elementos: Sequence[Mapping[str, Any]],
              espacios: tuple[spaces.Space, spaces.Space]) -> list[tuple[str, list[Trayectoria]]]:
    """(huella del NPZ original, partidas con su id original) de lo elegido."""
    origenes = []
    for elemento in elementos:
        npz, _ = _npz_de(biblioteca, elemento)
        if elemento["detalles"].get("tipo") == "subconjunto":
            partidas, datos = cargar_subconjunto(biblioteca, elemento["id"], *espacios)
            huella = datos["huella_dataset"]
            metadatos = resumen_demostraciones(npz)["metadatos"]
        else:
            partidas, metadatos = cargar_demostraciones(npz, *espacios)
            huella = huella_archivo(npz)
        identidades = identidades_originales(metadatos, huella,
                                             [p.id_episodio for p in partidas])
        origenes.extend(_por_origen(partidas, identidades))
    return origenes


def preparar_datos(biblioteca: Biblioteca, ids: Sequence[str], cfg: Mapping[str, Any]) -> str:
    """Conjunto con splits con el que se entrena una petición de BC o GAIL.

    - Un solo conjunto que ya tiene splits (como los congelados): ese mismo, así que
      se entrena exactamente como en la E1.
    - En otro caso (varios, un subconjunto o un conjunto sin splits): se combinan sus
      partidas sin repetir, por su NPZ y partida originales también si alguna llega a
      través de un combinado (identidades_originales, combinar_demostraciones); se
      reparten por NPZ original
      (repartir_combinado, con la proporción del perfil y la semilla de los splits de
      la E1, la segunda de semillas_independientes(demostraciones.semilla, 2)) y se
      guarda en la biblioteca como conjunto combinado, con origen «dashboard». Si ya
      hay uno con las mismas partidas y el mismo reparto, se reutiliza.

    Args:
        biblioteca: Índice de la biblioteca.
        ids: Conjuntos y subconjuntos elegidos (la lista ``datos`` de la petición).
        cfg: Configuración resuelta del entorno (origen.entorno, entorno.id y
            demostraciones).

    Returns:
        El id del conjunto con splits que hay que cargar (barrido.cargar_datos con su
        carpeta).

    Raises:
        ValueError: Si la selección está vacía, repite, o algún elemento no existe, no
            es de demostraciones o de ese entorno, o no es válido.
    """
    entorno, gym = cfg["origen"]["entorno"], cfg["entorno"]["id"]
    ids = _comprobar_ids(ids)
    elementos = [_elemento_datos(biblioteca, i, entorno) for i in ids]
    if (len(elementos) == 1 and elementos[0]["detalles"].get("tipo") == "conjunto"
            and elementos[0]["detalles"].get("splits")):
        return elementos[0]["id"]
    proporcion = proporcion_validacion(cfg)
    semilla_splits = semillas_independientes(
        validar_semilla(config_demostraciones(cfg)["semilla"]), 2)[1]
    espacios = espacios_entorno(dict(cfg))
    with turno_exclusivo(biblioteca.raiz / ARCHIVO_CERROJO_COMBINADOS):
        partidas, metadatos = combinar_demostraciones(_origenes(biblioteca, elementos, espacios),
                                                      gym)
        clave = clave_seleccion(metadatos, proporcion, semilla_splits)
        existente = _combinado_existente(biblioteca, entorno, clave)
        if existente is not None:
            return existente

        def escribir(carpeta: Path) -> None:
            """Guarda el NPZ combinado y sus splits."""
            npz = guardar_demostraciones(carpeta / ARCHIVO_DEMOSTRACIONES, partidas, metadatos,
                                         *espacios)
            guardar_splits(carpeta / ARCHIVO_SPLITS,
                           repartir_combinado(metadatos, huella_archivo(npz), proporcion,
                                              crear_generador(semilla_splits)))

        nombre = f"Combinado · {gym.split('-v')[0]} · {len(partidas)} partidas"
        return agregar_nuevo(biblioteca.raiz, entorno, "demostraciones", nombre, "dashboard",
                             escribir, fuente="combinado", seleccion=clave)["id"]


def resumen_seleccion(biblioteca: Biblioteca, ids: Sequence[str],
                      entorno: str) -> dict[str, int]:
    """Partidas y pares de unos datos elegidos, sin repetir, sin cargar las partidas.

    Una partida que llega por varios caminos (un conjunto y un subconjunto suyo, o un
    combinado que la incluye) cuenta una vez, como al combinar (identidades_originales).

    Args:
        biblioteca: Índice de la biblioteca.
        ids: Conjuntos y subconjuntos elegidos.
        entorno: Clave del entorno.

    Returns:
        ``{"partidas", "pares"}``.

    Raises:
        ValueError: Como preparar_datos.
    """
    pasos_de: dict[tuple[str, int], int] = {}
    for elemento in (_elemento_datos(biblioteca, i, entorno) for i in _comprobar_ids(ids)):
        npz, elegidas = _npz_de(biblioteca, elemento)
        resumen = resumen_demostraciones(npz)
        pasos = resumen["pasos_por_partida"]
        elegidas = list(pasos) if elegidas is None else elegidas
        identidades = identidades_originales(resumen["metadatos"], huella_archivo(npz),
                                             elegidas)
        pasos_de.update({identidades[i]: pasos[i] for i in elegidas})
    return {"partidas": len(pasos_de), "pares": sum(pasos_de.values())}


def retornos_train(biblioteca: Biblioteca, id_conjunto: str) -> list[float]:
    """Retorno de cada partida de train de un conjunto con splits, en el orden de train.

    Ayuda a elegir cuántas partidas usar en BC y GAIL: con N, se entrena con las N
    primeras de esta lista.

    Raises:
        ValueError: Si no es un conjunto válido con splits.
    """
    elemento = _conjunto(biblioteca, id_conjunto)
    if not elemento["detalles"].get("splits"):
        raise ValueError(f"El conjunto {id_conjunto} no tiene splits")
    carpeta = biblioteca.ruta(id_conjunto)
    npz = carpeta / ARCHIVO_DEMOSTRACIONES
    retornos = retornos_por_partida(npz)
    splits = cargar_splits(carpeta / ARCHIVO_SPLITS, huella_archivo(npz), list(retornos))
    return [retornos[i] for i in splits.train]

