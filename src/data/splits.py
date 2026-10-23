"""Reparto de demostraciones por trayectoria y subconjuntos anidados de train.

Las partidas se reparten enteras entre train y validación, de modo que ningún
paso de una misma partida aparece en ambos. Los subconjuntos (por ejemplo 1, 4,
7 y 10 partidas) salen de train y están anidados: cada uno contiene al anterior,
para que BC y GAIL comparen cantidades crecientes de los mismos datos.

El reparto se guarda en un JSON aparte del NPZ de demostraciones, junto con la
huella SHA-256 de ese NPZ, para que unos splits no se usen con otro dataset.

Cada archivo de splits corresponde a un único NPZ. Para usar demostraciones de
varios archivos, por ejemplo varias sesiones grabadas en el aplicativo, se
combinan antes en un NPZ nuevo con IDs sin repetir, y los splits se crean sobre
él (decisión del 13 de octubre de 2026). Así los IDs siguen siendo únicos y la
huella identifica exactamente los datos usados.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.data.tipos import Trayectoria, es_id_episodio
from src.utils.validacion import es_entero

_FORMATO = "tiiac-splits"
_VERSION = 1
_TAMANO_BLOQUE = 1 << 20


@dataclass(frozen=True, slots=True)
class SplitsDemostraciones:
    """IDs de las partidas de train, validación y de cada subconjunto de train.

    subconjuntos asocia cada cantidad con los IDs que la forman; están anidados
    y contenidos en train. huella_dataset es el SHA-256, en hexadecimal, del
    archivo de demostraciones para el que se crearon.
    """

    train: tuple[int, ...]
    validacion: tuple[int, ...]
    subconjuntos: Mapping[int, tuple[int, ...]]
    huella_dataset: str


def huella_archivo(ruta: str | Path) -> str:
    """Calcula la huella SHA-256 de un archivo, leyéndolo por bloques.

    Args:
        ruta: Archivo cuya huella se calcula, normalmente el NPZ de demostraciones.

    Returns:
        La huella en hexadecimal, 64 caracteres. Cambia si cambia un solo byte.

    Raises:
        FileNotFoundError: La ruta no existe.
    """
    resumen = hashlib.sha256()
    with Path(ruta).open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(_TAMANO_BLOQUE), b""):
            resumen.update(bloque)
    return resumen.hexdigest()


def comprobar_cantidades(
    cantidad_train: int, cantidad_validacion: int, cantidades_subconjuntos: Sequence[int],
) -> None:
    """Rechaza cantidades que no sean enteros positivos coherentes entre sí.

    Es la comprobación que aplica crear_splits. Sirve también para validar la
    configuración antes de grabar unas demostraciones que luego no se podrían
    repartir.

    Raises:
        ValueError: Cantidades no enteras o no positivas, subconjuntos vacíos o
            no estrictamente crecientes, o un subconjunto mayor que train.
    """
    for nombre, valor in (("cantidad_train", cantidad_train),
                          ("cantidad_validacion", cantidad_validacion)):
        if not es_entero(valor) or valor <= 0:
            raise ValueError(f"{nombre} debe ser un entero positivo, no {valor!r}")
    if (isinstance(cantidades_subconjuntos, (str, bytes))
            or not isinstance(cantidades_subconjuntos, Sequence) or not cantidades_subconjuntos):
        raise ValueError("cantidades_subconjuntos debe ser una secuencia no vacía de enteros")
    if not all(es_entero(c) and c > 0 for c in cantidades_subconjuntos):
        raise ValueError("Las cantidades de los subconjuntos deben ser enteros positivos")
    if any(a >= b for a, b in zip(cantidades_subconjuntos, cantidades_subconjuntos[1:])):
        raise ValueError("Las cantidades de los subconjuntos deben ser estrictamente crecientes")
    if cantidades_subconjuntos[-1] > cantidad_train:
        raise ValueError(f"El subconjunto mayor ({cantidades_subconjuntos[-1]}) supera las "
                         f"{cantidad_train} partidas de train")


def comprobar_huella(huella: object) -> None:
    """Exige una huella SHA-256 en hexadecimal en minúsculas, como la de huella_archivo.

    La usan los splits y los checkpoints para citar el dataset con el que se crearon.

    Raises:
        ValueError: Si no es un texto de 64 caracteres hexadecimales en minúsculas.
    """
    if (not isinstance(huella, str) or len(huella) != 64
            or any(c not in "0123456789abcdef" for c in huella)):
        raise ValueError("La huella del dataset debe ser un SHA-256 hexadecimal de 64 caracteres")


def crear_splits(
    trayectorias: Sequence[Trayectoria],
    cantidad_train: int,
    cantidad_validacion: int,
    cantidades_subconjuntos: Sequence[int],
    generador: np.random.Generator,
    huella_dataset: str,
) -> SplitsDemostraciones:
    """Reparte partidas enteras entre train y validación y elige los subconjuntos.

    Ordena los IDs, los baraja con el generador y asigna los cantidad_train
    primeros a train y el resto a validación. El orden barajado de train define
    los subconjuntos: cada uno son sus N primeras partidas, así que están
    anidados. El resultado depende solo de los IDs y del generador, no del orden
    en que lleguen las trayectorias. Se usan todas las partidas: el total debe
    ser exactamente cantidad_train + cantidad_validacion.

    Args:
        trayectorias: Partidas ya validadas; solo se usan sus IDs.
        cantidad_train: Número de partidas de train.
        cantidad_validacion: Número de partidas de validación.
        cantidades_subconjuntos: Tamaños crecientes de los subconjuntos de train.
        generador: Generador de src.utils.seed.crear_generador.
        huella_dataset: SHA-256 del archivo de demostraciones (huella_archivo).

    Returns:
        Los splits validados.

    Raises:
        ValueError: Cantidades incoherentes, IDs no enteros o repetidos, un total
            distinto del pedido, generador o huella no válidos.
    """
    comprobar_cantidades(cantidad_train, cantidad_validacion, cantidades_subconjuntos)
    comprobar_huella(huella_dataset)
    if not isinstance(generador, np.random.Generator):
        raise ValueError("generador debe ser un numpy.random.Generator de crear_generador")
    if (isinstance(trayectorias, (str, bytes)) or not isinstance(trayectorias, Sequence)
            or not all(isinstance(t, Trayectoria) for t in trayectorias)):
        raise ValueError("Se requiere una secuencia de objetos Trayectoria")
    ids = [t.id_episodio for t in trayectorias]
    if not all(es_id_episodio(i) for i in ids):
        raise ValueError("Los id_episodio deben ser enteros en el rango de int64")
    ids = [int(i) for i in ids]
    if len(set(ids)) != len(ids):
        raise ValueError("Hay id_episodio repetidos")
    total = cantidad_train + cantidad_validacion
    if len(ids) != total:
        raise ValueError(f"Se pidieron {cantidad_train} + {cantidad_validacion} = {total} "
                         f"partidas y hay {len(ids)}")

    # Se barajan posiciones y no los IDs: así los IDs nunca pasan por un array de
    # NumPy, que podría convertirlos a float64 y perder precisión.
    ordenados = sorted(ids)
    barajados = [ordenados[k] for k in generador.permutation(len(ordenados))]
    train = tuple(barajados[:cantidad_train])
    splits = SplitsDemostraciones(
        train=train,
        validacion=tuple(barajados[cantidad_train:]),
        subconjuntos={int(c): train[:c] for c in cantidades_subconjuntos},
        huella_dataset=huella_dataset,
    )
    validar_splits(splits, ids)
    return splits


def validar_splits(splits: SplitsDemostraciones, ids_disponibles: Iterable[int]) -> None:
    """Comprueba que los splits reparten exactamente las partidas disponibles.

    Train y validación no son vacíos, no repiten IDs, no comparten partidas y
    juntos cubren exactamente ids_disponibles. Cada subconjunto tiene tantas
    partidas como indica su cantidad, está contenido en train y contiene al
    subconjunto anterior.

    Args:
        splits: Splits a comprobar.
        ids_disponibles: IDs de las partidas del dataset con el que se usarán.

    Raises:
        ValueError: Si no se cumple alguna de las condiciones.
    """
    if not isinstance(splits, SplitsDemostraciones):
        raise ValueError("Se esperaba un objeto SplitsDemostraciones")
    comprobar_huella(splits.huella_dataset)
    disponibles = list(ids_disponibles)
    if (not all(es_id_episodio(i) for i in disponibles)
            or len(set(disponibles)) != len(disponibles)):
        raise ValueError("Los IDs disponibles deben ser enteros del rango de int64 sin repetir")
    for nombre, ids in (("train", splits.train), ("validacion", splits.validacion)):
        if not isinstance(ids, tuple) or not ids or not all(es_id_episodio(i) for i in ids):
            raise ValueError(f"{nombre} debe ser una tupla no vacía de enteros del rango de int64")
        if len(set(ids)) != len(ids):
            raise ValueError(f"{nombre} repite partidas")
    compartidas = set(splits.train) & set(splits.validacion)
    if compartidas:
        raise ValueError(f"Partidas en train y validación a la vez: {sorted(compartidas)}")
    usados = set(splits.train) | set(splits.validacion)
    if usados != {int(i) for i in disponibles}:
        raise ValueError(f"Los splits no cubren exactamente las partidas disponibles; faltan "
                         f"{sorted(set(map(int, disponibles)) - usados)}, sobran "
                         f"{sorted(usados - set(map(int, disponibles)))}")
    if not isinstance(splits.subconjuntos, Mapping) or not splits.subconjuntos:
        raise ValueError("subconjuntos debe ser un diccionario no vacío")
    for cantidad in splits.subconjuntos:
        if not es_entero(cantidad) or cantidad <= 0:
            raise ValueError(f"Cantidad de subconjunto no válida: {cantidad!r}")
    anterior: set[int] = set()
    for cantidad in sorted(splits.subconjuntos):
        ids = splits.subconjuntos[cantidad]
        if (not isinstance(ids, tuple) or not all(es_id_episodio(i) for i in ids)
                or len(ids) != cantidad or len(set(ids)) != cantidad):
            raise ValueError(f"El subconjunto {cantidad} debe tener {cantidad} partidas distintas")
        if not set(ids) <= set(splits.train):
            raise ValueError(f"El subconjunto {cantidad} contiene partidas fuera de train")
        if not anterior <= set(ids):
            raise ValueError(f"El subconjunto {cantidad} no contiene al anterior")
        anterior = set(ids)


def guardar_splits(ruta: str | Path, splits: SplitsDemostraciones) -> Path:
    """Guarda los splits en JSON, sin sobrescribir un archivo existente.

    El JSON incluye formato, versión, train, validación, subconjuntos y la huella
    del dataset. Las claves se ordenan para que los mismos splits produzcan
    siempre el mismo texto. Se crean los directorios necesarios.

    Args:
        ruta: Destino, terminado en .json.
        splits: Splits a guardar; se validan internamente sin conocer el dataset.

    Returns:
        La ruta del archivo creado.

    Raises:
        ValueError: Extensión distinta de .json o splits incoherentes.
        FileExistsError: El destino ya existe; se conserva sin cambios.
        OSError: Fallo de escritura; se elimina el archivo incompleto.
    """
    destino = Path(ruta)
    if destino.suffix != ".json":
        raise ValueError("La ruta de splits debe terminar en .json")
    if not isinstance(splits, SplitsDemostraciones):
        raise ValueError("Se esperaba un objeto SplitsDemostraciones")
    validar_splits(splits, sorted(set(splits.train) | set(splits.validacion)))
    contenido = {
        "formato": _FORMATO, "version": _VERSION, "huella_dataset": splits.huella_dataset,
        "train": [int(i) for i in splits.train],
        "validacion": [int(i) for i in splits.validacion],
        "subconjuntos": {str(int(c)): [int(i) for i in ids]
                         for c, ids in splits.subconjuntos.items()},
    }
    texto = json.dumps(contenido, ensure_ascii=False, sort_keys=True, indent=2)
    destino.parent.mkdir(parents=True, exist_ok=True)
    archivo = destino.open("x", encoding="utf-8")
    try:
        with archivo:
            archivo.write(texto + "\n")
    except BaseException:
        destino.unlink(missing_ok=True)
        raise
    return destino


def _rechazar_constante(texto: str) -> Any:
    """Rechaza NaN e Infinity, que no son IDs válidos."""
    raise ValueError(f"El archivo de splits contiene un valor no válido: {texto}")


def _lista_de_enteros(valor: object, nombre: str) -> tuple[int, ...]:
    """Convierte una lista JSON de enteros en tupla o lanza ValueError."""
    if not isinstance(valor, list) or not all(es_id_episodio(i) for i in valor):
        raise ValueError(f"{nombre} debe ser una lista de enteros del rango de int64")
    return tuple(valor)


def cargar_splits(
    ruta: str | Path, huella_dataset: str, ids_disponibles: Iterable[int],
) -> SplitsDemostraciones:
    """Carga splits guardados y comprueba que corresponden al dataset indicado.

    Comprueba formato y versión, tipos, que la huella guardada coincide con
    huella_dataset y, con validar_splits, que reparten exactamente
    ids_disponibles.

    Args:
        ruta: Archivo JSON creado por guardar_splits.
        huella_dataset: huella_archivo del NPZ de demostraciones que se va a usar.
        ids_disponibles: IDs de las partidas cargadas de ese NPZ.

    Returns:
        Los splits, con las cantidades de los subconjuntos como enteros.

    Raises:
        FileNotFoundError: La ruta no existe.
        ValueError: Extensión, JSON, formato o versión no válidos, huella
            distinta o splits incoherentes con las partidas disponibles.
    """
    origen = Path(ruta)
    if origen.suffix != ".json":
        raise ValueError("La ruta de splits debe terminar en .json")
    comprobar_huella(huella_dataset)
    try:
        contenido = json.loads(origen.read_text(encoding="utf-8"),
                               parse_constant=_rechazar_constante)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"El archivo de splits no es JSON válido: {origen}") from exc
    if not isinstance(contenido, dict):
        raise ValueError("El archivo de splits debe contener un objeto JSON")
    esperadas = {"formato", "version", "huella_dataset", "train", "validacion", "subconjuntos"}
    if set(contenido) != esperadas:
        raise ValueError(f"Claves del archivo de splits: se esperaban {sorted(esperadas)}")
    if contenido["formato"] != _FORMATO:
        raise ValueError(f"Formato desconocido: se esperaba {_FORMATO!r}")
    if contenido["version"] != _VERSION or isinstance(contenido["version"], bool):
        raise ValueError(
            f"Versión no soportada: {contenido['version']!r}; se admite {_VERSION}")
    if contenido["huella_dataset"] != huella_dataset:
        raise ValueError("Estos splits se crearon para otro archivo de demostraciones: "
                         "la huella SHA-256 no coincide")
    crudos = contenido["subconjuntos"]
    if not isinstance(crudos, dict):
        raise ValueError("subconjuntos debe ser un objeto JSON")
    subconjuntos: dict[int, tuple[int, ...]] = {}
    for clave, ids in crudos.items():
        if not clave.isdigit() or str(int(clave)) != clave:
            raise ValueError(f"Cantidad de subconjunto no válida: {clave!r}")
        subconjuntos[int(clave)] = _lista_de_enteros(ids, f"El subconjunto {clave}")
    splits = SplitsDemostraciones(
        train=_lista_de_enteros(contenido["train"], "train"),
        validacion=_lista_de_enteros(contenido["validacion"], "validacion"),
        subconjuntos=subconjuntos,
        huella_dataset=contenido["huella_dataset"],
    )
    validar_splits(splits, ids_disponibles)
    return splits
