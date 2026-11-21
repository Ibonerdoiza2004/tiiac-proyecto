"""Selecciones del dashboard: subconjuntos de partidas y de tramos de pares.

Una selección elige, de un único archivo de demostraciones, qué partidas se
usan y, de cada una, qué tramos de pasos (pares estado-acción). Como los
splits, guarda la huella SHA-256 de ese NPZ para que no se use con otro. Para
elegir partidas de varios archivos se combinan antes con combinar_demostraciones
y las selecciones de cada archivo se trasladan al combinado (trasladar_seleccion).

Los tramos son intervalos semiabiertos de pasos, [inicio, fin), contando desde
0: el tramo (0, 20) son los pares 0 a 19, que la interfaz muestra como 1–20. Una
selección guarda los tramos de cada partida ordenados y fundidos, sin solapes ni
tramos contiguos, así que al unir selecciones los pares repetidos cuentan una
sola vez.

Para entrenar BC con una selección, repartir_validacion separa con un generador
partidas enteras de validación, de modo que ningún paso de una misma partida
está en train y en validación.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.data.splits import comprobar_huella
from src.data.tipos import LoteTransiciones, Trayectoria, es_id_episodio
from src.utils.validacion import es_entero, es_real_finito

Tramo = tuple[int, int]

_FORMATO = "tiiac-seleccion"
_VERSION = 1


@dataclass(frozen=True, slots=True)
class SeleccionPares:
    """Partidas elegidas de un archivo de demostraciones y sus tramos de pasos.

    partidas asocia cada id_episodio, en orden creciente, con sus tramos
    [inicio, fin) ordenados, sin solapes ni contiguos (normalizar_tramos).
    huella_dataset es el SHA-256 del NPZ del que salen.
    """

    huella_dataset: str
    partidas: Mapping[int, tuple[Tramo, ...]]

    @property
    def numero_partidas(self) -> int:
        """Devuelve cuántas partidas aportan algún par."""
        return len(self.partidas)

    @property
    def numero_pares(self) -> int:
        """Devuelve cuántos pares distintos hay en la selección."""
        return sum(fin - inicio for tramos in self.partidas.values() for inicio, fin in tramos)


def _por_id(trayectorias: Sequence[Trayectoria]) -> dict[int, Trayectoria]:
    """Indexa las partidas por id_episodio y rechaza colecciones no válidas."""
    if (isinstance(trayectorias, (str, bytes)) or not isinstance(trayectorias, Sequence)
            or not all(isinstance(t, Trayectoria) for t in trayectorias)):
        raise ValueError("Se requiere una secuencia de objetos Trayectoria")
    if not all(es_id_episodio(t.id_episodio) for t in trayectorias):
        raise ValueError("Los id_episodio deben ser enteros en el rango de int64")
    por_id = {int(t.id_episodio): t for t in trayectorias}
    if len(por_id) != len(trayectorias):
        raise ValueError("Hay id_episodio repetidos")
    return por_id


def normalizar_tramos(tramos: Sequence[Sequence[int]], numero_pasos: int) -> tuple[Tramo, ...]:
    """Comprueba los tramos de una partida y los ordena y funde.

    Args:
        tramos: Secuencia no vacía de pares (inicio, fin), enteros con
            0 <= inicio < fin <= numero_pasos.
        numero_pasos: Pasos de la partida.

    Returns:
        Los tramos ordenados, con los que se solapan o se tocan fundidos en uno.

    Raises:
        ValueError: Sin tramos, un tramo que no es un par de enteros, vacío,
            invertido o fuera de la partida.
    """
    if not es_entero(numero_pasos) or numero_pasos <= 0:
        raise ValueError(f"numero_pasos debe ser un entero positivo, no {numero_pasos!r}")
    if isinstance(tramos, (str, bytes)) or not isinstance(tramos, Sequence) or not tramos:
        raise ValueError("Cada partida elegida necesita al menos un tramo de pares")
    limpios: list[list[int]] = []
    for tramo in tramos:
        if (isinstance(tramo, (str, bytes)) or not isinstance(tramo, Sequence)
                or len(tramo) != 2 or not all(es_entero(extremo) for extremo in tramo)):
            raise ValueError(f"Un tramo es un par de enteros (inicio, fin), no {tramo!r}")
        inicio, fin = int(tramo[0]), int(tramo[1])
        if not 0 <= inicio < fin <= numero_pasos:
            raise ValueError(f"Tramo ({inicio}, {fin}) no válido: hace falta "
                             f"0 <= inicio < fin <= {numero_pasos}")
        limpios.append([inicio, fin])
    limpios.sort()
    fundidos = [limpios[0]]
    for inicio, fin in limpios[1:]:
        if inicio <= fundidos[-1][1]:
            fundidos[-1][1] = max(fundidos[-1][1], fin)
        else:
            fundidos.append([inicio, fin])
    return tuple((inicio, fin) for inicio, fin in fundidos)


def crear_seleccion(
    trayectorias: Sequence[Trayectoria], huella_dataset: str,
    elecciones: Mapping[int, Sequence[Sequence[int]] | None],
) -> SeleccionPares:
    """Crea una selección de partidas enteras o de tramos de sus pares.

    Args:
        trayectorias: Partidas del archivo de demostraciones.
        huella_dataset: huella_archivo de ese NPZ.
        elecciones: Por id_episodio, sus tramos [inicio, fin), o None para la
            partida entera. Tiene que haber al menos una partida.

    Returns:
        La selección, con las partidas en orden de id_episodio y los tramos
        normalizados.

    Raises:
        ValueError: Ninguna partida elegida, IDs que no están en trayectorias,
            tramos no válidos o huella mal formada.
    """
    comprobar_huella(huella_dataset)
    por_id = _por_id(trayectorias)
    if not isinstance(elecciones, Mapping) or not elecciones:
        raise ValueError("La selección necesita al menos una partida")
    if not all(es_id_episodio(identificador) for identificador in elecciones):
        raise ValueError("Los id_episodio deben ser enteros en el rango de int64")
    faltan = sorted({int(i) for i in elecciones} - set(por_id))
    if faltan:
        raise ValueError(f"Partidas que no están en el archivo: {faltan}")
    partidas: dict[int, tuple[Tramo, ...]] = {}
    for identificador in sorted(elecciones, key=int):
        pasos = por_id[int(identificador)].numero_pasos
        tramos = elecciones[identificador]
        partidas[int(identificador)] = normalizar_tramos(
            [(0, pasos)] if tramos is None else tramos, pasos)
    return SeleccionPares(huella_dataset=huella_dataset, partidas=partidas)


def _comprobar_forma(seleccion: object) -> None:
    """Comprueba lo que se puede sin las partidas: tipo, huella, IDs y tramos normalizados."""
    if not isinstance(seleccion, SeleccionPares):
        raise ValueError("Se esperaba un objeto SeleccionPares")
    comprobar_huella(seleccion.huella_dataset)
    if not isinstance(seleccion.partidas, Mapping) or not seleccion.partidas:
        raise ValueError("La selección necesita al menos una partida")
    for identificador, tramos in seleccion.partidas.items():
        if not es_id_episodio(identificador):
            raise ValueError(f"id_episodio no válido en la selección: {identificador!r}")
        if not isinstance(tramos, tuple) or not tramos:
            raise ValueError(f"La partida {identificador} necesita una tupla de tramos")
        # Sin la partida no se conoce su longitud: basta con que el último paso elegido
        # quepa; validar_seleccion comprueba después la longitud real.
        limite = max((t[1] for t in tramos if isinstance(t, tuple) and len(t) == 2
                      and es_entero(t[1])), default=1)
        if normalizar_tramos(tramos, max(int(limite), 1)) != tramos:
            raise ValueError(f"Los tramos de la partida {identificador} no están normalizados")


def validar_seleccion(seleccion: SeleccionPares, trayectorias: Sequence[Trayectoria]) -> None:
    """Comprueba que la selección es coherente con las partidas del archivo.

    Hay al menos una partida, todas están en trayectorias y sus tramos están
    normalizados (ordenados, fundidos y sin pasar del final de la partida).

    Raises:
        ValueError: Si no se cumple alguna de las condiciones.
    """
    _comprobar_forma(seleccion)
    por_id = _por_id(trayectorias)
    for identificador, tramos in seleccion.partidas.items():
        if int(identificador) not in por_id:
            raise ValueError(f"La partida {identificador} no está en el archivo")
        pasos = por_id[int(identificador)].numero_pasos
        if tramos[-1][1] > pasos:
            raise ValueError(f"Los tramos de la partida {identificador} pasan de sus "
                             f"{pasos} pasos")


def unir_selecciones(selecciones: Sequence[SeleccionPares]) -> SeleccionPares:
    """Une selecciones del mismo archivo; los pares repetidos cuentan una vez.

    Raises:
        ValueError: Ninguna selección, objetos que no son SeleccionPares o
            selecciones de archivos distintos (hay que combinarlos y trasladarlas antes).
    """
    if (isinstance(selecciones, (str, bytes)) or not isinstance(selecciones, Sequence)
            or not selecciones):
        raise ValueError("Se requiere una secuencia no vacía de objetos SeleccionPares")
    for seleccion in selecciones:
        _comprobar_forma(seleccion)
    huellas = {s.huella_dataset for s in selecciones}
    if len(huellas) != 1:
        raise ValueError("Solo se pueden unir selecciones del mismo archivo; combina antes los "
                         "archivos y traslada las selecciones")
    tramos: dict[int, list[Tramo]] = {}
    for seleccion in selecciones:
        for identificador, propios in seleccion.partidas.items():
            tramos.setdefault(int(identificador), []).extend(propios)
    # normalizar_tramos solo necesita un límite superior: los tramos de cada
    # selección ya caben en su partida.
    partidas = {identificador: normalizar_tramos(tramos[identificador],
                                                 max(fin for _, fin in tramos[identificador]))
                for identificador in sorted(tramos)}
    return SeleccionPares(huella_dataset=huellas.pop(), partidas=partidas)


def trasladar_seleccion(seleccion: SeleccionPares, metadatos_combinado: Mapping[str, Any],
                        huella_combinado: str) -> SeleccionPares:
    """Pasa una selección de un archivo a los IDs del archivo combinado que lo incluye.

    Usa los pares [ID antiguo, ID nuevo] que combinar_demostraciones guarda en
    los origenes de cada archivo de entrada. Dos partidas del original que sean
    la misma en el combinado juntan sus tramos.

    Args:
        seleccion: Selección del archivo original.
        metadatos_combinado: Metadatos del archivo combinado.
        huella_combinado: huella_archivo del archivo combinado.

    Raises:
        ValueError: Si el combinado no incluye el archivo de la selección o sus
            origenes no cubren las partidas elegidas.
    """
    _comprobar_forma(seleccion)
    comprobar_huella(huella_combinado)
    origenes = (metadatos_combinado.get("origenes")
                if isinstance(metadatos_combinado, Mapping) else None)
    archivos = origenes.get("archivos") if isinstance(origenes, Mapping) else None
    if not isinstance(archivos, list):
        raise ValueError("Los metadatos no son de un archivo combinado")
    nuevos: dict[int, int] | None = None
    for archivo in archivos:
        if isinstance(archivo, Mapping) and archivo.get("huella_dataset") == seleccion.huella_dataset:
            ids = archivo.get("ids")
            if (not isinstance(ids, list) or not all(
                    isinstance(par, list) and len(par) == 2 and all(es_id_episodio(i) for i in par)
                    for par in ids)):
                raise ValueError("Los IDs del archivo en los origenes están mal formados")
            nuevos = {int(antiguo): int(nuevo) for antiguo, nuevo in ids}
            break
    if nuevos is None:
        raise ValueError("El archivo combinado no incluye el archivo de esta selección")
    faltan = sorted(set(seleccion.partidas) - set(nuevos))
    if faltan:
        raise ValueError(f"Los origenes del combinado no incluyen las partidas {faltan}")
    trasladadas = [SeleccionPares(huella_dataset=huella_combinado,
                                  partidas={nuevos[int(i)]: tramos})
                   for i, tramos in seleccion.partidas.items()]
    return unir_selecciones(trasladadas)


def _cantidad_validacion(partidas: int, fraccion: float) -> int:
    """Partidas de validación: la fracción redondeada hacia arriba en los empates,
    con al menos una en validación y otra en train."""
    return min(partidas - 1, max(1, math.floor(fraccion * partidas + 0.5)))


def repartir_validacion(
    seleccion: SeleccionPares, cfg: Mapping[str, Any], generador: np.random.Generator,
) -> tuple[SeleccionPares, SeleccionPares]:
    """Separa partidas enteras de validación de una selección, con el generador.

    Ordena los IDs, los baraja con el generador y lleva las primeras a
    validación, como crear_splits: el resultado depende solo de los IDs y del
    generador. Validación tiene demostraciones.fraccion_validacion de las
    partidas, redondeada (las mitades hacia arriba), con al menos una partida en
    cada lado. Cada partida conserva sus tramos.

    Args:
        seleccion: Selección con al menos dos partidas.
        cfg: Configuración resuelta con demostraciones.fraccion_validacion.
        generador: Generador de src.utils.seed.crear_generador.

    Returns:
        (train, validacion), dos selecciones del mismo archivo sin partidas comunes.

    Raises:
        ValueError: Menos de dos partidas, fracción fuera de (0, 1), configuración
            incompleta o generador no válido.
    """
    _comprobar_forma(seleccion)
    dcfg = cfg.get("demostraciones") if isinstance(cfg, Mapping) else None
    if not isinstance(dcfg, Mapping) or "fraccion_validacion" not in dcfg:
        raise ValueError("La configuración necesita demostraciones.fraccion_validacion")
    fraccion = dcfg["fraccion_validacion"]
    if not es_real_finito(fraccion) or not 0 < fraccion < 1:
        raise ValueError(f"demostraciones.fraccion_validacion debe estar entre 0 y 1, "
                         f"no {fraccion!r}")
    if not isinstance(generador, np.random.Generator):
        raise ValueError("generador debe ser un numpy.random.Generator de crear_generador")
    if seleccion.numero_partidas < 2:
        raise ValueError("Hacen falta al menos dos partidas para separar una de validación")
    ordenados = sorted(int(i) for i in seleccion.partidas)
    barajados = [ordenados[k] for k in generador.permutation(len(ordenados))]
    cantidad = _cantidad_validacion(len(ordenados), float(fraccion))
    validacion, train = set(barajados[:cantidad]), set(barajados[cantidad:])

    def parte(ids: set[int]) -> SeleccionPares:
        return SeleccionPares(huella_dataset=seleccion.huella_dataset,
                              partidas={i: seleccion.partidas[i] for i in sorted(ids)})

    return parte(train), parte(validacion)


def pares_de_seleccion(trayectorias: Sequence[Trayectoria],
                       seleccion: SeleccionPares) -> LoteTransiciones:
    """Devuelve los pares (estado, acción) de una selección.

    Toma exactamente los pasos de los tramos, sin submuestrear, en orden de
    id_episodio y, dentro de cada partida, de paso. Los arrays conservan el
    dtype de las partidas.

    Args:
        trayectorias: Partidas del archivo de la selección.
        seleccion: Selección de esas partidas.

    Raises:
        ValueError: Si la selección no corresponde a las partidas.
    """
    validar_seleccion(seleccion, trayectorias)
    por_id = _por_id(trayectorias)
    estados, acciones, ids, pasos = [], [], [], []
    for identificador, tramos in sorted(seleccion.partidas.items()):
        trayectoria = por_id[int(identificador)]
        elegidos = np.concatenate([np.arange(inicio, fin, dtype=np.int64)
                                   for inicio, fin in tramos])
        estados.append(trayectoria.estados[elegidos])
        acciones.append(trayectoria.acciones[elegidos])
        ids.append(np.full(len(elegidos), int(identificador), dtype=np.int64))
        pasos.append(elegidos)
    return LoteTransiciones(estados=np.concatenate(estados), acciones=np.concatenate(acciones),
                            ids_episodio=np.concatenate(ids), pasos=np.concatenate(pasos))


def guardar_seleccion(ruta: str | Path, seleccion: SeleccionPares) -> Path:
    """Guarda la selección en JSON, sin sobrescribir un archivo existente.

    Guarda formato, versión, huella del archivo de demostraciones y, por
    partida, sus tramos. Las mismas selecciones producen siempre el mismo texto.

    Raises:
        ValueError: Extensión distinta de .json o selección mal formada.
        FileExistsError: El destino ya existe; se conserva sin cambios.
        OSError: Fallo de escritura; se elimina el archivo incompleto.
    """
    destino = Path(ruta)
    if destino.suffix != ".json":
        raise ValueError("La ruta de la selección debe terminar en .json")
    _comprobar_forma(seleccion)
    contenido = {
        "formato": _FORMATO, "version": _VERSION, "huella_dataset": seleccion.huella_dataset,
        "partidas": [{"id_episodio": int(i), "tramos": [list(t) for t in tramos]}
                     for i, tramos in sorted(seleccion.partidas.items())],
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
    """Rechaza NaN e Infinity, que no son pasos ni IDs válidos."""
    raise ValueError(f"El archivo de la selección contiene un valor no válido: {texto}")


def cargar_seleccion(ruta: str | Path, huella_dataset: str,
                     trayectorias: Sequence[Trayectoria]) -> SeleccionPares:
    """Carga una selección y comprueba que corresponde al archivo de demostraciones.

    Args:
        ruta: Archivo JSON de guardar_seleccion.
        huella_dataset: huella_archivo del NPZ que se va a usar.
        trayectorias: Partidas cargadas de ese NPZ.

    Raises:
        FileNotFoundError: La ruta no existe.
        ValueError: Extensión, JSON, formato o versión no válidos, huella
            distinta o selección incoherente con las partidas.
    """
    origen = Path(ruta)
    if origen.suffix != ".json":
        raise ValueError("La ruta de la selección debe terminar en .json")
    comprobar_huella(huella_dataset)
    try:
        contenido = json.loads(origen.read_text(encoding="utf-8"),
                               parse_constant=_rechazar_constante)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"El archivo de la selección no es JSON válido: {origen}") from exc
    if not isinstance(contenido, dict) or set(contenido) != {"formato", "version",
                                                             "huella_dataset", "partidas"}:
        raise ValueError("El archivo de la selección no tiene las claves esperadas")
    if contenido["formato"] != _FORMATO:
        raise ValueError(f"Formato desconocido: se esperaba {_FORMATO!r}")
    if contenido["version"] != _VERSION or isinstance(contenido["version"], bool):
        raise ValueError(f"Versión no soportada: {contenido['version']!r}; se admite {_VERSION}")
    if contenido["huella_dataset"] != huella_dataset:
        raise ValueError("Esta selección se creó para otro archivo de demostraciones: "
                         "la huella SHA-256 no coincide")
    entradas = contenido["partidas"]
    if not isinstance(entradas, list) or not all(
            isinstance(e, dict) and set(e) == {"id_episodio", "tramos"}
            and es_id_episodio(e["id_episodio"]) and isinstance(e["tramos"], list)
            for e in entradas):
        raise ValueError("Las partidas de la selección están mal formadas")
    partidas: dict[int, tuple[Tramo, ...]] = {}
    for entrada in entradas:
        if entrada["id_episodio"] in partidas:
            raise ValueError(f"La selección repite la partida {entrada['id_episodio']}")
        if not all(isinstance(t, list) and len(t) == 2 and all(es_entero(x) for x in t)
                   for t in entrada["tramos"]):
            raise ValueError(f"Tramos mal formados en la partida {entrada['id_episodio']}")
        partidas[entrada["id_episodio"]] = tuple((t[0], t[1]) for t in entrada["tramos"])
    seleccion = SeleccionPares(huella_dataset=huella_dataset,
                               partidas=dict(sorted(partidas.items())))
    validar_seleccion(seleccion, trayectorias)
    return seleccion
