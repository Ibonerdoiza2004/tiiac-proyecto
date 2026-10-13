"""Validación, guardado y carga de episodios completos en NPZ con un manifiesto JSON versionado."""

from __future__ import annotations

import json
import tokenize
import zipfile
import zlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from src.data.tipos import ID_EPISODIO_MAXIMO, ID_EPISODIO_MINIMO, Trayectoria, es_id_episodio

_CAMPOS = (
    "estados", "acciones", "siguientes_estados", "recompensas_originales",
    "terminados", "truncados",
)
_METADATOS_REQUERIDOS = {"entorno", "experto", "procedencia", "configuracion_experto"}
_FORMATO = "tiiac-demostraciones"
_VERSION = 1


def _comprobar_claves_json(valor: object, visitados: set[int]) -> None:
    """Rechaza claves que JSON convertiría silenciosamente en cadenas."""
    if isinstance(valor, (Mapping, list, tuple)):
        if id(valor) in visitados:
            return
        visitados.add(id(valor))
        if isinstance(valor, Mapping):
            if any(not isinstance(clave, str) for clave in valor):
                raise ValueError("Las claves de los metadatos deben ser cadenas")
            elementos = valor.values()
        else:
            elementos = valor
        for elemento in elementos:
            _comprobar_claves_json(elemento, visitados)


def _escalar_numpy_a_python(valor: object) -> object:
    """Convierte escalares NumPy, como semillas np.int64, en tipos JSON nativos."""
    if isinstance(valor, np.generic):
        return valor.item()
    raise TypeError(f"Tipo no serializable en JSON: {type(valor).__name__}")


def _comprobar_espacios(espacio_observacion: spaces.Space, espacio_accion: spaces.Space) -> None:
    """Rechaza espacios que el formato todavía no admite."""
    if not isinstance(espacio_observacion, spaces.Box) or len(espacio_observacion.shape) != 1:
        raise ValueError("Espacio de observación no compatible: se admite Box de una dimensión")
    discreto = isinstance(espacio_accion, spaces.Discrete)
    continuo = isinstance(espacio_accion, spaces.Box) and len(espacio_accion.shape) == 1
    if not (discreto or continuo):
        raise ValueError(
            "Espacio de acción no compatible: se admite Discrete o Box de una dimensión")


def _dentro_de_caja(valores: NDArray[Any], caja: spaces.Box) -> bool:
    """Indica si todas las filas [T, d] respetan los límites de la caja."""
    return bool(np.all(valores >= caja.low) and np.all(valores <= caja.high))


def _validar_trayectoria(
    trayectoria: Trayectoria, indice: int,
    espacio_observacion: spaces.Box, espacio_accion: spaces.Discrete | spaces.Box,
) -> tuple[int, dict[str, NDArray[Any]]]:
    """Comprueba una partida frente a los espacios sin transformar sus arrays."""
    contexto = f"Trayectoria {indice}"
    if not isinstance(trayectoria, Trayectoria):
        raise ValueError(f"{contexto}: se esperaba un objeto Trayectoria")
    identificador = trayectoria.id_episodio
    if (isinstance(identificador, (bool, np.bool_))
            or not isinstance(identificador, (int, np.integer))):
        raise ValueError(f"{contexto}: id_episodio debe ser un entero")
    if not es_id_episodio(identificador):
        raise ValueError(f"{contexto}: id_episodio debe estar en el rango de int64 "
                         f"[{ID_EPISODIO_MINIMO}, {ID_EPISODIO_MAXIMO}]")
    identificador = int(identificador)

    arrays: dict[str, NDArray[Any]] = {}
    for campo in _CAMPOS:
        valor = getattr(trayectoria, campo)
        if not isinstance(valor, np.ndarray):
            raise ValueError(f"{contexto}: {campo} debe ser un array NumPy")
        arrays[campo] = valor
    estados = arrays["estados"]
    if estados.ndim != 2 or estados.shape[0] == 0 or estados.shape[1] == 0:
        raise ValueError(f"{contexto}: estados debe tener forma [T, d], con T y d positivos")
    pasos = len(estados)
    for campo, valor in arrays.items():
        if valor.ndim == 0 or len(valor) != pasos:
            raise ValueError(f"{contexto}: {campo} tiene una longitud incompatible")
        if campo in ("terminados", "truncados"):
            if valor.shape != (pasos,) or valor.dtype != np.bool_:
                raise ValueError(f"{contexto}: {campo} debe ser booleano y tener forma [T]")
        elif valor.dtype.kind not in "iuf" or not np.isfinite(valor).all():
            raise ValueError(f"{contexto}: {campo} debe contener números reales finitos")
    if arrays["siguientes_estados"].shape != estados.shape:
        raise ValueError(f"{contexto}: siguientes_estados debe tener la misma forma que estados")
    if arrays["recompensas_originales"].shape != (pasos,):
        raise ValueError(f"{contexto}: recompensas_originales debe tener forma [T]")
    finales = arrays["terminados"] | arrays["truncados"]
    if finales[:-1].any() or not finales[-1]:
        raise ValueError(f"{contexto}: solo el último paso debe finalizar el episodio")
    if not np.array_equal(estados[1:], arrays["siguientes_estados"][:-1]):
        raise ValueError(f"{contexto}: los estados no forman una secuencia continua")

    if estados.shape[1:] != espacio_observacion.shape:
        raise ValueError(
            f"{contexto}: estados debe tener forma [T, {espacio_observacion.shape[0]}]")
    for campo in ("estados", "siguientes_estados"):
        if not _dentro_de_caja(arrays[campo], espacio_observacion):
            raise ValueError(f"{contexto}: {campo} fuera de los límites del espacio de observación")
    acciones = arrays["acciones"]
    if isinstance(espacio_accion, spaces.Discrete):
        if acciones.ndim != 1 or acciones.dtype.kind not in "iu":
            raise ValueError(f"{contexto}: las acciones discretas deben ser enteros con forma [T]")
        inicio = int(espacio_accion.start)
        if acciones.min() < inicio or acciones.max() >= inicio + int(espacio_accion.n):
            raise ValueError(f"{contexto}: acciones fuera del espacio {espacio_accion}")
    else:
        if acciones.shape[1:] != espacio_accion.shape:
            raise ValueError(
                f"{contexto}: acciones debe tener forma [T, {espacio_accion.shape[0]}]")
        if not _dentro_de_caja(acciones, espacio_accion):
            raise ValueError(f"{contexto}: acciones fuera de los límites del espacio de acción")
    return identificador, arrays


def _validar_coleccion(
    trayectorias: Sequence[Trayectoria],
    espacio_observacion: spaces.Space, espacio_accion: spaces.Space,
) -> list[tuple[int, dict[str, NDArray[Any]]]]:
    """Valida la colección completa y devuelve IDs y arrays de cada partida."""
    if not isinstance(trayectorias, Sequence) or isinstance(trayectorias, str) or not trayectorias:
        raise ValueError("Se requiere una secuencia no vacía de trayectorias")
    _comprobar_espacios(espacio_observacion, espacio_accion)
    validadas: list[tuple[int, dict[str, NDArray[Any]]]] = []
    identificadores: set[int] = set()
    for indice, trayectoria in enumerate(trayectorias):
        identificador, arrays = _validar_trayectoria(
            trayectoria, indice, espacio_observacion, espacio_accion)
        if identificador in identificadores:
            raise ValueError(f"id_episodio duplicado: {identificador}")
        identificadores.add(identificador)
        validadas.append((identificador, arrays))
    return validadas


def validar_demostraciones(
    trayectorias: Sequence[Trayectoria],
    espacio_observacion: spaces.Space,
    espacio_accion: spaces.Space,
) -> None:
    """Comprueba que las partidas son completas y compatibles con el entorno.

    Cada partida es un objeto Trayectoria con id_episodio entero y único. Sus
    arrays tienen T pasos, valores finitos y continuidad entre estados; solo el
    último paso termina o se trunca. Las observaciones tienen la forma y los
    límites de espacio_observacion. Con Discrete(n, start) las acciones son
    enteros [T] en [start, start + n); con Box son reales [T, d] dentro de sus
    límites. No modifica ni copia los arrays.

    Args:
        trayectorias: Secuencia no vacía de objetos Trayectoria.
        espacio_observacion: Espacio Box de una dimensión del entorno.
        espacio_accion: Espacio Discrete o Box de una dimensión del entorno.

    Raises:
        ValueError: Partidas incompletas, incoherentes o incompatibles con los
            espacios, o espacios que el formato todavía no admite.
    """
    _validar_coleccion(trayectorias, espacio_observacion, espacio_accion)


def guardar_demostraciones(
    ruta: str | Path,
    trayectorias: Sequence[Trayectoria],
    metadatos: Mapping[str, object],
    espacio_observacion: spaces.Space,
    espacio_accion: spaces.Space,
) -> Path:
    """Valida y guarda partidas completas con su procedencia sin sobrescribir archivos.

    Antes de escribir aplica validar_demostraciones con los espacios del
    entorno. Los arrays conservan sus valores, dtype, orden y separación por
    episodio; no se normalizan.

    Los metadatos JSON incluyen entorno, experto, procedencia y
    configuracion_experto. Los escalares NumPy se guardan como números de
    Python y las tuplas se leerán como listas. La ruta termina en .npz y se
    crean sus directorios. El manifiesto de formato y versión vive dentro del
    mismo archivo comprimido, legible con np.load(..., allow_pickle=False).

    Args:
        ruta: Destino del archivo NPZ, elegido por quien llama.
        trayectorias: Secuencia no vacía de objetos Trayectoria.
        metadatos: Información JSON del entorno, experto y origen de los datos.
        espacio_observacion: Espacio de observación del entorno de origen.
        espacio_accion: Espacio de acción del entorno de origen.

    Returns:
        Ruta del archivo creado.

    Raises:
        ValueError: Ruta, datos, espacios o metadatos incompatibles con el formato.
        FileExistsError: El destino ya existe; se conserva sin cambios.
        OSError: No se puede crear o escribir el archivo. Si falla la escritura,
            se elimina el archivo incompleto creado por esta llamada.
    """
    destino = Path(ruta)
    if destino.suffix != ".npz":
        raise ValueError("La ruta de demostraciones debe terminar en .npz")
    if not isinstance(metadatos, Mapping) or not _METADATOS_REQUERIDOS <= metadatos.keys():
        raise ValueError(f"Metadatos obligatorios: {sorted(_METADATOS_REQUERIDOS)}")
    _comprobar_claves_json(metadatos, set())
    validadas = _validar_coleccion(trayectorias, espacio_observacion, espacio_accion)

    contenidos: dict[str, NDArray[Any]] = {}
    episodios: list[dict[str, object]] = []
    for indice, (identificador, arrays) in enumerate(validadas):
        prefijo = f"trayectoria_{indice}"
        episodios.append({"id_episodio": identificador, "prefijo": prefijo,
                          "numero_pasos": len(arrays["estados"])})
        contenidos.update({f"{prefijo}_{campo}": valor for campo, valor in arrays.items()})

    manifiesto = {"formato": _FORMATO, "version": _VERSION,
                  "episodios": episodios, "metadatos": dict(metadatos)}
    try:
        texto = json.dumps(manifiesto, ensure_ascii=False, allow_nan=False, sort_keys=True,
                           default=_escalar_numpy_a_python)
    except (TypeError, ValueError) as exc:
        raise ValueError("Los metadatos deben ser JSON con valores finitos "
                         "y sin referencias circulares") from exc
    contenidos["manifiesto"] = np.asarray(texto)
    destino.parent.mkdir(parents=True, exist_ok=True)
    archivo = destino.open("xb")
    try:
        with archivo:
            np.savez_compressed(archivo, allow_pickle=False, **contenidos)
    except BaseException:
        destino.unlink(missing_ok=True)
        raise
    return destino


def _leer_array(archivo: np.lib.npyio.NpzFile, clave: str) -> NDArray[Any]:
    """Devuelve un miembro del NPZ y rechaza los que no contienen un array .npy válido."""
    try:
        valor = archivo[clave]
    except (ValueError, SyntaxError, tokenize.TokenError) as exc:
        # NumPy señala así las cabeceras .npy rotas, los datos truncados y los arrays
        # que exigirían pickle; algunos de estos errores no son ValueError.
        raise ValueError(f"El miembro {clave!r} no contiene un array NumPy válido") from exc
    if not isinstance(valor, np.ndarray):
        # NpzFile devuelve bytes cuando el miembro no tiene cabecera .npy.
        raise ValueError(f"El miembro {clave!r} no contiene un array NumPy")
    return valor


def _rechazar_no_finito(texto: str) -> float:
    """Convierte números JSON y rechaza NaN, Infinity y desbordamientos como 1e999."""
    valor = float(texto)
    if not np.isfinite(valor):
        raise ValueError(f"El manifiesto contiene un número no finito: {texto}")
    return valor


def _leer_manifiesto(archivo: np.lib.npyio.NpzFile) -> dict[str, Any]:
    """Lee el manifiesto JSON y comprueba formato, versión, episodios y metadatos."""
    if len(archivo.files) != len(set(archivo.files)):
        raise ValueError("El archivo contiene miembros con nombres repetidos")
    if "manifiesto" not in archivo.files:
        raise ValueError("El archivo no contiene manifiesto")
    contenido = _leer_array(archivo, "manifiesto")
    if contenido.shape != () or contenido.dtype.kind != "U":
        raise ValueError("El manifiesto debe ser un texto escalar")
    try:
        manifiesto = json.loads(contenido.item(), parse_constant=_rechazar_no_finito,
                                parse_float=_rechazar_no_finito)
    except json.JSONDecodeError as exc:
        raise ValueError("El manifiesto no es JSON válido") from exc
    if not isinstance(manifiesto, dict):
        raise ValueError("El manifiesto debe ser un objeto JSON")
    if manifiesto.get("formato") != _FORMATO:
        raise ValueError(f"Formato desconocido: se esperaba {_FORMATO!r}")
    if manifiesto.get("version") != _VERSION or isinstance(manifiesto.get("version"), bool):
        raise ValueError(
            f"Versión no soportada: {manifiesto.get('version')!r}; se admite {_VERSION}")
    metadatos = manifiesto.get("metadatos")
    if not isinstance(metadatos, dict) or not _METADATOS_REQUERIDOS <= metadatos.keys():
        raise ValueError(f"Metadatos obligatorios: {sorted(_METADATOS_REQUERIDOS)}")
    episodios = manifiesto.get("episodios")
    if not isinstance(episodios, list) or not episodios:
        raise ValueError("El manifiesto debe listar al menos un episodio")
    for indice, episodio in enumerate(episodios):
        if (not isinstance(episodio, dict)
                or set(episodio) != {"id_episodio", "prefijo", "numero_pasos"}
                or not isinstance(episodio["prefijo"], str)
                or isinstance(episodio["numero_pasos"], bool)
                or not isinstance(episodio["numero_pasos"], int)):
            raise ValueError(f"Entrada {indice} del manifiesto mal formada")
    prefijos = [episodio["prefijo"] for episodio in episodios]
    if len(set(prefijos)) != len(prefijos):
        raise ValueError("El manifiesto repite prefijos de episodio")
    esperados = {"manifiesto"} | {f"{prefijo}_{campo}" for prefijo in prefijos for campo in _CAMPOS}
    if set(archivo.files) != esperados:
        faltan = sorted(esperados - set(archivo.files))
        sobran = sorted(set(archivo.files) - esperados)
        raise ValueError(f"Arrays del archivo incoherentes con el manifiesto; "
                         f"faltan {faltan}, sobran {sobran}")
    return manifiesto


def cargar_demostraciones(
    ruta: str | Path,
    espacio_observacion: spaces.Space,
    espacio_accion: spaces.Space,
) -> tuple[list[Trayectoria], dict[str, Any]]:
    """Carga y valida un archivo creado por guardar_demostraciones.

    Lee el NPZ sin pickle, comprueba formato y versión del manifiesto, que
    sus episodios coincidan exactamente con los arrays guardados y que cada
    numero_pasos sea la longitud real. Reconstruye las partidas en el orden
    guardado y las valida con validar_demostraciones frente a los espacios del
    entorno en el que se van a usar. Los arrays conservan dtype y valores.

    Args:
        ruta: Archivo NPZ de demostraciones.
        espacio_observacion: Espacio de observación del entorno de uso.
        espacio_accion: Espacio de acción del entorno de uso.

    Returns:
        Las trayectorias en el orden guardado y los metadatos del manifiesto.
        Las tuplas guardadas en los metadatos se recuperan como listas.

    Raises:
        FileNotFoundError: La ruta no existe.
        IsADirectoryError: La ruta es un directorio.
        ValueError: Extensión distinta de .npz, archivo dañado o con otro
            formato, miembros que no son arrays o con nombres repetidos,
            versión no soportada, manifiesto incoherente, números no finitos
            en el manifiesto o partidas incompatibles con los espacios.
    """
    origen = Path(ruta)
    if origen.suffix != ".npz":
        raise ValueError("La ruta de demostraciones debe terminar en .npz")
    if origen.is_dir():
        raise IsADirectoryError(f"La ruta de demostraciones es un directorio: {origen}")
    if not origen.is_file():
        raise FileNotFoundError(f"No existe el archivo de demostraciones: {origen}")
    try:
        # Se abre aquí porque np.load no cierra el archivo si falla al leer un ZIP dañado.
        with origen.open("rb") as fichero:
            try:
                archivo = np.load(fichero, allow_pickle=False)
            except ValueError as exc:
                # Un archivo que no es ZIP ni .npy hace que NumPy intente leerlo con pickle.
                raise ValueError(f"Archivo de demostraciones dañado o ilegible: {origen}") from exc
            if not isinstance(archivo, np.lib.npyio.NpzFile):
                raise ValueError("El archivo no es un NPZ de demostraciones")
            with archivo:
                manifiesto = _leer_manifiesto(archivo)
                trayectorias: list[Trayectoria] = []
                for indice, episodio in enumerate(manifiesto["episodios"]):
                    arrays = {campo: _leer_array(archivo, f"{episodio['prefijo']}_{campo}")
                              for campo in _CAMPOS}
                    if any(valor.ndim == 0 or len(valor) != episodio["numero_pasos"]
                           for valor in arrays.values()):
                        raise ValueError(
                            f"Episodio {indice}: numero_pasos no coincide con sus arrays")
                    trayectorias.append(Trayectoria(id_episodio=episodio["id_episodio"], **arrays))
    except (zipfile.BadZipFile, zlib.error, EOFError) as exc:
        raise ValueError(f"Archivo de demostraciones dañado o ilegible: {origen}") from exc
    validar_demostraciones(trayectorias, espacio_observacion, espacio_accion)
    return trayectorias, manifiesto["metadatos"]
