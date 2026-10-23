"""Grabación, validación, guardado y carga de episodios completos.

Las partidas se guardan en NPZ con un manifiesto JSON versionado. generar_dataset
graba las partidas con una política, las guarda y crea sus splits; es lo que usa
scripts/get_data.py con el experto TRPO.
"""

from __future__ import annotations

import hashlib
import json
import tokenize
import zipfile
import zlib
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from src.data.entornos import crear_entorno
from src.data.splits import comprobar_cantidades, crear_splits, guardar_splits, huella_archivo
from src.data.tipos import ID_EPISODIO_MAXIMO, ID_EPISODIO_MINIMO, Trayectoria, es_id_episodio
from src.utils.seed import (crear_generador, generador_torch_aislado, reiniciar_con_semilla,
                            semillas_independientes, validar_semilla)
from src.utils.validacion import es_entero

Politica = Callable[[NDArray[Any]], Any]
"""Recibe una observación en bruto del entorno y devuelve la acción que se le pasa a step."""

ARCHIVO_DEMOSTRACIONES = "demostraciones.npz"
ARCHIVO_SPLITS = "splits.json"

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


def comprobar_espacios(espacio_observacion: spaces.Space, espacio_accion: spaces.Space) -> None:
    """Rechaza espacios que el formato todavía no admite: Box de una dimensión para las
    observaciones y Discrete o Box de una dimensión para las acciones.

    Raises:
        ValueError: Si alguno de los espacios no es de esos tipos.
    """
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
    if not es_entero(identificador):
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
            raise ValueError(
                f"{contexto}: {campo} fuera de los límites del espacio de observación")
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
    comprobar_espacios(espacio_observacion, espacio_accion)
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
                or not es_entero(episodio["numero_pasos"])):
            raise ValueError(f"Entrada {indice} del manifiesto mal formada")
    prefijos = [episodio["prefijo"] for episodio in episodios]
    if len(set(prefijos)) != len(prefijos):
        raise ValueError("El manifiesto repite prefijos de episodio")
    esperados = {"manifiesto"} | {f"{prefijo}_{campo}"
                                  for prefijo in prefijos for campo in _CAMPOS}
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


def jugar_partida(entorno: gym.Env, politica: Politica, semilla: int,
                  id_episodio: int) -> Trayectoria:
    """Juega una partida completa con la política y la devuelve como Trayectoria.

    La partida empieza con reiniciar_con_semilla, así que es la primera de esa
    semilla, y todo ocurre dentro de generador_torch_aislado(semilla): una
    política que muestrea con el generador global de PyTorch (como predict de
    sb3) da la misma partida con la misma semilla y no altera lo que venga
    después. Las observaciones se guardan en bruto, tal como las da el entorno;
    si la política normaliza, lo hace por dentro. La política recibe una copia
    de cada observación, así que modificarla no cambia la partida. Las recompensas son las del
    entorno, en float64.

    Args:
        entorno: Entorno Gymnasium con límite de pasos (TimeLimit), como los de
            crear_entorno.
        politica: Función de observación a acción.
        semilla: Semilla no negativa de la partida.
        id_episodio: Identificador de la partida (es_id_episodio).

    Returns:
        La partida, sin validar frente a los espacios (grabar_demostraciones lo hace).

    Raises:
        TypeError: Si entorno no es un entorno Gymnasium.
        ValueError: Si el entorno no tiene límite de pasos, o la semilla o el
            identificador no son válidos.
    """
    if not isinstance(entorno, gym.Env):
        raise TypeError("entorno debe ser un entorno Gymnasium")
    if entorno.spec is None or entorno.spec.max_episode_steps is None:
        raise ValueError("El entorno necesita un límite de pasos (TimeLimit) para que la "
                         "partida termine")
    if not es_id_episodio(id_episodio):
        raise ValueError(f"id_episodio debe ser un entero del rango de int64, no {id_episodio!r}")
    semilla = validar_semilla(semilla)
    estados, acciones, siguientes, recompensas, terminados, truncados = [], [], [], [], [], []
    with generador_torch_aislado(semilla):
        observacion, _ = reiniciar_con_semilla(entorno, semilla)
        fin = False
        while not fin:
            # La política recibe su propia copia: si modificara la observación, no
            # alteraría ni el estado guardado ni un buffer que el entorno reutilice.
            estados.append(np.array(observacion))
            accion = politica(np.array(observacion))
            siguiente, recompensa, terminado, truncado, _ = entorno.step(accion)
            acciones.append(np.array(accion))
            siguientes.append(np.array(siguiente))
            recompensas.append(float(recompensa))
            terminados.append(bool(terminado))
            truncados.append(bool(truncado))
            observacion, fin = siguiente, terminado or truncado
    return Trayectoria(
        id_episodio=int(id_episodio), estados=np.stack(estados), acciones=np.stack(acciones),
        siguientes_estados=np.stack(siguientes),
        recompensas_originales=np.asarray(recompensas, dtype=np.float64),
        terminados=np.asarray(terminados, dtype=bool), truncados=np.asarray(truncados, dtype=bool))


def retorno_partida(trayectoria: Trayectoria) -> float:
    """Suma las recompensas originales en orden, paso a paso.

    Da el mismo valor que acumular la recompensa durante la partida. No se usa
    sum, que desde Python 3.12 compensa el redondeo y puede cambiar el último
    decimal respecto a la suma paso a paso.
    """
    retorno = 0.0
    for recompensa in trayectoria.recompensas_originales.tolist():
        retorno += recompensa
    return retorno

def huella_partidas(trayectorias: Sequence[Trayectoria]) -> str:
    """Calcula una huella SHA-256 del contenido de las partidas, sin metadatos.

    Recorre las partidas en orden y, de cada una, su id_episodio y el dtype,
    la forma y los bytes de cada array. A diferencia de huella_archivo del NPZ,
    no cambia con los metadatos (commit, versiones), así que sirve para
    comprobar que una regeneración da exactamente las mismas partidas.

    Raises:
        ValueError: Si no es una secuencia de objetos Trayectoria.
    """
    if (isinstance(trayectorias, (str, bytes)) or not isinstance(trayectorias, Sequence)
            or not all(isinstance(t, Trayectoria) for t in trayectorias)):
        raise ValueError("Se requiere una secuencia de objetos Trayectoria")
    resumen = hashlib.sha256()
    for trayectoria in trayectorias:
        resumen.update(f"id={int(trayectoria.id_episodio)};".encode())
        for campo in _CAMPOS:
            valor = np.ascontiguousarray(getattr(trayectoria, campo))
            resumen.update(f"{campo}:{valor.dtype.str}:{valor.shape};".encode())
            resumen.update(valor.tobytes())
    return resumen.hexdigest()

def grabar_demostraciones(entorno: gym.Env, politica: Politica, partidas: int,
                          semilla: int) -> list[Trayectoria]:
    """Juega partidas con la política y las valida frente a los espacios del entorno.

    La partida i usa la semilla i de semillas_independientes(semilla, partidas)
    y tiene id_episodio i. Así cada partida depende solo de su semilla: grabar
    menos partidas con la misma semilla da las primeras de grabar más. Se
    guardan todas, sin filtrar por retorno, como el código original.

    Args:
        entorno: Entorno Gymnasium con límite de pasos; quien llama lo cierra.
        politica: Función de observación a acción.
        partidas: Número de partidas, entero positivo.
        semilla: Semilla base no negativa.

    Returns:
        Las partidas en orden de id_episodio.

    Raises:
        ValueError: Partidas o semilla no válidas, o partidas incompatibles con
            los espacios del entorno (por ejemplo, acciones fuera de sus límites).
    """
    semillas = semillas_independientes(validar_semilla(semilla), partidas)
    trayectorias = [jugar_partida(entorno, politica, semilla_partida, indice)
                    for indice, semilla_partida in enumerate(semillas)]
    validar_demostraciones(trayectorias, entorno.observation_space, entorno.action_space)
    return trayectorias


def config_demostraciones(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Devuelve la sección demostraciones comprobada.

    partidas_train, partidas_validacion y subconjuntos deben poder repartirse
    con crear_splits (comprobar_cantidades); semilla pasa por validar_semilla
    y determinista es booleano.

    Raises:
        ValueError: Si falta la sección o alguna clave, o algún valor no es válido.
    """
    dcfg = cfg.get("demostraciones") if isinstance(cfg, Mapping) else None
    if not isinstance(dcfg, Mapping):
        raise ValueError("La configuración necesita la sección demostraciones")
    claves = ("partidas_train", "partidas_validacion", "subconjuntos", "semilla", "determinista")
    for clave in claves:
        if clave not in dcfg:
            raise ValueError(f"Falta demostraciones.{clave} en la configuración")
    comprobar_cantidades(dcfg["partidas_train"], dcfg["partidas_validacion"],
                         dcfg["subconjuntos"])
    try:
        validar_semilla(dcfg["semilla"])
    except ValueError as error:
        raise ValueError(f"demostraciones.semilla no es válida: {error}") from error
    if not isinstance(dcfg["determinista"], bool):
        raise ValueError(f"demostraciones.determinista debe ser booleano, "
                         f"no {dcfg['determinista']!r}")
    return dict(dcfg)


def generar_dataset(cfg: Mapping[str, Any], politica: Politica, directorio: str | Path,
                    metadatos: Mapping[str, object]) -> dict[str, Any]:
    """Graba las demostraciones de la configuración, las guarda y crea sus splits.

    De demostraciones.semilla salen dos semillas independientes: la de las
    partidas y la del reparto. Se graban partidas_train + partidas_validacion
    partidas en el entorno de la configuración, se guardan en
    directorio/demostraciones.npz con los metadatos recibidos más un apartado
    grabacion (semillas, huella_partidas, retornos y longitudes), y se
    reparten con crear_splits
    en directorio/splits.json, con la huella del NPZ. No sobrescribe: si falla
    después de guardar el NPZ, lo borra para no dejar un dataset sin splits.

    Args:
        cfg: Configuración resuelta con entorno.id y la sección demostraciones.
        politica: Función de observación a acción, por ejemplo el experto.
        directorio: Carpeta de salida; se crea si no existe.
        metadatos: Metadatos obligatorios de guardar_demostraciones (entorno,
            experto, procedencia, configuracion_experto), sin la clave grabacion.

    Returns:
        Rutas de los dos archivos, huella_dataset y huella_splits (SHA-256 de
        cada archivo), huella_partidas (solo del contenido),
        IDs de train y validación, pasos totales, retornos y longitudes.

    Raises:
        ValueError: Configuración o metadatos no válidos, o partidas
            incompatibles con el entorno.
        FileExistsError: Alguno de los archivos ya existe.
    """
    dcfg = config_demostraciones(cfg)
    if not isinstance(metadatos, Mapping) or "grabacion" in metadatos:
        raise ValueError("metadatos debe ser un diccionario sin la clave grabacion, "
                         "que añade generar_dataset")
    semilla = validar_semilla(dcfg["semilla"])
    semilla_partidas, semilla_splits = semillas_independientes(semilla, 2)
    total = dcfg["partidas_train"] + dcfg["partidas_validacion"]
    entorno = crear_entorno(dict(cfg), None)
    try:
        trayectorias = grabar_demostraciones(entorno, politica, total, semilla_partidas)
        espacio_observacion, espacio_accion = entorno.observation_space, entorno.action_space
    finally:
        entorno.close()
    retornos = [retorno_partida(t) for t in trayectorias]
    longitudes = [t.numero_pasos for t in trayectorias]
    contenido = huella_partidas(trayectorias)
    grabacion = {"semilla": semilla, "semilla_partidas": semilla_partidas,
                 "semilla_splits": semilla_splits, "partidas": total,
                 "determinista": dcfg["determinista"], "huella_partidas": contenido,
                 "retornos": retornos, "longitudes": longitudes}
    directorio = Path(directorio)
    ruta_demostraciones = guardar_demostraciones(
        directorio / ARCHIVO_DEMOSTRACIONES, trayectorias, {**metadatos, "grabacion": grabacion},
        espacio_observacion, espacio_accion)
    try:
        huella = huella_archivo(ruta_demostraciones)
        splits = crear_splits(trayectorias, dcfg["partidas_train"], dcfg["partidas_validacion"],
                              dcfg["subconjuntos"], crear_generador(semilla_splits), huella)
        ruta_splits = guardar_splits(directorio / ARCHIVO_SPLITS, splits)
    except BaseException:
        ruta_demostraciones.unlink(missing_ok=True)
        raise
    return {"demostraciones": ruta_demostraciones, "splits": ruta_splits,
            "huella_dataset": huella, "huella_splits": huella_archivo(ruta_splits),
            "huella_partidas": contenido,
            "train": list(splits.train), "validacion": list(splits.validacion),
            "pasos": sum(longitudes), "retornos": retornos, "longitudes": longitudes}
