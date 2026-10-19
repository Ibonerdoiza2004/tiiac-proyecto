"""Pruebas de grabación, validación, guardado, carga, integridad de episodios y
rechazo de datos inválidos."""

import gc
import io
import json
import re
import struct
import warnings
import zipfile
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

import torch
from gymnasium.envs.classic_control import CartPoleEnv

from src.data.demostraciones import (
    ARCHIVO_DEMOSTRACIONES, ARCHIVO_SPLITS, cargar_demostraciones, config_demostraciones,
    generar_dataset, grabar_demostraciones, guardar_demostraciones, huella_partidas,
    jugar_partida, retorno_partida, validar_demostraciones,
)
from src.data.entornos import crear_entorno
from src.data.splits import cargar_splits, crear_splits, huella_archivo
from src.data.tipos import ID_EPISODIO_MAXIMO, ID_EPISODIO_MINIMO, Trayectoria
from src.utils.configuracion import resolver_configuracion
from src.utils.seed import crear_generador, semillas_independientes

OBSERVACION = spaces.Box(-np.inf, np.inf, (4,), np.float32)
DISCRETA = spaces.Discrete(2)
CONTINUA = spaces.Box(-10.0, 10.0, (2,), np.float64)


@pytest.fixture
def metadatos() -> dict[str, object]:
    """Identifica explícitamente los episodios como datos artificiales de prueba."""
    return {"entorno": "CartPole-v1", "experto": "politica-de-prueba",
            "procedencia": "datos artificiales; no prueban calidad del experto",
            "configuracion_experto": {"capas": [8, 8]}, "licencia": "prueba"}


def _campos(pasos: int = 3, identificador: Any = 0, continua: bool = False) -> dict[str, Any]:
    """Construye los campos de una secuencia exacta sin utilizar aleatoriedad."""
    observaciones = np.arange((pasos + 1) * 4, dtype=np.float32).reshape(pasos + 1, 4)
    terminados = np.zeros(pasos, dtype=bool)
    if pasos:
        terminados[-1] = True
    acciones = (np.arange(pasos * 2, dtype=np.float64).reshape(pasos, 2)
                if continua else np.arange(pasos, dtype=np.int64) % 2)
    return {"id_episodio": identificador, "estados": observaciones[:-1],
            "acciones": acciones, "siguientes_estados": observaciones[1:],
            "recompensas_originales": np.ones(pasos, dtype=np.float32),
            "terminados": terminados, "truncados": np.zeros(pasos, dtype=bool)}


def _trayectoria(pasos: int = 3, identificador: Any = 0, continua: bool = False) -> Trayectoria:
    """Devuelve la secuencia exacta de _campos como objeto Trayectoria."""
    return Trayectoria(**_campos(pasos, identificador, continua))


def _guardar(ruta: Path | str, trayectorias: Any, metadatos: Any, continua: bool = False) -> Path:
    """Guarda con el espacio de observación de prueba y la acción indicada."""
    return guardar_demostraciones(ruta, trayectorias, metadatos, OBSERVACION,
                                  CONTINUA if continua else DISCRETA)


@pytest.mark.parametrize("continua", [False, True])
@pytest.mark.parametrize("final", [(True, False), (False, True), (True, True)])
def test_guardado_conserva_arrays_episodios_y_metadatos(
    tmp_path: Path, metadatos: dict[str, object], continua: bool,
    final: tuple[bool, bool],
) -> None:
    """Recuperar con NumPy conserva orden, valores, tipos y límites de partidas."""
    datos = [_campos(1, 3, continua), _campos(4, 7, continua)]
    for episodio in datos:
        episodio["terminados"][-1], episodio["truncados"][-1] = final
    originales = [{k: v.copy() if isinstance(v, np.ndarray) else v for k, v in d.items()}
                  for d in datos]
    ruta = tmp_path / "subdirectorio" / "demostraciones.npz"
    trayectorias = [Trayectoria(**d) for d in datos]
    assert _guardar(ruta, trayectorias, metadatos, continua) == ruta
    with np.load(ruta, allow_pickle=False) as archivo:
        manifiesto = json.loads(archivo["manifiesto"].item())
        assert manifiesto["formato"] == "tiiac-demostraciones"
        assert manifiesto["version"] == 1
        assert manifiesto["metadatos"] == metadatos
        assert [e["id_episodio"] for e in manifiesto["episodios"]] == [3, 7]
        assert [e["numero_pasos"] for e in manifiesto["episodios"]] == [1, 4]
        assert len(archivo.files) == 13
        for indice, episodio in enumerate(originales):
            for campo, valor in episodio.items():
                if campo == "id_episodio":
                    continue
                guardado = archivo[f"trayectoria_{indice}_{campo}"]
                np.testing.assert_array_equal(guardado, valor)
                assert guardado.dtype == valor.dtype
                np.testing.assert_array_equal(getattr(trayectorias[indice], campo), valor)
    with zipfile.ZipFile(ruta) as archivo:
        assert all(e.compress_type == zipfile.ZIP_DEFLATED for e in archivo.infolist())


def test_acepta_arrays_no_contiguos_y_id_numpy(
    tmp_path: Path, metadatos: dict[str, object],
) -> None:
    """Guardar no exige copias contiguas ni cambia los valores de entrada."""
    episodio = _campos(4, np.int64(5))
    for campo in ("estados", "siguientes_estados"):
        episodio[campo] = np.repeat(episodio[campo], 2, axis=1)[:, ::2]
        episodio[campo].flags.writeable = False
    trayectoria = Trayectoria(**episodio)
    ruta = _guardar(tmp_path / "datos.npz", (trayectoria,), metadatos)
    with np.load(ruta, allow_pickle=False) as archivo:
        np.testing.assert_array_equal(archivo["trayectoria_0_estados"], episodio["estados"])
        assert json.loads(archivo["manifiesto"].item())["episodios"][0]["id_episodio"] == 5


_NO_SECUENCIA = "Se requiere una secuencia no vacía de trayectorias"
_NO_TRAYECTORIA = "Trayectoria 0: se esperaba un objeto Trayectoria"


@pytest.mark.parametrize(("entrada", "mensaje"), [
    ([], _NO_SECUENCIA), ((), _NO_SECUENCIA), (None, _NO_SECUENCIA),
    ("episodios", _NO_SECUENCIA), ({"episodio": 0}, _NO_SECUENCIA),
    ([None], _NO_TRAYECTORIA), ([_campos()], _NO_TRAYECTORIA),
], ids=["lista_vacia", "tupla_vacia", "none", "texto", "diccionario", "elemento_none",
        "elemento_diccionario"])
def test_rechaza_coleccion_vacia_o_invalida(
    tmp_path: Path, metadatos: dict[str, object], entrada: Any, mensaje: str,
) -> None:
    """Solo se aceptan secuencias no vacías de Trayectoria; los diccionarios ya no."""
    ruta = tmp_path / "nuevo" / "datos.npz"
    with pytest.raises(ValueError, match=re.escape(mensaje)):
        _guardar(ruta, entrada, metadatos)
    with pytest.raises(ValueError, match=re.escape(mensaje)):
        validar_demostraciones(entrada, OBSERVACION, DISCRETA)
    assert not ruta.parent.exists()


_CASOS_ESTRUCTURA = [
    ("vacia", "estados debe tener forma [T, d], con T y d positivos"),
    ("longitud", "acciones tiene una longitud incompatible"),
    ("dimension_obs", "estados debe tener forma [T, d], con T y d positivos"),
    ("siguiente_forma", "siguientes_estados debe tener la misma forma que estados"),
    ("recompensa_matriz", "recompensas_originales debe tener forma [T]"),
    ("acciones_3d", "las acciones discretas deben ser enteros con forma [T]"),
    ("acciones_sin_dimension", "las acciones discretas deben ser enteros con forma [T]"),
    ("acciones_float", "las acciones discretas deben ser enteros con forma [T]"),
    ("lista", "acciones debe ser un array NumPy"),
    ("flags_enteros", "terminados debe ser booleano y tener forma [T]"),
    ("flags_matriz", "truncados debe ser booleano y tener forma [T]"),
    ("sin_final", "solo el último paso debe finalizar el episodio"),
    ("final_intermedio", "solo el último paso debe finalizar el episodio"),
    ("discontinuidad", "los estados no forman una secuencia continua"),
]


@pytest.mark.parametrize(("caso", "mensaje"), _CASOS_ESTRUCTURA,
                         ids=[caso for caso, _ in _CASOS_ESTRUCTURA])
def test_rechaza_estructura_o_limites_invalidos(
    tmp_path: Path, metadatos: dict[str, object], caso: str, mensaje: str,
) -> None:
    """Cubre formas, tipos, continuidad y límites de un episodio completo.

    Cada caso comprueba el mensaje para asegurar que falla por la comprobación
    prevista y no por otra; re.escape trata los corchetes como texto literal.
    """
    episodio = _campos()
    if caso == "vacia":
        episodio = _campos(0)
    elif caso == "longitud":
        episodio["acciones"] = episodio["acciones"][:-1]
    elif caso == "dimension_obs":
        episodio["estados"] = np.empty((3, 0))
    elif caso == "siguiente_forma":
        episodio["siguientes_estados"] = np.zeros((3, 2))
    elif caso == "recompensa_matriz":
        episodio["recompensas_originales"] = np.ones((3, 1))
    elif caso == "acciones_3d":
        episodio["acciones"] = np.ones((3, 1, 1), dtype=np.int64)
    elif caso == "acciones_sin_dimension":
        episodio["acciones"] = np.empty((3, 0), dtype=np.int64)
    elif caso == "acciones_float":
        episodio["acciones"] = episodio["acciones"].astype(float)
    elif caso == "lista":
        episodio["acciones"] = episodio["acciones"].tolist()
    elif caso == "flags_enteros":
        episodio["terminados"] = episodio["terminados"].astype(int)
    elif caso == "flags_matriz":
        episodio["truncados"] = np.zeros((3, 1), dtype=bool)
    elif caso == "sin_final":
        episodio["terminados"][:] = False
    elif caso == "final_intermedio":
        episodio["truncados"][0] = True
    elif caso == "discontinuidad":
        episodio["siguientes_estados"] = episodio["siguientes_estados"].copy()
        episodio["siguientes_estados"][0, 0] = -1
    ruta = tmp_path / "datos.npz"
    with pytest.raises(ValueError, match=re.escape(mensaje)):
        _guardar(ruta, [Trayectoria(**episodio)], metadatos)
    assert not ruta.exists()


@pytest.mark.parametrize("acciones", [np.array([[0], [1], [0]], dtype=np.int64),
                                      np.array([[0, 1], [1, 0], [0, 1]], dtype=np.int64)],
                         ids=["columna_T_1", "matriz_T_2"])
def test_rechaza_acciones_discretas_con_forma_de_matriz(
    tmp_path: Path, metadatos: dict[str, object], acciones: np.ndarray,
) -> None:
    """Regresión: acciones discretas [T, 1], típicas al apilar predict de un VecEnv,
    se guardaban como continuas."""
    episodio = _campos()
    episodio["acciones"] = acciones
    with pytest.raises(ValueError, match=r"enteros con forma \[T\]"):
        _guardar(tmp_path / "datos.npz", [Trayectoria(**episodio)], metadatos)


@pytest.mark.parametrize(("espacio", "acciones", "valida"), [
    (spaces.Discrete(2), [0, 1, 1], True),
    (spaces.Discrete(2), [-1, 0, 1], False),
    (spaces.Discrete(2), [0, 1, 2], False),
    (spaces.Discrete(3, start=1), [1, 2, 3], True),
    (spaces.Discrete(3, start=1), [0, 1, 2], False),
    (spaces.Discrete(3, start=1), [1, 2, 4], False),
], ids=["dentro", "negativa", "igual_a_n", "start_dentro", "bajo_start", "sobre_start_n"])
def test_rango_de_acciones_discretas(
    espacio: spaces.Discrete, acciones: list[int], valida: bool,
) -> None:
    """Los límites inferior y superior de Discrete(n, start) se respetan en ambos extremos."""
    episodio = _campos()
    episodio["acciones"] = np.array(acciones, dtype=np.int64)
    trayectoria = [Trayectoria(**episodio)]
    if valida:
        assert validar_demostraciones(trayectoria, OBSERVACION, espacio) is None
    else:
        with pytest.raises(ValueError, match="fuera del espacio"):
            validar_demostraciones(trayectoria, OBSERVACION, espacio)


@pytest.mark.parametrize(("valor", "valida"), [(-10.0, True), (10.0, True), (-10.5, False),
                                               (10.5, False)])
def test_limites_de_acciones_continuas(valor: float, valida: bool) -> None:
    """Las acciones Box se aceptan en los límites y se rechazan al superarlos."""
    episodio = _campos(continua=True)
    episodio["acciones"][1, 0] = valor
    trayectoria = [Trayectoria(**episodio)]
    if valida:
        validar_demostraciones(trayectoria, OBSERVACION, CONTINUA)
    else:
        with pytest.raises(ValueError, match="fuera de los límites"):
            validar_demostraciones(trayectoria, OBSERVACION, CONTINUA)


@pytest.mark.parametrize(("caso", "mensaje"), [
    ("obs_dimension", r"estados debe tener forma \[T, 5\]"),
    ("obs_fuera", "estados fuera de los límites"),
    ("siguiente_fuera", "siguientes_estados fuera de los límites"),
    ("accion_continua_dimension", r"acciones debe tener forma \[T, 3\]"),
])
def test_rechaza_datos_incompatibles_con_los_espacios(caso: str, mensaje: str) -> None:
    """Observaciones y acciones deben coincidir en forma y límites con el entorno.

    Con tres pasos, estados va de 0 a 11 y siguientes_estados de 4 a 15.
    """
    episodio = _campos(continua=caso.startswith("accion"))
    observacion, accion = OBSERVACION, CONTINUA if caso.startswith("accion") else DISCRETA
    if caso == "obs_dimension":
        observacion = spaces.Box(-np.inf, np.inf, (5,), np.float32)
    elif caso == "obs_fuera":
        observacion = spaces.Box(0.5, 100.0, (4,), np.float32)
    elif caso == "siguiente_fuera":
        observacion = spaces.Box(-100.0, 14.5, (4,), np.float32)
    else:
        accion = spaces.Box(-10.0, 10.0, (3,), np.float64)
    with pytest.raises(ValueError, match=mensaje):
        validar_demostraciones([Trayectoria(**episodio)], observacion, accion)


@pytest.mark.parametrize(("observacion", "accion", "mensaje"), [
    (spaces.Discrete(4), DISCRETA, "Espacio de observación no compatible"),
    (spaces.Box(-1.0, 1.0, (2, 2)), DISCRETA, "Espacio de observación no compatible"),
    (OBSERVACION, spaces.MultiDiscrete([2, 2]), "Espacio de acción no compatible"),
    (OBSERVACION, spaces.Box(-1.0, 1.0, (2, 1)), "Espacio de acción no compatible"),
], ids=["obs_discreta", "obs_2d", "accion_multidiscreta", "accion_box_2d"])
def test_rechaza_espacios_no_compatibles(
    tmp_path: Path, metadatos: dict[str, object], observacion: Any, accion: Any, mensaje: str,
) -> None:
    """Los espacios que el formato no admite se rechazan con un error que indica cuál."""
    with pytest.raises(ValueError, match=mensaje):
        guardar_demostraciones(tmp_path / "datos.npz", [_trayectoria()], metadatos,
                               observacion, accion)
    assert not (tmp_path / "datos.npz").exists()


def test_valida_con_los_espacios_reales_de_cartpole() -> None:
    """Un episodio real de CartPole pasa la validación y uno fuera de límites no.

    No se fija semilla: con acciones alternas, cualquier estado inicial produce
    un episodio completo que debe ser válido.
    """
    entorno = gym.make("CartPole-v1")
    try:
        estado, _ = entorno.reset()
        campos: dict[str, list[Any]] = {campo: [] for campo in (
            "estados", "acciones", "siguientes_estados", "recompensas_originales",
            "terminados", "truncados")}
        terminado = truncado = False
        while not (terminado or truncado):
            accion = len(campos["acciones"]) % 2
            siguiente, recompensa, terminado, truncado, _ = entorno.step(accion)
            for campo, valor in zip(campos, (estado, accion, siguiente, recompensa,
                                             terminado, truncado), strict=True):
                campos[campo].append(valor)
            estado = siguiente
        arrays = {campo: np.array(valores) for campo, valores in campos.items()}
        trayectoria = Trayectoria(id_episodio=0, **arrays)
        validar_demostraciones([trayectoria], entorno.observation_space, entorno.action_space)

        fuera = {**arrays, "estados": arrays["estados"].copy()}
        fuera["estados"][0, 0] = 5.0  # límite de posición del carro: 4.8
        with pytest.raises(ValueError, match="fuera de los límites"):
            validar_demostraciones([Trayectoria(id_episodio=1, **fuera)],
                                   entorno.observation_space, entorno.action_space)
    finally:
        entorno.close()


@pytest.mark.parametrize("campo", ["estados", "siguientes_estados", "acciones",
                                   "recompensas_originales"])
@pytest.mark.parametrize("valor", [np.nan, np.inf, -np.inf])
def test_rechaza_valores_no_finitos(
    tmp_path: Path, metadatos: dict[str, object], campo: str, valor: float,
) -> None:
    """Todos los arrays numéricos se comprueban frente a NaN e infinitos."""
    episodio = _campos(continua=True)
    episodio[campo] = episodio[campo].astype(float)
    episodio[campo].flat[0] = valor
    with pytest.raises(ValueError, match="finitos"):
        _guardar(tmp_path / "datos.npz", [Trayectoria(**episodio)], metadatos, continua=True)


@pytest.mark.parametrize("dtype", [object, complex, str, bool])
def test_rechaza_arrays_no_reales(
    tmp_path: Path, metadatos: dict[str, object], dtype: Any,
) -> None:
    """Los episodios no contienen arrays objeto, complejos, texto o estados booleanos."""
    episodio = _campos()
    episodio["estados"] = episodio["estados"].astype(dtype)
    with pytest.raises(ValueError, match="reales finitos"):
        _guardar(tmp_path / "datos.npz", [Trayectoria(**episodio)], metadatos)


@pytest.mark.parametrize("identificador", [True, np.bool_(False), "1", "", 1.5,
                                           np.float64(2.0), None])
def test_rechaza_identificadores_no_enteros(
    tmp_path: Path, metadatos: dict[str, object], identificador: Any,
) -> None:
    """Los IDs son enteros; cadenas, booleanos y reales se rechazan."""
    with pytest.raises(ValueError, match="id_episodio"):
        _guardar(tmp_path / "datos.npz", [_trayectoria(identificador=identificador)], metadatos)


@pytest.mark.parametrize("identificador", [
    ID_EPISODIO_MAXIMO + 1, ID_EPISODIO_MINIMO - 1, 2**70, np.uint64(ID_EPISODIO_MAXIMO + 1),
], ids=["maximo_mas_uno", "minimo_menos_uno", "2_elevado_70", "uint64_grande"])
def test_rechaza_ids_fuera_del_rango_int64(
    tmp_path: Path, metadatos: dict[str, object], identificador: Any,
) -> None:
    """Regresión: IDs válidos por tipo pero fuera de int64 rompían los splits y el lote."""
    with pytest.raises(ValueError, match="id_episodio debe estar en el rango de int64"):
        _guardar(tmp_path / "datos.npz", [_trayectoria(identificador=identificador)], metadatos)


@pytest.mark.parametrize("identificador", [
    ID_EPISODIO_MAXIMO, ID_EPISODIO_MINIMO, np.int64(ID_EPISODIO_MAXIMO),
    np.uint64(ID_EPISODIO_MAXIMO),
], ids=["maximo", "minimo", "maximo_int64", "maximo_uint64"])
def test_ids_en_los_limites_de_int64_se_guardan_y_cargan_exactos(
    tmp_path: Path, metadatos: dict[str, object], identificador: Any,
) -> None:
    """Los extremos del rango son válidos y vuelven exactamente iguales tras guardar y cargar."""
    ruta = _guardar(tmp_path / "datos.npz", [_trayectoria(identificador=identificador)], metadatos)
    trayectorias, _ = cargar_demostraciones(ruta, OBSERVACION, DISCRETA)
    assert trayectorias[0].id_episodio == int(identificador)
    assert type(trayectorias[0].id_episodio) is int


@pytest.mark.parametrize("ids", [(0, 0), (1, np.int64(1)), (np.int32(2), np.uint8(2))],
                         ids=["int", "int_y_numpy", "numpy_distintos_dtype"])
def test_rechaza_ids_duplicados(
    tmp_path: Path, metadatos: dict[str, object], ids: tuple[Any, Any],
) -> None:
    """Un mismo valor entero cuenta como duplicado aunque cambie su tipo NumPy."""
    episodios = [_trayectoria(identificador=ids[0]), _trayectoria(identificador=ids[1])]
    with pytest.raises(ValueError, match="duplicado"):
        _guardar(tmp_path / "datos.npz", episodios, metadatos)


@pytest.mark.parametrize("campo", ["entorno", "experto", "procedencia", "configuracion_experto"])
def test_exige_metadatos_de_procedencia(
    tmp_path: Path, metadatos: dict[str, object], campo: str,
) -> None:
    """La identidad del experto y el origen se conservan como parte del contrato."""
    del metadatos[campo]
    with pytest.raises(ValueError, match="Metadatos obligatorios"):
        _guardar(tmp_path / "datos.npz", [_trayectoria()], metadatos)


_NO_JSON = "Los metadatos deben ser JSON con valores finitos y sin referencias circulares"
_OBLIGATORIOS = "Metadatos obligatorios"


@pytest.mark.parametrize(("caso", "mensaje"), [
    ("nan", _NO_JSON), ("inf", _NO_JSON), ("nan_numpy", _NO_JSON), ("array", _NO_JSON),
    ("clave_no_str", "Las claves de los metadatos deben ser cadenas"),
    ("referencia_circular", _NO_JSON),
    ("no_mapping", _OBLIGATORIOS),
], ids=["nan", "inf", "nan_numpy", "array", "clave_no_str", "referencia_circular",
        "no_mapping"])
def test_rechaza_metadatos_no_json(
    tmp_path: Path, metadatos: dict[str, object], caso: str, mensaje: str,
) -> None:
    """Los metadatos deben ser JSON finito con claves de texto y sin referencias circulares."""
    if caso == "nan":
        metadatos["extra"] = np.nan
    elif caso == "inf":
        metadatos["extra"] = np.inf
    elif caso == "nan_numpy":
        metadatos["extra"] = np.float32(np.nan)
    elif caso == "array":
        metadatos["extra"] = np.ones(2)
    elif caso == "clave_no_str":
        metadatos["extra"] = {1: "se convertiría en texto"}
    elif caso == "referencia_circular":
        metadatos["extra"] = metadatos
    else:
        metadatos = None
    ruta = tmp_path / "nuevo" / "datos.npz"
    with pytest.raises(ValueError, match=re.escape(mensaje)):
        _guardar(ruta, [_trayectoria()], metadatos)
    assert not ruta.parent.exists()


def test_metadatos_con_escalares_numpy_y_tuplas(
    tmp_path: Path, metadatos: dict[str, object],
) -> None:
    """Regresión: np.int64 se rechazaba y np.float64 no. Los escalares NumPy se
    guardan como números de Python y las tuplas se recuperan como listas."""
    metadatos.update({"semilla": np.int64(42), "tasa": np.float32(0.5),
                      "determinista": np.bool_(False), "capas": (8, 8)})
    ruta = _guardar(tmp_path / "datos.npz", [_trayectoria()], metadatos)
    with np.load(ruta, allow_pickle=False) as archivo:
        guardados = json.loads(archivo["manifiesto"].item())["metadatos"]
    assert guardados["semilla"] == 42 and isinstance(guardados["semilla"], int)
    assert guardados["tasa"] == 0.5
    assert guardados["determinista"] is False
    assert guardados["capas"] == [8, 8]


def test_manifiesto_identico_con_metadatos_en_distinto_orden(
    tmp_path: Path, metadatos: dict[str, object],
) -> None:
    """Los mismos datos producen el mismo manifiesto aunque las claves de los
    metadatos, también las anidadas, se escriban en otro orden.

    Comprueba sort_keys=True: el texto del manifiesto es reproducible y sirve para
    identificar el dataset, por ejemplo mediante un hash.
    """
    metadatos["configuracion_experto"] = {"capas": [8, 8], "activacion": "tanh"}
    invertidos = dict(reversed(list(metadatos.items())))
    invertidos["configuracion_experto"] = {"activacion": "tanh", "capas": [8, 8]}
    assert list(invertidos) != list(metadatos)
    manifiestos = []
    for nombre, datos in (("a.npz", metadatos), ("b.npz", invertidos)):
        ruta = _guardar(tmp_path / nombre, [_trayectoria()], datos)
        with np.load(ruta, allow_pickle=False) as archivo:
            manifiestos.append(archivo["manifiesto"].item())
    assert manifiestos[0] == manifiestos[1]


@pytest.mark.parametrize("nombre", ["datos", "datos.npy", "datos.json", "datos.NPZ"])
def test_exige_extension_npz(
    tmp_path: Path, metadatos: dict[str, object], nombre: str,
) -> None:
    """No se agregan extensiones ni se escribe en una ruta diferente a la pedida."""
    with pytest.raises(ValueError, match="terminar en .npz"):
        _guardar(tmp_path / nombre, [_trayectoria()], metadatos)


def test_no_sobrescribe_archivo_existente(
    tmp_path: Path, metadatos: dict[str, object],
) -> None:
    """Un destino existente conserva exactamente sus bytes."""
    ruta = tmp_path / "datos.npz"
    ruta.write_bytes(b"contenido previo")
    with pytest.raises(FileExistsError):
        _guardar(str(ruta), [_trayectoria()], metadatos)
    assert ruta.read_bytes() == b"contenido previo"


def test_elimina_archivo_incompleto_si_falla_escritura(
    tmp_path: Path, metadatos: dict[str, object], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Un fallo real del escritor se propaga y no deja un dataset parcial."""
    def fallo(archivo: Any, **kwargs: Any) -> None:
        """Simula un fallo después de escribir parte del archivo."""
        archivo.write(b"parcial")
        raise OSError("fallo simulado de escritura")

    monkeypatch.setattr(np, "savez_compressed", fallo)
    ruta = tmp_path / "datos.npz"
    with pytest.raises(OSError, match="fallo simulado"):
        _guardar(ruta, [_trayectoria()], metadatos)
    assert not ruta.exists()


@pytest.mark.parametrize("pasos", [1, 3])
@pytest.mark.parametrize("continua", [False, True])
def test_trayectoria_conserva_campos_y_numero_pasos(pasos: int, continua: bool) -> None:
    """El contenedor conserva los arrays originales sin transformarlos."""
    datos = _campos(pasos, 9, continua)
    trayectoria = Trayectoria(**datos)
    assert trayectoria.numero_pasos == pasos
    assert trayectoria.id_episodio == 9
    for campo, valor in datos.items():
        if campo != "id_episodio":
            assert getattr(trayectoria, campo) is valor


def test_validar_no_modifica_ni_devuelve_datos() -> None:
    """La validación pública solo comprueba: no devuelve copias ni altera arrays."""
    datos = _campos(4, 1)
    originales = {k: v.copy() for k, v in datos.items() if isinstance(v, np.ndarray)}
    assert validar_demostraciones([Trayectoria(**datos)], OBSERVACION, DISCRETA) is None
    for campo, valor in originales.items():
        np.testing.assert_array_equal(datos[campo], valor)


# --- Carga -------------------------------------------------------------------


def _manipular(origen: Path, destino: Path, cambio: Any) -> Path:
    """Reescribe un NPZ válido aplicando un cambio; el manifiesto llega como dict."""
    with np.load(origen, allow_pickle=False) as archivo:
        contenidos: dict[str, Any] = {k: archivo[k] for k in archivo.files}
    contenidos["manifiesto"] = json.loads(contenidos["manifiesto"].item())
    cambio(contenidos)
    if isinstance(contenidos.get("manifiesto"), (dict, list)):
        contenidos["manifiesto"] = np.asarray(json.dumps(contenidos["manifiesto"]))
    np.savez_compressed(destino, **contenidos)
    return destino


@pytest.mark.parametrize("continua", [False, True])
@pytest.mark.parametrize("final", [(True, False), (False, True), (True, True)])
def test_carga_recupera_lo_guardado(
    tmp_path: Path, metadatos: dict[str, object], continua: bool, final: tuple[bool, bool],
) -> None:
    """Ida y vuelta: mismas partidas, orden, IDs, valores, dtypes y metadatos."""
    datos = [_campos(2, 5, continua), _campos(1, 2, continua), _campos(4, 9, continua)]
    for episodio in datos:
        episodio["terminados"][-1], episodio["truncados"][-1] = final
    metadatos["capas"] = (8, 8)
    ruta = _guardar(tmp_path / "datos.npz", [Trayectoria(**d) for d in datos], metadatos, continua)
    trayectorias, recuperados = cargar_demostraciones(
        ruta, OBSERVACION, CONTINUA if continua else DISCRETA)
    assert [t.id_episodio for t in trayectorias] == [5, 2, 9]
    assert [t.numero_pasos for t in trayectorias] == [2, 1, 4]
    assert all(isinstance(t, Trayectoria) for t in trayectorias)
    for trayectoria, original in zip(trayectorias, datos, strict=True):
        for campo, valor in original.items():
            if campo != "id_episodio":
                np.testing.assert_array_equal(getattr(trayectoria, campo), valor)
                assert getattr(trayectoria, campo).dtype == valor.dtype
    assert recuperados == {**metadatos, "capas": [8, 8]}


def test_carga_acepta_ruta_como_texto(tmp_path: Path, metadatos: dict[str, object]) -> None:
    """La ruta puede pasarse como str igual que al guardar."""
    ruta = _guardar(tmp_path / "datos.npz", [_trayectoria()], metadatos)
    trayectorias, _ = cargar_demostraciones(str(ruta), OBSERVACION, DISCRETA)
    assert len(trayectorias) == 1


def test_carga_ruta_inexistente(tmp_path: Path) -> None:
    """Una ruta que no existe se comunica como FileNotFoundError con mensaje propio.

    El mensaje distingue la comprobación explícita del módulo del error que daría
    open() por sí mismo, que también es FileNotFoundError.
    """
    with pytest.raises(FileNotFoundError, match="No existe el archivo de demostraciones"):
        cargar_demostraciones(tmp_path / "no_existe.npz", OBSERVACION, DISCRETA)


@pytest.mark.parametrize("nombre", ["datos", "datos.npy", "datos.NPZ"])
def test_carga_exige_extension_npz(tmp_path: Path, nombre: str) -> None:
    """La extensión se comprueba antes de leer."""
    (tmp_path / nombre).write_bytes(b"")
    with pytest.raises(ValueError, match="terminar en .npz"):
        cargar_demostraciones(tmp_path / nombre, OBSERVACION, DISCRETA)


def _miembro_npy_estropeado(caso: str) -> bytes:
    """Devuelve los bytes de un .npy de acciones roto de la forma indicada."""
    buffer = io.BytesIO()
    np.save(buffer, np.array([0, 1, 0], dtype=np.int64))
    bueno = buffer.getvalue()
    if caso == "cabecera_con_basura":      # NumPy falla con tokenize.TokenError
        return b"\x93NUMPY\x01\x00\x10\x00{basura cabecera}"
    if caso == "cabecera_claves_incorrectas":
        return b"\x93NUMPY\x01\x00" + (16).to_bytes(2, "little") + b"{'x': 1}       \n"
    return bueno[:-5]                       # datos truncados


def _deflate_roto(valido: Path, destino: Path) -> Path:
    """Sobrescribe el inicio de los datos comprimidos de un miembro: falla zlib, no el CRC."""
    datos = bytearray(valido.read_bytes())
    with zipfile.ZipFile(valido) as archivo:
        info = archivo.getinfo("trayectoria_0_estados.npy")
    nombre, extra = struct.unpack("<HH", datos[info.header_offset + 26:info.header_offset + 30])
    inicio = info.header_offset + 30 + nombre + extra
    datos[inicio:inicio + 6] = b"\xff" * 6
    destino.write_bytes(bytes(datos))
    return destino


_ILEGIBLE = "dañado o ilegible"
_ACCIONES_INVALIDAS = "El miembro 'trayectoria_0_acciones' no contiene un array NumPy válido"


@pytest.mark.parametrize(("caso", "mensaje"), [
    ("vacio", _ILEGIBLE),
    ("basura", _ILEGIBLE),
    ("npy_renombrado", "no es un NPZ de demostraciones"),
    ("zip_truncado", _ILEGIBLE),
    ("deflate_roto", _ILEGIBLE),
    ("array_con_pickle", _ACCIONES_INVALIDAS),
    ("cabecera_con_basura", _ACCIONES_INVALIDAS),
    ("cabecera_claves_incorrectas", _ACCIONES_INVALIDAS),
    ("datos_truncados", _ACCIONES_INVALIDAS),
], ids=["vacio", "basura", "npy_renombrado", "zip_truncado", "deflate_roto", "array_con_pickle",
        "cabecera_con_basura", "cabecera_claves_incorrectas", "datos_truncados"])
def test_carga_rechaza_archivos_que_no_son_demostraciones(
    tmp_path: Path, metadatos: dict[str, object], caso: str, mensaje: str,
) -> None:
    """Archivos vacíos, ajenos, truncados, con pickle o con miembros rotos producen el
    ValueError documentado con un mensaje propio.

    Regresión: "basura" y "array_con_pickle" daban el mensaje de NumPy, y
    "cabecera_con_basura" y "deflate_roto" escapaban como TokenError y zlib.error.
    """
    ruta = tmp_path / "datos.npz"
    valido = _guardar(tmp_path / "valido.npz", [_trayectoria()], metadatos)
    if caso == "vacio":
        ruta.write_bytes(b"")
    elif caso == "basura":
        ruta.write_bytes(b"no es un npz" * 10)
    elif caso == "npy_renombrado":
        with ruta.open("wb") as archivo:
            np.save(archivo, np.ones(3))
    elif caso == "zip_truncado":
        ruta.write_bytes(valido.read_bytes()[:200])
    elif caso == "deflate_roto":
        _deflate_roto(valido, ruta)
    elif caso == "array_con_pickle":
        _manipular(valido, ruta, lambda c: c.update(
            trayectoria_0_acciones=np.array([0, 1, 0], dtype=object)))
    else:
        _reescribir_zip(valido, ruta, sustituir={"trayectoria_0_acciones.npy": (
            "trayectoria_0_acciones.npy", _miembro_npy_estropeado(caso))})
    with pytest.raises(ValueError, match=re.escape(mensaje)):
        cargar_demostraciones(ruta, OBSERVACION, DISCRETA)


def _cambiar_episodio(clave: str, valor: Any) -> Any:
    """Devuelve un cambio que sustituye un campo del primer episodio del manifiesto."""
    return lambda c: c["manifiesto"]["episodios"][0].__setitem__(clave, valor)


_MAL_FORMADA = "Entrada 0 del manifiesto mal formada"
_OBLIGATORIOS_LISTA = ("Metadatos obligatorios: ['configuracion_experto', 'entorno', "
                       "'experto', 'procedencia']")

# Cada caso: (cambio sobre el NPZ válido, mensaje que debe producir la carga).
_MANIPULACIONES = {
    "sin_manifiesto": (lambda c: c.pop("manifiesto"), "El archivo no contiene manifiesto"),
    "manifiesto_no_escalar": (lambda c: c.update(manifiesto=np.array(["{}", "{}"])),
                              "El manifiesto debe ser un texto escalar"),
    "manifiesto_numerico": (lambda c: c.update(manifiesto=np.array(1.0)),
                            "El manifiesto debe ser un texto escalar"),
    "manifiesto_no_json": (lambda c: c.update(manifiesto=np.asarray("{no json")),
                           "El manifiesto no es JSON válido"),
    "manifiesto_lista": (lambda c: c.update(manifiesto=[]),
                         "El manifiesto debe ser un objeto JSON"),
    "formato_distinto": (lambda c: c["manifiesto"].update(formato="otro"),
                         "Formato desconocido: se esperaba 'tiiac-demostraciones'"),
    "version_2": (lambda c: c["manifiesto"].update(version=2), "Versión no soportada: 2"),
    "version_true": (lambda c: c["manifiesto"].update(version=True), "Versión no soportada: True"),
    "metadatos_incompletos": (lambda c: c["manifiesto"]["metadatos"].pop("experto"),
                              _OBLIGATORIOS_LISTA),
    "metadatos_no_dict": (lambda c: c["manifiesto"].update(metadatos=[]), _OBLIGATORIOS_LISTA),
    "episodios_vacios": (lambda c: c["manifiesto"].update(episodios=[]),
                         "El manifiesto debe listar al menos un episodio"),
    "episodio_sin_clave": (lambda c: c["manifiesto"]["episodios"][0].pop("prefijo"), _MAL_FORMADA),
    "episodio_clave_extra": (_cambiar_episodio("extra", 1), _MAL_FORMADA),
    "numero_pasos_texto": (_cambiar_episodio("numero_pasos", "3"), _MAL_FORMADA),
    "numero_pasos_bool": (_cambiar_episodio("numero_pasos", True), _MAL_FORMADA),
    "numero_pasos_distinto": (_cambiar_episodio("numero_pasos", 2),
                              "Episodio 0: numero_pasos no coincide con sus arrays"),
    "prefijo_no_texto": (_cambiar_episodio("prefijo", 0), _MAL_FORMADA),
    "prefijo_duplicado": (
        lambda c: c["manifiesto"]["episodios"][1].update(prefijo="trayectoria_0"),
        "El manifiesto repite prefijos de episodio"),
    "id_bool": (_cambiar_episodio("id_episodio", True),
                "Trayectoria 0: id_episodio debe ser un entero"),
    "id_texto": (_cambiar_episodio("id_episodio", "0"),
                 "Trayectoria 0: id_episodio debe ser un entero"),
    "id_duplicado": (lambda c: c["manifiesto"]["episodios"][1].update(id_episodio=0),
                     "id_episodio duplicado: 0"),
    "array_faltante": (lambda c: c.pop("trayectoria_1_truncados"),
                       "faltan ['trayectoria_1_truncados'], sobran []"),
    "array_sobrante": (lambda c: c.update(extra=np.zeros(1)), "faltan [], sobran ['extra']"),
    "array_alterado": (lambda c: c.update(trayectoria_0_acciones=np.array([0, 5, 0])),
                       "Trayectoria 0: acciones fuera del espacio Discrete(2)"),
}


@pytest.mark.parametrize("caso", list(_MANIPULACIONES))
def test_carga_rechaza_manifiesto_o_arrays_incoherentes(
    tmp_path: Path, metadatos: dict[str, object], caso: str,
) -> None:
    """Cada inconsistencia entre manifiesto, formato y arrays se rechaza al cargar.

    El mensaje confirma qué comprobación la detecta: un archivo manipulado suele
    romper varias cosas y otra comprobación posterior podría tapar la prevista.
    """
    cambio, mensaje = _MANIPULACIONES[caso]
    valido = _guardar(tmp_path / "valido.npz",
                      [_trayectoria(identificador=0), _trayectoria(identificador=1)], metadatos)
    ruta = _manipular(valido, tmp_path / "manipulado.npz", cambio)
    with pytest.raises(ValueError, match=re.escape(mensaje)):
        cargar_demostraciones(ruta, OBSERVACION, DISCRETA)


@pytest.mark.parametrize(("observacion", "accion", "mensaje"), [
    (spaces.Box(-np.inf, np.inf, (5,), np.float32), DISCRETA, "estados debe tener forma [T, 5]"),
    (spaces.Box(0.5, 100.0, (4,), np.float32), DISCRETA,
     "estados fuera de los límites del espacio de observación"),
    (OBSERVACION, spaces.Discrete(1), "acciones fuera del espacio Discrete(1)"),
    (OBSERVACION, CONTINUA, "acciones debe tener forma [T, 2]"),
], ids=["obs_otra_dimension", "obs_fuera_de_limites", "accion_fuera_de_rango", "accion_continua"])
def test_carga_valida_frente_al_entorno_de_uso(
    tmp_path: Path, metadatos: dict[str, object], observacion: Any, accion: Any, mensaje: str,
) -> None:
    """Un archivo válido se rechaza si no encaja con los espacios con los que se carga."""
    ruta = _guardar(tmp_path / "datos.npz", [_trayectoria()], metadatos)
    with pytest.raises(ValueError, match=re.escape(mensaje)):
        cargar_demostraciones(ruta, observacion, accion)


def _reescribir_zip(
    origen: Path, destino: Path, sustituir: dict[str, tuple[str, bytes]] | None = None,
    anadir: tuple[tuple[str, bytes], ...] = (),
) -> Path:
    """Copia los miembros del ZIP sustituyendo algunos por (nombre, bytes) y añadiendo otros."""
    sustituir = sustituir or {}
    with zipfile.ZipFile(origen) as entrada:
        miembros = [(nombre, entrada.read(nombre)) for nombre in entrada.namelist()]
    with zipfile.ZipFile(destino, "w") as salida, warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Duplicate name", UserWarning)
        for nombre, contenido in miembros:
            salida.writestr(*sustituir.get(nombre, (nombre, contenido)))
        for nombre, contenido in anadir:
            salida.writestr(nombre, contenido)
    return destino


@pytest.mark.parametrize("texto", ['NaN', 'Infinity', '-Infinity', '1e999', '-1e999',
                                   '{"a": [1, NaN]}', '[{"b": 1e999}]'],
                         ids=["nan", "inf", "menos_inf", "desborde", "menos_desborde",
                              "nan_anidado", "desborde_anidado"])
def test_carga_rechaza_numeros_no_finitos_en_metadatos(
    tmp_path: Path, metadatos: dict[str, object], texto: str,
) -> None:
    """Regresión: NaN, infinitos o desbordamientos del manifiesto se aceptaban al
    cargar aunque guardar los rechaza, también dentro de estructuras anidadas."""
    valido = _guardar(tmp_path / "valido.npz", [_trayectoria()], metadatos)

    def insertar(contenidos: dict[str, Any]) -> None:
        """Escribe el JSON a mano para conservar el literal no finito."""
        crudo = json.dumps(contenidos["manifiesto"])
        contenidos["manifiesto"] = np.asarray(crudo.replace('"licencia": "prueba"',
                                                            f'"licencia": {texto}'))

    ruta = _manipular(valido, tmp_path / "no_finito.npz", insertar)
    with pytest.raises(ValueError, match="no finito"):
        cargar_demostraciones(ruta, OBSERVACION, DISCRETA)


@pytest.mark.parametrize("miembro", ["manifiesto", "trayectoria_0_acciones",
                                     "trayectoria_0_estados"])
def test_carga_rechaza_miembros_que_no_son_arrays(
    tmp_path: Path, metadatos: dict[str, object], miembro: str,
) -> None:
    """Regresión: un miembro sin cabecera .npy llega como bytes y provocaba
    AttributeError en lugar del ValueError documentado.

    El "$" final exige el mensaje exacto: distingue este caso (bytes sin cabecera)
    del de cabecera rota, cuyo mensaje termina en "array NumPy válido".
    """
    valido = _guardar(tmp_path / "valido.npz", [_trayectoria()], metadatos)
    ruta = _reescribir_zip(valido, tmp_path / "bytes.npz",
                           sustituir={f"{miembro}.npy": (miembro, b"no es un array")})
    mensaje = f"El miembro '{miembro}' no contiene un array NumPy"
    with pytest.raises(ValueError, match=re.escape(mensaje) + "$"):
        cargar_demostraciones(ruta, OBSERVACION, DISCRETA)


@pytest.mark.parametrize("anadido", ["trayectoria_0_acciones.npy", "manifiesto"],
                         ids=["entrada_repetida", "mismo_nombre_sin_extension"])
def test_carga_rechaza_miembros_repetidos(
    tmp_path: Path, metadatos: dict[str, object], anadido: str,
) -> None:
    """Regresión: dos entradas con el mismo nombre de array se aceptaban y NumPy
    leía una de ellas sin avisar. NpzFile quita la extensión .npy, así que
    "manifiesto" y "manifiesto.npy" también chocan."""
    valido = _guardar(tmp_path / "valido.npz", [_trayectoria()], metadatos)
    with zipfile.ZipFile(valido) as archivo:
        contenido = archivo.read("trayectoria_0_acciones.npy")
    ruta = _reescribir_zip(valido, tmp_path / "repetido.npz", anadir=((anadido, contenido),))
    with pytest.raises(ValueError, match="repetidos"):
        cargar_demostraciones(ruta, OBSERVACION, DISCRETA)


def test_carga_distingue_directorio_de_ruta_inexistente(tmp_path: Path) -> None:
    """Regresión: un directorio con extensión .npz se notificaba como inexistente."""
    (tmp_path / "carpeta.npz").mkdir()
    with pytest.raises(IsADirectoryError, match="directorio"):
        cargar_demostraciones(tmp_path / "carpeta.npz", OBSERVACION, DISCRETA)


@pytest.mark.filterwarnings("error")
def test_carga_cierra_el_archivo_si_el_zip_esta_danado(
    tmp_path: Path, metadatos: dict[str, object],
) -> None:
    """Regresión: con un ZIP truncado, np.load fallaba sin cerrar el archivo y
    quedaba un ResourceWarning por descriptor abierto."""
    valido = _guardar(tmp_path / "valido.npz", [_trayectoria()], metadatos)
    ruta = tmp_path / "truncado.npz"
    ruta.write_bytes(valido.read_bytes()[:200])
    with pytest.raises(ValueError, match="dañado"):
        cargar_demostraciones(ruta, OBSERVACION, DISCRETA)
    gc.collect()



# --- grabación ---------------------------------------------------------------------
# Políticas fijas en lugar del experto, para no entrenar nada: equilibrio aguanta
# las 500 partidas de CartPole-v1 (truncadas) y cero cae enseguida (terminadas).

RAIZ = Path(__file__).resolve().parent.parent


def _equilibrio(observacion: np.ndarray) -> int:
    """Empuja hacia el lado al que cae el palo."""
    return int(observacion[2] + 0.5 * observacion[3] > 0)


def _cero(observacion: np.ndarray) -> int:
    """Empuja siempre a la izquierda."""
    return 0


def _al_azar_con_pytorch(observacion: np.ndarray) -> int:
    """Muestrea con el generador global de PyTorch, como predict de sb3."""
    return int(torch.rand(()) < 0.5)


def _par_constante(observacion: np.ndarray) -> np.ndarray:
    """Acción continua fija dentro de los límites de Pendulum."""
    return np.array([0.5], dtype=np.float32)


def _cfg_demostraciones(entorno: str = "CartPole-v1", **cambios: Any) -> dict[str, Any]:
    """Configuración del proyecto con un dataset pequeño: 4 + 2 partidas."""
    cfg = resolver_configuracion(RAIZ / "configs" / "default.yaml", "cartpole")
    cfg["entorno"]["id"] = entorno
    cfg["demostraciones"].update(partidas_train=4, partidas_validacion=2, subconjuntos=[1, 4])
    cfg["demostraciones"].update(cambios)
    return cfg


def _metadatos_grabacion() -> dict[str, object]:
    return {"entorno": "CartPole-v1", "experto": "política fija de prueba",
            "procedencia": "tests", "configuracion_experto": {}}


@pytest.mark.parametrize("politica,truncada", [(_equilibrio, True), (_cero, False)],
                         ids=["truncada", "terminada"])
def test_jugar_partida_devuelve_una_partida_completa_y_valida(politica, truncada) -> None:
    entorno = crear_entorno({"entorno": {"id": "CartPole-v1"}}, None)
    try:
        partida = jugar_partida(entorno, politica, 3, 7)
        validar_demostraciones([partida], entorno.observation_space, entorno.action_space)
    finally:
        entorno.close()
    assert partida.id_episodio == 7
    assert (partida.numero_pasos == 500) is truncada
    assert (bool(partida.truncados[-1]), bool(partida.terminados[-1])) == (truncada, not truncada)
    assert not (partida.terminados[:-1] | partida.truncados[:-1]).any()
    assert partida.estados.shape == (partida.numero_pasos, 4)
    assert partida.estados.dtype == np.float32
    assert partida.acciones.shape == (partida.numero_pasos,)
    assert partida.acciones.dtype.kind == "i"
    assert partida.recompensas_originales.dtype == np.float64
    # CartPole da 1 por paso: el retorno es la longitud.
    assert retorno_partida(partida) == partida.numero_pasos


def test_jugar_partida_empieza_por_el_primer_estado_de_su_semilla() -> None:
    # A diferencia de sembrar_entorno, que gasta un reset.
    entorno = crear_entorno({"entorno": {"id": "CartPole-v1"}}, None)
    otro = crear_entorno({"entorno": {"id": "CartPole-v1"}}, None)
    try:
        inicial, _ = otro.reset(seed=11)
        partida = jugar_partida(entorno, _cero, 11, 0)
    finally:
        entorno.close()
        otro.close()
    np.testing.assert_array_equal(partida.estados[0], inicial)
    np.testing.assert_array_equal(partida.estados[1:], partida.siguientes_estados[:-1])


def test_jugar_partida_es_reproducible_y_no_altera_pytorch() -> None:
    entorno = crear_entorno({"entorno": {"id": "CartPole-v1"}}, None)
    try:
        estado = torch.get_rng_state()
        a = jugar_partida(entorno, _al_azar_con_pytorch, 5, 0)
        assert torch.equal(torch.get_rng_state(), estado)
        torch.rand(9)  # el generador global cambia y no debe influir
        b = jugar_partida(entorno, _al_azar_con_pytorch, 5, 0)
        c = jugar_partida(entorno, _al_azar_con_pytorch, 6, 0)
    finally:
        entorno.close()
    assert huella_partidas([a]) == huella_partidas([b]) != huella_partidas([c])


def test_jugar_partida_con_acciones_continuas() -> None:
    entorno = crear_entorno({"entorno": {"id": "Pendulum-v1"}}, None)
    try:
        partida = jugar_partida(entorno, _par_constante, 2, 0)
        validar_demostraciones([partida], entorno.observation_space, entorno.action_space)
    finally:
        entorno.close()
    assert partida.numero_pasos == 200 and partida.truncados[-1]
    assert partida.acciones.shape == (200, 1) and partida.acciones.dtype == np.float32
    assert (partida.acciones == 0.5).all()


def test_jugar_partida_no_comparte_arrays_con_la_politica() -> None:
    # Una política que modifica la observación recibida no cambia lo grabado:
    # cada estado se copia al guardarlo.
    def destructiva(observacion: np.ndarray) -> int:
        accion = _equilibrio(observacion)
        observacion[:] = 99.0
        return accion

    entorno = crear_entorno({"entorno": {"id": "CartPole-v1"}}, None)
    try:
        partida = jugar_partida(entorno, destructiva, 1, 0)
    finally:
        entorno.close()
    assert not (partida.estados == 99.0).any()
    assert not (partida.siguientes_estados == 99.0).any()
    np.testing.assert_array_equal(partida.estados[1:], partida.siguientes_estados[:-1])


@pytest.mark.parametrize("semilla", [-1, 1.5, True, "3", None])
def test_jugar_partida_rechaza_semillas_no_validas(semilla: Any) -> None:
    entorno = crear_entorno({"entorno": {"id": "CartPole-v1"}}, None)
    try:
        with pytest.raises(ValueError, match="semilla"):
            jugar_partida(entorno, _cero, semilla, 0)
    finally:
        entorno.close()


@pytest.mark.parametrize("identificador", [True, 1.5, "0", ID_EPISODIO_MAXIMO + 1, None])
def test_jugar_partida_rechaza_ids_no_validos(identificador: Any) -> None:
    entorno = crear_entorno({"entorno": {"id": "CartPole-v1"}}, None)
    try:
        with pytest.raises(ValueError, match="id_episodio"):
            jugar_partida(entorno, _cero, 0, identificador)
    finally:
        entorno.close()


def test_jugar_partida_exige_un_entorno_con_limite_de_pasos() -> None:
    sin_limite = CartPoleEnv()  # sin TimeLimit: una buena política no acabaría nunca
    try:
        with pytest.raises(ValueError, match="límite de pasos"):
            jugar_partida(sin_limite, _equilibrio, 0, 0)
    finally:
        sin_limite.close()
    with pytest.raises(TypeError, match="Gymnasium"):
        jugar_partida(object(), _equilibrio, 0, 0)


def test_grabar_usa_una_semilla_independiente_por_partida() -> None:
    entorno = crear_entorno({"entorno": {"id": "CartPole-v1"}}, None)
    try:
        cuatro = grabar_demostraciones(entorno, _al_azar_con_pytorch, 4, 8)
        dos = grabar_demostraciones(entorno, _al_azar_con_pytorch, 2, 8)
        sueltas = [jugar_partida(entorno, _al_azar_con_pytorch, semilla, i)
                   for i, semilla in enumerate(semillas_independientes(8, 4))]
    finally:
        entorno.close()
    assert [p.id_episodio for p in cuatro] == [0, 1, 2, 3]
    assert huella_partidas(cuatro) == huella_partidas(sueltas)
    # Cada partida depende solo de su semilla: grabar menos da las primeras.
    assert huella_partidas(dos) == huella_partidas(cuatro[:2])
    assert len({huella_partidas([p]) for p in cuatro}) == 4


@pytest.mark.parametrize("partidas", [0, -1, True, 2.0, "3"])
def test_grabar_rechaza_un_numero_de_partidas_no_valido(partidas: Any) -> None:
    entorno = crear_entorno({"entorno": {"id": "CartPole-v1"}}, None)
    try:
        with pytest.raises(ValueError, match="cantidad"):
            grabar_demostraciones(entorno, _cero, partidas, 0)
    finally:
        entorno.close()


def test_grabar_rechaza_acciones_fuera_del_espacio() -> None:
    # Pendulum recorta la acción por dentro, así que solo la validación lo detecta.
    entorno = crear_entorno({"entorno": {"id": "Pendulum-v1"}}, None)
    try:
        with pytest.raises(ValueError, match="fuera de los límites"):
            grabar_demostraciones(entorno, lambda o: np.array([5.0], np.float32), 1, 0)
    finally:
        entorno.close()


def test_retorno_partida_suma_en_orden_paso_a_paso() -> None:
    # Regresión: sum de Python 3.12 compensa el redondeo y daba 1.0; acumulando
    # paso a paso, como durante la partida, el 1.0 se pierde y queda 0.0. Con
    # sum, evaluar_retorno cambiaba el último decimal de los retornos de Pendulum.
    partida = Trayectoria(**{**_campos(3), "recompensas_originales":
                             np.array([1e16, 1.0, -1e16])})
    assert sum(partida.recompensas_originales.tolist()) == 1.0
    assert retorno_partida(partida) == 0.0


@pytest.mark.parametrize("campo", ["estados", "acciones", "siguientes_estados",
                                   "recompensas_originales", "terminados", "truncados"])
def test_huella_partidas_cambia_con_cualquier_campo(campo: str) -> None:
    base = _campos(3)
    original = huella_partidas([Trayectoria(**base)])
    cambiado = dict(base)
    cambiado[campo] = base[campo].copy()
    cambiado[campo][0] = not cambiado[campo][0] if campo in ("terminados", "truncados") \
        else cambiado[campo][0] + 1
    assert huella_partidas([Trayectoria(**cambiado)]) != original
    assert huella_partidas([Trayectoria(**base)]) == original


def test_huella_partidas_distingue_id_tipo_y_orden() -> None:
    a, b = _trayectoria(3, 0), _trayectoria(4, 1)
    original = huella_partidas([a, b])
    assert huella_partidas([b, a]) != original
    assert huella_partidas([_trayectoria(3, 5), b]) != original
    # Mismos bytes con otro tipo (ceros enteros y reales) o con otra forma.
    enteras = Trayectoria(**{**_campos(3), "acciones": np.zeros(3, np.int64)})
    reales = Trayectoria(**{**_campos(3), "acciones": np.zeros(3, np.float64)})
    assert enteras.acciones.tobytes() == reales.acciones.tobytes()
    assert huella_partidas([enteras]) != huella_partidas([reales])
    matriz = Trayectoria(**{**_campos(3), "estados": a.estados.reshape(4, 3)})
    assert huella_partidas([matriz]) != huella_partidas([a])
    no_contiguo = Trayectoria(**{**_campos(3), "estados": np.asfortranarray(a.estados)})
    assert huella_partidas([no_contiguo]) == huella_partidas([a])
    for invalido in ("abc", [object()], None):
        with pytest.raises(ValueError, match="Trayectoria"):
            huella_partidas(invalido)


def test_huella_partidas_no_depende_de_los_metadatos(
    tmp_path: Path, metadatos: dict[str, object],
) -> None:
    trayectorias = [_trayectoria(3, 0), _trayectoria(2, 1)]
    otros = {**metadatos, "procedencia": "otro commit"}
    rutas = [_guardar(tmp_path / "a.npz", trayectorias, metadatos),
             _guardar(tmp_path / "b.npz", trayectorias, otros)]
    assert huella_archivo(rutas[0]) != huella_archivo(rutas[1])
    cargadas = [cargar_demostraciones(r, OBSERVACION, DISCRETA)[0] for r in rutas]
    assert huella_partidas(cargadas[0]) == huella_partidas(cargadas[1]) \
        == huella_partidas(trayectorias)


def test_config_demostraciones_de_cartpole_es_valida_y_es_una_copia() -> None:
    cfg = resolver_configuracion(RAIZ / "configs" / "default.yaml", "cartpole")
    dcfg = config_demostraciones(cfg)
    assert dcfg == cfg["demostraciones"]
    assert dcfg["determinista"] is False  # el original graba con el experto estocástico
    dcfg["semilla"] = 1
    assert cfg["demostraciones"]["semilla"] != 1
    config_demostraciones(_cfg_demostraciones(semilla=np.int64(4)))


@pytest.mark.parametrize("cambios,mensaje", [
    ({"semilla": -1}, "demostraciones.semilla"),
    ({"semilla": True}, "demostraciones.semilla"),
    ({"semilla": 1.5}, "demostraciones.semilla"),
    ({"determinista": "no"}, "demostraciones.determinista"),
    ({"determinista": 0}, "demostraciones.determinista"),
    ({"partidas_train": 0}, "cantidad_train"),
    ({"partidas_validacion": 0}, "cantidad_validacion"),
    ({"subconjuntos": [1, 9]}, "supera"),
    ({"subconjuntos": [4, 1]}, "crecientes"),
    ({"subconjuntos": []}, "no vacía"),
])
def test_config_demostraciones_rechaza_valores_no_validos(cambios: dict, mensaje: str) -> None:
    with pytest.raises(ValueError, match=mensaje):
        config_demostraciones(_cfg_demostraciones(**cambios))


@pytest.mark.parametrize("clave", ["partidas_train", "partidas_validacion", "subconjuntos",
                                   "semilla", "determinista"])
def test_config_demostraciones_exige_cada_clave(clave: str) -> None:
    cfg = _cfg_demostraciones()
    del cfg["demostraciones"][clave]
    with pytest.raises(ValueError, match=f"demostraciones.{clave}"):
        config_demostraciones(cfg)
    with pytest.raises(ValueError, match="sección demostraciones"):
        config_demostraciones({})


@pytest.mark.parametrize("entorno,politica", [("CartPole-v1", _equilibrio),
                                              ("Pendulum-v1", _par_constante)],
                         ids=["discreto", "continuo"])
def test_generar_dataset_guarda_partidas_y_splits_coherentes(
    tmp_path: Path, entorno: str, politica: Any,
) -> None:
    cfg = _cfg_demostraciones(entorno)
    resultado = generar_dataset(cfg, politica, tmp_path / "datos", _metadatos_grabacion())
    assert resultado["demostraciones"] == tmp_path / "datos" / ARCHIVO_DEMOSTRACIONES
    assert resultado["splits"] == tmp_path / "datos" / ARCHIVO_SPLITS
    espacios = crear_entorno(cfg, None)
    try:
        partidas, guardados = cargar_demostraciones(resultado["demostraciones"],
                                                    espacios.observation_space,
                                                    espacios.action_space)
    finally:
        espacios.close()
    assert [p.id_episodio for p in partidas] == list(range(6))
    assert resultado["huella_dataset"] == huella_archivo(resultado["demostraciones"])
    assert resultado["huella_splits"] == huella_archivo(resultado["splits"])
    assert resultado["huella_partidas"] == huella_partidas(partidas)
    splits = cargar_splits(resultado["splits"], resultado["huella_dataset"], range(6))
    assert (len(splits.train), len(splits.validacion)) == (4, 2)
    assert sorted(splits.subconjuntos) == [1, 4]
    assert [list(splits.train), list(splits.validacion)] == [resultado["train"],
                                                             resultado["validacion"]]
    grabacion = guardados["grabacion"]
    semilla_partidas, semilla_splits = semillas_independientes(cfg["demostraciones"]["semilla"], 2)
    assert grabacion == {"semilla": cfg["demostraciones"]["semilla"],
                         "semilla_partidas": semilla_partidas, "semilla_splits": semilla_splits,
                         "partidas": 6, "determinista": False,
                         "huella_partidas": resultado["huella_partidas"],
                         "retornos": resultado["retornos"], "longitudes": resultado["longitudes"]}
    assert resultado["retornos"] == [retorno_partida(p) for p in partidas]
    assert resultado["pasos"] == sum(p.numero_pasos for p in partidas)
    # El reparto usa su propia semilla, no la base ni la de las partidas.
    esperados = crear_splits(partidas, 4, 2, [1, 4], crear_generador(semilla_splits),
                             resultado["huella_dataset"])
    assert esperados == splits
    assert {k: guardados[k] for k in _metadatos_grabacion()} == _metadatos_grabacion()


def test_generar_dataset_es_reproducible_byte_a_byte(tmp_path: Path) -> None:
    cfg = _cfg_demostraciones()
    a = generar_dataset(cfg, _al_azar_con_pytorch, tmp_path / "a", _metadatos_grabacion())
    b = generar_dataset(cfg, _al_azar_con_pytorch, tmp_path / "b", _metadatos_grabacion())
    assert (a["huella_dataset"], a["huella_splits"]) == (b["huella_dataset"], b["huella_splits"])
    otra = generar_dataset(_cfg_demostraciones(semilla=7), _al_azar_con_pytorch, tmp_path / "c",
                           _metadatos_grabacion())
    assert otra["huella_partidas"] != a["huella_partidas"]


def test_generar_dataset_no_sobrescribe(tmp_path: Path) -> None:
    cfg = _cfg_demostraciones()
    resultado = generar_dataset(cfg, _equilibrio, tmp_path, _metadatos_grabacion())
    with pytest.raises(FileExistsError):
        generar_dataset(cfg, _cero, tmp_path, _metadatos_grabacion())
    assert huella_archivo(resultado["demostraciones"]) == resultado["huella_dataset"]
    assert huella_archivo(resultado["splits"]) == resultado["huella_splits"]


def test_generar_dataset_no_deja_un_npz_sin_splits(tmp_path: Path) -> None:
    # Si los splits no se pueden guardar, se borra el NPZ que acaba de crear,
    # y el archivo que ya estaba se conserva.
    (tmp_path / ARCHIVO_SPLITS).write_text("de otro dataset", encoding="utf-8")
    with pytest.raises(FileExistsError):
        generar_dataset(_cfg_demostraciones(), _equilibrio, tmp_path, _metadatos_grabacion())
    assert not (tmp_path / ARCHIVO_DEMOSTRACIONES).exists()
    assert (tmp_path / ARCHIVO_SPLITS).read_text(encoding="utf-8") == "de otro dataset"


@pytest.mark.parametrize("cfg_cambios,metadatos,mensaje", [
    ({"subconjuntos": [1, 9]}, None, "supera"),
    ({"semilla": -1}, None, "semilla"),
    ({}, {"grabacion": 1}, "grabacion"),
    ({}, "no es un diccionario", "grabacion"),
    ({}, {"entorno": "CartPole-v1"}, "Metadatos obligatorios"),
])
def test_generar_dataset_falla_sin_crear_archivos(
    tmp_path: Path, cfg_cambios: dict, metadatos: Any, mensaje: str,
) -> None:
    if isinstance(metadatos, dict) and "grabacion" in metadatos:
        metadatos = {**_metadatos_grabacion(), **metadatos}
    with pytest.raises(ValueError, match=mensaje):
        generar_dataset(_cfg_demostraciones(**cfg_cambios), _equilibrio, tmp_path / "datos",
                        _metadatos_grabacion() if metadatos is None else metadatos)
    assert not (tmp_path / "datos").exists()
