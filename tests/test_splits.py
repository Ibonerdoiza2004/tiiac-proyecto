"""Pruebas de crear_generador, LoteTransiciones, splits por trayectoria y submuestreo."""

import hashlib
import itertools
import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.data.loaders import submuestrear_pares
from src.data.splits import (
    SplitsDemostraciones, cargar_splits, crear_splits, guardar_splits, huella_archivo,
    validar_splits,
)
from src.data.tipos import (
    ID_EPISODIO_MAXIMO, ID_EPISODIO_MINIMO, LoteTransiciones, Trayectoria, es_id_episodio,
)
from src.utils.seed import crear_generador

HUELLA = "a" * 64
OTRA_HUELLA = "b" * 64


def _partida(identificador: int, pasos: int = 5, continua: bool = False) -> Trayectoria:
    """Partida válida y exacta: el estado del paso t vale 100 * id + t en cada columna."""
    base = 100 * identificador + np.arange(pasos + 1, dtype=np.float32)
    observaciones = np.repeat(base[:, None], 4, axis=1)
    terminados = np.zeros(pasos, dtype=bool)
    terminados[-1] = True
    acciones = (np.stack([np.arange(pasos), -np.arange(pasos)], axis=1).astype(np.float64)
                if continua else np.arange(pasos, dtype=np.int64) % 2)
    return Trayectoria(identificador, observaciones[:-1], acciones, observaciones[1:],
                       np.ones(pasos, dtype=np.float32), terminados,
                       np.zeros(pasos, dtype=bool))


def _quince() -> list[Trayectoria]:
    """Las 15 partidas del plan: IDs 0 a 14."""
    return [_partida(i) for i in range(15)]


def _splits(semilla: int = 0, cantidades: tuple[int, ...] = (1, 4, 7, 10)) -> SplitsDemostraciones:
    """Splits 10 + 5 de las 15 partidas con la semilla indicada."""
    return crear_splits(_quince(), 10, 5, cantidades, crear_generador(semilla), HUELLA)


# --- crear_generador ---------------------------------------------------------


@pytest.mark.parametrize("semilla", [0, 1, 402, np.int64(7)])
def test_generador_reproducible(semilla: Any) -> None:
    """La misma semilla produce exactamente la misma secuencia."""
    a = crear_generador(semilla).integers(0, 1000, 20)
    b = crear_generador(semilla).integers(0, 1000, 20)
    np.testing.assert_array_equal(a, b)


def test_generadores_con_semillas_distintas_difieren() -> None:
    """Semillas distintas dan secuencias distintas."""
    secuencias = {tuple(crear_generador(s).integers(0, 10**6, 5)) for s in range(10)}
    assert len(secuencias) == 10


@pytest.mark.parametrize("semilla", [-1, 1.5, True, np.bool_(False), "3", None])
def test_generador_rechaza_semillas_invalidas(semilla: Any) -> None:
    """Solo se aceptan enteros no negativos."""
    with pytest.raises(ValueError, match="entero no negativo"):
        crear_generador(semilla)


# --- LoteTransiciones --------------------------------------------------------


def test_lote_conserva_campos_y_numero_transiciones() -> None:
    """El contenedor guarda los arrays tal cual y cuenta sus pares."""
    campos = {"estados": np.zeros((3, 4)), "acciones": np.array([0, 1, 0]),
              "ids_episodio": np.array([2, 2, 5]), "pasos": np.array([0, 10, 3])}
    lote = LoteTransiciones(**campos)
    assert lote.numero_transiciones == 3
    for campo, valor in campos.items():
        assert getattr(lote, campo) is valor


# --- es_id_episodio ----------------------------------------------------------


@pytest.mark.parametrize(("valor", "valido"), [
    (0, True), (ID_EPISODIO_MAXIMO, True), (ID_EPISODIO_MINIMO, True),
    (np.int64(ID_EPISODIO_MAXIMO), True), (np.uint64(ID_EPISODIO_MAXIMO), True),
    (np.int8(-3), True),
    (ID_EPISODIO_MAXIMO + 1, False), (ID_EPISODIO_MINIMO - 1, False),
    (np.uint64(ID_EPISODIO_MAXIMO + 1), False),
    (True, False), (np.bool_(False), False), (1.0, False), ("1", False), (None, False),
])
def test_es_id_episodio_en_los_limites(valor: Any, valido: bool) -> None:
    """Enteros no booleanos dentro de int64, comprobando ambos extremos y uno más allá."""
    assert es_id_episodio(valor) is valido


# --- huella_archivo ----------------------------------------------------------


def test_huella_coincide_con_sha256_y_cambia_con_un_byte(tmp_path: Path) -> None:
    """La huella es el SHA-256 del archivo, también por encima del tamaño de bloque."""
    contenido = bytes(range(256)) * 5000          # ~1,3 MB: más de un bloque de 1 MB
    ruta = tmp_path / "datos.npz"
    ruta.write_bytes(contenido)
    assert huella_archivo(ruta) == hashlib.sha256(contenido).hexdigest()
    assert huella_archivo(str(ruta)) == huella_archivo(ruta)
    ruta.write_bytes(contenido[:-1] + b"\x00")
    assert huella_archivo(ruta) != hashlib.sha256(contenido).hexdigest()


def test_huella_de_archivo_inexistente(tmp_path: Path) -> None:
    """Una ruta inexistente se comunica como FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        huella_archivo(tmp_path / "no_existe.npz")


# --- crear_splits ------------------------------------------------------------


@pytest.mark.parametrize("semilla", range(20))
def test_splits_reparten_partidas_enteras_sin_solapes(semilla: int) -> None:
    """Para 20 semillas: 10 + 5 disjuntos que cubren las 15 y subconjuntos anidados."""
    splits = _splits(semilla)
    assert len(splits.train) == 10 and len(splits.validacion) == 5
    assert not set(splits.train) & set(splits.validacion)
    assert set(splits.train) | set(splits.validacion) == set(range(15))
    assert all(type(i) is int for i in splits.train + splits.validacion)
    assert list(splits.subconjuntos) == [1, 4, 7, 10]
    for cantidad, ids in splits.subconjuntos.items():
        assert ids == splits.train[:cantidad]
    assert splits.huella_dataset == HUELLA


def test_splits_reproducibles_e_independientes_del_orden_de_entrada() -> None:
    """Misma semilla, mismo reparto, aunque las partidas lleguen en otro orden."""
    partidas = _quince()
    invertidas = list(reversed(partidas))
    a = crear_splits(partidas, 10, 5, (1, 4, 7, 10), crear_generador(3), HUELLA)
    b = crear_splits(invertidas, 10, 5, (1, 4, 7, 10), crear_generador(3), HUELLA)
    assert a == b


def test_semillas_distintas_dan_repartos_distintos() -> None:
    """Con 20 semillas aparecen repartos y órdenes de train distintos."""
    assert len({_splits(s).train for s in range(20)}) > 1
    assert len({frozenset(_splits(s).validacion) for s in range(20)}) > 1


def test_todas_las_combinaciones_de_subconjuntos_estan_anidadas() -> None:
    """Dominio completo: las 1023 secuencias crecientes de cantidades entre 1 y 10."""
    combinaciones = [c for n in range(1, 11) for c in itertools.combinations(range(1, 11), n)]
    assert len(combinaciones) == 1023
    for cantidades in combinaciones:
        splits = _splits(5, cantidades)
        anterior: set[int] = set()
        for cantidad in cantidades:
            ids = splits.subconjuntos[cantidad]
            assert len(ids) == cantidad and set(ids) <= set(splits.train)
            assert anterior <= set(ids)
            anterior = set(ids)


@pytest.mark.parametrize(("argumentos", "mensaje"), [
    ({"cantidad_train": 0}, "cantidad_train debe ser un entero positivo"),
    ({"cantidad_train": -1}, "cantidad_train debe ser un entero positivo"),
    ({"cantidad_train": 10.0}, "cantidad_train debe ser un entero positivo"),
    ({"cantidad_train": True}, "cantidad_train debe ser un entero positivo"),
    ({"cantidad_validacion": 0}, "cantidad_validacion debe ser un entero positivo"),
    ({"cantidades_subconjuntos": ()}, "secuencia no vacía de enteros"),
    ({"cantidades_subconjuntos": "1,4"}, "secuencia no vacía de enteros"),
    ({"cantidades_subconjuntos": (0, 4)}, "enteros positivos"),
    ({"cantidades_subconjuntos": (1, 4.5)}, "enteros positivos"),
    ({"cantidades_subconjuntos": (4, 1)}, "estrictamente crecientes"),
    ({"cantidades_subconjuntos": (1, 4, 4)}, "estrictamente crecientes"),
    ({"cantidades_subconjuntos": (1, 11)}, "supera las 10 partidas de train"),
    ({"cantidad_train": 9, "cantidades_subconjuntos": (1, 4, 7)},
     "9 + 5 = 14 partidas y hay 15"),
    ({"cantidad_validacion": 6}, "10 + 6 = 16 partidas y hay 15"),
    ({"generador": np.random.RandomState(0)}, "numpy.random.Generator"),
    ({"generador": None}, "numpy.random.Generator"),
    ({"huella_dataset": "a" * 63}, "SHA-256 hexadecimal"),
    ({"huella_dataset": "A" * 64}, "SHA-256 hexadecimal"),
    ({"huella_dataset": None}, "SHA-256 hexadecimal"),
    ({"trayectorias": [_partida(0)] * 15}, "id_episodio repetidos"),
    ({"trayectorias": [_partida(i) for i in range(14)] + [{"id_episodio": 14}]},
     "objetos Trayectoria"),
    ({"trayectorias": "partidas"}, "objetos Trayectoria"),
])
def test_crear_splits_rechaza_entradas_invalidas(argumentos: dict[str, Any], mensaje: str) -> None:
    """Cada parámetro inválido produce su mensaje propio."""
    llamada = {"trayectorias": _quince(), "cantidad_train": 10, "cantidad_validacion": 5,
               "cantidades_subconjuntos": (1, 4, 7, 10), "generador": crear_generador(0),
               "huella_dataset": HUELLA, **argumentos}
    with pytest.raises(ValueError, match=re.escape(mensaje)):
        crear_splits(**llamada)


def test_splits_conservan_exactos_los_ids_extremos() -> None:
    """Los IDs en los límites de int64 se reparten sin perder precisión.

    Los IDs se ordenan y se barajan posiciones, así que nunca pasan por un array de
    NumPy que pudiera convertirlos a float64.
    """
    ids = [ID_EPISODIO_MINIMO, ID_EPISODIO_MINIMO + 1, -1, 0, ID_EPISODIO_MAXIMO - 1,
           ID_EPISODIO_MAXIMO]
    partidas = [_partida(0, 3) for _ in ids]
    partidas = [replace(p, id_episodio=i) for p, i in zip(partidas, ids, strict=True)]
    splits = crear_splits(partidas, 4, 2, (1, 4), crear_generador(0), HUELLA)
    assert set(splits.train) | set(splits.validacion) == set(ids)
    assert all(type(i) is int for i in splits.train + splits.validacion)


def test_crear_splits_rechaza_id_fuera_de_int64() -> None:
    """Un ID fuera de int64 se rechaza con un mensaje claro, no con un reparto erróneo."""
    partidas = _quince()
    partidas[0] = replace(partidas[0], id_episodio=ID_EPISODIO_MAXIMO + 1)
    with pytest.raises(ValueError, match="enteros en el rango de int64"):
        crear_splits(partidas, 10, 5, (1,), crear_generador(0), HUELLA)


def test_crear_splits_rechaza_id_no_entero() -> None:
    """Un id_episodio que no es entero se rechaza aunque el contenedor lo permita."""
    partidas = _quince()
    partidas[3] = replace(partidas[3], id_episodio="3")
    with pytest.raises(ValueError, match="deben ser enteros"):
        crear_splits(partidas, 10, 5, (1,), crear_generador(0), HUELLA)


# --- validar_splits ----------------------------------------------------------


def _modificar(**cambios: Any) -> SplitsDemostraciones:
    """Splits válidos con algunos campos sustituidos."""
    return replace(_splits(0), **cambios)


def _sin(ids: tuple[int, ...], quitar: int, poner: int) -> tuple[int, ...]:
    """Sustituye un ID por otro dentro de una tupla."""
    return tuple(poner if i == quitar else i for i in ids)


@pytest.mark.parametrize(("caso", "mensaje"), [
    ("solape", "Partidas en train y validación a la vez"),
    ("id_desconocido", "faltan [14], sobran [99]"),
    ("dataset_con_una_menos", "faltan [], sobran [14]"),
    ("train_vacio", "train debe ser una tupla no vacía de enteros"),
    ("validacion_lista", "validacion debe ser una tupla no vacía de enteros"),
    ("id_real", "train debe ser una tupla no vacía de enteros"),
    ("train_repetido", "train repite partidas"),
    ("subconjunto_tamano", "El subconjunto 4 debe tener 4 partidas distintas"),
    ("subconjunto_fuera_de_train", "El subconjunto 1 contiene partidas fuera de train"),
    ("no_anidado", "El subconjunto 4 no contiene al anterior"),
    ("cantidad_cero", "Cantidad de subconjunto no válida: 0"),
    ("cantidad_texto", "Cantidad de subconjunto no válida: '4'"),
    ("id_real_en_subconjunto", "El subconjunto 1 debe tener 1 partidas distintas"),
    ("sin_subconjuntos", "subconjuntos debe ser un diccionario no vacío"),
    ("huella", "SHA-256 hexadecimal"),
    ("no_es_splits", "Se esperaba un objeto SplitsDemostraciones"),
])
def test_validar_splits_detecta_cada_incoherencia(caso: str, mensaje: str) -> None:
    """Cada regla de validar_splits salta con su mensaje."""
    base = _splits(0)
    train, val = list(base.train), list(base.validacion)
    ids_disponibles: Any = range(15)
    if caso == "id_desconocido":
        splits: Any = replace(base, train=_sin(base.train, 14, 99),
                              validacion=_sin(base.validacion, 14, 99))
    elif caso == "dataset_con_una_menos":
        splits, ids_disponibles = base, range(14)
    else:
        splits = {
            "solape": _modificar(validacion=tuple(val[:-1] + [train[0]])),
            "train_vacio": _modificar(train=()),
            "validacion_lista": _modificar(validacion=val),
            "id_real": _modificar(train=tuple([float(train[0])] + train[1:])),
            "train_repetido": _modificar(train=tuple(train[:-1] + [train[0]])),
            "subconjunto_tamano": _modificar(subconjuntos={1: (train[0],),
                                                           4: tuple(train[:3])}),
            "subconjunto_fuera_de_train": _modificar(subconjuntos={1: (val[0],)}),
            "no_anidado": _modificar(subconjuntos={1: (train[0],), 4: tuple(train[1:5])}),
            "cantidad_cero": _modificar(subconjuntos={0: ()}),
            "cantidad_texto": _modificar(subconjuntos={"4": tuple(train[:4])}),
            "id_real_en_subconjunto": _modificar(subconjuntos={1: (float(train[0]),)}),
            "sin_subconjuntos": _modificar(subconjuntos={}),
            "huella": _modificar(huella_dataset="x"),
            "no_es_splits": {"train": train},
        }[caso]
    with pytest.raises(ValueError, match=re.escape(mensaje)):
        validar_splits(splits, ids_disponibles)


def test_validar_splits_rechaza_ids_disponibles_repetidos() -> None:
    """La lista de partidas del dataset tampoco puede repetir IDs."""
    with pytest.raises(ValueError, match="enteros del rango de int64 sin repetir"):
        validar_splits(_splits(0), list(range(15)) + [3])


# --- guardar_splits y cargar_splits ------------------------------------------


@pytest.mark.parametrize("semilla", [0, 1, 2])
def test_guardar_y_cargar_recupera_los_mismos_splits(tmp_path: Path, semilla: int) -> None:
    """Ida y vuelta en una carpeta nueva: mismos IDs, mismo orden, cantidades enteras."""
    splits = _splits(semilla)
    ruta = guardar_splits(tmp_path / "nueva" / "splits.json", splits)
    assert ruta == tmp_path / "nueva" / "splits.json"
    cargados = cargar_splits(ruta, HUELLA, range(15))
    assert cargados == splits
    assert all(type(c) is int for c in cargados.subconjuntos)


def test_mismos_splits_producen_el_mismo_texto(tmp_path: Path) -> None:
    """Las claves se ordenan: el archivo es idéntico aunque cambie el orden del diccionario."""
    splits = _splits(0)
    invertidos = replace(splits, subconjuntos=dict(reversed(list(splits.subconjuntos.items()))))
    a = guardar_splits(tmp_path / "a.json", splits).read_text(encoding="utf-8")
    b = guardar_splits(tmp_path / "b.json", invertidos).read_text(encoding="utf-8")
    assert a == b
    assert list(json.loads(a)) == sorted(json.loads(a))


def test_guardar_no_sobrescribe(tmp_path: Path) -> None:
    """Un destino existente conserva su contenido."""
    ruta = tmp_path / "splits.json"
    ruta.write_text("previo", encoding="utf-8")
    with pytest.raises(FileExistsError):
        guardar_splits(ruta, _splits(0))
    assert ruta.read_text(encoding="utf-8") == "previo"


@pytest.mark.parametrize("nombre", ["splits", "splits.npz", "splits.JSON"])
def test_guardar_y_cargar_exigen_extension_json(tmp_path: Path, nombre: str) -> None:
    """La extensión se comprueba antes de escribir o leer."""
    with pytest.raises(ValueError, match="terminar en .json"):
        guardar_splits(tmp_path / nombre, _splits(0))
    with pytest.raises(ValueError, match="terminar en .json"):
        cargar_splits(tmp_path / nombre, HUELLA, range(15))


def test_guardar_rechaza_splits_incoherentes_sin_crear_nada(tmp_path: Path) -> None:
    """Unos splits con solape no se escriben."""
    malos = _modificar(validacion=(_splits(0).train[0],) + _splits(0).validacion[1:])
    ruta = tmp_path / "nueva" / "splits.json"
    with pytest.raises(ValueError, match="train y validación a la vez"):
        guardar_splits(ruta, malos)
    assert not ruta.parent.exists()


@pytest.mark.parametrize("splits", [None, {"train": (0,), "validacion": (1,)}, "splits"])
def test_guardar_rechaza_objetos_que_no_son_splits(tmp_path: Path, splits: Any) -> None:
    """Sin esta comprobación, guardar fallaría con AttributeError al leer splits.train."""
    with pytest.raises(ValueError, match="Se esperaba un objeto SplitsDemostraciones"):
        guardar_splits(tmp_path / "splits.json", splits)
    assert not (tmp_path / "splits.json").exists()


def test_guardar_elimina_archivo_incompleto(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Si la escritura falla a medias, no queda un JSON parcial."""
    abrir = Path.open

    def abrir_con_fallo(self: Path, *args: Any, **kwargs: Any) -> Any:
        """Abre de verdad, pero el archivo falla tras escribir parte."""
        archivo = abrir(self, *args, **kwargs)
        escribir = archivo.write

        def escribir_y_fallar(texto: str) -> int:
            """Escribe la mitad y simula un disco lleno."""
            escribir(texto[: len(texto) // 2])
            raise OSError("fallo simulado de escritura")

        archivo.write = escribir_y_fallar
        return archivo

    monkeypatch.setattr(Path, "open", abrir_con_fallo)
    ruta = tmp_path / "splits.json"
    with pytest.raises(OSError, match="fallo simulado"):
        guardar_splits(ruta, _splits(0))
    monkeypatch.undo()
    assert not ruta.exists()


def test_cargar_ruta_inexistente(tmp_path: Path) -> None:
    """Un archivo de splits que no existe se comunica como FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        cargar_splits(tmp_path / "no_existe.json", HUELLA, range(15))


def _json_valido() -> dict[str, Any]:
    """Contenido de un archivo de splits válido, como diccionario editable."""
    splits = _splits(0)
    return {"formato": "tiiac-splits", "version": 1, "huella_dataset": HUELLA,
            "train": list(splits.train), "validacion": list(splits.validacion),
            "subconjuntos": {str(c): list(ids) for c, ids in splits.subconjuntos.items()}}


def _con(clave: str, valor: Any) -> dict[str, Any]:
    """JSON válido con una clave sustituida."""
    return {**_json_valido(), clave: valor}


@pytest.mark.parametrize(("contenido", "mensaje"), [
    ("{no json", "no es JSON válido"),
    (b"\xff\xfe", "no es JSON válido"),
    ([], "debe contener un objeto JSON"),
    ({k: v for k, v in _json_valido().items() if k != "train"}, "Claves del archivo de splits"),
    ({**_json_valido(), "extra": 1}, "Claves del archivo de splits"),
    (_con("formato", "otro"), "Formato desconocido"),
    (_con("version", 2), "Versión no soportada: 2"),
    (_con("version", True), "Versión no soportada: True"),
    (_con("huella_dataset", OTRA_HUELLA), "se crearon para otro archivo de demostraciones"),
    (_con("huella_dataset", None), "se crearon para otro archivo de demostraciones"),
    (_con("train", "0,1"), "train debe ser una lista de enteros"),
    (_con("train", [0.0] + _json_valido()["train"][1:]), "train debe ser una lista de enteros"),
    (_con("validacion", [True] * 5), "validacion debe ser una lista de enteros"),
    (_con("train", [2**63] + _json_valido()["train"][1:]),
     "train debe ser una lista de enteros del rango de int64"),
    (_con("subconjuntos", []), "subconjuntos debe ser un objeto JSON"),
    (_con("subconjuntos", {"01": [0]}), "Cantidad de subconjunto no válida: '01'"),
    (_con("subconjuntos", {"-1": [0]}), "Cantidad de subconjunto no válida: '-1'"),
    (_con("subconjuntos", {"uno": [0]}), "Cantidad de subconjunto no válida: 'uno'"),
    (_con("subconjuntos", {"1": "0"}), "El subconjunto 1 debe ser una lista de enteros"),
    ('{"formato": NaN}', "valor no válido: NaN"),
])
def test_cargar_rechaza_archivos_invalidos(tmp_path: Path, contenido: Any, mensaje: str) -> None:
    """Cada defecto del archivo produce su mensaje propio."""
    ruta = tmp_path / "splits.json"
    if isinstance(contenido, bytes):
        ruta.write_bytes(contenido)
    elif isinstance(contenido, str):
        ruta.write_text(contenido, encoding="utf-8")
    else:
        ruta.write_text(json.dumps(contenido), encoding="utf-8")
    with pytest.raises(ValueError, match=re.escape(mensaje)):
        cargar_splits(ruta, HUELLA, range(15))


@pytest.mark.parametrize(("ids_disponibles", "mensaje"), [
    (range(14), "faltan [], sobran [14]"),
    (range(16), "faltan [15], sobran []"),
    ([i for i in range(15) if i != 6] + [20], "faltan [20], sobran [6]"),
], ids=["borrada_una_partida", "anadida_una_partida", "cambiada_una_partida"])
def test_cargar_rechaza_splits_de_otro_conjunto_de_partidas(
    tmp_path: Path, ids_disponibles: Any, mensaje: str,
) -> None:
    """Aunque la huella coincida, los splits deben repartir exactamente las partidas cargadas."""
    ruta = guardar_splits(tmp_path / "splits.json", _splits(0))
    with pytest.raises(ValueError, match=re.escape(mensaje)):
        cargar_splits(ruta, HUELLA, ids_disponibles)


def test_dataset_regenerado_cambia_la_huella_y_los_splits_se_rechazan(tmp_path: Path) -> None:
    """Integración: si se rehace el NPZ sin una partida, la huella cambia y la carga falla."""
    from gymnasium import spaces

    from src.data.demostraciones import guardar_demostraciones

    espacios = (spaces.Box(-np.inf, np.inf, (4,), np.float32), spaces.Discrete(2))
    metadatos = {"entorno": "prueba", "experto": "prueba", "procedencia": "prueba",
                 "configuracion_experto": {}}
    original = guardar_demostraciones(tmp_path / "v1.npz", _quince(), metadatos, *espacios)
    splits = crear_splits(_quince(), 10, 5, (1, 4, 7, 10), crear_generador(0),
                          huella_archivo(original))
    ruta = guardar_splits(tmp_path / "splits.json", splits)
    assert cargar_splits(ruta, huella_archivo(original), range(15)) == splits
    sin_una = guardar_demostraciones(tmp_path / "v2.npz", _quince()[:14], metadatos, *espacios)
    with pytest.raises(ValueError, match="otro archivo de demostraciones"):
        cargar_splits(ruta, huella_archivo(sin_una), range(14))


# --- submuestrear_pares ------------------------------------------------------


def test_frecuencia_uno_toma_todos_los_pasos() -> None:
    """Con frecuencia 1 el lote contiene todas las transiciones, en orden."""
    partidas = [_partida(3, 4), _partida(8, 2)]
    lote = submuestrear_pares(partidas, 1, crear_generador(0))
    np.testing.assert_array_equal(lote.ids_episodio, [3, 3, 3, 3, 8, 8])
    np.testing.assert_array_equal(lote.pasos, [0, 1, 2, 3, 0, 1])
    np.testing.assert_array_equal(lote.estados, np.concatenate([p.estados for p in partidas]))


def test_submuestreo_en_todo_el_dominio_pequeno() -> None:
    """Dominio completo: longitudes 1 a 25, frecuencias 1 a 12 y tres semillas.

    Para cada combinación, los pasos empiezan antes de min(frecuencia, T), avanzan
    de frecuencia en frecuencia hasta el final y cada par es el de su paso.
    """
    for pasos, frecuencia, semilla in itertools.product(range(1, 26), range(1, 13), range(3)):
        partida = _partida(4, pasos)
        lote = submuestrear_pares([partida], frecuencia, crear_generador(semilla))
        inicio = int(lote.pasos[0])
        assert 0 <= inicio < min(frecuencia, pasos)
        np.testing.assert_array_equal(lote.pasos, np.arange(inicio, pasos, frecuencia))
        np.testing.assert_array_equal(lote.estados, partida.estados[lote.pasos])
        np.testing.assert_array_equal(lote.acciones, partida.acciones[lote.pasos])
        assert set(lote.ids_episodio.tolist()) == {4}
        assert lote.numero_transiciones >= 1


@pytest.mark.parametrize("continua", [False, True])
def test_lote_conserva_formas_y_dtypes(continua: bool) -> None:
    """Acciones [N] o [N, d] y los mismos dtypes que las partidas."""
    partidas = [_partida(i, 30, continua) for i in range(3)]
    lote = submuestrear_pares(partidas, 10, crear_generador(1))
    assert lote.estados.shape == (lote.numero_transiciones, 4)
    assert lote.acciones.shape == ((lote.numero_transiciones, 2) if continua
                                   else (lote.numero_transiciones,))
    assert lote.estados.dtype == np.float32
    assert lote.acciones.dtype == partidas[0].acciones.dtype
    assert lote.ids_episodio.dtype == lote.pasos.dtype == np.int64


def test_cada_partida_tiene_su_propio_inicio_y_es_reproducible() -> None:
    """Los inicios varían entre partidas y la misma semilla repite el lote."""
    partidas = [_partida(i, 50) for i in range(15)]
    lote = submuestrear_pares(partidas, 10, crear_generador(7))
    inicios = {int(lote.pasos[lote.ids_episodio == i][0]) for i in range(15)}
    assert len(inicios) > 1
    otra = submuestrear_pares(partidas, 10, crear_generador(7))
    for campo in ("estados", "acciones", "ids_episodio", "pasos"):
        np.testing.assert_array_equal(getattr(lote, campo), getattr(otra, campo))


def test_submuestreo_no_modifica_las_partidas() -> None:
    """Las partidas de entrada quedan intactas."""
    partida = _partida(1, 20)
    copia = partida.estados.copy()
    submuestrear_pares([partida], 3, crear_generador(0))
    np.testing.assert_array_equal(partida.estados, copia)


def test_submuestreo_conserva_ids_extremos() -> None:
    """Regresión: un ID cerca de los límites de int64 se guarda exacto en el lote."""
    partidas = [replace(_partida(0, 4), id_episodio=ID_EPISODIO_MAXIMO),
                replace(_partida(1, 4), id_episodio=ID_EPISODIO_MINIMO)]
    lote = submuestrear_pares(partidas, 1, crear_generador(0))
    assert lote.ids_episodio.tolist() == [ID_EPISODIO_MAXIMO] * 4 + [ID_EPISODIO_MINIMO] * 4


@pytest.mark.parametrize("identificador", [ID_EPISODIO_MAXIMO + 1, ID_EPISODIO_MINIMO - 1])
def test_submuestreo_rechaza_ids_fuera_de_int64(identificador: int) -> None:
    """Regresión: antes fallaba con OverflowError en lugar del ValueError documentado."""
    partida = replace(_partida(0), id_episodio=identificador)
    with pytest.raises(ValueError, match="enteros en el rango de int64"):
        submuestrear_pares([partida], 10, crear_generador(0))


@pytest.mark.parametrize(("argumentos", "mensaje"), [
    ({"frecuencia": 0}, "frecuencia debe ser un entero positivo"),
    ({"frecuencia": -2}, "frecuencia debe ser un entero positivo"),
    ({"frecuencia": 1.5}, "frecuencia debe ser un entero positivo"),
    ({"frecuencia": True}, "frecuencia debe ser un entero positivo"),
    ({"generador": np.random.RandomState(0)}, "numpy.random.Generator"),
    ({"trayectorias": []}, "secuencia no vacía de objetos Trayectoria"),
    ({"trayectorias": [None]}, "secuencia no vacía de objetos Trayectoria"),
    ({"trayectorias": "partidas"}, "secuencia no vacía de objetos Trayectoria"),
])
def test_submuestreo_rechaza_entradas_invalidas(argumentos: dict[str, Any], mensaje: str) -> None:
    """Cada parámetro inválido produce su mensaje propio."""
    llamada = {"trayectorias": [_partida(0)], "frecuencia": 10,
               "generador": crear_generador(0), **argumentos}
    with pytest.raises(ValueError, match=re.escape(mensaje)):
        submuestrear_pares(**llamada)
