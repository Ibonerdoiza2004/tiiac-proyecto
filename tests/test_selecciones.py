"""Pruebas de las selecciones del dashboard: tramos, uniones, traslado al archivo
combinado, reparto de validación, pares y guardado."""

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from gymnasium import spaces

from src.data.demostraciones import (cargar_demostraciones, combinar_demostraciones,
                                     guardar_demostraciones, metadatos_humanos)
from src.data.loaders import DatasetTransiciones
from src.data.selecciones import (
    SeleccionPares, cargar_seleccion, crear_seleccion, guardar_seleccion, normalizar_tramos,
    pares_de_seleccion, repartir_validacion, trasladar_seleccion, unir_selecciones,
    validar_seleccion,
)
from src.data.splits import huella_archivo
from src.data.tipos import ID_EPISODIO_MAXIMO, ID_EPISODIO_MINIMO, Trayectoria
from src.utils.configuracion import TipoEntorno, resolver_configuracion
from src.utils.seed import crear_generador

HUELLA = "a" * 64
OTRA_HUELLA = "b" * 64
OBSERVACION = spaces.Box(-np.inf, np.inf, (4,), np.float32)
DISCRETA = spaces.Discrete(2)
CONTINUA = spaces.Box(-100.0, 100.0, (2,), np.float64)
CFG = {"demostraciones": {"fraccion_validacion": 0.3}}


def _partida(identificador: int, pasos: int = 5, continua: bool = False) -> Trayectoria:
    """Partida exacta: el estado del paso t vale 100 * id + t en cada columna."""
    base = 100 * (identificador % 1000) + np.arange(pasos + 1, dtype=np.float32)
    observaciones = np.repeat(base[:, None], 4, axis=1)
    terminados = np.zeros(pasos, dtype=bool)
    terminados[-1] = True
    acciones = (np.stack([np.arange(pasos), -np.arange(pasos)], axis=1).astype(np.float64)
                if continua else np.arange(pasos, dtype=np.int64) % 2)
    return Trayectoria(identificador, observaciones[:-1], acciones, observaciones[1:],
                       np.ones(pasos, dtype=np.float32), terminados,
                       np.zeros(pasos, dtype=bool))


def _partidas(n: int = 6, pasos: int = 5) -> list[Trayectoria]:
    return [_partida(i, pasos) for i in range(n)]


def _pasos(tramos: tuple[tuple[int, int], ...]) -> set[int]:
    return {paso for inicio, fin in tramos for paso in range(inicio, fin)}


# --- normalizar_tramos ---------------------------------------------------------


def _referencia(pasos: set[int]) -> tuple[tuple[int, int], ...]:
    """Tramos máximos de pasos consecutivos, calculados de forma independiente."""
    tramos: list[tuple[int, int]] = []
    for paso in sorted(pasos):
        if tramos and tramos[-1][1] == paso:
            tramos[-1] = (tramos[-1][0], paso + 1)
        else:
            tramos.append((paso, paso + 1))
    return tuple(tramos)


def test_normalizar_tramos_en_todo_el_dominio_pequeno() -> None:
    # Todas las listas de 1 a 3 tramos válidos de una partida de 4 pasos (dominio
    # finito): el resultado cubre exactamente los mismos pasos, en tramos máximos.
    pasos = 4
    validos = [(i, f) for i in range(pasos) for f in range(i + 1, pasos + 1)]
    casos = 0
    for cantidad in (1, 2, 3):
        for tramos in itertools.product(validos, repeat=cantidad):
            resultado = normalizar_tramos(list(tramos), pasos)
            assert resultado == _referencia(_pasos(tramos))
            assert normalizar_tramos(resultado, pasos) == resultado
            casos += 1
    assert casos == 10 + 10 ** 2 + 10 ** 3


@pytest.mark.parametrize(("tramos", "esperado"), [
    ([(0, 2), (2, 4)], ((0, 4),)),
    ([(3, 5), (0, 1)], ((0, 1), (3, 5))),
    ([(1, 3), (0, 5)], ((0, 5),)),
    ([[0, 1], [1, 2]], ((0, 2),)),
    ([(np.int64(1), np.int32(2))], ((1, 2),)),
])
def test_normalizar_tramos_ordena_y_funde(tramos: Any, esperado: Any) -> None:
    resultado = normalizar_tramos(tramos, 5)
    assert resultado == esperado
    assert all(type(x) is int for tramo in resultado for x in tramo)


@pytest.mark.parametrize(("tramos", "mensaje"), [
    ([], "al menos un tramo"),
    ((), "al menos un tramo"),
    (None, "al menos un tramo"),
    ("01", "al menos un tramo"),
    ([(0,)], "par de enteros"),
    ([(0, 1, 2)], "par de enteros"),
    ([0, 1], "par de enteros"),
    (["01"], "par de enteros"),
    ([(0.0, 1)], "par de enteros"),
    ([(True, 2)], "par de enteros"),
    ([("0", 1)], "par de enteros"),
    ([(1, 1)], "no válido"),
    ([(2, 1)], "no válido"),
    ([(-1, 1)], "no válido"),
    ([(0, 6)], "no válido"),
    ([(0, 2), (5, 6)], "no válido"),
], ids=["lista_vacia", "tupla_vacia", "none", "texto", "un_extremo", "tres_extremos",
        "sin_pares", "tramo_texto", "real", "bool", "texto_en_tramo", "vacio", "invertido",
        "negativo", "pasa_del_final", "uno_de_varios_fuera"])
def test_normalizar_tramos_rechaza_tramos_no_validos(tramos: Any, mensaje: str) -> None:
    with pytest.raises(ValueError, match=mensaje):
        normalizar_tramos(tramos, 5)


@pytest.mark.parametrize("pasos", [0, -1, True, 2.0, None])
def test_normalizar_tramos_rechaza_partidas_sin_pasos(pasos: Any) -> None:
    with pytest.raises(ValueError, match="numero_pasos"):
        normalizar_tramos([(0, 1)], pasos)


# --- crear_seleccion -------------------------------------------------------------


def test_crear_seleccion_de_partidas_enteras_y_tramos() -> None:
    seleccion = crear_seleccion(_partidas(), HUELLA, {4: None, 1: [(3, 5), (0, 2), (2, 3)],
                                                      np.int64(2): [(1, 2)]})
    assert seleccion.huella_dataset == HUELLA
    assert list(seleccion.partidas) == [1, 2, 4]
    assert seleccion.partidas == {1: ((0, 5),), 2: ((1, 2),), 4: ((0, 5),)}
    assert seleccion.numero_partidas == 3
    assert seleccion.numero_pares == 11


def test_crear_seleccion_con_ids_extremos() -> None:
    partidas = [_partida(ID_EPISODIO_MINIMO), _partida(ID_EPISODIO_MAXIMO)]
    seleccion = crear_seleccion(partidas, HUELLA, {ID_EPISODIO_MAXIMO: [(0, 1)],
                                                   ID_EPISODIO_MINIMO: None})
    assert list(seleccion.partidas) == [ID_EPISODIO_MINIMO, ID_EPISODIO_MAXIMO]


@pytest.mark.parametrize(("argumentos", "mensaje"), [
    ({"elecciones": {}}, "al menos una partida"),
    ({"elecciones": None}, "al menos una partida"),
    ({"elecciones": [(0, None)]}, "al menos una partida"),
    ({"elecciones": {9: None}}, r"no están en el archivo: \[9\]"),
    ({"elecciones": {True: None}}, "id_episodio"),
    ({"elecciones": {"0": None}}, "id_episodio"),
    ({"elecciones": {0: []}}, "al menos un tramo"),
    ({"elecciones": {0: [(0, 6)]}}, "no válido"),
    ({"huella_dataset": "XYZ"}, "SHA-256"),
    ({"trayectorias": "partidas"}, "objetos Trayectoria"),
    ({"trayectorias": [_partida(0), _partida(0)]}, "repetidos"),
    ({"trayectorias": [_partida(0), 3]}, "objetos Trayectoria"),
])
def test_crear_seleccion_rechaza_entradas_no_validas(argumentos: dict[str, Any],
                                                     mensaje: str) -> None:
    datos = {"trayectorias": _partidas(), "huella_dataset": HUELLA, "elecciones": {0: None},
             **argumentos}
    with pytest.raises(ValueError, match=mensaje):
        crear_seleccion(**datos)


# --- unir_selecciones --------------------------------------------------------------


def test_unir_cuenta_una_vez_los_pares_repetidos() -> None:
    partidas = _partidas()
    a = crear_seleccion(partidas, HUELLA, {0: [(0, 3)], 1: None})
    b = crear_seleccion(partidas, HUELLA, {0: [(2, 5)], 3: [(1, 2)]})
    union = unir_selecciones([a, b, a])
    assert union.partidas == {0: ((0, 5),), 1: ((0, 5),), 3: ((1, 2),)}
    assert union.numero_pares == 11 < a.numero_pares + b.numero_pares


def test_unir_equivale_a_la_union_de_pasos_en_todo_el_dominio_pequeno() -> None:
    # Dos selecciones de una partida de 3 pasos con todos sus conjuntos de tramos
    # válidos de uno o dos tramos: los pares de la unión son la unión de los pares.
    partidas = [_partida(0, pasos=3)]
    validos = [(i, f) for i in range(3) for f in range(i + 1, 4)]
    opciones = [[t] for t in validos] + [list(p) for p in itertools.combinations(validos, 2)]
    for tramos_a, tramos_b in itertools.product(opciones, repeat=2):
        a = crear_seleccion(partidas, HUELLA, {0: tramos_a})
        b = crear_seleccion(partidas, HUELLA, {0: tramos_b})
        union = unir_selecciones([a, b])
        assert _pasos(union.partidas[0]) == _pasos(a.partidas[0]) | _pasos(b.partidas[0])
        assert union.numero_pares == len(_pasos(union.partidas[0]))


@pytest.mark.parametrize(("selecciones", "mensaje"), [
    ([], "no vacía"),
    (None, "no vacía"),
    ("ab", "no vacía"),
    ([crear_seleccion(_partidas(), HUELLA, {0: None}), "otra"], "SeleccionPares"),
    ([crear_seleccion(_partidas(), HUELLA, {0: None}),
      crear_seleccion(_partidas(), OTRA_HUELLA, {0: None})], "mismo archivo"),
    ([SeleccionPares(HUELLA, {0: ((0, 2), (1, 3))})], "no están normalizados"),
    ([SeleccionPares(HUELLA, {})], "al menos una partida"),
])
def test_unir_rechaza_combinaciones_vacias_o_de_otros_archivos(selecciones: Any,
                                                               mensaje: str) -> None:
    with pytest.raises(ValueError, match=mensaje):
        unir_selecciones(selecciones)


# --- validar_seleccion ---------------------------------------------------------------


@pytest.mark.parametrize(("seleccion", "mensaje"), [
    ("no", "SeleccionPares"),
    (SeleccionPares("XYZ", {0: ((0, 1),)}), "SHA-256"),
    (SeleccionPares(HUELLA, {}), "al menos una partida"),
    (SeleccionPares(HUELLA, {True: ((0, 1),)}), "id_episodio"),
    (SeleccionPares(HUELLA, {0: [(0, 1)]}), "tupla de tramos"),
    (SeleccionPares(HUELLA, {0: ()}), "tupla de tramos"),
    (SeleccionPares(HUELLA, {0: ((2, 3), (0, 1))}), "no están normalizados"),
    (SeleccionPares(HUELLA, {0: ((0, 1), (1, 2))}), "no están normalizados"),
    (SeleccionPares(HUELLA, {0: ((1, 1),)}), "no válido"),
    (SeleccionPares(HUELLA, {9: ((0, 1),)}), "no está en el archivo"),
    (SeleccionPares(HUELLA, {0: ((0, 6),)}), "pasan de sus 5 pasos"),
], ids=["tipo", "huella", "vacia", "id_bool", "lista", "sin_tramos", "desordenados",
        "contiguos", "tramo_vacio", "id_ajeno", "pasa_del_final"])
def test_validar_seleccion_detecta_cada_incoherencia(seleccion: Any, mensaje: str) -> None:
    with pytest.raises(ValueError, match=mensaje):
        validar_seleccion(seleccion, _partidas())


# --- pares_de_seleccion ------------------------------------------------------------


@pytest.mark.parametrize("continua", [False, True])
def test_pares_de_seleccion_son_exactamente_los_pasos_elegidos(continua: bool) -> None:
    partidas = [_partida(i, 6, continua) for i in (3, 8)]
    # Partidas en orden inverso: los pares salen igualmente por id y por paso.
    seleccion = SeleccionPares(HUELLA, {8: ((1, 3),), 3: ((0, 1), (4, 6))})
    lote = pares_de_seleccion(partidas, seleccion)
    assert lote.ids_episodio.tolist() == [3, 3, 3, 8, 8]
    assert lote.pasos.tolist() == [0, 4, 5, 1, 2]
    por_id = {t.id_episodio: t for t in partidas}
    for k, (identificador, paso) in enumerate(zip(lote.ids_episodio, lote.pasos)):
        np.testing.assert_array_equal(lote.estados[k], por_id[identificador].estados[paso])
        np.testing.assert_array_equal(lote.acciones[k], por_id[identificador].acciones[paso])
    assert lote.estados.dtype == np.float32
    assert lote.acciones.dtype == partidas[0].acciones.dtype
    assert lote.ids_episodio.dtype == lote.pasos.dtype == np.int64
    assert lote.numero_transiciones == seleccion.numero_pares


@pytest.mark.parametrize("continua", [False, True])
def test_pares_de_seleccion_sirven_para_bc(continua: bool) -> None:
    partidas = [_partida(i, 6, continua) for i in range(3)]
    seleccion = crear_seleccion(partidas, HUELLA, {0: None, 2: [(1, 4)]})
    dataset = DatasetTransiciones(pares_de_seleccion(partidas, seleccion),
                                  CONTINUA if continua else DISCRETA)
    assert len(dataset) == 9
    assert dataset.ids_episodio.tolist() == [0] * 6 + [2] * 3


def test_pares_de_seleccion_rechaza_selecciones_de_otras_partidas() -> None:
    seleccion = crear_seleccion(_partidas(pasos=8), HUELLA, {0: [(6, 8)]})
    with pytest.raises(ValueError, match="pasan de sus 5 pasos"):
        pares_de_seleccion(_partidas(pasos=5), seleccion)


# --- repartir_validacion ------------------------------------------------------------


def _cantidad_esperada(partidas: int, fraccion: float) -> int:
    return min(partidas - 1, max(1, int(np.floor(fraccion * partidas + 0.5))))


@pytest.mark.parametrize("partidas", range(2, 16))
@pytest.mark.parametrize("fraccion", [0.01, 0.3, 0.5, 0.99])
def test_repartir_validacion_separa_partidas_enteras(partidas: int, fraccion: float) -> None:
    trayectorias = _partidas(partidas)
    seleccion = crear_seleccion(trayectorias, HUELLA, {i: [(i % 5, 5)] for i in range(partidas)})
    train, validacion = repartir_validacion(
        seleccion, {"demostraciones": {"fraccion_validacion": fraccion}}, crear_generador(1))
    assert validacion.numero_partidas == _cantidad_esperada(partidas, fraccion)
    assert train.numero_partidas == partidas - validacion.numero_partidas
    assert not set(train.partidas) & set(validacion.partidas)
    assert set(train.partidas) | set(validacion.partidas) == set(seleccion.partidas)
    for parte in (train, validacion):
        assert parte.huella_dataset == HUELLA
        assert list(parte.partidas) == sorted(parte.partidas)
        assert all(parte.partidas[i] == seleccion.partidas[i] for i in parte.partidas)


@pytest.mark.parametrize(("partidas", "validacion"), [(2, 1), (3, 1), (5, 2), (10, 3),
                                                      (15, 5), (20, 6)])
def test_repartir_validacion_con_la_fraccion_de_la_config(partidas: int, validacion: int) -> None:
    # Con 15 partidas salen las 5 de validación de la E1.
    seleccion = crear_seleccion(_partidas(partidas), HUELLA, dict.fromkeys(range(partidas)))
    _, val = repartir_validacion(seleccion, CFG, crear_generador(0))
    assert val.numero_partidas == validacion


def test_repartir_validacion_es_reproducible_e_independiente_del_orden() -> None:
    partidas = _partidas(12)
    seleccion = crear_seleccion(partidas, HUELLA, dict.fromkeys(range(12)))
    al_reves = SeleccionPares(HUELLA, dict(reversed(list(seleccion.partidas.items()))))
    uno = repartir_validacion(seleccion, CFG, crear_generador(5))
    dos = repartir_validacion(al_reves, CFG, crear_generador(5))
    assert uno == dos
    distintos = {tuple(repartir_validacion(seleccion, CFG, crear_generador(s))[1].partidas)
                 for s in range(10)}
    assert len(distintos) > 1


@pytest.mark.parametrize(("cfg", "mensaje"), [
    ({"demostraciones": {"fraccion_validacion": valor}}, "entre 0 y 1")
    for valor in (0, 1, 0.0, 1.0, -0.1, 1.5, float("nan"), float("inf"), True, "0.3", None)
] + [
    ({"demostraciones": {}}, "fraccion_validacion"),
    ({}, "fraccion_validacion"),
    (None, "fraccion_validacion"),
])
def test_repartir_validacion_rechaza_fracciones_no_validas(cfg: Any, mensaje: str) -> None:
    seleccion = crear_seleccion(_partidas(), HUELLA, dict.fromkeys(range(6)))
    with pytest.raises(ValueError, match=mensaje):
        repartir_validacion(seleccion, cfg, crear_generador(0))


def test_repartir_validacion_necesita_dos_partidas_y_un_generador() -> None:
    una = crear_seleccion(_partidas(), HUELLA, {0: None})
    with pytest.raises(ValueError, match="al menos dos partidas"):
        repartir_validacion(una, CFG, crear_generador(0))
    dos = crear_seleccion(_partidas(), HUELLA, {0: None, 1: None})
    with pytest.raises(ValueError, match="generador"):
        repartir_validacion(dos, CFG, np.random.default_rng(0).integers)
    with pytest.raises(ValueError, match="SeleccionPares"):
        repartir_validacion({0: ((0, 1),)}, CFG, crear_generador(0))


@pytest.mark.parametrize("entorno", list(TipoEntorno))
def test_todos_los_perfiles_tienen_fraccion_de_validacion(entorno: TipoEntorno) -> None:
    cfg = resolver_configuracion(Path("configs/default.yaml"), entorno.value)
    seleccion = crear_seleccion(_partidas(), HUELLA, dict.fromkeys(range(6)))
    _, validacion = repartir_validacion(seleccion, cfg, crear_generador(0))
    assert cfg["demostraciones"]["fraccion_validacion"] == 0.3
    assert validacion.numero_partidas == 2


# --- trasladar_seleccion (con combinar_demostraciones) --------------------------------


def _metadatos_experto() -> dict[str, object]:
    return {"entorno": "CartPole-v1", "experto": "prueba", "procedencia": "prueba",
            "configuracion_experto": {}}


def _guardar(ruta: Path, partidas: list[Trayectoria], metadatos: dict[str, object]) -> Path:
    return guardar_demostraciones(ruta, partidas, metadatos, OBSERVACION, DISCRETA)


@pytest.fixture
def combinado(tmp_path: Path) -> dict[str, Any]:
    """Dos archivos (experto e humano) y su combinación, con una partida repetida."""
    experto = _guardar(tmp_path / "experto.npz", [_partida(i, 6) for i in (0, 1, 2)],
                       _metadatos_experto())
    humano = _guardar(tmp_path / "humano.npz", [_partida(i, 4) for i in (10, 11)],
                      metadatos_humanos("CartPole-v1", {10: 1, 11: 2}))
    resultado = combinar_demostraciones([experto, humano, experto], tmp_path / "c.npz",
                                        OBSERVACION, DISCRETA)
    partidas, metadatos = cargar_demostraciones(resultado["demostraciones"], OBSERVACION,
                                                DISCRETA)
    return {"experto": experto, "humano": humano, "partidas": partidas,
            "metadatos": metadatos, "huella": resultado["huella_dataset"]}


def _original(ruta: Path) -> list[Trayectoria]:
    return cargar_demostraciones(ruta, OBSERVACION, DISCRETA)[0]


def test_trasladar_da_los_mismos_pares_en_el_combinado(combinado: dict[str, Any]) -> None:
    partidas_humano = _original(combinado["humano"])
    seleccion = crear_seleccion(partidas_humano, huella_archivo(combinado["humano"]),
                                {11: [(1, 3)], 10: None})
    trasladada = trasladar_seleccion(seleccion, combinado["metadatos"], combinado["huella"])
    assert trasladada.huella_dataset == combinado["huella"]
    assert trasladada.partidas == {3: ((0, 4),), 4: ((1, 3),)}
    antes = pares_de_seleccion(partidas_humano, seleccion)
    despues = pares_de_seleccion(combinado["partidas"], trasladada)
    np.testing.assert_array_equal(antes.estados, despues.estados)
    np.testing.assert_array_equal(antes.acciones, despues.acciones)
    np.testing.assert_array_equal(antes.pasos, despues.pasos)


def test_selecciones_de_varios_archivos_se_unen_sin_repetir_pares(
    combinado: dict[str, Any],
) -> None:
    huella_experto = huella_archivo(combinado["experto"])
    a = crear_seleccion(_original(combinado["experto"]), huella_experto, {0: [(0, 4)], 2: None})
    b = crear_seleccion(_original(combinado["experto"]), huella_experto, {0: [(2, 6)]})
    c = crear_seleccion(_original(combinado["humano"]), huella_archivo(combinado["humano"]),
                        {10: [(0, 2)]})
    union = unir_selecciones([trasladar_seleccion(s, combinado["metadatos"], combinado["huella"])
                              for s in (a, b, c)])
    assert union.partidas == {0: ((0, 6),), 2: ((0, 6),), 3: ((0, 2),)}
    assert union.numero_pares == 14
    validar_seleccion(union, combinado["partidas"])


@pytest.mark.parametrize(("cambio", "mensaje"), [
    (lambda m: m.pop("origenes"), "no son de un archivo combinado"),
    (lambda m: m["origenes"].update(archivos="x"), "no son de un archivo combinado"),
    (lambda m: m["origenes"]["archivos"].pop(1), "no incluye el archivo"),
    (lambda m: m["origenes"]["archivos"][1].update(ids=[[10, 3]]), r"no incluyen las partidas"),
    (lambda m: m["origenes"]["archivos"][1].update(ids=[[10, 3, 1]]), "mal formados"),
    (lambda m: m["origenes"]["archivos"][1].update(ids=[[10, True]]), "mal formados"),
    (lambda m: m["origenes"]["archivos"][1].update(ids="10->3"), "mal formados"),
], ids=["sin_origenes", "archivos_texto", "sin_el_archivo", "falta_un_id", "par_de_tres",
        "id_bool", "ids_texto"])
def test_trasladar_rechaza_origenes_que_no_cubren_la_seleccion(
    combinado: dict[str, Any], cambio: Any, mensaje: str,
) -> None:
    metadatos = json.loads(json.dumps(combinado["metadatos"]))
    cambio(metadatos)
    seleccion = crear_seleccion(_original(combinado["humano"]),
                                huella_archivo(combinado["humano"]), {10: None, 11: None})
    with pytest.raises(ValueError, match=mensaje):
        trasladar_seleccion(seleccion, metadatos, combinado["huella"])


def test_trasladar_rechaza_entradas_no_validas(combinado: dict[str, Any]) -> None:
    seleccion = crear_seleccion(_original(combinado["humano"]),
                                huella_archivo(combinado["humano"]), {10: None})
    with pytest.raises(ValueError, match="SHA-256"):
        trasladar_seleccion(seleccion, combinado["metadatos"], "no")
    with pytest.raises(ValueError, match="no son de un archivo combinado"):
        trasladar_seleccion(seleccion, None, combinado["huella"])
    with pytest.raises(ValueError, match="SeleccionPares"):
        trasladar_seleccion({10: None}, combinado["metadatos"], combinado["huella"])


# --- guardar_seleccion y cargar_seleccion ---------------------------------------------


def _seleccion() -> SeleccionPares:
    return crear_seleccion(_partidas(), HUELLA, {3: [(0, 2), (3, 5)], 0: None})


def test_guardar_y_cargar_recupera_la_misma_seleccion(tmp_path: Path) -> None:
    ruta = guardar_seleccion(tmp_path / "sub" / "seleccion.json", _seleccion())
    cargada = cargar_seleccion(ruta, HUELLA, _partidas())
    assert cargada == _seleccion()
    assert json.loads(ruta.read_text(encoding="utf-8")) == {
        "formato": "tiiac-seleccion", "version": 1, "huella_dataset": HUELLA,
        "partidas": [{"id_episodio": 0, "tramos": [[0, 5]]},
                     {"id_episodio": 3, "tramos": [[0, 2], [3, 5]]}]}


def test_mismas_selecciones_dan_el_mismo_texto(tmp_path: Path) -> None:
    al_reves = SeleccionPares(HUELLA, dict(reversed(list(_seleccion().partidas.items()))))
    uno = guardar_seleccion(tmp_path / "uno.json", _seleccion())
    dos = guardar_seleccion(tmp_path / "dos.json", al_reves)
    assert uno.read_text(encoding="utf-8") == dos.read_text(encoding="utf-8")


def test_guardar_y_cargar_ids_extremos(tmp_path: Path) -> None:
    partidas = [_partida(ID_EPISODIO_MINIMO), _partida(ID_EPISODIO_MAXIMO)]
    seleccion = crear_seleccion(partidas, HUELLA, dict.fromkeys(
        (ID_EPISODIO_MINIMO, ID_EPISODIO_MAXIMO)))
    ruta = guardar_seleccion(tmp_path / "s.json", seleccion)
    assert cargar_seleccion(ruta, HUELLA, partidas) == seleccion


def test_guardar_no_sobrescribe(tmp_path: Path) -> None:
    ruta = tmp_path / "s.json"
    ruta.write_text("previo", encoding="utf-8")
    with pytest.raises(FileExistsError):
        guardar_seleccion(ruta, _seleccion())
    assert ruta.read_text(encoding="utf-8") == "previo"


@pytest.mark.parametrize("nombre", ["s.txt", "s", "s.json.tmp"])
def test_guardar_y_cargar_exigen_extension_json(tmp_path: Path, nombre: str) -> None:
    with pytest.raises(ValueError, match="terminar en .json"):
        guardar_seleccion(tmp_path / nombre, _seleccion())
    with pytest.raises(ValueError, match="terminar en .json"):
        cargar_seleccion(tmp_path / nombre, HUELLA, _partidas())


@pytest.mark.parametrize("seleccion", [
    SeleccionPares(HUELLA, {}), SeleccionPares(HUELLA, {0: ((1, 0),)}),
    SeleccionPares("x", {0: ((0, 1),)}), {0: ((0, 1),)},
])
def test_guardar_rechaza_selecciones_mal_formadas_sin_crear_nada(tmp_path: Path,
                                                                 seleccion: Any) -> None:
    with pytest.raises(ValueError):
        guardar_seleccion(tmp_path / "s.json", seleccion)
    assert not (tmp_path / "s.json").exists()


def test_guardar_elimina_archivo_incompleto(tmp_path: Path,
                                            monkeypatch: pytest.MonkeyPatch) -> None:
    abrir = Path.open

    def abrir_con_fallo(self: Path, *args: Any, **kwargs: Any) -> Any:
        archivo = abrir(self, *args, **kwargs)
        escribir = archivo.write

        def escribir_y_fallar(texto: str) -> int:
            escribir(texto[: len(texto) // 2])
            raise OSError("fallo simulado de escritura")

        archivo.write = escribir_y_fallar
        return archivo

    monkeypatch.setattr(Path, "open", abrir_con_fallo)
    ruta = tmp_path / "s.json"
    with pytest.raises(OSError, match="fallo simulado"):
        guardar_seleccion(ruta, _seleccion())
    monkeypatch.undo()
    assert not ruta.exists()


def test_cargar_ruta_inexistente(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        cargar_seleccion(tmp_path / "no.json", HUELLA, _partidas())


def _json_valido() -> dict[str, Any]:
    return {"formato": "tiiac-seleccion", "version": 1, "huella_dataset": HUELLA,
            "partidas": [{"id_episodio": 0, "tramos": [[0, 2]]},
                         {"id_episodio": 3, "tramos": [[1, 5]]}]}


def _con(**cambios: Any) -> dict[str, Any]:
    return {**_json_valido(), **cambios}


def _partida_json(identificador: Any, tramos: Any) -> dict[str, Any]:
    return _con(partidas=[{"id_episodio": identificador, "tramos": tramos}])


@pytest.mark.parametrize(("contenido", "mensaje"), [
    ("{no es json", "no es JSON válido"),
    ('{"formato": NaN}', "valor no válido"),
    ([], "claves esperadas"),
    ({k: v for k, v in _json_valido().items() if k != "partidas"}, "claves esperadas"),
    (_con(extra=1), "claves esperadas"),
    (_con(formato="tiiac-splits"), "Formato desconocido"),
    (_con(version=2), "Versión no soportada"),
    (_con(version=True), "Versión no soportada"),
    (_con(huella_dataset=OTRA_HUELLA), "otro archivo de demostraciones"),
    (_con(partidas={}), "mal formadas"),
    (_con(partidas=[{"id_episodio": 0}]), "mal formadas"),
    (_partida_json(True, [[0, 1]]), "mal formadas"),
    (_partida_json(0, "0-1"), "mal formadas"),
    (_con(partidas=[{"id_episodio": 0, "tramos": [[0, 1]]},
                    {"id_episodio": 0, "tramos": [[2, 3]]}]), "repite la partida 0"),
    (_partida_json(0, [[0, 1, 2]]), "Tramos mal formados"),
    (_partida_json(0, [[0.0, 1]]), "Tramos mal formados"),
    (_partida_json(0, [[0, 2], [1, 3]]), "no están normalizados"),
    (_partida_json(0, [[2, 2]]), "no válido"),
    (_partida_json(0, [[0, 9]]), "pasan de sus 5 pasos"),
    (_partida_json(9, [[0, 1]]), "no está en el archivo"),
    (_con(partidas=[]), "al menos una partida"),
], ids=["json_roto", "nan", "lista", "sin_partidas", "clave_extra", "formato", "version_2",
        "version_bool", "otra_huella", "partidas_dict", "sin_tramos", "id_bool", "tramos_texto",
        "id_repetido", "tramo_de_tres", "tramo_real", "solapados", "tramo_vacio",
        "pasa_del_final", "id_ajeno", "vacia"])
def test_cargar_rechaza_archivos_no_validos(tmp_path: Path, contenido: Any,
                                            mensaje: str) -> None:
    ruta = tmp_path / "s.json"
    ruta.write_text(contenido if isinstance(contenido, str) else json.dumps(contenido),
                    encoding="utf-8")
    with pytest.raises(ValueError, match=mensaje):
        cargar_seleccion(ruta, HUELLA, _partidas())


def test_cargar_rechaza_una_huella_mal_formada(tmp_path: Path) -> None:
    ruta = guardar_seleccion(tmp_path / "s.json", _seleccion())
    with pytest.raises(ValueError, match="SHA-256"):
        cargar_seleccion(ruta, "x", _partidas())
