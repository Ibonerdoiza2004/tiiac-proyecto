"""Tests de las selecciones de demostraciones (src/data/seleccion.py): subconjuntos."""

from __future__ import annotations

import dataclasses
import itertools
import json
import math
import re
import shutil
from pathlib import Path

import numpy as np
import pytest
from gymnasium import spaces

from src.app.contrato import DemostracionApi
from src.app.rutas_biblioteca import como_demostracion
from src.data.biblioteca import (ARCHIVO_SUBCONJUNTO, Biblioteca, agregar, agregar_nuevo,
                                 crear_ficha, nuevo_id)
from src.data.demostraciones import (cargar_demostraciones, fuente_demostraciones,
                                     guardar_demostraciones, resumen_demostraciones,
                                     retorno_partida, retornos_por_partida)
from src.data.seleccion import (FORMATO_SUBCONJUNTO, VERSION_SUBCONJUNTO, cargar_subconjunto,
                                clave_seleccion, combinar_demostraciones,
                                comprobar_subconjunto, crear_subconjunto,
                                identidades_originales, preparar_datos,
                                proporcion_validacion, repartir_combinado, resumen_seleccion,
                                retornos_train)
from src.data.splits import (cargar_splits, crear_splits, guardar_splits, huella_archivo,
                             validar_splits)
from src.data.tipos import Trayectoria
from src.utils.configuracion import resolver_configuracion
from src.utils.seed import crear_generador, semillas_independientes

OBSERVACION = spaces.Box(-np.inf, np.inf, (4,), np.float32)
DISCRETA = spaces.Discrete(3)
PASOS = 5


def _partida(pasos: int = PASOS, identificador: int = 7) -> Trayectoria:
    """Partida de prueba con valores distintos en cada paso, terminada en el último."""
    estados = np.arange((pasos + 1) * 4, dtype=np.float32).reshape(pasos + 1, 4)
    terminados = np.zeros(pasos, dtype=bool)
    terminados[-1] = True
    return Trayectoria(identificador, estados[:-1], np.arange(pasos, dtype=np.int64) % 3,
                       estados[1:], np.arange(pasos, dtype=np.float32) + 0.5, terminados,
                       np.zeros(pasos, dtype=bool))


# --- subconjuntos ----------------------------------------------------------------------

ENTORNOS = {"cartpole": "CartPole-v1", "acrobot": "Acrobot-v1"}
# Partidas del conjunto de prueba: id → pasos.
PASOS_CONJUNTO = {0: 4, 3: 6, 5: 5}
ELEGIDAS = [5, 0]


def _metadatos_experto(gym: str = "CartPole-v1") -> dict:
    return {"entorno": gym, "experto": "x", "procedencia": {}, "configuracion_experto": {}}


def _guardar_conjunto(ruta: Path, pasos: dict[int, int] = PASOS_CONJUNTO,
                      gym: str = "CartPole-v1") -> None:
    guardar_demostraciones(ruta, [_partida(pasos=p, identificador=i) for i, p in pasos.items()],
                           _metadatos_experto(gym), OBSERVACION, DISCRETA)


def _anadir_conjunto(biblioteca: Biblioteca, entorno: str = "cartpole", **extra) -> str:
    return agregar_nuevo(biblioteca.raiz, entorno, "demostraciones", "Conjunto", "dashboard",
                         lambda c: _guardar_conjunto(c / "demostraciones.npz",
                                                     gym=ENTORNOS[entorno]), **extra)["id"]


def _anadir_a_mano(biblioteca: Biblioteca, datos: object, entorno: str = "cartpole",
                   **extra) -> str:
    """Añade un subconjunto con el contenido que se quiera, sin comprobarlo."""
    ficha = crear_ficha(nuevo_id(), entorno, "demostraciones", "A mano", "dashboard", **extra)
    texto = datos if isinstance(datos, str) else json.dumps(datos)
    agregar(biblioteca.raiz, ficha,
            lambda c: (c / ARCHIVO_SUBCONJUNTO).write_text(texto, encoding="utf-8"))
    return ficha["id"]


def _datos(conjunto: str, huella: str, **cambios) -> dict:
    return {"formato": FORMATO_SUBCONJUNTO, "version": VERSION_SUBCONJUNTO, "conjunto": conjunto,
            "huella_dataset": huella, "partidas": [0], **cambios}


@pytest.fixture
def biblioteca(tmp_path: Path) -> Biblioteca:
    return Biblioteca(tmp_path / "biblioteca", ENTORNOS)


def test_crear_un_subconjunto_y_verlo_en_la_biblioteca(biblioteca):
    conjunto = _anadir_conjunto(biblioteca)
    ficha = crear_subconjunto(biblioteca, conjunto, ELEGIDAS, "Mis mejores partidas")
    assert (ficha["origen"], ficha["fuente"], ficha["tipo"]) == ("dashboard", "experto",
                                                                 "demostraciones")
    carpeta = biblioteca.ruta(ficha["id"])
    assert not (carpeta / "demostraciones.npz").exists()  # no copia datos
    datos = json.loads((carpeta / ARCHIVO_SUBCONJUNTO).read_text(encoding="utf-8"))
    assert datos == {"formato": FORMATO_SUBCONJUNTO, "version": VERSION_SUBCONJUNTO,
                     "conjunto": conjunto,
                     "huella_dataset": huella_archivo(biblioteca.ruta(conjunto)
                                                      / "demostraciones.npz"),
                     "partidas": [0, 5]}  # en orden, sea cual sea el de la petición
    elemento = biblioteca.obtener(ficha["id"])
    assert elemento["valido"], elemento["motivo"]
    assert elemento["detalles"] == {"tipo": "subconjunto", "conjunto": conjunto, "partidas": 2,
                                    "pares": 4 + 5, "fuente": "experto", "splits": False}
    api = DemostracionApi.model_validate(como_demostracion(elemento))
    assert (api.tipo, api.partidas, api.pares, api.nombre) == ("subconjunto", 2, 9,
                                                               "Mis mejores partidas")


def test_el_mismo_subconjunto_da_el_mismo_archivo(biblioteca):
    conjunto = _anadir_conjunto(biblioteca)
    contenidos = set()
    for partidas in ([5, 0], [0, 5], (np.int64(5), np.int32(0))):
        ficha = crear_subconjunto(biblioteca, conjunto, partidas, "s")
        contenidos.add((biblioteca.ruta(ficha["id"]) / ARCHIVO_SUBCONJUNTO).read_bytes())
    assert len(contenidos) == 1


def test_cargar_un_subconjunto_da_sus_partidas_enteras(biblioteca):
    conjunto = _anadir_conjunto(biblioteca)
    ficha = crear_subconjunto(biblioteca, conjunto, ELEGIDAS, "s")
    partidas, datos = cargar_subconjunto(biblioteca, ficha["id"], OBSERVACION, DISCRETA)
    assert [p.id_episodio for p in partidas] == [0, 5] and datos["conjunto"] == conjunto
    for partida in partidas:
        esperada = _partida(pasos=PASOS_CONJUNTO[partida.id_episodio],
                            identificador=partida.id_episodio)
        for campo in ("estados", "acciones", "siguientes_estados", "terminados", "truncados"):
            np.testing.assert_array_equal(getattr(partida, campo), getattr(esperada, campo))


def test_la_fuente_del_subconjunto_es_la_de_su_conjunto(biblioteca):
    conjunto = _anadir_conjunto(biblioteca, fuente="experto")
    ficha = crear_subconjunto(biblioteca, conjunto, [3], "s")
    assert ficha["fuente"] == biblioteca.obtener(ficha["id"])["fuente"] == "experto"


@pytest.mark.parametrize("partidas, mensaje", [
    ([], "al menos una partida"),
    ({0: 1}, "lista de id_episodio"),
    ("05", "lista de id_episodio"),
    (5, "lista de id_episodio"),
    ([True], "no es un id de partida"),
    (["0"], "no es un id de partida"),
    ([1.0], "no es un id de partida"),
    ([0, 3, 0], "repetidas"),
    ([0, 9, 7], "las partidas [7, 9] no están en el conjunto"),
])
def test_crear_rechaza_partidas_que_no_valen(biblioteca, partidas, mensaje):
    conjunto = _anadir_conjunto(biblioteca)
    with pytest.raises(ValueError, match=re.escape(mensaje)):
        crear_subconjunto(biblioteca, conjunto, partidas, "s")
    assert [e["id"] for e in biblioteca.listar()] == [conjunto]  # no deja nada a medias


def test_crear_rechaza_lo_que_no_es_un_conjunto_valido(biblioteca):
    with pytest.raises(ValueError, match="No existe el conjunto"):
        crear_subconjunto(biblioteca, "a" * 12, ELEGIDAS, "s")
    conjunto = _anadir_conjunto(biblioteca)
    sub = crear_subconjunto(biblioteca, conjunto, ELEGIDAS, "s")["id"]
    with pytest.raises(ValueError, match="no es un conjunto de demostraciones"):
        crear_subconjunto(biblioteca, sub, ELEGIDAS, "s")

    def escribir_experto(carpeta: Path) -> None:
        (carpeta / "modelo.zip").write_bytes(b"x")
        (carpeta / "experto.json").write_text("{}")

    experto = agregar_nuevo(biblioteca.raiz, "cartpole", "experto", "e", "dashboard",
                            escribir_experto)["id"]
    with pytest.raises(ValueError, match="no es un conjunto de demostraciones"):
        crear_subconjunto(biblioteca, experto, ELEGIDAS, "s")
    roto = agregar_nuevo(biblioteca.raiz, "cartpole", "demostraciones", "roto", "dashboard",
                         lambda c: (c / "demostraciones.npz").write_bytes(b"no es un npz"))["id"]
    with pytest.raises(ValueError, match="no es válido"):
        crear_subconjunto(biblioteca, roto, ELEGIDAS, "s")


def test_sin_su_conjunto_o_si_ha_cambiado_deja_de_ser_valido(biblioteca):
    conjunto = _anadir_conjunto(biblioteca)
    sub = crear_subconjunto(biblioteca, conjunto, ELEGIDAS, "s")["id"]
    npz = biblioteca.ruta(conjunto) / "demostraciones.npz"
    # Otro NPZ válido en su lugar: el conjunto sigue valiendo, el subconjunto no.
    npz.unlink()
    _guardar_conjunto(npz, {0: 4, 3: 6, 5: 7})
    assert biblioteca.obtener(conjunto)["valido"]
    elemento = biblioteca.obtener(sub)
    assert not elemento["valido"] and "el conjunto ha cambiado" in elemento["motivo"]
    assert elemento["detalles"] == {"tipo": "subconjunto"}
    assert como_demostracion(elemento)["tipo"] == "subconjunto"
    with pytest.raises(ValueError, match="no es válido: el conjunto ha cambiado"):
        cargar_subconjunto(biblioteca, sub, OBSERVACION, DISCRETA)
    shutil.rmtree(biblioteca.ruta(conjunto))
    elemento = biblioteca.obtener(sub)
    assert not elemento["valido"] and elemento["motivo"] == f"no existe su conjunto {conjunto}"


def test_un_subconjunto_con_partidas_que_su_conjunto_no_tiene_no_es_valido(biblioteca):
    conjunto = _anadir_conjunto(biblioteca)
    huella = huella_archivo(biblioteca.ruta(conjunto) / "demostraciones.npz")
    sub = _anadir_a_mano(biblioteca, _datos(conjunto, huella, partidas=[0, 4]))
    elemento = biblioteca.obtener(sub)
    assert not elemento["valido"] and "las partidas [4] no están" in elemento["motivo"]


def test_cargar_comprueba_otra_vez_el_npz(biblioteca, monkeypatch):
    """Si el NPZ cambia entre listar y cargar, se nota al cargar."""
    from src.data import seleccion
    conjunto = _anadir_conjunto(biblioteca)
    sub = crear_subconjunto(biblioteca, conjunto, ELEGIDAS, "s")["id"]
    monkeypatch.setattr(seleccion, "huella_archivo", lambda ruta: "cd" * 32)
    with pytest.raises(ValueError, match="el conjunto ha cambiado"):
        cargar_subconjunto(biblioteca, sub, OBSERVACION, DISCRETA)


def test_subconjuntos_mal_apuntados_no_son_validos_y_no_hay_bucles(biblioteca):
    conjunto = _anadir_conjunto(biblioteca)
    huella = huella_archivo(biblioteca.ruta(conjunto) / "demostraciones.npz")
    bueno = _anadir_a_mano(biblioteca, _datos(conjunto, huella))
    assert biblioteca.obtener(bueno)["valido"]
    otro_entorno = _anadir_conjunto(biblioteca, "acrobot")
    casos = {
        _anadir_a_mano(biblioteca, _datos(bueno, huella)):
            f"{bueno} es un subconjunto: solo se saca de un conjunto",
        _anadir_a_mano(biblioteca, _datos(otro_entorno, huella)):
            f"{otro_entorno} no son demostraciones de cartpole",
        _anadir_a_mano(biblioteca, "{no json"): "no se pudo leer subconjunto.json",
        _anadir_a_mano(biblioteca, _datos(conjunto, huella, version=2)):
            "Formato de subconjunto desconocido",
        _anadir_a_mano(biblioteca, _datos(conjunto, huella), fuente="humano"):
            "la ficha dice fuente 'humano' y su conjunto, 'experto'",
    }
    for id_sub, motivo in casos.items():
        elemento = biblioteca.obtener(id_sub)
        assert not elemento["valido"] and motivo in elemento["motivo"], (id_sub, elemento)
    # Uno que apunta a sí mismo no deja la lectura en bucle.
    a_si_mismo = crear_ficha(nuevo_id(), "cartpole", "demostraciones", "yo", "dashboard")
    agregar(biblioteca.raiz, a_si_mismo, lambda c: (c / ARCHIVO_SUBCONJUNTO).write_text(
        json.dumps(_datos(a_si_mismo["id"], huella))))
    assert "es un subconjunto" in biblioteca.obtener(a_si_mismo["id"])["motivo"]
    assert len(biblioteca.listar()) == 9


def test_cargar_rechaza_lo_que_no_es_un_subconjunto_valido(biblioteca):
    conjunto = _anadir_conjunto(biblioteca)
    with pytest.raises(ValueError, match="no es un subconjunto"):
        cargar_subconjunto(biblioteca, conjunto, OBSERVACION, DISCRETA)
    with pytest.raises(ValueError, match="No existe el subconjunto"):
        cargar_subconjunto(biblioteca, "b" * 12, OBSERVACION, DISCRETA)


@pytest.mark.parametrize("cambio, mensaje", [
    (lambda d: d.pop("partidas"), "exactamente"),
    (lambda d: d.update(otra=1), "exactamente"),
    (lambda d: d.update(formato="otro"), "Formato de subconjunto desconocido"),
    (lambda d: d.update(version=2), "Formato de subconjunto desconocido"),
    (lambda d: d.update(conjunto="../a"), "no es un id de la biblioteca"),
    (lambda d: d.update(huella_dataset="corta"), "huella"),
    (lambda d: d.update(partidas="todas"), "al menos una partida"),
    (lambda d: d.update(partidas=[]), "al menos una partida"),
    (lambda d: d.update(partidas=[{"id_episodio": 0, "tramo": [0, 1]}]),
     "no es un id de partida"),
    (lambda d: d.update(partidas=[2, 2]), "repetidas"),
])
def test_forma_del_subconjunto(cambio, mensaje):
    datos = _datos("a" * 12, "ab" * 32)
    assert comprobar_subconjunto(datos) == datos
    cambio(datos)
    with pytest.raises(ValueError, match=re.escape(mensaje)):
        comprobar_subconjunto(datos)


def test_el_resumen_da_los_pasos_de_cada_partida(tmp_path):
    _guardar_conjunto(tmp_path / "d.npz")
    assert resumen_demostraciones(tmp_path / "d.npz")["pasos_por_partida"] == PASOS_CONJUNTO


def test_si_su_conjunto_no_es_valido_el_subconjunto_tampoco(biblioteca):
    """Aunque su NPZ se lea bien: aquí la ficha del conjunto dice otra fuente."""
    conjunto = _anadir_conjunto(biblioteca, fuente="humano")
    assert not biblioteca.obtener(conjunto)["valido"]
    huella = huella_archivo(biblioteca.ruta(conjunto) / "demostraciones.npz")
    sub = _anadir_a_mano(biblioteca, _datos(conjunto, huella), fuente="humano")
    elemento = biblioteca.obtener(sub)
    assert not elemento["valido"]
    assert elemento["motivo"].startswith(f"su conjunto {conjunto} no es válido: la ficha dice")


# --- combinar_demostraciones -----------------------------------------------------------

HUELLA_A, HUELLA_B, HUELLA_C = "aa" * 32, "bb" * 32, "cc" * 32


def _partidas(pasos_por_id: dict[int, int]) -> list[Trayectoria]:
    return [_partida(pasos=p, identificador=i) for i, p in pasos_por_id.items()]


def test_combinar_da_ids_nuevos_en_orden_y_dice_de_donde_sale_cada_una():
    a, b = _partidas({3: 4, 0: 5}), _partidas({0: 2, 1: 3})
    partidas, metadatos = combinar_demostraciones([(HUELLA_B, b), (HUELLA_A, a)], "CartPole-v1")
    assert [p.id_episodio for p in partidas] == [0, 1, 2, 3]
    assert [p.numero_pasos for p in partidas] == [5, 4, 2, 3]
    assert metadatos == {"fuente": "combinado", "entorno": "CartPole-v1", "partidas": [
        {"id_episodio": 0, "huella_dataset": HUELLA_A, "id_original": 0},
        {"id_episodio": 1, "huella_dataset": HUELLA_A, "id_original": 3},
        {"id_episodio": 2, "huella_dataset": HUELLA_B, "id_original": 0},
        {"id_episodio": 3, "huella_dataset": HUELLA_B, "id_original": 1}]}
    # Sin copiar los arrays y sin tocar las partidas de entrada.
    assert partidas[0].estados is a[1].estados and partidas[2].acciones is b[0].acciones
    assert [p.id_episodio for p in a] == [3, 0]


def test_una_partida_elegida_dos_veces_cuenta_una():
    """Un conjunto y un subconjunto suyo (otras copias de las mismas partidas)."""
    conjunto, subconjunto = _partidas({0: 4, 3: 6, 5: 5}), _partidas({3: 6, 5: 5})
    partidas, metadatos = combinar_demostraciones(
        [(HUELLA_A, conjunto), (HUELLA_A, subconjunto), (HUELLA_A, conjunto[:1])], "CartPole-v1")
    assert [m["id_original"] for m in metadatos["partidas"]] == [0, 3, 5]
    assert sum(p.numero_pasos for p in partidas) == 4 + 6 + 5  # pares sin repetir


def test_el_mismo_id_de_dos_npz_son_dos_partidas():
    partidas, metadatos = combinar_demostraciones(
        [(HUELLA_A, _partidas({0: 4})), (HUELLA_B, _partidas({0: 4}))], "CartPole-v1")
    assert len(partidas) == 2
    assert [m["huella_dataset"] for m in metadatos["partidas"]] == [HUELLA_A, HUELLA_B]


def test_el_orden_de_los_datos_no_cambia_el_resultado():
    """Todas las permutaciones de tres orígenes, con sus partidas al revés o no."""
    origenes = [(HUELLA_A, _partidas({2: 3, 0: 4})), (HUELLA_B, _partidas({1: 5})),
                (HUELLA_C, _partidas({7: 2, 4: 6}))]
    esperado = combinar_demostraciones(origenes, "CartPole-v1")
    for orden in itertools.permutations(origenes):
        for invertir in (False, True):
            entrada = [(h, list(reversed(p)) if invertir else p) for h, p in orden]
            partidas, metadatos = combinar_demostraciones(entrada, "CartPole-v1")
            assert metadatos == esperado[1]
            for obtenida, prevista in zip(partidas, esperado[0]):
                assert obtenida.id_episodio == prevista.id_episodio
                np.testing.assert_array_equal(obtenida.estados, prevista.estados)


def test_la_misma_partida_con_otro_contenido_es_un_error():
    otra = _partidas({0: 4})[0]
    distinta = dataclasses.replace(otra, acciones=(otra.acciones + 1) % 3)
    with pytest.raises(ValueError, match="llega dos veces con contenidos distintos"):
        combinar_demostraciones([(HUELLA_A, [otra]), (HUELLA_A, [distinta])], "CartPole-v1")


@pytest.mark.parametrize("origenes, mensaje", [
    ([], "ninguna partida"),
    ([(HUELLA_A, [])], "ninguna partida"),
    ([(HUELLA_A, []), (HUELLA_B, [])], "ninguna partida"),
    ("origenes", r"lista de \(huella, partidas\)"),
    ({HUELLA_A: []}, r"lista de \(huella, partidas\)"),
    ([HUELLA_A], r"origenes\[0\] debe ser \(huella, partidas\)"),
    ([(HUELLA_A, [], 1)], r"origenes\[0\] debe ser \(huella, partidas\)"),
    ([("corta", [])], "huella"),
    ([(HUELLA_A, "partidas")], "lista de Trayectoria"),
    ([(HUELLA_A, [{"id_episodio": 0}])], "objetos Trayectoria"),
])
def test_combinar_rechaza_lo_que_no_vale(origenes, mensaje):
    with pytest.raises(ValueError, match=mensaje):
        combinar_demostraciones(origenes, "CartPole-v1")


@pytest.mark.parametrize("entorno", ["", "  ", None, 3])
def test_combinar_necesita_el_entorno(entorno):
    with pytest.raises(ValueError, match="entorno"):
        combinar_demostraciones([(HUELLA_A, _partidas({0: 2}))], entorno)


def test_lo_combinado_se_guarda_y_se_lee_como_un_npz_combinado(tmp_path):
    partidas, metadatos = combinar_demostraciones(
        [(HUELLA_A, _partidas({3: 4, 0: 5})), (HUELLA_B, _partidas({0: 2}))], "CartPole-v1")
    ruta = guardar_demostraciones(tmp_path / "combinado.npz", partidas, metadatos,
                                  OBSERVACION, DISCRETA)
    leidas, leidos = cargar_demostraciones(ruta, OBSERVACION, DISCRETA)
    assert leidos == metadatos and fuente_demostraciones(leidos) == "combinado"
    assert [p.id_episodio for p in leidas] == [0, 1, 2]
    assert resumen_demostraciones(ruta)["pares"] == 11


# --- reparto de un combinado -----------------------------------------------------------

HUELLA_COMBINADO = "dd" * 32
UN_TERCIO, UN_SEXTO = 5 / 15, 5 / 30


def _combinado(tamanos: dict[str, int]) -> dict:
    """Metadatos de un combinado con tantas partidas de cada NPZ como se diga."""
    partidas, nuevo = [], 0
    for huella, cuantas in tamanos.items():
        for original in range(cuantas):
            partidas.append({"id_episodio": nuevo, "huella_dataset": huella,
                             "id_original": original})
            nuevo += 1
    return {"fuente": "combinado", "entorno": "CartPole-v1", "partidas": partidas}


def _por_origen(metadatos: dict, ids) -> dict[str, int]:
    origen = {p["id_episodio"]: p["huella_dataset"] for p in metadatos["partidas"]}
    cuenta: dict[str, int] = {}
    for i in ids:
        cuenta[origen[i]] = cuenta.get(origen[i], 0) + 1
    return cuenta


@pytest.mark.parametrize("tamanos, proporcion, validacion", [
    ({HUELLA_A: 15}, UN_TERCIO, {HUELLA_A: 5}),                 # como la E1
    ({HUELLA_A: 9, HUELLA_B: 6}, UN_TERCIO, {HUELLA_A: 3, HUELLA_B: 2}),
    ({HUELLA_A: 30}, UN_SEXTO, {HUELLA_A: 5}),
    ({HUELLA_A: 12, HUELLA_B: 4, HUELLA_C: 2}, UN_TERCIO, {HUELLA_A: 4, HUELLA_B: 1,
                                                           HUELLA_C: 1}),
    ({HUELLA_A: 2}, UN_TERCIO, {HUELLA_A: 1}),                  # 0,67 redondea a 1
    ({HUELLA_A: 3}, 0.5, {HUELLA_A: 2}),                        # 1,5: los medios hacia arriba
])
def test_cada_origen_aporta_su_parte_a_validacion(tamanos, proporcion, validacion):
    metadatos = _combinado(tamanos)
    splits = repartir_combinado(metadatos, HUELLA_COMBINADO, proporcion, crear_generador(0))
    assert _por_origen(metadatos, splits.validacion) == validacion
    total = sum(tamanos.values())
    assert len(splits.train) + len(splits.validacion) == total
    assert splits.huella_dataset == HUELLA_COMBINADO
    # Cualquier cantidad son las N primeras de train, de 1 a todas.
    assert splits.subconjuntos == {n: splits.train[:n] for n in range(1, len(splits.train) + 1)}
    validar_splits(splits, range(total))


@pytest.mark.parametrize("tamanos, proporcion, validacion", [
    # El redondeo deja validación vacía: una del origen más grande (A, por huella si empatan).
    ({HUELLA_A: 1, HUELLA_B: 1, HUELLA_C: 1}, UN_TERCIO, {HUELLA_A: 1}),
    ({HUELLA_B: 1, HUELLA_C: 2}, UN_SEXTO, {HUELLA_C: 1}),
    # El redondeo deja train vacío: una vuelve del origen más grande.
    ({HUELLA_A: 1, HUELLA_B: 1}, 0.5, {HUELLA_B: 1}),
    ({HUELLA_A: 1, HUELLA_B: 2}, 0.9, {HUELLA_A: 1, HUELLA_B: 1}),
])
def test_siempre_hay_una_de_train_y_una_de_validacion(tamanos, proporcion, validacion):
    metadatos = _combinado(tamanos)
    splits = repartir_combinado(metadatos, HUELLA_COMBINADO, proporcion, crear_generador(3))
    assert _por_origen(metadatos, splits.validacion) == validacion
    assert splits.train and splits.validacion


@pytest.mark.parametrize("total", range(2, 13))
@pytest.mark.parametrize("proporcion", [UN_TERCIO, UN_SEXTO, 0.5])
def test_con_un_origen_siempre_reparte_todas(total, proporcion):
    """De 2 a 12 partidas de un solo NPZ, con las proporciones de los perfiles y 0,5."""
    splits = repartir_combinado(_combinado({HUELLA_A: total}), HUELLA_COMBINADO, proporcion,
                                crear_generador(total))
    esperadas = min(max(math.floor(total * proporcion + 0.5), 1), total - 1)
    assert len(splits.validacion) == esperadas
    assert sorted(splits.train + splits.validacion) == list(range(total))


def test_el_reparto_depende_de_la_semilla_y_no_del_orden():
    metadatos = _combinado({HUELLA_A: 9, HUELLA_B: 6})
    al_reves = {**metadatos, "partidas": metadatos["partidas"][::-1]}
    a = repartir_combinado(metadatos, HUELLA_COMBINADO, UN_TERCIO, crear_generador(7))
    assert a == repartir_combinado(al_reves, HUELLA_COMBINADO, UN_TERCIO, crear_generador(7))
    assert a != repartir_combinado(metadatos, HUELLA_COMBINADO, UN_TERCIO, crear_generador(8))


def test_train_mezcla_los_origenes():
    """train se baraja entero: sus primeras partidas no son siempre del primer NPZ."""
    metadatos = _combinado({HUELLA_A: 10, HUELLA_B: 10})
    origenes_primera = {_por_origen(metadatos, repartir_combinado(
        metadatos, HUELLA_COMBINADO, UN_TERCIO, crear_generador(semilla)).train[:1]).popitem()[0]
        for semilla in range(10)}
    assert origenes_primera == {HUELLA_A, HUELLA_B}


@pytest.mark.parametrize("metadatos, proporcion, generador, mensaje", [
    (_combinado({HUELLA_A: 1}), UN_TERCIO, crear_generador(0), "al menos dos partidas"),
    ({"fuente": "combinado", "entorno": "x", "partidas": []}, UN_TERCIO, crear_generador(0),
     "al menos dos partidas"),
    ({"entorno": "x", "partidas": []}, UN_TERCIO, crear_generador(0), "combinados"),
    ({"fuente": "humano"}, UN_TERCIO, crear_generador(0), "combinados"),
    ([], UN_TERCIO, crear_generador(0), "combinados"),
    (_combinado({HUELLA_A: 4}), 0, crear_generador(0), "entre 0 y 1"),
    (_combinado({HUELLA_A: 4}), 1, crear_generador(0), "entre 0 y 1"),
    (_combinado({HUELLA_A: 4}), -0.2, crear_generador(0), "entre 0 y 1"),
    (_combinado({HUELLA_A: 4}), float("nan"), crear_generador(0), "entre 0 y 1"),
    (_combinado({HUELLA_A: 4}), True, crear_generador(0), "entre 0 y 1"),
    (_combinado({HUELLA_A: 4}), "0.3", crear_generador(0), "entre 0 y 1"),
    (_combinado({HUELLA_A: 4}), UN_TERCIO, 0, "generador"),
])
def test_repartir_rechaza_lo_que_no_vale(metadatos, proporcion, generador, mensaje):
    with pytest.raises(ValueError, match=mensaje):
        repartir_combinado(metadatos, HUELLA_COMBINADO, proporcion, generador)


def test_repartir_comprueba_la_huella_del_combinado():
    with pytest.raises(ValueError, match="huella"):
        repartir_combinado(_combinado({HUELLA_A: 4}), "corta", UN_TERCIO, crear_generador(0))


@pytest.mark.parametrize("entorno, esperada", [("cartpole", UN_TERCIO), ("mountaincar", UN_TERCIO),
                                               ("acrobot", UN_TERCIO), ("reacher", UN_SEXTO),
                                               ("halfcheetah", UN_SEXTO)])
def test_la_proporcion_es_la_de_cada_perfil(entorno, esperada):
    cfg = resolver_configuracion(CONFIG, entorno)
    assert proporcion_validacion(cfg) == pytest.approx(esperada)


def test_la_proporcion_necesita_la_seccion_demostraciones():
    with pytest.raises(ValueError, match="demostraciones"):
        proporcion_validacion({})


# --- datos de una petición de entrenamiento --------------------------------------------

CONFIG = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
CARTPOLE_OBS = spaces.Box(np.array([-4.8, -np.inf, -0.418, -np.inf], np.float32),
                          np.array([4.8, np.inf, 0.418, np.inf], np.float32))


def _cfg() -> dict:
    return resolver_configuracion(CONFIG, "cartpole")


def _partida_cartpole(pasos: int, identificador: int, base: float = 0.0) -> Trayectoria:
    """Partida que cabe en los espacios de CartPole, con recompensas 1, 2, 3..."""
    estados = (np.arange((pasos + 1) * 4, dtype=np.float32).reshape(pasos + 1, 4) * 1e-3
               + np.float32(base))
    terminados = np.zeros(pasos, dtype=bool)
    terminados[-1] = True
    return Trayectoria(identificador, estados[:-1], np.arange(pasos, dtype=np.int64) % 2,
                       estados[1:], np.arange(1, pasos + 1, dtype=np.float32), terminados,
                       np.zeros(pasos, dtype=bool))


def _anadir_cartpole(biblioteca: Biblioteca, pasos: dict[int, int], base: float = 0.0,
                     splits: bool = False) -> str:
    """Conjunto de CartPole en la biblioteca; con splits, de 2/3 train y 1/3 validación."""
    partidas = [_partida_cartpole(p, i, base) for i, p in pasos.items()]

    def escribir(carpeta: Path) -> None:
        npz = guardar_demostraciones(carpeta / "demostraciones.npz", partidas,
                                     _metadatos_experto(), CARTPOLE_OBS, spaces.Discrete(2))
        if splits:
            validacion = len(partidas) // 3
            guardar_splits(carpeta / "splits.json", crear_splits(
                partidas, len(partidas) - validacion, validacion, [1], crear_generador(0),
                huella_archivo(npz)))

    return agregar_nuevo(biblioteca.raiz, "cartpole", "demostraciones", "C", "dashboard",
                         escribir)["id"]


def _cargar(biblioteca: Biblioteca, id_conjunto: str):
    carpeta = biblioteca.ruta(id_conjunto)
    partidas, metadatos = cargar_demostraciones(carpeta / "demostraciones.npz", CARTPOLE_OBS,
                                                spaces.Discrete(2))
    splits = cargar_splits(carpeta / "splits.json", huella_archivo(carpeta / "demostraciones.npz"),
                           [p.id_episodio for p in partidas])
    return partidas, metadatos, splits


def test_un_conjunto_con_splits_se_usa_tal_cual(biblioteca):
    """Como los congelados: se entrena exactamente como en la E1."""
    conjunto = _anadir_cartpole(biblioteca, {0: 3, 1: 4, 2: 5}, splits=True)
    assert preparar_datos(biblioteca, [conjunto], _cfg()) == conjunto
    assert [e["id"] for e in biblioteca.listar()] == [conjunto]  # no crea nada


def test_un_conjunto_sin_splits_se_reparte_en_un_combinado(biblioteca):
    conjunto = _anadir_cartpole(biblioteca, {i: 3 + i for i in range(6)})
    nuevo = preparar_datos(biblioteca, [conjunto], _cfg())
    assert nuevo != conjunto
    elemento = biblioteca.obtener(nuevo)
    assert elemento["valido"] and elemento["fuente"] == "combinado"
    assert elemento["origen"] == "dashboard" and elemento["detalles"]["splits"]
    assert elemento["nombre"] == "Combinado · CartPole · 6 partidas"
    partidas, metadatos, splits = _cargar(biblioteca, nuevo)
    assert [p.numero_pasos for p in partidas] == [3, 4, 5, 6, 7, 8]
    assert len(splits.validacion) == 2 and len(splits.train) == 4  # 1/3 de CartPole
    # Con la semilla de los splits de la E1 y la proporción del perfil.
    semilla = semillas_independientes(_cfg()["demostraciones"]["semilla"], 2)[1]
    assert splits == repartir_combinado(metadatos, splits.huella_dataset, 5 / 15,
                                        crear_generador(semilla))


def test_varios_conjuntos_y_subconjuntos_se_combinan_sin_repetir(biblioteca):
    a = _anadir_cartpole(biblioteca, {0: 3, 1: 4, 2: 5})
    b = _anadir_cartpole(biblioteca, {0: 6, 1: 7, 2: 8}, base=0.2)
    sub_a = crear_subconjunto(biblioteca, a, [1, 2], "s")["id"]
    combinado = preparar_datos(biblioteca, [sub_a, a, b], _cfg())
    partidas, metadatos, splits = _cargar(biblioteca, combinado)
    assert len(partidas) == 6  # las de sub_a ya estaban en a
    assert resumen_seleccion(biblioteca, [sub_a, a, b], "cartpole") == {
        "partidas": 6, "pares": 3 + 4 + 5 + 6 + 7 + 8}
    huellas = {p["huella_dataset"] for p in metadatos["partidas"]}
    assert huellas == {huella_archivo(biblioteca.ruta(x) / "demostraciones.npz") for x in (a, b)}
    # Cada origen aporta a validación su tercio.
    origen = {p["id_episodio"]: p["huella_dataset"] for p in metadatos["partidas"]}
    assert sorted(origen[i] for i in splits.validacion) == sorted(huellas)


def test_los_mismos_datos_reutilizan_el_combinado(biblioteca):
    a = _anadir_cartpole(biblioteca, {0: 3, 1: 4, 2: 5})
    b = _anadir_cartpole(biblioteca, {0: 6, 1: 7}, base=0.2)
    primero = preparar_datos(biblioteca, [a, b], _cfg())
    total = len(biblioteca.listar())
    assert preparar_datos(biblioteca, [b, a], _cfg()) == primero  # en otro orden
    sub_b = crear_subconjunto(biblioteca, b, [0, 1], "todo b")["id"]
    assert preparar_datos(biblioteca, [a, sub_b], _cfg()) == primero  # las mismas partidas
    assert len(biblioteca.listar()) == total + 1  # solo el subconjunto nuevo
    # Otras partidas: otro combinado.
    assert preparar_datos(biblioteca, [a], _cfg()) not in (primero, a)


def test_otro_reparto_es_otro_combinado(biblioteca):
    a = _anadir_cartpole(biblioteca, {0: 3, 1: 4, 2: 5})
    cfg = _cfg()
    primero = preparar_datos(biblioteca, [a], cfg)
    cfg["demostraciones"]["semilla"] = cfg["demostraciones"]["semilla"] + 1
    assert preparar_datos(biblioteca, [a], cfg) != primero


def test_la_clave_de_seleccion_no_depende_del_orden():
    metadatos = {"partidas": [{"huella_dataset": HUELLA_A, "id_original": 2},
                              {"huella_dataset": HUELLA_B, "id_original": 0}]}
    al_reves = {"partidas": metadatos["partidas"][::-1]}
    assert clave_seleccion(metadatos, 0.5, 1) == clave_seleccion(al_reves, 0.5, 1)
    assert clave_seleccion(metadatos, 0.5, 1) != clave_seleccion(metadatos, 0.5, 2)
    assert clave_seleccion(metadatos, 0.5, 1) != clave_seleccion(metadatos, 0.4, 1)


def test_preparar_rechaza_datos_que_no_valen(biblioteca):
    conjunto = _anadir_cartpole(biblioteca, {0: 3, 1: 4})
    acrobot = agregar_nuevo(biblioteca.raiz, "acrobot", "demostraciones", "A", "dashboard",
                            lambda c: _guardar_conjunto(c / "demostraciones.npz",
                                                        gym="Acrobot-v1"))["id"]
    roto = agregar_nuevo(biblioteca.raiz, "cartpole", "demostraciones", "roto", "dashboard",
                         lambda c: (c / "demostraciones.npz").write_bytes(b"no"))["id"]
    casos = [([], "al menos un conjunto"), ("a" * 12, "al menos un conjunto"),
             ([conjunto, conjunto], "repetidos"), (["b" * 12], "No existen"),
             ([3], "por su id"), ([acrobot], "de acrobot, no de cartpole"),
             ([roto], "no son válidas")]
    for ids, mensaje in casos:
        with pytest.raises(ValueError, match=mensaje):
            preparar_datos(biblioteca, ids, _cfg())
        with pytest.raises(ValueError, match=mensaje):
            resumen_seleccion(biblioteca, ids, "cartpole")


def test_preparar_con_un_modelo_no_vale(biblioteca):
    def escribir_experto(carpeta: Path) -> None:
        (carpeta / "modelo.zip").write_bytes(b"x")
        (carpeta / "experto.json").write_text("{}")

    experto = agregar_nuevo(biblioteca.raiz, "cartpole", "experto", "e", "dashboard",
                            escribir_experto)["id"]
    with pytest.raises(ValueError, match="no son demostraciones"):
        preparar_datos(biblioteca, [experto], _cfg())


def test_retornos_de_train_en_su_orden(biblioteca):
    conjunto = _anadir_cartpole(biblioteca, {0: 3, 1: 4, 2: 5, 3: 6, 4: 7, 5: 8}, splits=True)
    partidas, _, splits = _cargar(biblioteca, conjunto)
    por_id = {p.id_episodio: p for p in partidas}
    assert retornos_train(biblioteca, conjunto) == [retorno_partida(por_id[i])
                                                    for i in splits.train]
    # Recompensas 1, 2, 3...: una partida de n pasos devuelve n(n+1)/2.
    assert sorted(retornos_train(biblioteca, conjunto)) == sorted(
        n * (n + 1) / 2 for n in (por_id[i].numero_pasos for i in splits.train))


def test_retornos_de_train_necesitan_un_conjunto_con_splits(biblioteca):
    sin_splits = _anadir_cartpole(biblioteca, {0: 3, 1: 4})
    with pytest.raises(ValueError, match="no tiene splits"):
        retornos_train(biblioteca, sin_splits)
    sub = crear_subconjunto(biblioteca, sin_splits, [0], "s")["id"]
    with pytest.raises(ValueError, match="no es un conjunto"):
        retornos_train(biblioteca, sub)


def test_retornos_por_partida_suma_como_retorno_partida(tmp_path):
    partidas = [_partida_cartpole(5, 0), _partida_cartpole(3, 9)]
    ruta = guardar_demostraciones(tmp_path / "d.npz", partidas, _metadatos_experto(),
                                  CARTPOLE_OBS, spaces.Discrete(2))
    assert retornos_por_partida(ruta) == {0: retorno_partida(partidas[0]),
                                          9: retorno_partida(partidas[1])}
    (tmp_path / "malo.npz").write_bytes(b"no es un npz")
    with pytest.raises(ValueError, match="No se pudo leer"):
        retornos_por_partida(tmp_path / "malo.npz")



def test_preparar_un_combinado_se_hace_por_turnos(biblioteca, monkeypatch):
    """Dos entrenamientos con los mismos datos a la vez no crean dos combinados."""
    from contextlib import contextmanager

    from src.data import seleccion
    usados = []

    @contextmanager
    def espia(cerrojo):
        usados.append(Path(cerrojo))
        yield

    monkeypatch.setattr(seleccion, "turno_exclusivo", espia)
    conjunto = _anadir_cartpole(biblioteca, {0: 3, 1: 4})
    preparar_datos(biblioteca, [conjunto], _cfg())
    assert usados == [biblioteca.raiz / seleccion.ARCHIVO_CERROJO_COMBINADOS]
    # El cerrojo está oculto: no sale en la biblioteca.
    assert all(not e["id"].startswith(".") for e in biblioteca.listar())


def test_solo_se_reutilizan_combinados(biblioteca):
    """Un conjunto de otra fuente con la misma clave en su ficha no se toma por el combinado."""
    a = _anadir_cartpole(biblioteca, {0: 3, 1: 4, 2: 5})
    cfg = _cfg()
    semilla = semillas_independientes(cfg["demostraciones"]["semilla"], 2)[1]
    huella = huella_archivo(biblioteca.ruta(a) / "demostraciones.npz")
    clave = clave_seleccion({"partidas": [{"huella_dataset": huella, "id_original": i}
                                          for i in range(3)]}, 5 / 15, semilla)
    partidas = [_partida_cartpole(p, i) for i, p in {0: 3, 1: 4, 2: 5}.items()]

    def escribir(carpeta: Path) -> None:
        npz = guardar_demostraciones(carpeta / "demostraciones.npz", partidas,
                                     _metadatos_experto(), CARTPOLE_OBS, spaces.Discrete(2))
        guardar_splits(carpeta / "splits.json", crear_splits(
            partidas, 2, 1, [1], crear_generador(0), huella_archivo(npz)))

    impostor = agregar_nuevo(biblioteca.raiz, "cartpole", "demostraciones", "I", "dashboard",
                             escribir, seleccion=clave)["id"]
    combinado = preparar_datos(biblioteca, [a], cfg)
    assert combinado not in (a, impostor)
    assert biblioteca.obtener(combinado)["fuente"] == "combinado"


def test_el_resumen_de_un_subconjunto_cuenta_solo_sus_partidas(biblioteca):
    a = _anadir_cartpole(biblioteca, {0: 3, 1: 4, 2: 5})
    sub = crear_subconjunto(biblioteca, a, [0, 2], "s")["id"]
    assert resumen_seleccion(biblioteca, [sub], "cartpole") == {"partidas": 2, "pares": 3 + 5}


# --- partidas que llegan a través de un combinado (auditoría del 2026-11-26) -------------

def test_un_combinado_con_su_conjunto_original_no_repite_partidas(biblioteca):
    """Regresión: el combinado traía las mismas partidas con otra huella; 6 contaban 12
    y una misma partida podía acabar a la vez en train y en validación."""
    a = _anadir_cartpole(biblioteca, {i: 3 + i for i in range(6)})
    combinado = preparar_datos(biblioteca, [a], _cfg())
    assert resumen_seleccion(biblioteca, [a, combinado], "cartpole") == {
        "partidas": 6, "pares": sum(3 + i for i in range(6))}
    total = len(biblioteca.listar())
    # Son los mismos datos que [a]: se reutiliza el mismo combinado.
    assert preparar_datos(biblioteca, [a, combinado], _cfg()) == combinado
    assert preparar_datos(biblioteca, [combinado], _cfg()) == combinado  # tiene splits
    assert len(biblioteca.listar()) == total


def test_un_subconjunto_de_un_combinado_cuenta_por_sus_partidas_originales(biblioteca):
    a = _anadir_cartpole(biblioteca, {i: 3 + i for i in range(6)})
    combinado = preparar_datos(biblioteca, [a], _cfg())
    sub = crear_subconjunto(biblioteca, combinado, [0, 1], "del combinado")["id"]
    assert resumen_seleccion(biblioteca, [a, sub], "cartpole")["partidas"] == 6
    assert preparar_datos(biblioteca, [a, sub], _cfg()) == combinado
    assert resumen_seleccion(biblioteca, [sub], "cartpole") == {"partidas": 2, "pares": 3 + 4}


def test_un_combinado_de_un_combinado_apunta_a_los_npz_originales(biblioteca):
    a = _anadir_cartpole(biblioteca, {0: 3, 1: 4, 2: 5})
    b = _anadir_cartpole(biblioteca, {0: 6, 1: 7, 2: 8}, base=0.2)
    de_a = preparar_datos(biblioteca, [a], _cfg())
    nuevo = preparar_datos(biblioteca, [de_a, b], _cfg())
    partidas, metadatos, splits = _cargar(biblioteca, nuevo)
    originales = {huella_archivo(biblioteca.ruta(x) / "demostraciones.npz") for x in (a, b)}
    assert {p["huella_dataset"] for p in metadatos["partidas"]} == originales
    assert len(partidas) == 6
    # Cada partida original, una sola vez: ninguna puede estar en train y en validación.
    identidades = [(p["huella_dataset"], p["id_original"]) for p in metadatos["partidas"]]
    assert len(set(identidades)) == len(identidades)
    assert not set(splits.train) & set(splits.validacion)


def test_identidades_originales():
    experto = {"entorno": "CartPole-v1", "experto": "x", "procedencia": {},
               "configuracion_experto": {}}
    assert identidades_originales(experto, HUELLA_A, [3, 0]) == {3: (HUELLA_A, 3),
                                                                 0: (HUELLA_A, 0)}
    combinado = {"fuente": "combinado", "entorno": "CartPole-v1", "partidas": [
        {"id_episodio": 0, "huella_dataset": HUELLA_B, "id_original": 7},
        {"id_episodio": 1, "huella_dataset": HUELLA_C, "id_original": 2}]}
    assert identidades_originales(combinado, HUELLA_A, [1]) == {1: (HUELLA_C, 2)}



def test_las_partidas_de_un_combinado_recuperan_su_id_original(biblioteca):
    """En un combinado de A y B, las de B tienen ids nuevos (3, 4, 5): elegirlo con B
    tiene que reconocerlas por su id en B (0, 1, 2), no por el del combinado."""
    a = _anadir_cartpole(biblioteca, {0: 3, 1: 4, 2: 5})
    b = _anadir_cartpole(biblioteca, {0: 6, 1: 7, 2: 8}, base=0.2)
    de_ab = preparar_datos(biblioteca, [a, b], _cfg())
    assert resumen_seleccion(biblioteca, [de_ab, b], "cartpole")["partidas"] == 6
    assert preparar_datos(biblioteca, [de_ab, b], _cfg()) == de_ab
