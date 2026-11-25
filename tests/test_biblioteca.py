"""Tests de la biblioteca (src/data/biblioteca.py), su precarga (src/training/precarga.py)
y sus rutas (src/app/rutas_biblioteca.py).

Los elementos se fabrican aquí con archivos mínimos pero reales: un experto.json y un
gail.json como los de la E1, un politica.pt con sus metadatos y un NPZ guardado con
guardar_demostraciones. La precarga se prueba con descargas falsas, sin W&B.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from gymnasium import spaces

from src.data import biblioteca as modulo_biblioteca
from src.data.biblioteca import (ARCHIVO_FICHA, CARACTERES_ID, CARPETA_IDS, INTENTOS_ID, ORIGENES,
                                 TIPOS, Biblioteca, IdRepetido, agregar, agregar_nuevo,
                                 comprobar_ficha, crear_ficha, es_id, nuevo_id)
from src.data.demostraciones import guardar_demostraciones, resumen_demostraciones
from src.data.tipos import Trayectoria
from src.training import precarga

ENTORNOS = {"cartpole": "CartPole-v1", "reacher": "Reacher-v5"}
HUELLA = "a" * 64


# --- Fabricar elementos -------------------------------------------------------------

def _trayectoria(identificador: int, pasos: int) -> Trayectoria:
    """Partida de CartPole de pruebas con los pasos pedidos."""
    estados = np.zeros((pasos, 4), dtype=np.float32)
    terminados = np.zeros(pasos, dtype=bool)
    terminados[-1] = True
    return Trayectoria(id_episodio=identificador, estados=estados,
                       acciones=np.zeros(pasos, dtype=np.int64),
                       siguientes_estados=estados.copy(),
                       recompensas_originales=np.ones(pasos, dtype=np.float32),
                       terminados=terminados, truncados=np.zeros(pasos, dtype=bool))


def escribir_archivos(carpeta: Path, tipo: str, gym: str = "CartPole-v1") -> None:
    """Escribe en carpeta los archivos de un elemento del tipo, como los de la E1."""
    if tipo == "experto":
        (carpeta / "modelo.zip").write_bytes(b"zip")
        (carpeta / "experto.json").write_text(json.dumps({"resumen": {
            "entorno": gym, "pasos": 60000, "iteraciones": 12, "mejor_retorno": 500.0,
            "objetivo": 500, "objetivo_alcanzado": True, "segundos": 13.5}}), encoding="utf-8")
    elif tipo == "gail":
        for nombre in ("modelo.zip", "discriminador.pt"):
            (carpeta / nombre).write_bytes(b"x")
        (carpeta / "gail.json").write_text(json.dumps({
            "resumen": {"entorno": gym, "cantidad": 4, "repeticion": 1, "iteraciones": 20,
                        "pasos": 100000, "segundos": 31.0},
            "datos": {"huella_dataset": HUELLA}}), encoding="utf-8")
    elif tipo == "bc":
        torch.save({"estado": {}, "metadatos": {
            "entorno": gym, "subconjunto": 7, "semilla": 12, "huella_dataset": HUELLA,
            "adicional": {"resumen": {"repeticion": 2, "segundos": 1.2}}}},
            carpeta / "politica.pt")
    else:
        guardar_demostraciones(carpeta / "demostraciones.npz",
                               [_trayectoria(0, 5), _trayectoria(1, 3)],
                               {"entorno": gym, "experto": "x", "procedencia": {},
                                "configuracion_experto": {}},
                               spaces.Box(-np.inf, np.inf, (4,), np.float32), spaces.Discrete(2))


def nuevo(raiz: Path, tipo: str, entorno: str = "cartpole", origen: str = "dashboard",
          **extra: Any) -> str:
    """Añade un elemento válido a la biblioteca y devuelve su id."""
    ficha = crear_ficha(nuevo_id(), entorno, tipo, f"{tipo} de prueba", origen, **extra)
    agregar(raiz, ficha, lambda carpeta: escribir_archivos(carpeta, tipo, ENTORNOS[entorno]))
    return ficha["id"]


@pytest.fixture
def biblioteca(tmp_path: Path) -> Biblioteca:
    """Biblioteca vacía en una carpeta temporal."""
    return Biblioteca(tmp_path / "biblioteca", ENTORNOS)


# --- Ids y fichas -------------------------------------------------------------------

def test_los_ids_son_cortos_hexadecimales_y_distintos() -> None:
    ids = {nuevo_id() for _ in range(200)}
    assert len(ids) == 200
    assert all(len(i) == CARACTERES_ID and es_id(i) for i in ids)


@pytest.mark.parametrize("valor", ["", "abc", "G" * 12, "a" * 13, "a" * 11, None, 3,
                                   "../" + "a" * 9, "A" * 12])
def test_lo_que_no_es_un_id(valor) -> None:
    assert not es_id(valor)


def test_crear_ficha_valida() -> None:
    ficha = crear_ficha("0" * 12, "cartpole", "experto", "Experto", "congelado",
                        artefacto="experto-x", descripcion="d")
    assert ficha["formato"] == "tiiac-biblioteca" and ficha["version"] == 1
    assert ficha["creado"].endswith("+00:00")
    assert ficha["artefacto"] == "experto-x"


@pytest.mark.parametrize("cambio, motivo", [
    ({"formato": "otro"}, "formato"),
    ({"version": 2}, "formato"),
    ({"id": "corto"}, "id no válido"),
    ({"tipo": "ppo"}, "tipo no válido"),
    ({"origen": "otro"}, "origen no válido"),
    ({"nombre": "  "}, "falta nombre"),
    ({"entorno": ""}, "falta entorno"),
    ({"creado": None}, "falta creado"),
    ({"fuente": "robot"}, "fuente no válida"),
    ({"entorno": "cartpoel"}, "entorno no válido"),
    ({"entorno": "CartPole-v1"}, "entorno no válido"),
])
def test_fichas_que_no_valen(cambio, motivo) -> None:
    ficha = {**crear_ficha("0" * 12, "cartpole", "bc", "BC", "dashboard"), **cambio}
    with pytest.raises(ValueError, match=motivo):
        comprobar_ficha(ficha)


@pytest.mark.parametrize("campo, valor", [("entorno", "reacher"), ("tipo", "gail"),
                                          ("id", "1" * 12)])
def test_la_ficha_tiene_que_coincidir_con_su_carpeta(campo, valor) -> None:
    ficha = crear_ficha("0" * 12, "cartpole", "bc", "BC", "dashboard")
    with pytest.raises(ValueError, match=f"la ficha dice {campo}"):
        comprobar_ficha(ficha, **{"entorno": "cartpole", "tipo": "bc", "id_elemento": "0" * 12,
                                  {"id": "id_elemento"}.get(campo, campo): valor})


# --- Agregar --------------------------------------------------------------------------

@pytest.mark.parametrize("tipo", TIPOS)
def test_agregar_deja_cada_tipo_en_su_sitio_con_su_ficha(biblioteca, tipo) -> None:
    id_ = nuevo(biblioteca.raiz, tipo)
    carpeta = biblioteca.raiz / "cartpole" / tipo / id_
    assert (carpeta / ARCHIVO_FICHA).is_file()
    assert biblioteca.ruta(id_) == carpeta


def test_agregar_sin_los_archivos_del_tipo_no_deja_nada(biblioteca) -> None:
    ficha = crear_ficha(nuevo_id(), "cartpole", "gail", "GAIL", "dashboard")
    with pytest.raises(ValueError, match="Faltan archivos"):
        agregar(biblioteca.raiz, ficha, lambda c: (c / "modelo.zip").write_bytes(b"x"))
    assert biblioteca.listar() == []
    carpeta = biblioteca.raiz / "cartpole" / "gail"
    assert not any(p.name.startswith(".") for p in carpeta.iterdir())


def test_agregar_si_falla_al_escribir_no_deja_nada(biblioteca) -> None:
    def roto(carpeta: Path) -> None:
        (carpeta / "modelo.zip").write_bytes(b"x")
        raise RuntimeError("se cortó")

    ficha = crear_ficha(nuevo_id(), "cartpole", "experto", "E", "dashboard")
    with pytest.raises(RuntimeError):
        agregar(biblioteca.raiz, ficha, roto)
    assert biblioteca.listar() == []


def test_agregar_el_mismo_id_dos_veces_falla(biblioteca) -> None:
    id_ = nuevo(biblioteca.raiz, "bc")
    ficha = crear_ficha(id_, "cartpole", "bc", "otro", "dashboard")
    with pytest.raises(FileExistsError):
        agregar(biblioteca.raiz, ficha, lambda c: escribir_archivos(c, "bc"))


# --- Ids únicos en toda la biblioteca ------------------------------------------------

def _ficha(id_: str, entorno: str = "cartpole", tipo: str = "bc") -> dict[str, Any]:
    """Ficha de pruebas con un id dado."""
    return crear_ficha(id_, entorno, tipo, "prueba", "dashboard")


@pytest.mark.parametrize("entorno, tipo", [("cartpole", "gail"), ("reacher", "bc"),
                                           ("reacher", "demostraciones")])
def test_un_id_no_se_repite_ni_en_otro_entorno_ni_en_otro_tipo(biblioteca, entorno,
                                                              tipo) -> None:
    id_ = nuevo(biblioteca.raiz, "bc")
    with pytest.raises(IdRepetido, match=id_):
        agregar(biblioteca.raiz, _ficha(id_, entorno, tipo),
                lambda c: escribir_archivos(c, tipo, ENTORNOS[entorno]))
    assert [e["id"] for e in biblioteca.listar()] == [id_]
    assert biblioteca.obtener(id_)["tipo"] == "bc"


def test_tambien_cuenta_un_elemento_puesto_a_mano_sin_reserva(biblioteca) -> None:
    id_ = nuevo(biblioteca.raiz, "experto")
    (biblioteca.raiz / CARPETA_IDS / id_).unlink()  # como si se hubiera copiado a mano
    with pytest.raises(IdRepetido):
        agregar(biblioteca.raiz, _ficha(id_), lambda c: escribir_archivos(c, "bc"))
    # La comprobación no deja una reserva colgada: el elemento a mano sigue sin ella.
    assert not (biblioteca.raiz / CARPETA_IDS / id_).exists()


def test_un_id_reservado_por_otro_proceso_no_se_puede_usar(biblioteca) -> None:
    id_ = nuevo_id()
    (biblioteca.raiz / CARPETA_IDS).mkdir(parents=True)
    (biblioteca.raiz / CARPETA_IDS / id_).touch()
    with pytest.raises(IdRepetido):
        agregar(biblioteca.raiz, _ficha(id_), lambda c: escribir_archivos(c, "bc"))


def test_si_falla_al_guardar_el_id_queda_libre(biblioteca) -> None:
    id_ = nuevo_id()

    def roto(carpeta: Path) -> None:
        raise RuntimeError("se cortó")

    with pytest.raises(RuntimeError):
        agregar(biblioteca.raiz, _ficha(id_), roto)
    assert not (biblioteca.raiz / CARPETA_IDS / id_).exists()
    agregar(biblioteca.raiz, _ficha(id_), lambda c: escribir_archivos(c, "bc"))
    assert biblioteca.obtener(id_)["valido"]


def test_con_varios_a_la_vez_y_el_mismo_id_solo_entra_uno(biblioteca) -> None:
    import threading

    id_ = nuevo_id()
    destinos = [(e, t) for e in ENTORNOS for t in TIPOS]
    resultados: list[str] = []
    salida = threading.Barrier(len(destinos))

    def intentar(entorno: str, tipo: str) -> None:
        salida.wait()
        try:
            agregar(biblioteca.raiz, _ficha(id_, entorno, tipo),
                    lambda c: escribir_archivos(c, tipo, ENTORNOS[entorno]))
            resultados.append("ok")
        except IdRepetido:
            resultados.append("repetido")

    hilos = [threading.Thread(target=intentar, args=d) for d in destinos]
    for hilo in hilos:
        hilo.start()
    for hilo in hilos:
        hilo.join()
    assert sorted(resultados) == ["ok"] + ["repetido"] * (len(destinos) - 1)
    assert [e["id"] for e in biblioteca.listar()] == [id_]


def test_agregar_nuevo_prueba_otro_id_si_sale_repetido(biblioteca, monkeypatch) -> None:
    usado = nuevo(biblioteca.raiz, "gail", entorno="reacher")
    libre = nuevo_id()
    ids = iter([usado, usado, libre])
    monkeypatch.setattr(modulo_biblioteca, "nuevo_id", lambda: next(ids))
    ficha = agregar_nuevo(biblioteca.raiz, "cartpole", "bc", "BC", "dashboard",
                          lambda c: escribir_archivos(c, "bc"), descripcion="d")
    assert ficha["id"] == libre and ficha["descripcion"] == "d"
    assert biblioteca.obtener(libre)["valido"]


def test_agregar_nuevo_se_rinde_tras_varios_repetidos(biblioteca, monkeypatch) -> None:
    usado = nuevo(biblioteca.raiz, "bc")
    llamadas = []
    monkeypatch.setattr(modulo_biblioteca, "nuevo_id", lambda: llamadas.append(1) or usado)
    with pytest.raises(IdRepetido, match=f"{INTENTOS_ID} ids"):
        agregar_nuevo(biblioteca.raiz, "cartpole", "experto", "E", "dashboard",
                      lambda c: escribir_archivos(c, "experto"))
    assert len(llamadas) == INTENTOS_ID


def test_las_reservas_no_salen_en_el_indice(biblioteca) -> None:
    nuevo(biblioteca.raiz, "bc")
    assert (biblioteca.raiz / CARPETA_IDS).is_dir()
    assert len(biblioteca.listar()) == 1


# --- Índice -------------------------------------------------------------------------------

def test_indice_de_cada_tipo_con_sus_detalles(biblioteca) -> None:
    ids = {tipo: nuevo(biblioteca.raiz, tipo) for tipo in TIPOS}
    por_id = {e["id"]: e for e in biblioteca.listar()}
    assert {e["valido"] for e in por_id.values()} == {True}
    assert por_id[ids["experto"]]["detalles"] == {
        "pasos": 60000, "iteraciones": 12, "mejor_retorno": 500.0, "objetivo": 500,
        "objetivo_alcanzado": True, "segundos": 13.5}
    assert por_id[ids["gail"]]["detalles"] == {"cantidad": 4, "repeticion": 1,
                                               "iteraciones": 20, "pasos": 100000,
                                               "segundos": 31.0, "huella_dataset": HUELLA}
    assert por_id[ids["bc"]]["detalles"] == {"subconjunto": 7, "semilla": 12, "repeticion": 2,
                                             "segundos": 1.2, "huella_dataset": HUELLA}
    assert por_id[ids["demostraciones"]]["detalles"] == {"partidas": 2, "pares": 8,
                                                         "splits": False}
    assert por_id[ids["demostraciones"]]["fuente"] == "experto"  # por defecto
    assert por_id[ids["bc"]]["fuente"] is None


def test_listar_filtra_por_entorno_y_tipo_y_ordena(biblioteca) -> None:
    a = nuevo(biblioteca.raiz, "gail", creado="2026-01-02T00:00:00+00:00")
    b = nuevo(biblioteca.raiz, "gail", creado="2026-01-01T00:00:00+00:00")
    c = nuevo(biblioteca.raiz, "experto", entorno="reacher")
    d = nuevo(biblioteca.raiz, "experto")
    assert [e["id"] for e in biblioteca.listar()] == [d, b, a, c]
    assert [e["id"] for e in biblioteca.listar("reacher")] == [c]
    assert [e["id"] for e in biblioteca.listar(tipos=("gail",))] == [b, a]
    assert biblioteca.listar("cartpole", ("bc",)) == []


def test_una_biblioteca_que_no_existe_esta_vacia(tmp_path) -> None:
    assert Biblioteca(tmp_path / "no-existe", ENTORNOS).listar() == []


@pytest.mark.parametrize("estropear, motivo", [
    (lambda c: (c / ARCHIVO_FICHA).unlink(), "no se pudo leer ficha.json"),
    (lambda c: (c / ARCHIVO_FICHA).write_text("{roto", encoding="utf-8"),
     "no se pudo leer ficha.json"),
    (lambda c: (c / "experto.json").unlink(), "faltan experto.json"),
    (lambda c: (c / "experto.json").write_text("[", encoding="utf-8"),
     "no se pudo leer experto.json"),
    (lambda c: (c / "experto.json").write_text(json.dumps({"resumen": {"entorno": "Otro-v0"}}),
                                                encoding="utf-8"), "no de CartPole-v1"),
    (lambda c: (c / ARCHIVO_FICHA).write_text(json.dumps(
        {**json.loads((c / ARCHIVO_FICHA).read_text(encoding="utf-8")), "tipo": "bc"}),
        encoding="utf-8"), "la ficha dice tipo"),
])
def test_un_experto_roto_sale_como_no_valido_con_el_motivo(biblioteca, estropear,
                                                           motivo) -> None:
    id_ = nuevo(biblioteca.raiz, "experto")
    estropear(biblioteca.ruta(id_))
    elemento, = biblioteca.listar()
    assert elemento["id"] == id_ and elemento["valido"] is False
    assert motivo in elemento["motivo"]


@pytest.mark.parametrize("tipo, estropear, motivo", [
    ("bc", lambda c: (c / "politica.pt").write_bytes(b"no es torch"), "no se pudo leer"),
    ("bc", lambda c: torch.save({"estado": {}}, c / "politica.pt"), "no tiene metadatos"),
    ("bc", lambda c: torch.save({"metadatos": {"entorno": "Otro-v0"}}, c / "politica.pt"),
     "no de CartPole-v1"),
    ("gail", lambda c: (c / "discriminador.pt").unlink(), "faltan discriminador.pt"),
    ("gail", lambda c: (c / "gail.json").write_text(json.dumps({"resumen": {}}),
                                                     encoding="utf-8"), "no de CartPole-v1"),
    ("demostraciones", lambda c: (c / "demostraciones.npz").write_bytes(b"x"), "No se pudo leer"),
])
def test_otros_elementos_rotos(biblioteca, tipo, estropear, motivo) -> None:
    id_ = nuevo(biblioteca.raiz, tipo)
    estropear(biblioteca.ruta(id_))
    elemento = biblioteca.obtener(id_)
    assert elemento["valido"] is False and motivo in elemento["motivo"]


def _json(carpeta: Path, nombre: str, contenido: Any) -> None:
    """Sustituye un JSON de un elemento."""
    (carpeta / nombre).write_text(json.dumps(contenido), encoding="utf-8")


@pytest.mark.parametrize("tipo, estropear, motivo", [
    ("experto", lambda c: _json(c, "experto.json", [1, 2]), "experto.json no es un objeto"),
    ("experto", lambda c: _json(c, "experto.json", {"resumen": [1]}),
     "el resumen de experto.json no es un objeto"),
    ("gail", lambda c: _json(c, "gail.json", {"resumen": {"entorno": "CartPole-v1"}, "datos": []}),
     "los datos de gail.json no es un objeto"),
    ("gail", lambda c: _json(c, "gail.json", "texto"), "gail.json no es un objeto"),
    ("bc", lambda c: torch.save({"metadatos": {"entorno": "CartPole-v1", "adicional": [1]}},
                                c / "politica.pt"), "adicional de politica.pt no es un objeto"),
    ("bc", lambda c: torch.save({"metadatos": {"entorno": "CartPole-v1",
                                               "adicional": {"resumen": 3}}}, c / "politica.pt"),
     "el resumen de politica.pt no es un objeto"),
])
def test_metadatos_que_no_son_objetos_no_tumban_el_indice(biblioteca, tipo, estropear,
                                                          motivo) -> None:
    """Regresión: un gail.json con "datos": [] o un experto.json que es una lista daban
    AttributeError y fallaba el listado entero. Ahora ese elemento sale como no válido y
    los demás siguen."""
    bueno = nuevo(biblioteca.raiz, "demostraciones")
    roto = nuevo(biblioteca.raiz, tipo)
    estropear(biblioteca.ruta(roto))
    elementos = {e["id"]: e for e in biblioteca.listar()}
    assert elementos[bueno]["valido"] is True
    assert elementos[roto]["valido"] is False and motivo in elementos[roto]["motivo"]


def test_un_fallo_inesperado_al_leer_un_elemento_no_tumba_el_indice(biblioteca,
                                                                   monkeypatch) -> None:
    bueno = nuevo(biblioteca.raiz, "experto")
    roto = nuevo(biblioteca.raiz, "gail")

    def falla(carpeta: Path, gym: str) -> dict:
        raise RuntimeError("fallo raro")

    monkeypatch.setitem(modulo_biblioteca.DETALLES, "gail", falla)
    elementos = {e["id"]: e for e in biblioteca.listar()}
    assert elementos[bueno]["valido"] is True
    assert not elementos[roto]["valido"] and "fallo raro" in elementos[roto]["motivo"]


def test_fichas_con_descripcion_o_artefacto_que_no_son_texto() -> None:
    for campo in ("descripcion", "artefacto"):
        with pytest.raises(ValueError, match=f"{campo} debe ser un texto"):
            crear_ficha("0" * 12, "cartpole", "bc", "BC", "dashboard", **{campo: 5})
        crear_ficha("0" * 12, "cartpole", "bc", "BC", "dashboard", **{campo: None})


def test_unas_demostraciones_sin_pares_no_son_validas(biblioteca, monkeypatch) -> None:
    from src.data import demostraciones

    id_ = nuevo(biblioteca.raiz, "demostraciones")
    original = demostraciones.resumen_demostraciones
    monkeypatch.setattr(demostraciones, "resumen_demostraciones",
                        lambda ruta: {**original(ruta), "pares": 0})
    elemento = biblioteca.obtener(id_)
    assert not elemento["valido"] and "no tiene pares" in elemento["motivo"]


def test_no_se_guarda_un_elemento_de_un_entorno_desconocido(biblioteca) -> None:
    """Regresión: agregar_nuevo aceptaba «cartpoel», escribía la carpeta y el índice, que
    solo mira los entornos conocidos, no la mostraba nunca."""
    with pytest.raises(ValueError, match="entorno no válido"):
        agregar_nuevo(biblioteca.raiz, "cartpoel", "experto", "E", "dashboard",
                      lambda c: escribir_archivos(c, "experto"))
    assert not biblioteca.raiz.exists() or not (biblioteca.raiz / "cartpoel").exists()


def test_demostraciones_de_otro_entorno(biblioteca) -> None:
    ficha = crear_ficha(nuevo_id(), "reacher", "demostraciones", "D", "dashboard")
    agregar(biblioteca.raiz, ficha, lambda c: escribir_archivos(c, "demostraciones"))
    elemento, = biblioteca.listar()
    assert not elemento["valido"] and "no de Reacher-v5" in elemento["motivo"]


def test_se_ignoran_las_carpetas_ocultas_y_los_entornos_y_tipos_desconocidos(biblioteca) -> None:
    id_ = nuevo(biblioteca.raiz, "bc")
    (biblioteca.raiz / "cartpole" / "bc" / ".temporal.tmp").mkdir()
    (biblioteca.raiz / "pong" / "bc" / ("a" * 12)).mkdir(parents=True)
    (biblioteca.raiz / "cartpole" / "ppo" / ("b" * 12)).mkdir(parents=True)
    (biblioteca.raiz / "cartpole" / "bc" / "suelto.txt").write_text("x", encoding="utf-8")
    assert [e["id"] for e in biblioteca.listar()] == [id_]


def test_obtener_y_ruta(biblioteca) -> None:
    id_ = nuevo(biblioteca.raiz, "gail", entorno="reacher")
    assert biblioteca.obtener(id_)["entorno"] == "reacher"
    for malo in ("no-existe", "0" * 12, "../" + id_):
        with pytest.raises(KeyError):
            biblioteca.ruta(malo)
    # Un id con «..» no puede salir de la carpeta de su tipo hacia la de otro.
    with pytest.raises(KeyError):
        biblioteca.ruta(f"../gail/{id_}")
    with pytest.raises(KeyError):
        biblioteca.obtener(f"../../reacher/gail/{id_}")


def test_experto_por_defecto(biblioteca) -> None:
    assert biblioteca.experto_por_defecto("cartpole") is None
    nuevo(biblioteca.raiz, "experto", origen="dashboard")
    assert biblioteca.experto_por_defecto("cartpole") is None
    roto = nuevo(biblioteca.raiz, "experto", origen="congelado",
                 creado="2025-01-01T00:00:00+00:00")
    (biblioteca.ruta(roto) / "modelo.zip").unlink()
    bueno = nuevo(biblioteca.raiz, "experto", origen="congelado")
    assert biblioteca.experto_por_defecto("cartpole")["id"] == bueno
    assert biblioteca.experto_por_defecto("reacher") is None


def test_resumen_de_demostraciones(tmp_path) -> None:
    escribir_archivos(tmp_path, "demostraciones")
    resumen = resumen_demostraciones(tmp_path / "demostraciones.npz")
    assert (resumen["partidas"], resumen["pares"]) == (2, 8)
    assert resumen["metadatos"]["entorno"] == "CartPole-v1"
    with pytest.raises(ValueError, match="No se pudo leer"):
        resumen_demostraciones(tmp_path / "no-existe.npz")


# --- Precarga (sin W&B) --------------------------------------------------------------------

def _cfg(entorno: str = "cartpole") -> dict[str, Any]:
    """Configuración mínima que usa la precarga."""
    return {"origen": {"entorno": entorno}, "entorno": {"id": ENTORNOS[entorno]},
            "datos": {"artefacto_experto": "experto-a", "artefacto_demostraciones": "demos-a",
                      "huella_partidas": "h", "huella_experto": "h"}}


@pytest.fixture
def congelados_falsos(monkeypatch) -> list[tuple[Path, Path]]:
    """Sustituye la comprobación de huellas por una que solo anota lo que comprueba."""
    comprobados = []
    monkeypatch.setattr(precarga, "comprobar_datos_congelados",
                        lambda cfg, datos, experto: comprobados.append((datos, experto)))
    return comprobados


def _descargar(nombre: str, carpeta: Path) -> Path:
    """Descarga falsa de un artefacto congelado."""
    if nombre.startswith("experto"):
        escribir_archivos(carpeta, "experto")
    else:
        escribir_archivos(carpeta, "demostraciones")
        (carpeta / "splits.json").write_text("{}", encoding="utf-8")
    return carpeta


def test_precargar_congelados_y_repetirlo_no_duplica(biblioteca, congelados_falsos) -> None:
    lineas = []
    anadidos = precarga.precargar_congelados(_cfg(), biblioteca, _descargar, lineas.append)
    assert len(anadidos) == 2 and len(congelados_falsos) == 1
    elementos = {e["tipo"]: e for e in biblioteca.listar()}
    assert {e["origen"] for e in elementos.values()} == {"congelado"}
    assert elementos["experto"]["artefacto"] == "experto-a"
    assert elementos["demostraciones"]["detalles"]["splits"] is True
    assert elementos["demostraciones"]["fuente"] == "experto"
    assert precarga.precargar_congelados(_cfg(), biblioteca, _descargar, lineas.append) == []
    assert len(biblioteca.listar()) == 2
    assert "ya están" in lineas[-1]


def test_precargar_congelados_si_no_valen_no_anade_nada(biblioteca, monkeypatch) -> None:
    def falla(cfg, datos, experto):
        raise ValueError("la huella de las partidas no coincide")

    monkeypatch.setattr(precarga, "comprobar_datos_congelados", falla)
    with pytest.raises(ValueError, match="no coincide"):
        precarga.precargar_congelados(_cfg(), biblioteca, _descargar, lambda _: None)
    assert biblioteca.listar() == []
    # Sin temporales: solo queda el cerrojo de las precargas.
    assert [p.name for p in biblioteca.raiz.iterdir()] == [precarga.ARCHIVO_CERROJO]


def test_precargar_congelados_con_el_perfil_pendiente(biblioteca) -> None:
    cfg = _cfg()
    cfg["datos"]["artefacto_experto"] = "pendiente"
    with pytest.raises(ValueError, match="pendiente"):
        precarga.precargar_congelados(cfg, biblioteca, _descargar, lambda _: None)


class ArtefactoFalso:
    """Lo que usa la precarga de un artefacto de W&B.

    ``huella`` es la que anuncia el artefacto en sus metadatos; ``interna``, la que
    guarda el propio modelo (por defecto, la misma).
    """

    def __init__(self, nombre: str, tipo: str, huella: str, cantidad: int = 4,
                 repeticion: int = 0, pesos: int = 1, interna: str | None = None):
        self.name, self.type = nombre, tipo
        self.metadata = {"huella_dataset": huella, "cantidad": cantidad,
                         "repeticion": repeticion}
        self.pesos = pesos
        self.interna = huella if interna is None else interna
        self.descargas = 0

    def download(self, root: str) -> str:
        self.descargas += 1
        carpeta = Path(root)
        if self.type == "gail":
            escribir_archivos(carpeta, "gail")
            datos = json.loads((carpeta / "gail.json").read_text(encoding="utf-8"))
            datos["datos"]["huella_dataset"] = self.interna
            _json(carpeta, "gail.json", datos)
        else:
            for i in range(self.pesos):
                escribir_archivos(carpeta, "bc")
                carga = torch.load(carpeta / "politica.pt", weights_only=True)
                carga["metadatos"]["huella_dataset"] = self.interna
                torch.save(carga, carpeta / f"repeticion_{i}.pt")
                (carpeta / "politica.pt").unlink()
        return str(carpeta)


def test_precargar_entrega(biblioteca, congelados_falsos) -> None:
    precarga.precargar_congelados(_cfg(), biblioteca, _descargar, lambda _: None)
    huella = precarga.huella_demostraciones_congeladas(_cfg(), biblioteca)
    lineas = []
    artefactos = [ArtefactoFalso("bc-a:v0", "bc", huella, 7, 2),
                  ArtefactoFalso("gail-a:v0", "gail", huella, 4, 1),
                  ArtefactoFalso("gail-otros-datos:v0", "gail", "b" * 64),
                  ArtefactoFalso("bc-dos-pt:v0", "bc", huella, pesos=2)]
    anadidos = precarga.precargar_entrega(_cfg(), biblioteca, artefactos, lineas.append)
    assert len(anadidos) == 2
    nuevos = {e["artefacto"]: e for e in biblioteca.listar(tipos=("bc", "gail"))}
    assert set(nuevos) == {"bc-a:v0", "gail-a:v0"}
    assert all(e["valido"] and e["origen"] == "entrega" for e in nuevos.values())
    assert nuevos["bc-a:v0"]["nombre"] == "BC · 7 partidas · repetición 2"
    assert any("no se entrenó con las demostraciones congeladas" in l for l in lineas)
    assert any("no tiene un único .pt" in l for l in lineas)
    # Repetir no duplica.
    assert precarga.precargar_entrega(_cfg(), biblioteca, artefactos[:2], lineas.append) == []


def test_precargar_entrega_sin_congelados_falla(biblioteca) -> None:
    with pytest.raises(ValueError, match="faltan las demostraciones congeladas"):
        precarga.precargar_entrega(_cfg(), biblioteca, [], lambda _: None)


def test_un_modelo_cuya_huella_interna_no_coincide_se_omite(biblioteca,
                                                           congelados_falsos) -> None:
    """Regresión: bastaba con la huella que anuncia el artefacto en W&B; ahora también
    tiene que coincidir la que guarda el propio modelo."""
    precarga.precargar_congelados(_cfg(), biblioteca, _descargar, lambda _: None)
    huella = precarga.huella_demostraciones_congeladas(_cfg(), biblioteca)
    lineas = []
    artefactos = [ArtefactoFalso("gail-x:v0", "gail", huella, interna="b" * 64),
                  ArtefactoFalso("bc-x:v0", "bc", huella, interna="c" * 64),
                  ArtefactoFalso("bc-bueno:v0", "bc", huella)]
    anadidos = precarga.precargar_entrega(_cfg(), biblioteca, artefactos, lineas.append)
    assert [e["artefacto"] for e in biblioteca.listar(tipos=("bc", "gail"))] == ["bc-bueno:v0"]
    assert len(anadidos) == 1
    assert sum("no con las demostraciones congeladas" in l for l in lineas) == 2
    # Los omitidos liberan su id: quedan los 2 congelados y el BC bueno.
    assert len(list((biblioteca.raiz / CARPETA_IDS).iterdir())) == 3


def test_la_huella_es_la_del_congelado_del_perfil(biblioteca, congelados_falsos) -> None:
    """Regresión: con dos congelados del mismo entorno se usaba el más antiguo, aunque el
    perfil citara el otro, y se descartaban los modelos buenos."""
    viejo = _cfg()
    viejo["datos"]["artefacto_demostraciones"] = "demos-viejo"
    precarga.precargar_congelados(viejo, biblioteca, _descargar, lambda _: None)
    # El congelado actual del perfil, con otro NPZ (otra huella).
    def descargar_actual(nombre: str, carpeta: Path) -> Path:
        _descargar(nombre, carpeta)
        if not nombre.startswith("experto"):
            guardar = carpeta / "demostraciones.npz"
            guardar.unlink()
            guardar_demostraciones(guardar, [_trayectoria(0, 9)],
                                   {"entorno": "CartPole-v1", "experto": "y", "procedencia": {},
                                    "configuracion_experto": {}},
                                   spaces.Box(-np.inf, np.inf, (4,), np.float32),
                                   spaces.Discrete(2))
        return carpeta
    precarga.precargar_congelados(_cfg(), biblioteca, descargar_actual, lambda _: None)
    huella_vieja = precarga.huella_demostraciones_congeladas(viejo, biblioteca)
    huella = precarga.huella_demostraciones_congeladas(_cfg(), biblioteca)
    assert huella and huella_vieja and huella != huella_vieja
    anadidos = precarga.precargar_entrega(_cfg(), biblioteca,
                                          [ArtefactoFalso("bc-a:v0", "bc", huella)],
                                          lambda _: None)
    assert len(anadidos) == 1


def _huellas_falsas(cfg: dict, datos: Path, experto: Path) -> None:
    """Como comprobar_datos_congelados, con lo que miran sus huellas: un modelo.zip con
    contenido y unos splits.json que existan."""
    if not (experto / "modelo.zip").is_file() or not (experto / "modelo.zip").stat().st_size:
        raise ValueError("la huella del experto no coincide")
    if not (datos / "splits.json").is_file():
        raise ValueError("no están los splits")


@pytest.mark.parametrize("estropear, por_huellas", [
    (lambda e, d: (e / "modelo.zip").unlink(), False),  # el índice ya lo da por no válido
    (lambda e, d: (e / "modelo.zip").write_bytes(b""), True),  # existe, pero vacío
    (lambda e, d: (d / "splits.json").unlink(), True),  # opcional para el índice
], ids=["sin_modelo", "modelo_vacio", "sin_splits"])
def test_unos_congelados_estropeados_se_sustituyen(biblioteca, monkeypatch, estropear,
                                                   por_huellas) -> None:
    """Regresión: la precarga solo miraba que existieran los archivos; con modelo.zip
    vacío o sin splits.json decía «ya están». Ahora comprueba sus huellas y, si no pasan,
    borra los dos y los vuelve a descargar."""
    monkeypatch.setattr(precarga, "comprobar_datos_congelados", _huellas_falsas)
    precarga.precargar_congelados(_cfg(), biblioteca, _descargar, lambda _: None)
    antes = {e["tipo"]: e["id"] for e in biblioteca.listar()}
    estropear(biblioteca.ruta(antes["experto"]), biblioteca.ruta(antes["demostraciones"]))
    lineas = []
    anadidos = precarga.precargar_congelados(_cfg(), biblioteca, _descargar, lineas.append)
    assert len(anadidos) == 2
    despues = {e["tipo"]: e for e in biblioteca.listar()}
    assert set(despues) == {"experto", "demostraciones"}  # sin restos de los estropeados
    assert all(e["valido"] and e["id"] not in antes.values() for e in despues.values())
    assert any("no pasan sus huellas" in l for l in lineas) is por_huellas
    # Y ya íntegros, repetir no descarga nada.
    assert precarga.precargar_congelados(_cfg(), biblioteca, _descargar, lambda _: None) == []


def test_quitar_un_elemento_libera_su_id(biblioteca) -> None:
    id_ = nuevo(biblioteca.raiz, "experto")
    biblioteca.quitar(id_)
    assert biblioteca.listar() == []
    assert not (biblioteca.raiz / CARPETA_IDS / id_).exists()
    with pytest.raises(KeyError):
        biblioteca.quitar(id_)


def test_un_fallo_al_descargar_un_modelo_no_impide_los_demas(biblioteca,
                                                            congelados_falsos) -> None:
    """Regresión: un CommError de W&B al descargar un BC salía del script y dejaba sin
    procesar lo que quedaba."""
    class CommError(Exception):
        pass

    class Roto(ArtefactoFalso):
        def download(self, root: str) -> str:
            raise CommError("red caída")

    precarga.precargar_congelados(_cfg(), biblioteca, _descargar, lambda _: None)
    huella = precarga.huella_demostraciones_congeladas(_cfg(), biblioteca)
    lineas = []
    with pytest.raises(RuntimeError, match="no se pudieron descargar bc-roto:v0"):
        precarga.precargar_entrega(_cfg(), biblioteca,
                                   [Roto("bc-roto:v0", "bc", huella),
                                    ArtefactoFalso("gail-a:v0", "gail", huella)], lineas.append)
    assert [e["artefacto"] for e in biblioteca.listar(tipos=("bc", "gail"))] == ["gail-a:v0"]
    assert any("no se pudo descargar bc-roto:v0: red caída" in l for l in lineas)


def test_el_mismo_artefacto_dos_veces_se_anade_una(biblioteca, congelados_falsos) -> None:
    """Regresión: recibir el mismo artefacto dos veces en una llamada creaba dos elementos."""
    precarga.precargar_congelados(_cfg(), biblioteca, _descargar, lambda _: None)
    huella = precarga.huella_demostraciones_congeladas(_cfg(), biblioteca)
    dos = [ArtefactoFalso("bc-a:v0", "bc", huella), ArtefactoFalso("bc-a:v0", "bc", huella)]
    assert len(precarga.precargar_entrega(_cfg(), biblioteca, dos, lambda _: None)) == 1
    assert dos[1].descargas == 0


def test_dos_precargas_a_la_vez_no_duplican(biblioteca, congelados_falsos) -> None:
    """Regresión: dos precargas simultáneas creaban cuatro fichas para dos artefactos."""
    import threading
    import time

    def lenta(nombre: str, carpeta: Path) -> Path:
        time.sleep(0.3)
        return _descargar(nombre, carpeta)

    errores = []

    def precargar() -> None:
        try:
            precarga.precargar_congelados(_cfg(), biblioteca, lenta, lambda _: None)
        except Exception as error:  # noqa: BLE001
            errores.append(error)

    hilos = [threading.Thread(target=precargar) for _ in range(2)]
    for hilo in hilos:
        hilo.start()
    for hilo in hilos:
        hilo.join(30)
    assert errores == []
    assert sorted(e["tipo"] for e in biblioteca.listar()) == ["demostraciones", "experto"]


# --- Rutas ---------------------------------------------------------------------------------

@pytest.fixture
def cliente(tmp_path: Path) -> Iterator[tuple[TestClient, Path]]:
    """API con una biblioteca en una carpeta temporal."""
    from src.app.api import crear_app
    from src.app.trabajos import GestorTrabajos

    configs = tmp_path / "configs"
    shutil.copytree(Path(__file__).resolve().parents[1] / "configs", configs)
    base = configs / "default.yaml"
    texto = base.read_text(encoding="utf-8").replace(
        "  biblioteca: biblioteca", f"  biblioteca: {tmp_path / 'biblioteca'}")
    base.write_text(texto, encoding="utf-8")
    gestor = GestorTrabajos(1)
    with TestClient(crear_app(base, gestor)) as c:
        yield c, tmp_path / "biblioteca"
    gestor.cerrar()


def test_rutas_de_la_biblioteca(cliente) -> None:
    c, raiz = cliente
    assert c.get("/api/biblioteca/modelos").json() == []
    experto = nuevo(raiz, "experto", origen="congelado")
    bc = nuevo(raiz, "bc", entorno="reacher")
    demos = nuevo(raiz, "demostraciones", fuente="humano")
    modelos = c.get("/api/biblioteca/modelos").json()
    assert [m["id"] for m in modelos] == [experto, bc]
    assert [m["id"] for m in c.get("/api/biblioteca/modelos",
                                    params={"tipo": "bc"}).json()] == [bc]
    assert [m["id"] for m in c.get("/api/biblioteca/modelos",
                                    params={"entorno": "cartpole"}).json()] == [experto]
    lista = c.get("/api/biblioteca/demostraciones", params={"entorno": "cartpole"}).json()
    assert lista == [{"id": demos, "entorno": "cartpole", "nombre": "demostraciones de prueba",
                      "tipo": "conjunto", "partidas": 2, "pares": 8, "fuente": "humano",
                      "origen": "dashboard", "valido": True, "motivo": None}]
    uno = c.get(f"/api/biblioteca/{experto}").json()
    assert uno["tipo"] == "experto" and uno["detalles"]["mejor_retorno"] == 500.0
    assert c.get("/api/biblioteca/no-existe").status_code == 404
    assert c.get("/api/biblioteca/modelos", params={"tipo": "demostraciones"}).status_code == 422
    assert c.get("/api/biblioteca/modelos", params={"entorno": "pong"}).status_code == 422


def test_una_ficha_rara_no_tumba_los_listados(cliente, monkeypatch) -> None:
    """Regresión: una ficha con descripción numérica (escrita a mano) o un manifiesto sin
    pares daban 500 en el listado entero. Ahora ese elemento sale como no válido."""
    from src.data import demostraciones

    c, raiz = cliente
    sano = nuevo(raiz, "experto")
    raro = nuevo(raiz, "bc")
    ficha = Biblioteca(raiz, ENTORNOS).ruta(raro) / ARCHIVO_FICHA
    ficha.write_text(json.dumps({**json.loads(ficha.read_text(encoding="utf-8")),
                                 "descripcion": 5}), encoding="utf-8")
    respuesta = c.get("/api/biblioteca/modelos")
    assert respuesta.status_code == 200
    elementos = {e["id"]: e for e in respuesta.json()}
    assert elementos[sano]["valido"] and not elementos[raro]["valido"]
    assert "descripcion" in elementos[raro]["motivo"]
    assert c.get(f"/api/biblioteca/{raro}").status_code == 200

    demos = nuevo(raiz, "demostraciones")
    original = demostraciones.resumen_demostraciones
    monkeypatch.setattr(demostraciones, "resumen_demostraciones",
                        lambda ruta: {**original(ruta), "pares": 0})
    respuesta = c.get("/api/biblioteca/demostraciones")
    assert respuesta.status_code == 200
    assert respuesta.json()[0]["id"] == demos and not respuesta.json()[0]["valido"]


@pytest.mark.parametrize("ruta", ["/api/biblioteca/modelos", "/api/biblioteca/demostraciones",
                                  "/api/biblioteca/{id}"])
def test_las_rutas_no_dan_500_aunque_el_indice_devuelva_algo_raro(cliente, monkeypatch,
                                                                  ruta) -> None:
    """Las tres rutas pasan cada elemento por su modelo de respuesta: aunque el índice
    devolviera algo que no encaja (un caso que hoy ya filtra la ficha), sale no válido."""
    c, raiz = cliente
    tipo = "demostraciones" if "demostraciones" in ruta else "bc"
    id_ = nuevo(raiz, tipo)
    leer = Biblioteca._leer

    def raro(self, carpeta, entorno, tipo_carpeta):
        elemento = leer(self, carpeta, entorno, tipo_carpeta)
        return {**elemento, "nombre": 3, "detalles": {**elemento["detalles"], "partidas": 0}}

    monkeypatch.setattr(Biblioteca, "_leer", raro)
    respuesta = c.get(ruta.format(id=id_))
    assert respuesta.status_code == 200
    datos = respuesta.json()
    elemento = datos[0] if isinstance(datos, list) else datos
    assert elemento["id"] == id_ and elemento["valido"] is False


def test_como_respuesta_convierte_lo_que_no_encaja_en_no_valido() -> None:
    from src.app.contrato import DemostracionApi, ElementoBibliotecaApi
    from src.app.rutas_biblioteca import como_respuesta

    raro = {"id": "a" * 12, "entorno": "cartpole", "tipo": "bc", "nombre": 3, "origen": "x",
            "creado": None, "descripcion": None, "fuente": None, "artefacto": None,
            "valido": True, "motivo": None, "detalles": {"a": 1}}
    salida = como_respuesta(ElementoBibliotecaApi, raro)
    assert salida["valido"] is False and salida["nombre"] == "a" * 12
    assert salida["detalles"] == {} and "nombre" in salida["motivo"]
    demos = {"id": "b" * 12, "entorno": "cartpole", "nombre": "d", "tipo": "conjunto",
             "partidas": 0, "pares": 3, "fuente": None, "origen": None, "valido": True,
             "motivo": None}
    salida = como_respuesta(DemostracionApi, demos)
    assert salida["valido"] is False and salida["partidas"] is None
    bueno = {**demos, "partidas": 2}
    assert como_respuesta(DemostracionApi, bueno) == {**bueno, "entorno": "cartpole"}


def test_un_elemento_roto_sale_por_la_api_como_no_valido(cliente) -> None:
    c, raiz = cliente
    demos = nuevo(raiz, "demostraciones")
    (Biblioteca(raiz, ENTORNOS).ruta(demos) / "demostraciones.npz").write_bytes(b"x")
    roto, = c.get("/api/biblioteca/demostraciones").json()
    assert roto["valido"] is False and roto["partidas"] is None and roto["motivo"]


def test_constantes() -> None:
    assert set(ORIGENES) == {"congelado", "entrega", "dashboard"}
    assert TIPOS == ("experto", "bc", "gail", "demostraciones")


# --- Script de precarga ----------------------------------------------------------------

def _script():
    """Importa scripts/preparar_biblioteca.py como módulo."""
    import importlib.util

    ruta = Path(__file__).resolve().parents[1] / "scripts" / "preparar_biblioteca.py"
    especificacion = importlib.util.spec_from_file_location("preparar_biblioteca", ruta)
    modulo = importlib.util.module_from_spec(especificacion)
    especificacion.loader.exec_module(modulo)
    return modulo


def test_script_precarga_cada_entorno_y_sigue_si_uno_falla(tmp_path, monkeypatch,
                                                            capsys) -> None:
    script = _script()
    hechos = []

    def congelados(cfg, biblioteca):
        if cfg["origen"]["entorno"] == "reacher":
            raise RuntimeError("sin red")
        hechos.append(("congelados", cfg["origen"]["entorno"]))

    monkeypatch.setattr(script, "precargar_congelados", congelados)
    monkeypatch.setattr(script, "artefactos_de_comparacion", lambda cfg, id_run: [id_run])
    monkeypatch.setattr(script, "precargar_entrega",
                        lambda cfg, b, arts: hechos.append(("entrega", cfg["origen"]["entorno"],
                                                            arts)))
    config = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    codigo = script.main(["--config", str(config), "--entorno", "cartpole",
                          "--entorno", "reacher"])
    assert codigo == 1
    assert hechos == [("congelados", "cartpole"), ("entrega", "cartpole", ["ft49odps"])]
    assert "error en reacher: sin red" in capsys.readouterr().err
    hechos.clear()
    assert script.main(["--config", str(config), "--entorno", "cartpole", "--sin-entrega"]) == 0
    assert hechos == [("congelados", "cartpole")]


def test_script_sigue_con_los_demas_entornos_ante_cualquier_error(monkeypatch, capsys) -> None:
    """Regresión: un error que no era ValueError, RuntimeError ni OSError (como el
    CommError de W&B) salía del script y los entornos siguientes no se procesaban."""
    script = _script()
    hechos = []

    class CommError(Exception):
        pass

    def congelados(cfg, biblioteca):
        if cfg["origen"]["entorno"] == "cartpole":
            raise CommError("red caída")
        hechos.append(cfg["origen"]["entorno"])

    monkeypatch.setattr(script, "precargar_congelados", congelados)
    config = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    assert script.main(["--config", str(config), "--entorno", "cartpole", "--entorno",
                        "reacher", "--sin-entrega"]) == 1
    assert hechos == ["reacher"]
    assert "error en cartpole: red caída" in capsys.readouterr().err


def test_script_con_config_que_no_vale_sale_con_2(tmp_path, capsys) -> None:
    assert _script().main(["--config", str(tmp_path / "no-existe.yaml")]) == 2
    assert "configuración no válida" in capsys.readouterr().err
