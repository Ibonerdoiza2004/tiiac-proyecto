"""Tests de las comparaciones en la API: el contrato (ComparacionCrear y lo que manda el
trabajo) y la ruta POST /api/comparaciones, de punta a punta con el gestor de trabajos y
modelos pequeños de verdad (entrenados con los trabajadores, con los tamaños del smoke).
"""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient
from pydantic import ValidationError
from test_rutas_laboratorio import CONFIGS, elemento
from test_rutas_trabajos import cerrar_y_esperar, es_estado, leer_hasta, leer_hasta_cierre
from test_trabajadores import CanalFalso, _argumentos, _argumentos_imitacion, _conjunto

from src.app.api import crear_app
from src.app.contrato import (ArgumentosComparacionApi, ComparacionCrear,
                              ProgresoComparacionApi, ResultadoComparacion)
from src.app.rutas_comparar import TAREA_COMPARAR
from src.app.trabajos import GestorTrabajos
from src.training.trabajadores import (tarea_comparar, tarea_entrenar_bc,
                                       tarea_entrenar_experto, tarea_entrenar_gail)

ID = "a" * 12


# --- ComparacionCrear ------------------------------------------------------------------------

@pytest.mark.parametrize("cuerpo", [
    {"entorno": "cartpole", "aleatorio": True},
    {"entorno": "reacher", "modelos": [ID]},
    {"entorno": "acrobot", "modelos": [ID, "b" * 12], "aleatorio": True, "partidas": 1000,
     "vista": True},
    {"entorno": "halfcheetah", "modelos": [ID], "partidas": 1, "aleatorio": False},
])
def test_peticiones_que_valen(cuerpo):
    peticion = ComparacionCrear.model_validate(cuerpo)
    assert peticion.entorno.value == cuerpo["entorno"]
    assert peticion.modelos == cuerpo.get("modelos", [])
    assert peticion.aleatorio is cuerpo.get("aleatorio", False)
    assert peticion.partidas == cuerpo.get("partidas")
    assert peticion.vista is cuerpo.get("vista", False)


@pytest.mark.parametrize("cambio, mensaje", [
    ({"modelos": [ID, ID]}, "repetidos"),
    ({"modelos": [], "aleatorio": False}, "nada que comparar"),
    ({"modelos": [ID], "extra": 1}, "Extra inputs"),
    ({"entorno": "pong"}, "entorno"),
    ({"modelos": [""]}, "at least 1 character"),
    ({"modelos": ID}, "valid list"),
    ({"partidas": 0}, "greater than or equal to 1"),
    ({"partidas": -3}, "greater than or equal to 1"),
    ({"partidas": True}, "valid integer"),
    ({"partidas": "3"}, "valid integer"),
    ({"partidas": 2.0}, "valid integer"),
    ({"aleatorio": "true"}, "valid boolean"),
    ({"aleatorio": 1}, "valid boolean"),
    ({"vista": 1}, "valid boolean"),
])
def test_peticiones_que_no_valen(cambio, mensaje):
    cuerpo = {"entorno": "cartpole", "modelos": [ID], **cambio}
    with pytest.raises(ValidationError, match=mensaje):
        ComparacionCrear.model_validate(cuerpo)


def test_sin_entorno_no_vale():
    with pytest.raises(ValidationError, match="entorno"):
        ComparacionCrear.model_validate({"aleatorio": True})


# --- lo que manda el trabajo ----------------------------------------------------------------

def test_argumentos_publicos_con_partidas_y_semilla_efectivas():
    argumentos = {"entorno": "cartpole", "modelos": [ID], "aleatorio": True, "partidas": 50,
                  "semilla": 9000, "vista": False}
    validados = ArgumentosComparacionApi.model_validate(argumentos)
    assert validados.model_dump(mode="json") == argumentos
    for falta in ("partidas", "semilla"):
        with pytest.raises(ValidationError, match=falta):
            ArgumentosComparacionApi.model_validate(
                {k: v for k, v in argumentos.items() if k != falta})


@pytest.mark.parametrize("medidas, vale", [
    ({"modelo": ID, "partida": 0, "retorno": 9.5, "pasos": 10}, True),
    ({"modelo": "aleatorio", "partida": 3, "retorno": -200.0, "pasos": 200}, True),
    ({"modelo": ID, "partida": -1, "retorno": 1.0, "pasos": 1}, False),
    ({"modelo": ID, "partida": 0, "retorno": 1.0, "pasos": 0}, False),
    ({"modelo": ID, "partida": 0, "pasos": 1}, False),
])
def test_progreso_de_una_comparacion(medidas, vale):
    if vale:
        assert ProgresoComparacionApi.model_validate(medidas).modelo == medidas["modelo"]
    else:
        with pytest.raises(ValidationError):
            ProgresoComparacionApi.model_validate(medidas)


# --- la ruta, de punta a punta -----------------------------------------------------------------

LIMITE = 120.0


@pytest.fixture(scope="module")
def biblioteca(tmp_path_factory) -> tuple[Path, dict[str, str]]:
    """Biblioteca de CartPole con un experto, un BC y un GAIL pequeños y sus datos."""
    base = tmp_path_factory.mktemp("comparar")
    raiz = base / "biblioteca"
    ids = {"experto": tarea_entrenar_experto(_argumentos(base), CanalFalso())["modelo"]["id"]}
    ids["demostraciones"] = _conjunto(raiz)
    for metodo, tarea in (("bc", tarea_entrenar_bc), ("gail", tarea_entrenar_gail)):
        argumentos = _argumentos_imitacion(base, [ids["demostraciones"]])
        ids[metodo] = tarea(argumentos, CanalFalso())["modelo"]["id"]
    return raiz, ids


def _configs(tmp_path: Path, raiz: Path) -> Path:
    """Copia configs/ con la biblioteca en raiz; devuelve el YAML base."""
    copia = tmp_path / "configs"
    shutil.copytree(CONFIGS, copia)
    base = copia / "default.yaml"
    cfg = yaml.safe_load(base.read_text(encoding="utf-8"))
    cfg["aplicativo"]["biblioteca"] = str(raiz)
    base.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return base


@pytest.fixture
def app(tmp_path, biblioteca) -> Iterator[tuple[TestClient, GestorTrabajos, Path]]:
    """Cliente, gestor de un hueco y ruta de la configuración de la aplicación."""
    gestor = GestorTrabajos(1)
    ruta = _configs(tmp_path, biblioteca[0])
    with TestClient(crear_app(ruta, gestor)) as cliente:
        yield cliente, gestor, ruta
    gestor.cerrar()


def _seguir(cliente: TestClient, id_trabajo: str) -> list:
    """Sigue el trabajo por el WebSocket hasta que el servidor cierra."""
    with cliente.websocket_connect(f"/api/trabajos/{id_trabajo}") as ws:
        mensajes, codigo = leer_hasta_cierre(ws)
    assert codigo == 1000
    return mensajes


def test_comparar_de_punta_a_punta(app, biblioteca):
    cliente, gestor, ruta = app
    raiz, ids = biblioteca
    modelos = [ids["gail"], ids["experto"], ids["bc"]]
    respuesta = cliente.post("/api/comparaciones", json={
        "entorno": "cartpole", "modelos": modelos, "aleatorio": True, "partidas": 2})
    assert respuesta.status_code == 201, respuesta.json()
    trabajo = respuesta.json()
    assert trabajo["tipo"] == "comparacion" and trabajo["interactivo"] is False
    # Los argumentos públicos, con la semilla de evaluacion; nada de rutas del servidor.
    assert trabajo["argumentos"] == {"entorno": "cartpole", "modelos": modelos,
                                     "aleatorio": True, "partidas": 2, "semilla": 9000,
                                     "vista": False}
    mensajes = _seguir(cliente, trabajo["id"])
    progresos = [m["medidas"] for m in mensajes if isinstance(m, dict) and m["tipo"] == "progreso"]
    historial = next(m for m in mensajes if isinstance(m, dict) and m["tipo"] == "historial")
    todos = historial["progreso"] + progresos
    assert [(p["partida"], p["modelo"]) for p in todos] == [
        (i, m) for i in range(2) for m in [*modelos, "aleatorio"]]
    for medidas in todos:
        ProgresoComparacionApi.model_validate(medidas)
    final = mensajes[-1]
    assert final["tipo"] == "estado" and final["trabajo"]["estado"] == "terminado"
    resultado = ResultadoComparacion.model_validate(final["trabajo"]["resultado"])
    assert resultado.partidas == 2 and not resultado.incompleta and resultado.semilla == 9000
    # El mismo resultado que el trabajador con los mismos argumentos.
    directo = tarea_comparar({"ruta_config": str(ruta), "biblioteca": str(raiz),
                              "calidad_jpeg": 80, "entorno": "cartpole", "modelos": modelos,
                              "aleatorio": True, "partidas": 2, "semilla": 9000,
                              "vista": False}, CanalFalso())
    assert final["trabajo"]["resultado"] == directo
    # Aparece en la lista de comparaciones con sus argumentos públicos.
    lista = cliente.get("/api/trabajos", params={"tipo": "comparacion"}).json()
    assert [t["argumentos"] for t in lista] == [trabajo["argumentos"]]


def test_sin_partidas_usa_las_de_la_evaluacion(app):
    cliente, gestor, _ = app
    respuesta = cliente.post("/api/comparaciones", json={"entorno": "cartpole",
                                                         "aleatorio": True})
    assert respuesta.status_code == 201
    argumentos = respuesta.json()["argumentos"]
    assert argumentos["partidas"] == 50 and argumentos["semilla"] == 9000
    final = gestor.esperar(respuesta.json()["id"], LIMITE)
    assert final["resultado"]["partidas"] == 50


def test_vista_y_finalizar_por_el_websocket(app, biblioteca):
    cliente, gestor, _ = app
    _, ids = biblioteca
    respuesta = cliente.post("/api/comparaciones", json={
        "entorno": "cartpole", "modelos": [ids["experto"]], "aleatorio": True,
        "partidas": 1000, "vista": True})
    id_trabajo = respuesta.json()["id"]
    with cliente.websocket_connect(f"/api/trabajos/{id_trabajo}") as ws:
        # A velocidad real, 1000 partidas no acaban antes de finalizar.
        leer_hasta(ws, lambda m: isinstance(m, bytes))
        ws.send_json({"tipo": "vista", "activa": False})
        leer_hasta(ws, lambda m: isinstance(m, dict) and m.get("tipo") == "progreso")
        ws.send_json({"tipo": "finalizar"})
        mensajes = leer_hasta(ws, es_estado("terminado"))
        cerrar_y_esperar(ws, gestor)
    resultado = ResultadoComparacion.model_validate(mensajes[-1]["trabajo"]["resultado"])
    assert resultado.incompleta and resultado.partidas < 1000
    assert all(len(m.retornos) == resultado.partidas for m in resultado.modelos)


def test_una_comparacion_sin_hueco_espera_en_la_cola(app):
    cliente, gestor, _ = app
    ocupado = gestor.crear("entrenamiento", "tareas_app:contar_hasta_orden")
    respuesta = cliente.post("/api/comparaciones", json={"entorno": "cartpole",
                                                         "aleatorio": True, "partidas": 1})
    assert respuesta.status_code == 201 and respuesta.json()["estado"] == "en_cola"
    gestor.enviar_orden(ocupado["id"], {"tipo": "finalizar"})
    assert gestor.esperar(respuesta.json()["id"], LIMITE)["estado"] == "terminado"


def test_usa_la_evaluacion_del_perfil_y_la_vista_del_dashboard(tmp_path, biblioteca):
    """Con otros valores en configs/: partidas, semilla y calidad salen de ahí."""
    raiz, ids = biblioteca
    ruta = _configs(tmp_path, raiz)
    _perfil_con(ruta, episodios=3, semilla=77)
    dashboard = ruta.parent / "dashboard.yaml"
    dashboard.write_text(dashboard.read_text(encoding="utf-8").replace(
        "calidad_jpeg: 80", "calidad_jpeg: 30"), encoding="utf-8")
    gestor = GestorTrabajos(1)
    creados = []
    original = gestor.crear

    def espia(tipo, tarea, argumentos=None, interactivo=False, publicos=None):
        creados.append((tipo, tarea, argumentos, interactivo))
        return original(tipo, tarea, argumentos, interactivo, publicos)

    gestor.crear = espia
    try:
        with TestClient(crear_app(ruta, gestor)) as cliente:
            trabajo = cliente.post("/api/comparaciones", json={
                "entorno": "cartpole", "modelos": [ids["bc"]], "aleatorio": True}).json()
            final = gestor.esperar(trabajo["id"], LIMITE)
    finally:
        gestor.cerrar()
    assert trabajo["argumentos"]["partidas"] == 3 and trabajo["argumentos"]["semilla"] == 77
    assert final["resultado"]["partidas"] == 3 and final["resultado"]["semilla"] == 77
    # Lo que recibe el trabajador: las rutas del servidor, la calidad del dashboard y las
    # partidas ya resueltas (las mismas que ve el navegador).
    (tipo, tarea, argumentos, interactivo), = creados
    assert (tipo, tarea, interactivo) == ("comparacion", TAREA_COMPARAR, False)
    assert argumentos == {"ruta_config": str(ruta), "biblioteca": str(raiz), "calidad_jpeg": 30,
                          "entorno": "cartpole", "modelos": [ids["bc"]], "aleatorio": True,
                          "partidas": 3, "semilla": 77, "vista": False}


def _perfil_con(ruta: Path, **evaluacion) -> None:
    """Cambia la sección evaluacion del perfil de CartPole de la copia de configs/."""
    perfil = ruta.parent / "entornos" / "cartpole.yaml"
    datos = yaml.safe_load(perfil.read_text(encoding="utf-8"))
    datos["evaluacion"].update(evaluacion)
    perfil.write_text(yaml.safe_dump(datos, allow_unicode=True, sort_keys=False),
                      encoding="utf-8")


def test_unas_partidas_por_defecto_por_encima_del_maximo_dan_422(app):
    """Regresión: el máximo solo se comprobaba con las partidas pedidas."""
    cliente, gestor, ruta = app
    _perfil_con(ruta, episodios=5000)
    respuesta = cliente.post("/api/comparaciones", json={"entorno": "cartpole",
                                                         "aleatorio": True})
    assert respuesta.status_code == 422
    assert respuesta.json()["detail"] == ("Como mucho 1000 partidas, no 5000 (las de "
                                          "evaluacion del perfil)")
    assert gestor.listar() == []
    _perfil_con(ruta, episodios=1000)  # justo en el máximo, vale
    respuesta = cliente.post("/api/comparaciones", json={"entorno": "cartpole",
                                                         "aleatorio": True})
    assert respuesta.status_code == 201 and respuesta.json()["argumentos"]["partidas"] == 1000
    gestor.cancelar(respuesta.json()["id"])


def test_la_semilla_no_cambia_si_configs_cambia_en_la_cola(app):
    """Regresión: el trabajador volvía a leer la semilla de configs/ al arrancar."""
    cliente, gestor, ruta = app
    ocupado = gestor.crear("entrenamiento", "tareas_app:contar_hasta_orden")
    respuesta = cliente.post("/api/comparaciones", json={"entorno": "cartpole",
                                                         "aleatorio": True, "partidas": 2})
    assert respuesta.json()["estado"] == "en_cola"
    _perfil_con(ruta, semilla=1234, episodios=7)
    gestor.enviar_orden(ocupado["id"], {"tipo": "finalizar"})
    final = gestor.esperar(respuesta.json()["id"], LIMITE)
    assert final["argumentos"]["semilla"] == final["resultado"]["semilla"] == 9000
    assert final["resultado"]["partidas"] == 2


def test_mas_partidas_que_el_maximo_da_422(app):
    cliente, gestor, _ = app
    respuesta = cliente.post("/api/comparaciones", json={"entorno": "cartpole",
                                                         "aleatorio": True, "partidas": 1001})
    assert respuesta.status_code == 422
    assert respuesta.json()["detail"] == "Como mucho 1000 partidas, no 1001"
    assert gestor.listar() == []


def test_modelos_que_no_valen_dan_422(app, biblioteca):
    cliente, gestor, _ = app
    raiz, ids = biblioteca
    casos = [("0" * 12, "No hay ningún elemento"), ("no-es-un-id", "No hay ningún elemento"),
             (ids["demostraciones"], "es de tipo demostraciones, no experto, bc o gail"),
             (elemento(raiz, "experto", "mountaincar"), "es de mountaincar, no de cartpole"),
             (elemento(raiz, "bc", "cartpole", valido=False), "no es válido")]
    for id_modelo, motivo in casos:
        respuesta = cliente.post("/api/comparaciones", json={
            "entorno": "cartpole", "modelos": [ids["experto"], id_modelo]})
        assert respuesta.status_code == 422, id_modelo
        detalle = respuesta.json()["detail"]
        assert detalle.startswith("modelos: ") and motivo in detalle, detalle
    assert gestor.listar() == []


@pytest.mark.parametrize("cuerpo", [
    {"entorno": "cartpole"}, {"entorno": "cartpole", "modelos": [ID, ID]},
    {"entorno": "cartpole", "aleatorio": True, "semilla": 3},
    {"entorno": "pong", "aleatorio": True}, {"aleatorio": True},
])
def test_peticiones_con_mala_forma_dan_422(app, cuerpo):
    cliente, gestor, _ = app
    assert cliente.post("/api/comparaciones", json=cuerpo).status_code == 422
    assert gestor.listar() == []


def test_la_tarea_es_la_del_trabajador():
    modulo, funcion = TAREA_COMPARAR.split(":")
    assert (modulo, funcion) == ("src.training.trabajadores", tarea_comparar.__name__)
