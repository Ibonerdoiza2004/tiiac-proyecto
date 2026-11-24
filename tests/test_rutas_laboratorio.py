"""Tests del contrato (src/app/contrato.py) y de las rutas del Laboratorio
(src/app/rutas_laboratorio.py): configuración del juego, esquema de ajustes y crear
sesiones y entrenamientos (que validan y responden 501 hasta sus ramas -api).
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.app.ajustes import METODOS, cargar_dashboard, esquema_ajustes
from src.app.api import crear_app
from src.app.contrato import ORDENES_POR_TIPO, validar_orden
from src.app.rutas_laboratorio import PENDIENTE_ENTRENAR, PENDIENTE_JUGAR, fps_entorno
from src.app.trabajos import GestorTrabajos
from src.utils.configuracion import TipoEntorno, resolver_configuracion

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
CONFIG = CONFIGS / "default.yaml"
DASHBOARD = cargar_dashboard(CONFIGS / "dashboard.yaml")
ENTORNOS = [t.value for t in TipoEntorno]
FPS = {"cartpole": 50, "mountaincar": 30, "acrobot": 15, "reacher": 50, "halfcheetah": 20}


@pytest.fixture
def cliente_y_gestor() -> Iterator[tuple[TestClient, GestorTrabajos]]:
    """Aplicación con un gestor pequeño; se cierra al acabar."""
    gestor = GestorTrabajos(1)
    with TestClient(crear_app(CONFIG, gestor)) as cliente:
        yield cliente, gestor
    gestor.cerrar()


@pytest.fixture
def cliente(cliente_y_gestor) -> TestClient:
    """Solo el cliente."""
    return cliente_y_gestor[0]


# --- Órdenes del WebSocket (contrato) ---------------------------------------------------

@pytest.mark.parametrize("tipo_trabajo, orden", [
    ("sesion", {"tipo": "accion", "accion": 0}),
    ("sesion", {"tipo": "accion", "accion": 2}),
    ("sesion", {"tipo": "accion", "accion": None}),
    ("sesion", {"tipo": "finalizar"}),
    ("entrenamiento", {"tipo": "vista", "activa": True}),
    ("entrenamiento", {"tipo": "vista", "activa": False}),
    ("entrenamiento", {"tipo": "finalizar"}),
    ("comparacion", {"tipo": "vista", "activa": True}),
    ("comparacion", {"tipo": "finalizar"}),
])
def test_ordenes_validas(tipo_trabajo, orden) -> None:
    assert validar_orden(tipo_trabajo, orden) == orden


@pytest.mark.parametrize("tipo_trabajo, orden, mensaje", [
    ("sesion", {"tipo": "vista", "activa": True}, "no válida para un trabajo de tipo sesion"),
    ("entrenamiento", {"tipo": "accion", "accion": 1}, "no válida para un trabajo"),
    ("entrenamiento", {"tipo": "otra"}, "valen: pausar, reanudar, cancelar, vista, finalizar"),
    ("desconocido", {"tipo": "finalizar"}, "valen: pausar, reanudar, cancelar"),
    ("sesion", {"tipo": "accion"}, "accion"),
    ("sesion", {"tipo": "accion", "accion": -1}, "accion"),
    ("sesion", {"tipo": "accion", "accion": True}, "accion"),
    ("sesion", {"tipo": "accion", "accion": "1"}, "accion"),
    ("sesion", {"tipo": "accion", "accion": 1.0}, "accion"),
    ("sesion", {"tipo": "accion", "accion": 1, "extra": 2}, "extra"),
    ("entrenamiento", {"tipo": "vista"}, "activa"),
    ("entrenamiento", {"tipo": "vista", "activa": "true"}, "activa"),
    ("entrenamiento", {"tipo": "vista", "activa": 1}, "activa"),
    ("entrenamiento", {"tipo": "finalizar", "ya": True}, "ya"),
    ("sesion", [1], "no válida"),
])
def test_ordenes_que_no_valen(tipo_trabajo, orden, mensaje) -> None:
    with pytest.raises(ValueError, match=mensaje):
        validar_orden(tipo_trabajo, orden)


def test_cada_tipo_de_trabajo_admite_finalizar() -> None:
    for tipo_trabajo in ORDENES_POR_TIPO:
        assert validar_orden(tipo_trabajo, {"tipo": "finalizar"}) == {"tipo": "finalizar"}


# --- Configuración del juego -------------------------------------------------------------

@pytest.mark.parametrize("entorno", ENTORNOS)
def test_configuracion_del_juego(cliente, entorno) -> None:
    respuesta = cliente.get(f"/api/laboratorio/{entorno}")
    assert respuesta.status_code == 200
    datos = respuesta.json()
    demostraciones = resolver_configuracion(CONFIG, entorno)["demostraciones"]
    assert datos == {"entorno": entorno, "fps": FPS[entorno],
                     "jugar_a_mano": entorno in ("cartpole", "mountaincar"),
                     "partidas_experto": demostraciones["partidas_train"]
                     + demostraciones["partidas_validacion"],
                     "partidas_maximas": 1000, "semilla": demostraciones["semilla"]}


def test_fps_se_calcula_una_vez() -> None:
    fps_entorno.cache_clear()
    for _ in range(3):
        fps_entorno("CartPole-v1")
    assert fps_entorno.cache_info().misses == 1


def test_entorno_desconocido_da_422(cliente) -> None:
    assert cliente.get("/api/laboratorio/pong").status_code == 422
    assert cliente.get("/api/laboratorio/pong/ajustes/bc").status_code == 422


# --- Esquema de ajustes --------------------------------------------------------------------

@pytest.mark.parametrize("entorno, metodo", list(itertools.product(ENTORNOS, METODOS)))
def test_esquema_de_ajustes(cliente, entorno, metodo) -> None:
    respuesta = cliente.get(f"/api/laboratorio/{entorno}/ajustes/{metodo}")
    assert respuesta.status_code == 200
    assert respuesta.json() == esquema_ajustes(CONFIG, entorno, metodo, DASHBOARD)


def test_metodo_desconocido_da_422(cliente) -> None:
    assert cliente.get("/api/laboratorio/cartpole/ajustes/ppo").status_code == 422


# --- Crear sesiones ------------------------------------------------------------------------

@pytest.mark.parametrize("cuerpo", [
    {"entorno": "cartpole", "modo": "manual"},
    {"entorno": "mountaincar", "modo": "manual", "semilla": 5},
    {"entorno": "halfcheetah", "modo": "experto"},
    {"entorno": "reacher", "modo": "experto", "partidas": 1000, "experto": "abc", "semilla": 0},
])
def test_una_sesion_valida_responde_501_hasta_su_rama(cliente, cuerpo) -> None:
    respuesta = cliente.post("/api/sesiones", json=cuerpo)
    assert respuesta.status_code == 501
    assert respuesta.json() == {"detail": PENDIENTE_JUGAR}


@pytest.mark.parametrize("cuerpo, mensaje", [
    ({"entorno": "reacher", "modo": "manual"}, "no se puede jugar a mano"),
    ({"entorno": "cartpole", "modo": "experto", "partidas": 1001}, "Como mucho 1000"),
])
def test_sesion_que_no_vale_por_la_configuracion(cliente, cuerpo, mensaje) -> None:
    respuesta = cliente.post("/api/sesiones", json=cuerpo)
    assert respuesta.status_code == 422
    assert mensaje in respuesta.json()["detail"]


@pytest.mark.parametrize("cuerpo", [
    {"entorno": "cartpole"},
    {"entorno": "pong", "modo": "manual"},
    {"entorno": "cartpole", "modo": "jugar"},
    {"entorno": "cartpole", "modo": "manual", "partidas": 3},
    {"entorno": "cartpole", "modo": "manual", "experto": "x"},
    {"entorno": "cartpole", "modo": "experto", "partidas": 0},
    {"entorno": "cartpole", "modo": "experto", "partidas": "3"},
    {"entorno": "cartpole", "modo": "experto", "partidas": 2.0},
    {"entorno": "cartpole", "modo": "experto", "semilla": -1},
    {"entorno": "cartpole", "modo": "manual", "otra": 1},
])
def test_sesion_que_no_vale_por_su_forma(cliente, cuerpo) -> None:
    assert cliente.post("/api/sesiones", json=cuerpo).status_code == 422


# --- Guardar sesiones ----------------------------------------------------------------------

GUARDAR = {"nombre": "cartpole_humano_1", "partidas": [{"indice": 0, "tramo": [0, 20]},
                                                      {"indice": 2, "tramo": [5, 6]}]}


def test_guardar_una_sesion_responde_501_hasta_su_rama(cliente_y_gestor) -> None:
    cliente, gestor = cliente_y_gestor
    sesion = gestor.crear("sesion", "tareas_app:sumar", {"numeros": [1]}, interactivo=True)
    respuesta = cliente.post(f"/api/sesiones/{sesion['id']}/guardar", json=GUARDAR)
    assert respuesta.status_code == 501
    assert respuesta.json() == {"detail": PENDIENTE_JUGAR}


def test_guardar_algo_que_no_es_una_sesion(cliente_y_gestor) -> None:
    cliente, gestor = cliente_y_gestor
    otro = gestor.crear("entrenamiento", "tareas_app:sumar", {"numeros": [1]})
    respuesta = cliente.post(f"/api/trabajos/{otro['id']}", json=GUARDAR)
    assert respuesta.status_code == 405
    respuesta = cliente.post(f"/api/sesiones/{otro['id']}/guardar", json=GUARDAR)
    assert respuesta.status_code == 409
    assert "no es una sesión" in respuesta.json()["detail"]
    assert cliente.post("/api/sesiones/no-existe/guardar", json=GUARDAR).status_code == 404


@pytest.mark.parametrize("cuerpo", [
    {"nombre": "", "partidas": GUARDAR["partidas"]},
    {"nombre": "   ", "partidas": GUARDAR["partidas"]},
    {"nombre": "x" * 101, "partidas": GUARDAR["partidas"]},
    {"nombre": "a", "partidas": []},
    {"nombre": "a", "partidas": [{"indice": 0, "tramo": [3, 3]}]},
    {"nombre": "a", "partidas": [{"indice": 0, "tramo": [4, 3]}]},
    {"nombre": "a", "partidas": [{"indice": 0, "tramo": [-1, 3]}]},
    {"nombre": "a", "partidas": [{"indice": 0, "tramo": [0, 1, 2]}]},
    {"nombre": "a", "partidas": [{"indice": -1, "tramo": [0, 1]}]},
    {"nombre": "a", "partidas": [{"indice": 0, "tramo": [0, 1]}, {"indice": 0, "tramo": [1, 2]}]},
    {"nombre": "a", "partidas": [{"indice": 0, "tramo": [0, 1], "x": 1}]},
])
def test_guardar_con_un_cuerpo_que_no_vale(cliente_y_gestor, cuerpo) -> None:
    cliente, gestor = cliente_y_gestor
    sesion = gestor.crear("sesion", "tareas_app:sumar", {"numeros": [1]}, interactivo=True)
    assert cliente.post(f"/api/sesiones/{sesion['id']}/guardar", json=cuerpo).status_code == 422


# --- Crear entrenamientos ---------------------------------------------------------------------

@pytest.mark.parametrize("cuerpo", [
    {"entorno": "cartpole", "metodo": "experto"},
    {"entorno": "cartpole", "metodo": "experto", "vista": True, "wandb": True, "semilla": 4,
     "ajustes": {"experto": {"pasos_totales": 2048, "trpo": {"gamma": 0.95}}}},
    {"entorno": "acrobot", "metodo": "bc", "datos": ["a", "b"],
     "ajustes": {"bc": {"learning_rate": 0.01}}},
    {"entorno": "halfcheetah", "metodo": "gail", "datos": ["a"], "vista": True,
     "ajustes": {"gail": {"iteraciones": 3, "discriminador": {"learning_rate": 0.1}}}},
    {"entorno": "cartpole", "metodo": "experto", "nombre": "Mi experto"},
    {"entorno": "cartpole", "metodo": "experto", "nombre": "x" * 100},
    {"entorno": "cartpole", "metodo": "experto", "nombre": None},
])
def test_un_entrenamiento_valido_responde_501_hasta_su_rama(cliente, cuerpo) -> None:
    respuesta = cliente.post("/api/entrenamientos", json=cuerpo)
    assert respuesta.status_code == 501
    assert respuesta.json() == {"detail": PENDIENTE_ENTRENAR}


@pytest.mark.parametrize("cuerpo, mensaje", [
    ({"entorno": "cartpole", "metodo": "bc", "datos": ["a"], "ajustes": {"bc": {"batch_size": 0}}},
     "bc.batch_size: tiene que ser como mínimo 1"),
    ({"entorno": "cartpole", "metodo": "bc", "datos": ["a"], "ajustes": {"trpo": {"gamma": 0.5}}},
     "trpo.gamma: no es un ajuste de bc"),
    ({"entorno": "cartpole", "metodo": "gail", "datos": ["a"],
      "ajustes": {"gail": {"recompensa": "otra"}}}, "gail.recompensa"),
])
def test_entrenamiento_con_ajustes_que_no_valen(cliente, cuerpo, mensaje) -> None:
    respuesta = cliente.post("/api/entrenamientos", json=cuerpo)
    assert respuesta.status_code == 422
    assert mensaje in respuesta.json()["detail"]


@pytest.mark.parametrize("cuerpo", [
    {"entorno": "cartpole", "metodo": "experto", "datos": ["a"]},
    {"entorno": "cartpole", "metodo": "bc"},
    {"entorno": "cartpole", "metodo": "gail", "datos": []},
    {"entorno": "cartpole", "metodo": "bc", "datos": ["a", "a"]},
    {"entorno": "cartpole", "metodo": "bc", "datos": [""]},
    {"entorno": "cartpole", "metodo": "bc", "datos": ["a"], "vista": True},
    {"entorno": "cartpole", "metodo": "ppo"},
    {"entorno": "pong", "metodo": "experto"},
    {"entorno": "cartpole", "metodo": "experto", "semilla": -1},
    {"entorno": "cartpole", "metodo": "experto", "vista": "true"},
    {"entorno": "cartpole", "metodo": "experto", "wandb": 1},
    {"entorno": "cartpole", "metodo": "experto", "ajustes": [1]},
    {"entorno": "cartpole", "metodo": "experto", "otra": 1},
    {"entorno": "cartpole", "metodo": "experto", "nombre": ""},
    {"entorno": "cartpole", "metodo": "experto", "nombre": "   "},
    {"entorno": "cartpole", "metodo": "experto", "nombre": "x" * 101},
    {"entorno": "cartpole", "metodo": "experto", "nombre": 3},
])
def test_entrenamiento_que_no_vale_por_su_forma(cliente, cuerpo) -> None:
    assert cliente.post("/api/entrenamientos", json=cuerpo).status_code == 422


# --- Documentación ---------------------------------------------------------------------------

def test_la_documentacion_tiene_todo_el_contrato(cliente) -> None:
    esquema = cliente.get("/openapi.json").json()
    for ruta in ("/api/estado", "/api/trabajos", "/api/trabajos/{id_trabajo}",
                 "/api/laboratorio/{entorno}", "/api/laboratorio/{entorno}/ajustes/{metodo}",
                 "/api/sesiones", "/api/sesiones/{id_trabajo}/guardar", "/api/entrenamientos"):
        assert ruta in esquema["paths"], ruta
    for modelo in ("SesionCrear", "SesionGuardar", "EntrenamientoCrear", "EsquemaAjustesApi",
                   "ConfiguracionJuegoApi", "TrabajoApi", "ResultadoGuardadoApi"):
        assert modelo in esquema["components"]["schemas"], modelo


def test_un_dashboard_que_no_vale_impide_crear_la_app(tmp_path) -> None:
    import shutil

    copia = tmp_path / "configs"
    shutil.copytree(CONFIGS, copia)
    (copia / "dashboard.yaml").write_text("laboratorio: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="falta la sección"):
        crear_app(copia / "default.yaml", GestorTrabajos(1))


# --- Nombre del modelo ---------------------------------------------------------------------

def test_nombre_del_modelo_el_del_usuario_o_uno_generado() -> None:
    from datetime import datetime

    from src.app.contrato import EntrenamientoCrear
    from src.app.rutas_laboratorio import nombre_modelo
    from src.data.biblioteca import nombre_por_defecto

    con = EntrenamientoCrear(entorno="cartpole", metodo="experto", nombre="  Mi experto ")
    assert nombre_modelo(con, "CartPole-v1") == "Mi experto"
    sin = EntrenamientoCrear(entorno="reacher", metodo="gail", datos=["a"])
    generado = nombre_modelo(sin, "Reacher-v5")
    assert generado.startswith("GAIL · Reacher · ")
    momento = datetime(2026, 11, 24, 18, 30)
    assert nombre_por_defecto("gail", "CartPole-v1", momento) == "GAIL · CartPole · 24/11 18:30"
    assert nombre_por_defecto("bc", "HalfCheetah-v5", momento) == "BC · HalfCheetah · 24/11 18:30"
    assert nombre_por_defecto("experto", "MountainCar-v0", momento) == \
        "Experto · MountainCar · 24/11 18:30"
