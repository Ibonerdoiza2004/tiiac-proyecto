"""Tests del contrato (src/app/contrato.py) y de las rutas del Laboratorio
(src/app/rutas_laboratorio.py): configuración del juego, esquema de ajustes y crear
sesiones y entrenamientos (que validan y responden 501 hasta sus ramas -api).
"""

from __future__ import annotations

import itertools
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient
from test_biblioteca import escribir_archivos

from src.app.ajustes import METODOS, cargar_dashboard, esquema_ajustes
from src.app.api import crear_app
from src.app.contrato import ORDENES_POR_TIPO, validar_orden
from src.app.rutas_laboratorio import PENDIENTE_ENTRENAR, PENDIENTE_JUGAR, fps_entorno
from src.app.trabajos import GestorTrabajos
from src.data.biblioteca import ARCHIVOS, agregar, crear_ficha, nuevo_id
from src.utils.configuracion import TipoEntorno, resolver_configuracion

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
CONFIG = CONFIGS / "default.yaml"
DASHBOARD = cargar_dashboard(CONFIGS / "dashboard.yaml")
ENTORNOS = [t.value for t in TipoEntorno]
FPS = {"cartpole": 50, "mountaincar": 30, "acrobot": 15, "reacher": 50, "halfcheetah": 20}
GYM = {entorno: resolver_configuracion(CONFIG, entorno)["entorno"]["id"] for entorno in ENTORNOS}


def configs_con_biblioteca(tmp_path: Path) -> Path:
    """Copia configs/ con la biblioteca en tmp_path/biblioteca; devuelve el YAML base."""
    copia = tmp_path / "configs"
    shutil.copytree(CONFIGS, copia)
    base = copia / "default.yaml"
    cfg = yaml.safe_load(base.read_text(encoding="utf-8"))
    cfg["aplicativo"]["biblioteca"] = str(tmp_path / "biblioteca")
    base.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return base


def elemento(raiz: Path, tipo: str, entorno: str, valido: bool = True) -> str:
    """Añade a la biblioteca un elemento del tipo y entorno (roto si no es válido)."""
    ficha = crear_ficha(nuevo_id(), entorno, tipo, f"{tipo} de prueba", "dashboard")
    carpeta = agregar(raiz, ficha, lambda c: escribir_archivos(c, tipo, GYM[entorno]))
    if not valido:
        (carpeta / ARCHIVOS[tipo][0]).unlink()
    return ficha["id"]


@pytest.fixture
def raiz(tmp_path: Path) -> Path:
    """Carpeta de la biblioteca de la aplicación de los tests."""
    return tmp_path / "biblioteca"


@pytest.fixture
def cliente_y_gestor(tmp_path: Path) -> Iterator[tuple[TestClient, GestorTrabajos]]:
    """Aplicación con un gestor pequeño y una biblioteca vacía propia; se cierra al acabar."""
    gestor = GestorTrabajos(1)
    with TestClient(crear_app(configs_con_biblioteca(tmp_path), gestor)) as cliente:
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
    esperado = esquema_ajustes(CONFIG, entorno, metodo, DASHBOARD)
    for seccion in esperado["secciones"]:
        seccion["campos"] = [{k: v for k, v in campo.items() if v is not None}
                             for campo in seccion["campos"]]
    assert respuesta.json() == esperado


@pytest.mark.parametrize("entorno, metodo", list(itertools.product(ENTORNOS, METODOS)))
def test_el_esquema_no_manda_limites_ni_opciones_nulos(cliente, entorno, metodo) -> None:
    """Regresión: min: null hacía que el formulario rechazara retorno_objetivo −90 («como
    mínimo null»). Un límite que no existe no aparece."""
    esquema = cliente.get(f"/api/laboratorio/{entorno}/ajustes/{metodo}").json()
    campos = [c for s in esquema["secciones"] for c in s["campos"]]
    assert all(None not in campo.values() for campo in campos)
    objetivo = [c for c in campos if c["clave"] == "retorno_objetivo"]
    if metodo == "experto":
        assert objetivo and "min" not in objetivo[0] and "max" not in objetivo[0]


def test_metodo_desconocido_da_422(cliente) -> None:
    assert cliente.get("/api/laboratorio/cartpole/ajustes/ppo").status_code == 422


# --- Crear sesiones ------------------------------------------------------------------------

@pytest.mark.parametrize("cuerpo", [
    {"entorno": "cartpole", "modo": "manual"},
    {"entorno": "mountaincar", "modo": "manual", "semilla": 5},
    {"entorno": "halfcheetah", "modo": "experto"},
    {"entorno": "reacher", "modo": "experto", "partidas": 1000, "experto": "@", "semilla": 0},
])
def test_una_sesion_valida_responde_501_hasta_su_rama(cliente, raiz, cuerpo) -> None:
    if cuerpo.get("experto") == "@":
        cuerpo = {**cuerpo, "experto": elemento(raiz, "experto", cuerpo["entorno"])}
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

GUARDAR = {"nombre": "cartpole_humano_1", "partidas": [0, 2]}


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
    {"nombre": "a", "partidas": [-1]},
    {"nombre": "a", "partidas": [0, 2, 0]},
    {"nombre": "a", "partidas": ["0"]},
    {"nombre": "a", "partidas": [1.5]},
    {"nombre": "a", "partidas": [True]},
    {"nombre": "a", "partidas": 0},
    # El formato anterior, con tramos, ya no vale.
    {"nombre": "a", "partidas": [{"indice": 0, "tramo": [0, 1]}]},
])
def test_guardar_con_un_cuerpo_que_no_vale(cliente_y_gestor, cuerpo) -> None:
    cliente, gestor = cliente_y_gestor
    sesion = gestor.crear("sesion", "tareas_app:sumar", {"numeros": [1]}, interactivo=True)
    assert cliente.post(f"/api/sesiones/{sesion['id']}/guardar", json=cuerpo).status_code == 422


# --- Crear entrenamientos ---------------------------------------------------------------------

@pytest.mark.parametrize("cuerpo", [
    {"entorno": "cartpole", "metodo": "experto"},
    {"entorno": "cartpole", "metodo": "experto", "vista": True, "semilla": 4,
     "ajustes": {"experto": {"pasos_totales": 2048, "trpo": {"gamma": 0.95}}}},
    {"entorno": "acrobot", "metodo": "bc", "datos": ["@", "@"],
     "ajustes": {"bc": {"learning_rate": 0.01}}},
    {"entorno": "halfcheetah", "metodo": "gail", "datos": ["@"], "vista": True,
     "ajustes": {"gail": {"iteraciones": 3, "discriminador": {"learning_rate": 0.1}}}},
    {"entorno": "cartpole", "metodo": "experto", "nombre": "Mi experto"},
    {"entorno": "cartpole", "metodo": "experto", "nombre": "x" * 100},
    {"entorno": "cartpole", "metodo": "experto", "nombre": None},
])
def test_un_entrenamiento_valido_responde_501_hasta_su_rama(cliente, raiz, cuerpo) -> None:
    if "datos" in cuerpo:
        cuerpo = {**cuerpo, "datos": [elemento(raiz, "demostraciones", cuerpo["entorno"])
                                      for _ in cuerpo["datos"]]}
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


@pytest.mark.parametrize("entorno, metodo, ajustes, mensaje", [
    # Cada campo en su rango, pero el entrenador lo rechaza (regresión: llegaban al 501).
    ("cartpole", "bc", {"bc": {"learning_rate": 0}}, "bc.learning_rate debe ser un real positivo"),
    ("cartpole", "gail", {"trpo": {"gamma": 0}}, "trpo.gamma no es válido"),
    ("cartpole", "gail", {"trpo": {"n_entornos": 3}},
     "trpo.pasos_por_iteracion debe ser divisible por trpo.n_entornos"),
    ("cartpole", "gail", {"trpo": {"batch_size_valor": 300}},
     "trpo.batch_size_valor debe dividir trpo.pasos_por_iteracion"),
    ("cartpole", "gail", {"gail": {"discriminador": {"learning_rate": 0}}},
     "gail.discriminador.learning_rate debe ser positivo"),
    ("reacher", "experto", {"experto": {"trpo": {"n_entornos": 7}}},
     "trpo.pasos_por_iteracion debe ser divisible por trpo.n_entornos"),
    # Regresión: con la normalización activada, BC aceptaba épsilon 0 y la política lo
    # rechazaba al ajustar la normalización.
    ("cartpole", "bc", {"bc": {"epsilon_normalizacion": 0}},
     "bc.epsilon_normalizacion debe ser un real positivo si se normaliza"),
])
def test_entrenamiento_que_el_entrenador_rechazaria(cliente, raiz, entorno, metodo, ajustes,
                                                    mensaje) -> None:
    datos = [elemento(raiz, "demostraciones", entorno)] if metodo != "experto" else []
    respuesta = cliente.post("/api/entrenamientos", json={
        "entorno": entorno, "metodo": metodo, "datos": datos, "ajustes": ajustes})
    assert respuesta.status_code == 422
    assert mensaje in respuesta.json()["detail"]


def test_bc_sin_normalizar_no_mira_el_epsilon(cliente, raiz) -> None:
    datos = [elemento(raiz, "demostraciones", "cartpole")]
    respuesta = cliente.post("/api/entrenamientos", json={
        "entorno": "cartpole", "metodo": "bc", "datos": datos,
        "ajustes": {"bc": {"normalizar_observaciones": False, "epsilon_normalizacion": 0}}})
    assert respuesta.status_code == 501


@pytest.mark.parametrize("objetivo", [10**20, 1e20, -10**20])
def test_un_objetivo_enorme_no_da_500(cliente, objetivo) -> None:
    """Regresión: 10**20 como entero daba 500 (np.isfinite no admite enteros de Python
    tan grandes) y como 1e20 pasaba."""
    respuesta = cliente.post("/api/entrenamientos", json={
        "entorno": "mountaincar", "metodo": "experto",
        "ajustes": {"experto": {"retorno_objetivo": objetivo}}})
    assert respuesta.status_code == 501


def test_un_objetivo_que_no_cabe_en_un_float_da_422(cliente) -> None:
    respuesta = cliente.post(
        "/api/entrenamientos",
        content=('{"entorno": "mountaincar", "metodo": "experto", "ajustes": '
                 '{"experto": {"retorno_objetivo": 1' + "0" * 400 + "}}}"),
        headers={"content-type": "application/json"})
    assert respuesta.status_code == 422
    assert "retorno_objetivo" in respuesta.json()["detail"]


# --- Referencias a la biblioteca -------------------------------------------------------------

def _referencias(raiz: Path, tipo: str) -> list[tuple[str, str]]:
    """Ids que no valen como tipo en cartpole, con el motivo esperado."""
    otro_tipo = "demostraciones" if tipo == "experto" else "experto"
    return [("0" * 12, "No hay ningún elemento"),
            ("no-es-un-id", "No hay ningún elemento"),
            (elemento(raiz, otro_tipo, "cartpole"), f"es de tipo {otro_tipo}"),
            (elemento(raiz, "bc", "cartpole"), "es de tipo bc"),
            (elemento(raiz, tipo, "mountaincar"), "es de mountaincar, no de cartpole"),
            (elemento(raiz, tipo, "cartpole", valido=False), "no es válido: faltan")]


def test_ver_un_experto_que_no_vale_da_422(cliente, raiz) -> None:
    for id_experto, motivo in _referencias(raiz, "experto"):
        respuesta = cliente.post("/api/sesiones", json={"entorno": "cartpole", "modo": "experto",
                                                        "experto": id_experto})
        assert respuesta.status_code == 422, id_experto
        assert motivo in respuesta.json()["detail"], respuesta.json()


@pytest.mark.parametrize("metodo", ["bc", "gail"])
def test_entrenar_con_datos_que_no_valen_da_422(cliente, raiz, metodo) -> None:
    bueno = elemento(raiz, "demostraciones", "cartpole")
    for id_datos, motivo in _referencias(raiz, "demostraciones"):
        respuesta = cliente.post("/api/entrenamientos", json={
            "entorno": "cartpole", "metodo": metodo, "datos": [bueno, id_datos]})
        assert respuesta.status_code == 422, id_datos
        assert motivo in respuesta.json()["detail"], respuesta.json()


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


@pytest.mark.parametrize("cambio, mensaje", [
    (lambda d: d["campos"]["bc"].pop("learning_rate"), "no describe bc.learning_rate"),
    (lambda d: d["campos"]["trpo"].pop("gamma"), "no describe experto.trpo.gamma"),
    (lambda d: d["laboratorio"].update(jugar_a_mano="cartpole"), "jugar_a_mano"),
    (lambda d: d["laboratorio"].update(jugar_a_mano=["pong"]), "jugar_a_mano"),
    (lambda d: d["laboratorio"].update(jugar_a_mano=5), "jugar_a_mano"),
    # Regresión: con un título de sección numérico la app arrancaba y el formulario de
    # BC daba 500 (el arranque no validaba contra el modelo de la respuesta).
    (lambda d: d["secciones"].update(bc=7), "título de la sección bc"),
    (lambda d: d["secciones"].update(bc="  "), "título de la sección bc"),
])
def test_un_formulario_roto_impide_crear_la_app(tmp_path, cambio, mensaje) -> None:
    """Regresión: sin la descripción de un campo, la app arrancaba y ese formulario daba
    500; con jugar_a_mano que no es una lista, TypeError en lugar de ValueError."""
    base = configs_con_biblioteca(tmp_path)
    ruta = base.parent / "dashboard.yaml"
    datos = yaml.safe_load(ruta.read_text(encoding="utf-8"))
    cambio(datos)
    ruta.write_text(yaml.safe_dump(datos, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError, match=mensaje):
        crear_app(base, GestorTrabajos(1))


def test_jugar_a_mano_puede_ir_vacio(tmp_path) -> None:
    base = configs_con_biblioteca(tmp_path)
    ruta = base.parent / "dashboard.yaml"
    datos = yaml.safe_load(ruta.read_text(encoding="utf-8"))
    datos["laboratorio"]["jugar_a_mano"] = []
    ruta.write_text(yaml.safe_dump(datos, allow_unicode=True), encoding="utf-8")
    assert cargar_dashboard(ruta)["laboratorio"]["jugar_a_mano"] == []


def test_un_dashboard_que_no_vale_impide_crear_la_app(tmp_path) -> None:
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


# --- W&B ---------------------------------------------------------------------------------------

class ComprobacionFalsa:
    """Comprobación de W&B que devuelve lo que se le diga y cuenta las llamadas."""

    def __init__(self, motivo: str | None = None):
        self.motivo, self.llamadas = motivo, 0

    def __call__(self) -> str | None:
        self.llamadas += 1
        return self.motivo


def _cliente_wandb(comprobacion) -> tuple[TestClient, GestorTrabajos]:
    """App con una comprobación de W&B dada (hay que cerrar el gestor)."""
    gestor = GestorTrabajos(1)
    return TestClient(crear_app(CONFIG, gestor, comprobacion)), gestor


def _esperar_comprobado(cliente: TestClient) -> dict:
    """Espera a que termine la comprobación de W&B del arranque."""
    import time

    for _ in range(200):
        estado = cliente.get("/api/estado").json()["wandb"]
        if estado["disponible"] is not None:
            return estado
        time.sleep(0.02)
    pytest.fail("no terminó la comprobación de W&B")


def test_al_arrancar_se_comprueba_wandb_y_se_puede_repetir() -> None:
    comprobacion = ComprobacionFalsa("no hay sesión")
    cliente, gestor = _cliente_wandb(comprobacion)
    with cliente:
        assert _esperar_comprobado(cliente) == {"disponible": False, "motivo": "no hay sesión"}
        comprobacion.motivo = None  # el usuario hace wandb login
        assert cliente.post("/api/wandb/comprobar").json() == {"disponible": True,
                                                              "motivo": None}
        assert cliente.get("/api/estado").json()["wandb"]["disponible"] is True
    gestor.cerrar()
    assert comprobacion.llamadas == 2


ENTRENAR_CON_WANDB = {"entorno": "cartpole", "metodo": "experto", "wandb": True}


def test_entrenar_con_wandb_sin_sesion_da_503_con_el_motivo() -> None:
    cliente, gestor = _cliente_wandb(ComprobacionFalsa("la clave no vale"))
    with cliente:
        _esperar_comprobado(cliente)
        respuesta = cliente.post("/api/entrenamientos", json=ENTRENAR_CON_WANDB)
        assert respuesta.status_code == 503
        assert "la clave no vale" in respuesta.json()["detail"]
        assert "wandb login" in respuesta.json()["detail"]
        # Sin W&B no hace falta sesión.
        assert cliente.post("/api/entrenamientos",
                            json={**ENTRENAR_CON_WANDB, "wandb": False}).status_code == 501
    gestor.cerrar()


def test_entrenar_con_wandb_con_sesion_sigue_adelante() -> None:
    cliente, gestor = _cliente_wandb(ComprobacionFalsa(None))
    with cliente:
        _esperar_comprobado(cliente)
        for cuerpo in (ENTRENAR_CON_WANDB, {**ENTRENAR_CON_WANDB, "grupo": "mis-pruebas.1"}):
            assert cliente.post("/api/entrenamientos", json=cuerpo).status_code == 501
    gestor.cerrar()


@pytest.mark.parametrize("cuerpo", [
    {"entorno": "cartpole", "metodo": "experto", "grupo": "a"},  # grupo sin wandb
    {**ENTRENAR_CON_WANDB, "grupo": ""},
    {**ENTRENAR_CON_WANDB, "grupo": "con espacios"},
    {**ENTRENAR_CON_WANDB, "grupo": "a/b"},
    {**ENTRENAR_CON_WANDB, "grupo": "x" * 129},
    {**ENTRENAR_CON_WANDB, "grupo": 3},
])
def test_grupos_que_no_valen(cliente, cuerpo) -> None:
    assert cliente.post("/api/entrenamientos", json=cuerpo).status_code == 422


def test_config_wandb_de_un_entrenamiento() -> None:
    from src.app.contrato import EntrenamientoCrear
    from src.app.rutas_laboratorio import config_wandb

    sin = EntrenamientoCrear(entorno="cartpole", metodo="experto")
    assert config_wandb(sin, DASHBOARD) == {"activo": False, "grupo": None}
    por_defecto = EntrenamientoCrear(entorno="halfcheetah", metodo="experto", wandb=True)
    assert config_wandb(por_defecto, DASHBOARD) == {"activo": True,
                                                    "grupo": "dashboard-halfcheetah"}
    propio = EntrenamientoCrear(entorno="cartpole", metodo="gail", datos=["a"], wandb=True,
                                grupo="x" * 128)
    assert config_wandb(propio, DASHBOARD)["grupo"] == "x" * 128


def test_una_comprobacion_antigua_no_pisa_a_una_mas_reciente() -> None:
    """Regresión: la comprobación lenta del arranque (sin sesión) acababa después de otra
    pedida más tarde (con sesión) y la sobrescribía."""
    import threading

    from src.app.estado_wandb import EstadoWandb

    dentro, soltar = threading.Event(), threading.Event()
    respuestas = iter(["sin sesión todavía", None])

    def comprobacion() -> str | None:
        respuesta = next(respuestas)
        if respuesta is not None:  # la primera, lenta
            dentro.set()
            soltar.wait(5)
        return respuesta

    estado = EstadoWandb(comprobacion)
    resultados = []
    lenta = threading.Thread(target=lambda: resultados.append(estado.comprobar()))
    lenta.start()
    assert dentro.wait(5)
    assert estado.comprobar() == {"disponible": True, "motivo": None}
    soltar.set()
    lenta.join(5)
    assert estado.describir() == {"disponible": True, "motivo": None}
    assert resultados == [{"disponible": True, "motivo": None}]


def test_estado_wandb() -> None:
    from src.app.estado_wandb import EstadoWandb

    comprobacion = ComprobacionFalsa(None)
    estado = EstadoWandb(comprobacion)
    assert estado.describir() == {"disponible": None, "motivo": None}
    estado.exigir()  # sin comprobar todavía: comprueba ahora
    assert comprobacion.llamadas == 1
    estado.exigir()  # ya comprobado: no vuelve a hacerlo
    assert comprobacion.llamadas == 1

    def rota() -> str | None:
        raise ConnectionError("sin red")

    roto = EstadoWandb(rota)
    with pytest.raises(RuntimeError, match="no se pudo comprobar W&B"):
        roto.exigir()
    assert roto.describir()["disponible"] is False


def test_comprobar_sesion_wandb_devuelve_el_motivo(monkeypatch) -> None:
    from src.app import estado_wandb

    monkeypatch.undo()  # quita la comprobación falsa de conftest.py

    def sin_sesion(cfg, exigir_sesion=False):
        raise RuntimeError("no hay una sesión válida de W&B")

    monkeypatch.setattr(estado_wandb, "comprobar_wandb", sin_sesion)
    assert estado_wandb.comprobar_sesion_wandb() == "no hay una sesión válida de W&B"
    monkeypatch.setattr(estado_wandb, "comprobar_wandb", lambda cfg, exigir_sesion=False: None)
    assert estado_wandb.comprobar_sesion_wandb() is None
