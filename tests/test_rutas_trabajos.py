"""Tests de las rutas de trabajos (src/app/rutas_trabajos.py): HTTP y WebSocket.

Los trabajos se crean directamente en el gestor, con las tareas de juguete de
tareas_app.py, porque la API no tiene una ruta genérica para crearlos. Arrancan
procesos de verdad.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from src.app.api import crear_app
from src.app.rutas_trabajos import CIERRE_NO_EXISTE, ColaEnvio, a_json_finito
from src.app.trabajos import GestorTrabajos, SinHueco

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
LIMITE = 30.0


def esperar_hasta(condicion: Callable[[], bool], limite: float = LIMITE) -> None:
    """Espera a que la condición se cumpla o falla el test."""
    fin = time.monotonic() + limite
    while not condicion():
        if time.monotonic() > fin:
            pytest.fail("la condición no se cumplió a tiempo")
        time.sleep(0.02)


@pytest.fixture
def app_y_gestor() -> Iterator[tuple[TestClient, GestorTrabajos]]:
    """Aplicación con un gestor de dos procesos; al salir se apaga y lo cierra."""
    gestor = GestorTrabajos(2)
    app = crear_app(CONFIG, gestor)

    @app.get("/api/prueba-sin-hueco")
    def sin_hueco() -> None:
        """Ruta de prueba para el manejador de SinHueco."""
        raise SinHueco("No hay núcleos libres")

    with TestClient(app) as cliente:
        yield cliente, gestor
    gestor.cerrar()


def contar(gestor: GestorTrabajos, interactivo: bool = False) -> str:
    """Trabajo que cuenta hasta recibir {"tipo": "fin"}; espera a que avance si arranca."""
    trabajo = gestor.crear("sesion" if interactivo else "entrenamiento",
                           "tareas_app:contar_hasta_orden", interactivo=interactivo)
    if trabajo["estado"] == "en_marcha":
        esperar_hasta(lambda: len(gestor.historial(trabajo["id"])) >= 3)
    return trabajo["id"]


def leer(ws: Any) -> Any:
    """Siguiente mensaje: dict si es JSON, bytes si es binario; lanza si se cierra."""
    mensaje = ws.receive()
    if mensaje["type"] == "websocket.close":
        raise WebSocketDisconnect(mensaje.get("code", 1000))
    if mensaje.get("text") is not None:
        return json.loads(mensaje["text"])
    return mensaje["bytes"]


def leer_hasta_cierre(ws: Any, maximo: int = 100_000) -> tuple[list[Any], int]:
    """Lee todos los mensajes hasta que el servidor cierra; devuelve (mensajes, código)."""
    mensajes = []
    for _ in range(maximo):
        try:
            mensajes.append(leer(ws))
        except WebSocketDisconnect as cierre:
            return mensajes, cierre.code
    pytest.fail("el servidor no cerró la conexión")


def leer_hasta(ws: Any, condicion: Callable[[Any], bool], maximo: int = 100_000) -> list[Any]:
    """Lee mensajes hasta uno que cumpla la condición (incluido)."""
    mensajes = []
    for _ in range(maximo):
        mensajes.append(leer(ws))
        if condicion(mensajes[-1]):
            return mensajes
    pytest.fail("no llegó el mensaje esperado")


def cerrar_y_esperar(ws: Any, gestor: GestorTrabajos) -> None:
    """Cierra desde el navegador y espera a que el servidor termine con la conexión.

    El cliente de pruebas de Starlette interrumpe la función del WebSocket al salir
    del ``with`` aunque no haya acabado de procesar el cierre (Uvicorn no lo hace).
    Esperar a que se quite su suscripción evita esa carrera, que es del cliente de
    pruebas y no de la API.
    """
    ws.close()
    esperar_hasta(lambda: not gestor._observadores)


def es_estado(estado: str) -> Callable[[Any], bool]:
    """Condición: mensaje de estado con ese estado."""
    return lambda m: isinstance(m, dict) and m.get("tipo") == "estado" and \
        m["trabajo"]["estado"] == estado


# --- HTTP ---------------------------------------------------------------------------

def test_listar_obtener_y_filtrar(app_y_gestor) -> None:
    cliente, gestor = app_y_gestor
    assert cliente.get("/api/trabajos").json() == []
    a = gestor.crear("entrenamiento", "tareas_app:sumar", {"numeros": [1, 2]})["id"]
    b = gestor.crear("comparacion", "tareas_app:sumar", {"numeros": [3]})["id"]
    for id_ in (a, b):
        gestor.esperar(id_, LIMITE)
    lista = cliente.get("/api/trabajos").json()
    assert [t["id"] for t in lista] == [a, b]
    assert [t["id"] for t in cliente.get("/api/trabajos", params={"tipo": "comparacion"}).json()] == [b]
    assert cliente.get("/api/trabajos", params={"tipo": "otro"}).json() == []
    uno = cliente.get(f"/api/trabajos/{a}").json()
    assert uno["estado"] == "terminado"
    assert uno["resultado"] == {"suma": 3}
    assert uno["progresos"] == 2
    assert uno["argumentos"] == {"numeros": [1, 2]}
    assert set(uno) == {"id", "tipo", "estado", "argumentos", "interactivo", "creado",
                        "iniciado", "terminado", "progresos", "resultado", "error"}


def test_un_trabajo_que_no_existe_da_404(app_y_gestor) -> None:
    cliente, _ = app_y_gestor
    for metodo, ruta in [("get", ""), ("post", "/pausar"), ("post", "/reanudar"),
                         ("post", "/cancelar")]:
        respuesta = getattr(cliente, metodo)(f"/api/trabajos/no-existe{ruta}")
        assert respuesta.status_code == 404
        assert respuesta.json() == {"detail": "No existe el trabajo no-existe"}


def test_pausar_reanudar_y_cancelar_por_http(app_y_gestor) -> None:
    cliente, gestor = app_y_gestor
    trabajo = contar(gestor)
    assert cliente.post(f"/api/trabajos/{trabajo}/pausar").json()["estado"] == "pausado"
    assert cliente.post(f"/api/trabajos/{trabajo}/reanudar").json()["estado"] == "en_marcha"
    cliente.post(f"/api/trabajos/{trabajo}/cancelar")
    assert gestor.esperar(trabajo, LIMITE)["estado"] == "cancelado"
    assert cliente.get(f"/api/trabajos/{trabajo}").json()["estado"] == "cancelado"


@pytest.mark.parametrize("accion", ["reanudar", "cancelar"])
def test_una_operacion_que_no_vale_en_su_estado_da_409(app_y_gestor, accion) -> None:
    cliente, gestor = app_y_gestor
    trabajo = gestor.crear("entrenamiento", "tareas_app:sumar", {"numeros": [1]})["id"]
    gestor.esperar(trabajo, LIMITE)
    respuesta = cliente.post(f"/api/trabajos/{trabajo}/{accion}")
    assert respuesta.status_code == 409
    assert "terminado" in respuesta.json()["detail"]


def test_pausar_uno_en_cola_da_409(app_y_gestor) -> None:
    cliente, gestor = app_y_gestor
    ocupados = [contar(gestor), contar(gestor)]
    en_cola = gestor.crear("entrenamiento", "tareas_app:sumar", {"numeros": [1]})["id"]
    respuesta = cliente.post(f"/api/trabajos/{en_cola}/pausar")
    assert respuesta.status_code == 409
    assert "en_cola" in respuesta.json()["detail"]
    for trabajo in ocupados:
        gestor.enviar_orden(trabajo, {"tipo": "fin"})


def test_sin_hueco_da_503(app_y_gestor) -> None:
    cliente, _ = app_y_gestor
    respuesta = cliente.get("/api/prueba-sin-hueco")
    assert respuesta.status_code == 503
    assert respuesta.json() == {"detail": "No hay núcleos libres"}


def test_estado_cuenta_los_huecos_libres(app_y_gestor) -> None:
    cliente, gestor = app_y_gestor
    assert cliente.get("/api/estado").json()["libres"] == 2
    trabajo = contar(gestor)
    estado = cliente.get("/api/estado").json()
    assert (estado["procesos"], estado["libres"]) == (2, 1)
    gestor.enviar_orden(trabajo, {"tipo": "fin"})
    gestor.esperar(trabajo, LIMITE)
    assert cliente.get("/api/estado").json()["libres"] == 2


def test_los_numeros_no_finitos_salen_como_null_por_http(app_y_gestor) -> None:
    cliente, gestor = app_y_gestor
    trabajo = gestor.crear("entrenamiento", "tareas_app:progreso_no_finito")["id"]
    gestor.esperar(trabajo, LIMITE)
    respuesta = cliente.get(f"/api/trabajos/{trabajo}")
    assert respuesta.status_code == 200
    assert respuesta.json()["resultado"] == {"perdida": None}


def test_apagar_la_api_cancela_los_trabajos() -> None:
    gestor = GestorTrabajos(1)
    with TestClient(crear_app(CONFIG, gestor)):
        trabajo = contar(gestor)
        proceso = gestor._trabajos[trabajo].proceso
    assert gestor.obtener(trabajo)["estado"] == "cancelado"
    assert not proceso.is_alive()


# --- WebSocket -------------------------------------------------------------------------

def test_conectarse_a_uno_acabado_da_la_foto_y_cierra(app_y_gestor) -> None:
    cliente, gestor = app_y_gestor
    trabajo = gestor.crear("entrenamiento", "tareas_app:mandar_imagenes", {"cuantas": 3})["id"]
    gestor.esperar(trabajo, LIMITE)
    esperar_hasta(lambda: gestor.ultima_imagen(trabajo) == b"imagen-2")
    with cliente.websocket_connect(f"/api/trabajos/{trabajo}") as ws:
        mensajes, codigo = leer_hasta_cierre(ws)
    assert codigo == 1000
    assert mensajes[0]["tipo"] == "estado"
    assert mensajes[0]["trabajo"]["estado"] == "terminado"
    assert mensajes[1] == {"tipo": "historial", "progreso": [{"imagenes": 3}]}
    assert mensajes[2:] == [b"imagen-2"]


def test_seguir_en_directo_hasta_el_final(app_y_gestor) -> None:
    cliente, gestor = app_y_gestor
    trabajo = contar(gestor)
    with cliente.websocket_connect(f"/api/trabajos/{trabajo}") as ws:
        estado, historial = leer(ws), leer(ws)
        assert estado["trabajo"]["estado"] == "en_marcha"
        assert historial["tipo"] == "historial"
        en_directo = leer_hasta(ws, lambda m: m["tipo"] == "progreso")
        ws.send_json({"tipo": "fin"})  # una orden cualquiera pasa tal cual al trabajo
        resto, codigo = leer_hasta_cierre(ws)
    assert codigo == 1000
    mensajes = en_directo + resto
    assert es_estado("terminado")(mensajes[-1])
    assert mensajes[-1]["trabajo"]["resultado"]["ordenes"] == [{"tipo": "fin"}]
    progresos = [m["medidas"] for m in mensajes if m["tipo"] == "progreso"]
    # Foto más directo: exactamente todo el progreso, sin huecos ni repetidos.
    assert historial["progreso"] + progresos == gestor.historial(trabajo)


def test_reconectarse_a_mitad_recupera_todo(app_y_gestor) -> None:
    cliente, gestor = app_y_gestor
    trabajo = contar(gestor)
    with cliente.websocket_connect(f"/api/trabajos/{trabajo}") as ws:
        leer(ws), leer(ws)
        leer_hasta(ws, lambda m: m["tipo"] == "progreso")
        cerrar_y_esperar(ws, gestor)
    # La conexión se cerró, pero un entrenamiento sigue.
    time.sleep(0.2)
    assert gestor.obtener(trabajo)["estado"] == "en_marcha"
    with cliente.websocket_connect(f"/api/trabajos/{trabajo}") as ws:
        _, historial = leer(ws), leer(ws)
        ws.send_json({"tipo": "fin"})
        resto, _ = leer_hasta_cierre(ws)
    progresos = [m["medidas"] for m in resto if m["tipo"] == "progreso"]
    assert historial["progreso"] + progresos == gestor.historial(trabajo)
    assert len(historial["progreso"]) >= 3


def test_controles_por_websocket(app_y_gestor) -> None:
    cliente, gestor = app_y_gestor
    trabajo = contar(gestor)
    with cliente.websocket_connect(f"/api/trabajos/{trabajo}") as ws:
        leer(ws), leer(ws)
        ws.send_json({"tipo": "pausar"})
        leer_hasta(ws, es_estado("pausado"))
        ws.send_json({"tipo": "reanudar"})
        leer_hasta(ws, es_estado("en_marcha"))
        ws.send_json({"tipo": "cancelar"})
        mensajes, codigo = leer_hasta_cierre(ws)
    assert codigo == 1000
    assert es_estado("cancelado")(mensajes[-1])


@pytest.mark.parametrize("enviado, error", [
    ("esto no es json", "no es JSON"),
    ("[1, 2]", "«tipo»"),
    ('{"accion": 1}', "«tipo»"),
    ('{"tipo": ""}', "«tipo»"),
    ('{"tipo": 3}', "«tipo»"),
])
def test_un_mensaje_que_no_vale_da_error_y_la_conexion_sigue(app_y_gestor, enviado,
                                                              error) -> None:
    cliente, gestor = app_y_gestor
    trabajo = contar(gestor)
    with cliente.websocket_connect(f"/api/trabajos/{trabajo}") as ws:
        leer(ws), leer(ws)
        ws.send_text(enviado)
        respuesta = leer_hasta(ws, lambda m: m["tipo"] == "error")[-1]
        assert error in respuesta["mensaje"]
        ws.send_json({"tipo": "fin"})
        leer_hasta_cierre(ws)
    assert gestor.obtener(trabajo)["estado"] == "terminado"


def test_un_control_que_no_vale_da_error_y_la_conexion_sigue(app_y_gestor) -> None:
    cliente, gestor = app_y_gestor
    trabajo = contar(gestor)
    with cliente.websocket_connect(f"/api/trabajos/{trabajo}") as ws:
        leer(ws), leer(ws)
        ws.send_json({"tipo": "reanudar"})
        respuesta = leer_hasta(ws, lambda m: m["tipo"] == "error")[-1]
        assert "en_marcha" in respuesta["mensaje"]
        ws.send_json({"tipo": "fin"})
        mensajes, codigo = leer_hasta_cierre(ws)
    assert codigo == 1000
    assert es_estado("terminado")(mensajes[-1])


def test_un_trabajo_que_no_existe_da_error_y_cierra_con_4404(app_y_gestor) -> None:
    cliente, _ = app_y_gestor
    with cliente.websocket_connect("/api/trabajos/no-existe") as ws:
        mensajes, codigo = leer_hasta_cierre(ws)
    assert codigo == CIERRE_NO_EXISTE
    assert mensajes == [{"tipo": "error", "mensaje": "No existe el trabajo no-existe"}]


def test_cerrar_la_conexion_de_una_sesion_la_cancela(app_y_gestor) -> None:
    cliente, gestor = app_y_gestor
    sesion = contar(gestor, interactivo=True)
    with cliente.websocket_connect(f"/api/trabajos/{sesion}") as ws:
        leer(ws), leer(ws)
        cerrar_y_esperar(ws, gestor)
    assert gestor.esperar(sesion, LIMITE)["estado"] == "cancelado"


def test_cerrar_la_conexion_de_un_entrenamiento_no_lo_para(app_y_gestor) -> None:
    cliente, gestor = app_y_gestor
    trabajo = contar(gestor)
    with cliente.websocket_connect(f"/api/trabajos/{trabajo}") as ws:
        leer(ws), leer(ws)
        cerrar_y_esperar(ws, gestor)
    antes = len(gestor.historial(trabajo))
    esperar_hasta(lambda: len(gestor.historial(trabajo)) > antes + 3)
    assert gestor.obtener(trabajo)["estado"] == "en_marcha"
    gestor.enviar_orden(trabajo, {"tipo": "fin"})


def numero(imagen: bytes) -> int:
    """Número de una imagen de juguete «imagen-N»."""
    return int(imagen.decode().split("-")[1])


def test_las_imagenes_llegan_en_binario_y_en_orden(app_y_gestor) -> None:
    cliente, gestor = app_y_gestor
    trabajo = gestor.crear("entrenamiento", "tareas_app:imagenes_hasta_orden")["id"]
    esperar_hasta(lambda: gestor.ultima_imagen(trabajo) is not None)
    with cliente.websocket_connect(f"/api/trabajos/{trabajo}") as ws:
        leer(ws), leer(ws)
        imagenes = leer_hasta(ws, lambda m: isinstance(m, bytes) and numero(m) >= 20)
        ws.send_json({"tipo": "fin"})
        resto, _ = leer_hasta_cierre(ws)
    numeros = [numero(m) for m in imagenes + resto if isinstance(m, bytes)]
    # Pueden saltarse imágenes si el navegador va lento, pero nunca van hacia atrás.
    assert numeros == sorted(numeros)
    assert len(set(numeros)) == len(numeros)


def test_los_numeros_no_finitos_salen_como_null_por_websocket(app_y_gestor) -> None:
    cliente, gestor = app_y_gestor
    trabajo = gestor.crear("entrenamiento", "tareas_app:progreso_no_finito")["id"]
    gestor.esperar(trabajo, LIMITE)
    with cliente.websocket_connect(f"/api/trabajos/{trabajo}") as ws:
        mensajes, _ = leer_hasta_cierre(ws)
    assert mensajes[1]["progreso"] == [{"nan": None, "infinito": None, "normal": 1.5}]
    assert mensajes[0]["trabajo"]["resultado"] == {"perdida": None}


# --- a_json_finito -------------------------------------------------------------------

def test_a_json_finito() -> None:
    entrada = {"a": float("nan"), "b": [float("inf"), -float("inf"), 2.0, (1, np.float64(0.5))],
               "c": {"d": np.int64(3), "e": np.float32("nan")}, "f": "texto", "g": None,
               "h": True, 4: 1}
    salida = a_json_finito(entrada)
    assert salida == {"a": None, "b": [None, None, 2.0, [1, 0.5]],
                      "c": {"d": 3, "e": None}, "f": "texto", "g": None, "h": True, "4": 1}
    assert type(salida["c"]["d"]) is int
    json.dumps(salida, allow_nan=False)


# --- ColaEnvio ---------------------------------------------------------------------

def _imagen(i: int) -> dict[str, Any]:
    return {"tipo": "imagen", "datos": f"imagen-{i}".encode()}


def test_cola_envio_manda_una_vez_la_ultima_de_varias_imagenes() -> None:
    async def probar() -> list[Any]:
        cola = ColaEnvio()
        for i in range(5):
            cola.poner(_imagen(i))
        primero = await cola.sacar()
        return [primero, cola.vacia()]
    assert asyncio.run(probar()) == [b"imagen-4", True]


def test_cola_envio_conserva_el_orden_de_los_eventos() -> None:
    progreso = {"tipo": "progreso", "medidas": {"paso": 1}}
    estado = {"tipo": "estado", "trabajo": {"estado": "terminado"}}

    async def probar() -> list[Any]:
        cola = ColaEnvio()
        cola.poner(_imagen(0))
        cola.poner(progreso)
        cola.poner(_imagen(1))  # sustituye a la 0, que sigue en su sitio de la cola
        cola.poner(estado)
        sacados = [await cola.sacar() for _ in range(3)]
        vacia = cola.vacia()
        cola.poner(_imagen(2))  # tras enviar, una imagen nueva vuelve a entrar
        return sacados + [vacia, await cola.sacar()]

    assert asyncio.run(probar()) == [b"imagen-1", progreso, estado, True, b"imagen-2"]
