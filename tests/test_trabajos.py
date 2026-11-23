"""Tests del gestor de trabajos (src/app/trabajos.py) con las tareas de tareas_app.py.

Cada trabajo arranca un proceso spawn de verdad: así se prueban la cola, el límite,
las señales de pausa, la muerte de un proceso y las tuberías tal como funcionarán
con los entrenamientos.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator

import numpy as np
import pytest

from src.app.trabajos import (ESTADOS_FINALES, GestorTrabajos, OperacionNoValida, SinHueco,
                              TrabajoDesconocido, comprobar_ruta_tarea, resolver_tarea)

LIMITE = 30.0  # segundos como mucho para cualquier espera


def esperar_hasta(condicion: Callable[[], bool], limite: float = LIMITE) -> None:
    """Espera a que la condición se cumpla o falla el test."""
    fin = time.monotonic() + limite
    while not condicion():
        if time.monotonic() > fin:
            pytest.fail("la condición no se cumplió a tiempo")
        time.sleep(0.02)


@pytest.fixture
def crear_gestor() -> Iterator[Callable[[int], GestorTrabajos]]:
    """Crea gestores y los cierra al acabar el test, para no dejar procesos vivos."""
    gestores = []

    def crear(procesos: int = 2) -> GestorTrabajos:
        gestor = GestorTrabajos(procesos)
        gestores.append(gestor)
        return gestor

    yield crear
    for gestor in gestores:
        gestor.cerrar()


def contar(gestor: GestorTrabajos, procesos: int = 1) -> str:
    """Crea un trabajo que cuenta hasta recibir la orden de fin y espera a que avance."""
    trabajo = gestor.crear("entrenamiento", "tareas_app:contar_hasta_orden")
    if trabajo["estado"] == "en_marcha":
        esperar_hasta(lambda: len(gestor.historial(trabajo["id"])) >= 3)
    return trabajo["id"]


# --- Un trabajo de principio a fin -----------------------------------------------

def test_un_trabajo_termina_con_su_resultado_y_su_progreso(crear_gestor) -> None:
    gestor = crear_gestor()
    eventos = []
    gestor.observar(lambda id_, evento: eventos.append((id_, evento)))
    trabajo = gestor.crear("entrenamiento", "tareas_app:sumar", {"numeros": [1, 2, 3]})
    assert trabajo["estado"] == "en_marcha"
    assert trabajo["iniciado"] is not None
    final = gestor.esperar(trabajo["id"], LIMITE)
    assert final["estado"] == "terminado"
    assert final["resultado"] == {"suma": 6}
    assert final["error"] is None
    assert final["creado"] <= final["iniciado"] <= final["terminado"]
    assert gestor.historial(trabajo["id"]) == [{"paso": 0, "valor": 1}, {"paso": 1, "valor": 2},
                                               {"paso": 2, "valor": 3}]
    assert final["progresos"] == 3
    # El repartidor entrega los eventos en su hilo: puede ir algo por detrás de esperar.
    esperar_hasta(lambda: any(e["tipo"] == "estado" and e["trabajo"]["estado"] == "terminado"
                              for _, e in eventos))
    progresos =[e["medidas"] for i, e in eventos if e["tipo"] == "progreso"]
    assert progresos == gestor.historial(trabajo["id"])
    estados = [e["trabajo"]["estado"] for i, e in eventos if e["tipo"] == "estado"]
    assert estados == ["en_marcha", "terminado"]
    assert {i for i, _ in eventos} == {trabajo["id"]}


def test_listar_y_obtener(crear_gestor) -> None:
    gestor = crear_gestor()
    a = gestor.crear("entrenamiento", "tareas_app:sumar", {"numeros": [1]})
    b = gestor.crear("comparacion", "tareas_app:sumar", {"numeros": [2]})
    for trabajo in (a, b):
        gestor.esperar(trabajo["id"], LIMITE)
    assert [t["id"] for t in gestor.listar()] == [a["id"], b["id"]]
    assert gestor.obtener(b["id"])["tipo"] == "comparacion"
    assert gestor.obtener(b["id"])["argumentos"] == {"numeros": [2]}


def test_el_proceso_hijo_trabaja_con_un_hilo_y_sin_cargar_el_entrenamiento(crear_gestor) -> None:
    gestor = crear_gestor()
    final = gestor.esperar(gestor.crear("entrenamiento", "tareas_app:hilos")["id"], LIMITE)
    resultado = final["resultado"]
    assert resultado["numpy_antes"] is False
    assert set(resultado["entorno"].values()) == {"1"}
    assert resultado["torch"] == 1
    assert resultado["training"] == []


# --- Cola y límite ---------------------------------------------------------------

def test_con_el_limite_lleno_los_demas_esperan_en_cola_por_orden(crear_gestor) -> None:
    gestor = crear_gestor(1)
    primero = contar(gestor)
    segundo = gestor.crear("entrenamiento", "tareas_app:sumar", {"numeros": [2]})
    tercero = gestor.crear("entrenamiento", "tareas_app:sumar", {"numeros": [3]})
    assert segundo["estado"] == tercero["estado"] == "en_cola"
    assert segundo["iniciado"] is None
    assert gestor.libres() == 0
    gestor.enviar_orden(primero, {"tipo": "fin"})
    finales = [gestor.esperar(t, LIMITE) for t in (primero, segundo["id"], tercero["id"])]
    assert [f["estado"] for f in finales] == ["terminado"] * 3
    # Por orden de llegada: cada uno empieza cuando acaba el anterior.
    assert finales[0]["terminado"] <= finales[1]["iniciado"]
    assert finales[1]["terminado"] <= finales[2]["iniciado"]
    assert gestor.libres() == 1


def test_varios_a_la_vez_hasta_el_limite(crear_gestor) -> None:
    gestor = crear_gestor(2)
    a, b = contar(gestor), contar(gestor)
    c = gestor.crear("entrenamiento", "tareas_app:sumar", {"numeros": [1]})
    assert [gestor.obtener(t)["estado"] for t in (a, b)] == ["en_marcha"] * 2
    assert c["estado"] == "en_cola"
    gestor.enviar_orden(b, {"tipo": "fin"})
    assert gestor.esperar(c["id"], LIMITE)["estado"] == "terminado"
    assert gestor.obtener(a)["estado"] == "en_marcha"
    gestor.enviar_orden(a, {"tipo": "fin"})
    gestor.esperar(a, LIMITE)


def test_un_interactivo_sin_hueco_se_rechaza_y_con_hueco_empieza(crear_gestor) -> None:
    gestor = crear_gestor(1)
    entrenamiento = contar(gestor)
    with pytest.raises(SinHueco, match="No hay núcleos libres"):
        gestor.crear("sesion", "tareas_app:sumar", {"numeros": [1]}, interactivo=True)
    assert len(gestor.listar()) == 1  # el rechazado no queda registrado
    gestor.enviar_orden(entrenamiento, {"tipo": "fin"})
    gestor.esperar(entrenamiento, LIMITE)
    sesion = gestor.crear("sesion", "tareas_app:sumar", {"numeros": [1]}, interactivo=True)
    assert sesion["estado"] == "en_marcha"
    assert sesion["interactivo"] is True


def test_una_sesion_ocupa_hueco_y_los_entrenamientos_esperan(crear_gestor) -> None:
    gestor = crear_gestor(1)
    sesion = gestor.crear("sesion", "tareas_app:contar_hasta_orden", interactivo=True)
    entrenamiento = gestor.crear("entrenamiento", "tareas_app:sumar", {"numeros": [1]})
    assert entrenamiento["estado"] == "en_cola"
    gestor.enviar_orden(sesion["id"], {"tipo": "fin"})
    assert gestor.esperar(entrenamiento["id"], LIMITE)["estado"] == "terminado"


# --- Pausa y reanudación ---------------------------------------------------------

def test_en_pausa_no_avanza_y_al_reanudar_sigue(crear_gestor) -> None:
    gestor = crear_gestor(1)
    trabajo = contar(gestor)
    assert gestor.pausar(trabajo)["estado"] == "pausado"
    time.sleep(0.3)  # lo que ya estaba en la tubería termina de llegar
    antes = len(gestor.historial(trabajo))
    time.sleep(0.5)
    assert len(gestor.historial(trabajo)) == antes
    assert gestor.reanudar(trabajo)["estado"] == "en_marcha"
    esperar_hasta(lambda: len(gestor.historial(trabajo)) > antes + 2)
    gestor.enviar_orden(trabajo, {"tipo": "fin"})
    assert gestor.esperar(trabajo, LIMITE)["estado"] == "terminado"
    # Sigue donde estaba: los pasos no se saltan ni se repiten.
    pasos = [p["paso"] for p in gestor.historial(trabajo)]
    assert pasos == list(range(len(pasos)))


def test_en_pausa_sigue_ocupando_su_hueco(crear_gestor) -> None:
    gestor = crear_gestor(1)
    trabajo = contar(gestor)
    gestor.pausar(trabajo)
    otro = gestor.crear("entrenamiento", "tareas_app:sumar", {"numeros": [1]})
    assert otro["estado"] == "en_cola"
    with pytest.raises(SinHueco):
        gestor.crear("sesion", "tareas_app:sumar", {"numeros": [1]}, interactivo=True)
    gestor.reanudar(trabajo)
    gestor.enviar_orden(trabajo, {"tipo": "fin"})
    assert gestor.esperar(otro["id"], LIMITE)["estado"] == "terminado"


def test_las_ordenes_esperan_mientras_esta_en_pausa(crear_gestor) -> None:
    gestor = crear_gestor(1)
    trabajo = contar(gestor)
    gestor.pausar(trabajo)
    gestor.enviar_orden(trabajo, {"tipo": "vista", "activa": True})
    gestor.enviar_orden(trabajo, {"tipo": "fin"})
    time.sleep(0.3)
    assert gestor.obtener(trabajo)["estado"] == "pausado"
    gestor.reanudar(trabajo)
    final = gestor.esperar(trabajo, LIMITE)
    assert final["resultado"]["ordenes"] == [{"tipo": "vista", "activa": True}, {"tipo": "fin"}]
    # Al reanudar, una sola llamada a ordenes() devuelve todas las que esperaban.
    assert final["resultado"]["lotes"] == [2]


@pytest.mark.parametrize("accion, estado", [("pausar", "en_cola"), ("reanudar", "en_cola"),
                                            ("reanudar", "en_marcha")])
def test_pausar_o_reanudar_en_otro_estado_no_es_valido(crear_gestor, accion, estado) -> None:
    gestor = crear_gestor(1)
    primero = contar(gestor)
    objetivo = primero if estado == "en_marcha" else gestor.crear(
        "entrenamiento", "tareas_app:sumar", {"numeros": [1]})["id"]
    assert gestor.obtener(objetivo)["estado"] == estado
    with pytest.raises(OperacionNoValida, match=accion):
        getattr(gestor, accion)(objetivo)
    gestor.enviar_orden(primero, {"tipo": "fin"})


def test_pausar_dos_veces_no_es_valido(crear_gestor) -> None:
    gestor = crear_gestor(1)
    trabajo = contar(gestor)
    gestor.pausar(trabajo)
    with pytest.raises(OperacionNoValida, match="pausado"):
        gestor.pausar(trabajo)
    gestor.reanudar(trabajo)
    gestor.enviar_orden(trabajo, {"tipo": "fin"})


# --- Cancelar ----------------------------------------------------------------------

def test_cancelar_uno_en_cola_no_lo_arranca_nunca(crear_gestor) -> None:
    gestor = crear_gestor(1)
    primero = contar(gestor)
    en_cola = gestor.crear("entrenamiento", "tareas_app:sumar", {"numeros": [1]})
    assert gestor.cancelar(en_cola["id"])["estado"] == "cancelado"
    gestor.enviar_orden(primero, {"tipo": "fin"})
    gestor.esperar(primero, LIMITE)
    final = gestor.obtener(en_cola["id"])
    assert final["estado"] == "cancelado"
    assert final["iniciado"] is None
    assert final["resultado"] is None


@pytest.mark.parametrize("pausado", [False, True], ids=["en_marcha", "pausado"])
def test_cancelar_en_marcha_o_en_pausa_termina_el_proceso_y_libera_el_hueco(
        crear_gestor, pausado) -> None:
    gestor = crear_gestor(1)
    trabajo = contar(gestor)
    proceso = gestor._trabajos[trabajo].proceso
    siguiente = gestor.crear("entrenamiento", "tareas_app:sumar", {"numeros": [5]})
    if pausado:
        gestor.pausar(trabajo)
    gestor.cancelar(trabajo)
    final = gestor.esperar(trabajo, LIMITE)
    assert final["estado"] == "cancelado"
    assert final["resultado"] is None and final["error"] is None
    assert not proceso.is_alive()
    assert gestor.esperar(siguiente["id"], LIMITE)["resultado"] == {"suma": 5}


@pytest.mark.parametrize("tarea, argumentos", [("tareas_app:sumar", {"numeros": [1]}),
                                               ("tareas_app:fallar", {})])
def test_cancelar_uno_acabado_no_es_valido(crear_gestor, tarea, argumentos) -> None:
    gestor = crear_gestor()
    trabajo = gestor.crear("entrenamiento", tarea, argumentos)
    gestor.esperar(trabajo["id"], LIMITE)
    with pytest.raises(OperacionNoValida, match="cancelar"):
        gestor.cancelar(trabajo["id"])


def test_cancelar_dos_veces_no_es_valido(crear_gestor) -> None:
    gestor = crear_gestor(1)
    trabajo = contar(gestor)
    gestor.cancelar(trabajo)
    with pytest.raises(OperacionNoValida):
        gestor.cancelar(trabajo)
    with pytest.raises(OperacionNoValida):
        gestor.pausar(trabajo)
    with pytest.raises(OperacionNoValida):
        gestor.enviar_orden(trabajo, {"tipo": "fin"})


# --- Fallos ------------------------------------------------------------------------

@pytest.mark.parametrize("tarea, argumentos, error", [
    ("tareas_app:fallar", {"mensaje": "datos rotos"}, "ValueError('datos rotos')"),
    ("tareas_app:no_devolver_diccionario", {}, "TypeError"),
    ("tareas_app:no_existe", {}, "No se encuentra la tarea"),
    ("modulo_que_no_existe:tarea", {}, "No se encuentra la tarea"),
    ("tareas_app:no_es_funcion", {}, "no es una función"),
])
def test_un_error_de_la_tarea_deja_el_trabajo_fallido(crear_gestor, tarea, argumentos,
                                                      error) -> None:
    gestor = crear_gestor()
    final = gestor.esperar(gestor.crear("entrenamiento", tarea, argumentos)["id"], LIMITE)
    assert final["estado"] == "fallido"
    assert error in final["error"]
    assert final["resultado"] is None


def test_si_el_proceso_muere_cuenta_como_fallido_y_los_demas_siguen(crear_gestor) -> None:
    gestor = crear_gestor(1)
    muere = gestor.crear("entrenamiento", "tareas_app:morir")
    despues = gestor.crear("entrenamiento", "tareas_app:sumar", {"numeros": [4]})
    final = gestor.esperar(muere["id"], LIMITE)
    assert final["estado"] == "fallido"
    assert "código de salida -9" in final["error"]
    assert gestor.historial(muere["id"]) == [{"paso": 0}]
    assert gestor.esperar(despues["id"], LIMITE)["resultado"] == {"suma": 4}


# --- Órdenes e imágenes --------------------------------------------------------------

def test_las_ordenes_llegan_en_orden(crear_gestor) -> None:
    gestor = crear_gestor()
    trabajo = contar(gestor)
    ordenes = [{"tipo": "vista", "activa": True}, {"tipo": "vista", "activa": False},
               {"tipo": "fin"}]
    for orden in ordenes:
        gestor.enviar_orden(trabajo, orden)
    assert gestor.esperar(trabajo, LIMITE)["resultado"]["ordenes"] == ordenes


def test_una_orden_a_un_trabajo_acabado_o_en_cola_no_es_valida(crear_gestor) -> None:
    gestor = crear_gestor(1)
    primero = contar(gestor)
    en_cola = gestor.crear("entrenamiento", "tareas_app:sumar", {"numeros": [1]})
    with pytest.raises(OperacionNoValida, match="en_cola"):
        gestor.enviar_orden(en_cola["id"], {"tipo": "fin"})
    with pytest.raises(ValueError, match="diccionario"):
        gestor.enviar_orden(primero, "fin")
    gestor.enviar_orden(primero, {"tipo": "fin"})
    gestor.esperar(primero, LIMITE)
    with pytest.raises(OperacionNoValida, match="terminado"):
        gestor.enviar_orden(primero, {"tipo": "fin"})


def test_las_imagenes_llegan_y_se_guarda_la_ultima(crear_gestor) -> None:
    gestor = crear_gestor()
    imagenes = []
    gestor.observar(lambda id_, e: imagenes.append(e["datos"]) if e["tipo"] == "imagen" else None)
    trabajo = gestor.crear("entrenamiento", "tareas_app:mandar_imagenes", {"cuantas": 5})
    assert gestor.ultima_imagen(trabajo["id"]) is None or isinstance(
        gestor.ultima_imagen(trabajo["id"]), bytes)
    gestor.esperar(trabajo["id"], LIMITE)
    esperar_hasta(lambda: gestor.ultima_imagen(trabajo["id"]) == b"imagen-4")
    esperar_hasta(lambda: len(imagenes) == 5)
    assert imagenes == [f"imagen-{i}".encode() for i in range(5)]
    assert gestor.historial(trabajo["id"]) == [{"imagenes": 5}]


def test_un_trabajo_sin_imagenes_no_tiene_ultima_imagen(crear_gestor) -> None:
    gestor = crear_gestor()
    trabajo = gestor.crear("entrenamiento", "tareas_app:sumar", {"numeros": [1]})
    gestor.esperar(trabajo["id"], LIMITE)
    assert gestor.ultima_imagen(trabajo["id"]) is None


# --- Observadores --------------------------------------------------------------------

def test_un_observador_que_falla_no_para_el_gestor_y_se_puede_quitar(crear_gestor) -> None:
    gestor = crear_gestor()
    vistos = []

    def roto(id_, evento):
        raise RuntimeError("observador roto")

    gestor.observar(roto)
    dejar = gestor.observar(lambda id_, e: vistos.append(e["tipo"]))
    primero = gestor.crear("entrenamiento", "tareas_app:sumar", {"numeros": [1]})
    assert gestor.esperar(primero["id"], LIMITE)["estado"] == "terminado"
    esperar_hasta(lambda: "progreso" in vistos)
    dejar()
    dejar()  # quitarlo dos veces no falla
    cuantos = len(vistos)
    segundo = gestor.crear("entrenamiento", "tareas_app:sumar", {"numeros": [1]})
    gestor.esperar(segundo["id"], LIMITE)
    assert len(vistos) == cuantos


def test_el_estado_final_se_avisa_una_vez_por_trabajo(crear_gestor) -> None:
    gestor = crear_gestor(1)
    finales = []
    cerrojo = threading.Lock()

    def anotar(id_, evento):
        if evento["tipo"] == "estado" and evento["trabajo"]["estado"] in {
                e.value for e in ESTADOS_FINALES}:
            with cerrojo:
                finales.append(id_)

    gestor.observar(anotar)
    ids = [gestor.crear("entrenamiento", "tareas_app:sumar", {"numeros": [i]})["id"]
           for i in range(3)]
    for id_ in ids:
        gestor.esperar(id_, LIMITE)
    esperar_hasta(lambda: len(finales) == 3)
    assert sorted(finales) == sorted(ids)


def test_los_estados_de_cada_trabajo_llegan_en_orden(crear_gestor) -> None:
    """Con trabajos rápidos y cola, ningún aviso llega después del de un estado posterior."""
    gestor = crear_gestor(2)
    estados: dict[str, list[str]] = {}
    gestor.observar(lambda id_, e: estados.setdefault(id_, []).append(e["trabajo"]["estado"])
                    if e["tipo"] == "estado" else None)
    ids = [gestor.crear("entrenamiento", "tareas_app:sumar", {"numeros": [i]})["id"]
           for i in range(6)]
    pausado = contar(gestor)  # entra en cola detrás de los otros seis
    esperar_hasta(lambda: len(gestor.historial(pausado)) >= 3)
    gestor.pausar(pausado)
    gestor.reanudar(pausado)
    gestor.enviar_orden(pausado, {"tipo": "fin"})
    for id_ in [*ids, pausado]:
        gestor.esperar(id_, LIMITE)
    esperar_hasta(lambda: all(estados.get(i, [""])[-1] == "terminado" for i in [*ids, pausado]))
    for id_ in ids:
        assert estados[id_] in (["en_marcha", "terminado"],
                                ["en_cola", "en_marcha", "terminado"])
    assert estados[pausado] in (["en_marcha", "pausado", "en_marcha", "terminado"],
                                ["en_cola", "en_marcha", "pausado", "en_marcha", "terminado"])
    assert any(estados[i][0] == "en_cola" for i in ids)


def test_cancelar_avisa_del_estado_cancelado(crear_gestor) -> None:
    gestor = crear_gestor(1)
    estados = []
    gestor.observar(lambda id_, e: estados.append((id_, e["trabajo"]["estado"]))
                    if e["tipo"] == "estado" else None)
    primero = contar(gestor)
    en_cola = gestor.crear("entrenamiento", "tareas_app:sumar", {"numeros": [1]})["id"]
    gestor.cancelar(en_cola)
    gestor.cancelar(primero)
    gestor.esperar(primero, LIMITE)
    esperar_hasta(lambda: (primero, "cancelado") in estados)
    assert [e for i, e in estados if i == en_cola] == ["en_cola", "cancelado"]
    assert [e for i, e in estados if i == primero] == ["en_marcha", "cancelado"]


# --- Cerrar --------------------------------------------------------------------------

def test_cerrar_cancela_todo_y_no_deja_procesos(crear_gestor) -> None:
    gestor = crear_gestor(2)
    a, b = contar(gestor), contar(gestor)
    gestor.pausar(b)
    en_cola = gestor.crear("entrenamiento", "tareas_app:sumar", {"numeros": [1]})["id"]
    procesos = [gestor._trabajos[t].proceso for t in (a, b)]
    gestor.cerrar()
    assert [gestor.obtener(t)["estado"] for t in (a, b, en_cola)] == ["cancelado"] * 3
    assert not any(p.is_alive() for p in procesos)
    with pytest.raises(RuntimeError, match="cerrado"):
        gestor.crear("entrenamiento", "tareas_app:sumar", {"numeros": [1]})


# --- Validación ----------------------------------------------------------------------

@pytest.mark.parametrize("procesos", [0, -1, True, 1.5, "2", None])
def test_procesos_no_validos(procesos) -> None:
    with pytest.raises(ValueError, match="procesos"):
        GestorTrabajos(procesos)


def test_procesos_admite_enteros_de_numpy() -> None:
    assert GestorTrabajos(np.int64(3)).procesos == 3


@pytest.mark.parametrize("ruta", ["", "tareas_app", "tareas_app:", ":sumar", "a:b:c",
                                  " tareas_app:sumar", None, 3])
def test_rutas_de_tarea_no_validas(ruta) -> None:
    with pytest.raises(ValueError, match="modulo:funcion"):
        comprobar_ruta_tarea(ruta)


def test_resolver_tarea_devuelve_la_funcion() -> None:
    import tareas_app

    assert resolver_tarea("tareas_app:sumar") is tareas_app.sumar


@pytest.mark.parametrize("tipo, tarea, argumentos, mensaje", [
    ("", "tareas_app:sumar", {}, "tipo"),
    (None, "tareas_app:sumar", {}, "tipo"),
    ("entrenamiento", "sin_dos_puntos", {}, "modulo:funcion"),
    ("entrenamiento", "tareas_app:sumar", [1], "argumentos"),
])
def test_crear_rechaza_datos_no_validos_sin_registrar_nada(crear_gestor, tipo, tarea,
                                                           argumentos, mensaje) -> None:
    gestor = crear_gestor()
    with pytest.raises(ValueError, match=mensaje):
        gestor.crear(tipo, tarea, argumentos)
    assert gestor.listar() == []


@pytest.mark.parametrize("accion", ["obtener", "historial", "ultima_imagen", "pausar",
                                    "reanudar", "cancelar", "esperar"])
def test_un_id_desconocido_da_trabajo_desconocido(crear_gestor, accion) -> None:
    with pytest.raises(TrabajoDesconocido):
        getattr(crear_gestor(), accion)("no-existe")
    with pytest.raises(TrabajoDesconocido):
        crear_gestor().enviar_orden("no-existe", {"tipo": "fin"})


def test_esperar_con_plazo_lanza_timeout(crear_gestor) -> None:
    gestor = crear_gestor()
    trabajo = contar(gestor)
    with pytest.raises(TimeoutError, match="en_marcha"):
        gestor.esperar(trabajo, 0.1)
    gestor.enviar_orden(trabajo, {"tipo": "fin"})
