"""Tests del gestor de trabajos (src/app/trabajos.py) con las tareas de tareas_app.py.

Cada trabajo arranca un proceso spawn de verdad: así se prueban la cola, el límite,
las señales de pausa, la muerte de un proceso y las tuberías tal como funcionarán
con los entrenamientos.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from collections.abc import Callable, Iterator

import numpy as np
import pytest

from src.app import trabajos
from src.app.trabajos import (ESTADOS_FINALES, GestorTrabajos, OperacionNoValida, SinHueco,
                              TrabajoDesconocido, comprobar_ruta_tarea, resolver_tarea)

LIMITE = 30.0  # segundos como mucho para cualquier espera


def vivo(pid: int) -> bool:
    """Si el proceso existe todavía (tampoco como zombi sin recoger)."""
    return Path(f"/proc/{pid}").exists()


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
    b = gestor.crear("comparacion", "tareas_app:sumar", {"numeros": [2]},
                     publicos={"cuantos": 1})
    for trabajo in (a, b):
        gestor.esperar(trabajo["id"], LIMITE)
    assert [t["id"] for t in gestor.listar()] == [a["id"], b["id"]]
    assert gestor.obtener(b["id"])["tipo"] == "comparacion"
    assert gestor.obtener(b["id"])["argumentos"] == {"cuantos": 1}


# --- argumentos públicos ---------------------------------------------------------------

def test_el_navegador_ve_los_publicos_y_la_tarea_recibe_los_suyos(crear_gestor) -> None:
    gestor = crear_gestor()
    publicos = {"entorno": "cartpole", "partidas": 3}
    trabajo = gestor.crear("comparacion", "tareas_app:sumar",
                           {"numeros": [4, 5], "ruta": "/interna/configs"}, publicos=publicos)
    assert trabajo["argumentos"] == publicos
    final = gestor.esperar(trabajo["id"], LIMITE)
    assert final["resultado"] == {"suma": 9}  # la tarea recibió los suyos
    assert final["argumentos"] == publicos
    assert all(t["argumentos"] == publicos for t in gestor.listar())


def test_sin_publicos_no_se_ensena_nada_de_la_tarea(crear_gestor) -> None:
    gestor = crear_gestor()
    trabajo = gestor.crear("entrenamiento", "tareas_app:sumar", {"numeros": [1]})
    assert trabajo["argumentos"] == {}
    assert gestor.esperar(trabajo["id"], LIMITE)["argumentos"] == {}


def test_los_publicos_son_una_copia(crear_gestor) -> None:
    gestor = crear_gestor()
    publicos = {"modelos": ["a"]}
    trabajo = gestor.crear("comparacion", "tareas_app:sumar", {"numeros": [1]},
                           publicos=publicos)
    publicos["modelos"].append("b")
    assert gestor.obtener(trabajo["id"])["argumentos"] == {"modelos": ["a"]}


@pytest.mark.parametrize("publicos, mensaje", [
    ([1], "diccionario"), ("x", "diccionario"),
    ({"ruta": Path("/a")}, "JSON"), ({"x": float("nan")}, "JSON"), ({"x": {1, 2}}, "JSON"),
])
def test_unos_publicos_que_no_valen(crear_gestor, publicos, mensaje) -> None:
    gestor = crear_gestor()
    with pytest.raises(ValueError, match=mensaje):
        gestor.crear("comparacion", "tareas_app:sumar", {"numeros": [1]}, publicos=publicos)
    assert gestor.listar() == []


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
    gestor.enviar_orden(primero, {"tipo": "finalizar"})
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
    gestor.enviar_orden(b, {"tipo": "finalizar"})
    assert gestor.esperar(c["id"], LIMITE)["estado"] == "terminado"
    assert gestor.obtener(a)["estado"] == "en_marcha"
    gestor.enviar_orden(a, {"tipo": "finalizar"})
    gestor.esperar(a, LIMITE)


def test_un_interactivo_sin_hueco_se_rechaza_y_con_hueco_empieza(crear_gestor) -> None:
    gestor = crear_gestor(1)
    entrenamiento = contar(gestor)
    with pytest.raises(SinHueco, match="No hay núcleos libres"):
        gestor.crear("sesion", "tareas_app:sumar", {"numeros": [1]}, interactivo=True)
    assert len(gestor.listar()) == 1  # el rechazado no queda registrado
    gestor.enviar_orden(entrenamiento, {"tipo": "finalizar"})
    gestor.esperar(entrenamiento, LIMITE)
    sesion = gestor.crear("sesion", "tareas_app:sumar", {"numeros": [1]}, interactivo=True)
    assert sesion["estado"] == "en_marcha"
    assert sesion["interactivo"] is True


def test_una_sesion_ocupa_hueco_y_los_entrenamientos_esperan(crear_gestor) -> None:
    gestor = crear_gestor(1)
    sesion = gestor.crear("sesion", "tareas_app:contar_hasta_orden", interactivo=True)
    entrenamiento = gestor.crear("entrenamiento", "tareas_app:sumar", {"numeros": [1]})
    assert entrenamiento["estado"] == "en_cola"
    gestor.enviar_orden(sesion["id"], {"tipo": "finalizar"})
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
    gestor.enviar_orden(trabajo, {"tipo": "finalizar"})
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
    gestor.enviar_orden(trabajo, {"tipo": "finalizar"})
    assert gestor.esperar(otro["id"], LIMITE)["estado"] == "terminado"


def test_las_ordenes_esperan_mientras_esta_en_pausa(crear_gestor) -> None:
    gestor = crear_gestor(1)
    trabajo = contar(gestor)
    gestor.pausar(trabajo)
    gestor.enviar_orden(trabajo, {"tipo": "vista", "activa": True})
    gestor.enviar_orden(trabajo, {"tipo": "finalizar"})
    time.sleep(0.3)
    assert gestor.obtener(trabajo)["estado"] == "pausado"
    gestor.reanudar(trabajo)
    final = gestor.esperar(trabajo, LIMITE)
    assert final["resultado"]["ordenes"] == [{"tipo": "vista", "activa": True},
                                             {"tipo": "finalizar"}]
    # Al reanudar, una sola llamada a ordenes() devuelve todas las que esperaban.
    assert final["resultado"]["lotes"] == [2]


def test_una_orden_que_reanuda_saca_al_trabajo_de_la_pausa(crear_gestor) -> None:
    """Regresión: finalizar en pausa dejaba el trabajo pausado, ocupando su hueco, hasta
    que alguien lo reanudaba (y el frontend deshabilita Reanudar tras Finalizar)."""
    gestor = crear_gestor(1)
    estados = []
    gestor.observar(lambda id_, e: estados.append(e["trabajo"]["estado"])
                    if e["tipo"] == "estado" else None)
    trabajo = contar(gestor)
    gestor.pausar(trabajo)
    gestor.enviar_orden(trabajo, {"tipo": "finalizar"}, reanudar=True)
    final = gestor.esperar(trabajo, LIMITE)
    assert final["estado"] == "terminado"
    assert final["resultado"]["ordenes"] == [{"tipo": "finalizar"}]
    esperar_hasta(lambda: "terminado" in estados)
    assert estados == ["en_marcha", "pausado", "en_marcha", "terminado"]


def test_reanudar_con_la_orden_no_hace_nada_si_ya_estaba_en_marcha(crear_gestor) -> None:
    gestor = crear_gestor(1)
    trabajo = contar(gestor)
    gestor.enviar_orden(trabajo, {"tipo": "finalizar"}, reanudar=True)
    assert gestor.esperar(trabajo, LIMITE)["estado"] == "terminado"


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
    gestor.enviar_orden(primero, {"tipo": "finalizar"})


def test_pausar_dos_veces_no_es_valido(crear_gestor) -> None:
    gestor = crear_gestor(1)
    trabajo = contar(gestor)
    gestor.pausar(trabajo)
    with pytest.raises(OperacionNoValida, match="pausado"):
        gestor.pausar(trabajo)
    gestor.reanudar(trabajo)
    gestor.enviar_orden(trabajo, {"tipo": "finalizar"})


# --- Cancelar ----------------------------------------------------------------------

def test_cancelar_uno_en_cola_no_lo_arranca_nunca(crear_gestor) -> None:
    gestor = crear_gestor(1)
    primero = contar(gestor)
    en_cola = gestor.crear("entrenamiento", "tareas_app:sumar", {"numeros": [1]})
    assert gestor.cancelar(en_cola["id"])["estado"] == "cancelado"
    gestor.enviar_orden(primero, {"tipo": "finalizar"})
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
    proceso = gestor._trabajos[trabajo].proceso.pid
    siguiente = gestor.crear("entrenamiento", "tareas_app:sumar", {"numeros": [5]})
    if pausado:
        gestor.pausar(trabajo)
    gestor.cancelar(trabajo)
    final = gestor.esperar(trabajo, LIMITE)
    assert final["estado"] == "cancelado"
    assert final["resultado"] is None and final["error"] is None
    assert not vivo(proceso)
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
        gestor.enviar_orden(trabajo, {"tipo": "finalizar"})


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
               {"tipo": "finalizar"}]
    for orden in ordenes:
        gestor.enviar_orden(trabajo, orden)
    assert gestor.esperar(trabajo, LIMITE)["resultado"]["ordenes"] == ordenes


def test_una_orden_a_un_trabajo_acabado_o_en_cola_no_es_valida(crear_gestor) -> None:
    gestor = crear_gestor(1)
    primero = contar(gestor)
    en_cola = gestor.crear("entrenamiento", "tareas_app:sumar", {"numeros": [1]})
    with pytest.raises(OperacionNoValida, match="en_cola"):
        gestor.enviar_orden(en_cola["id"], {"tipo": "finalizar"})
    with pytest.raises(ValueError, match="diccionario"):
        gestor.enviar_orden(primero, "fin")
    gestor.enviar_orden(primero, {"tipo": "finalizar"})
    gestor.esperar(primero, LIMITE)
    with pytest.raises(OperacionNoValida, match="terminado"):
        gestor.enviar_orden(primero, {"tipo": "finalizar"})


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
    gestor.enviar_orden(pausado, {"tipo": "finalizar"})
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
    procesos = [gestor._trabajos[t].proceso.pid for t in (a, b)]
    gestor.cerrar()
    assert [gestor.obtener(t)["estado"] for t in (a, b, en_cola)] == ["cancelado"] * 3
    assert not any(vivo(p) for p in procesos)
    with pytest.raises(RuntimeError, match="cerrado"):
        gestor.crear("entrenamiento", "tareas_app:sumar", {"numeros": [1]})


def test_cerrar_mata_a_los_que_no_terminan_por_las_buenas(crear_gestor, monkeypatch) -> None:
    """Regresión: si un proceso seguía vivo al acabar el plazo, cerrar volvía con él
    vivo y el trabajo en_marcha."""
    monkeypatch.setattr(trabajos, "ESPERA_TERMINAR", 60.0)  # que no lo mate el temporizador
    gestor = crear_gestor(1)
    terco = gestor.crear("entrenamiento", "tareas_app:ignorar_terminar")["id"]
    esperar_hasta(lambda: len(gestor.historial(terco)) >= 3)
    proceso = gestor._trabajos[terco].proceso.pid
    inicio = time.monotonic()
    gestor.cerrar(timeout=0.3)
    assert time.monotonic() - inicio < 10
    assert not vivo(proceso)
    assert gestor.obtener(terco)["estado"] == "cancelado"


# --- Procesos que no arrancan o no se dejan terminar ----------------------------------

def test_un_trabajo_que_no_arranca_falla_y_la_cola_sigue(crear_gestor) -> None:
    """Regresión: si el proceso no llegaba a arrancar (argumentos que no se pueden pasar
    por pickle), el trabajo se quedaba en_cola para siempre, el hilo que lo lanzaba
    moría y los siguientes de la cola no empezaban nunca."""
    gestor = crear_gestor(1)
    primero = contar(gestor)
    roto = gestor.crear("entrenamiento", "tareas_app:sumar", {"numeros": lambda: 1})["id"]
    siguiente = gestor.crear("entrenamiento", "tareas_app:sumar", {"numeros": [2, 3]})["id"]
    gestor.enviar_orden(primero, {"tipo": "finalizar"})
    final = gestor.esperar(roto, LIMITE)
    assert final["estado"] == "fallido"
    assert "No se pudo arrancar el proceso" in final["error"]
    assert final["terminado"] is not None
    assert gestor.esperar(siguiente, LIMITE)["resultado"] == {"suma": 5}
    assert gestor.libres() == 1


def test_un_trabajo_que_no_arranca_con_hueco_libre_falla_al_crearlo(crear_gestor) -> None:
    gestor = crear_gestor(1)
    roto = gestor.crear("sesion", "tareas_app:sumar", {"numeros": lambda: 1}, interactivo=True)
    assert roto["estado"] == "fallido" and "No se pudo arrancar" in roto["error"]
    assert gestor.libres() == 1
    assert gestor.esperar(gestor.crear("entrenamiento", "tareas_app:sumar",
                                       {"numeros": [1]})["id"], LIMITE)["estado"] == "terminado"


def test_si_falla_algo_tras_arrancar_el_proceso_no_queda_vivo(crear_gestor,
                                                              monkeypatch) -> None:
    """Regresión: si fallaba algo justo después de start() (abrir el pidfd sin
    descriptores libres), el trabajo quedaba fallido pero su proceso seguía vivo y sin
    vigilar."""
    import multiprocessing

    def sin_descriptores(pid: int) -> int:
        raise OSError(24, "Too many open files")

    gestor = crear_gestor(1)
    monkeypatch.setattr(trabajos.os, "pidfd_open", sin_descriptores, raising=False)
    trabajo = gestor.crear("entrenamiento", "tareas_app:contar_hasta_orden")
    monkeypatch.undo()
    assert trabajo["estado"] == "fallido" and "Too many open files" in trabajo["error"]
    assert multiprocessing.active_children() == []
    assert gestor.libres() == 1


def test_enviar_ordenes_nunca_bloquea_el_gestor(crear_gestor) -> None:
    """Regresión: las órdenes se escribían con el cerrojo tomado; si la tarea no las leía
    y se llenaba la tubería (unas 1600), el gestor entero se quedaba esperando."""
    gestor = crear_gestor(2)
    sordo = gestor.crear("entrenamiento", "tareas_app:esperar", {"segundos": 60})["id"]
    inicio = time.monotonic()
    for _ in range(5000):
        gestor.enviar_orden(sordo, {"tipo": "accion", "accion": 1})
    assert time.monotonic() - inicio < 5
    otro = gestor.crear("entrenamiento", "tareas_app:sumar", {"numeros": [4]})["id"]
    assert gestor.esperar(otro, LIMITE)["resultado"] == {"suma": 4}
    gestor.cancelar(sordo)
    assert gestor.esperar(sordo, LIMITE)["estado"] == "cancelado"


def test_las_ordenes_siguen_llegando_en_orden_por_su_hilo(crear_gestor) -> None:
    gestor = crear_gestor(1)
    trabajo = contar(gestor)
    gestor.pausar(trabajo)
    for i in range(2000):  # más de lo que cabe en la tubería
        gestor.enviar_orden(trabajo, {"tipo": "accion", "accion": i})
    gestor.enviar_orden(trabajo, {"tipo": "finalizar"}, reanudar=True)
    final = gestor.esperar(trabajo, LIMITE)
    assert [o.get("accion") for o in final["resultado"]["ordenes"][:-1]] == list(range(2000))


def test_los_trabajos_terminados_no_dejan_descriptores_abiertos(crear_gestor) -> None:
    """Regresión: cada trabajo terminado dejaba dos descriptores abiertos (los del
    objeto Process, que no se cerraba tras recogerlo)."""
    gestor = crear_gestor(1)

    def abiertos() -> int:
        return len(list(Path("/proc/self/fd").iterdir()))

    cuentas = []
    for _ in range(6):
        trabajo = gestor.crear("entrenamiento", "tareas_app:sumar", {"numeros": [1]})["id"]
        gestor.esperar(trabajo, LIMITE)
        esperar_hasta(lambda: gestor._trabajos[trabajo].proceso is None)
        time.sleep(0.2)  # que acaben los hilos del trabajo
        cuentas.append(abiertos())
    assert cuentas[-1] <= cuentas[1], cuentas


def test_cancelar_mata_a_un_proceso_que_ignora_sigterm(crear_gestor, monkeypatch) -> None:
    monkeypatch.setattr(trabajos, "ESPERA_TERMINAR", 0.5)
    gestor = crear_gestor(1)
    terco = gestor.crear("entrenamiento", "tareas_app:ignorar_terminar")["id"]
    esperar_hasta(lambda: len(gestor.historial(terco)) >= 3)
    pid = gestor._trabajos[terco].proceso.pid
    gestor.cancelar(terco)
    assert gestor.esperar(terco, LIMITE)["estado"] == "cancelado"
    assert not vivo(pid)
    assert gestor.libres() == 1


@pytest.mark.skipif(not hasattr(os, "pidfd_open"), reason="pidfd solo existe en Linux")
def test_las_senales_van_por_el_pidfd_y_no_por_el_numero(crear_gestor, monkeypatch) -> None:
    """Las señales no usan el número del proceso, que el sistema puede reutilizar en
    cuanto el proceso termina: van por su pidfd."""
    gestor = crear_gestor(1)
    trabajo = contar(gestor)
    monkeypatch.setattr(trabajos.os, "kill",
                        lambda *a: pytest.fail("no debería usar os.kill"))
    gestor.pausar(trabajo)
    gestor.reanudar(trabajo)
    gestor.cancelar(trabajo)
    assert gestor.esperar(trabajo, LIMITE)["estado"] == "cancelado"
    # Acabado, su pidfd está cerrado y ya no se le puede mandar nada.
    assert gestor._trabajos[trabajo].pidfd is None
    with pytest.raises(OperacionNoValida):
        gestor.pausar(trabajo)


def test_la_ultima_imagen_llega_antes_que_el_estado_final(crear_gestor, monkeypatch) -> None:
    """Regresión: con el lector de imágenes retrasado, el estado terminado se avisaba
    antes que la última imagen (y el WebSocket ya podía haberse cerrado)."""
    leer = GestorTrabajos._leer_imagenes

    def lento(self, trabajo, lectura):
        time.sleep(0.5)
        leer(self, trabajo, lectura)

    monkeypatch.setattr(GestorTrabajos, "_leer_imagenes", lento)
    # Regresión: la espera al hilo de imágenes tenía tope (ESPERA_TERMINAR); con el lector
    # más lento que ese tope, el estado final volvía a salir antes que la imagen.
    monkeypatch.setattr(trabajos, "ESPERA_TERMINAR", 0.1)
    gestor = crear_gestor(1)
    eventos = []
    gestor.observar(lambda id_, e: eventos.append(e))
    trabajo = gestor.crear("entrenamiento", "tareas_app:imagenes_y_terminar", {"cuantas": 3})
    final = gestor.esperar(trabajo["id"], LIMITE)
    assert final["estado"] == "terminado"
    assert gestor.ultima_imagen(trabajo["id"]) == b"imagen-2"
    esperar_hasta(lambda: any(e["tipo"] == "estado" and e["trabajo"]["estado"] == "terminado"
                              for e in eventos))
    tipos = [e["tipo"] if e["tipo"] != "estado" else e["trabajo"]["estado"] for e in eventos]
    assert tipos[-1] == "terminado"
    assert tipos.count("imagen") == 3


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
        crear_gestor().enviar_orden("no-existe", {"tipo": "finalizar"})


def test_esperar_con_plazo_lanza_timeout(crear_gestor) -> None:
    gestor = crear_gestor()
    trabajo = contar(gestor)
    with pytest.raises(TimeoutError, match="en_marcha"):
        gestor.esperar(trabajo, 0.1)
    gestor.enviar_orden(trabajo, {"tipo": "finalizar"})


# --- Suscripción (reconectarse a mitad) ------------------------------------------------

@pytest.mark.parametrize("espera", [0.0, 0.05, 0.2])
def test_suscribirse_a_mitad_no_pierde_ni_repite_progreso(crear_gestor, espera) -> None:
    """Foto más eventos posteriores tiene que ser exactamente el historial completo.

    Se suscribe mientras el trabajo manda progreso sin parar, en varios momentos, para
    que haya eventos en la cola del repartidor justo al tomar la foto.
    """
    gestor = crear_gestor()
    trabajo = contar(gestor)
    time.sleep(espera)
    recibidos = []
    foto, dejar = gestor.suscribir(trabajo, recibidos.append)
    assert foto["trabajo"]["estado"] == "en_marcha"
    esperar_hasta(lambda: sum(e["tipo"] == "progreso" for e in recibidos) >= 5)
    gestor.enviar_orden(trabajo, {"tipo": "finalizar"})
    gestor.esperar(trabajo, LIMITE)
    esperar_hasta(lambda: any(e["tipo"] == "estado" and e["trabajo"]["estado"] == "terminado"
                              for e in recibidos))
    dejar()
    vistos = foto["historial"] + [e["medidas"] for e in recibidos if e["tipo"] == "progreso"]
    assert vistos == gestor.historial(trabajo)
    numeros = [e["n"] for e in recibidos]
    assert numeros == list(range(numeros[0], numeros[0] + len(numeros)))


def test_suscribirse_a_uno_acabado_da_la_foto_final_y_ningun_evento(crear_gestor) -> None:
    gestor = crear_gestor()
    trabajo = gestor.crear("entrenamiento", "tareas_app:mandar_imagenes", {"cuantas": 3})["id"]
    gestor.esperar(trabajo, LIMITE)
    esperar_hasta(lambda: gestor.ultima_imagen(trabajo) == b"imagen-2")
    time.sleep(0.2)  # que el repartidor acabe con lo que tuviera de este trabajo
    recibidos = []
    foto, dejar = gestor.suscribir(trabajo, recibidos.append)
    time.sleep(0.2)
    dejar()
    assert foto["trabajo"]["estado"] == "terminado"
    assert foto["historial"] == [{"imagenes": 3}]
    assert foto["imagen"] == b"imagen-2"
    assert recibidos == []


def test_la_suscripcion_solo_recibe_su_trabajo_y_se_puede_quitar(crear_gestor) -> None:
    gestor = crear_gestor(2)
    a, b = contar(gestor), contar(gestor)
    recibidos = []
    _, dejar = gestor.suscribir(a, recibidos.append)
    esperar_hasta(lambda: len(recibidos) >= 3)
    dejar()
    dejar()
    cuantos = len(recibidos)
    time.sleep(0.2)
    assert len(recibidos) <= cuantos + 1  # como mucho el que ya se estaba repartiendo
    assert all(e["tipo"] == "progreso" for e in recibidos)
    for trabajo in (a, b):
        gestor.enviar_orden(trabajo, {"tipo": "finalizar"})


def test_los_eventos_de_cada_trabajo_van_numerados_desde_1(crear_gestor) -> None:
    gestor = crear_gestor()
    numeros: dict[str, list[int]] = {}
    gestor.observar(lambda id_, e: numeros.setdefault(id_, []).append(e["n"]))
    ids = [gestor.crear("entrenamiento", "tareas_app:sumar", {"numeros": [1, 2]})["id"]
           for _ in range(2)]
    for id_ in ids:
        gestor.esperar(id_, LIMITE)
    # en_marcha, dos progresos y terminado
    esperar_hasta(lambda: all(len(numeros.get(i, [])) == 4 for i in ids))
    assert all(numeros[i] == [1, 2, 3, 4] for i in ids)


def test_suscribirse_a_un_desconocido_falla(crear_gestor) -> None:
    with pytest.raises(TrabajoDesconocido):
        crear_gestor().suscribir("no-existe", lambda e: None)



# --- Eventos propios de la tarea -----------------------------------------------------

def test_los_eventos_propios_se_guardan_y_se_reparten(crear_gestor) -> None:
    gestor = crear_gestor()
    recibidos = []
    gestor.observar(lambda id_, e: recibidos.append(e) if e["tipo"] == "partida_terminada"
                    else None)
    trabajo = gestor.crear("sesion", "tareas_app:eventos_propios")["id"]
    esperar_hasta(lambda: len(gestor.eventos_propios(trabajo)) == 2)
    foto, dejar = gestor.suscribir(trabajo, lambda e: None)
    dejar()
    assert [e["partida"]["indice"] for e in foto["eventos"]] == [0, 1]
    gestor.enviar_orden(trabajo, {"tipo": "finalizar"})
    assert gestor.esperar(trabajo, LIMITE)["resultado"] == {"partidas": 3}
    esperar_hasta(lambda: len(recibidos) == 3)
    assert [e["partida"]["indice"] for e in recibidos] == [0, 1, 2]
    assert [e["partida"]["indice"] for e in gestor.eventos_propios(trabajo)] == [0, 1, 2]
    assert gestor.historial(trabajo) == [{"paso": 1}]  # el progreso va aparte


@pytest.mark.parametrize("tipo, datos, error", [
    ("progreso", None, "Tipo de evento no válido"),
    ("estado", None, "Tipo de evento no válido"),
    ("historial", None, "Tipo de evento no válido"),
    ("imagen", None, "Tipo de evento no válido"),
    ("error", None, "Tipo de evento no válido"),
    ("", None, "Tipo de evento no válido"),
    (3, None, "Tipo de evento no válido"),
    ("mio", {"tipo": "x"}, "claves «tipo» ni «n»"),
    ("mio", {"n": 1}, "claves «tipo» ni «n»"),
])
def test_un_evento_que_no_vale_hace_fallar_la_tarea(crear_gestor, tipo, datos, error) -> None:
    gestor = crear_gestor()
    final = gestor.esperar(gestor.crear("entrenamiento", "tareas_app:evento_no_valido",
                                        {"tipo": tipo, "datos": datos})["id"], LIMITE)
    assert final["estado"] == "fallido"
    assert error in final["error"]
