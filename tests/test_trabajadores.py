"""Tests de los trabajadores del dashboard (src/training/trabajadores.py).

Primero las piezas (argumentos, progreso y órdenes) con un canal falso; después la
tarea del experto en este proceso, comparada con entrenar_experto, y por último con el
gestor de trabajos de verdad (proceso spawn, tuberías, órdenes), como la lanzará la API.
"""

from __future__ import annotations

import copy
import dataclasses
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from gymnasium import spaces

from src.app.contrato import ResultadoComparacion, ResultadoEntrenamiento
from src.app.trabajos import GestorTrabajos
from src.data.biblioteca import Biblioteca, agregar_nuevo
from src.data.demostraciones import guardar_demostraciones
from src.data.splits import crear_splits, guardar_splits, huella_archivo
from src.data.tipos import Trayectoria
from src.training import trabajadores
from src.training.barrido import cargar_datos
from src.training.bc import cargar_politica_bc, entrenar_bc_repeticion
from src.training.gail import cargar_gail, entrenar_gail
from src.training.evaluacion import evaluar_final, evaluar_politica
from src.training.trabajadores import (ControlEntrenamiento, ProgresoEpocas, ProgresoTrabajo,
                                       cargar_agente, comprobar_argumentos_comparacion,
                                       comprobar_argumentos_entrenamiento,
                                       datos_de_entrenamiento, tarea_comparar,
                                       tarea_entrenar_bc, tarea_entrenar_experto,
                                       tarea_entrenar_gail)
from src.training.trpo import cargar_experto, entrenar_experto
from src.training.vista import CallbackVista
from src.utils.configuracion import resolver_configuracion
from src.utils.seed import crear_generador

RAIZ = Path(__file__).resolve().parent.parent
CONFIG = RAIZ / "configs" / "default.yaml"
TAREA_EXPERTO = "src.training.trabajadores:tarea_entrenar_experto"
LIMITE = 120.0
# Tamaños del smoke: 2 iteraciones de 500 pasos, una evaluación de 2 partidas por iteración.
SMOKE = yaml.safe_load((RAIZ / "configs" / "smoke.yaml").read_text(encoding="utf-8"))


class CanalFalso:
    """Imita CanalTrabajo: guarda progreso e imágenes y da las órdenes que se le pongan."""

    def __init__(self, ordenes: list[list[dict]] | None = None) -> None:
        self.progresos: list[dict] = []
        self.imagenes: list[bytes] = []
        self._ordenes = list(ordenes or [])

    def progreso(self, medidas: dict) -> None:
        self.progresos.append(dict(medidas))

    def imagen(self, datos: bytes) -> None:
        self.imagenes.append(datos)

    def ordenes(self) -> list[dict]:
        return self._ordenes.pop(0) if self._ordenes else []


class RunFalso:
    """Imita un run de W&B."""

    def __init__(self) -> None:
        self.registros: list[tuple[dict, int | None]] = []

    def log(self, metricas: dict, step: int | None = None) -> None:
        self.registros.append((dict(metricas), step))


def _argumentos(tmp_path: Path, **cambios) -> dict:
    """Argumentos de un experto de CartPole con los tamaños del smoke."""
    argumentos = {"ruta_config": str(CONFIG), "biblioteca": str(tmp_path / "biblioteca"),
                  "calidad_jpeg": 80, "entorno": "cartpole", "nombre": "Mi experto",
                  "ajustes": copy.deepcopy(SMOKE), "semilla": 123, "vista": False,
                  "wandb": {"activo": False, "grupo": None}, "checkpoint": None}
    argumentos.update(cambios)
    return argumentos


# --- argumentos ------------------------------------------------------------------------

def test_argumentos_validos(tmp_path):
    argumentos = _argumentos(tmp_path)
    copia = comprobar_argumentos_entrenamiento(argumentos)
    assert copia == argumentos and copia is not argumentos


@pytest.mark.parametrize("cambio, mensaje", [
    (lambda a: a.pop("semilla"), "exactamente"),
    (lambda a: a.update(otra=1), "exactamente"),
    (lambda a: a.update(ruta_config=""), "ruta_config"),
    (lambda a: a.update(biblioteca=3), "biblioteca"),
    (lambda a: a.update(entorno="  "), "entorno"),
    (lambda a: a.update(nombre=None), "nombre"),
    (lambda a: a.update(ajustes=[]), "ajustes"),
    (lambda a: a.update(semilla=-1), "semilla"),
    (lambda a: a.update(semilla=True), "semilla"),
    (lambda a: a.update(vista="sí"), "vista"),
    (lambda a: a.update(wandb=True), "wandb"),
    (lambda a: a.update(wandb={"activo": True}), "wandb"),
    (lambda a: a.update(wandb={"activo": True, "grupo": None}), "wandb"),
    (lambda a: a.update(wandb={"activo": "no", "grupo": None}), "wandb"),
    (lambda a: a.update(wandb={"activo": False, "grupo": 3}), "wandb"),
])
def test_argumentos_no_validos(tmp_path, cambio, mensaje):
    argumentos = _argumentos(tmp_path)
    cambio(argumentos)
    with pytest.raises(ValueError, match=mensaje):
        comprobar_argumentos_entrenamiento(argumentos)


def test_argumentos_que_no_son_un_diccionario():
    with pytest.raises(ValueError, match="exactamente"):
        comprobar_argumentos_entrenamiento([("entorno", "cartpole")])


# --- progreso --------------------------------------------------------------------------

def test_un_mensaje_por_iteracion_con_su_evaluacion():
    canal, run = CanalFalso(), RunFalso()
    progreso = ProgresoTrabajo(canal, run)
    progreso.log({"trpo/retorno_real_medio": 10.0, "trpo/partidas": 3.0}, 500)
    progreso.log({"trpo/retorno_real_medio": 20.0, "trpo/partidas": 2.0}, 1000)
    # La evaluación del mismo paso va con su iteración; las listas no salen.
    progreso.log({"evaluacion/retorno_medio": 30.0, "evaluacion/retornos": [29.0, 31.0],
                  "evaluacion/iteracion": 2}, 1000)
    progreso.log({"trpo/retorno_real_medio": 40.0, "trpo/partidas": 1.0}, 1500)
    assert canal.progresos == [
        {"paso": 500, "iteracion": 1, "trpo/retorno_real_medio": 10.0, "trpo/partidas": 3.0},
        {"paso": 1000, "iteracion": 2, "trpo/retorno_real_medio": 20.0, "trpo/partidas": 2.0,
         "evaluacion/retorno_medio": 30.0, "evaluacion/iteracion": 2.0},
    ]
    progreso.vaciar()
    assert canal.progresos[-1] == {"paso": 1500, "iteracion": 3,
                                   "trpo/retorno_real_medio": 40.0, "trpo/partidas": 1.0}
    progreso.vaciar()  # sin nada pendiente no manda nada
    assert len(canal.progresos) == 3
    # A W&B llega cada llamada al momento, con registrar (solo escalares, como en la E1).
    assert [paso for _, paso in run.registros] == [500, 1000, 1000, 1500]
    assert run.registros[2][0] == {"evaluacion/retorno_medio": 30.0, "evaluacion/iteracion": 2}


def test_en_iteraciones_el_paso_es_el_numero_de_iteracion():
    canal = CanalFalso()
    progreso = ProgresoTrabajo(canal, unidad="iteraciones")
    for step in (64, 128):
        progreso.log({"trpo/partidas": 1.0}, step)
    progreso.log({"validacion/retorno_medio": 5.0}, 128)
    assert [(m["paso"], m["iteracion"]) for m in canal.progresos] == [(1, 1), (2, 2)]
    assert canal.progresos[1]["validacion/retorno_medio"] == 5.0


def test_una_evaluacion_sin_iteracion_antes_sale_sola():
    canal = CanalFalso()
    ProgresoTrabajo(canal).log({"evaluacion/retorno_medio": 5.0}, 0)
    assert canal.progresos == [{"paso": 0, "iteracion": 0, "evaluacion/retorno_medio": 5.0}]


@pytest.mark.parametrize("valor", [float("nan"), float("inf"), True, "texto", None, [1.0]])
def test_solo_salen_numeros_finitos(valor):
    canal = CanalFalso()
    progreso = ProgresoTrabajo(canal)
    progreso.log({"trpo/x": valor, "trpo/y": 1}, 10)
    progreso.vaciar()
    assert canal.progresos == [{"paso": 10, "iteracion": 1, "trpo/y": 1.0}]


def test_unidad_no_valida():
    with pytest.raises(ValueError, match="unidad"):
        ProgresoTrabajo(CanalFalso(), unidad="epocas")


# --- órdenes ---------------------------------------------------------------------------

def test_las_ordenes_encienden_la_vista_y_piden_finalizar():
    canal = CanalFalso([[{"tipo": "vista", "activa": True}],
                        [{"tipo": "desconocida"}, {"tipo": "vista", "activa": False}],
                        [], [{"tipo": "finalizar"}]])
    vista = CallbackVista(lambda datos: None, 50, 80)
    control = ControlEntrenamiento(canal, vista, ProgresoTrabajo(canal))
    control.atender_ordenes()
    assert vista.activa and not control.finalizar_pedido
    control.atender_ordenes()
    assert not vista.activa
    assert control.parar() is False
    assert control.parar() is True
    assert control.parar() is True  # se queda pedido


# --- la tarea del experto, en este proceso ---------------------------------------------

def _pesos(modelo) -> dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in modelo.policy.state_dict().items()}


def test_el_trabajador_entrena_lo_mismo_que_entrenar_experto_y_lo_guarda(tmp_path):
    """Mismo modelo que la función de la E1 con la misma config y semilla."""
    argumentos = _argumentos(tmp_path)
    canal = CanalFalso()
    resultado = tarea_entrenar_experto(argumentos, canal)
    ResultadoEntrenamiento.model_validate(resultado)
    assert resultado["semilla"] == 123
    assert resultado["modelo"]["nombre"] == "Mi experto"
    assert resultado["modelo"]["incompleto"] is False

    cfg = resolver_configuracion(CONFIG, "cartpole", SMOKE)
    cfg["experto"]["semilla"] = 123
    esperado, resumen = entrenar_experto(cfg)
    assert resultado["resumen"]["evaluaciones"] == resumen["evaluaciones"]
    assert resultado["resumen"]["pasos"] == resumen["pasos"] == 1000

    biblioteca = Biblioteca(tmp_path / "biblioteca", {"cartpole": "CartPole-v1"})
    elemento = biblioteca.obtener(resultado["modelo"]["id"])
    assert elemento["valido"] and elemento["tipo"] == "experto"
    assert elemento["origen"] == "dashboard"
    assert elemento["nombre"] == "Mi experto"
    guardado, metadatos = cargar_experto(biblioteca.ruta(resultado["modelo"]["id"]))
    pesos_guardados, pesos_esperados = _pesos(guardado), _pesos(esperado)
    assert all(torch.equal(pesos_guardados[k], pesos_esperados[k]) for k in pesos_esperados)
    assert metadatos["resumen"]["incompleto"] is False

    # Un progreso por iteración, con su evaluación (el smoke evalúa en cada una).
    assert [(m["paso"], m["iteracion"]) for m in canal.progresos] == [(500, 1), (1000, 2)]
    assert all("trpo/retorno_real_medio" in m and "evaluacion/retorno_medio" in m
               for m in canal.progresos)
    assert canal.imagenes == []


def test_con_la_vista_encendida_manda_imagenes_sin_cambiar_el_modelo(tmp_path):
    sin = tarea_entrenar_experto(_argumentos(tmp_path / "a"), CanalFalso())
    canal = CanalFalso()
    con = tarea_entrenar_experto(_argumentos(tmp_path / "b", vista=True), canal)
    assert canal.imagenes and all(datos[:2] == b"\xff\xd8" for datos in canal.imagenes)
    assert sin["resumen"]["evaluaciones"] == con["resumen"]["evaluaciones"]


def test_finalizar_guarda_el_mejor_hasta_entonces_como_incompleto(tmp_path):
    ajustes = copy.deepcopy(SMOKE)
    ajustes["experto"]["pasos_totales"] = 5000  # 10 iteraciones de 500
    ajustes["experto"]["retorno_objetivo"] = 10**6
    canal = CanalFalso([[{"tipo": "finalizar"}]])  # llega en el primer paso
    resultado = tarea_entrenar_experto(_argumentos(tmp_path, ajustes=ajustes), canal)
    assert resultado["modelo"]["incompleto"] is True
    assert resultado["resumen"]["iteraciones"] == 1 and resultado["resumen"]["pasos"] == 500
    assert resultado["resumen"]["incompleto"] is True
    assert len(canal.progresos) == 1


def test_finalizar_en_la_ultima_iteracion_no_es_incompleto(tmp_path):
    # El smoke da 2 iteraciones; la orden llega en la segunda consulta de parar, al
    # acabar la última: el entrenamiento estaba completo.
    canal = CanalFalso()
    llamadas = []
    original = canal.ordenes

    def ordenes() -> list[dict]:
        llamadas.append(1)
        return original()

    canal.ordenes = ordenes
    resultado = tarea_entrenar_experto(_argumentos(tmp_path), canal)
    total = len(llamadas)
    canal = CanalFalso([[]] * (total - 1) + [[{"tipo": "finalizar"}]])
    resultado = tarea_entrenar_experto(_argumentos(tmp_path / "otra"), canal)
    assert resultado["resumen"]["pasos"] == 1000
    assert resultado["modelo"]["incompleto"] is False


def test_con_wandb_registra_medidas_resumen_y_modelo(tmp_path, monkeypatch):
    run = RunFalso()
    run.summary = {}
    llamadas: dict[str, list] = {"abrir": [], "subir": [], "terminar": []}

    def abrir(cfg, **opciones):
        llamadas["abrir"].append((cfg["wandb"]["modo"], opciones))
        return run

    monkeypatch.setattr(trabajadores, "abrir_run", abrir)
    monkeypatch.setattr(trabajadores, "subir_artefacto",
                        lambda run, nombre, tipo, archivos, metadatos: llamadas["subir"].append(
                            (nombre, tipo, [Path(a).name for a in archivos], metadatos)))
    monkeypatch.setattr(trabajadores, "terminar_run",
                        lambda run, error=None: llamadas["terminar"].append(error))
    argumentos = _argumentos(tmp_path, wandb={"activo": True, "grupo": "dashboard-cartpole"})
    argumentos["ajustes"]["wandb"]["modo"] = "online"
    resultado = tarea_entrenar_experto(argumentos, CanalFalso())
    modo, opciones = llamadas["abrir"][0]
    assert modo == "online"
    assert opciones["grupo"] == "dashboard-cartpole" and opciones["job_type"] == "experto"
    assert opciones["nombre"] == "Mi experto" and "origen:dashboard" in opciones["etiquetas"]
    assert any("trpo/retorno_real_medio" in m for m, _ in run.registros)
    assert run.summary["pasos"] == 1000 and run.summary["incompleto"] is False
    (nombre, tipo, archivos, metadatos), = llamadas["subir"]
    assert nombre.startswith("experto-CartPole-v1-") and tipo == "experto"
    assert archivos == ["modelo.zip", "experto.json"] and metadatos["origen"] == "dashboard"
    assert llamadas["terminar"] == [None]
    assert resultado["modelo"]["incompleto"] is False


def test_sin_wandb_no_abre_ningun_run(tmp_path, monkeypatch):
    modos = []
    original = trabajadores.abrir_run
    monkeypatch.setattr(trabajadores, "abrir_run",
                        lambda cfg, **o: modos.append(cfg["wandb"]["modo"]) or original(cfg, **o))
    argumentos = _argumentos(tmp_path)
    argumentos["ajustes"]["wandb"]["modo"] = "online"  # el perfil diría online
    tarea_entrenar_experto(argumentos, CanalFalso())
    assert modos == ["disabled"]


def test_si_falla_cierra_el_run_con_el_error_y_no_guarda_nada(tmp_path, monkeypatch):
    errores = []
    monkeypatch.setattr(trabajadores, "abrir_run", lambda cfg, **o: RunFalso())
    monkeypatch.setattr(trabajadores, "terminar_run",
                        lambda run, error=None: errores.append(error))

    def fallar(*args, **kwargs):
        raise RuntimeError("se rompió")

    monkeypatch.setattr(trabajadores, "entrenar_experto", fallar)
    argumentos = _argumentos(tmp_path, wandb={"activo": True, "grupo": "g"})
    with pytest.raises(RuntimeError, match="se rompió"):
        tarea_entrenar_experto(argumentos, CanalFalso())
    assert len(errores) == 1 and isinstance(errores[0], RuntimeError)
    assert not (tmp_path / "biblioteca" / "cartpole").exists()


def test_una_configuracion_no_valida_falla_antes_de_entrenar(tmp_path, monkeypatch):
    monkeypatch.setattr(trabajadores, "entrenar_experto",
                        lambda *a, **k: pytest.fail("no debía entrenar"))
    with pytest.raises(ValueError):
        tarea_entrenar_experto(_argumentos(tmp_path, entorno="pong"), CanalFalso())


# --- con el gestor de trabajos de verdad -----------------------------------------------

def _esperar_hasta(condicion: Callable[[], bool], limite: float = LIMITE) -> None:
    fin = time.monotonic() + limite
    while not condicion():
        if time.monotonic() > fin:
            pytest.fail("la condición no se cumplió a tiempo")
        time.sleep(0.05)


@pytest.fixture
def gestor():
    gestor = GestorTrabajos(1)
    yield gestor
    gestor.cerrar()


def test_con_el_gestor_entrena_guarda_y_devuelve_el_resultado(gestor, tmp_path):
    trabajo = gestor.crear("entrenamiento", TAREA_EXPERTO, _argumentos(tmp_path, vista=True))
    final = gestor.esperar(trabajo["id"], LIMITE)
    assert final["estado"] == "terminado", final["error"]
    resultado = ResultadoEntrenamiento.model_validate(final["resultado"])
    assert not resultado.modelo.incompleto and resultado.semilla == 123
    assert [m["iteracion"] for m in gestor.historial(trabajo["id"])] == [1, 2]
    assert gestor.ultima_imagen(trabajo["id"])[:2] == b"\xff\xd8"
    biblioteca = Biblioteca(tmp_path / "biblioteca", {"cartpole": "CartPole-v1"})
    assert [e["id"] for e in biblioteca.listar("cartpole", ("experto",))] == [resultado.modelo.id]


def test_con_el_gestor_la_vista_y_finalizar_llegan_al_trabajador(gestor, tmp_path):
    ajustes = copy.deepcopy(SMOKE)
    ajustes["experto"].update(pasos_totales=100_000, retorno_objetivo=10**6, evaluar_cada=50)
    trabajo = gestor.crear("entrenamiento", TAREA_EXPERTO, _argumentos(tmp_path, ajustes=ajustes))
    _esperar_hasta(lambda: len(gestor.historial(trabajo["id"])) >= 1)
    assert gestor.ultima_imagen(trabajo["id"]) is None
    gestor.enviar_orden(trabajo["id"], {"tipo": "vista", "activa": True})
    _esperar_hasta(lambda: gestor.ultima_imagen(trabajo["id"]) is not None)
    gestor.enviar_orden(trabajo["id"], {"tipo": "finalizar"})
    final = gestor.esperar(trabajo["id"], LIMITE)
    assert final["estado"] == "terminado", final["error"]
    resultado = final["resultado"]
    assert resultado["modelo"]["incompleto"] is True
    assert resultado["resumen"]["pasos"] < 100_000
    # Al finalizar se evalúa la iteración en curso, aunque no tocara por evaluar_cada.
    resumen = resultado["resumen"]
    assert resumen["evaluaciones"][-1]["iteracion"] == resumen["iteraciones"]


def test_con_el_gestor_un_error_deja_el_trabajo_fallido(gestor, tmp_path):
    trabajo = gestor.crear("entrenamiento", TAREA_EXPERTO, _argumentos(tmp_path, semilla=-1))
    final = gestor.esperar(trabajo["id"], LIMITE)
    assert final["estado"] == "fallido" and "semilla" in final["error"]


# --- detalles del control y del trabajador -----------------------------------------------

def _modelo_dibujable():
    from src.data.entornos import crear_entorno_vectorizado
    from src.training.trpo import crear_trpo
    from src.utils.seed import fijar_seed
    cfg = resolver_configuracion(CONFIG, "cartpole", SMOKE)
    fijar_seed(0)
    venv = crear_entorno_vectorizado(cfg, cfg["trpo"]["n_entornos"], 3, dibujable=True)
    return crear_trpo(cfg, venv), venv


def test_el_mensaje_de_una_iteracion_sale_al_empezar_la_siguiente():
    """Sin esperar a la siguiente llamada de log, que tarda una iteración entera."""
    canal = CanalFalso()
    progreso = ProgresoTrabajo(canal)
    control = ControlEntrenamiento(canal, CallbackVista(lambda d: None, 50, 80), progreso)
    progreso.log({"trpo/partidas": 1.0}, 500)
    assert canal.progresos == []
    control.on_rollout_start()
    assert canal.progresos == [{"paso": 500, "iteracion": 1, "trpo/partidas": 1.0}]


def test_las_ordenes_se_leen_en_cada_paso_no_solo_al_acabar_la_iteracion():
    imagenes = []
    canal = CanalFalso([[{"tipo": "vista", "activa": True}], [{"tipo": "finalizar"}]])
    control = ControlEntrenamiento(canal, CallbackVista(imagenes.append, 50, 80),
                                   ProgresoTrabajo(canal))
    modelo, venv = _modelo_dibujable()
    try:
        venv.reset()
        control.init_callback(modelo)
        assert control.on_step() is True
        assert control.vista.activa and len(imagenes) == 1
        control.on_step()
        assert control.finalizar_pedido
    finally:
        venv.close()


def test_finalizar_cuando_se_alcanza_el_objetivo_no_es_incompleto(tmp_path):
    ajustes = copy.deepcopy(SMOKE)
    ajustes["experto"].update(pasos_totales=5000, retorno_objetivo=1.0)  # se alcanza a la primera
    canal = CanalFalso([[{"tipo": "finalizar"}]])
    resultado = tarea_entrenar_experto(_argumentos(tmp_path, ajustes=ajustes), canal)
    assert resultado["resumen"]["objetivo_alcanzado"] and resultado["resumen"]["pasos"] == 500
    assert resultado["modelo"]["incompleto"] is False


def test_fps_entorno():
    cfg = resolver_configuracion(CONFIG, "cartpole")
    assert trabajadores.fps_entorno(cfg) == 50


@pytest.mark.parametrize("metadata", [{}, {"render_fps": 0}, {"render_fps": True},
                                      {"render_fps": 12.5}])
def test_fps_entorno_sin_render_fps(metadata, monkeypatch):
    class Entorno:
        def __init__(self):
            self.metadata = metadata
            self.cerrado = False

        def close(self):
            self.cerrado = True

    entorno = Entorno()
    monkeypatch.setattr(trabajadores, "crear_entorno", lambda cfg, semilla: entorno)
    with pytest.raises(ValueError, match="render_fps"):
        trabajadores.fps_entorno({"entorno": {"id": "Raro-v0"}})
    assert entorno.cerrado


# --- BC y GAIL ---------------------------------------------------------------------------


TAREA_GAIL = "src.training.trabajadores:tarea_entrenar_gail"
TAREA_BC = "src.training.trabajadores:tarea_entrenar_bc"
CARTPOLE_OBS = spaces.Box(np.array([-4.8, -np.inf, -0.418, -np.inf], np.float32),
                          np.array([4.8, np.inf, 0.418, np.inf], np.float32))


def _partida_cartpole(pasos: int, identificador: int) -> Trayectoria:
    """Partida que cabe en CartPole (no tiene por qué ser buena: solo se imita)."""
    estados = (np.arange((pasos + 1) * 4, dtype=np.float32).reshape(pasos + 1, 4) * 1e-3
               + np.float32(identificador * 1e-2))
    terminados = np.zeros(pasos, dtype=bool)
    terminados[-1] = True
    return Trayectoria(identificador, estados[:-1], np.arange(pasos, dtype=np.int64) % 2,
                       estados[1:], np.ones(pasos, dtype=np.float32), terminados,
                       np.zeros(pasos, dtype=bool))


def _conjunto(raiz: Path, partidas: int = 6, splits: bool = True) -> str:
    """Conjunto de CartPole en la biblioteca; con splits, 2/3 train y subconjunto [1]."""
    trayectorias = [_partida_cartpole(20 + i, i) for i in range(partidas)]

    def escribir(carpeta: Path) -> None:
        npz = guardar_demostraciones(
            carpeta / "demostraciones.npz", trayectorias,
            {"entorno": "CartPole-v1", "experto": "x", "procedencia": {},
             "configuracion_experto": {}}, CARTPOLE_OBS, spaces.Discrete(2))
        if splits:
            validacion = partidas // 3
            guardar_splits(carpeta / "splits.json", crear_splits(
                trayectorias, partidas - validacion, validacion, [1], crear_generador(0),
                huella_archivo(npz)))

    return agregar_nuevo(raiz, "cartpole", "demostraciones", "Demos", "dashboard",
                         escribir)["id"]


def _argumentos_imitacion(tmp_path: Path, datos: list[str], **cambios) -> dict:
    argumentos = _argumentos(tmp_path, datos=datos, cantidad=None)
    argumentos.update(cambios)
    return argumentos


@pytest.mark.parametrize("metodo", ["gail", "bc"])
def test_argumentos_de_imitacion(tmp_path, metodo):
    argumentos = _argumentos_imitacion(tmp_path, ["a" * 12], cantidad=3)
    assert comprobar_argumentos_entrenamiento(argumentos, metodo)["cantidad"] == 3
    with pytest.raises(ValueError, match="exactamente"):
        comprobar_argumentos_entrenamiento(_argumentos(tmp_path), metodo)  # sin datos
    with pytest.raises(ValueError, match="exactamente"):
        comprobar_argumentos_entrenamiento(argumentos, "experto")  # el experto no los lleva


@pytest.mark.parametrize("cambio, mensaje", [
    ({"datos": []}, "lista no vacía"), ({"datos": "a" * 12}, "lista no vacía"),
    ({"datos": [3]}, "lista no vacía"), ({"datos": [""]}, "lista no vacía"),
    ({"cantidad": 0}, "cantidad"), ({"cantidad": -2}, "cantidad"),
    ({"cantidad": True}, "cantidad"), ({"cantidad": 1.5}, "cantidad"),
])
def test_argumentos_de_imitacion_que_no_valen(tmp_path, cambio, mensaje):
    argumentos = _argumentos_imitacion(tmp_path, ["a" * 12])
    argumentos.update(cambio)
    with pytest.raises(ValueError, match=mensaje):
        comprobar_argumentos_entrenamiento(argumentos, "gail")


def test_bc_no_tiene_vista(tmp_path):
    argumentos = _argumentos_imitacion(tmp_path, ["a" * 12], vista=True)
    assert comprobar_argumentos_entrenamiento(argumentos, "gail")
    with pytest.raises(ValueError, match="BC no tiene vista"):
        comprobar_argumentos_entrenamiento(argumentos, "bc")


def test_progreso_de_bc_uno_por_epoca_con_la_perdida_media():
    canal, run = CanalFalso(), RunFalso()
    progreso = ProgresoEpocas(canal, run)
    for paso, perdida in enumerate([1.0, 2.0, 6.0], 1):
        progreso.log({"bc/train/perdida": perdida}, paso)
    progreso.log({"bc/val/perdida": 0.5, "bc/val/acierto": 0.75}, 3)
    progreso.log({"bc/train/perdida": 4.0}, 4)
    progreso.log({"bc/val/perdida": 0.25, "bc/val/acierto": 0.5}, 4)
    assert canal.progresos == [
        {"paso": 1, "epoca": 1, "bc/train/perdida": 3.0, "bc/val/perdida": 0.5,
         "bc/val/acierto": 0.75},
        {"paso": 2, "epoca": 2, "bc/train/perdida": 4.0, "bc/val/perdida": 0.25,
         "bc/val/acierto": 0.5}]
    # A W&B llega cada batch, como en la E1.
    assert [paso for _, paso in run.registros] == [1, 2, 3, 3, 4, 4]


def test_control_sin_vista_ignora_la_vista_y_atiende_finalizar():
    control = ControlEntrenamiento(CanalFalso([[{"tipo": "vista", "activa": True}],
                                               [{"tipo": "finalizar"}]]))
    assert control.parar() is False
    assert control.parar() is True


def test_datos_con_la_cantidad_pedida(tmp_path):
    conjunto = _conjunto(tmp_path / "biblioteca")
    cfg = resolver_configuracion(CONFIG, "cartpole", SMOKE)
    # Sin cantidad: todas las de train (4 de 6), aunque los splits solo traían [1].
    _, splits, procedencia, cantidad, usado = datos_de_entrenamiento(
        _argumentos_imitacion(tmp_path, [conjunto]), cfg)
    assert usado == conjunto and cantidad == 4
    assert splits.subconjuntos[4] == splits.train and splits.subconjuntos[1] == splits.train[:1]
    assert procedencia["huella_dataset"] == splits.huella_dataset
    _, splits, _, cantidad, _ = datos_de_entrenamiento(
        _argumentos_imitacion(tmp_path, [conjunto], cantidad=3), cfg)
    assert cantidad == 3 and splits.subconjuntos[3] == splits.train[:3]
    with pytest.raises(ValueError, match="Se pidieron 5 partidas de train y los datos tienen 4"):
        datos_de_entrenamiento(_argumentos_imitacion(tmp_path, [conjunto], cantidad=5), cfg)


def test_datos_sin_splits_se_combinan(tmp_path):
    conjunto = _conjunto(tmp_path / "biblioteca", splits=False)
    cfg = resolver_configuracion(CONFIG, "cartpole", SMOKE)
    _, splits, _, cantidad, usado = datos_de_entrenamiento(
        _argumentos_imitacion(tmp_path, [conjunto]), cfg)
    assert usado != conjunto and cantidad == len(splits.train)


def _cfg_imitacion(semilla: int = 123) -> dict:
    cfg = resolver_configuracion(CONFIG, "cartpole", SMOKE)
    cfg["experimento"]["semilla_repeticiones"] = semilla
    return cfg


def _iguales(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_el_trabajador_de_gail_entrena_lo_mismo_que_entrenar_gail(tmp_path):
    raiz = tmp_path / "biblioteca"
    conjunto = _conjunto(raiz)
    canal = CanalFalso()
    resultado = tarea_entrenar_gail(_argumentos_imitacion(tmp_path, [conjunto], cantidad=3),
                                    canal)
    ResultadoEntrenamiento.model_validate(resultado)
    assert resultado["semilla"] == 123 and resultado["modelo"]["incompleto"] is False
    cfg = _cfg_imitacion()
    trayectorias, splits, _ = cargar_datos(cfg, Biblioteca(raiz, {"cartpole": "CartPole-v1"})
                                           .ruta(conjunto))
    splits = dataclasses.replace(splits, subconjuntos={**splits.subconjuntos,
                                                       3: splits.train[:3]})
    agente, discriminador, resumen = entrenar_gail(cfg, trayectorias, splits, 3, 0)
    biblioteca = Biblioteca(raiz, {"cartpole": "CartPole-v1"})
    elemento = biblioteca.obtener(resultado["modelo"]["id"])
    assert elemento["valido"] and elemento["tipo"] == "gail" and elemento["origen"] == "dashboard"
    assert elemento["detalles"]["cantidad"] == 3 and elemento["detalles"]["repeticion"] == 0
    guardado, d_guardado, _ = cargar_gail(biblioteca.ruta(resultado["modelo"]["id"]))
    assert _iguales(guardado.policy.state_dict(), agente.policy.state_dict())
    assert _iguales(d_guardado.state_dict(), discriminador.state_dict())
    assert resultado["resumen"]["validaciones"] == [
        {k: v for k, v in val.items()} for val in resumen["validaciones"]]
    # Un progreso por iteración, en iteraciones, con su validación (el smoke valida en cada una).
    assert [(m["paso"], m["iteracion"]) for m in canal.progresos] == [(1, 1), (2, 2)]
    assert all("validacion/retorno_medio" in m and "trpo/partidas" in m
               for m in canal.progresos)


def test_gail_con_vista_y_finalizar(tmp_path):
    conjunto = _conjunto(tmp_path / "biblioteca")
    ajustes = copy.deepcopy(SMOKE)
    ajustes["gail"]["iteraciones"] = 10
    canal = CanalFalso([[{"tipo": "finalizar"}]])
    resultado = tarea_entrenar_gail(_argumentos_imitacion(tmp_path, [conjunto], vista=True,
                                                          ajustes=ajustes), canal)
    assert resultado["modelo"]["incompleto"] is True
    assert resultado["resumen"]["iteraciones"] == 1 and resultado["resumen"]["incompleto"]
    assert canal.imagenes and canal.imagenes[0][:2] == b"\xff\xd8"


def test_el_trabajador_de_bc_entrena_lo_mismo_que_entrenar_bc_repeticion(tmp_path):
    raiz = tmp_path / "biblioteca"
    conjunto = _conjunto(raiz)
    canal = CanalFalso()
    resultado = tarea_entrenar_bc(_argumentos_imitacion(tmp_path, [conjunto]), canal)
    ResultadoEntrenamiento.model_validate(resultado)
    assert resultado["modelo"]["incompleto"] is False
    cfg = _cfg_imitacion()
    biblioteca = Biblioteca(raiz, {"cartpole": "CartPole-v1"})
    trayectorias, splits, _ = cargar_datos(cfg, biblioteca.ruta(conjunto))
    splits = dataclasses.replace(splits, subconjuntos={**splits.subconjuntos,
                                                       4: splits.train})
    politica, resumen = entrenar_bc_repeticion(cfg, trayectorias, splits, 4, 0)
    elemento = biblioteca.obtener(resultado["modelo"]["id"])
    assert elemento["valido"] and elemento["tipo"] == "bc" and elemento["origen"] == "dashboard"
    assert elemento["detalles"]["subconjunto"] == 4
    guardada, metadatos = cargar_politica_bc(biblioteca.ruta(resultado["modelo"]["id"])
                                             / "politica.pt")
    assert _iguales(guardada.state_dict(), politica.state_dict())
    assert metadatos["semilla"] == resumen["semillas"]["modelo"]
    assert resultado["resumen"]["epocas"] == resumen["epocas"]
    assert [m["epoca"] for m in canal.progresos] == list(range(1, resumen["epocas"] + 1))
    assert all("bc/train/perdida" in m and "bc/val/perdida" in m for m in canal.progresos)


def test_bc_finalizar_guarda_la_mejor_epoca_como_incompleto(tmp_path):
    conjunto = _conjunto(tmp_path / "biblioteca")
    ajustes = copy.deepcopy(SMOKE)
    ajustes["bc"].update(epocas_maximas=10, paciencia=100)
    resultado = tarea_entrenar_bc(_argumentos_imitacion(tmp_path, [conjunto], ajustes=ajustes),
                                  CanalFalso([[{"tipo": "finalizar"}]]))
    assert resultado["modelo"]["incompleto"] is True and resultado["resumen"]["epocas"] == 1


def test_bc_que_finaliza_en_la_ultima_epoca_no_es_incompleto(tmp_path):
    conjunto = _conjunto(tmp_path / "biblioteca")
    ajustes = copy.deepcopy(SMOKE)
    ajustes["bc"].update(epocas_maximas=2, paciencia=100)
    # parar se consulta tras cada época: la orden llega al acabar la segunda, la última.
    resultado = tarea_entrenar_bc(_argumentos_imitacion(tmp_path, [conjunto], ajustes=ajustes),
                                  CanalFalso([[], [{"tipo": "finalizar"}]]))
    assert resultado["resumen"]["epocas"] == 2
    assert resultado["modelo"]["incompleto"] is False


def test_bc_que_agota_la_paciencia_no_es_incompleto(tmp_path, monkeypatch):
    """La validación nunca mejora: para por la paciencia sin consultar la orden."""
    import src.training.bc as bc
    monkeypatch.setattr(bc, "evaluar_bc", lambda politica, loader: {"perdida": 1.0,
                                                                        "acierto": 0.5})
    conjunto = _conjunto(tmp_path / "biblioteca")
    ajustes = copy.deepcopy(SMOKE)
    ajustes["bc"].update(epocas_maximas=10, paciencia=1)
    resultado = tarea_entrenar_bc(_argumentos_imitacion(tmp_path, [conjunto], ajustes=ajustes),
                                  CanalFalso([[], [{"tipo": "finalizar"}]]))
    assert resultado["resumen"]["epocas"] == 2
    assert resultado["modelo"]["incompleto"] is False


@pytest.mark.parametrize("metodo, tarea, archivos", [
    ("gail", tarea_entrenar_gail, ["modelo.zip", "discriminador.pt", "gail.json"]),
    ("bc", tarea_entrenar_bc, ["politica.pt"]),
])
def test_con_wandb_bc_y_gail_suben_su_modelo(tmp_path, monkeypatch, metodo, tarea, archivos):
    run = RunFalso()
    run.summary = {}
    subidos = []
    monkeypatch.setattr(trabajadores, "abrir_run", lambda cfg, **o: run)
    monkeypatch.setattr(trabajadores, "subir_artefacto",
                        lambda run, nombre, tipo, rutas, metadatos: subidos.append(
                            (nombre, tipo, [Path(r).name for r in rutas], metadatos)))
    monkeypatch.setattr(trabajadores, "terminar_run", lambda run, error=None: None)
    conjunto = _conjunto(tmp_path / "biblioteca")
    tarea(_argumentos_imitacion(tmp_path, [conjunto],
                                wandb={"activo": True, "grupo": "dashboard-cartpole"}),
          CanalFalso())
    (nombre, tipo, rutas, metadatos), = subidos
    assert nombre.startswith(f"{metodo}-CartPole-v1-") and tipo == metodo and rutas == archivos
    assert metadatos["origen"] == "dashboard" and metadatos["cantidad"] == 4
    assert run.summary["incompleto"] is False and "huella_dataset" in run.summary


@pytest.mark.parametrize("tarea", [TAREA_GAIL, TAREA_BC])
def test_bc_y_gail_con_el_gestor(gestor, tmp_path, tarea):
    conjunto = _conjunto(tmp_path / "biblioteca")
    trabajo = gestor.crear("entrenamiento", tarea,
                           _argumentos_imitacion(tmp_path, [conjunto], cantidad=2))
    final = gestor.esperar(trabajo["id"], LIMITE)
    assert final["estado"] == "terminado", final["error"]
    resultado = ResultadoEntrenamiento.model_validate(final["resultado"])
    assert not resultado.modelo.incompleto and gestor.historial(trabajo["id"])


def test_unos_datos_que_no_existen_dejan_el_trabajo_fallido(gestor, tmp_path):
    trabajo = gestor.crear("entrenamiento", TAREA_BC,
                           _argumentos_imitacion(tmp_path, ["b" * 12]))
    final = gestor.esperar(trabajo["id"], LIMITE)
    assert final["estado"] == "fallido" and "No existen" in final["error"]


def test_gail_que_finaliza_en_la_ultima_iteracion_no_es_incompleto(tmp_path):
    conjunto = _conjunto(tmp_path / "biblioteca")
    # Cuántas veces lee las órdenes un GAIL del smoke (2 iteraciones), para que la orden
    # llegue en la última lectura, la de parar al acabar la última iteración.
    canal = CanalFalso()
    lecturas = []
    original = canal.ordenes
    canal.ordenes = lambda: lecturas.append(1) or original()
    tarea_entrenar_gail(_argumentos_imitacion(tmp_path, [conjunto]), canal)
    canal = CanalFalso([[]] * (len(lecturas) - 1) + [[{"tipo": "finalizar"}]])
    resultado = tarea_entrenar_gail(_argumentos_imitacion(tmp_path, [conjunto]), canal)
    assert resultado["resumen"]["iteraciones"] == 2
    assert resultado["modelo"]["incompleto"] is False


# --- comparación ---------------------------------------------------------------------------

TAREA_COMPARAR = "src.training.trabajadores:tarea_comparar"
ENTORNOS = {"cartpole": "CartPole-v1", "mountaincar": "MountainCar-v0"}


@pytest.fixture(scope="module")
def modelos(tmp_path_factory) -> tuple[Path, dict[str, str]]:
    """Biblioteca de CartPole con un experto, otro que normaliza, un BC y un GAIL."""
    base = tmp_path_factory.mktemp("comparar")
    raiz = base / "biblioteca"
    ids = {"experto": tarea_entrenar_experto(_argumentos(base), CanalFalso())["modelo"]["id"]}
    ajustes = copy.deepcopy(SMOKE)
    ajustes.setdefault("trpo", {})["normalizar_observaciones"] = True
    ids["normaliza"] = tarea_entrenar_experto(_argumentos(base, ajustes=ajustes),
                                              CanalFalso())["modelo"]["id"]
    conjunto = _conjunto(raiz)
    for metodo, tarea in (("bc", tarea_entrenar_bc), ("gail", tarea_entrenar_gail)):
        ids[metodo] = tarea(_argumentos_imitacion(base, [conjunto]), CanalFalso())["modelo"]["id"]
    ids["demostraciones"] = conjunto
    return raiz, ids


def _argumentos_comparar(raiz: Path, modelos: list[str], **cambios) -> dict:
    argumentos = {"ruta_config": str(CONFIG), "biblioteca": str(raiz), "calidad_jpeg": 80,
                  "entorno": "cartpole", "modelos": modelos, "aleatorio": True,
                  "partidas": 3, "vista": False}
    argumentos.update(cambios)
    return argumentos


class CanalFinalizar(CanalFalso):
    """Pide finalizar cuando ya se han mandado ``tras`` progresos."""

    def __init__(self, tras: int) -> None:
        super().__init__()
        self.tras = tras

    def ordenes(self) -> list[dict]:
        return [{"tipo": "finalizar"}] if len(self.progresos) == self.tras else []


def test_argumentos_de_comparacion_validos(tmp_path):
    argumentos = _argumentos_comparar(tmp_path, ["a" * 12], partidas=None, aleatorio=False)
    assert comprobar_argumentos_comparacion(argumentos) == argumentos
    assert comprobar_argumentos_comparacion(argumentos) is not argumentos
    solo_aleatorio = _argumentos_comparar(tmp_path, [], partidas=np.int64(4))
    assert comprobar_argumentos_comparacion(solo_aleatorio)["partidas"] == 4


@pytest.mark.parametrize("cambio, mensaje", [
    ({"modelos": "a" * 12}, "lista de ids"), ({"modelos": [3]}, "lista de ids"),
    ({"modelos": [""]}, "lista de ids"), ({"modelos": ["a" * 12, "a" * 12]}, "repetidos"),
    ({"modelos": [], "aleatorio": False}, "nada que comparar"),
    ({"aleatorio": 1}, "aleatorio"), ({"vista": None}, "vista"),
    ({"partidas": 0}, "partidas"), ({"partidas": -1}, "partidas"),
    ({"partidas": True}, "partidas"), ({"partidas": 2.0}, "partidas"),
    ({"entorno": " "}, "entorno"), ({"ruta_config": None}, "ruta_config"),
    ({"biblioteca": ""}, "biblioteca"),
])
def test_argumentos_de_comparacion_que_no_valen(tmp_path, cambio, mensaje):
    with pytest.raises(ValueError, match=mensaje):
        comprobar_argumentos_comparacion({**_argumentos_comparar(tmp_path, ["a" * 12]),
                                          **cambio})


@pytest.mark.parametrize("argumentos", [
    None, [], {"modelos": []},
    {**_argumentos_comparar(Path("x"), []), "semilla": 3},
])
def test_argumentos_de_comparacion_con_claves_de_mas_o_de_menos(argumentos):
    with pytest.raises(ValueError, match="exactamente"):
        comprobar_argumentos_comparacion(argumentos)


def test_compara_como_la_evaluacion_de_la_e1(modelos):
    """Cada modelo da los retornos de evaluar_politica con las mismas partidas."""
    raiz, ids = modelos
    orden = [ids["gail"], ids["experto"], ids["bc"], ids["normaliza"]]
    canal = CanalFalso()
    resultado = tarea_comparar(_argumentos_comparar(raiz, orden), canal)
    ResultadoComparacion.model_validate(resultado)
    cfg = resolver_configuracion(CONFIG, "cartpole")
    semilla = cfg["evaluacion"]["semilla"]
    assert resultado["partidas"] == 3 and resultado["semilla"] == semilla
    assert resultado["incompleta"] is False
    assert [m["id"] for m in resultado["modelos"]] == [*orden, "aleatorio"]
    biblioteca = Biblioteca(raiz, ENTORNOS)
    esperados = {}
    for id_modelo in [*orden, "aleatorio"]:
        agente, estadisticas = (("aleatorio", None) if id_modelo == "aleatorio"
                                else cargar_agente(biblioteca, id_modelo, "cartpole"))
        esperados[id_modelo] = evaluar_politica(agente, cfg, 3, semilla,
                                                cfg["evaluacion"]["determinista"], estadisticas)
    assert {m["id"]: m["retornos"] for m in resultado["modelos"]} == {
        k: v["retornos"] for k, v in esperados.items()}
    # Las estadísticas del que normaliza se usan (sin ellas jugaría otra cosa).
    assert cargar_agente(biblioteca, ids["normaliza"], "cartpole")[1] is not None
    # Un progreso por partida de cada modelo: todos juegan la i antes de la i + 1.
    assert [(m["partida"], m["modelo"]) for m in canal.progresos] == [
        (i, id_modelo) for i in range(3) for id_modelo in [*orden, "aleatorio"]]
    for mensaje in canal.progresos:
        i, evaluado = mensaje["partida"], esperados[mensaje["modelo"]]
        assert mensaje["retorno"] == evaluado["retornos"][i]
        assert mensaje["pasos"] == evaluado["longitudes"][i]
    assert canal.imagenes == []


def test_sin_aleatorio_solo_juegan_los_modelos(modelos):
    raiz, ids = modelos
    canal = CanalFalso()
    resultado = tarea_comparar(_argumentos_comparar(raiz, [ids["bc"]], aleatorio=False,
                                                    partidas=2), canal)
    assert [m["id"] for m in resultado["modelos"]] == [ids["bc"]]
    assert {m["modelo"] for m in canal.progresos} == {ids["bc"]}


def test_sin_partidas_usa_las_de_la_evaluacion(tmp_path):
    canal = CanalFalso()
    resultado = tarea_comparar(_argumentos_comparar(tmp_path / "vacia", [], partidas=None),
                               canal)
    cfg = resolver_configuracion(CONFIG, "cartpole")
    assert resultado["partidas"] == cfg["evaluacion"]["episodios"] == len(canal.progresos)
    assert resultado["modelos"] == [{"id": "aleatorio",
                                     "retornos": evaluar_final("aleatorio", cfg)["retornos"]}]


def test_las_primeras_partidas_no_dependen_de_cuantas_se_pidan(modelos):
    raiz, ids = modelos
    cortas = tarea_comparar(_argumentos_comparar(raiz, [ids["bc"]], partidas=2), CanalFalso())
    largas = tarea_comparar(_argumentos_comparar(raiz, [ids["bc"]], partidas=4), CanalFalso())
    for corta, larga in zip(cortas["modelos"], largas["modelos"], strict=True):
        assert corta["retornos"] == larga["retornos"][:2]


@pytest.mark.parametrize("tras, rondas", [(0, 0), (1, 0), (2, 1), (3, 1), (4, 2), (5, 2)])
def test_finalizar_se_queda_con_las_rondas_completas(modelos, tras, rondas):
    raiz, ids = modelos
    canal = CanalFinalizar(tras)
    resultado = tarea_comparar(_argumentos_comparar(raiz, [ids["experto"]]), canal)
    completa = tarea_comparar(_argumentos_comparar(raiz, [ids["experto"]]), CanalFalso())
    assert resultado["incompleta"] is True and resultado["partidas"] == rondas
    ResultadoComparacion.model_validate(resultado)
    for parcial, entera in zip(resultado["modelos"], completa["modelos"], strict=True):
        assert parcial["retornos"] == entera["retornos"][:rondas]
    # Lo jugado sí salió como progreso; la partida cortada, no.
    assert len(canal.progresos) == tras


def test_finalizar_tras_la_ultima_partida_no_la_deja_incompleta(modelos):
    raiz, ids = modelos
    resultado = tarea_comparar(_argumentos_comparar(raiz, [ids["experto"]]), CanalFinalizar(6))
    assert resultado["incompleta"] is False and resultado["partidas"] == 3


def test_la_vista_dibuja_sin_cambiar_los_retornos(tmp_path):
    sin = tarea_comparar(_argumentos_comparar(tmp_path, [], partidas=2), CanalFalso())
    canal = CanalFalso()
    con = tarea_comparar(_argumentos_comparar(tmp_path, [], partidas=2, vista=True), canal)
    assert con == sin
    pasos = sum(m["pasos"] for m in canal.progresos)
    assert len(canal.imagenes) == pasos + 2  # el estado inicial de cada partida y cada paso
    assert canal.imagenes[0][:2] == b"\xff\xd8"


def test_la_vista_va_a_velocidad_real(tmp_path):
    canal = CanalFalso()
    inicio = time.monotonic()
    tarea_comparar(_argumentos_comparar(tmp_path, [], partidas=1, vista=True), canal)
    pasos = canal.progresos[0]["pasos"]
    # CartPole dibuja a 50 fps: la partida dura al menos pasos / 50 s.
    assert time.monotonic() - inicio >= pasos / 50


def test_la_vista_se_enciende_y_apaga_con_ordenes(tmp_path):
    canal = CanalFalso([[], [], [{"tipo": "vista", "activa": True}], [],
                        [{"tipo": "vista", "activa": False}]])
    tarea_comparar(_argumentos_comparar(tmp_path, [], partidas=1), canal)
    # Se enciende al leer la tercera orden (antes del tercer paso) y se apaga en la
    # quinta: se dibujan el tercer y el cuarto paso.
    assert len(canal.imagenes) == 2


def _escribir(ruta: Path) -> None:
    ruta.write_bytes(b"x")


@pytest.mark.parametrize("caso, mensaje", [
    ("no_existe", "no está en la biblioteca"), ("demostraciones", "no un modelo"),
    ("otro_entorno", "es de mountaincar"), ("roto", "no es válido"),
])
def test_modelos_que_no_se_pueden_comparar(modelos, tmp_path, caso, mensaje):
    raiz, ids = modelos
    if caso == "no_existe":
        id_modelo = "0" * 12
    elif caso == "demostraciones":
        id_modelo = ids["demostraciones"]
    else:
        raiz = tmp_path / "biblioteca"
        entorno = "mountaincar" if caso == "otro_entorno" else "cartpole"
        id_modelo = agregar_nuevo(raiz, entorno, "bc", "Roto", "dashboard",
                                  lambda carpeta: _escribir(carpeta / "politica.pt"))["id"]
    canal = CanalFalso()
    with pytest.raises(ValueError, match=mensaje):
        tarea_comparar(_argumentos_comparar(raiz, [id_modelo]), canal)
    assert canal.progresos == []


def test_con_el_gestor_compara_y_devuelve_el_resultado(gestor, modelos):
    raiz, ids = modelos
    argumentos = _argumentos_comparar(raiz, [ids["experto"], ids["gail"]], partidas=2)
    trabajo = gestor.crear("comparacion", TAREA_COMPARAR, argumentos)
    final = gestor.esperar(trabajo["id"], LIMITE)
    assert final["estado"] == "terminado", final["error"]
    assert final["resultado"] == tarea_comparar(argumentos, CanalFalso())
    assert len(gestor.historial(trabajo["id"])) == 6


def test_con_el_gestor_la_vista_y_finalizar_llegan_a_la_comparacion(gestor, tmp_path):
    # Con la vista encendida va a velocidad real: 1000 partidas no acaban antes de finalizar.
    argumentos = _argumentos_comparar(tmp_path, [], partidas=1000, vista=True)
    trabajo = gestor.crear("comparacion", TAREA_COMPARAR, argumentos)
    _esperar_hasta(lambda: gestor.ultima_imagen(trabajo["id"]) is not None)
    _esperar_hasta(lambda: len(gestor.historial(trabajo["id"])) >= 1)
    gestor.enviar_orden(trabajo["id"], {"tipo": "finalizar"})
    final = gestor.esperar(trabajo["id"], LIMITE)
    assert final["estado"] == "terminado", final["error"]
    resultado = ResultadoComparacion.model_validate(final["resultado"])
    assert resultado.incompleta and 1 <= resultado.partidas < 1000
    assert len(resultado.modelos[0].retornos) == resultado.partidas


# --- checkpoints de BC ----------------------------------------------------------------------

class CanalQueSeCae(CanalFalso):
    """Imita un corte: falla al mandar el progreso número ``en`` (desde 1)."""

    def __init__(self, en: int) -> None:
        super().__init__()
        self.en = en

    def progreso(self, medidas: dict) -> None:
        if len(self.progresos) + 1 == self.en:
            raise KeyboardInterrupt("corte")
        super().progreso(medidas)


def _argumentos_bc_reanudable(tmp_path: Path, conjunto: str, **cambios) -> dict:
    ajustes = copy.deepcopy(SMOKE)
    ajustes["bc"].update(epocas_maximas=5, paciencia=100)
    return _argumentos_imitacion(tmp_path, [conjunto], ajustes=ajustes,
                                 checkpoint=str(tmp_path / "biblioteca" / ".en_curso" / "t1"),
                                 **cambios)


def _politica_de(raiz: Path, resultado: dict):
    biblioteca = Biblioteca(raiz, {"cartpole": "CartPole-v1"})
    return cargar_politica_bc(biblioteca.ruta(resultado["modelo"]["id"]) / "politica.pt")[0]


def test_bc_con_checkpoint_sin_cortes_entrena_lo_mismo_y_borra_la_carpeta(tmp_path):
    raiz = tmp_path / "biblioteca"
    conjunto = _conjunto(raiz)
    argumentos = _argumentos_bc_reanudable(tmp_path, conjunto)
    resultado = tarea_entrenar_bc(argumentos, CanalFalso())
    sin = tarea_entrenar_bc({**argumentos, "checkpoint": None, "nombre": "Otro"}, CanalFalso())
    assert _iguales(_politica_de(raiz, resultado).state_dict(),
                    _politica_de(raiz, sin).state_dict())
    assert resultado["resumen"]["reanudado_desde"] == []
    assert not Path(argumentos["checkpoint"]).exists()


@pytest.mark.parametrize("corte", [1, 2, 3, 5])
def test_bc_cortado_en_cualquier_epoca_se_reanuda_y_acaba_igual(tmp_path, corte):
    raiz = tmp_path / "biblioteca"
    conjunto = _conjunto(raiz)
    argumentos = _argumentos_bc_reanudable(tmp_path, conjunto)
    completo = CanalFalso()
    referencia = tarea_entrenar_bc({**argumentos, "checkpoint": None}, completo)

    with pytest.raises(KeyboardInterrupt):
        tarea_entrenar_bc(argumentos, CanalQueSeCae(corte))
    carpeta = Path(argumentos["checkpoint"])
    assert (carpeta / "argumentos.json").is_file()
    # El corte llega al mandar la época «corte»: el último checkpoint es el de la anterior.
    assert (carpeta / "checkpoint.pt").is_file() == (corte > 1)

    canal = CanalFalso()
    resultado = tarea_entrenar_bc({**argumentos, "vista": False}, canal)
    assert _iguales(_politica_de(raiz, resultado).state_dict(),
                    _politica_de(raiz, referencia).state_dict())
    # Primero el progreso guardado y después el resto: el mismo que sin cortes.
    assert canal.progresos == completo.progresos
    assert resultado["resumen"]["reanudado_desde"] == ([corte - 1] if corte > 1 else [])
    sin_tiempos = {"segundos", "reanudado_desde"}
    assert ({k: v for k, v in resultado["resumen"].items() if k not in sin_tiempos}
            == {k: v for k, v in referencia["resumen"].items() if k not in sin_tiempos})
    assert not carpeta.exists()
    # El intento cortado no dejó nada en la biblioteca: solo los dos terminados.
    biblioteca = Biblioteca(raiz, {"cartpole": "CartPole-v1"})
    assert len(biblioteca.listar("cartpole", ("bc",))) == 2


def test_bc_reanudado_suma_los_segundos_de_antes_del_corte(tmp_path):
    from src.training.checkpoints import escribir_checkpoint, leer_checkpoint
    argumentos = _argumentos_bc_reanudable(tmp_path, _conjunto(tmp_path / "biblioteca"))
    with pytest.raises(KeyboardInterrupt):
        tarea_entrenar_bc(argumentos, CanalQueSeCae(3))
    carpeta = Path(argumentos["checkpoint"])
    contenido = leer_checkpoint(carpeta, "bc")
    escribir_checkpoint(carpeta, "bc", {**{k: v for k, v in contenido.items()
                                           if k not in ("formato", "version", "metodo")},
                                        "segundos": 1000.0})
    resultado = tarea_entrenar_bc(argumentos, CanalFalso())
    assert 1000 < resultado["resumen"]["segundos"] < 1100


def test_bc_cortado_dos_veces(tmp_path):
    raiz = tmp_path / "biblioteca"
    conjunto = _conjunto(raiz)
    argumentos = _argumentos_bc_reanudable(tmp_path, conjunto)
    with pytest.raises(KeyboardInterrupt):
        tarea_entrenar_bc(argumentos, CanalQueSeCae(3))
    # La segunda vez se vuelven a mandar 2 épocas: el corte en el 4.º mensaje es en la 4.ª.
    with pytest.raises(KeyboardInterrupt):
        tarea_entrenar_bc(argumentos, CanalQueSeCae(5))
    resultado = tarea_entrenar_bc(argumentos, CanalFalso())
    assert resultado["resumen"]["reanudado_desde"] == [2, 4]


def test_bc_finalizado_tras_reanudar_borra_la_carpeta_y_es_incompleto(tmp_path):
    raiz = tmp_path / "biblioteca"
    argumentos = _argumentos_bc_reanudable(tmp_path, _conjunto(raiz))
    with pytest.raises(KeyboardInterrupt):
        tarea_entrenar_bc(argumentos, CanalQueSeCae(3))
    resultado = tarea_entrenar_bc(argumentos, CanalFalso([[{"tipo": "finalizar"}]]))
    assert resultado["modelo"]["incompleto"] is True and resultado["resumen"]["epocas"] == 3
    assert not Path(argumentos["checkpoint"]).exists()


@pytest.mark.parametrize("cambio", [{"cantidad": 2}, {"semilla": 5}, {"nombre": "Otro"}])
def test_bc_no_se_reanuda_con_otros_argumentos(tmp_path, cambio):
    raiz = tmp_path / "biblioteca"
    argumentos = _argumentos_bc_reanudable(tmp_path, _conjunto(raiz))
    with pytest.raises(KeyboardInterrupt):
        tarea_entrenar_bc(argumentos, CanalQueSeCae(3))
    canal = CanalFalso()
    with pytest.raises(ValueError, match="No se puede reanudar"):
        tarea_entrenar_bc({**argumentos, **cambio}, canal)
    assert canal.progresos == [] and Path(argumentos["checkpoint"], "checkpoint.pt").is_file()


def test_bc_no_se_reanuda_si_cambio_la_configuracion(tmp_path, monkeypatch):
    raiz = tmp_path / "biblioteca"
    argumentos = _argumentos_bc_reanudable(tmp_path, _conjunto(raiz))
    with pytest.raises(KeyboardInterrupt):
        tarea_entrenar_bc(argumentos, CanalQueSeCae(3))
    original = trabajadores._config

    def otra(argumentos):
        cfg = original(argumentos)
        cfg["bc"]["learning_rate"] = 0.5  # como si se hubiera editado configs/
        return cfg

    monkeypatch.setattr(trabajadores, "_config", otra)
    with pytest.raises(ValueError, match="configuración cambió"):
        tarea_entrenar_bc(argumentos, CanalFalso())


def test_bc_reanudado_con_wandb_abre_un_run_nuevo_marcado(tmp_path, monkeypatch):
    abiertos = []

    def abrir(cfg, **opciones):
        run = RunFalso()
        run.id, run.summary = f"run{len(abiertos)}", {}
        abiertos.append((cfg, opciones))
        return run

    monkeypatch.setattr(trabajadores, "abrir_run", abrir)
    monkeypatch.setattr(trabajadores, "subir_artefacto", lambda *a, **k: None)
    monkeypatch.setattr(trabajadores, "terminar_run", lambda run, error=None: None)
    raiz = tmp_path / "biblioteca"
    argumentos = _argumentos_bc_reanudable(
        tmp_path, _conjunto(raiz), wandb={"activo": True, "grupo": "dashboard-cartpole"})
    with pytest.raises(KeyboardInterrupt):
        tarea_entrenar_bc(argumentos, CanalQueSeCae(3))
    tarea_entrenar_bc(argumentos, CanalFalso())
    (cfg0, op0), (cfg1, op1) = abiertos
    assert "reanudado" not in cfg0 and "reanudado" not in op0["etiquetas"]
    assert cfg1["reanudado"] == {"run_anterior": "run0", "desde_epoca": 2}
    assert "reanudado" in op1["etiquetas"] and op1["grupo"] == "dashboard-cartpole"


@pytest.mark.parametrize("checkpoint", ["", " ", 3])
def test_checkpoint_no_valido(tmp_path, checkpoint):
    with pytest.raises(ValueError, match="checkpoint"):
        comprobar_argumentos_entrenamiento(
            _argumentos_imitacion(tmp_path, ["a" * 12], checkpoint=checkpoint), "bc")


# --- checkpoints del experto ------------------------------------------------------------------

def _argumentos_experto_reanudable(tmp_path: Path, **cambios) -> dict:
    """4 iteraciones de 500 pasos con una evaluación en cada una."""
    ajustes = copy.deepcopy(SMOKE)
    ajustes["experto"].update(pasos_totales=2000, evaluar_cada=1, retorno_objetivo=10**6)
    return _argumentos(tmp_path, ajustes=ajustes,
                       checkpoint=str(tmp_path / "biblioteca" / ".en_curso" / "t1"), **cambios)


def _experto_de(raiz: Path, resultado: dict):
    biblioteca = Biblioteca(raiz, {"cartpole": "CartPole-v1"})
    return cargar_experto(biblioteca.ruta(resultado["modelo"]["id"]))[0]


def test_experto_con_checkpoint_sin_cortes_entrena_lo_mismo_y_borra_la_carpeta(tmp_path):
    raiz = tmp_path / "biblioteca"
    argumentos = _argumentos_experto_reanudable(tmp_path)
    resultado = tarea_entrenar_experto(argumentos, CanalFalso())
    sin = tarea_entrenar_experto({**argumentos, "checkpoint": None}, CanalFalso())
    assert _iguales(_experto_de(raiz, resultado).policy.state_dict(),
                    _experto_de(raiz, sin).policy.state_dict())
    assert resultado["resumen"]["reanudado_desde"] == []
    assert not Path(argumentos["checkpoint"]).exists()


@pytest.mark.parametrize("corte", [1, 2, 4])
def test_experto_cortado_se_reanuda_desde_la_ultima_evaluacion(tmp_path, corte):
    raiz = tmp_path / "biblioteca"
    argumentos = _argumentos_experto_reanudable(tmp_path)
    completo = CanalFalso()
    referencia = tarea_entrenar_experto({**argumentos, "checkpoint": None}, completo)
    with pytest.raises(KeyboardInterrupt):
        tarea_entrenar_experto(argumentos, CanalQueSeCae(corte))
    carpeta = Path(argumentos["checkpoint"])
    assert (carpeta / "checkpoint.pt").is_file() == (corte > 1)

    canal = CanalFalso()
    resultado = tarea_entrenar_experto({**argumentos, "vista": True}, canal)
    resumen = resultado["resumen"]
    # El progreso de antes del corte se vuelve a mandar tal cual y después sigue.
    assert canal.progresos[:corte - 1] == completo.progresos[:corte - 1]
    assert [m["iteracion"] for m in canal.progresos] == [1, 2, 3, 4]
    assert resumen["iteraciones"] == 4 and resumen["pasos"] == 2000
    assert (resumen["evaluaciones"][:corte - 1]
            == referencia["resumen"]["evaluaciones"][:corte - 1])
    assert resumen["reanudado_desde"] == ([corte - 1] if corte > 1 else [])
    assert ("entornos_reanudacion" in resumen["semillas"]) == (corte > 1)
    assert canal.imagenes and not carpeta.exists()
    assert _experto_de(raiz, resultado) is not None


def test_experto_reanudado_dos_veces_desde_el_mismo_checkpoint_da_lo_mismo(tmp_path):
    import shutil
    argumentos = _argumentos_experto_reanudable(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        tarea_entrenar_experto(argumentos, CanalQueSeCae(3))
    copia = tmp_path / "copia"
    shutil.copytree(argumentos["checkpoint"], copia)
    a = tarea_entrenar_experto(argumentos, CanalFalso())
    otra_biblioteca = tmp_path / "otra"
    b = tarea_entrenar_experto({**argumentos, "checkpoint": str(copia),
                                "biblioteca": str(otra_biblioteca)}, CanalFalso())
    assert _iguales(_experto_de(tmp_path / "biblioteca", a).policy.state_dict(),
                    _experto_de(otra_biblioteca, b).policy.state_dict())
    assert a["resumen"]["evaluaciones"] == b["resumen"]["evaluaciones"]


def test_experto_reanudado_con_wandb_abre_un_run_nuevo_marcado(tmp_path, monkeypatch):
    abiertos = []

    def abrir(cfg, **opciones):
        run = RunFalso()
        run.id, run.summary = f"run{len(abiertos)}", {}
        abiertos.append((cfg, opciones))
        return run

    monkeypatch.setattr(trabajadores, "abrir_run", abrir)
    monkeypatch.setattr(trabajadores, "subir_artefacto", lambda *a, **k: None)
    monkeypatch.setattr(trabajadores, "terminar_run", lambda run, error=None: None)
    argumentos = _argumentos_experto_reanudable(
        tmp_path, wandb={"activo": True, "grupo": "dashboard-cartpole"})
    with pytest.raises(KeyboardInterrupt):
        tarea_entrenar_experto(argumentos, CanalQueSeCae(3))
    tarea_entrenar_experto(argumentos, CanalFalso())
    (_, op0), (cfg1, op1) = abiertos
    assert "reanudado" not in op0["etiquetas"] and "reanudado" in op1["etiquetas"]
    assert cfg1["reanudado"] == {"run_anterior": "run0", "desde_iteracion": 2}


@pytest.mark.parametrize("metodo", ["experto", "gail", "bc"])
def test_checkpoint_con_el_gestor_se_reanuda_tras_matar_el_proceso(gestor, tmp_path, metodo):
    """Un corte de verdad: el proceso muere (cancelar) y otro trabajo sigue desde ahí."""
    if metodo == "experto":
        argumentos = _argumentos_experto_reanudable(tmp_path)
        argumentos["ajustes"]["experto"]["pasos_totales"] = 10**7
        tarea, unidad = TAREA_EXPERTO, "iteracion"
    elif metodo == "gail":
        argumentos = _argumentos_gail_reanudable(tmp_path, _conjunto(tmp_path / "biblioteca"))
        argumentos["ajustes"]["gail"]["iteraciones"] = 10**5
        tarea, unidad = TAREA_GAIL, "iteracion"
    else:
        argumentos = _argumentos_bc_reanudable(tmp_path, _conjunto(tmp_path / "biblioteca"))
        argumentos["ajustes"]["bc"].update(epocas_maximas=100_000)
        tarea, unidad = TAREA_BC, "epoca"
    trabajo = gestor.crear("entrenamiento", tarea, argumentos)
    _esperar_hasta(lambda: len(gestor.historial(trabajo["id"])) >= 3)
    gestor.cancelar(trabajo["id"])
    gestor.esperar(trabajo["id"], LIMITE)
    assert Path(argumentos["checkpoint"], "checkpoint.pt").is_file()
    guardadas = len(gestor.historial(trabajo["id"]))
    otro = gestor.crear("entrenamiento", tarea, argumentos)
    _esperar_hasta(lambda: len(gestor.historial(otro["id"])) >= guardadas + 1)
    gestor.enviar_orden(otro["id"], {"tipo": "finalizar"})
    final = gestor.esperar(otro["id"], LIMITE)
    assert final["estado"] == "terminado", final["error"]
    desde, = final["resultado"]["resumen"]["reanudado_desde"]
    assert desde >= 2
    # El historial del nuevo trabajo empieza por el del anterior.
    historial = gestor.historial(otro["id"])
    assert historial[:desde] == gestor.historial(trabajo["id"])[:desde]
    assert [m[unidad] for m in historial] == list(range(1, len(historial) + 1))
    assert not Path(argumentos["checkpoint"]).exists()


# --- checkpoints de GAIL ----------------------------------------------------------------------

def _argumentos_gail_reanudable(tmp_path: Path, conjunto: str, **cambios) -> dict:
    """4 iteraciones con una validación en cada una."""
    ajustes = copy.deepcopy(SMOKE)
    ajustes["gail"].update(iteraciones=4, evaluar_cada=1)
    return _argumentos_imitacion(tmp_path, [conjunto], ajustes=ajustes, cantidad=3,
                                 checkpoint=str(tmp_path / "biblioteca" / ".en_curso" / "t1"),
                                 **cambios)


def _gail_de(raiz: Path, resultado: dict):
    biblioteca = Biblioteca(raiz, {"cartpole": "CartPole-v1"})
    return cargar_gail(biblioteca.ruta(resultado["modelo"]["id"]))


def test_gail_con_checkpoint_sin_cortes_entrena_lo_mismo_y_borra_la_carpeta(tmp_path):
    raiz = tmp_path / "biblioteca"
    argumentos = _argumentos_gail_reanudable(tmp_path, _conjunto(raiz))
    resultado = tarea_entrenar_gail(argumentos, CanalFalso())
    sin = tarea_entrenar_gail({**argumentos, "checkpoint": None}, CanalFalso())
    (agente, discriminador, _), (agente_sin, discriminador_sin, _) = (
        _gail_de(raiz, resultado), _gail_de(raiz, sin))
    assert _iguales(agente.policy.state_dict(), agente_sin.policy.state_dict())
    assert _iguales(discriminador.state_dict(), discriminador_sin.state_dict())
    assert resultado["resumen"]["reanudado_desde"] == []
    assert not Path(argumentos["checkpoint"]).exists()


@pytest.mark.parametrize("corte", [1, 2, 4])
def test_gail_cortado_se_reanuda_desde_la_ultima_validacion(tmp_path, corte):
    raiz = tmp_path / "biblioteca"
    argumentos = _argumentos_gail_reanudable(tmp_path, _conjunto(raiz))
    completo = CanalFalso()
    referencia = tarea_entrenar_gail({**argumentos, "checkpoint": None}, completo)
    with pytest.raises(KeyboardInterrupt):
        tarea_entrenar_gail(argumentos, CanalQueSeCae(corte))
    carpeta = Path(argumentos["checkpoint"])
    assert (carpeta / "checkpoint.pt").is_file() == (corte > 1)

    canal = CanalFalso()
    resultado = tarea_entrenar_gail({**argumentos, "vista": True}, canal)
    resumen = resultado["resumen"]
    assert canal.progresos[:corte - 1] == completo.progresos[:corte - 1]
    assert [m["iteracion"] for m in canal.progresos] == [1, 2, 3, 4]
    assert [m["paso"] for m in canal.progresos] == [1, 2, 3, 4]
    assert resumen["iteraciones"] == 4
    assert (resumen["validaciones"][:corte - 1]
            == referencia["resumen"]["validaciones"][:corte - 1])
    assert resumen["reanudado_desde"] == ([corte - 1] if corte > 1 else [])
    assert ("entornos_reanudacion" in resumen["semillas"]) == (corte > 1)
    assert canal.imagenes and not carpeta.exists()
    assert len(Biblioteca(raiz, {"cartpole": "CartPole-v1"}).listar("cartpole", ("gail",))) == 2


def test_gail_reanudado_dos_veces_desde_el_mismo_checkpoint_da_lo_mismo(tmp_path):
    import shutil
    raiz = tmp_path / "biblioteca"
    argumentos = _argumentos_gail_reanudable(tmp_path, _conjunto(raiz))
    with pytest.raises(KeyboardInterrupt):
        tarea_entrenar_gail(argumentos, CanalQueSeCae(3))
    copia = tmp_path / "copia"
    shutil.copytree(argumentos["checkpoint"], copia)
    a = tarea_entrenar_gail(argumentos, CanalFalso())
    b = tarea_entrenar_gail({**argumentos, "checkpoint": str(copia)}, CanalFalso())
    (agente_a, d_a, _), (agente_b, d_b, _) = _gail_de(raiz, a), _gail_de(raiz, b)
    assert _iguales(agente_a.policy.state_dict(), agente_b.policy.state_dict())
    assert _iguales(d_a.state_dict(), d_b.state_dict())
    assert a["resumen"]["validaciones"] == b["resumen"]["validaciones"]
    assert a["resumen"]["reanudado_desde"] == b["resumen"]["reanudado_desde"] == [2]


def test_gail_reanudado_con_wandb_abre_un_run_nuevo_marcado(tmp_path, monkeypatch):
    abiertos = []

    def abrir(cfg, **opciones):
        run = RunFalso()
        run.id, run.summary = f"run{len(abiertos)}", {}
        abiertos.append((cfg, opciones))
        return run

    monkeypatch.setattr(trabajadores, "abrir_run", abrir)
    monkeypatch.setattr(trabajadores, "subir_artefacto", lambda *a, **k: None)
    monkeypatch.setattr(trabajadores, "terminar_run", lambda run, error=None: None)
    argumentos = _argumentos_gail_reanudable(
        tmp_path, _conjunto(tmp_path / "biblioteca"),
        wandb={"activo": True, "grupo": "dashboard-cartpole"})
    with pytest.raises(KeyboardInterrupt):
        tarea_entrenar_gail(argumentos, CanalQueSeCae(3))
    tarea_entrenar_gail(argumentos, CanalFalso())
    (_, op0), (cfg1, op1) = abiertos
    assert "reanudado" not in op0["etiquetas"] and "reanudado" in op1["etiquetas"]
    assert cfg1["reanudado"] == {"run_anterior": "run0", "desde_iteracion": 2}


def test_gail_no_se_reanuda_con_otros_datos(tmp_path):
    raiz = tmp_path / "biblioteca"
    argumentos = _argumentos_gail_reanudable(tmp_path, _conjunto(raiz))
    with pytest.raises(KeyboardInterrupt):
        tarea_entrenar_gail(argumentos, CanalQueSeCae(3))
    for cambio in ({"datos": [_conjunto(raiz, partidas=9)]}, {"cantidad": 2}):
        with pytest.raises(ValueError, match="No se puede reanudar"):
            tarea_entrenar_gail({**argumentos, **cambio}, CanalFalso())


# --- reserva de la carpeta y checkpoint incompleto (auditoría) ----------------------------------

@pytest.mark.parametrize("tarea, metodo", [(tarea_entrenar_experto, "experto"),
                                           (tarea_entrenar_gail, "gail"),
                                           (tarea_entrenar_bc, "bc")])
def test_con_la_carpeta_en_uso_se_rechaza_sin_entrenar(tmp_path, tarea, metodo):
    """Regresión: dos entrenamientos con la misma carpeta se pisaban."""
    from src.training.checkpoints import CarpetaOcupada, en_uso
    raiz = tmp_path / "biblioteca"
    argumentos = (_argumentos_experto_reanudable(tmp_path) if metodo == "experto" else
                  _argumentos_gail_reanudable(tmp_path, _conjunto(raiz)) if metodo == "gail"
                  else _argumentos_bc_reanudable(tmp_path, _conjunto(raiz)))
    canal = CanalFalso()
    with en_uso(Path(argumentos["checkpoint"])):
        with pytest.raises(CarpetaOcupada, match="Ya hay un entrenamiento"):
            tarea(argumentos, canal)
    assert canal.progresos == []
    biblioteca = Biblioteca(raiz, {"cartpole": "CartPole-v1"})
    assert biblioteca.listar("cartpole", (metodo,)) == []
    # Suelta la reserva, el siguiente entrena.
    assert tarea(argumentos, CanalFalso())["modelo"]["incompleto"] is False


def test_dos_reanudaciones_a_la_vez_solo_sigue_una(tmp_path):
    """Regresión: la que terminaba borraba la carpeta y la otra fallaba tras entrenar."""
    import threading
    from src.training.checkpoints import CarpetaOcupada
    raiz = tmp_path / "biblioteca"
    argumentos = _argumentos_bc_reanudable(tmp_path, _conjunto(raiz))
    argumentos["ajustes"]["bc"]["epocas_maximas"] = 300
    with pytest.raises(KeyboardInterrupt):
        tarea_entrenar_bc(argumentos, CanalQueSeCae(3))
    resultados = {}

    def lanzar(n: int) -> None:
        try:
            resultados[n] = tarea_entrenar_bc(dict(argumentos), CanalFalso())
        except Exception as error:  # noqa: BLE001 - se mira cuál falló y cómo
            resultados[n] = error

    hilos = [threading.Thread(target=lanzar, args=(n,)) for n in range(2)]
    for hilo in hilos:
        hilo.start()
    for hilo in hilos:
        hilo.join()
    terminados = [r for r in resultados.values() if isinstance(r, dict)]
    rechazados = [r for r in resultados.values() if isinstance(r, CarpetaOcupada)]
    assert len(terminados) == 1 and len(rechazados) == 1
    assert terminados[0]["resumen"]["reanudado_desde"] == [2]
    assert len(Biblioteca(raiz, {"cartpole": "CartPole-v1"}).listar("cartpole", ("bc",))) == 1
    assert not Path(argumentos["checkpoint"]).exists()


@pytest.mark.parametrize("cambio, mensaje", [
    (lambda c: c.pop("config"), "no está completo"),
    (lambda c: c.pop("segundos"), "no está completo"),
    (lambda c: c.update(extra=1), "no está completo"),
    (lambda c: c.update(estado={}), "el estado"),
    (lambda c: c.update(estado=[]), "el estado"),
    (lambda c: c.update(progreso={}), "el progreso"),
    (lambda c: c.update(progreso=[1]), "el progreso"),
    (lambda c: c.update(reanudado_desde=[0]), "el progreso"),
    (lambda c: c.update(reanudado_desde=None), "el progreso"),
    (lambda c: c.update(run=3), "el progreso"),
    (lambda c: c.update(segundos=-1.0), "el progreso"),
    (lambda c: c.update(segundos=float("nan")), "el progreso"),
])
def test_un_checkpoint_incompleto_da_un_error_claro(tmp_path, cambio, mensaje):
    """Regresión: le faltaba una clave y daba KeyError."""
    from src.training.checkpoints import escribir_checkpoint, leer_checkpoint
    raiz = tmp_path / "biblioteca"
    argumentos = _argumentos_bc_reanudable(tmp_path, _conjunto(raiz))
    with pytest.raises(KeyboardInterrupt):
        tarea_entrenar_bc(argumentos, CanalQueSeCae(3))
    carpeta = Path(argumentos["checkpoint"])
    contenido = {k: v for k, v in leer_checkpoint(carpeta, "bc").items()
                 if k not in ("formato", "version", "metodo")}
    cambio(contenido)
    escribir_checkpoint(carpeta, "bc", contenido)
    canal = CanalFalso()
    with pytest.raises(ValueError, match=mensaje):
        tarea_entrenar_bc(argumentos, canal)
    assert canal.progresos == []
    # El error suelta la reserva: con la carpeta ya libre, otro intento da el mismo error.
    with pytest.raises(ValueError, match=mensaje):
        tarea_entrenar_bc(argumentos, CanalFalso())


def _argumentos_en_curso(carpeta: Path) -> dict:
    return {"entorno": "cartpole", "nombre": "x", "ajustes": {}, "semilla": 1, "datos": ["a"],
            "cantidad": None, "checkpoint": str(carpeta)}


def test_en_curso_suelta_la_carpeta_al_salir_aunque_siga_vivo(tmp_path):
    """La reserva se suelta al salir del bloque, no cuando se libera el objeto."""
    from src.training.checkpoints import CarpetaOcupada, en_uso
    carpeta = tmp_path / "t"
    with trabajadores.EnCurso(_argumentos_en_curso(carpeta), "bc", {}, "epoca") as en_curso:
        with pytest.raises(CarpetaOcupada):
            with en_uso(carpeta):
                pass
    with en_uso(carpeta):  # en_curso sigue vivo aquí
        pass
    assert en_curso.previo is None


def test_en_curso_suelta_la_carpeta_si_falla_al_prepararla(tmp_path):
    """Con la excepción (y su traceback) aún viva, la carpeta ya está libre."""
    from src.training.checkpoints import en_uso, escribir_checkpoint
    carpeta = tmp_path / "t"
    trabajadores.EnCurso(_argumentos_en_curso(carpeta), "bc", {}, "epoca").__exit__()
    escribir_checkpoint(carpeta, "bc", {"estado": {}})  # incompleto
    with pytest.raises(ValueError, match="no está completo") as fallo:
        trabajadores.EnCurso(_argumentos_en_curso(carpeta), "bc", {}, "epoca")
    with en_uso(carpeta):
        pass
    assert fallo.value is not None
