"""Pruebas del TRPO de sb3-contrib: creación, iteraciones, experto y guardado.

Cada comportamiento se prueba con acciones discretas (CartPole-v1) y continuas
(Pendulum-v1, en lugar de Reacher y HalfCheetah, que necesitan MuJoCo). Las
configuraciones parten de las del proyecto con presupuestos pequeños para que
los tests sean rápidos.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import src.training.trpo as trpo
from src.data.entornos import crear_entorno, crear_entorno_vectorizado
from src.models.factory import construir_discriminador
from src.training.gail import EntornoRecompensaGAIL
from src.data.demostraciones import (generar_dataset, huella_partidas, jugar_partida,
                                     retorno_partida)
from src.data.splits import huella_archivo
from src.training.trpo import (cargar_experto, cargar_politica_experto, config_trpo,
                               config_trpo_experto, crear_trpo,
                               entrenar_experto, estadisticas_normalizacion, evaluar_retorno,
                               guardar_experto, huella_politica, iteracion_trpo,
                               normalizar_observacion, politica_trpo)
from src.utils import archivos, procedencia
from src.utils.configuracion import resolver_configuracion
from src.utils.seed import (fijar_seed, generador_torch_aislado, reiniciar_con_semilla,
                            semillas_independientes)

RAIZ = Path(__file__).resolve().parent.parent
ENTORNOS = pytest.mark.parametrize("entorno", ["CartPole-v1", "Pendulum-v1"],
                                   ids=["discreto", "continuo"])


def _cfg(entorno: str = "CartPole-v1", **trpo_cambios) -> dict:
    """Configuración del proyecto con un entorno y un presupuesto pequeño."""
    cfg = resolver_configuracion(RAIZ / "configs" / "default.yaml", "cartpole")
    cfg["entorno"]["id"] = entorno
    cfg["trpo"].update(pasos_por_iteracion=64, batch_size_valor=32, n_entornos=1,
                       capas_ocultas=[16, 16])
    cfg["trpo"].update(trpo_cambios)
    cfg["experto"].update(pasos_totales=256, evaluar_cada=2, episodios_evaluacion=2)
    return cfg


def _agente(cfg: dict, semilla: int = 0):
    fijar_seed(semilla)
    venv = crear_entorno_vectorizado(cfg, cfg["trpo"]["n_entornos"], 3)
    return crear_trpo(cfg, venv), venv


# --- crear_trpo -----------------------------------------------------------------

@ENTORNOS
@pytest.mark.parametrize("n_entornos", [1, 2])
def test_crear_trpo_usa_los_hiperparametros_de_la_configuracion(entorno, n_entornos):
    cfg = resolver_configuracion(RAIZ / "configs" / "default.yaml", "cartpole")
    cfg["entorno"]["id"] = entorno
    cfg["trpo"]["n_entornos"] = n_entornos
    modelo, venv = _agente(cfg)
    try:
        tcfg = cfg["trpo"]
        assert modelo.n_steps * n_entornos == tcfg["pasos_por_iteracion"] == 5000
        assert modelo.batch_size == tcfg["batch_size_valor"]
        assert (modelo.gamma, modelo.gae_lambda) == (tcfg["gamma"], tcfg["gae_lambda"])
        assert (modelo.target_kl, modelo.cg_damping) == (tcfg["target_kl"], tcfg["cg_damping"])
        assert modelo.cg_max_steps == tcfg["pasos_gradiente_conjugado"]
        assert modelo.line_search_shrinking_factor == tcfg["busqueda_lineal_factor"]
        assert modelo.line_search_max_iter == tcfg["busqueda_lineal_maximo"]
        assert modelo.n_critic_updates == tcfg["pasos_valor"]
        assert modelo.normalize_advantage is tcfg["normalizar_ventajas"]
        assert modelo.learning_rate == tcfg["learning_rate_valor"]
        assert modelo.device.type == "cpu"
        # Política y valor: 2 × 100 tanh, como el paper.
        for red in (modelo.policy.mlp_extractor.policy_net, modelo.policy.mlp_extractor.value_net):
            lineales = [m for m in red if isinstance(m, torch.nn.Linear)]
            assert [m.out_features for m in lineales] == [100, 100]
            assert sum(isinstance(m, torch.nn.Tanh) for m in red) == 2
    finally:
        venv.close()


def test_crear_trpo_no_siembra_por_su_cuenta(monkeypatch):
    # sb3 resiembra los generadores globales y los entornos si recibe seed=.
    def prohibido(*args, **kwargs):
        raise AssertionError("crear_trpo no debe sembrar desde sb3")
    monkeypatch.setattr("stable_baselines3.common.base_class.set_random_seed", prohibido)
    modelo, venv = _agente(_cfg())
    iteracion_trpo(modelo)
    venv.close()


@pytest.mark.parametrize("clave,valor,mensaje", [
    ("pasos_por_iteracion", 65, "divisible por trpo.n_entornos"),
    ("batch_size_valor", 30, "debe dividir"),
    ("capas_ocultas", [], "capas_ocultas"),
    ("capas_ocultas", [0], "capas_ocultas"),
    ("activacion", "sigmoide", "activacion"),
    ("gamma", 0, "gamma"),
    ("gamma", 1.5, "gamma"),
    ("gae_lambda", -0.1, "gae_lambda"),
    ("target_kl", 0, "target_kl"),
    ("cg_damping", -1, "cg_damping"),
    ("n_entornos", True, "n_entornos"),
    ("normalizar_observaciones", "si", "normalizar_observaciones"),
    ("log_std_inicial", float("inf"), "log_std_inicial"),
    ("log_std_inicial", "-1", "log_std_inicial"),
    ("inicializacion_ortogonal", 1, "inicializacion_ortogonal"),
    ("limite_normalizacion", 0, "limite_normalizacion"),
    ("normalizar_ventajas", 1, "normalizar_ventajas"),
    ("pasos_gradiente_conjugado", 0, "pasos_gradiente_conjugado"),
    ("busqueda_lineal_factor", 1.0, "busqueda_lineal_factor"),
    ("busqueda_lineal_maximo", 1.5, "busqueda_lineal_maximo"),
    ("learning_rate_valor", float("nan"), "learning_rate_valor"),
    ("pasos_valor", -1, "pasos_valor"),
    # Regresión: con una sola transición, sb3 fallaba con un AssertionError.
    ("pasos_por_iteracion", 1, "al menos 2"),
    ("batch_size_valor", 1, "al menos 2"),
])
def test_crear_trpo_rechaza_configuraciones_no_validas(clave, valor, mensaje):
    cfg = _cfg(**{"n_entornos": 2} if (clave, valor) == ("pasos_por_iteracion", 65) else {})
    if (clave, valor) == ("pasos_por_iteracion", 1):
        cfg["trpo"]["batch_size_valor"] = 1
    venv = crear_entorno_vectorizado(cfg, cfg["trpo"]["n_entornos"], 0)
    cfg["trpo"][clave] = valor
    try:
        with pytest.raises(ValueError, match=mensaje):
            crear_trpo(cfg, venv)
    finally:
        venv.close()


def test_crear_trpo_exige_la_seccion_y_un_vecenv_coherente():
    cfg = _cfg()
    venv = crear_entorno_vectorizado(cfg, 2, 0)
    try:
        with pytest.raises(ValueError, match="2 entornos"):
            crear_trpo(cfg, venv)
        with pytest.raises(ValueError, match="VecEnv"):
            crear_trpo(cfg, crear_entorno(cfg, 0))
        sin_clave = _cfg()
        del sin_clave["trpo"]["pasos_valor"]
        with pytest.raises(ValueError, match="Falta trpo.pasos_valor"):
            crear_trpo(sin_clave, venv)
    finally:
        venv.close()


# --- iteracion_trpo ---------------------------------------------------------------

@ENTORNOS
@pytest.mark.parametrize("n_entornos", [1, 2])
def test_cada_iteracion_da_exactamente_un_bloque_sin_reiniciar(entorno, n_entornos, monkeypatch):
    cfg = _cfg(entorno, n_entornos=n_entornos)
    modelo, venv = _agente(cfg)
    reinicios = []
    original = venv.reset
    monkeypatch.setattr(venv, "reset", lambda: (reinicios.append(1), original())[1])
    try:
        for i in range(1, 4):
            medidas = iteracion_trpo(modelo)
            assert medidas["pasos"] == modelo.num_timesteps == 64 * i
        # learn solo reinicia el VecEnv la primera vez; después siguen las partidas.
        assert len(reinicios) == 1
    finally:
        venv.close()


def test_iteracion_mide_el_retorno_real_bajo_el_envoltorio_de_gail():
    cfg = _cfg(pasos_por_iteracion=128)
    fijar_seed(0)
    venv = crear_entorno_vectorizado(cfg, 1, 3)
    discriminador = construir_discriminador(cfg, venv.observation_space, venv.action_space)
    envuelto = EntornoRecompensaGAIL(venv, discriminador, cfg["gail"]["recompensa"])
    modelo = crear_trpo(cfg, envuelto)
    try:
        envuelto.iniciar_bloque()
        medidas = iteracion_trpo(modelo)
        # CartPole da 1 por paso: el retorno real es la longitud de la partida,
        # aunque el agente haya recibido log 2 por paso del discriminador.
        assert medidas["retorno_real_medio"] == medidas["longitud_media"]
        assert envuelto.extraer_lote().numero_transiciones == 128
        assert np.allclose(modelo.rollout_buffer.rewards, math.log(2))
    finally:
        envuelto.close()


def test_las_medidas_son_solo_de_las_partidas_del_bloque():
    # Regresión: el retorno promediaba las últimas 100 partidas de sb3, también
    # las de iteraciones anteriores. Monitor cuenta las partidas de verdad.
    modelo, venv = _agente(_cfg())
    monitor = venv.envs[0]
    try:
        for _ in range(6):
            antes = len(monitor.get_episode_lengths())
            medidas = iteracion_trpo(modelo)
            nuevas = monitor.get_episode_lengths()[antes:]
            assert medidas["partidas"] == len(nuevas)
            if nuevas:
                assert medidas["longitud_media"] == pytest.approx(np.mean(nuevas))
            else:
                assert math.isnan(medidas["retorno_real_medio"])
    finally:
        venv.close()


def test_sin_partidas_terminadas_el_retorno_es_nan():
    modelo, venv = _agente(_cfg("Pendulum-v1"))  # Pendulum dura 200 pasos > 64
    try:
        assert math.isnan(iteracion_trpo(modelo)["retorno_real_medio"])
    finally:
        venv.close()


@ENTORNOS
@pytest.mark.parametrize("n_entornos", [1, 2])
def test_misma_semilla_mismos_parametros_tras_dos_iteraciones(entorno, n_entornos):
    # También los estados recogidos, desde el primero: el reset que hace sb3 al
    # empezar continúa la secuencia sembrada en vez de usar una semilla nueva.
    resultados, estados = [], []
    for _ in range(2):
        modelo, venv = _agente(_cfg(entorno, n_entornos=n_entornos), semilla=11)
        iteracion_trpo(modelo)
        estados.append(modelo.rollout_buffer.observations.copy())
        iteracion_trpo(modelo)
        resultados.append({k: v.clone() for k, v in modelo.policy.state_dict().items()})
        venv.close()
    assert np.array_equal(*estados)
    assert all(torch.equal(v, resultados[1][k]) for k, v in resultados[0].items())


def test_iteracion_rechaza_lo_que_no_es_trpo():
    with pytest.raises(ValueError, match="TRPO"):
        iteracion_trpo(object())


# --- evaluación y normalización ---------------------------------------------------

@ENTORNOS
@pytest.mark.parametrize("determinista", [True, False])
def test_evaluar_retorno_es_reproducible(entorno, determinista):
    # Regresión: las acciones muestreadas gastaban el generador global de PyTorch,
    # así que la misma semilla podía dar otro resultado y evaluar alteraba el
    # entrenamiento. Sin fijar_seed entre medias, el resultado es el mismo y el
    # generador global queda como estaba.
    cfg = _cfg(entorno)
    modelo, venv = _agente(cfg)
    venv.close()
    estado = torch.get_rng_state()
    a = evaluar_retorno(modelo, cfg, 3, 9, determinista)
    assert torch.equal(torch.get_rng_state(), estado)
    torch.rand(7)  # el generador global cambia y no debe influir
    b = evaluar_retorno(modelo, cfg, 3, 9, determinista)
    assert a == b
    assert set(a) == {"retorno_medio", "retorno_desviacion", "longitud_media",
                      "retornos", "longitudes"}
    assert len(a["longitudes"]) == len(a["retornos"]) == 3
    assert a["longitud_media"] == np.mean(a["longitudes"])
    assert len(a["retornos"]) == 3


@ENTORNOS
def test_cada_episodio_depende_solo_de_su_semilla(entorno):
    cfg = _cfg(entorno)
    modelo, venv = _agente(cfg)
    venv.close()
    cinco = evaluar_retorno(modelo, cfg, 5, 9, False)["retornos"]
    dos = evaluar_retorno(modelo, cfg, 2, 9, False)["retornos"]
    assert cinco[:2] == dos


def test_evaluar_mas_a_menudo_no_cambia_el_entrenamiento():
    # Regresión: con la evaluación gastando el generador global, evaluar en cada
    # iteración cambiaba las acciones de entrenamiento siguientes.
    resultados = []
    for evaluar_cada in (1, 4):
        cfg = _cfg()
        cfg["experto"].update(pasos_totales=64 * 4, evaluar_cada=evaluar_cada,
                              retorno_objetivo=10_000)
        _, resumen = entrenar_experto(cfg)
        resultados.append(resumen["evaluaciones"][-1])
    assert resultados[0]["iteracion"] == resultados[1]["iteracion"] == 4
    assert resultados[0]["retornos"] == resultados[1]["retornos"]


def test_entrenar_experto_registra_cada_medida_en_su_seccion_de_wandb():
    """Las medidas de TRPO se devolvían sin prefijo y caían en la sección genérica."""
    registros = []
    run = SimpleNamespace(log=lambda metricas, step=None: registros.append(dict(metricas)))
    cfg = _cfg()
    cfg["experto"].update(pasos_totales=64 * 2, evaluar_cada=2, retorno_objetivo=10_000)
    entrenar_experto(cfg, run)
    assert {clave.split("/")[0] for m in registros for clave in m} == {"trpo", "evaluacion"}
    assert {"trpo/pasos", "trpo/partidas", "trpo/retorno_real_medio",
            "trpo/longitud_media"} == set(registros[0])


def test_evaluar_retorno_juega_episodios_completos_con_su_semilla():
    cfg = _cfg()
    modelo, venv = _agente(cfg)
    venv.close()
    resultado = evaluar_retorno(modelo, cfg, 4, 2, True)
    # CartPole da 1 por paso: retorno = longitud.
    assert resultado["retorno_medio"] == resultado["longitud_media"]
    otra = evaluar_retorno(modelo, cfg, 4, 3, True)
    assert otra != resultado
    with pytest.raises(ValueError, match="episodios"):
        evaluar_retorno(modelo, cfg, 0, 2, True)
    with pytest.raises(ValueError, match="semilla"):
        evaluar_retorno(modelo, cfg, 2, -1, True)


def test_normalizar_observacion_coincide_con_vecnormalize():
    cfg = _cfg("Pendulum-v1")
    venv = VecNormalize(crear_entorno_vectorizado(cfg, 2, 0), norm_obs=True, norm_reward=False,
                        clip_obs=0.5)
    try:
        venv.reset()
        for _ in range(50):
            observaciones, _, _, _ = venv.step(np.zeros((2, 1), np.float32))
        estadisticas = estadisticas_normalizacion(venv)
        crudas = venv.get_original_obs()
        normalizadas = normalizar_observacion(crudas, estadisticas)
        np.testing.assert_allclose(normalizadas, venv.normalize_obs(crudas), rtol=1e-6, atol=1e-6)
        # Con un límite bajo, alguna componente se recorta: el límite se aplica.
        assert estadisticas["limite"] == 0.5 and np.isclose(np.abs(normalizadas).max(), 0.5)
    finally:
        venv.close()
    sin = crear_entorno_vectorizado(cfg, 1, 0)
    assert estadisticas_normalizacion(sin) is None
    assert normalizar_observacion(np.ones(3), None) is not None
    sin.close()


# --- entrenar_experto ---------------------------------------------------------------

def _evaluaciones_falsas(monkeypatch, retornos):
    """Sustituye la evaluación por una secuencia de retornos conocida."""
    pendientes = list(retornos)
    llamadas = []

    def falsa(modelo, cfg, episodios, semilla, determinista, estadisticas=None):
        llamadas.append({"pesos": {k: v.clone() for k, v in modelo.policy.state_dict().items()},
                         "determinista": determinista, "episodios": episodios})
        return {"retorno_medio": pendientes.pop(0), "retorno_desviacion": 0.0,
                "longitud_media": 1.0}
    monkeypatch.setattr(trpo, "evaluar_retorno", falsa)
    return llamadas


def test_el_experto_se_queda_con_la_mejor_evaluacion(monkeypatch):
    cfg = _cfg()
    cfg["experto"].update(pasos_totales=64 * 6, evaluar_cada=2, retorno_objetivo=1000)
    llamadas = _evaluaciones_falsas(monkeypatch, [10.0, 50.0, 30.0])
    modelo, resumen = entrenar_experto(cfg)
    assert len(llamadas) == 3 and resumen["iteraciones"] == 6
    assert resumen["mejor_retorno"] == 50.0 and resumen["mejor_iteracion"] == 4
    assert not resumen["objetivo_alcanzado"]
    finales = modelo.policy.state_dict()
    assert all(torch.equal(v, llamadas[1]["pesos"][k]) for k, v in finales.items())
    assert [e["retorno_medio"] for e in resumen["evaluaciones"]] == [10.0, 50.0, 30.0]
    # Por defecto se evalúa la política estocástica, que es la que graba las demostraciones.
    assert all(not llamada["determinista"] for llamada in llamadas)


def test_el_experto_para_al_alcanzar_el_objetivo(monkeypatch):
    cfg = _cfg()
    cfg["experto"].update(pasos_totales=64 * 20, evaluar_cada=1, retorno_objetivo=40)
    llamadas = _evaluaciones_falsas(monkeypatch, [10.0, 39.9, 40.0, 99.0])
    _, resumen = entrenar_experto(cfg)
    assert len(llamadas) == 3
    assert resumen["objetivo_alcanzado"] and resumen["pasos"] == 64 * 3


def test_el_experto_respeta_el_presupuesto_y_evalua_al_final(monkeypatch):
    cfg = _cfg()
    # 4 bloques de 64; evaluar_cada 3 evalúa en la 3 y, por acabarse el
    # presupuesto, en la 4.
    cfg["experto"].update(pasos_totales=256, evaluar_cada=3, retorno_objetivo=1000)
    llamadas = _evaluaciones_falsas(monkeypatch, [1.0, 2.0])
    _, resumen = entrenar_experto(cfg)
    assert resumen["iteraciones"] == 4 and resumen["pasos"] == 256 and len(llamadas) == 2
    assert [e["iteracion"] for e in resumen["evaluaciones"]] == [3, 4]
    assert resumen["segundos"] > 0
    assert set(resumen["semillas"]) == {"base", "modelo", "entornos", "evaluacion"}


@pytest.mark.parametrize("presupuesto,iteraciones", [(192, 3), (193, 4), (200, 4), (255, 4),
                                                     (256, 4), (257, 5), (1, 1)])
def test_el_experto_para_en_la_primera_iteracion_que_llega_al_presupuesto(
    presupuesto, iteraciones, monkeypatch
):
    """Sin múltiplo exacto, la última iteración se pasa y los pasos reales quedan en el resumen."""
    cfg = _cfg()  # Bloques de 64 pasos.
    cfg["experto"].update(pasos_totales=presupuesto, evaluar_cada=100, retorno_objetivo=1000)
    llamadas = _evaluaciones_falsas(monkeypatch, [1.0])
    _, resumen = entrenar_experto(cfg)
    assert resumen["iteraciones"] == iteraciones
    assert resumen["pasos"] == 64 * iteraciones >= presupuesto
    assert resumen["pasos"] - presupuesto < 64
    # Siempre se evalúa al terminar, aunque no toque por evaluar_cada.
    assert len(llamadas) == 1 and resumen["evaluaciones"][0]["iteracion"] == iteraciones


def test_el_experto_es_reproducible():
    cfg = _cfg()
    resultados = [entrenar_experto(cfg) for _ in range(2)]
    (modelo_a, resumen_a), (modelo_b, resumen_b) = resultados
    assert resumen_a["evaluaciones"] == resumen_b["evaluaciones"]
    estado_b = modelo_b.policy.state_dict()
    assert all(torch.equal(v, estado_b[k]) for k, v in modelo_a.policy.state_dict().items())


def test_el_experto_normaliza_si_lo_pide_el_perfil():
    cfg = _cfg("Pendulum-v1", normalizar_observaciones=True)
    cfg["experto"].update(pasos_totales=128, evaluar_cada=1, retorno_objetivo=0)
    _, resumen = entrenar_experto(cfg)
    normalizacion = resumen["normalizacion"]
    assert normalizacion is not None and len(normalizacion["media"]) == 3
    _, sin = entrenar_experto(_cfg("Pendulum-v1"))
    assert sin["normalizacion"] is None


@pytest.mark.parametrize("log_std,ortogonal", [(0.0, True), (-1.0, False)])
def test_crear_trpo_aplica_la_inicializacion_de_la_politica(log_std, ortogonal):
    cfg = _cfg("Pendulum-v1", log_std_inicial=log_std, inicializacion_ortogonal=ortogonal)
    agente, venv = _agente(cfg)
    try:
        assert torch.allclose(agente.policy.log_std, torch.full_like(agente.policy.log_std,
                                                                     log_std))
        assert agente.policy.ortho_init is ortogonal
    finally:
        venv.close()


def test_los_valores_por_defecto_de_la_politica_son_los_de_sb3():
    """Regresión: las claves nuevas no cambian la política que ya se creaba."""
    from sb3_contrib import TRPO
    cfg = _cfg("Pendulum-v1")
    fijar_seed(0)
    venv = crear_entorno_vectorizado(cfg, 1, 3)
    try:
        propio = crear_trpo(cfg, venv)
        fijar_seed(0)
        sb3 = TRPO("MlpPolicy", venv, policy_kwargs={
            "net_arch": {"pi": [16, 16], "vf": [16, 16]}, "activation_fn": torch.nn.Tanh},
            device="cpu")
        estado = sb3.policy.state_dict()
        assert all(torch.equal(v, estado[k]) for k, v in propio.policy.state_dict().items())
    finally:
        venv.close()


# --- experto.trpo: hiperparámetros propios del experto ------------------------------

def test_config_trpo_experto_cambia_solo_sus_claves_y_no_toca_cfg():
    cfg = _cfg()
    original = copy.deepcopy(cfg)
    assert config_trpo_experto(cfg)["trpo"] == cfg["trpo"]
    cfg["experto"]["trpo"] = {"target_kl": 0.04, "capas_ocultas": [8]}
    combinada = config_trpo_experto(cfg)
    assert combinada["trpo"] == {**original["trpo"], "target_kl": 0.04, "capas_ocultas": [8]}
    assert cfg["trpo"] == original["trpo"]
    assert config_trpo(cfg)["target_kl"] == original["trpo"]["target_kl"]


@pytest.mark.parametrize("cambios,mensaje", [
    ({"target_kl": 0}, "trpo.target_kl"),
    ({"pasos_por_iteracion": 48}, "debe dividir"),
    ([], "experto.trpo debe ser un diccionario"),
])
def test_config_trpo_experto_comprueba_la_seccion_resultante(cambios, mensaje):
    cfg = _cfg()
    cfg["experto"]["trpo"] = cambios
    with pytest.raises(ValueError, match=mensaje):
        config_trpo_experto(cfg)
    with pytest.raises(ValueError, match=mensaje):
        entrenar_experto(cfg)


def test_el_experto_entrena_con_experto_trpo():
    cfg = _cfg()
    cfg["experto"]["trpo"] = {"capas_ocultas": [8]}
    modelo, _ = entrenar_experto(cfg)
    assert modelo.policy.mlp_extractor.policy_net[0].out_features == 8
    sin_cambios, _ = entrenar_experto(_cfg())
    assert sin_cambios.policy.mlp_extractor.policy_net[0].out_features == 16


def test_un_experto_trpo_con_los_mismos_valores_da_el_mismo_experto():
    """Regresión: sin cambios efectivos, el experto sale idéntico al de antes."""
    cfg = _cfg()
    igual = _cfg()
    igual["experto"]["trpo"] = {"target_kl": cfg["trpo"]["target_kl"]}
    (modelo_a, resumen_a), (modelo_b, resumen_b) = entrenar_experto(cfg), entrenar_experto(igual)
    assert resumen_a["evaluaciones"] == resumen_b["evaluaciones"]
    estado_b = modelo_b.policy.state_dict()
    assert all(torch.equal(v, estado_b[k]) for k, v in modelo_a.policy.state_dict().items())


@pytest.mark.parametrize("clave,valor", [
    ("pasos_totales", 0), ("retorno_objetivo", "500"), ("semilla", -1), ("semilla", True),
    ("semilla", 1.5), ("semilla", np.int64(-3)),
    ("evaluar_cada", 0), ("episodios_evaluacion", 1.5), ("evaluacion_determinista", "no"),
])
def test_entrenar_experto_rechaza_configuraciones_no_validas(clave, valor):
    cfg = _cfg()
    cfg["experto"][clave] = valor
    with pytest.raises(ValueError, match=f"experto.{clave}"):
        entrenar_experto(cfg)


def test_cartpole_mejora_en_pocas_iteraciones():
    """Integración con presupuesto pequeño: el retorno sube respecto al inicial."""
    cfg = _cfg(pasos_por_iteracion=512, batch_size_valor=128, capas_ocultas=[64, 64])
    cfg["experto"].update(pasos_totales=512 * 6, evaluar_cada=6, episodios_evaluacion=5)
    modelo_inicial, venv = _agente(cfg, semilla=0)
    venv.close()
    inicial = evaluar_retorno(modelo_inicial, cfg, 5, 1, False)["retorno_medio"]
    _, resumen = entrenar_experto(cfg)
    assert resumen["mejor_retorno"] > inicial


# --- guardar y cargar ---------------------------------------------------------------

@ENTORNOS
def test_guardar_y_cargar_dan_las_mismas_acciones(entorno, tmp_path):
    cfg = _cfg(entorno)
    modelo, resumen = entrenar_experto(cfg)
    guardar_experto(tmp_path / "experto", modelo, cfg, resumen)
    cargado, metadatos = cargar_experto(tmp_path / "experto")
    env = crear_entorno(cfg, None)
    try:
        observacion, _ = reiniciar_con_semilla(env, 4)
        for _ in range(20):
            a, _ = modelo.predict(observacion, deterministic=True)
            b, _ = cargado.predict(observacion, deterministic=True)
            np.testing.assert_array_equal(a, b)
            observacion, _, fin, corte, _ = env.step(a)
            if fin or corte:
                break
    finally:
        env.close()
    assert metadatos["configuracion"] == json.loads(json.dumps(cfg))
    texto = (tmp_path / "experto" / "experto.json").read_text(encoding="utf-8")

    def rechazar(constante):
        raise AssertionError(f"JSON no estricto: {constante}")
    json.loads(texto, parse_constant=rechazar)
    assert metadatos["resumen"]["mejor_retorno"] == resumen["mejor_retorno"]
    assert {"python", "torch", "sb3_contrib", "gymnasium", "procesador"} <= set(
        metadatos["entorno_ejecucion"])
    assert metadatos["codigo"] == procedencia.estado_git()


def test_guardar_no_sobrescribe_y_cargar_exige_los_archivos(tmp_path):
    cfg = _cfg()
    modelo, resumen = entrenar_experto(cfg)
    guardar_experto(tmp_path / "experto", modelo, cfg, resumen)
    with pytest.raises(FileExistsError, match="no se sobrescribe"):
        guardar_experto(tmp_path / "experto", modelo, cfg, resumen)
    with pytest.raises(FileNotFoundError, match="modelo.zip"):
        cargar_experto(tmp_path / "vacio")
    (tmp_path / "experto" / "experto.json").unlink()
    with pytest.raises(FileNotFoundError, match="experto.json"):
        cargar_experto(tmp_path / "experto")
    with pytest.raises(ValueError, match="TRPO"):
        guardar_experto(tmp_path / "otro", object(), cfg, resumen)
    # JSON estricto: un NaN da error y no deja nada a medias.
    with pytest.raises(ValueError):
        guardar_experto(tmp_path / "nan", modelo, cfg, {**resumen, "segundos": float("nan")})
    assert not (tmp_path / "nan").exists()


def test_dummyvecenv_sin_monitor_no_rompe_la_iteracion():
    # Un VecEnv ajeno (sin Monitor) funciona; solo no hay retorno real que medir.
    cfg = _cfg()
    fijar_seed(0)
    venv = DummyVecEnv([lambda: crear_entorno(cfg, 0)])
    modelo = crear_trpo(cfg, venv)
    try:
        assert math.isnan(iteracion_trpo(modelo)["retorno_real_medio"])
    finally:
        venv.close()


# --- escribir_carpeta_nueva -----------------------------------------------------------

def _escribir_dos(carpeta: Path) -> None:
    (carpeta / "a.txt").write_text("a")
    (carpeta / "b.txt").write_text("b")


@pytest.mark.parametrize("previa", ["no_existe", "vacia"])
def test_escribir_carpeta_nueva_deja_la_carpeta_entera(tmp_path, previa):
    destino = tmp_path / "padre" / "resultado"
    if previa == "vacia":
        destino.mkdir(parents=True)
    archivos.escribir_carpeta_nueva(destino, _escribir_dos)
    assert sorted(p.name for p in destino.iterdir()) == ["a.txt", "b.txt"]
    assert [p.name for p in destino.parent.iterdir()] == ["resultado"]  # sin temporales


@pytest.mark.parametrize("error", [OSError("disco lleno"), KeyboardInterrupt()])
def test_escribir_carpeta_nueva_no_deja_nada_si_falla(tmp_path, error):
    """Regresión: un fallo a mitad (disco lleno, Ctrl+C) dejaba una carpeta incompleta que
    además impedía volver a guardar ahí."""
    destino = tmp_path / "resultado"

    def falla_a_mitad(carpeta: Path) -> None:
        (carpeta / "a.txt").write_text("a")
        raise error

    with pytest.raises(type(error)):
        archivos.escribir_carpeta_nueva(destino, falla_a_mitad)
    assert list(tmp_path.iterdir()) == []  # ni la carpeta ni la temporal
    archivos.escribir_carpeta_nueva(destino, _escribir_dos)  # se puede volver a guardar
    assert sorted(p.name for p in destino.iterdir()) == ["a.txt", "b.txt"]


def test_escribir_carpeta_nueva_no_sobrescribe(tmp_path):
    destino = tmp_path / "resultado"
    destino.mkdir()
    (destino / "notas.txt").write_text("x")
    with pytest.raises(FileExistsError):
        archivos.escribir_carpeta_nueva(destino, _escribir_dos)

    # Si otro proceso la llena mientras se escribe, tampoco: la temporal se borra.
    otra = tmp_path / "otra"

    def ocupar(carpeta: Path) -> None:
        _escribir_dos(carpeta)
        otra.mkdir()
        (otra / "ajeno.txt").write_text("x")

    with pytest.raises(FileExistsError):
        archivos.escribir_carpeta_nueva(otra, ocupar)
    assert [p.name for p in otra.iterdir()] == ["ajeno.txt"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["otra", "resultado"]


def test_escribir_archivo_nuevo_deja_el_archivo_entero_y_crea_las_carpetas(tmp_path):
    destino = tmp_path / "padre" / "modelo.pt"
    archivos.escribir_archivo_nuevo(destino, lambda temporal: temporal.write_text("todo"))
    assert destino.read_text() == "todo"
    assert [p.name for p in destino.parent.iterdir()] == ["modelo.pt"]  # sin temporales


@pytest.mark.parametrize("error", [OSError("disco lleno"), KeyboardInterrupt()])
def test_escribir_archivo_nuevo_no_deja_nada_si_falla(tmp_path, error):
    destino = tmp_path / "modelo.pt"

    def falla_a_mitad(temporal: Path) -> None:
        temporal.write_text("a medias")
        raise error

    with pytest.raises(type(error)):
        archivos.escribir_archivo_nuevo(destino, falla_a_mitad)
    assert list(tmp_path.iterdir()) == []  # ni el archivo ni el temporal
    archivos.escribir_archivo_nuevo(destino, lambda temporal: temporal.write_text("bien"))
    assert destino.read_text() == "bien"


def test_escribir_archivo_nuevo_no_sobrescribe(tmp_path):
    destino = tmp_path / "modelo.pt"
    destino.write_text("anterior")
    escritos = []
    with pytest.raises(FileExistsError):
        archivos.escribir_archivo_nuevo(destino, lambda temporal: escritos.append(1))
    assert not escritos and destino.read_text() == "anterior"

    # Si otro proceso lo crea mientras se escribe, tampoco: el temporal se borra.
    otro = tmp_path / "otro.pt"

    def ocupar(temporal: Path) -> None:
        temporal.write_text("mío")
        otro.write_text("ajeno")

    with pytest.raises(FileExistsError):
        archivos.escribir_archivo_nuevo(otro, ocupar)
    assert otro.read_text() == "ajeno"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["modelo.pt", "otro.pt"]


def test_guardar_experto_no_deja_una_carpeta_a_medias(tmp_path, monkeypatch):
    cfg = _cfg()
    modelo, resumen = entrenar_experto(cfg)

    def falla(*a, **k):
        raise OSError("disco lleno")

    monkeypatch.setattr(modelo, "save", falla)
    with pytest.raises(OSError, match="disco lleno"):
        trpo.guardar_experto(tmp_path / "experto", modelo, cfg, resumen)
    assert list(tmp_path.iterdir()) == []


# --- scripts/entrenar_experto.py ----------------------------------------------------

def _script():
    """Carga el script como módulo (scripts/ no es un paquete)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("entrenar_experto",
                                                  RAIZ / "scripts" / "entrenar_experto.py")
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def test_estado_git_da_el_commit_actual_y_los_cambios():
    import subprocess
    estado = procedencia.estado_git()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=RAIZ, capture_output=True,
                            text=True, check=True).stdout.strip()
    cambios = subprocess.run(["git", "status", "--porcelain"], cwd=RAIZ, capture_output=True,
                             text=True, check=True).stdout.strip()
    assert estado == {"commit": commit, "cambios_sin_commit": bool(cambios)}
    assert len(commit) == 40


@pytest.mark.parametrize("error", [FileNotFoundError("git"), "proceso"])
def test_estado_git_sin_git_no_falla(error, monkeypatch):
    import subprocess

    def falla(*args, **kwargs):
        if error == "proceso":
            raise subprocess.CalledProcessError(128, args[0])
        raise error
    monkeypatch.setattr(procedencia.subprocess, "run", falla)
    assert procedencia.estado_git() == {"commit": None, "cambios_sin_commit": None}


@pytest.mark.parametrize("codigo,aviso", [
    ({"commit": "a" * 40, "cambios_sin_commit": False}, None),
    ({"commit": "a" * 40, "cambios_sin_commit": True}, "había cambios sin commit"),
    ({"commit": None, "cambios_sin_commit": None}, "no se pudo leer el commit"),
])
def test_avisar_estado_git(codigo, aviso, capsys):
    procedencia.avisar_estado_git(codigo)
    salida = capsys.readouterr()
    assert salida.out == ""
    assert (aviso in salida.err and salida.err.count("\n") == 1) if aviso else salida.err == ""


@pytest.mark.parametrize("codigo,aviso", [
    ({"commit": "a" * 40, "cambios_sin_commit": False}, None),
    ({"commit": "a" * 40, "cambios_sin_commit": True}, "cambios sin commit"),
    ({"commit": None, "cambios_sin_commit": None}, "no se pudo leer el commit"),
])
def test_el_script_avisa_del_estado_de_git(codigo, aviso, monkeypatch, tmp_path, capsys):
    script = _script()
    resumen = {"entorno": "CartPole-v1", "mejor_retorno": 500.0, "mejor_iteracion": 3,
               "objetivo": 500, "objetivo_alcanzado": True, "pasos": 15000, "segundos": 12.0}
    monkeypatch.setattr("src.training.flujo.entrenar_experto",
                        lambda cfg, run: ("modelo", resumen))
    monkeypatch.setattr("src.training.flujo.guardar_experto", lambda *a: None)
    monkeypatch.setattr("src.training.flujo.estado_git", lambda: codigo)
    monkeypatch.setattr("sys.argv", ["entrenar_experto.py", "--sin-wandb",
                                     "--salida", str(tmp_path / "experto")])
    assert script.main() == 0
    error = capsys.readouterr().err
    assert (aviso in error) if aviso else "aviso" not in error


def test_el_script_entrena_y_guarda_con_la_configuracion_resuelta(monkeypatch, tmp_path, capsys):
    script = _script()
    llamadas = {}

    def entrenar(cfg, run):
        llamadas["cfg"], llamadas["run"] = cfg, run
        return "modelo", {"entorno": cfg["entorno"]["id"], "mejor_retorno": 500.0,
                          "mejor_iteracion": 3, "objetivo": 500, "objetivo_alcanzado": True,
                          "pasos": 15000, "segundos": 12.0}

    monkeypatch.setattr("src.training.flujo.entrenar_experto", entrenar)
    monkeypatch.setattr("src.training.flujo.guardar_experto",
                        lambda salida, modelo, cfg, resumen: llamadas.update(salida=salida))
    monkeypatch.setattr("sys.argv", ["entrenar_experto.py", "--sin-wandb",
                                     "--salida", str(tmp_path / "experto")])
    assert script.main() == 0
    assert llamadas["cfg"]["wandb"]["modo"] == "disabled" and llamadas["run"] is None
    assert llamadas["cfg"]["entorno"]["id"] == "CartPole-v1"
    assert llamadas["salida"] == tmp_path / "experto"
    assert "objetivo 500 alcanzado" in capsys.readouterr().out


def test_el_artefacto_del_experto_se_identifica_por_la_huella_del_modelo(
    monkeypatch, tmp_path
):
    """El nombre del artefacto es estable entre grupos y depende del modelo guardado."""
    script = _script()
    resumen = {"entorno": "CartPole-v1", "mejor_retorno": 500.0,
               "mejor_iteracion": 3, "objetivo": 500, "objetivo_alcanzado": True,
               "pasos": 15000, "segundos": 12.0}
    subidas = []
    run = object()

    def guardar(salida, *_):
        salida.mkdir(parents=True)
        (salida / "modelo.zip").write_bytes(b"modelo")
        (salida / "experto.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(script, "comprobar_wandb", lambda *a, **k: object())
    monkeypatch.setattr("src.training.flujo.abrir_run", lambda *a, **k: run)
    monkeypatch.setattr("src.training.flujo.entrenar_experto", lambda *a: (object(), resumen))
    monkeypatch.setattr("src.training.flujo.guardar_experto", guardar)
    monkeypatch.setattr("src.training.flujo.registrar_resumen", lambda *a: None)
    monkeypatch.setattr("src.training.flujo.subir_artefacto", lambda *a: subidas.append(a))
    monkeypatch.setattr("src.training.flujo.terminar_run", lambda *a: None)
    monkeypatch.setattr("src.training.flujo.avisar_estado_git", lambda *a: None)
    destino = tmp_path / "experto"
    monkeypatch.setattr("sys.argv", ["entrenar_experto.py", "--salida", str(destino)])
    assert script.main() == 0
    huella = huella_archivo(destino / "modelo.zip")
    assert subidas[0][1] == f"experto-CartPole-v1-{huella[:12]}"
    assert subidas[0][4]["huella_modelo"] == huella


@pytest.mark.parametrize("fallo,codigo", [(None, 0), ("entrenar", 1), ("guardar", 1)])
def test_el_script_cierra_el_run_con_su_codigo_de_salida(fallo, codigo, monkeypatch, tmp_path):
    """Regresión: finish() sin código marcaba como correcto en W&B un experto fallido."""
    script = _script()
    cierres = []

    class _Run:
        def log(self, *a, **k) -> None:
            pass

        def finish(self, exit_code=None) -> None:
            cierres.append(exit_code)

    resumen = {"entorno": "CartPole-v1", "mejor_retorno": 500.0, "mejor_iteracion": 3,
               "objetivo": 500, "objetivo_alcanzado": True, "pasos": 15000, "segundos": 12.0}

    def entrenar(cfg, run):
        if fallo == "entrenar":
            raise RuntimeError("fallo de prueba")
        return "modelo", resumen

    def guardar(*a):
        if fallo == "guardar":
            raise OSError("disco lleno")
        destino = a[0]
        destino.mkdir(parents=True)
        (destino / "modelo.zip").write_bytes(b"modelo")
        (destino / "experto.json").write_text("{}", encoding="utf-8")

    nombres = []
    monkeypatch.setattr("src.training.flujo.abrir_run",
                        lambda cfg, **k: nombres.append(k["nombre"]) or _Run())
    monkeypatch.setattr(script, "comprobar_wandb", lambda *a, **k: None)
    monkeypatch.setattr("src.training.flujo.entrenar_experto", entrenar)
    monkeypatch.setattr("src.training.flujo.guardar_experto", guardar)
    monkeypatch.setattr("sys.argv", ["entrenar_experto.py", "--salida", str(tmp_path / "e")])
    if fallo is None:
        assert script.main() == 0
    else:
        with pytest.raises((RuntimeError, OSError)):
            script.main()
    assert cierres == [codigo]
    assert nombres == ["tiiac-experto-cartpole"]


def test_el_script_informa_de_una_configuracion_no_valida(monkeypatch, tmp_path, capsys):
    script = _script()
    monkeypatch.setattr("sys.argv", ["entrenar_experto.py", "--config",
                                     str(tmp_path / "no_existe.yaml")])
    assert script.main() == 2
    assert "error de configuración" in capsys.readouterr().err


# --- espacios y normalización en la evaluación --------------------------------------

class _EntornoEspacios(__import__("gymnasium").Env):
    """Entorno mínimo con los espacios que se le pasen."""

    def __init__(self, observacion, accion) -> None:
        self.observation_space, self.action_space = observacion, accion

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self.observation_space.sample(), {}

    def step(self, accion):
        return self.observation_space.sample(), 0.0, True, False, {}


@pytest.mark.parametrize("observacion,accion,mensaje", [
    ("box", "start", "start 0"),
    ("box", "multidiscreto", "Espacio de acción"),
    ("box", "box2d", "Espacio de acción"),
    ("box2d", "discreto", "Espacio de observación"),
])
def test_crear_trpo_rechaza_espacios_no_compatibles(observacion, accion, mensaje):
    from gymnasium import spaces
    observaciones = {"box": spaces.Box(-1, 1, (3,), np.float32),
                     "box2d": spaces.Box(-1, 1, (2, 2), np.float32)}
    acciones = {"start": spaces.Discrete(3, start=-1), "discreto": spaces.Discrete(2),
                "multidiscreto": spaces.MultiDiscrete([2, 2]),
                "box2d": spaces.Box(-1, 1, (2, 2), np.float32)}
    venv = DummyVecEnv([lambda: _EntornoEspacios(observaciones[observacion], acciones[accion])])
    try:
        with pytest.raises(ValueError, match=mensaje):
            crear_trpo(_cfg(), venv)
    finally:
        venv.close()


def test_evaluar_retorno_aplica_la_normalizacion():
    cfg = _cfg("Pendulum-v1")
    modelo, venv = _agente(cfg)
    venv.close()
    for capa in [m for m in modelo.policy.modules() if isinstance(m, torch.nn.Linear)]:
        torch.nn.init.normal_(capa.weight, std=1.0)  # salida sensible a la entrada
    desplazada = {"media": [5.0, 5.0, 5.0], "varianza": [1.0, 1.0, 1.0],
                  "epsilon": 1e-8, "limite": 10.0}
    sin = evaluar_retorno(modelo, cfg, 2, 4, True)
    con = evaluar_retorno(modelo, cfg, 2, 4, True, desplazada)
    identidad = {"media": [0.0] * 3, "varianza": [1.0 - 1e-8] * 3, "epsilon": 1e-8,
                 "limite": 1e6}
    assert con != sin
    assert evaluar_retorno(modelo, cfg, 2, 4, True, identidad) == pytest.approx(sin)


def test_sin_estandarizar_ventajas_se_admite_una_transicion():
    cfg = _cfg(pasos_por_iteracion=2, batch_size_valor=1, normalizar_ventajas=False)
    modelo, venv = _agente(cfg)
    try:
        assert iteracion_trpo(modelo)["pasos"] == 2
    finally:
        venv.close()


def test_comprobar_directorio_libre(tmp_path):
    from src.utils.archivos import comprobar_directorio_libre
    assert comprobar_directorio_libre(tmp_path / "nuevo") == tmp_path / "nuevo"
    (tmp_path / "vacio").mkdir()
    comprobar_directorio_libre(tmp_path / "vacio")
    (tmp_path / "lleno").mkdir()
    (tmp_path / "lleno" / "algo.txt").write_text("x", encoding="utf-8")
    (tmp_path / "archivo").write_text("x", encoding="utf-8")
    for ocupado in ("lleno", "archivo"):
        with pytest.raises(FileExistsError, match="no se sobrescribe"):
            comprobar_directorio_libre(tmp_path / ocupado)


@pytest.mark.parametrize("ocupado", ["carpeta", "archivo"])
def test_el_script_no_entrena_si_la_salida_esta_ocupada(ocupado, monkeypatch, tmp_path, capsys):
    # Regresión: entrenaba y fallaba después, al guardar, perdiendo el entrenamiento.
    script = _script()
    salida = tmp_path / "experto"
    if ocupado == "carpeta":
        salida.mkdir()
        (salida / "modelo.zip").write_text("x", encoding="utf-8")
    else:
        salida.write_text("x", encoding="utf-8")
    entrenado = []
    monkeypatch.setattr("src.training.flujo.entrenar_experto",
                        lambda *a: entrenado.append(1))
    monkeypatch.setattr("sys.argv",
                        ["entrenar_experto.py", "--sin-wandb", "--salida", str(salida)])
    assert script.main() == 2
    assert not entrenado
    assert "elige otra --salida" in capsys.readouterr().err


def test_la_semilla_del_experto_admite_enteros_de_numpy(tmp_path):
    # Regresión: experto.semilla tenía su propia comprobación y rechazaba np.int64,
    # que validar_semilla (la única del proyecto) acepta. También se debe poder
    # guardar: experto.json convierte los escalares de NumPy.
    con_python, con_numpy = _cfg(), _cfg()
    con_numpy["experto"]["semilla"] = np.int64(con_python["experto"]["semilla"])
    modelo, resumen = entrenar_experto(con_numpy)
    assert resumen["evaluaciones"] == entrenar_experto(con_python)[1]["evaluaciones"]
    assert type(resumen["semillas"]["base"]) is int
    guardar_experto(tmp_path / "experto", modelo, con_numpy, resumen)
    _, metadatos = cargar_experto(tmp_path / "experto")
    assert metadatos["configuracion"]["experto"]["semilla"] == con_python["experto"]["semilla"]


def test_evaluar_retorno_anota_lo_que_devuelve():
    import typing
    devuelve = typing.get_type_hints(evaluar_retorno)["return"]
    assert typing.get_args(devuelve)[1] is typing.Any




# --- política del experto y grabación de demostraciones ---------------------------

@ENTORNOS
@pytest.mark.parametrize("determinista", [True, False])
def test_politica_trpo_es_predict_sobre_la_observacion_normalizada(entorno, determinista):
    cfg = _cfg(entorno)
    modelo, venv = _agente(cfg)
    venv.close()
    estadisticas = {"media": [0.3] * modelo.observation_space.shape[0],
                    "varianza": [2.0] * modelo.observation_space.shape[0],
                    "epsilon": 1e-8, "limite": 10.0}
    observacion = modelo.observation_space.sample()
    for est in (None, estadisticas):
        politica = politica_trpo(modelo, determinista, est)
        with generador_torch_aislado(0):
            esperada, _ = modelo.predict(normalizar_observacion(observacion, est),
                                         deterministic=determinista)
        with generador_torch_aislado(0):
            np.testing.assert_array_equal(politica(observacion), esperada)


def test_politica_trpo_rechaza_lo_que_no_es_trpo_o_un_modo_no_booleano():
    modelo, venv = _agente(_cfg())
    venv.close()
    with pytest.raises(ValueError, match="TRPO"):
        politica_trpo(object(), True)
    for modo in (1, "no", None):
        with pytest.raises(ValueError, match="determinista"):
            politica_trpo(modelo, modo)


@ENTORNOS
def test_evaluar_retorno_juega_las_mismas_partidas_que_jugar_partida(entorno):
    cfg = _cfg(entorno)
    modelo, venv = _agente(cfg)
    venv.close()
    resultado = evaluar_retorno(modelo, cfg, 3, 9, False)
    entorno_ = crear_entorno(cfg, None)
    try:
        partidas = [jugar_partida(entorno_, politica_trpo(modelo, False), semilla, i)
                    for i, semilla in enumerate(semillas_independientes(9, 3))]
    finally:
        entorno_.close()
    assert resultado["retornos"] == [retorno_partida(p) for p in partidas]
    assert resultado["longitud_media"] == np.mean([p.numero_pasos for p in partidas])


@pytest.fixture(scope="module")
def experto_guardado(tmp_path_factory):
    """Experto de CartPole con un presupuesto mínimo, guardado como lo hace el script."""
    cfg = _cfg()
    modelo, resumen = entrenar_experto(cfg)
    directorio = tmp_path_factory.mktemp("experto") / "cartpole"
    guardar_experto(directorio, modelo, cfg, resumen)
    return directorio


def _reescribir_zip(origen: Path, destino: Path, cambios: dict[str, bytes],
                   fecha: tuple[int, ...] | None = None) -> None:
    """Copia un zip cambiando el contenido de algunas entradas y, si se pide, su fecha."""
    with zipfile.ZipFile(origen) as entrada, zipfile.ZipFile(destino, "w") as salida:
        for info in entrada.infolist():
            if fecha is not None:
                info.date_time = fecha
            salida.writestr(info, cambios.get(info.filename, entrada.read(info.filename)))


def test_huella_politica_es_la_de_policy_pth(experto_guardado):
    modelo = experto_guardado / "modelo.zip"
    with zipfile.ZipFile(modelo) as archivo:
        pesos = archivo.read("policy.pth")
    assert huella_politica(modelo) == hashlib.sha256(pesos).hexdigest()


def test_huella_politica_no_cambia_con_los_metadatos_del_zip(experto_guardado, tmp_path):
    """Regresión: la huella de modelo.zip cambiaba al reentrenar con los mismos pesos."""
    modelo = experto_guardado / "modelo.zip"
    copia = tmp_path / "modelo.zip"
    _reescribir_zip(modelo, copia, {"data": b"{}", "system_info.txt": b"otra"},
                    fecha=(2030, 1, 1, 0, 0, 0))
    assert huella_archivo(copia) != huella_archivo(modelo)
    assert huella_politica(copia) == huella_politica(modelo)


def test_huella_politica_cambia_si_cambian_los_pesos(experto_guardado, tmp_path):
    modelo = experto_guardado / "modelo.zip"
    with zipfile.ZipFile(modelo) as archivo:
        pesos = bytearray(archivo.read("policy.pth"))
    pesos[-1] ^= 1
    _reescribir_zip(modelo, tmp_path / "modelo.zip", {"policy.pth": bytes(pesos)})
    assert huella_politica(tmp_path / "modelo.zip") != huella_politica(modelo)


def test_reentrenar_con_la_misma_config_da_la_misma_huella_de_politica(tmp_path):
    cfg = _cfg()
    for nombre in ("primero", "segundo"):
        modelo, resumen = entrenar_experto(cfg)
        guardar_experto(tmp_path / nombre, modelo, cfg, resumen)
    assert (huella_politica(tmp_path / "primero" / "modelo.zip")
            == huella_politica(tmp_path / "segundo" / "modelo.zip"))


def test_huella_politica_rechaza_archivos_que_no_son_un_modelo(tmp_path):
    no_zip = tmp_path / "no_zip.zip"
    no_zip.write_bytes(b"modelo de prueba")
    sin_politica = tmp_path / "sin_politica.zip"
    with zipfile.ZipFile(sin_politica, "w") as archivo:
        archivo.writestr("data", "{}")
    for ruta in (no_zip, sin_politica):
        with pytest.raises(ValueError, match="policy.pth"):
            huella_politica(ruta)
    with pytest.raises(FileNotFoundError):
        huella_politica(tmp_path / "falta.zip")


def _cfg_grabacion(**cambios) -> dict:
    cfg = _cfg()
    cfg["demostraciones"].update(partidas_train=4, partidas_validacion=2, subconjuntos=[1, 4])
    cfg["demostraciones"].update(cambios)
    return cfg


@pytest.mark.parametrize("determinista", [True, False])
def test_cargar_politica_experto_juega_como_el_experto(experto_guardado, determinista):
    cfg = _cfg_grabacion(determinista=determinista)
    politica, _ = cargar_politica_experto(experto_guardado, cfg)
    modelo, _ = cargar_experto(experto_guardado)
    entorno = crear_entorno(cfg, None)
    try:
        cargada = jugar_partida(entorno, politica, 4, 0)
        directa = jugar_partida(entorno, politica_trpo(modelo, determinista), 4, 0)
    finally:
        entorno.close()
    assert huella_partidas([cargada]) == huella_partidas([directa])


def test_cargar_politica_experto_describe_su_procedencia(experto_guardado):
    cfg = _cfg_grabacion()
    _, metadatos = cargar_politica_experto(experto_guardado, cfg)
    _, guardado = cargar_experto(experto_guardado)
    assert set(metadatos) == {"entorno", "experto", "procedencia", "configuracion_experto"}
    assert metadatos["entorno"] == "CartPole-v1"
    experto = metadatos["experto"]
    assert experto["huella_modelo"] == huella_archivo(experto_guardado / "modelo.zip")
    assert experto["huella_metadatos"] == huella_archivo(experto_guardado / "experto.json")
    resumen = guardado["resumen"]
    claves = ("mejor_retorno", "mejor_iteracion", "pasos", "objetivo_alcanzado")
    assert [experto[clave] for clave in claves] == [resumen[clave] for clave in claves]
    assert experto["codigo"] == guardado["codigo"]
    # Sin rutas absolutas: fuera de la carpeta de trabajo solo queda el nombre.
    assert experto["directorio"] == "cartpole"
    assert set(metadatos["procedencia"]) == {"generador", "codigo", "entorno_ejecucion",
                                             "configuracion_demostraciones"}
    assert metadatos["procedencia"]["configuracion_demostraciones"] == cfg["demostraciones"]
    assert metadatos["configuracion_experto"] == guardado["configuracion"]
    json.dumps(metadatos, allow_nan=False)


def test_la_ruta_del_experto_se_guarda_relativa_a_la_carpeta_de_trabajo(
        experto_guardado, monkeypatch):
    monkeypatch.chdir(experto_guardado.parent.parent)
    _, metadatos = cargar_politica_experto(experto_guardado, _cfg_grabacion())
    assert metadatos["experto"]["directorio"] == f"{experto_guardado.parent.name}/cartpole"


def test_regenerar_con_el_experto_da_el_mismo_archivo(experto_guardado, tmp_path):
    # Regresión de diseño: con la fecha en los metadatos, cada regeneración daba
    # otro NPZ y otra huella aunque las partidas fueran idénticas.
    cfg = _cfg_grabacion()
    huellas = []
    for nombre in ("a", "b"):
        politica, metadatos = cargar_politica_experto(experto_guardado, cfg)
        huellas.append(generar_dataset(cfg, politica, tmp_path / nombre, metadatos)
                       ["huella_dataset"])
    assert huellas[0] == huellas[1]


def test_cargar_politica_experto_rechaza_otro_entorno_y_carpetas_sin_experto(
        experto_guardado, tmp_path):
    with pytest.raises(ValueError, match="se entrenó en 'CartPole-v1'"):
        cargar_politica_experto(experto_guardado, {**_cfg_grabacion(),
                                                   "entorno": {"id": "Pendulum-v1"}})
    with pytest.raises(FileNotFoundError, match="modelo.zip"):
        cargar_politica_experto(tmp_path, _cfg_grabacion())
    with pytest.raises(ValueError, match="demostraciones.determinista"):
        cargar_politica_experto(experto_guardado, _cfg_grabacion(determinista="no"))


def test_la_politica_del_experto_aplica_su_normalizacion(tmp_path):
    cfg = _cfg("Pendulum-v1", normalizar_observaciones=True)
    cfg["experto"].update(pasos_totales=128)
    modelo, resumen = entrenar_experto(cfg)
    assert resumen["normalizacion"] is not None
    guardar_experto(tmp_path / "experto", modelo, cfg, resumen)
    cfg["demostraciones"]["determinista"] = True
    politica, _ = cargar_politica_experto(tmp_path / "experto", cfg)
    observacion = np.array([0.5, -0.5, 3.0], dtype=np.float32)
    con, _ = modelo.predict(normalizar_observacion(observacion, resumen["normalizacion"]),
                            deterministic=True)
    sin, _ = modelo.predict(observacion, deterministic=True)
    np.testing.assert_array_equal(politica(observacion), con)
    assert not np.array_equal(con, sin)


# --- scripts/get_data.py ----------------------------------------------------------

def _script_datos():
    """Carga scripts/get_data.py como módulo."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("get_data", RAIZ / "scripts" / "get_data.py")
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def test_get_data_graba_con_el_experto_y_verifica_la_huella(experto_guardado, tmp_path,
                                                            monkeypatch, capsys):
    script = _script_datos()
    resumenes = []
    monkeypatch.setattr("src.training.flujo.registrar_resumen",
                        lambda run, resumen: resumenes.append(resumen))
    argumentos = ["get_data.py", "--sin-wandb", "--experto", str(experto_guardado)]
    monkeypatch.setattr("sys.argv", [*argumentos, "--salida", str(tmp_path / "a")])
    assert script.main() == 0
    salida = capsys.readouterr().out
    huella = re.search(r"huella de las partidas: ([0-9a-f]{64})", salida).group(1)
    assert "15 partidas" in salida  # 10 + 5 del perfil de CartPole
    npz = tmp_path / "a" / "demostraciones.npz"
    assert npz.is_file() and (tmp_path / "a" / "splits.json").is_file()
    assert f"huella de demostraciones.npz: {huella_archivo(npz)}" in salida
    assert resumenes[0]["semillas"].keys() == {"base", "partidas", "splits"}
    assert all(isinstance(semilla, int) for semilla in resumenes[0]["semillas"].values())
    assert resumenes[0]["segundos"] >= 0

    monkeypatch.setattr("sys.argv", [*argumentos, "--salida", str(tmp_path / "b"),
                                     "--huella-esperada", huella])
    assert script.main() == 0
    assert "coinciden con las congeladas" in capsys.readouterr().out

    monkeypatch.setattr("sys.argv", [*argumentos, "--salida", str(tmp_path / "c"),
                                     "--huella-esperada", "0" * 64])
    assert script.main() == 1
    salida = capsys.readouterr()
    assert "no coincide" in salida.err
    assert "huella de las partidas:" in salida.out
    assert "huella de demostraciones.npz:" in salida.out
    assert "huella de splits.json:" in salida.out
    assert (tmp_path / "c" / "demostraciones.npz").is_file()  # se conserva para compararlo


def test_get_data_busca_el_artefacto_del_experto_por_huella(
    experto_guardado, tmp_path, monkeypatch
):
    """El run de datos encuentra el experto aunque se use otro grupo W&B."""
    script = _script_datos()
    _, metadatos = cargar_politica_experto(
        experto_guardado, resolver_configuracion(RAIZ / "configs" / "default.yaml", "cartpole")
    )
    nombre_esperado = (f"experto-CartPole-v1-{metadatos['experto']['huella_modelo'][:12]}:latest")
    class CommError(Exception):
        pass

    class Run:
        def __init__(self):
            self.usados = []

        def use_artifact(self, nombre, type):
            self.usados.append((nombre, type))

    run = Run()
    nombres = []
    monkeypatch.setattr(script, "comprobar_wandb", lambda *a, **k: object())
    monkeypatch.setattr("src.training.flujo.abrir_run",
                        lambda *a, **k: nombres.append(k["nombre"]) or run)
    monkeypatch.setattr("src.training.flujo.registrar_resumen", lambda *a: None)
    subidas = []
    monkeypatch.setattr("src.training.flujo.subir_artefacto", lambda *a: subidas.append(a))
    monkeypatch.setattr("src.training.flujo.terminar_run", lambda *a: None)
    monkeypatch.setattr("src.training.flujo.avisar_estado_git", lambda *a: None)
    monkeypatch.setattr("src.utils.tracking.comprobar_wandb",
                        lambda cfg: SimpleNamespace(errors=SimpleNamespace(CommError=CommError)))
    monkeypatch.setattr("src.training.flujo.generar_dataset", lambda cfg, politica, salida, meta: {
        "pasos": 1, "semillas": {"base": 1, "partidas": 2, "splits": 3},
        "retornos": [1.0], "huella_dataset": "a" * 64, "huella_splits": "b" * 64,
        "huella_partidas": "c" * 64, "demostraciones": Path("demostraciones.npz"),
        "splits": Path("splits.json"), "train": [0], "validacion": [],
    })
    salida = tmp_path / "datos"
    monkeypatch.setattr("sys.argv", ["get_data.py", "--experto", str(experto_guardado),
                                     "--salida", str(salida), "--grupo", "grupo-distinto"])
    assert script.main() == 0
    assert run.usados == [(nombre_esperado, "experto")]
    assert subidas[0][1:3] == ("demostraciones-aaaaaaaaaaaa", "demostraciones")
    assert nombres == ["tiiac-datos-cartpole"]


@pytest.mark.parametrize("caso,mensaje", [
    ("salida ocupada", "elige otra --salida"),
    ("sin experto", "error con el experto"),
    ("config", "error de configuración"),
])
def test_get_data_no_graba_si_algo_no_es_valido(caso, mensaje, experto_guardado, tmp_path,
                                                 monkeypatch, capsys):
    script = _script_datos()
    salida = tmp_path / "datos"
    argumentos = ["get_data.py", "--sin-wandb", "--experto", str(experto_guardado),
                  "--salida", str(salida)]
    if caso == "salida ocupada":
        salida.mkdir()
        (salida / "algo.txt").write_text("x", encoding="utf-8")
    elif caso == "sin experto":
        argumentos[3] = str(tmp_path / "no-existe")
    else:
        argumentos += ["--config", str(tmp_path / "no-existe.yaml")]
    grabado = []
    monkeypatch.setattr("src.training.flujo.generar_dataset", lambda *a: grabado.append(1))
    monkeypatch.setattr("sys.argv", argumentos)
    assert script.main() == 2
    assert not grabado
    assert mensaje in capsys.readouterr().err


@pytest.mark.parametrize("codigo,aviso", [
    ({"commit": "a" * 40, "cambios_sin_commit": False}, None),
    ({"commit": "a" * 40, "cambios_sin_commit": True}, "cambios sin commit"),
    ({"commit": None, "cambios_sin_commit": None}, "no se pudo leer el commit"),
])
def test_get_data_avisa_del_estado_de_git(codigo, aviso, monkeypatch, tmp_path, capsys):
    script = _script_datos()
    metadatos = {"entorno": "CartPole-v1", "experto": {}, "configuracion_experto": {},
                 "procedencia": {"codigo": codigo}}
    monkeypatch.setattr("src.training.flujo.cargar_politica_experto",
                        lambda directorio, cfg: (lambda o: int(o[2] > 0), metadatos))
    monkeypatch.setattr("sys.argv", ["get_data.py", "--sin-wandb", "--experto", "x",
                                     "--salida", str(tmp_path / "datos")])
    assert script.main() == 0
    error = capsys.readouterr().err
    assert (aviso in error) if aviso else "aviso" not in error


def test_bc_aprende_de_las_demostraciones_grabadas_con_el_experto(experto_guardado, tmp_path):
    """Integración de punta a punta: experto guardado → demostraciones y splits en
    disco → carga con comprobación de huella → BC con cada subconjunto."""
    from src.data.demostraciones import cargar_demostraciones
    from src.data.splits import cargar_splits
    from src.training.bc import entrenar_bc_subconjunto
    from src.utils.seed import crear_generador

    cfg = _cfg_grabacion()
    cfg["bc"].update(epocas_maximas=3, paciencia=1)
    politica, metadatos = cargar_politica_experto(experto_guardado, cfg)
    resultado = generar_dataset(cfg, politica, tmp_path / "datos", metadatos)
    entorno = crear_entorno(cfg, None)
    espacio_observacion, espacio_accion = entorno.observation_space, entorno.action_space
    try:
        trayectorias, guardados = cargar_demostraciones(
            resultado["demostraciones"], espacio_observacion, espacio_accion)
        splits = cargar_splits(resultado["splits"], huella_archivo(resultado["demostraciones"]),
                               [t.id_episodio for t in trayectorias])
        assert guardados["grabacion"]["huella_partidas"] == resultado["huella_partidas"]
        resumenes = []
        for cantidad in cfg["demostraciones"]["subconjuntos"]:
            fijar_seed(0)
            politica_bc, resumen = entrenar_bc_subconjunto(
                trayectorias, splits, cantidad, espacio_observacion, espacio_accion, cfg,
                crear_generador(0))
            resumenes.append(resumen)
            assert math.isfinite(resumen["val_perdida"]) and 1 <= resumen["mejor_epoca"] <= 3

            def actuar(observacion, red=politica_bc):
                with torch.no_grad():
                    entrada = torch.as_tensor(observacion, dtype=torch.float32).unsqueeze(0)
                    return int(red.eval()(entrada).argmax(dim=1))
            partida = jugar_partida(entorno, actuar, 7, 0)
            assert partida.numero_pasos >= 1
    finally:
        entorno.close()
    # Más partidas de train dan más pares; la validación es la misma para todos.
    assert resumenes[0]["pares_train"] < resumenes[1]["pares_train"]
    assert len({r["pares_validacion"] for r in resumenes}) == 1
