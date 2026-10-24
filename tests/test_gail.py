"""Pruebas de la recompensa de GAIL: discriminador, recompensas, muestreo de
expertos, actualización y envoltorio del entorno.

Como en BC, los pares son inventados y no hace falta experto. Se cubren acciones
Discrete (con start distinto de 0) y Box de una dimensión. Los entornos son
inventados y deterministas, para comprobar exactamente qué par se guarda en cada
paso; CartPole y TRPO solo se usan para comprobar que todo encaja con sb3.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
import yaml
from gymnasium import spaces
from sb3_contrib import TRPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.data.entornos import crear_entorno_vectorizado
from src.data.tipos import LoteTransiciones
from src.models.factory import Discriminador, construir_discriminador
from src.training.gail import (MODOS_RECOMPENSA, EntornoRecompensaGAIL,
                               actualizar_discriminador, calcular_recompensas_gail,
                               muestrear_expertos, preparar_discriminador)
from src.utils.configuracion import resolver_configuracion
from src.utils.seed import crear_generador, fijar_seed

RAIZ = Path(__file__).resolve().parent.parent
OBSERVACION = spaces.Box(-10.0, 10.0, shape=(3,), dtype=np.float32)
DISCRETO = spaces.Discrete(3, start=-1)
CONTINUO = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
TIPOS = pytest.mark.parametrize("continuo", [False, True], ids=["discreto", "continuo"])


def _espacio(continuo: bool) -> spaces.Space:
    return CONTINUO if continuo else DISCRETO


def _cfg(**cambios) -> dict:
    """Configuración del proyecto para CartPole con cambios en gail.discriminador."""
    cfg = resolver_configuracion(RAIZ / "configs" / "default.yaml", "cartpole")
    cfg["gail"]["discriminador"]["capas_ocultas"] = [8]
    cfg["gail"]["discriminador"].update(cambios)
    return cfg


def _discriminador(continuo: bool, capas=(8,)) -> Discriminador:
    return Discriminador(3, 2 if continuo else 3, continuo, list(capas), "tanh")


def _con_pesos_aleatorios(discriminador: Discriminador, semilla: int = 0) -> Discriminador:
    """Da pesos no nulos a la capa de salida, que empieza a cero."""
    generador = crear_generador(semilla)
    salida = [m for m in discriminador.red if isinstance(m, torch.nn.Linear)][-1]
    with torch.no_grad():
        for parametro in (salida.weight, salida.bias):
            parametro.copy_(torch.tensor(generador.normal(size=tuple(parametro.shape))))
    return discriminador


def _lote(n: int, continuo: bool, semilla: int = 0, centro: float = 0.0) -> LoteTransiciones:
    """Pares válidos: estados alrededor de centro y acciones del espacio del entorno."""
    generador = crear_generador(semilla)
    estados = (centro + generador.normal(size=(n, 3))).astype(np.float32)
    if continuo:
        acciones = generador.uniform(-1, 1, size=(n, 2)).astype(np.float32)
    else:
        acciones = generador.integers(-1, 2, size=n)
    return LoteTransiciones(estados, acciones, np.arange(n, dtype=np.int64) // 4,
                            np.arange(n, dtype=np.int64) % 4)


def _tensores(lote: LoteTransiciones, continuo: bool) -> tuple[torch.Tensor, torch.Tensor]:
    estados = torch.tensor(lote.estados)
    if continuo:
        return estados, torch.tensor(lote.acciones)
    return estados, torch.tensor(lote.acciones, dtype=torch.int64) + 1


# --- Discriminador -----------------------------------------------------------
# Arquitecturas: sin capas ocultas, la del paper reducida y dos capas distintas.

@TIPOS
@pytest.mark.parametrize("capas", [(), (8,), (8, 4)])
@pytest.mark.parametrize("batch", [1, 5])
def test_discriminador_da_un_logit_por_par(continuo, capas, batch):
    discriminador = _discriminador(continuo, capas)
    logits = discriminador(*_tensores(_lote(batch, continuo), continuo))
    assert logits.shape == (batch,)
    assert logits.dtype == torch.float32


@TIPOS
def test_discriminador_ve_lo_que_ve_el_original(continuo):
    discriminador = _discriminador(continuo, (8, 4))
    lineales = [m for m in discriminador.red if isinstance(m, torch.nn.Linear)]
    # Discretas: solo el estado y una salida por acción. Continuas: estado y
    # acción concatenados y una sola salida.
    assert lineales[0].in_features == (5 if continuo else 3)
    assert lineales[-1].out_features == (1 if continuo else 3)
    assert [m.out_features for m in lineales[:-1]] == [8, 4]
    assert sum(isinstance(m, torch.nn.Tanh) for m in discriminador.red) == 2


@TIPOS
def test_discriminador_se_inicializa_como_el_original(continuo):
    discriminador = _discriminador(continuo, (8, 4))
    lineales = [m for m in discriminador.red if isinstance(m, torch.nn.Linear)]
    assert torch.all(lineales[-1].weight == 0)
    assert all(torch.all(m.bias == 0) for m in lineales)
    assert all(m.weight.abs().sum() > 0 for m in lineales[:-1])
    # Con la salida a cero, D = 0.5 para cualquier par.
    logits = discriminador(*_tensores(_lote(6, continuo), continuo))
    assert torch.all(logits == 0)


def test_discriminador_discreto_elige_la_salida_de_cada_accion():
    discriminador = _discriminador(False, ())
    with torch.no_grad():
        discriminador.red[0].bias.copy_(torch.tensor([10.0, 20.0, 30.0]))
    logits = discriminador(torch.zeros(4, 3), torch.tensor([0, 2, 1, 2]))
    assert torch.equal(logits, torch.tensor([10.0, 30.0, 20.0, 30.0]))


def test_discriminador_continuo_depende_del_estado_y_de_la_accion():
    discriminador = _con_pesos_aleatorios(_discriminador(True))
    estados = torch.zeros(1, 3, requires_grad=True)
    acciones = torch.zeros(1, 2, requires_grad=True)
    discriminador(estados, acciones).sum().backward()
    assert estados.grad.abs().sum() > 0 and acciones.grad.abs().sum() > 0


@TIPOS
def test_discriminador_no_mezcla_el_batch(continuo):
    discriminador = _con_pesos_aleatorios(_discriminador(continuo)).eval()
    estados, acciones = _tensores(_lote(6, continuo), continuo)
    juntos = discriminador(estados, acciones)
    separados = torch.cat([discriminador(estados[i:i + 1], acciones[i:i + 1]) for i in range(6)])
    torch.testing.assert_close(juntos, separados)


@TIPOS
def test_los_gradientes_llegan_a_todos_los_parametros(continuo):
    discriminador = _con_pesos_aleatorios(_discriminador(continuo, (8, 4)))
    discriminador(*_tensores(_lote(6, continuo), continuo)).sum().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in discriminador.parameters())


@TIPOS
def test_la_misma_semilla_da_el_mismo_discriminador(continuo):
    fijar_seed(5)
    a = _discriminador(continuo)
    fijar_seed(5)
    b = _discriminador(continuo)
    assert all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))


@TIPOS
def test_normalizacion_del_discriminador(continuo):
    discriminador = _con_pesos_aleatorios(_discriminador(continuo))
    estados, acciones = _tensores(_lote(20, continuo, centro=4.0), continuo)
    antes = discriminador(estados, acciones)
    assert torch.equal(discriminador.media, torch.zeros(5 if continuo else 3))
    discriminador.ajustar_normalizacion(estados, acciones, 1e-6)

    # Con Box se normalizan estado y acción; con Discrete, solo el estado.
    entrada = torch.cat([estados, acciones], dim=1) if continuo else estados
    datos = entrada.double()
    torch.testing.assert_close(discriminador.media, datos.mean(0).float())
    torch.testing.assert_close(discriminador.escala, datos.std(0, unbiased=False).float())
    normalizada = (entrada - discriminador.media) / discriminador.escala
    esperado = discriminador.red(normalizada)
    esperado = esperado[:, 0] if continuo else esperado.gather(1, acciones.unsqueeze(1))[:, 0]
    torch.testing.assert_close(discriminador(estados, acciones), esperado)
    assert not torch.allclose(antes, discriminador(estados, acciones))
    assert {"media", "escala"} <= discriminador.state_dict().keys()
    assert all(p is not discriminador.media for p in discriminador.parameters())


@pytest.mark.parametrize("argumentos,mensaje", [
    ((0, 2, False, [8], "tanh"), "dim_observacion"),
    ((3, 0, False, [8], "tanh"), "dim_accion"),
    ((3, 2.0, True, [8], "tanh"), "dim_accion"),
    ((3, 2, 1, [8], "tanh"), "continua"),
    ((3, 2, False, [0], "tanh"), "capas_ocultas"),
    ((3, 2, False, "8", "tanh"), "capas_ocultas"),
    ((3, 2, False, [8], "sigmoide"), "activacion"),
])
def test_discriminador_rechaza_argumentos_no_validos(argumentos, mensaje):
    with pytest.raises(ValueError, match=mensaje):
        Discriminador(*argumentos)


@pytest.mark.parametrize("continuo,estados,acciones,mensaje", [
    (False, torch.zeros(2, 4), torch.zeros(2, dtype=torch.int64), r"estados \[B, 3\]"),
    (False, torch.zeros(3), torch.zeros(3, dtype=torch.int64), r"estados \[B, 3\]"),
    (False, torch.zeros(2, 3), torch.zeros(2), "enteras"),
    (False, torch.zeros(2, 3), torch.zeros(3, dtype=torch.int64), "enteras"),
    (False, torch.zeros(2, 3), torch.tensor([0, 3]), r"\[0, 3\)"),
    (False, torch.zeros(2, 3), torch.tensor([-1, 0]), r"\[0, 3\)"),
    (True, torch.zeros(2, 3), torch.zeros(2, 2, dtype=torch.int64), "reales"),
    (True, torch.zeros(2, 3), torch.zeros(2, 3), "reales"),
    (True, torch.zeros(2, 3), torch.zeros(2), "reales"),
])
def test_discriminador_rechaza_pares_no_validos(continuo, estados, acciones, mensaje):
    discriminador = _discriminador(continuo)
    with pytest.raises(ValueError, match=mensaje):
        discriminador(estados, acciones)
    with pytest.raises(ValueError, match=mensaje):
        discriminador.ajustar_normalizacion(estados, acciones, 1e-6)


@TIPOS
def test_ajustar_normalizacion_rechaza_datos_no_validos(continuo):
    discriminador = _discriminador(continuo)
    estados, acciones = _tensores(_lote(4, continuo), continuo)
    with pytest.raises(ValueError, match="tensores"):
        discriminador.ajustar_normalizacion(estados.numpy(), acciones, 1e-6)
    with pytest.raises(ValueError, match="epsilon"):
        discriminador.ajustar_normalizacion(estados, acciones, 0.0)
    estados[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finitos"):
        discriminador.ajustar_normalizacion(estados, acciones, 1e-6)
    with pytest.raises(ValueError, match="N > 0"):
        discriminador.ajustar_normalizacion(estados[:0], acciones[:0], 1e-6)


@pytest.mark.parametrize("entorno", ["CartPole-v1", "Pendulum-v1"])
def test_construye_el_discriminador_del_paper_con_los_espacios_reales(entorno):
    env = gym.make(entorno)
    try:
        cfg = resolver_configuracion(RAIZ / "configs" / "default.yaml", "cartpole")
        discriminador = construir_discriminador(cfg, env.observation_space, env.action_space)
        lineales = [m for m in discriminador.red if isinstance(m, torch.nn.Linear)]
        assert [m.out_features for m in lineales[:-1]] == [100, 100]
        assert sum(isinstance(m, torch.nn.Tanh) for m in discriminador.red) == 2
        continuo = isinstance(env.action_space, spaces.Box)
        assert discriminador.continua == continuo
        assert discriminador.dim_observacion == env.observation_space.shape[0]
        assert discriminador.dim_accion == (env.action_space.shape[0] if continuo
                                            else env.action_space.n)
    finally:
        env.close()


@pytest.mark.parametrize("observacion,accion,mensaje", [
    (spaces.Discrete(4), DISCRETO, "observación"),
    (spaces.Box(-1, 1, (2, 2)), DISCRETO, "observación"),
    (OBSERVACION, spaces.MultiDiscrete([2, 2]), "acción"),
    (OBSERVACION, spaces.Box(-1, 1, (2, 2)), "acción"),
])
def test_construir_discriminador_rechaza_espacios_no_compatibles(observacion, accion, mensaje):
    with pytest.raises(ValueError, match=mensaje):
        construir_discriminador(_cfg(), observacion, accion)


@pytest.mark.parametrize("cfg", [
    {}, {"gail": {}}, {"gail": {"discriminador": {"activacion": "tanh"}}},
    {"gail": {"discriminador": {"capas_ocultas": [8]}}}, None,
])
def test_construir_discriminador_exige_su_configuracion(cfg):
    with pytest.raises(ValueError, match="gail.discriminador"):
        construir_discriminador(cfg, OBSERVACION, DISCRETO)


# --- calcular_recompensas_gail -----------------------------------------------

def _discriminador_con_logits(logits: list[float]) -> Discriminador:
    """Discriminador discreto lineal cuyo logit para la acción i es logits[i]."""
    discriminador = Discriminador(3, len(logits), False, [], "tanh")
    with torch.no_grad():
        discriminador.red[0].bias.copy_(torch.tensor(logits))
    return discriminador


def test_las_recompensas_siguen_las_formulas_del_original():
    logits = [-3.0, -0.5, 0.0, 0.7, 4.0]
    discriminador = _discriminador_con_logits(logits)
    espacio = spaces.Discrete(5)
    estados, acciones = np.zeros((5, 3), np.float32), np.arange(5)
    d = 1 / (1 + np.exp(-np.array(logits)))
    menos_log = calcular_recompensas_gail(discriminador, estados, acciones, espacio,
                                          "menos_log_uno_menos_d")
    log_d = calcular_recompensas_gail(discriminador, estados, acciones, espacio, "log_d")
    np.testing.assert_allclose(menos_log, -np.log(1 - d), rtol=1e-5)
    np.testing.assert_allclose(log_d, np.log(d), rtol=1e-5)
    assert menos_log.dtype == log_d.dtype == np.float32 and menos_log.shape == (5,)


def test_las_recompensas_son_finitas_con_logits_extremos():
    discriminador = _discriminador_con_logits([-80.0, 80.0])
    espacio = spaces.Discrete(2)
    estados, acciones = np.zeros((2, 3), np.float32), np.array([0, 1])
    menos_log = calcular_recompensas_gail(discriminador, estados, acciones, espacio,
                                          "menos_log_uno_menos_d")
    log_d = calcular_recompensas_gail(discriminador, estados, acciones, espacio, "log_d")
    assert np.all(np.isfinite(menos_log)) and np.all(np.isfinite(log_d))
    np.testing.assert_allclose(menos_log, [0.0, 80.0], atol=1e-6)
    np.testing.assert_allclose(log_d, [-80.0, 0.0], atol=1e-6)


@TIPOS
def test_signo_de_cada_modo_y_valor_inicial(continuo):
    lote = _lote(30, continuo)
    inicial = _discriminador(continuo)
    for modo, valor in [("menos_log_uno_menos_d", math.log(2)), ("log_d", -math.log(2))]:
        np.testing.assert_allclose(calcular_recompensas_gail(
            inicial, lote.estados, lote.acciones, _espacio(continuo), modo), valor, rtol=1e-6)
    entrenado = _con_pesos_aleatorios(_discriminador(continuo))
    assert np.all(calcular_recompensas_gail(entrenado, lote.estados, lote.acciones,
                                            _espacio(continuo), "menos_log_uno_menos_d") >= 0)
    assert np.all(calcular_recompensas_gail(entrenado, lote.estados, lote.acciones,
                                            _espacio(continuo), "log_d") <= 0)


def test_las_recompensas_traducen_el_start_del_espacio_discreto():
    discriminador = _discriminador_con_logits([1.0, 2.0, 3.0])
    recompensas = calcular_recompensas_gail(discriminador, np.zeros((3, 3), np.float32),
                                            np.array([1, -1, 0]), DISCRETO, "log_d")
    np.testing.assert_allclose(recompensas, np.log(1 / (1 + np.exp(-np.array([3.0, 1.0, 2.0])))),
                               rtol=1e-6)


@TIPOS
def test_las_recompensas_se_calculan_en_evaluacion_y_sin_gradientes(continuo):
    discriminador = _con_pesos_aleatorios(_discriminador(continuo)).train()
    lote = _lote(4, continuo)
    calcular_recompensas_gail(discriminador, lote.estados, lote.acciones, _espacio(continuo),
                              "menos_log_uno_menos_d")
    assert not discriminador.training
    assert all(p.grad is None for p in discriminador.parameters())


def test_calcular_recompensas_rechaza_modo_y_pares_no_validos():
    discriminador = _discriminador(False)
    estados = np.zeros((2, 3), np.float32)
    with pytest.raises(ValueError, match="Modo de recompensa desconocido"):
        calcular_recompensas_gail(discriminador, estados, np.array([0, 1]), DISCRETO, "log")
    with pytest.raises(ValueError, match="fuera del espacio Discrete"):
        calcular_recompensas_gail(discriminador, estados, np.array([0, 2]), DISCRETO, "log_d")


# --- muestrear_expertos ------------------------------------------------------

def test_muestrear_expertos_elige_pares_completos_con_reemplazo():
    pares = _lote(3, False)
    muestra = muestrear_expertos(pares, 50, crear_generador(0))
    assert muestra.numero_transiciones == 50
    # Cada par elegido conserva su estado, acción, id y paso.
    for i in range(50):
        origen = [j for j in range(3) if np.array_equal(pares.estados[j], muestra.estados[i])]
        assert len(origen) == 1
        j = origen[0]
        assert muestra.acciones[i] == pares.acciones[j]
        assert muestra.ids_episodio[i] == pares.ids_episodio[j]
        assert muestra.pasos[i] == pares.pasos[j]
    assert len({tuple(fila) for fila in muestra.estados}) == 3


def test_muestrear_expertos_es_reproducible_y_solo_usa_su_generador():
    pares = _lote(20, True)
    estado_numpy, estado_torch = np.random.get_state(), torch.get_rng_state()
    a = muestrear_expertos(pares, 30, crear_generador(1))
    b = muestrear_expertos(pares, 30, crear_generador(1))
    c = muestrear_expertos(pares, 30, crear_generador(2))
    assert np.array_equal(a.estados, b.estados) and np.array_equal(a.pasos, b.pasos)
    assert not np.array_equal(a.estados, c.estados)
    assert np.array_equal(np.random.get_state()[1], estado_numpy[1])
    assert torch.equal(torch.get_rng_state(), estado_torch)


def test_muestrear_expertos_devuelve_copias():
    pares = _lote(5, False)
    original = pares.estados.copy()
    muestrear_expertos(pares, 5, crear_generador(0)).estados[:] = 99
    assert np.array_equal(pares.estados, original)


@pytest.mark.parametrize("pares,cantidad,generador,mensaje", [
    ("lote", 0, "gen", "cantidad"),
    ("lote", -2, "gen", "cantidad"),
    ("lote", 1.5, "gen", "cantidad"),
    ("lote", True, "gen", "cantidad"),
    ("lote", 3, np.random, "Generator"),
    ("lote", 3, None, "Generator"),
    ("vacio", 3, "gen", "No hay pares"),
    ("incoherente", 3, "gen", "longitudes"),
    ((np.zeros((2, 3)), np.zeros(2)), 3, "gen", "LoteTransiciones"),
])
def test_muestrear_expertos_rechaza_argumentos_no_validos(pares, cantidad, generador, mensaje):
    lote = _lote(4, False)
    pares = {
        "lote": lote,
        "vacio": LoteTransiciones(lote.estados[:0], lote.acciones[:0], lote.ids_episodio[:0],
                                  lote.pasos[:0]),
        "incoherente": LoteTransiciones(lote.estados, lote.acciones[:3], lote.ids_episodio,
                                        lote.pasos),
    }.get(pares, pares) if isinstance(pares, str) else pares
    generador = crear_generador(0) if generador == "gen" else generador
    with pytest.raises(ValueError, match=mensaje):
        muestrear_expertos(pares, cantidad, generador)


# --- preparar_discriminador ----------------------------------------------------

@TIPOS
def test_preparar_normaliza_solo_con_los_pares_expertos(continuo):
    expertos = _lote(30, continuo, centro=5.0)
    discriminador, _ = preparar_discriminador(_cfg(), OBSERVACION, _espacio(continuo), expertos)
    estados, acciones = _tensores(expertos, continuo)
    entrada = torch.cat([estados, acciones], dim=1) if continuo else estados
    torch.testing.assert_close(discriminador.media, entrada.double().mean(0).float())

    sin_normalizar, _ = preparar_discriminador(_cfg(normalizar_entrada=False), OBSERVACION,
                                               _espacio(continuo), expertos)
    assert torch.all(sin_normalizar.media == 0) and torch.all(sin_normalizar.escala == 1)


def test_preparar_crea_adam_con_el_learning_rate_de_la_config():
    discriminador, optimizador = preparar_discriminador(_cfg(learning_rate=0.05), OBSERVACION,
                                                        DISCRETO, _lote(5, False))
    assert isinstance(optimizador, torch.optim.Adam)
    assert optimizador.param_groups[0]["lr"] == 0.05
    optimizados = {id(p) for grupo in optimizador.param_groups for p in grupo["params"]}
    assert optimizados == {id(p) for p in discriminador.parameters()}


def test_preparar_da_discriminadores_nuevos_e_independientes():
    expertos = _lote(5, False)
    fijar_seed(0)
    a, _ = preparar_discriminador(_cfg(), OBSERVACION, DISCRETO, expertos)
    b, _ = preparar_discriminador(_cfg(), OBSERVACION, DISCRETO, expertos)
    assert a is not b
    assert any(not torch.equal(x, y) for x, y in zip(a.parameters(), b.parameters()))


@pytest.mark.parametrize("clave,valor,mensaje", [
    ("optimizador", "sgd", "optimizador desconocido"),
    ("learning_rate", 0, "learning_rate"),
    ("learning_rate", -0.1, "learning_rate"),
    ("learning_rate", float("nan"), "learning_rate"),
    ("learning_rate", True, "learning_rate"),
    ("learning_rate", "0.01", "learning_rate"),
    ("pasos_adam", 0, "pasos_adam"),
    ("pasos_adam", 1.0, "pasos_adam"),
    ("pasos_adam", True, "pasos_adam"),
    ("peso_entropia", -0.001, "peso_entropia"),
    ("peso_entropia", float("inf"), "peso_entropia"),
    ("normalizar_entrada", 1, "normalizar_entrada"),
    ("epsilon_normalizacion", 0.0, "epsilon_normalizacion"),
])
def test_la_configuracion_del_discriminador_se_comprueba(clave, valor, mensaje):
    cfg = _cfg(**{clave: valor})
    with pytest.raises(ValueError, match=mensaje):
        preparar_discriminador(cfg, OBSERVACION, DISCRETO, _lote(5, False))
    with pytest.raises(ValueError, match=mensaje):
        actualizar_discriminador(_discriminador(False), torch.optim.Adam(
            _discriminador(False).parameters()), _lote(4, False), _lote(4, False), DISCRETO, cfg)


def test_la_configuracion_del_discriminador_exige_todas_sus_claves():
    cfg = _cfg()
    del cfg["gail"]["discriminador"]["peso_entropia"]
    with pytest.raises(ValueError, match="peso_entropia"):
        preparar_discriminador(cfg, OBSERVACION, DISCRETO, _lote(5, False))


# --- actualizar_discriminador --------------------------------------------------

def _esperadas(discriminador, agente, experto, continuo, peso) -> tuple[float, float]:
    """Pérdida del original (media por grupo de BCE − peso·H, sumada) y entropía
    media por grupo, calculadas con las fórmulas de libro en float64."""
    def grupo(lote, etiqueta):
        logits = discriminador(*_tensores(lote, continuo)).double()
        p = torch.sigmoid(logits)
        bce = -(etiqueta * torch.log(p) + (1 - etiqueta) * torch.log(1 - p))
        entropia = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
        return (bce - peso * entropia).mean(), entropia.mean()
    with torch.no_grad():
        (perdida_a, entropia_a), (perdida_e, entropia_e) = grupo(agente, 0.0), grupo(experto, 1.0)
        return float(perdida_a + perdida_e), float((entropia_a + entropia_e) / 2)


@TIPOS
@pytest.mark.parametrize("peso", [0.0, 0.001, 0.5])
def test_la_perdida_pondera_cada_grupo_por_igual(continuo, peso):
    discriminador = _con_pesos_aleatorios(_discriminador(continuo))
    # Grupos de tamaños distintos: con una media conjunta el resultado cambiaría.
    agente, experto = _lote(7, continuo, 1), _lote(3, continuo, 2, centro=2.0)
    perdida, entropia = _esperadas(copy.deepcopy(discriminador), agente, experto, continuo, peso)
    medidas = actualizar_discriminador(discriminador, torch.optim.Adam(discriminador.parameters()),
                                       agente, experto, _espacio(continuo),
                                       _cfg(peso_entropia=peso))
    assert medidas["perdida"] == pytest.approx(perdida, rel=1e-5)
    # La entropía registrada también pesa igual cada grupo.
    assert medidas["entropia"] == pytest.approx(entropia, rel=1e-5)


def test_con_el_discriminador_inicial_la_perdida_es_la_esperada():
    # Todos los logits valen 0: BCE = log 2 y entropía = log 2 en cada grupo.
    discriminador = _discriminador(False)
    medidas = actualizar_discriminador(discriminador, torch.optim.Adam(discriminador.parameters()),
                                       _lote(4, False), _lote(4, False), DISCRETO, _cfg())
    assert medidas["perdida"] == pytest.approx(2 * (math.log(2) - 0.001 * math.log(2)))
    assert medidas["entropia"] == pytest.approx(math.log(2))
    assert medidas["acierto_agente"] == medidas["acierto_experto"] == 0.0


@TIPOS
def test_el_discriminador_aprende_a_separar_pares_distintos(continuo):
    fijar_seed(0)
    agente, experto = _lote(64, continuo, 1, centro=-4.0), _lote(64, continuo, 2, centro=4.0)
    discriminador, optimizador = preparar_discriminador(_cfg(), OBSERVACION, _espacio(continuo),
                                                        experto)
    primera = actualizar_discriminador(discriminador, optimizador, agente, experto,
                                       _espacio(continuo), _cfg())
    for _ in range(100):
        ultima = actualizar_discriminador(discriminador, optimizador, agente, experto,
                                          _espacio(continuo), _cfg())
    assert ultima["perdida"] < primera["perdida"] / 2
    assert ultima["acierto_agente"] == ultima["acierto_experto"] == 1.0
    agente_d = calcular_recompensas_gail(discriminador, agente.estados, agente.acciones,
                                         _espacio(continuo), "log_d")
    experto_d = calcular_recompensas_gail(discriminador, experto.estados, experto.acciones,
                                          _espacio(continuo), "log_d")
    assert experto_d.mean() > agente_d.mean()


@pytest.mark.parametrize("pasos", [1, 3])
def test_da_tantos_pasos_como_pasos_adam_y_conserva_el_optimizador(pasos, monkeypatch):
    discriminador = _discriminador(False)
    optimizador = torch.optim.Adam(discriminador.parameters())
    llamadas = []
    original = optimizador.step
    monkeypatch.setattr(optimizador, "step", lambda *a, **k: (llamadas.append(1), original())[1])
    for _ in range(2):
        actualizar_discriminador(discriminador, optimizador, _lote(4, False), _lote(4, False),
                                 DISCRETO, _cfg(pasos_adam=pasos))
    assert len(llamadas) == 2 * pasos
    estado = optimizador.state[next(iter(discriminador.parameters()))]
    assert int(estado["step"]) == 2 * pasos


def test_actualizar_deja_el_discriminador_en_entrenamiento():
    discriminador = _discriminador(False).eval()
    actualizar_discriminador(discriminador, torch.optim.Adam(discriminador.parameters()),
                             _lote(4, False), _lote(4, False), DISCRETO, _cfg())
    assert discriminador.training


@pytest.mark.parametrize("cual,lote,mensaje", [
    ("agente", (np.zeros((2, 3)),), "lote_agente"),
    ("experto", (np.zeros((2, 3)),), "lote_experto"),
    ("agente", "vacio", "N > 0"),
    ("experto", "fuera", "fuera del espacio Discrete"),
    ("agente", "incoherente", "longitudes"),
])
def test_actualizar_rechaza_lotes_no_validos(cual, lote, mensaje):
    bueno = _lote(4, False)
    lote = {
        "vacio": LoteTransiciones(bueno.estados[:0], bueno.acciones[:0], bueno.ids_episodio[:0],
                                  bueno.pasos[:0]),
        "fuera": LoteTransiciones(bueno.estados, bueno.acciones + 5, bueno.ids_episodio,
                                  bueno.pasos),
        "incoherente": LoteTransiciones(bueno.estados, bueno.acciones[:2], bueno.ids_episodio,
                                        bueno.pasos),
    }.get(lote, lote) if isinstance(lote, str) else lote
    discriminador = _discriminador(False)
    argumentos = (lote, bueno) if cual == "agente" else (bueno, lote)
    with pytest.raises(ValueError, match=mensaje):
        actualizar_discriminador(discriminador, torch.optim.Adam(discriminador.parameters()),
                                 *argumentos, DISCRETO, _cfg())


# --- EntornoRecompensaGAIL -----------------------------------------------------
# Los entornos envueltos usan Discrete con start 0: sb3 no admite otro start.

DISCRETO_SB3 = spaces.Discrete(3)


def _espacio_entorno(continuo: bool) -> spaces.Space:
    return CONTINUO if continuo else DISCRETO_SB3


class _EntornoContador(gym.Env):
    """Entorno determinista: la observación codifica el paso y la última acción.

    Tras el paso k devuelve [k, valor de la acción, 1]; al reiniciar, [0, 0, 0].
    Termina tras fin pasos y siempre da recompensa 2.
    """

    def __init__(self, espacio_accion: spaces.Space, fin: int) -> None:
        self.observation_space = OBSERVACION
        self.action_space = espacio_accion
        self.fin = fin
        self.paso = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.paso = 0
        return np.zeros(3, dtype=np.float32), {}

    def step(self, accion):
        self.paso += 1
        valor = float(np.asarray(accion).reshape(-1)[0])
        observacion = np.array([self.paso, valor, 1.0], dtype=np.float32)
        return observacion, 2.0, self.paso >= self.fin, False, {}


def _envuelto(continuo: bool, fines=(3, 5), modo="menos_log_uno_menos_d", monitor=False):
    def crear(fin):
        def hacer():
            env = _EntornoContador(_espacio_entorno(continuo), fin)
            return Monitor(env) if monitor else env
        return hacer
    venv = DummyVecEnv([crear(fin) for fin in fines])
    discriminador = _con_pesos_aleatorios(_discriminador(continuo))
    return EntornoRecompensaGAIL(venv, discriminador, modo), discriminador


def _acciones(continuo: bool, paso: int, entornos: int = 2) -> np.ndarray:
    if continuo:
        return np.full((entornos, 2), (paso % 5) / 5 - 0.4, dtype=np.float32)
    return np.array([(paso + i) % 3 for i in range(entornos)])


@TIPOS
def test_el_envoltorio_guarda_cada_accion_con_el_estado_previo(continuo):
    envuelto, _ = _envuelto(continuo)
    previas = [envuelto.reset()]
    acciones = []
    for paso in range(7):
        acciones.append(_acciones(continuo, paso))
        observaciones, _, _, _ = envuelto.step(acciones[-1])
        previas.append(observaciones)
    lote = envuelto.extraer_lote()

    # Orden de recogida: paso a paso y, dentro de cada paso, entorno a entorno.
    np.testing.assert_array_equal(lote.estados, np.concatenate(previas[:7]))
    np.testing.assert_array_equal(lote.acciones, np.concatenate(acciones))
    # Tras terminar (paso 3 en el primer entorno), el estado previo es el del reinicio.
    np.testing.assert_array_equal(lote.estados[2 * 3], np.zeros(3))
    assert lote.estados.dtype == np.float32


def test_el_envoltorio_numera_partidas_y_pasos():
    envuelto, _ = _envuelto(False, fines=(3, 5))
    envuelto.reset()
    for paso in range(7):
        envuelto.step(_acciones(False, paso))
    lote = envuelto.extraer_lote()
    np.testing.assert_array_equal(lote.pasos[0::2], [0, 1, 2, 0, 1, 2, 0])
    np.testing.assert_array_equal(lote.pasos[1::2], [0, 1, 2, 3, 4, 0, 1])
    np.testing.assert_array_equal(lote.ids_episodio[0::2], [0, 0, 0, 2, 2, 2, 4])
    np.testing.assert_array_equal(lote.ids_episodio[1::2], [1, 1, 1, 1, 1, 3, 3])
    assert lote.ids_episodio.dtype == lote.pasos.dtype == np.int64


@TIPOS
@pytest.mark.parametrize("modo", MODOS_RECOMPENSA)
def test_el_envoltorio_da_la_recompensa_del_discriminador(continuo, modo):
    envuelto, discriminador = _envuelto(continuo, modo=modo)
    previas = envuelto.reset()
    for paso in range(4):
        acciones = _acciones(continuo, paso)
        esperadas = calcular_recompensas_gail(discriminador, previas, acciones,
                                              _espacio_entorno(continuo), modo)
        previas, recompensas, _, infos = envuelto.step(acciones)
        np.testing.assert_array_equal(recompensas, esperadas)
        assert recompensas.dtype == np.float32
        assert [info["recompensa_original"] for info in infos] == [2.0, 2.0]


def test_monitor_por_debajo_sigue_midiendo_el_retorno_real():
    envuelto, _ = _envuelto(False, fines=(3, 5), monitor=True)
    envuelto.reset()
    episodios = []
    for paso in range(5):
        _, _, terminados, infos = envuelto.step(_acciones(False, paso))
        episodios += [info["episode"] for info, fin in zip(infos, terminados) if fin]
    assert sorted((e["r"], e["l"]) for e in episodios) == [(6.0, 3), (10.0, 5)]


def test_el_envoltorio_no_entrena_ni_cambia_el_discriminador():
    envuelto, discriminador = _envuelto(False)
    antes = copy.deepcopy(discriminador.state_dict())
    envuelto.reset()
    for paso in range(5):
        envuelto.step(_acciones(False, paso))
    assert all(torch.equal(v, discriminador.state_dict()[k]) for k, v in antes.items())
    assert all(p.grad is None for p in discriminador.parameters())


def test_extraer_lote_vacia_el_bloque_e_iniciar_bloque_descarta():
    envuelto, _ = _envuelto(False)
    envuelto.reset()
    envuelto.step(_acciones(False, 0))
    assert envuelto.extraer_lote().numero_transiciones == 2
    with pytest.raises(ValueError, match="No hay pares"):
        envuelto.extraer_lote()
    envuelto.step(_acciones(False, 1))
    envuelto.iniciar_bloque()
    envuelto.step(_acciones(False, 2))
    lote = envuelto.extraer_lote()
    assert lote.numero_transiciones == 2
    np.testing.assert_array_equal(lote.pasos, [2, 2])


def test_las_partidas_truncadas_tambien_cambian_de_id():
    # Sin terminar nunca (fin muy alto), TimeLimit corta cada partida a los 2 pasos.
    venv = DummyVecEnv([lambda: gym.wrappers.TimeLimit(_EntornoContador(DISCRETO_SB3, 100),
                                                       max_episode_steps=2)])
    envuelto = EntornoRecompensaGAIL(venv, _discriminador(False), "log_d")
    previas = [envuelto.reset()]
    for paso in range(5):
        observaciones, _, terminados, infos = envuelto.step(_acciones(False, paso, 1))
        previas.append(observaciones)
        assert bool(terminados[0]) == infos[0]["TimeLimit.truncated"] == (paso % 2 == 1)
    lote = envuelto.extraer_lote()
    np.testing.assert_array_equal(lote.ids_episodio, [0, 0, 1, 1, 2])
    np.testing.assert_array_equal(lote.pasos, [0, 1, 0, 1, 0])
    # Tras cada corte, el estado previo es el del reinicio.
    np.testing.assert_array_equal(lote.estados, np.concatenate(previas[:5]))
    np.testing.assert_array_equal(lote.estados[[2, 4]], np.zeros((2, 3)))


@TIPOS
def test_el_envoltorio_guarda_copias_de_acciones_y_observaciones(continuo):
    envuelto, _ = _envuelto(continuo)
    observaciones = envuelto.reset()
    acciones = _acciones(continuo, 0)
    originales = observaciones.copy(), acciones.copy()
    siguientes, _, _, _ = envuelto.step(acciones)
    siguientes_originales = siguientes.copy()
    # Quien llama puede reutilizar sus arrays sin cambiar los pares guardados.
    observaciones[:] = 7
    acciones[:] = 0
    siguientes[:] = 7
    envuelto.step(_acciones(continuo, 1))
    lote = envuelto.extraer_lote()
    np.testing.assert_array_equal(lote.estados[:2], originales[0])
    np.testing.assert_array_equal(lote.acciones[:2], originales[1])
    np.testing.assert_array_equal(lote.estados[2:], siguientes_originales)


@TIPOS
def test_el_entorno_ejecuta_la_misma_accion_que_se_guarda(continuo):
    # Regresión: el entorno recibía el array original y el envoltorio guardaba una
    # copia; si se cambiaba entre step_async y step_wait, ejecutaba otra acción.
    envuelto, _ = _envuelto(continuo)
    envuelto.reset()
    acciones = _acciones(continuo, 1)
    original = acciones.copy()
    envuelto.step_async(acciones)
    acciones[:] = _acciones(continuo, 2)
    observaciones, _, _, _ = envuelto.step_wait()
    # _EntornoContador devuelve en la columna 1 el valor de la acción ejecutada.
    ejecutadas = observaciones[:, 1]
    guardadas = envuelto.extraer_lote().acciones
    np.testing.assert_array_equal(guardadas, original)
    np.testing.assert_allclose(ejecutadas, original[:, 0] if continuo else original)


def test_el_envoltorio_funciona_sobre_crear_entorno_vectorizado():
    # Monitor queda dentro de cada instancia: mide el retorno real de CartPole
    # (1 por paso) aunque el agente reciba la recompensa del discriminador.
    cfg = resolver_configuracion(RAIZ / "configs" / "default.yaml", "cartpole")
    venv = crear_entorno_vectorizado(cfg, 2, 3)
    discriminador = construir_discriminador(cfg, venv.observation_space, venv.action_space)
    envuelto = EntornoRecompensaGAIL(venv, discriminador, cfg["gail"]["recompensa"])
    try:
        envuelto.reset()
        episodios = []
        for _ in range(200):
            _, recompensas, terminados, infos = envuelto.step(np.zeros(2, dtype=np.int64))
            np.testing.assert_allclose(recompensas, math.log(2), rtol=1e-6)
            episodios += [info["episode"] for info, fin in zip(infos, terminados) if fin]
        assert episodios
        assert all(e["r"] == e["l"] for e in episodios)
    finally:
        envuelto.close()


def test_el_envoltorio_exige_reiniciar_antes_de_dar_pasos():
    envuelto, _ = _envuelto(False)
    with pytest.raises(RuntimeError, match="reset"):
        envuelto.step(_acciones(False, 0))


@pytest.mark.parametrize("caso,mensaje", [
    ("modo", "Modo de recompensa desconocido"),
    ("discriminador", "Discriminador"),
    ("observacion", "observación"),
    ("tipo_accion", "acción"),
    ("tamano_accion", "acción"),
    # Regresión: sb3 daría índices 0..2 a un entorno que espera -1..1.
    ("start", "start distinto de 0"),
])
def test_el_envoltorio_rechaza_piezas_no_compatibles(caso, mensaje):
    espacio = DISCRETO if caso == "start" else DISCRETO_SB3
    venv = DummyVecEnv([lambda: _EntornoContador(espacio, 3)])
    discriminador = {
        "discriminador": torch.nn.Linear(3, 1),
        "observacion": Discriminador(4, 3, False, [8], "tanh"),
        "tipo_accion": Discriminador(3, 3, True, [8], "tanh"),
        "tamano_accion": Discriminador(3, 2, False, [8], "tanh"),
    }.get(caso, _discriminador(False))
    with pytest.raises(ValueError, match=mensaje):
        EntornoRecompensaGAIL(venv, discriminador, "log" if caso == "modo" else "log_d")


def test_trpo_de_sb3_entrena_sobre_el_envoltorio_con_cartpole():
    fijar_seed(0)
    cfg = resolver_configuracion(RAIZ / "configs" / "default.yaml", "cartpole")
    venv = DummyVecEnv([lambda: Monitor(gym.make(cfg["entorno"]["id"]))] * 2)
    discriminador = construir_discriminador(cfg, venv.observation_space, venv.action_space)
    envuelto = EntornoRecompensaGAIL(venv, discriminador, cfg["gail"]["recompensa"])
    agente = TRPO("MlpPolicy", envuelto, n_steps=32, batch_size=16, device="cpu")
    envuelto.iniciar_bloque()
    agente.learn(64)
    lote = envuelto.extraer_lote()
    assert lote.numero_transiciones == 64
    assert lote.estados.shape == (64, 4) and lote.acciones.shape == (64,)
    # Con el discriminador inicial, cada paso recibe log 2; el retorno real lo da Monitor.
    assert np.allclose(agente.rollout_buffer.rewards[:-1], math.log(2))
    assert all(info["r"] == info["l"] for info in agente.ep_info_buffer)


# --- recompensas del bloque y reajuste de la red de valor ----------------------------
# Los agentes se entrenan en dos entornos deterministas: el primero termina sus
# partidas cada 5 pasos y el segundo se trunca por tiempo cada 8. Con 16 pasos por
# bloque hay partidas terminadas, truncadas (con el bootstrap de sb3) y cortadas.

PASOS_BLOQUE = 16


def _venv_reajuste(continuo: bool) -> DummyVecEnv:
    espacio = _espacio_entorno(continuo)
    return DummyVecEnv([
        lambda: _EntornoContador(espacio, 5),
        lambda: gym.wrappers.TimeLimit(_EntornoContador(espacio, 100), max_episode_steps=8),
    ])


def _agente_reajuste(continuo: bool, n_critic_updates: int = 3, semilla: int = 0) -> TRPO:
    """TRPO pequeño entrenado un bloque."""
    fijar_seed(semilla)
    agente = TRPO("MlpPolicy", _venv_reajuste(continuo), n_steps=PASOS_BLOQUE, batch_size=8,
                  n_critic_updates=n_critic_updates, device="cpu",
                  policy_kwargs={"net_arch": {"pi": [8], "vf": [8]}})
    agente.learn(PASOS_BLOQUE * 2)
    return agente


def _parametros(agente: TRPO, de_valor: bool) -> dict[str, torch.Tensor]:
    return {nombre: p.detach().clone() for nombre, p in agente.policy.named_parameters()
            if ("value" in nombre) == de_valor}


def _recompensas_bloque(semilla: int) -> np.ndarray:
    return crear_generador(semilla).normal(size=(PASOS_BLOQUE, 2)).astype(np.float32)


@pytest.mark.parametrize("entornos", [1, 3])
@pytest.mark.parametrize("gamma", [0.9, 0.995])
@pytest.mark.parametrize("lam", [0.0, 0.97, 1.0])
def test_retornos_gae_coinciden_con_sb3(entornos, gamma, lam):
    from stable_baselines3.common.buffers import RolloutBuffer

    from src.training.gail import _retornos_gae
    pasos = 12
    generador = crear_generador(entornos * 100 + int(lam * 10))
    buffer = RolloutBuffer(pasos, spaces.Box(-1, 1, (2,)), spaces.Discrete(2), device="cpu",
                           gae_lambda=lam, gamma=gamma, n_envs=entornos)
    recompensas = generador.normal(size=(pasos, entornos)).astype(np.float32)
    valores = generador.normal(size=(pasos, entornos)).astype(np.float32)
    inicios = (generador.random((pasos, entornos)) < 0.3).astype(np.float32)
    for paso in range(pasos):
        buffer.add(np.zeros((entornos, 2), np.float32), np.zeros((entornos, 1)),
                   recompensas[paso], inicios[paso], torch.tensor(valores[paso]),
                   torch.zeros(entornos))
    ultimos_valores = generador.normal(size=entornos).astype(np.float32)
    ultimos_terminados = generador.random(entornos) < 0.5
    buffer.compute_returns_and_advantage(torch.tensor(ultimos_valores), ultimos_terminados)
    retornos = _retornos_gae(recompensas, valores, inicios, ultimos_valores, ultimos_terminados,
                             gamma, lam)
    assert retornos.dtype == np.float32
    np.testing.assert_allclose(retornos, buffer.returns, rtol=1e-6, atol=1e-6)


@TIPOS
def test_reajuste_solo_mueve_la_red_de_valor(continuo):
    from src.training.gail import reajustar_valor_tras_discriminador
    agente = _agente_reajuste(continuo)
    actor, valor = _parametros(agente, False), _parametros(agente, True)
    anteriores = _recompensas_bloque(1)
    reajustar_valor_tras_discriminador(agente, anteriores, anteriores + 2.0)
    assert all(torch.equal(p, _parametros(agente, False)[n]) for n, p in actor.items())
    despues = _parametros(agente, True)
    assert all(not torch.equal(p, despues[n]) for n, p in valor.items())
    if continuo:
        assert "log_std" in actor


@TIPOS
def test_reajuste_suma_la_diferencia_y_conserva_el_bootstrap(continuo):
    from src.training.gail import reajustar_valor_tras_discriminador
    agente = _agente_reajuste(continuo)
    antes = agente.rollout_buffer.rewards.copy()
    # sb3 suma gamma · V(estado final) a la recompensa (2) del último paso de cada
    # partida truncada: pasos 8 y 16 del segundo entorno, y solo esos.
    truncados = np.zeros((PASOS_BLOQUE, 2), dtype=bool)
    truncados[[7, 15], 1] = True
    assert np.all(antes[truncados] != 2.0) and np.all(antes[~truncados] == 2.0)
    anteriores, nuevas = _recompensas_bloque(1), _recompensas_bloque(2)
    reajustar_valor_tras_discriminador(agente, anteriores, nuevas)
    np.testing.assert_allclose(agente.rollout_buffer.rewards, antes + nuevas - anteriores,
                               rtol=1e-6, atol=1e-6)


@TIPOS
def test_los_retornos_usan_la_red_de_valor_actual(continuo):
    from src.training.gail import _retornos_gae, reajustar_valor_tras_discriminador
    agente = _agente_reajuste(continuo, n_critic_updates=5)
    buffer = agente.rollout_buffer
    with torch.no_grad():
        planos = agente.policy.predict_values(torch.as_tensor(buffer.observations)).numpy()
        ultimos = agente.policy.predict_values(torch.as_tensor(agente._last_obs)).numpy().ravel()
    # learn ajustó la red de valor después de recoger: los valores guardados en el
    # buffer son de antes y no sirven para los retornos nuevos.
    assert not np.allclose(planos.ravel(), buffer.values.ravel())
    valores = planos.reshape(2, PASOS_BLOQUE).T
    anteriores, nuevas = _recompensas_bloque(1), _recompensas_bloque(2)
    esperados = _retornos_gae(buffer.rewards + nuevas - anteriores, valores, buffer.episode_starts,
                              ultimos, agente._last_episode_starts, agente.gamma,
                              agente.gae_lambda)
    reajustar_valor_tras_discriminador(agente, anteriores, nuevas)
    np.testing.assert_allclose(buffer.returns.ravel(),
                               buffer.swap_and_flatten(esperados).ravel(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(buffer.values.ravel(), planos.ravel(), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(buffer.advantages, buffer.returns - buffer.values)


@TIPOS
def test_reajuste_acerca_la_red_de_valor_a_los_retornos_nuevos(continuo):
    from src.training.gail import reajustar_valor_tras_discriminador
    agente = _agente_reajuste(continuo, n_critic_updates=10)
    observaciones = torch.as_tensor(agente.rollout_buffer.observations)
    anteriores = _recompensas_bloque(1)
    resultado = reajustar_valor_tras_discriminador(agente, anteriores, anteriores + 3.0)
    retornos = torch.as_tensor(agente.rollout_buffer.returns.ravel())
    with torch.no_grad():
        error = float(((agente.policy.predict_values(observaciones).ravel() - retornos) ** 2)
                      .mean())
    # Tras 10 pasadas el error es menor que la media de las pérdidas de las pasadas.
    assert np.isfinite(resultado["perdida_valor"]) and error < resultado["perdida_valor"]


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # media vacía en los registros de sb3
def test_sin_pasadas_del_critico_la_perdida_es_nan():
    from src.training.gail import reajustar_valor_tras_discriminador
    agente = _agente_reajuste(False, n_critic_updates=0)
    valor = _parametros(agente, True)
    resultado = reajustar_valor_tras_discriminador(agente, _recompensas_bloque(1),
                                                   _recompensas_bloque(2))
    assert math.isnan(resultado["perdida_valor"])
    assert all(torch.equal(p, _parametros(agente, True)[n]) for n, p in valor.items())


@TIPOS
def test_reajuste_es_reproducible(continuo):
    from src.training.gail import reajustar_valor_tras_discriminador
    resultados = []
    for _ in range(2):
        agente = _agente_reajuste(continuo, semilla=3)
        reajustar_valor_tras_discriminador(agente, _recompensas_bloque(1), _recompensas_bloque(2))
        resultados.append(_parametros(agente, True))
    assert all(torch.equal(p, resultados[1][n]) for n, p in resultados[0].items())


@pytest.mark.parametrize("modo", [False, True])
def test_reajuste_restaura_el_modo_de_la_politica(modo):
    from src.training.gail import reajustar_valor_tras_discriminador
    agente = _agente_reajuste(False)
    agente.policy.set_training_mode(modo)
    reajustar_valor_tras_discriminador(agente, _recompensas_bloque(1), _recompensas_bloque(2))
    assert agente.policy.training is modo


@pytest.mark.parametrize("cambio,mensaje", [
    ("forma", r"\[16, 2\]"), ("traspuesta", r"\[16, 2\]"), ("bool", "reales"),
    ("compleja", "reales"), ("nan", "finitas"), ("inf", "finitas"),
])
@pytest.mark.parametrize("cual", ["anteriores", "nuevas"])
def test_reajuste_rechaza_recompensas_no_validas(cambio, mensaje, cual):
    from src.training.gail import reajustar_valor_tras_discriminador
    agente = _agente_reajuste(False)
    buenas = _recompensas_bloque(1)
    marcada = np.arange(buenas.size).reshape(buenas.shape) == 3
    malas = {"forma": buenas[:-1], "traspuesta": buenas.T, "bool": buenas > 0,
             "compleja": buenas.astype(np.complex64),
             "nan": np.where(marcada, np.nan, buenas),
             "inf": np.where(marcada, np.inf, buenas)}[cambio]
    antes = agente.rollout_buffer.rewards.copy()
    argumentos = (malas, buenas) if cual == "anteriores" else (buenas, malas)
    with pytest.raises(ValueError, match=f"recompensas_{cual}.*{mensaje}"):
        reajustar_valor_tras_discriminador(agente, *argumentos)
    np.testing.assert_array_equal(agente.rollout_buffer.rewards, antes)


def test_reajuste_exige_un_bloque_recien_entrenado():
    from src.training.gail import reajustar_valor_tras_discriminador
    fijar_seed(0)
    sin_entrenar = TRPO("MlpPolicy", _venv_reajuste(False), n_steps=PASOS_BLOQUE, batch_size=8,
                        device="cpu")
    ceros = np.zeros((PASOS_BLOQUE, 2))
    with pytest.raises(ValueError, match="llamar justo después de learn"):
        reajustar_valor_tras_discriminador(sin_entrenar, ceros, ceros)
    # Recogido pero sin entrenar todavía (como en el callback on_rollout_end).
    agente = _agente_reajuste(False)
    agente.rollout_buffer.generator_ready = False
    with pytest.raises(ValueError, match="llamar justo después de learn"):
        reajustar_valor_tras_discriminador(agente, ceros, ceros)


def test_reajuste_solo_una_vez_por_bloque():
    from src.training.gail import reajustar_valor_tras_discriminador
    agente = _agente_reajuste(False)
    anteriores, nuevas = _recompensas_bloque(1), _recompensas_bloque(2)
    reajustar_valor_tras_discriminador(agente, anteriores, nuevas)
    recompensas = agente.rollout_buffer.rewards.copy()
    with pytest.raises(ValueError, match="ya se reajustó"):
        reajustar_valor_tras_discriminador(agente, anteriores, nuevas)
    np.testing.assert_array_equal(agente.rollout_buffer.rewards, recompensas)
    agente.learn(PASOS_BLOQUE * 2, reset_num_timesteps=False)
    reajustar_valor_tras_discriminador(agente, anteriores, nuevas)  # bloque nuevo: se admite


def test_reajuste_solo_admite_trpo():
    from stable_baselines3 import PPO

    from src.training.gail import reajustar_valor_tras_discriminador
    fijar_seed(0)
    ppo = PPO("MlpPolicy", _venv_reajuste(False), n_steps=PASOS_BLOQUE, batch_size=8,
              device="cpu")
    with pytest.raises(ValueError, match="TRPO"):
        reajustar_valor_tras_discriminador(ppo, np.zeros((PASOS_BLOQUE, 2)),
                                           np.zeros((PASOS_BLOQUE, 2)))


@TIPOS
def test_extraer_bloque_da_las_recompensas_de_cada_paso(continuo):
    envuelto, _ = _envuelto(continuo)
    envuelto.reset()
    devueltas = []
    for paso in range(5):
        devueltas.append(envuelto.step(_acciones(continuo, paso))[1])
    lote, recompensas = envuelto.extraer_bloque()
    assert recompensas.shape == (5, 2) and recompensas.dtype == np.float32
    np.testing.assert_array_equal(recompensas, np.stack(devueltas))
    # El par i corresponde a la recompensa [i // entornos, i % entornos].
    assert lote.numero_transiciones == recompensas.size
    with pytest.raises(ValueError, match="No hay pares"):
        envuelto.extraer_bloque()
    envuelto.step(_acciones(continuo, 0))
    envuelto.iniciar_bloque()
    envuelto.step(_acciones(continuo, 1))
    assert envuelto.extraer_bloque()[1].shape == (1, 2)


def test_las_recompensas_del_bloque_coinciden_con_el_buffer_de_trpo():
    """Integración: el orden de extraer_bloque es el del rollout_buffer de sb3, así que
    puntuar el lote con el mismo discriminador no cambia las recompensas."""
    from src.training.gail import calcular_recompensas_gail, reajustar_valor_tras_discriminador
    cfg = resolver_configuracion(RAIZ / "configs" / "default.yaml", "cartpole")
    fijar_seed(0)
    venv = crear_entorno_vectorizado(cfg, 2, 3)
    discriminador = _con_pesos_aleatorios(
        construir_discriminador(cfg, venv.observation_space, venv.action_space))
    envuelto = EntornoRecompensaGAIL(venv, discriminador, cfg["gail"]["recompensa"])
    agente = TRPO("MlpPolicy", envuelto, n_steps=32, batch_size=16, device="cpu")
    try:
        envuelto.iniciar_bloque()
        agente.learn(64)
        lote, anteriores = envuelto.extraer_bloque()
        # 32 pasos de CartPole no llegan al límite de 500: sin bootstrap.
        np.testing.assert_array_equal(agente.rollout_buffer.rewards, anteriores)
        nuevas = calcular_recompensas_gail(discriminador, lote.estados, lote.acciones,
                                           envuelto.action_space, cfg["gail"]["recompensa"])
        nuevas = nuevas.reshape(anteriores.shape)
        # Mismo orden; solo cambia el redondeo de float32 al puntuar lotes de otro tamaño.
        np.testing.assert_allclose(nuevas, anteriores, rtol=1e-5, atol=1e-6)
        reajustar_valor_tras_discriminador(agente, anteriores, nuevas)
        np.testing.assert_allclose(agente.rollout_buffer.rewards, anteriores, rtol=1e-5,
                                   atol=1e-6)
    finally:
        envuelto.close()


# --- iteración de GAIL ------------------------------------------------------------
# Mismos entornos deterministas que el reajuste: en cada bloque de 16 pasos hay
# partidas terminadas, truncadas y cortadas. Los pasos internos se espían para
# comprobar el orden y qué recibe cada uno.

def _pares_expertos(continuo: bool, n: int = 40, semilla: int = 5) -> LoteTransiciones:
    """Pares expertos inventados en el espacio de acción del entorno (Discrete con start 0)."""
    lote = _lote(n, continuo, semilla, centro=2.0)
    acciones = lote.acciones if continuo else lote.acciones + 1
    return LoteTransiciones(lote.estados, acciones, lote.ids_episodio, lote.pasos)


def _piezas_gail(continuo: bool, semilla: int = 0, normalizar: bool = False,
                 norm_reward: bool = False):
    """TRPO pequeño sobre el envoltorio, sin entrenar, y el optimizador del discriminador."""
    fijar_seed(semilla)
    discriminador = _con_pesos_aleatorios(_discriminador(continuo), semilla)
    envuelto = EntornoRecompensaGAIL(_venv_reajuste(continuo), discriminador,
                                     "menos_log_uno_menos_d")
    capa = (VecNormalize(envuelto, norm_obs=True, norm_reward=norm_reward)
            if normalizar or norm_reward else envuelto)
    agente = TRPO("MlpPolicy", capa, n_steps=PASOS_BLOQUE, batch_size=8, n_critic_updates=3,
                  device="cpu", policy_kwargs={"net_arch": {"pi": [8], "vf": [8]}})
    return agente, envuelto, torch.optim.Adam(discriminador.parameters(), lr=0.01)


def _estado(modulo: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {nombre: valor.detach().clone() for nombre, valor in modulo.state_dict().items()}


def _iguales(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[n], b[n]) for n in a)


def _espiar(monkeypatch, registro: dict) -> None:
    """Sustituye los pasos de iteracion_gail por versiones que anotan lo que reciben."""
    import src.training.gail as gail
    originales = {nombre: getattr(gail, nombre) for nombre in (
        "iteracion_trpo", "muestrear_expertos", "actualizar_discriminador",
        "reajustar_valor_tras_discriminador")}
    registro["orden"] = []

    def trpo(agente):
        discriminador = registro["discriminador"]
        antes = _estado(discriminador)
        resultado = originales["iteracion_trpo"](agente)
        registro["fijo_en_el_bloque"] = _iguales(antes, _estado(discriminador))
        registro["orden"].append("trpo")
        return resultado

    def muestrear(pares, cantidad, generador):
        registro["orden"].append("muestrear")
        registro["expertos"] = originales["muestrear_expertos"](pares, cantidad, generador)
        return registro["expertos"]

    def actualizar(discriminador, optimizador, lote_agente, lote_experto, espacio, cfg):
        registro["orden"].append("discriminador")
        registro.update(lote=lote_agente, experto_recibido=lote_experto,
                        d_antes=copy.deepcopy(discriminador))
        return originales["actualizar_discriminador"](discriminador, optimizador, lote_agente,
                                                      lote_experto, espacio, cfg)

    def reajustar(agente, anteriores, nuevas):
        registro["orden"].append("reajuste")
        registro.update(anteriores=anteriores, nuevas=nuevas,
                        buffer_antes=agente.rollout_buffer.rewards.copy(),
                        d_despues=copy.deepcopy(registro["discriminador"]))
        return originales["reajustar_valor_tras_discriminador"](agente, anteriores, nuevas)

    for nombre, espia in (("iteracion_trpo", trpo), ("muestrear_expertos", muestrear),
                          ("actualizar_discriminador", actualizar),
                          ("reajustar_valor_tras_discriminador", reajustar)):
        monkeypatch.setattr(gail, nombre, espia)


@TIPOS
def test_iteracion_gail_encadena_los_pasos_en_orden(continuo, monkeypatch):
    from src.training.gail import iteracion_gail
    agente, envuelto, optimizador = _piezas_gail(continuo)
    registro = {"discriminador": envuelto.discriminador}
    _espiar(monkeypatch, registro)
    iteracion_gail(agente, envuelto, optimizador, _pares_expertos(continuo),
                   crear_generador(7), _cfg())

    assert registro["orden"] == ["trpo", "muestrear", "discriminador", "reajuste"]
    assert registro["fijo_en_el_bloque"]
    lote = registro["lote"]
    assert lote.numero_transiciones == PASOS_BLOQUE * 2
    assert registro["experto_recibido"] is registro["expertos"]
    assert registro["expertos"].numero_transiciones == lote.numero_transiciones

    def puntuar(discriminador):
        return calcular_recompensas_gail(discriminador, lote.estados, lote.acciones,
                                         envuelto.action_space, envuelto.modo
                                         ).reshape(PASOS_BLOQUE, 2)

    # anteriores: las del discriminador del bloque, las que guardó sb3 salvo el
    # bootstrap de las truncadas (pasos 8 y 16 del segundo entorno).
    truncados = np.zeros((PASOS_BLOQUE, 2), dtype=bool)
    truncados[[7, 15], 1] = True
    anteriores, nuevas = registro["anteriores"], registro["nuevas"]
    np.testing.assert_allclose(anteriores, puntuar(registro["d_antes"]), rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(registro["buffer_antes"][~truncados], anteriores[~truncados])
    assert np.all(registro["buffer_antes"][truncados] != anteriores[truncados])
    # nuevas: las del discriminador ya actualizado, que es distinto.
    assert not _iguales(_estado(registro["d_antes"]), _estado(registro["d_despues"]))
    np.testing.assert_allclose(nuevas, puntuar(registro["d_despues"]), rtol=1e-5, atol=1e-6)
    assert not np.allclose(nuevas, anteriores)
    np.testing.assert_allclose(agente.rollout_buffer.rewards,
                               registro["buffer_antes"] + nuevas - anteriores,
                               rtol=1e-6, atol=1e-6)


def test_iteracion_gail_muestrea_los_expertos_con_su_generador(monkeypatch):
    from src.training.gail import iteracion_gail
    agente, envuelto, optimizador = _piezas_gail(False)
    registro = {"discriminador": envuelto.discriminador}
    _espiar(monkeypatch, registro)
    pares, generador = _pares_expertos(False), crear_generador(7)
    copia = copy.deepcopy(generador)
    iteracion_gail(agente, envuelto, optimizador, pares, generador, _cfg())
    esperados = muestrear_expertos(pares, PASOS_BLOQUE * 2, copia)
    for campo in ("estados", "acciones", "ids_episodio", "pasos"):
        np.testing.assert_array_equal(getattr(registro["expertos"], campo),
                                      getattr(esperados, campo))
    # No usa el generador para nada más.
    assert generador.random() == copia.random()


@TIPOS
def test_iteracion_gail_devuelve_las_medidas_de_cada_paso(continuo):
    from src.training.gail import iteracion_gail
    agente, envuelto, optimizador = _piezas_gail(continuo)
    pares, generador = _pares_expertos(continuo), crear_generador(7)
    claves = {"pasos", "partidas", "retorno_real_medio", "longitud_media",
              "recompensa_gail_media", "discriminador/perdida", "discriminador/entropia",
              "discriminador/acierto_agente", "discriminador/acierto_experto",
              "reajuste/perdida_valor"}
    for numero in range(1, 4):
        medidas = iteracion_gail(agente, envuelto, optimizador, pares, generador, _cfg())
        assert medidas.keys() == claves
        assert all(isinstance(valor, float) for valor in medidas.values())
        # Cada iteración es un bloque nuevo, que se reajusta una vez.
        assert medidas["pasos"] == PASOS_BLOQUE * 2 * numero
        assert agente._pasos_ultimo_reajuste == agente.num_timesteps
        assert np.isfinite(medidas["reajuste/perdida_valor"])
        assert 0 <= medidas["discriminador/acierto_agente"] <= 1
        assert 0 <= medidas["discriminador/acierto_experto"] <= 1
        # Sin Monitor debajo, sb3 no registra partidas: el retorno real queda en nan
        # (con Monitor, en la prueba de integración).
        assert medidas["partidas"] == 0 and math.isnan(medidas["retorno_real_medio"])


def test_la_recompensa_media_es_la_que_recibio_el_agente(monkeypatch):
    from src.training.gail import iteracion_gail
    agente, envuelto, optimizador = _piezas_gail(False)
    registro = {"discriminador": envuelto.discriminador}
    _espiar(monkeypatch, registro)
    medidas = iteracion_gail(agente, envuelto, optimizador, _pares_expertos(False),
                             crear_generador(7), _cfg())
    assert medidas["recompensa_gail_media"] == pytest.approx(float(registro["anteriores"].mean()))
    assert medidas["recompensa_gail_media"] != pytest.approx(float(registro["nuevas"].mean()))


@TIPOS
def test_iteracion_gail_es_reproducible(continuo):
    from src.training.gail import iteracion_gail
    resultados = []
    for _ in range(2):
        agente, envuelto, optimizador = _piezas_gail(continuo, semilla=3)
        pares, generador = _pares_expertos(continuo), crear_generador(7)
        medidas = [iteracion_gail(agente, envuelto, optimizador, pares, generador, _cfg())
                   for _ in range(2)]
        resultados.append((medidas, _estado(agente.policy), _estado(envuelto.discriminador)))
    np.testing.assert_equal(resultados[0][0], resultados[1][0])
    assert _iguales(resultados[0][1], resultados[1][1])
    assert _iguales(resultados[0][2], resultados[1][2])


def test_iteracion_gail_con_vecnormalize_encima(monkeypatch):
    """El agente ve observaciones normalizadas; el discriminador, las originales."""
    from src.training.gail import iteracion_gail
    agente, envuelto, optimizador = _piezas_gail(False, normalizar=True)
    registro = {"discriminador": envuelto.discriminador}
    _espiar(monkeypatch, registro)
    for _ in range(2):
        iteracion_gail(agente, envuelto, optimizador, _pares_expertos(False),
                       crear_generador(7), _cfg())
    # _EntornoContador da [paso, acción, 0 o 1]: enteros sin normalizar.
    estados = registro["lote"].estados
    np.testing.assert_array_equal(estados, np.round(estados))
    assert set(np.unique(estados[:, 2])) <= {0.0, 1.0}
    assert not np.array_equal(agente.rollout_buffer.observations, np.round(
        agente.rollout_buffer.observations))
    assert agente._pasos_ultimo_reajuste == PASOS_BLOQUE * 2 * 2


@pytest.mark.parametrize("caso,mensaje", [
    ("agente", "TRPO"),
    ("envoltorio", "EntornoRecompensaGAIL"),
    ("otro_envoltorio", "no juega en este envoltorio"),
    ("norm_reward", "norm_reward"),
    ("optimizador", "optimizador no es el del discriminador"),
    ("optimizador_parcial", "optimizador no es el del discriminador"),
    ("cfg", "gail.discriminador"),
    ("pares_tipo", "LoteTransiciones"),
    ("pares_vacios", "N > 0"),
    ("pares_accion", "fuera del espacio"),
    ("pares_dimension", "dimensión de observación"),
    ("generador", "numpy.random.Generator"),
])
def test_iteracion_gail_comprueba_todo_antes_de_jugar(caso, mensaje):
    from src.training.gail import iteracion_gail
    agente, envuelto, optimizador = _piezas_gail(False, norm_reward=caso == "norm_reward")
    pares, generador, cfg = _pares_expertos(False), crear_generador(7), _cfg()
    argumentos = {"agente": agente, "envoltorio": envuelto, "optimizador": optimizador,
                  "pares_expertos": pares, "generador": generador, "cfg": cfg}
    otro, _ = _envuelto(False)
    cambios = {
        "agente": {"agente": object()},
        "envoltorio": {"envoltorio": envuelto.venv},
        "otro_envoltorio": {"envoltorio": otro},
        "norm_reward": {},
        "optimizador": {"optimizador": torch.optim.Adam(otro.discriminador.parameters())},
        "optimizador_parcial": {"optimizador": torch.optim.Adam(
            list(envuelto.discriminador.parameters())[:1])},
        "cfg": {"cfg": {}},
        "pares_tipo": {"pares_expertos": (pares.estados, pares.acciones)},
        "pares_vacios": {"pares_expertos": LoteTransiciones(
            pares.estados[:0], pares.acciones[:0], pares.ids_episodio[:0], pares.pasos[:0])},
        "pares_accion": {"pares_expertos": LoteTransiciones(
            pares.estados, np.full_like(pares.acciones, 3), pares.ids_episodio, pares.pasos)},
        "pares_dimension": {"pares_expertos": LoteTransiciones(
            np.zeros((len(pares.estados), 4), np.float32), pares.acciones,
            pares.ids_episodio, pares.pasos)},
        "generador": {"generador": np.random.RandomState(0)},
    }[caso]
    discriminador = _estado(envuelto.discriminador)
    with pytest.raises(ValueError, match=mensaje):
        iteracion_gail(**{**argumentos, **cambios})
    assert agente.num_timesteps == 0
    assert _iguales(discriminador, _estado(envuelto.discriminador))


def test_iteracion_gail_con_las_piezas_del_proyecto():
    """Integración: crear_trpo, crear_entorno_vectorizado y preparar_discriminador con
    la config de CartPole (reducida), como lo montará el entrenamiento de GAIL."""
    from src.training.gail import iteracion_gail
    from src.training.trpo import crear_trpo
    cfg = _cfg(normalizar_entrada=True)
    cfg["trpo"].update(pasos_por_iteracion=64, n_entornos=2, batch_size_valor=32,
                       capas_ocultas=[8], pasos_valor=2)
    fijar_seed(0)
    generador = crear_generador(1)
    pares = LoteTransiciones(generador.normal(scale=0.05, size=(50, 4)).astype(np.float32),
                             generador.integers(0, 2, size=50), np.arange(50) // 10,
                             np.arange(50) % 10)
    venv = crear_entorno_vectorizado(cfg, 2, 3)
    try:
        discriminador, optimizador = preparar_discriminador(
            cfg, venv.observation_space, venv.action_space, pares)
        envuelto = EntornoRecompensaGAIL(venv, discriminador, cfg["gail"]["recompensa"])
        agente = crear_trpo(cfg, envuelto)
        for numero in range(1, 3):
            medidas = iteracion_gail(agente, envuelto, optimizador, pares, generador, cfg)
            assert medidas["pasos"] == 64 * numero
            # Monitor va debajo del envoltorio: el retorno es el real de CartPole.
            assert medidas["retorno_real_medio"] == medidas["longitud_media"]
            assert all(np.isfinite(valor) for valor in medidas.values())
    finally:
        venv.close()


def test_iteracion_gail_descarta_pares_que_queden_de_antes(monkeypatch):
    """Si una iteración anterior se interrumpió tras jugar, sus pares no se mezclan."""
    from src.training.gail import iteracion_gail
    agente, envuelto, optimizador = _piezas_gail(False)
    registro = {"discriminador": envuelto.discriminador}
    _espiar(monkeypatch, registro)
    envuelto.reset()
    for paso in range(3):
        envuelto.step(_acciones(False, paso))
    iteracion_gail(agente, envuelto, optimizador, _pares_expertos(False), crear_generador(7),
                   _cfg())
    assert registro["lote"].numero_transiciones == PASOS_BLOQUE * 2


# --- entrenamiento de GAIL --------------------------------------------------------
# CartPole real con la config del proyecto reducida: 3 iteraciones de 64 pasos en
# 2 entornos y partidas expertas inventadas (6 partidas: 4 de train y 2 de validación).

def _cfg_gail(**gail) -> dict:
    cfg = _cfg()
    cfg["trpo"].update(pasos_por_iteracion=64, n_entornos=2, batch_size_valor=32,
                       capas_ocultas=[8], pasos_valor=2)
    cfg["gail"].update(iteraciones=3, evaluar_cada=2, episodios_evaluacion=2)
    cfg["demostraciones"]["frecuencia_pares"] = 2
    cfg["gail"].update(gail)
    return cfg


def _datos_gail(acciones_validas: bool = True):
    from src.data.splits import crear_splits
    from src.data.tipos import Trayectoria
    generador = crear_generador(11)
    trayectorias = []
    for identificador in range(6):
        pasos = 12 + identificador
        estados = generador.normal(scale=0.05, size=(pasos + 1, 4)).astype(np.float32)
        acciones = generador.integers(0, 2, size=pasos) + (0 if acciones_validas else 2)
        finales = np.zeros(pasos, dtype=bool)
        finales[-1] = True
        trayectorias.append(Trayectoria(identificador, estados[:-1], acciones, estados[1:],
                                        np.ones(pasos, np.float32), finales,
                                        np.zeros(pasos, dtype=bool)))
    return trayectorias, crear_splits(trayectorias, 4, 2, (1, 4), crear_generador(0), "a" * 64)


def _entrenar(cfg=None, cantidad=4, repeticion=0, run=None):
    from src.training.gail import entrenar_gail
    trayectorias, splits = _datos_gail()
    return entrenar_gail(cfg or _cfg_gail(), trayectorias, splits, cantidad, repeticion, run)


def test_entrenar_gail_da_sus_iteraciones_y_devuelve_el_resumen():
    from src.utils.seed import semillas_repeticion
    cfg = _cfg_gail()
    agente, discriminador, resumen = _entrenar(cfg, cantidad=4, repeticion=2)
    _, splits = _datos_gail()
    assert isinstance(agente, TRPO) and isinstance(discriminador, Discriminador)
    assert agente.num_timesteps == resumen["pasos"] == 3 * 64
    assert resumen["entorno"] == "CartPole-v1"
    assert (resumen["cantidad"], resumen["repeticion"], resumen["iteraciones"]) == (4, 2, 3)
    assert resumen["ids_partidas"] == list(splits.subconjuntos[4])
    assert resumen["huella_dataset"] == "a" * 64
    semilla = cfg["experimento"]["semilla_repeticiones"]
    assert resumen["semillas"] == {"base": semilla, **semillas_repeticion(semilla, 2)}
    assert [m["iteracion"] for m in resumen["historial"]] == [1, 2, 3]
    assert [m["pasos"] for m in resumen["historial"]] == [64.0, 128.0, 192.0]
    assert resumen["normalizacion"] is None and resumen["segundos"] > 0
    for validacion in resumen["validaciones"]:
        assert len(validacion["retornos"]) == 2
        # Recompensa real de CartPole: 1 por paso.
        assert validacion["retorno_medio"] == validacion["longitud_media"]


@pytest.mark.parametrize("evaluar_cada,esperadas", [(1, [1, 2, 3]), (2, [2, 3]), (3, [3]),
                                                    (5, [3])])
def test_entrenar_gail_valida_cada_evaluar_cada_y_al_final(evaluar_cada, esperadas):
    _, _, resumen = _entrenar(_cfg_gail(evaluar_cada=evaluar_cada))
    assert [v["iteracion"] for v in resumen["validaciones"]] == esperadas
    assert [v["pasos"] for v in resumen["validaciones"]] == [64 * i for i in esperadas]


def test_la_validacion_usa_siempre_las_mismas_partidas(monkeypatch):
    import src.training.gail as gail
    llamadas = []
    original = gail.evaluar_retorno

    def espia(modelo, cfg, episodios, semilla, determinista, estadisticas=None):
        llamadas.append((episodios, semilla, determinista, estadisticas))
        return original(modelo, cfg, episodios, semilla, determinista, estadisticas)

    monkeypatch.setattr(gail, "evaluar_retorno", espia)
    _, _, resumen = _entrenar(_cfg_gail(evaluar_cada=1, evaluacion_determinista=False))
    assert len(llamadas) == 3 and len(set(llamadas)) == 1
    assert llamadas[0] == (2, resumen["semillas"]["evaluacion"], False, None)


def test_entrenar_gail_es_reproducible():
    resultados = [_entrenar(repeticion=1) for _ in range(2)]
    (agente_a, d_a, resumen_a), (agente_b, d_b, resumen_b) = resultados
    assert _iguales(_estado(agente_a.policy), _estado(agente_b.policy))
    assert _iguales(_estado(d_a), _estado(d_b))
    np.testing.assert_equal(resumen_a["historial"], resumen_b["historial"])
    np.testing.assert_equal(resumen_a["validaciones"], resumen_b["validaciones"])


def test_validar_no_cambia_el_entrenamiento():
    """Las partidas de validación no tocan los generadores del entrenamiento."""
    agente_a, d_a, resumen_a = _entrenar(_cfg_gail(evaluar_cada=1))
    agente_b, d_b, resumen_b = _entrenar(_cfg_gail(evaluar_cada=3))
    assert _iguales(_estado(agente_a.policy), _estado(agente_b.policy))
    assert _iguales(_estado(d_a), _estado(d_b))
    np.testing.assert_equal(resumen_a["historial"], resumen_b["historial"])


def test_cada_repeticion_entrena_distinto():
    agente_a, d_a, resumen_a = _entrenar(repeticion=0)
    agente_b, d_b, resumen_b = _entrenar(repeticion=1)
    assert resumen_a["semillas"] != resumen_b["semillas"]
    assert not _iguales(_estado(agente_a.policy), _estado(agente_b.policy))
    assert not _iguales(_estado(d_a), _estado(d_b))


@pytest.mark.parametrize("cantidad", [1, 4])
def test_gail_usa_los_mismos_pares_expertos_que_bc(cantidad, monkeypatch):
    import src.training.gail as gail
    from src.data.loaders import construir_loaders_bc
    from src.utils.seed import semillas_repeticion
    recibidos = []
    original = gail.preparar_discriminador

    def espia(cfg, observacion, accion, pares):
        recibidos.append(pares)
        return original(cfg, observacion, accion, pares)

    monkeypatch.setattr(gail, "preparar_discriminador", espia)
    cfg = _cfg_gail(iteraciones=1)
    _, _, resumen = _entrenar(cfg, cantidad=cantidad, repeticion=3)
    trayectorias, splits = _datos_gail()
    semilla = semillas_repeticion(cfg["experimento"]["semilla_repeticiones"], 3)["pares"]
    train, _ = construir_loaders_bc(trayectorias, splits, cantidad, spaces.Discrete(2), cfg,
                                    crear_generador(semilla))
    np.testing.assert_array_equal(recibidos[0].estados, train.dataset.estados.numpy())
    assert resumen["pares_expertos"] == len(train.dataset)
    assert set(recibidos[0].ids_episodio.tolist()) == set(splits.subconjuntos[cantidad])


class _RunFalso:
    def __init__(self) -> None:
        self.registros: list[tuple[dict, int]] = []

    def log(self, metricas: dict, step: int) -> None:
        self.registros.append((metricas, step))


def test_entrenar_gail_registra_cada_iteracion_y_cada_validacion():
    run = _RunFalso()
    _, _, resumen = _entrenar(_cfg_gail(), run=run)
    iteraciones = [(m, paso) for m, paso in run.registros if "pasos" in m]
    validaciones = [(m, paso) for m, paso in run.registros if "validacion/retorno_medio" in m]
    assert len(run.registros) == len(iteraciones) + len(validaciones)
    assert [paso for _, paso in iteraciones] == [64, 128, 192]
    assert [paso for _, paso in validaciones] == [128, 192]
    for metricas, _ in validaciones:
        assert set(metricas) == {"validacion/retorno_medio", "validacion/retorno_desviacion",
                                 "validacion/longitud_media"}
    assert validaciones[-1][0]["validacion/retorno_medio"] == \
        resumen["validaciones"][-1]["retorno_medio"]


def test_entrenar_gail_normaliza_observaciones_si_el_perfil_lo_pide(monkeypatch):
    import src.training.gail as gail
    estadisticas = []
    original = gail.evaluar_retorno

    def espia(modelo, cfg, episodios, semilla, determinista, normalizacion=None):
        estadisticas.append(normalizacion)
        return original(modelo, cfg, episodios, semilla, determinista, normalizacion)

    monkeypatch.setattr(gail, "evaluar_retorno", espia)
    cfg = _cfg_gail()
    # Un límite distinto del de sb3 (10) comprueba que se usa el de la config.
    cfg["trpo"].update(normalizar_observaciones=True, limite_normalizacion=5.0)
    agente, _, resumen = _entrenar(cfg)
    assert isinstance(agente.get_env(), VecNormalize)
    assert isinstance(agente.get_env().venv, EntornoRecompensaGAIL)
    assert len(resumen["normalizacion"]["media"]) == 4
    assert resumen["normalizacion"]["limite"] == cfg["trpo"]["limite_normalizacion"]
    assert all(e is not None for e in estadisticas)
    assert estadisticas[-1] == resumen["normalizacion"]


def _cfg_sin(seccion: str, clave: str) -> dict:
    cfg = _cfg_gail()
    del cfg[seccion][clave]
    return cfg


@pytest.mark.parametrize("cambio,mensaje", [
    (lambda c: c["gail"].update(iteraciones=0), "gail.iteraciones"),
    (lambda c: c["gail"].update(iteraciones=True), "gail.iteraciones"),
    (lambda c: c["gail"].update(evaluar_cada=0), "gail.evaluar_cada"),
    (lambda c: c["gail"].update(episodios_evaluacion=2.0), "gail.episodios_evaluacion"),
    (lambda c: c["gail"].update(evaluacion_determinista="si"), "gail.evaluacion_determinista"),
    (lambda c: c["gail"].update(recompensa="d"), "gail.recompensa"),
    (lambda c: c["gail"].pop("evaluar_cada"), "Falta gail.evaluar_cada"),
    (lambda c: c.pop("gail"), "sección gail"),
    (lambda c: c["trpo"].update(pasos_por_iteracion=63), "trpo.pasos_por_iteracion"),
    (lambda c: c["gail"]["discriminador"].update(learning_rate=0), "learning_rate"),
    (lambda c: c["experimento"].pop("semilla_repeticiones"), "semilla_repeticiones"),
    (lambda c: c["experimento"].update(semilla_repeticiones=-1), "semilla"),
    (lambda c: c["demostraciones"].pop("frecuencia_pares"), "frecuencia_pares"),
])
def test_entrenar_gail_comprueba_la_config_antes_de_crear_entornos(cambio, mensaje,
                                                                   monkeypatch):
    import src.training.gail as gail

    def no_crear(*_):
        raise AssertionError("no debería llegar a crear los entornos")

    monkeypatch.setattr(gail, "crear_entorno_vectorizado", no_crear)
    cfg = _cfg_gail()
    cambio(cfg)
    with pytest.raises(ValueError, match=mensaje):
        _entrenar(cfg)


@pytest.mark.parametrize("cantidad,repeticion,mensaje", [
    (2, 0, "subconjunto"), (4, -1, "repeticion"), (4, 1.0, "repeticion"),
])
def test_entrenar_gail_comprueba_los_argumentos_antes_de_crear_entornos(cantidad, repeticion,
                                                                        mensaje, monkeypatch):
    import src.training.gail as gail

    def no_crear(*_):
        raise AssertionError("no debería llegar a crear los entornos")

    monkeypatch.setattr(gail, "crear_entorno_vectorizado", no_crear)
    with pytest.raises(ValueError, match=mensaje):
        _entrenar(cantidad=cantidad, repeticion=repeticion)
    trayectorias, splits = _datos_gail()
    from src.training.gail import entrenar_gail
    with pytest.raises(ValueError, match="faltan"):
        entrenar_gail(_cfg_gail(), trayectorias[:-1], splits, 4, 0)


@pytest.mark.parametrize("fallo", ["iteracion", "pares"])
def test_entrenar_gail_cierra_los_entornos_si_falla(fallo, monkeypatch):
    import src.training.gail as gail
    cerrados = []
    original = gail.crear_entorno_vectorizado

    def crear(*argumentos):
        venv = original(*argumentos)
        cerrar = venv.close
        venv.close = lambda: (cerrados.append(True), cerrar())
        return venv

    monkeypatch.setattr(gail, "crear_entorno_vectorizado", crear)
    trayectorias, splits = _datos_gail(acciones_validas=fallo != "pares")
    if fallo == "iteracion":
        def romper(*_):
            raise RuntimeError("fallo a mitad")
        monkeypatch.setattr(gail, "iteracion_gail", romper)
    with pytest.raises((RuntimeError, ValueError)):
        gail.entrenar_gail(_cfg_gail(), trayectorias, splits, 4, 0)
    assert cerrados == [True]


# --- guardado y carga de GAIL -------------------------------------------------------
# El NPZ de estas pruebas son bytes cualquiera: procedencia_datos solo calcula su
# huella. Los splits se crean con esa huella, como en generar_dataset.

def _datos_en_disco(carpeta: Path):
    from src.data.splits import crear_splits, guardar_splits, huella_archivo
    trayectorias, _ = _datos_gail()
    ruta_npz = carpeta / "datos" / "demostraciones.npz"
    ruta_npz.parent.mkdir(parents=True)
    ruta_npz.write_bytes(b"demostraciones de prueba")
    splits = crear_splits(trayectorias, 4, 2, (1, 4), crear_generador(0),
                          huella_archivo(ruta_npz))
    ruta_splits = guardar_splits(carpeta / "datos" / "splits.json", splits)
    return trayectorias, splits, ruta_npz, ruta_splits


def _entrenado(carpeta: Path, cfg=None):
    """Entrena 2 iteraciones con datos en disco y devuelve todo lo que necesita guardar_gail."""
    from src.training.barrido import procedencia_datos
    from src.training.gail import entrenar_gail
    trayectorias, splits, ruta_npz, ruta_splits = _datos_en_disco(carpeta)
    cfg = cfg or _cfg_gail(iteraciones=2)
    agente, discriminador, resumen = entrenar_gail(cfg, trayectorias, splits, 4, 1)
    return agente, discriminador, cfg, resumen, procedencia_datos(ruta_npz, ruta_splits,
                                                                  trayectorias)


def _json_estricto(ruta: Path) -> dict:
    def rechazar(constante):
        raise ValueError(f"constante no estándar: {constante}")
    return json.loads(ruta.read_text(encoding="utf-8"), parse_constant=rechazar)


def test_guardar_y_cargar_gail_conserva_agente_discriminador_y_metadatos(tmp_path):
    from src.training.gail import cargar_gail, guardar_gail
    agente, discriminador, cfg, resumen, datos = _entrenado(tmp_path)
    destino = tmp_path / "gail" / "cantidad_4_rep_1"
    guardar_gail(destino, agente, discriminador, cfg, resumen, datos)
    assert sorted(p.name for p in destino.iterdir()) == ["discriminador.pt", "gail.json",
                                                          "modelo.zip"]

    cargado, d_cargado, metadatos = cargar_gail(destino)
    assert _iguales(_estado(cargado.policy), _estado(agente.policy))
    assert _iguales(_estado(d_cargado), _estado(discriminador)) and not d_cargado.training
    observaciones = crear_generador(2).normal(scale=0.1, size=(20, 4)).astype(np.float32)
    np.testing.assert_array_equal(cargado.predict(observaciones, deterministic=True)[0],
                                  agente.predict(observaciones, deterministic=True)[0])
    estados = torch.as_tensor(observaciones)
    acciones = torch.zeros(20, dtype=torch.int64)
    with torch.no_grad():
        assert torch.equal(d_cargado(estados, acciones), discriminador.eval()(estados, acciones))

    assert set(metadatos) == {"configuracion", "resumen", "datos", "discriminador", "codigo",
                              "entorno_ejecucion"}
    assert metadatos["datos"] == datos
    assert metadatos["discriminador"] == {"dim_observacion": 4, "dim_accion": 2,
                                          "continua": False, "capas_ocultas": [8],
                                          "activacion": cfg["gail"]["discriminador"]["activacion"]}
    for clave in ("cantidad", "repeticion", "ids_partidas", "pares_expertos", "semillas",
                  "huella_dataset", "pasos", "segundos", "normalizacion"):
        assert metadatos["resumen"][clave] == resumen[clave]
    assert len(metadatos["resumen"]["historial"]) == 2
    assert metadatos["configuracion"]["gail"] == cfg["gail"]
    assert set(metadatos["codigo"]) == {"commit", "cambios_sin_commit"}
    assert "torch" in metadatos["entorno_ejecucion"]


def test_procedencia_de_los_datos(tmp_path, monkeypatch):
    from src.data.demostraciones import huella_partidas
    from src.data.splits import huella_archivo
    from src.training.barrido import procedencia_datos
    trayectorias, _, ruta_npz, ruta_splits = _datos_en_disco(tmp_path)
    monkeypatch.chdir(tmp_path)
    datos = procedencia_datos(ruta_npz, ruta_splits, trayectorias)
    assert datos == {"demostraciones": "datos/demostraciones.npz", "splits": "datos/splits.json",
                     "huella_dataset": huella_archivo(ruta_npz),
                     "huella_splits": huella_archivo(ruta_splits),
                     "huella_partidas": huella_partidas(trayectorias)}
    # Fuera de la carpeta de trabajo solo se guarda el nombre: nunca una ruta absoluta.
    monkeypatch.chdir(tmp_path / "datos")
    fuera = procedencia_datos(ruta_npz, ruta_splits, trayectorias)
    assert (fuera["demostraciones"], fuera["splits"]) == ("demostraciones.npz", "splits.json")
    monkeypatch.chdir(RAIZ)
    lejos = procedencia_datos(ruta_npz, ruta_splits, trayectorias)
    assert not any(Path(lejos[c]).is_absolute() for c in ("demostraciones", "splits"))
    with pytest.raises(FileNotFoundError):
        procedencia_datos(tmp_path / "no_existe.npz", ruta_splits, trayectorias)
    with pytest.raises(ValueError, match="Trayectoria"):
        procedencia_datos(ruta_npz, ruta_splits, "partidas")


def test_gail_json_es_estricto_y_guarda_nan_como_null(tmp_path):
    from src.training.gail import guardar_gail
    agente, discriminador, cfg, resumen, datos = _entrenado(tmp_path)
    resumen = copy.deepcopy(resumen)
    resumen["historial"][0].update(retorno_real_medio=float("nan"),
                                   longitud_media=np.float32("inf"), partidas=np.int64(0))
    guardar_gail(tmp_path / "gail", agente, discriminador, cfg, resumen, datos)
    guardado = _json_estricto(tmp_path / "gail" / "gail.json")["resumen"]["historial"][0]
    assert guardado["retorno_real_medio"] is None and guardado["longitud_media"] is None
    assert guardado["partidas"] == 0 and isinstance(guardado["partidas"], int)


def test_guardar_gail_admite_carpeta_vacia_y_no_sobrescribe(tmp_path):
    from src.training.gail import guardar_gail
    agente, discriminador, cfg, resumen, datos = _entrenado(tmp_path)
    vacia = tmp_path / "vacia"
    vacia.mkdir()
    guardar_gail(vacia, agente, discriminador, cfg, resumen, datos)
    contenido = {p.name: p.read_bytes() for p in vacia.iterdir()}
    with pytest.raises(FileExistsError):
        guardar_gail(vacia, agente, discriminador, cfg, resumen, datos)
    assert {p.name: p.read_bytes() for p in vacia.iterdir()} == contenido
    archivo = tmp_path / "archivo"
    archivo.write_text("x")
    with pytest.raises(FileExistsError):
        guardar_gail(archivo, agente, discriminador, cfg, resumen, datos)
    # Una carpeta con cualquier otro archivo tampoco: no se mezclan resultados.
    ocupada = tmp_path / "ocupada"
    ocupada.mkdir()
    (ocupada / "notas.txt").write_text("x")
    with pytest.raises(FileExistsError):
        guardar_gail(ocupada, agente, discriminador, cfg, resumen, datos)
    assert [p.name for p in ocupada.iterdir()] == ["notas.txt"]


@pytest.mark.parametrize("caso,error,mensaje", [
    ("agente", ValueError, "TRPO"),
    ("discriminador", ValueError, "Discriminador"),
    ("datos_incompletos", ValueError, "procedencia_datos"),
    ("otros_datos", ValueError, "huella del NPZ"),
    ("otra_arquitectura", ValueError, "no corresponde a gail.discriminador"),
    # Regresión: la activación no tiene pesos y load_state_dict no la detectaba.
    ("otra_activacion", ValueError, "activación 'tanh'.*pide \\[8\\] y 'relu'"),
    ("sin_config_discriminador", ValueError, "no corresponde a gail.discriminador"),
    ("valor_no_json", TypeError, "no se puede guardar"),
])
def test_guardar_gail_lo_comprueba_todo_antes_de_crear_la_carpeta(tmp_path, caso, error,
                                                                  mensaje):
    from src.training.gail import guardar_gail
    agente, discriminador, cfg, resumen, datos = _entrenado(tmp_path)
    argumentos = {"agente": agente, "discriminador": discriminador, "cfg": cfg,
                  "resumen": resumen, "datos": datos}
    otra_cfg = copy.deepcopy(cfg)
    otra_cfg["gail"]["discriminador"]["capas_ocultas"] = [16]
    assert discriminador.activacion == "tanh"
    relu_cfg = copy.deepcopy(cfg)
    relu_cfg["gail"]["discriminador"]["activacion"] = "relu"
    sin_d = copy.deepcopy(cfg)
    del sin_d["gail"]["discriminador"]
    argumentos.update({
        "agente": {"agente": discriminador},
        "discriminador": {"discriminador": agente.policy},
        "datos_incompletos": {"datos": {"huella_dataset": datos["huella_dataset"]}},
        "otros_datos": {"datos": {**datos, "huella_dataset": "b" * 64}},
        "otra_arquitectura": {"cfg": otra_cfg},
        "otra_activacion": {"cfg": relu_cfg},
        "sin_config_discriminador": {"cfg": sin_d},
        "valor_no_json": {"resumen": {**resumen, "raro": object()}},
    }[caso])
    destino = tmp_path / "gail"
    with pytest.raises(error, match=mensaje):
        guardar_gail(destino, **argumentos)
    assert not destino.exists()


@pytest.mark.parametrize("archivo", ["discriminador.pt", "modelo.zip"])
def test_guardar_gail_no_deja_una_carpeta_a_medias(tmp_path, archivo, monkeypatch):
    """Regresión: si fallaba al escribir el modelo o el discriminador, quedaba una carpeta
    con gail.json que además impedía volver a guardar ahí."""
    from src.training.gail import cargar_gail, guardar_gail
    agente, discriminador, cfg, resumen, datos = _entrenado(tmp_path / "datos")
    destino = tmp_path / "gail"

    def falla(*a, **k):
        raise OSError("disco lleno")

    with monkeypatch.context() as parche:
        if archivo == "modelo.zip":
            # En la clase: un save en el propio agente se guardaría luego con él.
            parche.setattr(type(agente), "save", falla)
        else:
            parche.setattr("src.training.gail.torch.save", falla)
        with pytest.raises(OSError, match="disco lleno"):
            guardar_gail(destino, agente, discriminador, cfg, resumen, datos)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["datos"]  # ni carpeta ni temporal
    guardar_gail(destino, agente, discriminador, cfg, resumen, datos)
    cargar_gail(destino)


@pytest.mark.parametrize("activacion", ["tanh", "relu"])
def test_guardar_y_cargar_gail_conserva_la_activacion(tmp_path, activacion):
    """Lo guardado sale del propio discriminador: al cargarlo da las mismas salidas,
    sea cual sea la activación (todas las de ACTIVACIONES)."""
    from src.models.factory import ACTIVACIONES
    from src.training.gail import cargar_gail, guardar_gail
    assert set(ACTIVACIONES) == {"tanh", "relu"}
    cfg = _cfg_gail(iteraciones=2)
    cfg["gail"]["discriminador"]["activacion"] = activacion
    agente, discriminador, cfg, resumen, datos = _entrenado(tmp_path, cfg)
    guardar_gail(tmp_path / "gail", agente, discriminador, cfg, resumen, datos)
    _, cargado, metadatos = cargar_gail(tmp_path / "gail")
    assert metadatos["discriminador"]["activacion"] == activacion
    assert cargado.activacion == activacion and cargado.capas_ocultas == (8,)
    # Entradas lejos de 0, donde tanh y relu dan valores distintos.
    estados = torch.as_tensor(crear_generador(4).normal(scale=3.0, size=(20, 4)),
                              dtype=torch.float32)
    acciones = torch.zeros(20, dtype=torch.int64)
    with torch.no_grad():
        assert torch.equal(cargado(estados, acciones), discriminador.eval()(estados, acciones))


def test_la_red_guarda_su_arquitectura_sin_cambiar_el_state_dict():
    from src.models.factory import PoliticaBC
    discriminador = Discriminador(3, 2, False, [8, 4], "relu")
    politica = PoliticaBC(3, 2, False, [8, 4], "relu")
    for red in (discriminador, politica):
        assert (red.capas_ocultas, red.activacion) == ((8, 4), "relu")
        # Son atributos, no buffers: los pesos guardados (y los de BC) no cambian.
        assert not any("capas" in clave or "activacion" in clave for clave in red.state_dict())


@pytest.mark.parametrize("falta", ["modelo.zip", "discriminador.pt", "gail.json"])
def test_cargar_gail_exige_los_tres_archivos(tmp_path, falta):
    from src.training.gail import cargar_gail, guardar_gail
    agente, discriminador, cfg, resumen, datos = _entrenado(tmp_path)
    guardar_gail(tmp_path / "gail", agente, discriminador, cfg, resumen, datos)
    (tmp_path / "gail" / falta).unlink()
    with pytest.raises(FileNotFoundError, match=falta):
        cargar_gail(tmp_path / "gail")


def test_cargar_gail_rechaza_un_discriminador_que_no_corresponde(tmp_path):
    from src.training.gail import cargar_gail, guardar_gail
    agente, discriminador, cfg, resumen, datos = _entrenado(tmp_path)
    guardar_gail(tmp_path / "gail", agente, discriminador, cfg, resumen, datos)
    otro = Discriminador(4, 2, False, [16], "tanh")
    torch.save(otro.state_dict(), tmp_path / "gail" / "discriminador.pt")
    with pytest.raises(ValueError, match="no corresponde a gail.json"):
        cargar_gail(tmp_path / "gail")


def test_cargar_gail_solo_acepta_tensores(tmp_path):
    """weights_only=True: un archivo con objetos de Python no se ejecuta al cargar."""
    import pickle

    from src.training.gail import cargar_gail, guardar_gail
    agente, discriminador, cfg, resumen, datos = _entrenado(tmp_path)
    guardar_gail(tmp_path / "gail", agente, discriminador, cfg, resumen, datos)
    torch.save({"objeto": _RunFalso()}, tmp_path / "gail" / "discriminador.pt")
    with pytest.raises(pickle.UnpicklingError):
        cargar_gail(tmp_path / "gail")


def test_guardar_gail_con_normalizacion_conserva_sus_estadisticas(tmp_path):
    from src.training.gail import cargar_gail, guardar_gail
    from src.training.trpo import politica_trpo
    cfg = _cfg_gail(iteraciones=2)
    cfg["trpo"]["normalizar_observaciones"] = True
    agente, discriminador, cfg, resumen, datos = _entrenado(tmp_path, cfg)
    guardar_gail(tmp_path / "gail", agente, discriminador, cfg, resumen, datos)
    cargado, _, metadatos = cargar_gail(tmp_path / "gail")
    estadisticas = metadatos["resumen"]["normalizacion"]
    assert estadisticas == resumen["normalizacion"] and estadisticas is not None
    # Con las estadísticas guardadas, el modelo cargado juega igual que el original.
    original = politica_trpo(agente, True, resumen["normalizacion"])
    recuperada = politica_trpo(cargado, True, estadisticas)
    for observacion in crear_generador(4).normal(scale=0.1, size=(10, 4)).astype(np.float32):
        assert np.array_equal(original(observacion), recuperada(observacion))


# --- scripts/entrenar_gail.py -------------------------------------------------------
# Datos reales de CartPole grabados con una política fija (generar_dataset) y la
# config reducida: el script se prueba de punta a punta sin experto.

def _script_gail():
    """Carga el script como módulo (scripts/ no es un paquete)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("entrenar_gail",
                                                  RAIZ / "scripts" / "entrenar_gail.py")
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def _cfg_script() -> dict:
    cfg = _cfg_gail(iteraciones=2)
    cfg["demostraciones"].update(partidas_train=4, partidas_validacion=2, subconjuntos=[1, 4])
    cfg["experimento"]["repeticiones"] = 2
    # En serie salvo que el test diga otra cosa: no depende del valor de la máquina en la base.
    cfg["experimento"]["procesos"] = 1
    cfg["wandb"]["modo"] = "disabled"
    return cfg


@pytest.fixture(scope="module")
def datos_grabados(tmp_path_factory):
    from src.data.demostraciones import generar_dataset
    carpeta = tmp_path_factory.mktemp("datos") / "cartpole"
    generar_dataset(_cfg_script(), lambda observacion: np.int64(observacion[2] > 0), carpeta,
                    {"entorno": "CartPole-v1", "experto": "política fija de prueba",
                     "procedencia": "tests", "configuracion_experto": {}})
    return carpeta


def _preparar_script(monkeypatch, argumentos: list[str], cfg: dict | None = None):
    script = _script_gail()
    resueltas = []

    def resolver(base, entorno, ajustes=None):
        resueltas.append((base, entorno, ajustes))
        return copy.deepcopy(cfg or _cfg_script())

    monkeypatch.setattr("src.training.barrido.resolver_configuracion", resolver)
    monkeypatch.setattr("sys.argv", ["entrenar_gail.py", *argumentos])
    return script, resueltas


def test_el_script_entrena_y_guarda_cada_cantidad_y_repeticion(datos_grabados, tmp_path,
                                                               monkeypatch, capsys):
    from src.training.gail import cargar_gail
    salida = tmp_path / "gail"
    script, resueltas = _preparar_script(monkeypatch, [
        "--datos", str(datos_grabados), "--cantidades", "1", "4", "--repeticiones", "1",
        "--salida", str(salida), "--sin-wandb"])
    assert script.main() == 0
    assert resueltas[0][1:] == ("cartpole", {"wandb": {"modo": "disabled"}})
    for cantidad in (1, 4):
        _, _, metadatos = cargar_gail(salida / f"cantidad_{cantidad}" / "repeticion_1")
        assert (metadatos["resumen"]["cantidad"], metadatos["resumen"]["repeticion"]) == \
            (cantidad, 1)
        assert metadatos["datos"]["huella_dataset"] == metadatos["resumen"]["huella_dataset"]
    assert sorted(p.name for p in salida.iterdir()) == ["cantidad_1", "cantidad_4"]
    salida_texto = capsys.readouterr().out
    assert "cantidad 1, repetición 1: retorno de validación" in salida_texto
    # Progreso durante el entrenamiento: una línea por validación (2 iteraciones, cada 2).
    assert salida_texto.count("cantidad 4, repetición 1 · iteración 2/2 · 128 pasos") == 1
    assert "2 entrenamientos guardados" in salida_texto


def _resumen_falso(*_):
    return "agente", "d", {"validaciones": [{"retorno_medio": 1.0, "iteracion": 2}],
                           "pasos": 1, "segundos": 1.0}


def test_el_script_entrena_por_defecto_todo_el_barrido(datos_grabados, tmp_path, monkeypatch):
    llamadas = []
    script, _ = _preparar_script(monkeypatch, ["--datos", str(datos_grabados),
                                               "--salida", str(tmp_path / "gail")])
    monkeypatch.setattr("src.training.gail.entrenar_gail", lambda cfg, t, s, c, r, run:
                        llamadas.append((c, r, run.run)) or _resumen_falso())
    guardados = []
    monkeypatch.setattr("src.training.gail.guardar_gail",
                        lambda destino, *a: guardados.append(destino))
    assert script.main() == 0
    # demostraciones.subconjuntos × experimento.repeticiones, sin W&B (modo disabled).
    assert llamadas == [(1, 0, None), (1, 1, None), (4, 0, None), (4, 1, None)]
    assert guardados == [tmp_path / "gail" / f"cantidad_{c}" / f"repeticion_{r}"
                         for c in (1, 4) for r in (0, 1)]


def test_el_barrido_por_defecto_es_el_de_la_configuracion(datos_grabados, tmp_path,
                                                          monkeypatch):
    """Regresión: sin --cantidades se entrenaban todos los subconjuntos de los splits
    ([1, 4]) aunque la configuración pidiera menos."""
    cfg = _cfg_script()
    cfg["demostraciones"]["subconjuntos"] = [1]
    llamadas = []
    script, _ = _preparar_script(monkeypatch, ["--datos", str(datos_grabados), "--salida",
                                               str(tmp_path / "gail")], cfg)
    monkeypatch.setattr("src.training.gail.entrenar_gail", lambda cfg, t, s, c, r, run:
                        llamadas.append((c, r)) or _resumen_falso())
    monkeypatch.setattr("src.training.gail.guardar_gail", lambda *a: None)
    assert script.main() == 0
    assert llamadas == [(1, 0), (1, 1)]


def test_el_barrido_de_la_configuracion_debe_estar_en_los_splits(datos_grabados, tmp_path,
                                                                 monkeypatch, capsys):
    cfg = _cfg_script()
    cfg["demostraciones"]["subconjuntos"] = [1, 3]  # los splits solo tienen 1 y 4
    script, _ = _preparar_script(monkeypatch, ["--datos", str(datos_grabados), "--salida",
                                               str(tmp_path / "gail")], cfg)
    entrenados = []
    monkeypatch.setattr("src.training.gail.entrenar_gail", lambda *a: entrenados.append(1))
    assert script.main() == 2 and not entrenados
    assert "los splits tienen subconjuntos de [1, 4]" in capsys.readouterr().err


def test_el_script_abre_un_run_por_entrenamiento(datos_grabados, tmp_path, monkeypatch):
    cfg = _cfg_script()
    cfg["wandb"]["modo"] = "offline"
    script, _ = _preparar_script(monkeypatch, ["--datos", str(datos_grabados), "--cantidades",
                                               "4", "--salida", str(tmp_path / "g")], cfg)
    abiertos, recibidos = [], []

    class _Run(_RunFalso):
        def finish(self, exit_code=None) -> None:
            self.terminado = exit_code

    def abrir(cfg_run):
        abiertos.append((cfg_run["experimento"]["nombre"], _Run()))
        return abiertos[-1][1]

    monkeypatch.setattr("src.utils.tracking.abrir_run", abrir)
    monkeypatch.setattr("src.training.gail.entrenar_gail", lambda cfg, t, s, c, r, run:
                        recibidos.append((cfg, run)) or _resumen_falso())
    monkeypatch.setattr("src.training.gail.guardar_gail", lambda *a: None)
    assert script.main() == 0
    assert [nombre for nombre, _ in abiertos] == [f"{cfg['experimento']['nombre']}-gail-"
                                                 f"cartpole-c4-r{r}" for r in (0, 1)]
    # entrenar_gail recibe el progreso, que reenvía al run de W&B.
    assert [progreso.run for _, progreso in recibidos] == [run for _, run in abiertos]
    assert all(run.terminado == 0 for _, run in abiertos)  # cerrados como correctos
    # El nombre del run no cambia la configuración con la que se entrena y se guarda.
    assert all(c["experimento"]["nombre"] == cfg["experimento"]["nombre"] for c, _ in recibidos)


class _RunCerrable(_RunFalso):
    def __init__(self) -> None:
        super().__init__()
        self.cierres: list = []

    def finish(self, exit_code=None) -> None:
        self.cierres.append(exit_code)


def test_terminar_run_da_el_codigo_de_salida_de_cada_caso():
    from src.utils.tracking import terminar_run
    terminar_run(None)  # sin run no hace nada
    terminar_run(None, ValueError("x"))
    run = _RunCerrable()
    for error in (None, ValueError("x"), FileExistsError("x"), KeyboardInterrupt()):
        terminar_run(run, error)
    # W&B: 0 es Finished y cualquier otro, Failed; 130 como el Ctrl+C de los scripts.
    assert run.cierres == [0, 1, 1, 130]


@pytest.mark.parametrize("fallo,error,codigo", [
    (None, None, 0),
    ("entrenar", RuntimeError, 1),
    ("guardar", FileExistsError, 1),
    ("entrenar", KeyboardInterrupt, 130),
])
def test_un_entrenamiento_fallido_se_cierra_como_fallido_en_wandb(fallo, error, codigo,
                                                                  monkeypatch):
    """Regresión: finish() sin código marcaba como correctos los entrenamientos fallidos."""
    from src.training.gail import entrenar_y_guardar_gail
    run = _RunCerrable()
    monkeypatch.setattr("src.utils.tracking.abrir_run", lambda cfg: run)

    def entrenar(*_):
        if fallo == "entrenar":
            raise error("fallo de prueba")
        return _resumen_falso()

    def guardar(*_):
        if fallo == "guardar":
            raise error("fallo de prueba")

    monkeypatch.setattr("src.training.gail.entrenar_gail", entrenar)
    monkeypatch.setattr("src.training.gail.guardar_gail", guardar)
    argumentos = (_cfg_script(), [], None, 4, 0, Path("destino"), {})
    if error is None:
        entrenar_y_guardar_gail(*argumentos)
    else:
        with pytest.raises(error):  # el error sigue propagándose
            entrenar_y_guardar_gail(*argumentos)
    assert run.cierres == [codigo]  # se cierra una sola vez


@pytest.mark.parametrize("caso,mensaje", [
    ("salida_ocupada", "elige otra --salida"),
    ("sin_datos", "error con los datos"),
    ("splits_de_otros_datos", "error con los datos"),
    ("cantidad", "cantidades no válidas"),
    ("cantidad_repetida", "cantidades no válidas"),
    ("repeticion", "repeticiones no válidas"),
    ("repeticion_negativa", "repeticiones no válidas"),
    ("repeticion_repetida", "repeticiones no válidas"),
])
def test_el_script_no_entrena_si_algo_no_es_valido(caso, mensaje, datos_grabados, tmp_path,
                                                   monkeypatch, capsys):
    import shutil
    datos, salida = datos_grabados, tmp_path / "gail"
    extra = []
    if caso == "salida_ocupada":
        salida.mkdir()
        (salida / "notas.txt").write_text("x")
    elif caso == "sin_datos":
        datos = tmp_path / "no_existe"
    elif caso == "splits_de_otros_datos":
        datos = tmp_path / "copia"
        shutil.copytree(datos_grabados, datos)
        with (datos / "demostraciones.npz").open("ab") as archivo:
            archivo.write(b"\0")
    else:
        extra = {"cantidad": ["--cantidades", "2"],
                 "cantidad_repetida": ["--cantidades", "4", "4"],
                 "repeticion": ["--repeticiones", "2"],
                 "repeticion_negativa": ["--repeticiones", "-1"],
                 "repeticion_repetida": ["--repeticiones", "0", "0"]}[caso]
    script, _ = _preparar_script(monkeypatch, ["--datos", str(datos), "--salida", str(salida),
                                               *extra])
    entrenados = []
    monkeypatch.setattr("src.training.gail.entrenar_gail", lambda *a: entrenados.append(1))
    assert script.main() == 2
    assert not entrenados
    assert mensaje in capsys.readouterr().err


def test_el_script_informa_de_una_configuracion_no_valida(monkeypatch, tmp_path, capsys):
    script = _script_gail()
    monkeypatch.setattr("sys.argv", ["entrenar_gail.py", "--datos", str(tmp_path),
                                     "--config", str(tmp_path / "no_existe.yaml")])
    assert script.main() == 2
    assert "error de configuración" in capsys.readouterr().err


@pytest.mark.parametrize("codigo,aviso", [
    ({"commit": "a" * 40, "cambios_sin_commit": False}, None),
    ({"commit": "a" * 40, "cambios_sin_commit": True}, "cambios sin commit"),
    ({"commit": None, "cambios_sin_commit": None}, "no se pudo leer el commit"),
])
def test_el_script_avisa_del_estado_de_git(codigo, aviso, datos_grabados, tmp_path,
                                           monkeypatch, capsys):
    script, _ = _preparar_script(monkeypatch, ["--datos", str(datos_grabados), "--cantidades",
                                               "1", "--repeticiones", "0", "--salida",
                                               str(tmp_path / "g")])
    monkeypatch.setattr("src.training.gail.entrenar_gail", _resumen_falso)
    monkeypatch.setattr("src.training.gail.guardar_gail", lambda *a: None)
    monkeypatch.setattr("src.training.barrido.estado_git", lambda: codigo)
    assert script.main() == 0
    error = capsys.readouterr().err
    assert (aviso in error) if aviso else "aviso" not in error


def test_el_progreso_reenvia_al_run_e_imprime_una_linea_por_validacion(capsys):
    from src.utils.tracking import RegistroProgreso
    run = _RunFalso()
    progreso = RegistroProgreso(run, "cantidad 4, repetición 0", 5000, 300)
    entrenamiento = {"pasos": 50000.0, "discriminador/acierto_agente": 0.61,
                     "discriminador/acierto_experto": 0.58}
    validacion = {"validacion/retorno_medio": 23.4, "validacion/retorno_desviacion": 1.0,
                  "validacion/longitud_media": 23.4}
    progreso.log(entrenamiento, step=50000)
    assert capsys.readouterr().out == ""
    progreso.log(validacion, step=50000)
    linea = capsys.readouterr().out
    assert run.registros == [(entrenamiento, 50000), (validacion, 50000)]
    assert linea.startswith("cantidad 4, repetición 0 · iteración 10/300 · 50000 pasos · "
                            "validación 23.4 · D acierta 0.61 (agente) 0.58 (experto) · ")
    assert "quedan ~" in linea and linea.count("\n") == 1


def test_el_progreso_sin_run_solo_imprime(capsys):
    from src.utils.tracking import RegistroProgreso
    progreso = RegistroProgreso(None, "c1 r0", 64, 3)
    progreso.log({"pasos": 64.0}, step=64)
    progreso.log({"validacion/retorno_medio": 9.0}, step=192)
    assert "c1 r0 · iteración 3/3 · 192 pasos · validación 9.0" in capsys.readouterr().out


def test_el_script_rechaza_procesos_no_validos(datos_grabados, tmp_path, monkeypatch, capsys):
    for procesos in (0, -1, 1.5, True, None):
        cfg = _cfg_script()
        cfg["experimento"]["procesos"] = procesos
        script, _ = _preparar_script(monkeypatch, ["--datos", str(datos_grabados), "--salida",
                                                   str(tmp_path / "g")], cfg)
        entrenados = []
        monkeypatch.setattr("src.training.gail.entrenar_gail", lambda *a: entrenados.append(1))
        assert script.main() == 2 and not entrenados
        assert "experimento.procesos" in capsys.readouterr().err


@pytest.mark.parametrize("seccion,clave,valor,mensaje", [
    # Regresión: con 0 o −1 se lanzaba un barrido de cero procesos, con texto saltaba
    # TypeError y con true se entrenaba una repetición.
    ("experimento", "repeticiones", 0, "experimento.repeticiones"),
    ("experimento", "repeticiones", -1, "experimento.repeticiones"),
    ("experimento", "repeticiones", "dos", "experimento.repeticiones"),
    ("experimento", "repeticiones", True, "experimento.repeticiones"),
    ("experimento", "repeticiones", 2.0, "experimento.repeticiones"),
    ("experimento", "semilla_repeticiones", -1, "experimento.semilla_repeticiones"),
    ("experimento", "semilla_repeticiones", None, "experimento.semilla_repeticiones"),
    # Regresión: estos fallaban dentro de cada entrenamiento y el script salía con 1.
    ("gail", "evaluar_cada", 0, "gail.evaluar_cada"),
    ("gail", "iteraciones", None, "gail.iteraciones"),
    ("gail", "recompensa", "otra", "gail.recompensa"),
    ("demostraciones", "frecuencia_pares", 0, "demostraciones.frecuencia_pares"),
    ("demostraciones", "frecuencia_pares", None, "demostraciones.frecuencia_pares"),
    ("demostraciones", "subconjuntos", [4, 1], "subconjuntos"),
    ("trpo", "n_entornos", 3, "divisible por trpo.n_entornos"),
])
def test_el_script_rechaza_la_configuracion_antes_de_entrenar(seccion, clave, valor, mensaje,
                                                              datos_grabados, tmp_path,
                                                              monkeypatch, capsys):
    cfg = _cfg_script()
    cfg[seccion][clave] = valor
    script, _ = _preparar_script(monkeypatch, ["--datos", str(datos_grabados), "--salida",
                                               str(tmp_path / "g")], cfg)
    entrenados = []
    monkeypatch.setattr("src.training.gail.entrenar_gail", lambda *a: entrenados.append(1))
    assert script.main() == 2 and not entrenados
    error = capsys.readouterr().err
    assert "error de configuración" in error and mensaje in error
    assert not (tmp_path / "g").exists()


@pytest.mark.parametrize("cambio,mensaje", [
    # Regresión: la arquitectura del discriminador no se comprobaba antes del barrido y
    # cada entrenamiento fallaba por separado (salida 1 en lugar de 2).
    ({"activacion": "sigmoide"}, "activacion desconocida"),
    ({"activacion": None}, "activacion desconocida"),
    ({"capas_ocultas": [-1]}, "capas_ocultas"),
    ({"capas_ocultas": [8, 0]}, "capas_ocultas"),
    ({"capas_ocultas": [8.5]}, "capas_ocultas"),
    ({"capas_ocultas": "100"}, "capas_ocultas"),
    ({"capas_ocultas": None}, "capas_ocultas"),
])
def test_el_script_rechaza_una_arquitectura_del_discriminador_no_valida(
        cambio, mensaje, datos_grabados, tmp_path, monkeypatch, capsys):
    cfg = _cfg_script()
    cfg["gail"]["discriminador"].update(cambio)
    script, _ = _preparar_script(monkeypatch, ["--datos", str(datos_grabados), "--salida",
                                               str(tmp_path / "g")], cfg)
    entrenados = []
    monkeypatch.setattr("src.training.gail.entrenar_gail", lambda *a: entrenados.append(1))
    assert script.main() == 2 and not entrenados
    error = capsys.readouterr().err
    assert "gail.discriminador no es válido" in error and mensaje in error


@pytest.mark.parametrize("clave", ["capas_ocultas", "activacion"])
def test_la_config_del_discriminador_exige_su_arquitectura(clave):
    from src.training.gail import comprobar_config_barrido
    cfg = _cfg_script()
    del cfg["gail"]["discriminador"][clave]
    with pytest.raises(ValueError, match=f"gail.discriminador con .*{clave}"):
        comprobar_config_barrido(cfg)


def test_comprobar_config_barrido_acepta_la_del_proyecto_y_enteros_de_numpy():
    from src.training.gail import comprobar_config_barrido
    comprobar_config_barrido(resolver_configuracion(RAIZ / "configs" / "default.yaml",
                                                    "cartpole"))
    cfg = _cfg_script()
    cfg["experimento"].update(repeticiones=np.int64(2), procesos=np.int32(1),
                              semilla_repeticiones=np.int64(7))
    cfg["demostraciones"]["frecuencia_pares"] = np.int64(2)
    comprobar_config_barrido(cfg)
    with pytest.raises(ValueError, match="sección experimento"):
        comprobar_config_barrido({**_cfg_script(), "experimento": None})


class _FalloPrevisto(RuntimeError):
    pass


def _entrenar_o_fallar(cfg, trayectorias, splits, cantidad, repeticion, run):
    if (cantidad, repeticion) == (4, 0):
        raise _FalloPrevisto("fallo de prueba")
    return _resumen_falso()


def test_si_un_entrenamiento_falla_los_demas_siguen(datos_grabados, tmp_path, monkeypatch,
                                                    capsys):
    script, _ = _preparar_script(monkeypatch, ["--datos", str(datos_grabados), "--salida",
                                               str(tmp_path / "g")])
    guardados = []
    monkeypatch.setattr("src.training.gail.entrenar_gail", _entrenar_o_fallar)
    monkeypatch.setattr("src.training.gail.guardar_gail",
                        lambda destino, *a: guardados.append(destino.parts[-2:]))
    assert script.main() == 1
    assert guardados == [("cantidad_1", "repeticion_0"), ("cantidad_1", "repeticion_1"),
                         ("cantidad_4", "repeticion_1")]
    salida = capsys.readouterr()
    assert "error en cantidad 4, repetición 0" in salida.err and "fallo de prueba" in salida.err
    assert "3 de 4 entrenamientos guardados" in salida.out


class _ContextoFalso:
    """Sustituye al contexto spawn: cada «proceso» es un hilo, para poder sustituir el
    entrenamiento. El Pipe es real y el hilo recibe su propia copia del extremo de
    escritura, como un proceso nuevo, así que el script puede cerrar la suya."""

    def __init__(self, registro: dict) -> None:
        self.registro = registro
        registro.update(lanzados=0, vivos=0, maximo=0, terminados=0, objetivos=set())

    def Pipe(self, duplex):  # noqa: N802 - mismo nombre que multiprocessing
        import multiprocessing
        return multiprocessing.Pipe(duplex)

    def Process(self, target, args):  # noqa: N802 - mismo nombre que multiprocessing
        import os
        import threading
        from multiprocessing.connection import Connection
        registro = self.registro

        class _Hilo:
            exitcode = 0
            unido = False

            def start(self) -> None:
                registro["lanzados"] += 1
                registro["vivos"] += 1
                registro["maximo"] = max(registro["maximo"], registro["vivos"])
                registro["objetivos"].add(target)
                escritura, *resto = args
                copia = Connection(os.dup(escritura.fileno()), readable=False)
                self.hilo = threading.Thread(target=target, args=(copia, *resto))
                self.hilo.start()

            def terminate(self) -> None:
                registro["terminados"] += 1

            def join(self) -> None:
                self.hilo.join()
                if not self.unido:  # el script puede esperar dos veces al mismo
                    self.unido = True
                    registro["vivos"] -= 1

        return _Hilo()


def _con_procesos_falsos(monkeypatch, script) -> dict:
    import src.training.barrido as barrido
    registro: dict = {"preparados": 0}
    original = barrido.multiprocessing.get_context

    def get_context(metodo=None):
        if metodo is None:
            return original()
        registro["metodo"] = metodo
        return _ContextoFalso(registro)

    def preparar() -> None:
        # La de verdad toca las señales, que en un hilo no se puede.
        registro["preparados"] += 1

    monkeypatch.setattr(barrido.multiprocessing, "get_context", get_context)
    monkeypatch.setattr("src.training.barrido.preparar_proceso_barrido", preparar)
    return registro


def test_con_varios_procesos_reparte_las_tareas(datos_grabados, tmp_path, monkeypatch, capsys):
    from src.training.barrido import ejecutar_tarea_barrido
    cfg = _cfg_script()
    cfg["experimento"]["procesos"] = 3
    script, _ = _preparar_script(monkeypatch, ["--datos", str(datos_grabados), "--salida",
                                               str(tmp_path / "g")], cfg)
    registro = _con_procesos_falsos(monkeypatch, script)
    guardados = []
    monkeypatch.setattr("src.training.gail.entrenar_gail", _entrenar_o_fallar)
    monkeypatch.setattr("src.training.gail.guardar_gail",
                        lambda destino, *a: guardados.append(destino.parts[-2:]))
    assert script.main() == 1
    # 4 tareas, cada una en un proceso nuevo (spawn), como mucho 3 a la vez, todos
    # preparados con un hilo y sin Ctrl+C y esperados al terminar.
    assert registro["metodo"] == "spawn"
    assert (registro["lanzados"], registro["maximo"]) == (4, 3)
    assert registro["objetivos"] == {ejecutar_tarea_barrido}
    assert registro["preparados"] == 4
    assert (registro["vivos"], registro["terminados"]) == (0, 0)
    assert sorted(guardados) == [("cantidad_1", "repeticion_0"), ("cantidad_1", "repeticion_1"),
                                 ("cantidad_4", "repeticion_1")]
    salida = capsys.readouterr()
    assert "error en cantidad 4, repetición 0" in salida.err
    assert "3 de 4 entrenamientos guardados" in salida.out


def test_no_lanza_mas_procesos_que_entrenamientos(datos_grabados, tmp_path, monkeypatch):
    cfg = _cfg_script()
    cfg["experimento"]["procesos"] = 8
    script, _ = _preparar_script(monkeypatch, ["--datos", str(datos_grabados), "--cantidades",
                                               "4", "--salida", str(tmp_path / "g")], cfg)
    registro = _con_procesos_falsos(monkeypatch, script)
    monkeypatch.setattr("src.training.gail.entrenar_gail", _resumen_falso)
    monkeypatch.setattr("src.training.gail.guardar_gail", lambda *a: None)
    assert script.main() == 0
    assert (registro["lanzados"], registro["maximo"]) == (2, 2)  # 1 cantidad × 2 repeticiones


def test_un_envio_truncado_cuenta_como_fallo_y_el_resto_sigue(datos_grabados, tmp_path,
                                                              monkeypatch, capsys):
    """Regresión: si un proceso moría a mitad del envío, recv() daba OSError (no
    EOFError), que abortaba el barrido y terminaba los demás entrenamientos."""
    import os
    import struct
    from src.training.barrido import ejecutar_tarea_barrido
    cfg = _cfg_script()
    cfg["experimento"]["procesos"] = 2
    script, _ = _preparar_script(monkeypatch, ["--datos", str(datos_grabados), "--salida",
                                               str(tmp_path / "g")], cfg)
    _con_procesos_falsos(monkeypatch, script)
    monkeypatch.setattr("src.training.gail.entrenar_gail", _resumen_falso)
    monkeypatch.setattr("src.training.gail.guardar_gail", lambda *a: None)

    def ejecutar_o_truncar(conexion, tarea, argumentos):
        if tuple(argumentos[3:5]) != (4, 0):
            return ejecutar_tarea_barrido(conexion, tarea, argumentos)
        # Cabecera de un mensaje de 1000 bytes y solo 10: como si muriera enviando.
        os.write(conexion.fileno(), struct.pack("!i", 1000) + b"x" * 10)
        conexion.close()

    monkeypatch.setattr("src.training.barrido.ejecutar_tarea_barrido", ejecutar_o_truncar)
    assert script.main() == 1
    salida = capsys.readouterr()
    assert "error en cantidad 4, repetición 0: el proceso terminó sin resultado" in salida.err
    assert "OSError" in salida.err
    assert "3 de 4 entrenamientos guardados" in salida.out


def test_ctrl_c_mientras_se_recibe_tambien_termina_ese_proceso(datos_grabados, tmp_path,
                                                                monkeypatch, capsys):
    """Regresión: el proceso que se estaba recibiendo salía de la lista antes de recibir,
    y un Ctrl+C en ese momento lo dejaba fuera de la limpieza, vivo."""
    from multiprocessing.connection import Connection
    cfg = _cfg_script()
    cfg["experimento"]["procesos"] = 3
    script, _ = _preparar_script(monkeypatch, ["--datos", str(datos_grabados), "--salida",
                                               str(tmp_path / "g")], cfg)
    registro = _con_procesos_falsos(monkeypatch, script)
    monkeypatch.setattr("src.training.gail.entrenar_gail", _resumen_falso)
    monkeypatch.setattr("src.training.gail.guardar_gail", lambda *a: None)

    def interrumpir(self):
        raise KeyboardInterrupt

    monkeypatch.setattr(Connection, "recv", interrumpir)
    assert script.main() == 130
    # Los 3 en marcha, también el que se estaba recibiendo, se terminan y se esperan.
    assert (registro["lanzados"], registro["terminados"], registro["vivos"]) == (3, 3, 0)


def _morir_en_4_0(cfg, trayectorias, splits, cantidad, repeticion, destino, datos):
    """Tarea de prueba para procesos de verdad: la (4, 0) muere como la mata el sistema."""
    import os
    import signal
    if (cantidad, repeticion) == (4, 0):
        os.kill(os.getpid(), signal.SIGKILL)
    return _resumen_falso()[2]


def test_si_un_proceso_muere_el_barrido_no_se_queda_esperando(tmp_path):
    """Regresión: con Pool, un proceso muerto (por ejemplo por falta de memoria) dejaba el
    barrido esperando para siempre. Procesos spawn de verdad; la alarma corta el test si
    vuelve a colgarse."""
    import signal

    from src.training.barrido import ejecutar_barrido

    def colgado(*_):
        raise TimeoutError("el barrido se ha quedado esperando")

    resultados = {}
    previa = signal.signal(signal.SIGALRM, colgado)
    signal.alarm(120)
    try:
        fallos = ejecutar_barrido([(1, 0), (4, 0), (4, 1)], 2, _morir_en_4_0,
                                  lambda c, r: ({}, [], None, c, r, tmp_path, {}),
                                  lambda resultado: resultados.update(
                                      {tuple(resultado[:2]): resultado[2:]}))
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previa)
    assert fallos == [(4, 0)]
    resumen, error = resultados[(4, 0)]
    assert resumen is None and "el proceso terminó sin resultado" in error
    assert f"código de salida {-signal.SIGKILL}" in error
    # Los otros dos terminan y se informan.
    assert resultados[(1, 0)] == (_resumen_falso()[2], None)
    assert resultados[(4, 1)] == (_resumen_falso()[2], None)


def test_ctrl_c_en_serie_para_el_barrido(datos_grabados, tmp_path, monkeypatch, capsys):
    script, _ = _preparar_script(monkeypatch, ["--datos", str(datos_grabados), "--salida",
                                               str(tmp_path / "g")])
    empezados = []

    def interrumpir(*argumentos):
        empezados.append(1)
        raise KeyboardInterrupt

    monkeypatch.setattr("src.training.gail.entrenar_gail", interrumpir)
    antes = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        assert script.main() == 130
        assert empezados == [1]  # no empieza ningún otro
        assert torch.get_num_threads() == 2  # también se restaura al interrumpir
    finally:
        torch.set_num_threads(antes)
    assert "interrumpido" in capsys.readouterr().err


def test_ctrl_c_en_paralelo_termina_los_procesos(datos_grabados, tmp_path, monkeypatch, capsys):
    """Regresión: con ProcessPoolExecutor, Ctrl+C esperaba a que terminase todo el barrido."""
    cfg = _cfg_script()
    cfg["experimento"]["procesos"] = 3
    script, _ = _preparar_script(monkeypatch, ["--datos", str(datos_grabados), "--salida",
                                               str(tmp_path / "g")], cfg)
    registro = _con_procesos_falsos(monkeypatch, script)
    monkeypatch.setattr("src.training.gail.entrenar_gail", _resumen_falso)
    monkeypatch.setattr("src.training.gail.guardar_gail", lambda *a: None)

    def interrumpir(conexiones):
        # Ctrl+C llega al proceso principal mientras espera resultados.
        raise KeyboardInterrupt

    monkeypatch.setattr("src.training.barrido.wait", interrumpir)
    assert script.main() == 130
    # Los 3 que estaban en marcha se terminan y se esperan; el cuarto no empieza.
    assert (registro["lanzados"], registro["terminados"], registro["vivos"]) == (3, 3, 0)
    assert "interrumpido" in capsys.readouterr().err


def test_preparar_proceso_barrido_deja_un_hilo_y_sin_ctrl_c():
    """En un proceso spawn de verdad, como los del script."""
    import multiprocessing
    import signal
    from src.training.barrido import preparar_proceso_barrido
    with multiprocessing.get_context("spawn").Pool(
            1, initializer=preparar_proceso_barrido) as pool:
        assert pool.apply(torch.get_num_threads) == 1
        assert pool.apply(signal.getsignal, (signal.SIGINT,)) == signal.SIG_IGN


def test_cada_tarea_devuelve_el_error_sin_propagarlo():
    from src.training.barrido import _ejecutar

    def fallar(*argumentos):
        raise ValueError("datos raros")

    assert _ejecutar(fallar, ("cfg", 4, 2)) == (None, "ValueError('datos raros')")
    assert _ejecutar(lambda *a: {"pasos": len(a)}, ("cfg", 4, 2)) == ({"pasos": 3}, None)


def test_en_paralelo_da_los_mismos_modelos_que_en_serie(datos_grabados, tmp_path, monkeypatch,
                                                         capfd):
    """Integración con procesos de verdad (spawn): cada entrenamiento solo depende de sus
    argumentos, así que repartirlos no cambia nada."""
    from src.training.gail import cargar_gail
    for procesos, carpeta in ((1, "serie"), (2, "paralelo")):
        cfg = _cfg_script()
        cfg["experimento"]["procesos"] = procesos
        # Redes del tamaño real: con las pequeñas no se notaba el número de hilos.
        cfg["trpo"]["capas_ocultas"] = [100, 100]
        cfg["gail"]["discriminador"]["capas_ocultas"] = [100, 100]
        script, _ = _preparar_script(monkeypatch, [
            "--datos", str(datos_grabados), "--cantidades", "1", "4", "--repeticiones", "1",
            "--salida", str(tmp_path / carpeta)], cfg)
        assert script.main() == 0
    # capfd: los procesos escriben directamente en la salida estándar, no en sys.stdout.
    salida = capfd.readouterr().out
    assert salida.count("2 de 2 entrenamientos guardados") == 2
    # El progreso también se imprime desde los procesos.
    assert salida.count("cantidad 4, repetición 1 · iteración 2/2") == 2
    for cantidad in (1, 4):
        ruta = Path("cantidad_" + str(cantidad)) / "repeticion_1"
        agente_s, d_s, meta_s = cargar_gail(tmp_path / "serie" / ruta)
        agente_p, d_p, meta_p = cargar_gail(tmp_path / "paralelo" / ruta)
        assert _iguales(_estado(agente_s.policy), _estado(agente_p.policy))
        assert _iguales(_estado(d_s), _estado(d_p))
        assert meta_s["resumen"]["historial"] == meta_p["resumen"]["historial"]


def test_en_serie_tambien_entrena_con_un_hilo(datos_grabados, tmp_path, monkeypatch):
    """Regresión: en serie se usaban todos los hilos de PyTorch y, con redes del tamaño
    real, el modelo salía distinto del entrenado en paralelo (un hilo por proceso)."""
    script, _ = _preparar_script(monkeypatch, ["--datos", str(datos_grabados), "--cantidades",
                                               "4", "--repeticiones", "0", "--salida",
                                               str(tmp_path / "g")])
    hilos = []
    monkeypatch.setattr("src.training.gail.entrenar_gail",
                        lambda *a: hilos.append(torch.get_num_threads()) or _resumen_falso())
    monkeypatch.setattr("src.training.gail.guardar_gail", lambda *a: None)
    antes = torch.get_num_threads()
    torch.set_num_threads(2)  # un valor conocido distinto de 1
    try:
        assert script.main() == 0
        assert hilos == [1]
        assert torch.get_num_threads() == 2  # se restaura al terminar
    finally:
        torch.set_num_threads(antes)
