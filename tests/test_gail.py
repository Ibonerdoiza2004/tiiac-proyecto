"""Pruebas de la recompensa de GAIL: discriminador, recompensas, muestreo de
expertos, actualización y envoltorio del entorno.

Como en BC, los pares son inventados y no hace falta experto. Se cubren acciones
Discrete (con start distinto de 0) y Box de una dimensión. Los entornos son
inventados y deterministas, para comprobar exactamente qué par se guarda en cada
paso; CartPole y TRPO solo se usan para comprobar que todo encaja con sb3.
"""

from __future__ import annotations

import copy
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
from stable_baselines3.common.vec_env import DummyVecEnv

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
    generador = torch.Generator().manual_seed(semilla)
    salida = [m for m in discriminador.red if isinstance(m, torch.nn.Linear)][-1]
    with torch.no_grad():
        salida.weight.copy_(torch.randn(salida.weight.shape, generator=generador))
        salida.bias.copy_(torch.randn(salida.bias.shape, generator=generador))
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
    ("lote", 3, np.random.RandomState(0), "Generator"),
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

def _perdida_esperada(discriminador, agente, experto, continuo, peso) -> float:
    """Pérdida del original: media por grupo de BCE − peso·H, sumada."""
    def grupo(lote, etiqueta):
        logits = discriminador(*_tensores(lote, continuo)).double()
        p = torch.sigmoid(logits)
        bce = -(etiqueta * torch.log(p) + (1 - etiqueta) * torch.log(1 - p))
        entropia = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
        return (bce - peso * entropia).mean()
    with torch.no_grad():
        return float(grupo(agente, 0.0) + grupo(experto, 1.0))


@TIPOS
@pytest.mark.parametrize("peso", [0.0, 0.001, 0.5])
def test_la_perdida_pondera_cada_grupo_por_igual(continuo, peso):
    discriminador = _con_pesos_aleatorios(_discriminador(continuo))
    # Grupos de tamaños distintos: con una media conjunta el resultado cambiaría.
    agente, experto = _lote(7, continuo, 1), _lote(3, continuo, 2, centro=2.0)
    esperada = _perdida_esperada(copy.deepcopy(discriminador), agente, experto, continuo, peso)
    medidas = actualizar_discriminador(discriminador, torch.optim.Adam(discriminador.parameters()),
                                       agente, experto, _espacio(continuo),
                                       _cfg(peso_entropia=peso))
    assert medidas["perdida"] == pytest.approx(esperada, rel=1e-5)


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
            env = _EntornoContador(_espacio(continuo), fin)
            return Monitor(env) if monitor else env
        return hacer
    venv = DummyVecEnv([crear(fin) for fin in fines])
    discriminador = _con_pesos_aleatorios(_discriminador(continuo))
    return EntornoRecompensaGAIL(venv, discriminador, modo), discriminador


def _acciones(continuo: bool, paso: int, entornos: int = 2) -> np.ndarray:
    if continuo:
        return np.full((entornos, 2), (paso % 5) / 5 - 0.4, dtype=np.float32)
    return np.array([(paso + i) % 3 - 1 for i in range(entornos)])


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
                                              _espacio(continuo), modo)
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
])
def test_el_envoltorio_rechaza_piezas_no_compatibles(caso, mensaje):
    venv = DummyVecEnv([lambda: _EntornoContador(DISCRETO, 3)])
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
