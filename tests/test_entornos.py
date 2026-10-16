"""Pruebas de creación, siembra y monitorización de entornos."""

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from src.data.entornos import crear_entorno, crear_entorno_vectorizado
from src.utils.seed import (crear_generador, generador_torch_aislado, reiniciar_con_semilla,
                            sembrar_entorno,
                            sembrar_entorno_vectorizado, semillas_independientes,
                            validar_semilla)


CFG = {"entorno": {"id": "CartPole-v1"}}


def test_entorno_repite_observacion_trayectoria_y_muestra_de_accion():
    entornos = [crear_entorno(CFG, 27), crear_entorno(CFG, 27)]
    try:
        iniciales = [env.reset()[0] for env in entornos]
        assert np.array_equal(iniciales[0], iniciales[1])
        for accion in (0, 1, 1, 0, 0):
            salidas = [env.step(accion) for env in entornos]
            assert np.array_equal(salidas[0][0], salidas[1][0])
            assert salidas[0][1:] == salidas[1][1:]
        assert entornos[0].action_space.sample() == entornos[1].action_space.sample()
    finally:
        for env in entornos:
            env.close()


def test_semillas_distintas_cambian_observacion_inicial():
    env_a, env_b = crear_entorno(CFG, 27), crear_entorno(CFG, 28)
    try:
        assert not np.array_equal(env_a.reset()[0], env_b.reset()[0])
    finally:
        env_a.close()
        env_b.close()


@pytest.mark.parametrize("semilla", [-1, 1.5, True])
def test_sembrar_entorno_rechaza_semilla_invalida(semilla):
    env = crear_entorno(CFG, None)
    try:
        with pytest.raises(ValueError, match="entero no negativo"):
            sembrar_entorno(env, semilla)
    finally:
        env.close()


@pytest.mark.parametrize("cfg", [{}, {"entorno": {}}, {"entorno": {"id": "NoExiste-v0"}}])
def test_crear_entorno_rechaza_configuracion_o_id_desconocidos(cfg):
    with pytest.raises(ValueError, match="entorno.id|No se pudo crear"):
        crear_entorno(cfg, None)


@pytest.mark.parametrize("cantidad", [0, -1, True, 1.5])
def test_vectorizado_rechaza_numero_de_entornos_invalido(cantidad):
    with pytest.raises(ValueError, match="n_entornos"):
        crear_entorno_vectorizado(CFG, cantidad, 5)


@pytest.mark.parametrize("semilla", [-1, 2.5, True])
def test_vectorizado_rechaza_semilla_invalida(semilla):
    with pytest.raises(ValueError, match="semilla"):
        crear_entorno_vectorizado(CFG, 1, semilla)


def test_vectorizado_tiene_forma_semillas_distintas_y_monitor_por_entorno():
    venv = crear_entorno_vectorizado(CFG, 2, 31)
    try:
        observaciones = venv.reset()
        assert observaciones.shape == (2, 4)
        assert not np.array_equal(observaciones[0], observaciones[1])
        for _ in range(600):
            _, _, dones, infos = venv.step(np.zeros(2, dtype=np.int64))
            if dones.all():
                break
        assert dones.all()
        assert all("episode" in info for info in infos)
        assert all(type(env).__name__ == "Monitor" for env in venv.envs)
    finally:
        venv.close()


def _recorrido(semilla: int, pasos: int = 300) -> tuple[np.ndarray, list]:
    """Observaciones de un vectorizado de 2 instancias con acciones fijas."""
    venv = crear_entorno_vectorizado(CFG, 2, semilla)
    try:
        observaciones = [venv.reset()]
        muestras = [int(venv.action_space.sample()) for _ in range(5)]
        acciones = crear_generador(0).integers(0, 2, size=(pasos, 2))
        for accion in acciones:
            observaciones.append(venv.step(accion)[0])
        return np.array(observaciones), muestras
    finally:
        venv.close()


def test_dos_vectorizados_con_la_misma_semilla_son_identicos():
    # 300 pasos con acciones al azar incluyen varios reinicios automáticos.
    (a, muestras_a), (b, muestras_b) = _recorrido(5), _recorrido(5)
    assert np.array_equal(a, b)
    assert muestras_a == muestras_b
    c, _ = _recorrido(6)
    assert not np.array_equal(a, c)


@pytest.mark.parametrize("n_entornos", [2, 3])
def test_bases_cercanas_no_comparten_instancias(n_entornos):
    # Regresión: con semilla + i, la base 5 y la 6 compartían una instancia.
    iniciales = []
    for base in range(5, 9):
        venv = crear_entorno_vectorizado(CFG, n_entornos, base)
        try:
            iniciales += [tuple(fila) for fila in venv.reset()]
        finally:
            venv.close()
    assert len(set(iniciales)) == len(iniciales)


def test_semillas_independientes_son_deterministas_y_no_se_solapan():
    assert semillas_independientes(5, 3) == semillas_independientes(np.int64(5), 3)
    todas = [s for base in range(50) for s in semillas_independientes(base, 4)]
    assert len(set(todas)) == len(todas)
    assert all(isinstance(s, int) and s >= 0 for s in todas)
    assert semillas_independientes(5, 1) == semillas_independientes(5, 3)[:1]


@pytest.mark.parametrize("semilla,cantidad,mensaje", [
    (-1, 2, "entero no negativo"), (True, 2, "entero no negativo"), (1.0, 2, "entero no negativo"),
    (1, 0, "cantidad"), (1, -1, "cantidad"), (1, True, "cantidad"), (1, 2.0, "cantidad"),
])
def test_semillas_independientes_rechaza_argumentos_no_validos(semilla, cantidad, mensaje):
    with pytest.raises(ValueError, match=mensaje):
        semillas_independientes(semilla, cantidad)


def test_la_validacion_de_semillas_es_la_misma_en_todas_las_funciones():
    # Antes el vectorizado rechazaba np.int64 y el resto lo aceptaba.
    assert validar_semilla(np.int64(3)) == 3 and type(validar_semilla(np.int64(3))) is int
    crear_entorno(CFG, np.int64(3)).close()
    crear_entorno_vectorizado(CFG, np.int64(1), np.int64(3)).close()
    crear_generador(np.int64(3))
    for funcion in (validar_semilla, crear_generador, lambda s: crear_entorno(CFG, s),
                    lambda s: crear_entorno_vectorizado(CFG, 1, s)):
        with pytest.raises(ValueError, match="La semilla debe ser un entero no negativo"):
            funcion(-1)


class _EntornoQueMuestreaEnReset(gym.Env):
    """Entorno cuyo reset devuelve observation_space.sample() y lo apunta."""

    def __init__(self) -> None:
        self.observation_space = spaces.Box(-1, 1, (3,), np.float32)
        self.action_space = spaces.Discrete(2)
        self.observaciones_reset = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        observacion = self.observation_space.sample()
        self.observaciones_reset.append(observacion)
        return observacion, {}

    def step(self, accion):
        return self.observation_space.sample(), 0.0, False, False, {}


def test_los_espacios_se_siembran_antes_del_reset():
    # Regresión: el reset de siembra muestreaba el espacio antes de sembrarlo.
    a, b = _EntornoQueMuestreaEnReset(), _EntornoQueMuestreaEnReset()
    sembrar_entorno(a, 9)
    sembrar_entorno(b, 9)
    assert np.array_equal(a.observaciones_reset[0], b.observaciones_reset[0])


def test_crear_entorno_admite_render_mode():
    env = crear_entorno(CFG, 0, render_mode="rgb_array")
    try:
        assert env.render_mode == "rgb_array"
        env.reset()
        assert env.render().ndim == 3
    finally:
        env.close()


def test_sembrar_rechaza_objetos_que_no_son_entornos():
    with pytest.raises(TypeError, match="entorno Gymnasium"):
        sembrar_entorno(object(), 1)
    with pytest.raises(TypeError, match="VecEnv"):
        sembrar_entorno_vectorizado(object(), 1)


def test_reiniciar_con_semilla_da_la_primera_partida_de_esa_semilla():
    env, otro = crear_entorno(CFG, None), crear_entorno(CFG, None)
    try:
        # Sin gastar ningún reset: coincide con reset(seed=...) directo.
        observacion, _ = reiniciar_con_semilla(env, 12)
        assert np.array_equal(observacion, otro.reset(seed=12)[0])
        # Tras otras partidas, la misma semilla vuelve a dar la misma partida.
        env.reset()
        assert np.array_equal(reiniciar_con_semilla(env, 12)[0], observacion)
        assert not np.array_equal(reiniciar_con_semilla(env, 13)[0], observacion)
        # Siembra también los espacios.
        reiniciar_con_semilla(env, 12)
        reiniciar_con_semilla(otro, 12)
        assert env.action_space.sample() == otro.action_space.sample()
        with pytest.raises(ValueError, match="entero no negativo"):
            reiniciar_con_semilla(env, -1)
        with pytest.raises(TypeError, match="entorno Gymnasium"):
            reiniciar_con_semilla(object(), 1)
    finally:
        env.close()
        otro.close()


def test_generador_torch_aislado_depende_de_su_semilla_y_restaura_el_global():
    import torch
    estado = torch.get_rng_state()
    with generador_torch_aislado(4):
        a = torch.rand(3)
    assert torch.equal(torch.get_rng_state(), estado)
    torch.rand(5)
    with generador_torch_aislado(4):
        b = torch.rand(3)
    with generador_torch_aislado(5):
        c = torch.rand(3)
    assert torch.equal(a, b) and not torch.equal(a, c)
    with pytest.raises(ValueError, match="entero no negativo"):
        with generador_torch_aislado(-1):
            pass

