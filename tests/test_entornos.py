"""Pruebas de creación, siembra y monitorización de entornos."""

import numpy as np
import pytest

from src.data.entornos import crear_entorno, crear_entorno_vectorizado
from src.utils.seed import sembrar_entorno


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
