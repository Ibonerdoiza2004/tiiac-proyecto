"""Pruebas del protocolo común de predicción, evaluación y checkpoints."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from gymnasium import spaces

from src.models.factory import PoliticaBC
from src.data.entornos import crear_entorno_vectorizado
from src.training.evaluacion import (cargar_checkpoint_trpo, cargar_politica_bc,
                                     evaluar_politica, guardar_checkpoint_trpo,
                                     guardar_politica_bc, predecir_accion)
from src.training.trpo import crear_trpo
from src.utils.configuracion import resolver_configuracion
from src.utils.seed import fijar_seed
from stable_baselines3.common.vec_env import VecNormalize


def _cfg() -> dict:
    return {"entorno": {"id": "CartPole-v1"},
            "bc": {"activacion": "tanh", "capas_ocultas": [], "log_std_inicial": 0.0}}


def test_bc_discreta_respeta_start_y_modo_determinista():
    espacio = spaces.Discrete(3, start=-2)
    politica = PoliticaBC(2, 3, False, [], "tanh")
    with torch.no_grad():
        politica.red[-1].bias.copy_(torch.tensor([0.0, 3.0, 1.0]))
    accion = predecir_accion(politica, np.zeros(2, dtype=np.float32), espacio, True)
    assert accion == -1 and espacio.contains(accion)
    assert not politica.training


@pytest.mark.parametrize("semilla", [1, 5, 101])
def test_bc_estocastica_es_reproducible_y_no_altera_rng_global(semilla):
    espacio = spaces.Discrete(2)
    politica = PoliticaBC(2, 2, False, [], "tanh")
    antes = torch.random.get_rng_state()
    a = predecir_accion(politica, np.ones(2, dtype=np.float32), espacio, False, semilla)
    b = predecir_accion(politica, np.ones(2, dtype=np.float32), espacio, False, semilla)
    assert a == b and torch.equal(antes, torch.random.get_rng_state())


def test_acciones_box_de_bc_se_recortan_y_son_float32():
    espacio = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
    politica = PoliticaBC(2, 2, True, [], "tanh", 0.0)
    with torch.no_grad():
        politica.red[-1].bias.copy_(torch.tensor([4.0, -4.0]))
    accion = predecir_accion(politica, np.zeros(2, dtype=np.float32), espacio, True)
    assert accion.dtype == np.float32
    assert np.array_equal(accion, np.array([1.0, -1.0], dtype=np.float32))
    assert espacio.contains(accion)


def test_predecir_rechaza_box_escalar_incompatible_sin_indexerror():
    politica = PoliticaBC(2, 2, True, [], "tanh", 0.0)
    espacio = spaces.Box(-1.0, 1.0, shape=(), dtype=np.float32)
    with pytest.raises(ValueError, match="no coincide"):
        predecir_accion(politica, np.zeros(2, dtype=np.float32), espacio, True)


def test_aleatorio_y_evaluacion_se_reproducen_con_las_mismas_semillas():
    cfg = _cfg()
    a = evaluar_politica("aleatorio", cfg, 3, 42)
    b = evaluar_politica("aleatorio", cfg, 3, 42)
    assert a == b
    assert len(a["retornos"]) == len(a["longitudes"]) == 3
    assert a["retorno_medio"] == float(np.mean(a["retornos"]))


@pytest.mark.parametrize("episodios,semilla", [(0, 0), (-1, 0), (1.5, 0), (1, -2), (1, True)])
def test_evaluacion_rechaza_episodios_o_semilla_invalidos(episodios, semilla):
    with pytest.raises(ValueError):
        evaluar_politica("aleatorio", _cfg(), episodios, semilla)


def test_checkpoint_bc_conserva_pesos_buffers_y_metadatos(tmp_path: Path):
    politica = PoliticaBC(4, 2, False, [5, 3], "tanh")
    politica.ajustar_normalizacion(torch.arange(20, dtype=torch.float32).reshape(5, 4), 1e-6)
    politica.eval()
    datos = {"semilla": 17, "subconjunto": 4, "huella_dataset": "a" * 64}
    ruta = tmp_path / "bc.pt"
    cfg = _cfg()
    cfg["bc"]["capas_ocultas"] = [5, 3]
    guardar_politica_bc(ruta, politica, cfg, datos)
    cargada, metadatos = cargar_politica_bc(ruta)
    assert not cargada.training
    assert metadatos["entorno"] == "CartPole-v1"
    assert metadatos["huella_dataset"] == "a" * 64
    for clave, tensor in politica.state_dict().items():
        assert torch.equal(tensor, cargada.state_dict()[clave])
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    assert torch.equal(politica(x), cargada(x))


def test_checkpoint_bc_no_sobrescribe_y_rechaza_archivo_danado(tmp_path: Path):
    ruta = tmp_path / "bc.pt"
    politica = PoliticaBC(2, 2, False, [], "tanh")
    datos = {"semilla": 0, "subconjunto": 1, "huella_dataset": "b" * 64}
    guardar_politica_bc(ruta, politica, _cfg(), datos)
    with pytest.raises(FileExistsError):
        guardar_politica_bc(ruta, politica, _cfg(), datos)
    ruta.write_bytes(b"no es un checkpoint")
    with pytest.raises(ValueError):
        cargar_politica_bc(ruta)


def test_checkpoint_bc_rechaza_configuracion_que_no_reconstruye_la_red(tmp_path: Path):
    politica = PoliticaBC(2, 2, False, [4], "tanh")
    datos = {"semilla": 0, "subconjunto": 1, "huella_dataset": "d" * 64}
    with pytest.raises(ValueError, match="capas_ocultas"):
        guardar_politica_bc(tmp_path / "bc.pt", politica, _cfg(), datos)


def test_evaluacion_no_modifica_politica_bc_ni_crea_gradientes():
    politica = PoliticaBC(4, 2, False, [8], "tanh")
    estado = copy.deepcopy(politica.state_dict())
    evaluar_politica(politica, _cfg(), 2, 8)
    assert not politica.training
    assert all(param.grad is None for param in politica.parameters())
    assert all(torch.equal(valor, politica.state_dict()[clave])
               for clave, valor in estado.items())


@pytest.mark.parametrize("normalizar", [False, True])
def test_checkpoint_trpo_conserva_prediccion_y_vecnormalize(tmp_path: Path, normalizar: bool):
    raiz = Path(__file__).resolve().parent.parent
    cfg = resolver_configuracion(raiz / "configs" / "default.yaml", "cartpole")
    cfg["trpo"].update(pasos_por_iteracion=64, n_entornos=1, batch_size_valor=32,
                       capas_ocultas=[8, 8])
    cfg["trpo"]["normalizar_observaciones"] = normalizar
    fijar_seed(13)
    venv = crear_entorno_vectorizado(cfg, 1, 14)
    if normalizar:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=False)
        venv.reset()
    modelo = crear_trpo(cfg, venv)
    ruta = tmp_path / ("trpo_normalizado" if normalizar else "trpo")
    datos = {"semilla": 15, "subconjunto": 4, "huella_dataset": "c" * 64}
    try:
        observacion = np.zeros(4, dtype=np.float32)
        antes, _ = modelo.predict(observacion, deterministic=True)
        guardar_checkpoint_trpo(ruta, modelo, cfg, datos)
        env_carga = crear_entorno_vectorizado(cfg, 1, 99)
        cargado, env_cargado, metadatos = cargar_checkpoint_trpo(ruta, env_carga)
        try:
            despues, _ = cargado.predict(observacion, deterministic=True)
            assert np.array_equal(antes, despues)
            assert isinstance(env_cargado, VecNormalize) is normalizar
            assert metadatos["huella_dataset"] == "c" * 64
            assert metadatos["configuracion"] == cfg
        finally:
            env_cargado.close()
        with pytest.raises(FileExistsError):
            guardar_checkpoint_trpo(ruta, modelo, cfg, datos)
    finally:
        venv.close()

