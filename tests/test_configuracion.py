"""Cobertura del esquema y la resolución por capas de configuración."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import yaml

import run
from src.data.loaders import construir_loaders_bc
from src.data.splits import crear_splits
from src.data.tipos import Trayectoria
from src.models.factory import construir_politica_bc
from src.utils.configuracion import TipoEntorno, resolver_configuracion
from src.utils.seed import crear_generador

RAIZ = Path(__file__).resolve().parents[1]
BASE = RAIZ / "configs" / "default.yaml"


@pytest.mark.parametrize("entorno", list(TipoEntorno))
def test_todos_los_entornos_activos_resuelven_base_y_smoke(entorno):
    """Todo miembro activo tiene perfil y completa también la capa smoke."""
    smoke = yaml.safe_load((RAIZ / "configs" / "smoke.yaml").read_text(encoding="utf-8"))
    normal = resolver_configuracion(BASE, entorno)
    mini = resolver_configuracion(BASE, entorno.value, smoke)
    perfil = yaml.safe_load(
        (RAIZ / "configs" / "entornos" / f"{entorno.value}.yaml").read_text(encoding="utf-8")
    )

    assert normal["entorno"] == perfil["entorno"]
    assert normal["bc"]["frecuencia_pares"] == perfil["bc"]["frecuencia_pares"]
    assert normal["origen"]["entorno"] == entorno.value
    assert normal["origen"]["archivos"] == [
        "configs/default.yaml", f"configs/entornos/{entorno.value}.yaml",
    ]
    assert mini["entrenamiento"]["max_steps"] == 10
    assert mini["wandb"]["modo"] == "disabled"
    assert mini["origen"]["archivos"] == normal["origen"]["archivos"]


def test_enum_acepta_valor_textual_y_rechaza_entorno_desconocido():
    assert TipoEntorno("cartpole") is TipoEntorno.CARTPOLE
    with pytest.raises(ValueError, match="Entorno desconocido.*Opciones: cartpole"):
        resolver_configuracion(BASE, "no-existe")


def test_ajustes_sobrescriben_recursivamente_y_listas_se_reemplazan():
    cfg = resolver_configuracion(BASE, "cartpole", {
        "bc": {"capas_ocultas": [8], "learning_rate": 0.02},
        "datos": {"subset": 12},
    })
    assert cfg["bc"]["capas_ocultas"] == [8]
    assert cfg["bc"]["learning_rate"] == 0.02
    assert cfg["bc"]["frecuencia_pares"] == 10
    assert cfg["datos"]["subset"] == 12


def test_ajustes_vacios_o_none_no_cambian_la_configuracion():
    original = resolver_configuracion(BASE, "cartpole")
    assert resolver_configuracion(BASE, "cartpole", {}) == original
    assert resolver_configuracion(BASE, "cartpole", None) == original


def test_error_yaml_del_smoke_se_muestra_como_error_de_configuracion(monkeypatch, capsys):
    def cargar_config_malformada(_: Path) -> dict:
        """Simula un smoke.yaml con sintaxis YAML inválida."""
        raise yaml.YAMLError("sintaxis")

    monkeypatch.setattr(sys, "argv", ["run.py", "--smoke"])
    monkeypatch.setattr(run, "cargar_config", cargar_config_malformada)

    assert run.main() == 2
    salida = capsys.readouterr()
    assert "error de configuración" in salida.err
    assert "Traceback" not in salida.err


@pytest.mark.parametrize("capa", [
    {"bc": {"frecuencia_par": 10}},
    {"inventada": 1},
])
def test_clave_desconocida_falla_con_ruta(capa):
    with pytest.raises(ValueError, match="Clave de configuración desconocida:"):
        resolver_configuracion(BASE, "cartpole", capa)


def test_se_rechazan_capas_con_tipo_estructural_incorrecto():
    with pytest.raises(ValueError, match="diccionario en bc"):
        resolver_configuracion(BASE, "cartpole", {"bc": 1})
    with pytest.raises(ValueError, match="ajustes debe ser un mapeo"):
        resolver_configuracion(BASE, "cartpole", [])


def test_perfil_que_deja_null_falla_con_ruta_completa(tmp_path):
    (tmp_path / "entornos").mkdir()
    (tmp_path / "base.yaml").write_text("a:\n  b: null\n", encoding="utf-8")
    (tmp_path / "entornos" / "cartpole.yaml").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="a.b"):
        resolver_configuracion(tmp_path / "base.yaml", "cartpole")


def test_entradas_no_se_mutan_y_resultado_es_independiente():
    ajustes = {"bc": {"capas_ocultas": [32]}}
    copia = copy.deepcopy(ajustes)
    resultado = resolver_configuracion(BASE, "cartpole", ajustes)
    resultado["bc"]["capas_ocultas"].append(16)
    assert ajustes == copia


def test_error_si_falta_archivo_base_o_perfil(tmp_path):
    with pytest.raises(ValueError, match="No se pudo leer"):
        resolver_configuracion(tmp_path / "ausente.yaml", "cartpole")
    (tmp_path / "base.yaml").write_text("a: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="No se pudo leer"):
        resolver_configuracion(tmp_path / "base.yaml", "cartpole")


def test_config_resuelta_se_integra_con_politica_y_loaders_bc():
    cfg = resolver_configuracion(BASE, "cartpole")
    env = gym.make(cfg["entorno"]["id"])
    try:
        politica = construir_politica_bc(cfg, env.observation_space, env.action_space)
        trayectorias = []
        for identificador in range(3):
            estados = np.zeros((12, 4), dtype=np.float32)
            acciones = np.zeros(12, dtype=np.int64)
            siguientes = np.zeros((12, 4), dtype=np.float32)
            finales = np.zeros(12, dtype=bool)
            finales[-1] = True
            trayectorias.append(Trayectoria(
                identificador, estados, acciones, siguientes,
                np.zeros(12, dtype=np.float32), finales, np.zeros(12, dtype=bool),
            ))
        splits = crear_splits(trayectorias, 2, 1, (1,), crear_generador(0), "a" * 64)
        train, validacion = construir_loaders_bc(
            trayectorias, splits, 1, env.action_space, cfg, crear_generador(1),
        )
        assert politica.dim_accion == env.action_space.n
        assert len(train.dataset) > 0
        assert len(validacion.dataset) > 0
    finally:
        env.close()
