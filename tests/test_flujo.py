"""Integración de las etapas de run.py con un experto y datos temporales."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from src.data.demostraciones import generar_dataset
from src.training.flujo import (comprobar_datos_congelados, ejecutar_experimento,
                                evaluar_carpeta, generar_y_registrar_datos,
                                entrenar_y_guardar_experto)
from src.utils.configuracion import resolver_configuracion

RAIZ = Path(__file__).resolve().parents[1]


def _cfg() -> dict:
    """Devuelve la configuración del smoke con W&B desactivado."""
    ajustes = yaml.safe_load((RAIZ / "configs" / "smoke.yaml").read_text(encoding="utf-8"))
    cfg = resolver_configuracion(RAIZ / "configs" / "default.yaml", "cartpole", ajustes)
    cfg["wandb"]["modo"] = "disabled"
    return cfg


def _preparar(carpeta: Path) -> tuple[dict, Path, Path]:
    """Genera un experto reducido y las partidas que consumen BC y GAIL."""
    cfg = _cfg()
    torch.set_num_threads(1)
    experto = carpeta / "experto"
    datos = carpeta / "datos"
    entrenar_y_guardar_experto(cfg, experto, "test-flujo")
    resultado = generar_y_registrar_datos(cfg, experto, datos, "test-flujo")
    assert resultado["coincide"]
    cfg["datos"]["huella_partidas"] = resultado["huella_partidas"]
    return cfg, experto, datos


def test_ejecutar_experimento_entrena_y_evalua_todos_los_agentes(tmp_path):
    cfg, experto, datos = _preparar(tmp_path)
    resultado = ejecutar_experimento(cfg, datos, experto, tmp_path / "run", "test-flujo")
    assert resultado["fallos"] == []
    assert len(resultado["resultados"]) == 2 + 2 * len(cfg["demostraciones"]["subconjuntos"])
    assert (tmp_path / "run" / "bc").is_dir()
    assert (tmp_path / "run" / "gail").is_dir()
    assert (tmp_path / "run" / "evaluacion.json").is_file()


def test_dos_ejecuciones_y_reevaluacion_conservan_los_retorno(tmp_path):
    cfg, experto, datos = _preparar(tmp_path)
    primero = ejecutar_experimento(cfg, datos, experto, tmp_path / "primero", "test-1")
    segundo = ejecutar_experimento(cfg, datos, experto, tmp_path / "segundo", "test-2")
    retorno = lambda resultado: [fila["retornos"] for fila in resultado["resultados"]]
    assert retorno(primero) == retorno(segundo)
    reevaluado = evaluar_carpeta(cfg, tmp_path / "primero", experto, "test-3")
    assert retorno(primero) == retorno(reevaluado)


def test_comprobar_datos_congelados_acepta_solo_experto_datos_y_huella_validos(tmp_path):
    cfg, experto, datos = _preparar(tmp_path)
    comprobar_datos_congelados(cfg, datos, experto)
    with pytest.raises(ValueError, match="experto congelado"):
        comprobar_datos_congelados(cfg, datos, tmp_path / "falta-experto")
    with pytest.raises(ValueError, match="no están los datos congelados"):
        comprobar_datos_congelados(cfg, tmp_path / "falta-datos", experto)
    cfg_erronea = copy.deepcopy(cfg)
    cfg_erronea["datos"]["huella_partidas"] = "0" * 64
    with pytest.raises(ValueError, match="huella de las partidas"):
        comprobar_datos_congelados(cfg_erronea, datos, experto)
