"""Pruebas de las comprobaciones numéricas comunes (src/utils/validacion.py).

Se cubren los tipos que llegan en la práctica: enteros y reales de Python (de
la config YAML), escalares de NumPy (de arrays o del código), booleanos de los
dos (que son subtipos de int y no deben contar), no finitos y otros tipos.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.utils.validacion import es_entero, es_entero_positivo, es_real_finito

ENTEROS = [0, 1, -3, 10**12, np.int64(5), np.int32(-2), np.uint8(0)]
NO_ENTEROS = [True, False, np.bool_(True), 1.0, np.float32(2.0), "3", None, [1], 2 + 0j]


@pytest.mark.parametrize("valor", ENTEROS)
def test_es_entero_acepta_enteros_de_python_y_numpy(valor):
    assert es_entero(valor)


@pytest.mark.parametrize("valor", NO_ENTEROS)
def test_es_entero_rechaza_booleanos_reales_y_otros_tipos(valor):
    assert not es_entero(valor)


@pytest.mark.parametrize("valor,esperado", [
    (1, True), (7, True), (np.int64(3), True), (0, False), (-1, False), (np.int16(-4), False),
    (True, False), (np.bool_(True), False), (1.0, False), ("1", False), (None, False),
])
def test_es_entero_positivo(valor, esperado):
    assert es_entero_positivo(valor) is esperado


@pytest.mark.parametrize("valor,esperado", [
    (0, True), (1, True), (-2.5, True), (1e-300, True), (np.float32(0.5), True),
    (np.float64(-1e10), True), (np.int64(4), True),
    (math.nan, False), (math.inf, False), (-math.inf, False), (np.float64("nan"), False),
    (np.float32("inf"), False), (True, False), (np.bool_(False), False), ("1.0", False),
    (None, False), (2 + 0j, False), ([1.0], False),
])
def test_es_real_finito(valor, esperado):
    assert es_real_finito(valor) is esperado


def test_las_secciones_de_config_admiten_enteros_de_numpy():
    """Antes, trpo y gail rechazaban enteros de NumPy y evaluacion o factory los admitían."""
    from pathlib import Path

    from src.training.gail import _config_discriminador, _config_gail
    from src.training.trpo import config_trpo
    from src.utils.configuracion import resolver_configuracion
    raiz = Path(__file__).resolve().parents[1]
    cfg = resolver_configuracion(raiz / "configs" / "default.yaml", "cartpole")
    cfg["trpo"]["n_entornos"] = np.int64(cfg["trpo"]["n_entornos"])
    cfg["gail"]["iteraciones"] = np.int64(cfg["gail"]["iteraciones"])
    cfg["gail"]["discriminador"]["pasos_adam"] = np.int32(1)
    cfg["gail"]["discriminador"]["learning_rate"] = np.float32(0.01)
    config_trpo(cfg)
    _config_gail(cfg)
    _config_discriminador(cfg)
    for seccion, clave in (("trpo", "n_entornos"), ("gail", "iteraciones")):
        cfg[seccion][clave] = True
        with pytest.raises(ValueError, match=clave):
            config_trpo(cfg) if seccion == "trpo" else _config_gail(cfg)
        cfg[seccion][clave] = 1
