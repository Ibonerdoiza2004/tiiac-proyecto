"""Tests de las reglas compartidas de la configuración (src/utils/reglas_config.py).

Las comprobaciones de cada sección ya se prueban a fondo en los tests de cada
entrenador (test_trpo.py, test_bc.py, test_gail.py), que siguen usándolas con los
mismos nombres. Aquí se prueba lo propio del módulo: que no necesita PyTorch, que sus
nombres coinciden con las clases de los entrenadores y comprobar_config_metodo.
"""

from __future__ import annotations

import copy
import itertools
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.utils import reglas_config
from src.utils.configuracion import TipoEntorno, resolver_configuracion
from src.utils.reglas_config import METODOS, comprobar_config_metodo
from src.utils.validacion import es_semilla

RAIZ = Path(__file__).resolve().parents[1]
CONFIG = RAIZ / "configs" / "default.yaml"


def test_no_importa_pytorch_ni_sb3() -> None:
    codigo = ("import sys, src.utils.reglas_config; "
              "print(sorted(m for m in ('torch', 'stable_baselines3', 'sb3_contrib', "
              "'gymnasium') if m in sys.modules))")
    salida = subprocess.run([sys.executable, "-c", codigo], cwd=RAIZ, capture_output=True,
                            text=True, check=True).stdout.strip()
    assert salida == "[]"


def test_los_nombres_coinciden_con_las_clases_de_los_entrenadores() -> None:
    from src.models.factory import ACTIVACIONES
    from src.training import bc, gail, trpo

    assert set(reglas_config.ACTIVACIONES) == set(ACTIVACIONES)
    assert set(reglas_config.OPTIMIZADORES_BC) == set(bc.OPTIMIZADORES)
    # Los entrenadores usan estas mismas funciones, no copias.
    assert trpo.config_trpo is reglas_config.config_trpo
    assert bc.config_bc is reglas_config.config_bc
    assert gail._config_discriminador is reglas_config.config_discriminador
    assert gail.MODOS_RECOMPENSA is reglas_config.MODOS_RECOMPENSA


@pytest.mark.parametrize("entorno, metodo", list(itertools.product(TipoEntorno, METODOS)))
def test_todos_los_perfiles_cumplen_las_reglas(entorno, metodo) -> None:
    comprobar_config_metodo(resolver_configuracion(CONFIG, entorno), metodo)


def _cambiar(cfg: dict, ruta: str, valor: object) -> dict:
    """Copia de cfg con la clave de ruta (con puntos) cambiada."""
    copia = copy.deepcopy(cfg)
    *secciones, clave = ruta.split(".")
    destino = copia
    for seccion in secciones:
        destino = destino[seccion]
    destino[clave] = valor
    return copia


@pytest.mark.parametrize("metodo, ruta, valor, mensaje", [
    ("experto", "experto.semilla", -1, "experto.semilla"),
    ("experto", "experto.trpo", {"gamma": 0}, "trpo.gamma"),
    ("experto", "experto.trpo", [1], "experto.trpo debe ser un diccionario"),
    ("bc", "bc.batch_size", 0, "bc.batch_size"),
    ("bc", "bc.activacion", "sigmoid", "activacion desconocida"),
    ("bc", "demostraciones.frecuencia_pares", 0, "frecuencia_pares"),
    ("gail", "gail.recompensa", "otra", "gail.recompensa"),
    ("gail", "trpo.n_entornos", 3, "divisible por trpo.n_entornos"),
    ("gail", "gail.discriminador.optimizador", "sgd", "optimizador desconocido"),
    ("gail", "demostraciones.frecuencia_pares", True, "frecuencia_pares"),
])
def test_comprobar_config_metodo_rechaza(metodo, ruta, valor, mensaje) -> None:
    cfg = _cambiar(resolver_configuracion(CONFIG, "cartpole"), ruta, valor)
    with pytest.raises(ValueError, match=mensaje):
        comprobar_config_metodo(cfg, metodo)


def test_cada_metodo_solo_mira_sus_secciones() -> None:
    cfg = resolver_configuracion(CONFIG, "cartpole")
    comprobar_config_metodo(_cambiar(cfg, "bc.learning_rate", 0), "gail")
    comprobar_config_metodo(_cambiar(cfg, "gail.recompensa", "otra"), "bc")
    comprobar_config_metodo(_cambiar(cfg, "experto.semilla", -1), "bc")


def test_metodo_desconocido() -> None:
    with pytest.raises(ValueError, match="Método desconocido"):
        comprobar_config_metodo(resolver_configuracion(CONFIG, "cartpole"), "ppo")


@pytest.mark.parametrize("valor, esperado", [
    (0, True), (5, True), (np.int64(3), True), (-1, False), (True, False), (1.0, False),
    ("1", False), (None, False), (np.int32(-2), False),
])
def test_es_semilla(valor, esperado) -> None:
    assert es_semilla(valor) is esperado
