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
from src.utils.configuracion import (DATOS_PENDIENTES, TipoEntorno, dato_congelado,
                                     resolver_configuracion)
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
    assert normal["trpo"]["n_entornos"] == perfil["trpo"]["n_entornos"]
    assert (normal["demostraciones"]["frecuencia_pares"]
            == perfil["demostraciones"]["frecuencia_pares"])
    assert normal["origen"]["entorno"] == entorno.value
    assert normal["origen"]["archivos"] == [
        "configs/default.yaml", f"configs/entornos/{entorno.value}.yaml",
    ]
    assert mini["gail"]["iteraciones"] == 2
    assert mini["wandb"]["modo"] == "disabled"
    assert mini["experimento"]["repeticiones"] == 1
    assert mini["origen"]["archivos"] == normal["origen"]["archivos"]


def test_enum_acepta_valor_textual_y_rechaza_entorno_desconocido():
    assert TipoEntorno("cartpole") is TipoEntorno.CARTPOLE
    with pytest.raises(ValueError, match="Entorno desconocido.*Opciones: cartpole"):
        resolver_configuracion(BASE, "no-existe")


def test_ajustes_sobrescriben_recursivamente_y_listas_se_reemplazan():
    cfg = resolver_configuracion(BASE, "cartpole", {
        "bc": {"capas_ocultas": [8], "learning_rate": 0.02},
        "demostraciones": {"subconjuntos": [1, 12]},
    })
    assert cfg["bc"]["capas_ocultas"] == [8]
    assert cfg["bc"]["learning_rate"] == 0.02
    assert cfg["demostraciones"]["frecuencia_pares"] == 10
    assert cfg["demostraciones"]["subconjuntos"] == [1, 12]


def test_ajustes_vacios_o_none_no_cambian_la_configuracion():
    original = resolver_configuracion(BASE, "cartpole")
    assert resolver_configuracion(BASE, "cartpole", {}) == original
    assert resolver_configuracion(BASE, "cartpole", None) == original


def test_error_yaml_del_smoke_se_muestra_como_error_de_configuracion(monkeypatch, capsys):
    def cargar_config_malformada(_: Path) -> dict:
        """Simula un smoke.yaml con sintaxis YAML inválida."""
        raise yaml.YAMLError("sintaxis")

    monkeypatch.setattr(sys, "argv", ["run.py", "--smoke"])
    monkeypatch.setattr(run.yaml, "safe_load", cargar_config_malformada)

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


TAMANOS = [("experimento", "repeticiones"), ("demostraciones", "partidas_train"),
           ("demostraciones", "partidas_validacion"), ("demostraciones", "subconjuntos"),
           ("demostraciones", "frecuencia_pares"), ("gail", "iteraciones"),
           ("gail", "evaluar_cada"), ("gail", "episodios_evaluacion"),
           ("gail", "evaluacion_determinista"), ("evaluacion", "episodios"),
           ("evaluacion", "semilla"), ("evaluacion", "determinista")]


@pytest.mark.parametrize("entorno", list(TipoEntorno))
def test_cada_perfil_fija_sus_tamanos_del_experimento(entorno):
    """Los tamaños están en el perfil, no solo en la base."""
    perfil = yaml.safe_load(
        (RAIZ / "configs" / "entornos" / f"{entorno.value}.yaml").read_text(encoding="utf-8"))
    cfg = resolver_configuracion(BASE, entorno)
    for seccion, clave in TAMANOS:
        assert clave in perfil.get(seccion, {}), f"{entorno.value}: falta {seccion}.{clave}"
        assert cfg[seccion][clave] == perfil[seccion][clave]


def test_perfil_cartpole_fija_rutas_y_huella_de_datos_congelados():
    cfg = resolver_configuracion(BASE, "cartpole")
    assert cfg["datos"]["demostraciones"] == "data/demostraciones/cartpole/congelado"
    assert cfg["datos"]["experto"] == "checkpoints/expertos/cartpole/congelado"
    assert cfg["datos"]["huella_partidas"]
    assert cfg["datos"]["huella_experto"] == (
        "92c44313863ebe30c616be30664b2742650ee80cd833f2b799550fe742d517ac")


def test_cambiar_la_base_no_cambia_los_tamanos_de_cartpole(tmp_path):
    base = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    base["experimento"]["repeticiones"] = 3
    base["demostraciones"].update(partidas_train=30, partidas_validacion=9,
                                  subconjuntos=[5, 30])
    base["demostraciones"]["frecuencia_pares"] = 3
    base["gail"].update(iteraciones=20, evaluar_cada=1, episodios_evaluacion=50,
                        evaluacion_determinista=False)
    base["evaluacion"].update(episodios=12, semilla=123, determinista=False)
    (tmp_path / "entornos").mkdir()
    (tmp_path / "base.yaml").write_text(yaml.safe_dump(base), encoding="utf-8")
    perfil = RAIZ / "configs" / "entornos" / "cartpole.yaml"
    (tmp_path / "entornos" / "cartpole.yaml").write_text(perfil.read_text(encoding="utf-8"),
                                                         encoding="utf-8")
    cfg = resolver_configuracion(tmp_path / "base.yaml", "cartpole")
    assert cfg["experimento"]["repeticiones"] == 5
    tamanos = {clave: cfg["demostraciones"][clave] for clave in
               ("partidas_train", "partidas_validacion", "subconjuntos", "frecuencia_pares")}
    assert tamanos == {"partidas_train": 10, "partidas_validacion": 5,
                       "subconjuntos": [1, 4, 7, 10], "frecuencia_pares": 10}
    evaluacion = {clave: cfg["gail"][clave] for clave in
                  ("iteraciones", "evaluar_cada", "episodios_evaluacion",
                   "evaluacion_determinista")}
    assert evaluacion == {"iteraciones": 20, "evaluar_cada": 10, "episodios_evaluacion": 10,
                          "evaluacion_determinista": True}
    assert cfg["evaluacion"] == {"episodios": 50, "semilla": 9000, "determinista": True}


@pytest.mark.parametrize("entorno", list(TipoEntorno))
@pytest.mark.parametrize("smoke", [False, True], ids=["normal", "smoke"])
def test_los_tamanos_del_experimento_son_coherentes(entorno, smoke):
    ajustes = (yaml.safe_load((RAIZ / "configs" / "smoke.yaml").read_text(encoding="utf-8"))
               if smoke else None)
    cfg = resolver_configuracion(BASE, entorno, ajustes)
    demostraciones = cfg["demostraciones"]
    for valor in (cfg["experimento"]["repeticiones"], cfg["gail"]["iteraciones"],
                  demostraciones["partidas_validacion"]):
        assert isinstance(valor, int) and not isinstance(valor, bool) and valor > 0
    # crear_splits acepta las cantidades: subconjuntos crecientes dentro de train.
    total = demostraciones["partidas_train"] + demostraciones["partidas_validacion"]
    trayectorias = [Trayectoria(i, np.zeros((3, 4), np.float32), np.zeros(3, np.int64),
                                np.zeros((3, 4), np.float32), np.zeros(3, np.float32),
                                np.array([False, False, True]), np.zeros(3, bool))
                    for i in range(total)]
    splits = crear_splits(trayectorias, demostraciones["partidas_train"],
                          demostraciones["partidas_validacion"],
                          demostraciones["subconjuntos"], crear_generador(0), "a" * 64)
    assert sorted(splits.subconjuntos) == demostraciones["subconjuntos"]
    assert len(splits.train) == demostraciones["subconjuntos"][-1]


def test_la_frecuencia_de_pares_ya_no_esta_en_bc():
    """Es común a BC y GAIL: vive en demostraciones y la sección bc la rechaza."""
    cfg = resolver_configuracion(BASE, "cartpole")
    assert "frecuencia_pares" not in cfg["bc"]
    with pytest.raises(ValueError, match="desconocida: bc.frecuencia_pares"):
        resolver_configuracion(BASE, "cartpole", {"bc": {"frecuencia_pares": 10}})


def test_la_semilla_de_las_repeticiones_esta_en_la_config():
    from src.utils.seed import semillas_repeticion
    cfg = resolver_configuracion(BASE, "cartpole")
    semilla = cfg["experimento"]["semilla_repeticiones"]
    for repeticion in range(cfg["experimento"]["repeticiones"]):
        semillas_repeticion(semilla, repeticion)



CLAVES_DATOS = ("demostraciones", "experto", "huella_partidas", "huella_experto",
          "artefacto_demostraciones", "artefacto_experto")


def test_un_perfil_con_datos_pendientes_se_resuelve():
    """Con el marcador, un entorno nuevo puede entrenar su experto y grabar sus datos."""
    cfg = resolver_configuracion(BASE, "cartpole",
                                 {"datos": {c: DATOS_PENDIENTES for c in CLAVES_DATOS}})
    assert all(cfg["datos"][clave] == DATOS_PENDIENTES for clave in CLAVES_DATOS)


@pytest.mark.parametrize("clave", CLAVES_DATOS)
def test_dato_congelado_devuelve_el_valor_real_y_rechaza_el_pendiente(clave):
    cfg = resolver_configuracion(BASE, "cartpole")
    assert dato_congelado(cfg, clave) == cfg["datos"][clave]
    cfg["datos"][clave] = DATOS_PENDIENTES
    with pytest.raises(ValueError, match=rf"datos\.{clave} sigue «pendiente».*CartPole-v1"):
        dato_congelado(cfg, clave)


@pytest.mark.parametrize("valor", ["Pendiente", "PENDIENTE", "pendiente ", "", None, 0])
def test_dato_congelado_solo_reconoce_el_marcador_exacto(valor):
    """Cualquier otro valor se devuelve tal cual: lo rechazarán las huellas o las rutas."""
    cfg = resolver_configuracion(BASE, "cartpole")
    cfg["datos"]["huella_experto"] = valor
    assert dato_congelado(cfg, "huella_experto") == valor


@pytest.mark.parametrize("entorno,gym_id,frecuencia,recompensa,subconjuntos,repeticiones,"
                         "iteraciones,pasos", [
    ("mountaincar", "MountainCar-v0", 5, "log_d", [1, 4, 7, 10], 5, 300, 5000),
    ("acrobot", "Acrobot-v1", 5, "log_d", [1, 4, 7, 10], 5, 300, 5000),
    ("reacher", "Reacher-v5", 20, "menos_log_uno_menos_d", [4, 11, 18, 25], 1, 500, 50000),
    ("halfcheetah", "HalfCheetah-v5", 20, "menos_log_uno_menos_d", [4, 11, 18, 25], 1, 500,
     50000),
])
def test_perfiles_nuevos_siguen_al_codigo_original_y_citan_sus_datos_congelados(
    entorno, gym_id, frecuencia, recompensa, subconjuntos, repeticiones, iteraciones, pasos
):
    """Valores de im_classic_pipeline.yaml / im_pipeline.yaml; objetivo, el umbral de Gymnasium.

    Las repeticiones de MountainCar y Acrobot son 5, como en CartPole (elección propia).
    """
    cfg = resolver_configuracion(BASE, entorno)
    assert cfg["entorno"]["id"] == gym_id
    assert cfg["experto"]["retorno_objetivo"] == gym.spec(gym_id).reward_threshold
    assert cfg["demostraciones"]["frecuencia_pares"] == frecuencia
    assert cfg["demostraciones"]["subconjuntos"] == subconjuntos
    assert cfg["demostraciones"]["partidas_train"] == subconjuntos[-1]
    assert cfg["gail"]["recompensa"] == recompensa
    assert cfg["experimento"]["repeticiones"] == repeticiones
    assert cfg["gail"]["iteraciones"] == iteraciones
    assert cfg["trpo"]["pasos_por_iteracion"] == pasos
    assert cfg["experto"]["pasos_totales"] % pasos == 0
    assert DATOS_PENDIENTES not in cfg["datos"].values()
    assert cfg["datos"]["demostraciones"] == f"data/demostraciones/{entorno}/congelado"
    assert cfg["datos"]["experto"] == f"checkpoints/expertos/{entorno}/congelado"
