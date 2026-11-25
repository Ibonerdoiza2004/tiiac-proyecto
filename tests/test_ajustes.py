"""Tests del esquema de hiperparámetros del dashboard (src/app/ajustes.py y
configs/dashboard.yaml).

El dominio es finito: cinco entornos por tres métodos. Se recorren todos.
"""

from __future__ import annotations

import copy
import itertools
import math
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.app.ajustes import (METODOS, campo, cargar_dashboard, comprobar_ajustes,
                             esquema_ajustes, pasos_reales_experto, secciones_de_metodo,
                             semilla_por_defecto)
from src.utils.configuracion import TipoEntorno, resolver_configuracion

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
CONFIG = CONFIGS / "default.yaml"
DASHBOARD = cargar_dashboard(CONFIGS / "dashboard.yaml")
ENTORNOS = [t.value for t in TipoEntorno]
TODOS = list(itertools.product(ENTORNOS, METODOS))


def _campos(esquema: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Campos del esquema por su ruta «seccion.clave»."""
    return {f"{s['clave']}.{c['clave']}": c for s in esquema["secciones"] for c in s["campos"]}


# --- El esquema ---------------------------------------------------------------------

@pytest.mark.parametrize("entorno, metodo", TODOS)
def test_cada_valor_del_esquema_es_el_del_perfil(entorno, metodo) -> None:
    cfg = resolver_configuracion(CONFIG, entorno)
    esquema = esquema_ajustes(CONFIG, entorno, metodo, DASHBOARD)
    valores = dict(secciones_de_metodo(cfg, metodo))
    for ruta, c in _campos(esquema).items():
        seccion = ruta.rsplit(".", 1)[0]
        assert c["valor"] == valores[seccion][c["clave"]], ruta
        assert c["etiqueta"] and c["ayuda"] and c["nivel"] in ("basico", "avanzado")


@pytest.mark.parametrize("entorno, metodo", TODOS)
def test_duracion_y_semilla_por_defecto(entorno, metodo) -> None:
    cfg = resolver_configuracion(CONFIG, entorno)
    esquema = esquema_ajustes(CONFIG, entorno, metodo, DASHBOARD)
    duracion = {"experto": {"unidad": "pasos", "total": pasos_reales_experto(cfg),
                            "campo": "experto.pasos_totales",
                            "bloque": "experto.trpo.pasos_por_iteracion"},
                "bc": {"unidad": "epocas", "total": cfg["bc"]["epocas_maximas"],
                       "campo": "bc.epocas_maximas"},
                "gail": {"unidad": "iteraciones", "total": cfg["gail"]["iteraciones"],
                         "campo": "gail.iteraciones"}}[metodo]
    assert esquema["duracion"] == duracion
    # Los campos que cita existen en el formulario y el total sale de sus valores.
    campos = _campos(esquema)
    valor = campos[duracion["campo"]]["valor"]
    if "bloque" in duracion:
        bloque = campos[duracion["bloque"]]["valor"]
        valor = -(-valor // bloque) * bloque
    assert valor == duracion["total"]
    assert esquema["semilla"] == (cfg["experto"]["semilla"] if metodo == "experto"
                                  else cfg["experimento"]["semilla_repeticiones"])
    assert esquema["semilla"] == semilla_por_defecto(cfg, metodo)


@pytest.mark.parametrize("presupuesto, bloque, esperado", [
    (25_000_000, 2048, 25_001_984), (100_000, 5000, 100_000), (1, 8, 8), (8, 8, 8),
    (9, 8, 16), (5000, 1, 5000),
])
def test_pasos_reales_del_experto(presupuesto, bloque, esperado) -> None:
    """Regresión: se anunciaba el presupuesto (25 000 000 en Reacher y HalfCheetah) y TRPO
    da bloques completos hasta pasarlo (25 001 984). Mismo bucle que entrenar_experto."""
    cfg = resolver_configuracion(CONFIG, "cartpole")
    cfg["experto"]["pasos_totales"] = presupuesto
    cfg["experto"]["trpo"] = {"pasos_por_iteracion": bloque}
    pasos = 0
    while pasos < presupuesto:  # como el bucle de entrenar_experto
        pasos += bloque
    assert pasos_reales_experto(cfg) == pasos == esperado


@pytest.mark.parametrize("entorno, metodo", TODOS)
def test_secciones_de_cada_metodo(entorno, metodo) -> None:
    esquema = esquema_ajustes(CONFIG, entorno, metodo, DASHBOARD)
    claves = [s["clave"] for s in esquema["secciones"]]
    assert claves == {"experto": ["experto", "experto.trpo"],
                      "bc": ["bc", "demostraciones"],
                      "gail": ["gail", "gail.discriminador", "trpo", "demostraciones"]}[metodo]
    campos = _campos(esquema)
    # Ni la semilla (campo propio) ni las subsecciones salen como campos.
    assert "experto.semilla" not in campos and "experto.trpo" not in campos
    assert "gail.discriminador" not in campos
    assert all(s["titulo"] for s in esquema["secciones"])


@pytest.mark.parametrize("entorno", ENTORNOS)
def test_el_experto_muestra_trpo_con_sus_propios_valores_encima(entorno) -> None:
    cfg = resolver_configuracion(CONFIG, entorno)
    campos = _campos(esquema_ajustes(CONFIG, entorno, "experto", DASHBOARD))
    esperado = {**cfg["trpo"], **cfg["experto"]["trpo"]}
    assert {r.split(".", 2)[2]: c["valor"] for r, c in campos.items()
            if r.startswith("experto.trpo.")} == esperado


@pytest.mark.parametrize("entorno", ENTORNOS)
def test_tipos_deducidos(entorno) -> None:
    campos = _campos(esquema_ajustes(CONFIG, entorno, "gail", DASHBOARD))
    assert campos["trpo.capas_ocultas"]["tipo"] == "lista_enteros"
    assert campos["trpo.activacion"]["tipo"] == "opcion"
    assert campos["trpo.activacion"]["opciones"] == ["relu", "tanh"]
    assert campos["gail.discriminador.optimizador"]["opciones"] == ["adam"]
    assert campos["trpo.normalizar_ventajas"]["tipo"] == "booleano"
    assert campos["trpo.pasos_por_iteracion"]["tipo"] == "entero"
    assert campos["trpo.gamma"]["tipo"] == "real"
    assert isinstance(campos["trpo.gamma"]["valor"], float)
    experto = _campos(esquema_ajustes(CONFIG, entorno, "experto", DASHBOARD))
    # Entero en unos perfiles (500) y real en otros (-3.5): siempre real.
    assert experto["experto.retorno_objetivo"]["tipo"] == "real"
    assert isinstance(experto["experto.retorno_objetivo"]["valor"], float)
    bc = _campos(esquema_ajustes(CONFIG, entorno, "bc", DASHBOARD))
    assert bc["bc.optimizador"]["opciones"] == ["adam", "rmsprop", "sgd"]


def test_hay_campos_basicos_en_cada_metodo() -> None:
    for metodo in METODOS:
        campos = _campos(esquema_ajustes(CONFIG, "cartpole", metodo, DASHBOARD))
        basicos = [r for r, c in campos.items() if c["nivel"] == "basico"]
        assert 0 < len(basicos) < len(campos), metodo


def test_dashboard_no_describe_claves_que_no_existen() -> None:
    usadas = set()
    for entorno, metodo in TODOS:
        for ruta in _campos(esquema_ajustes(CONFIG, entorno, metodo, DASHBOARD)):
            seccion, clave = ruta.rsplit(".", 1)
            usadas.add(("trpo" if seccion == "experto.trpo" else seccion, clave))
    descritas = {(s, c) for s, campos in DASHBOARD["campos"].items() for c in campos}
    assert descritas == usadas


def test_las_opciones_coinciden_con_el_codigo() -> None:
    from src.models.factory import ACTIVACIONES
    from src.training.bc import OPTIMIZADORES
    from src.training.gail import MODOS_RECOMPENSA

    opciones = DASHBOARD["opciones"]
    assert sorted(opciones["activacion"]) == sorted(ACTIVACIONES)
    assert sorted(opciones["bc.optimizador"]) == sorted(OPTIMIZADORES)
    assert sorted(opciones["recompensa"]) == sorted(MODOS_RECOMPENSA)
    assert opciones["gail.discriminador.optimizador"] == ["adam"]


def test_un_campo_sin_descripcion_falla() -> None:
    with pytest.raises(ValueError, match="no describe bc.inventado"):
        campo(DASHBOARD, "bc", "inventado", 1)


def test_un_metodo_desconocido_falla() -> None:
    with pytest.raises(ValueError, match="Método desconocido"):
        esquema_ajustes(CONFIG, "cartpole", "ppo", DASHBOARD)


# --- Comprobar ajustes --------------------------------------------------------------

def test_ajustes_validos_se_aplican() -> None:
    cfg = comprobar_ajustes(CONFIG, "cartpole", "gail", {
        "gail": {"iteraciones": 5, "discriminador": {"learning_rate": 0.02}},
        "trpo": {"gamma": 0.98, "capas_ocultas": [32, 32], "activacion": "relu"},
        "demostraciones": {"frecuencia_pares": 2}}, DASHBOARD)
    assert cfg["gail"]["iteraciones"] == 5
    assert cfg["gail"]["discriminador"]["learning_rate"] == 0.02
    assert cfg["trpo"]["gamma"] == 0.98 and cfg["trpo"]["capas_ocultas"] == [32, 32]
    assert cfg["demostraciones"]["frecuencia_pares"] == 2


def test_ajustes_vacios_dan_el_perfil() -> None:
    assert comprobar_ajustes(CONFIG, "acrobot", "bc", {}, DASHBOARD) == \
        resolver_configuracion(CONFIG, "acrobot")


def test_los_ajustes_del_experto_van_a_experto_trpo() -> None:
    cfg = comprobar_ajustes(CONFIG, "cartpole", "experto",
                            {"experto": {"trpo": {"gamma": 0.9}, "pasos_totales": 2048}},
                            DASHBOARD)
    assert cfg["experto"]["trpo"]["gamma"] == 0.9
    assert cfg["experto"]["pasos_totales"] == 2048
    # La sección trpo de GAIL no cambia.
    assert cfg["trpo"]["gamma"] == resolver_configuracion(CONFIG, "cartpole")["trpo"]["gamma"]


def test_un_entero_vale_donde_se_pide_un_real() -> None:
    cfg = comprobar_ajustes(CONFIG, "cartpole", "experto",
                            {"experto": {"retorno_objetivo": 400}}, DASHBOARD)
    assert cfg["experto"]["retorno_objetivo"] == 400


@pytest.mark.parametrize("metodo, ajustes, motivo", [
    ("bc", {"bc": {"inventado": 1}}, "bc.inventado: no es un ajuste de bc"),
    ("bc", {"gail": {"iteraciones": 3}}, "gail.iteraciones: no es un ajuste de bc"),
    ("bc", {"trpo": {"gamma": 0.9}}, "trpo.gamma: no es un ajuste de bc"),
    ("gail", {"experto": {"pasos_totales": 10}}, "no es un ajuste de gail"),
    ("experto", {"trpo": {"gamma": 0.9}}, "trpo.gamma: no es un ajuste de experto"),
    ("experto", {"experto": {"semilla": 3}}, "experto.semilla: no es un ajuste"),
    ("gail", {"gail": {"discriminador": 3}}, "gail.discriminador: no es un ajuste"),
    ("bc", {"bc": [1]}, "bc: no es un ajuste de bc"),
    # Tipos
    ("bc", {"bc": {"batch_size": 1.5}}, "tiene que ser un entero"),
    ("bc", {"bc": {"batch_size": True}}, "tiene que ser un entero"),
    ("bc", {"bc": {"learning_rate": "0.1"}}, "número finito"),
    ("bc", {"bc": {"learning_rate": math.nan}}, "número finito"),
    ("bc", {"bc": {"learning_rate": math.inf}}, "número finito"),
    ("bc", {"bc": {"learning_rate": True}}, "número finito"),
    ("bc", {"bc": {"normalizar_observaciones": 1}}, "true o false"),
    ("bc", {"bc": {"optimizador": "adagrad"}}, "tiene que ser una de"),
    ("gail", {"gail": {"discriminador": {"optimizador": "sgd"}}}, "tiene que ser una de"),
    ("bc", {"bc": {"capas_ocultas": []}}, "lista de enteros"),
    ("bc", {"bc": {"capas_ocultas": [1.5]}}, "lista de enteros"),
    ("bc", {"bc": {"capas_ocultas": "100"}}, "lista de enteros"),
    # Rangos
    ("bc", {"bc": {"capas_ocultas": [100, 0]}}, "como mínimo 1"),
    ("bc", {"bc": {"batch_size": 0}}, "como mínimo 1"),
    ("gail", {"trpo": {"gamma": 1.01}}, "como mucho 1"),
    ("gail", {"trpo": {"gamma": -0.1}}, "como mínimo 0"),
    ("bc", {"bc": {"learning_rate": -0.001}}, "como mínimo 0"),
])
def test_ajustes_que_no_valen(metodo, ajustes, motivo) -> None:
    with pytest.raises(ValueError, match="Ajustes no válidos") as error:
        comprobar_ajustes(CONFIG, "cartpole", metodo, ajustes, DASHBOARD)
    assert motivo in str(error.value)


def test_se_informan_todos_los_errores_a_la_vez() -> None:
    with pytest.raises(ValueError) as error:
        comprobar_ajustes(CONFIG, "cartpole", "bc",
                          {"bc": {"batch_size": 0, "optimizador": "x", "otro": 1}}, DASHBOARD)
    for parte in ("bc.batch_size", "bc.optimizador", "bc.otro"):
        assert parte in str(error.value)


@pytest.mark.parametrize("ajustes", [[1], "bc", 3])
def test_ajustes_que_no_son_diccionarios(ajustes) -> None:
    with pytest.raises(ValueError, match="diccionario anidado"):
        comprobar_ajustes(CONFIG, "cartpole", "bc", ajustes, DASHBOARD)


# --- dashboard.yaml que no vale ------------------------------------------------------

def _dashboard_con(tmp_path: Path, cambiar) -> Path:
    """Copia dashboard.yaml con un cambio y devuelve su ruta."""
    datos = copy.deepcopy(yaml.safe_load((CONFIGS / "dashboard.yaml").read_text(encoding="utf-8")))
    cambiar(datos)
    ruta = tmp_path / "dashboard.yaml"
    ruta.write_text(yaml.safe_dump(datos, allow_unicode=True), encoding="utf-8")
    return ruta


@pytest.mark.parametrize("cambiar, mensaje", [
    (lambda d: d.pop("campos"), "falta la sección campos"),
    (lambda d: d.pop("opciones"), "falta la sección opciones"),
    (lambda d: d["laboratorio"].update(jugar_a_mano=["pong"]), "jugar_a_mano"),
    (lambda d: d["laboratorio"].update(partidas_maximas=0), "partidas_maximas"),
    (lambda d: d["laboratorio"].update(partidas_maximas=True), "partidas_maximas"),
    (lambda d: d["secciones"].pop("bc"), "no tiene título"),
    (lambda d: d["campos"]["bc"]["batch_size"].pop("ayuda"), "falta ayuda"),
    (lambda d: d["campos"]["bc"]["batch_size"].update(etiqueta="  "), "falta etiqueta"),
    (lambda d: d["campos"]["bc"]["batch_size"].update(nivel="medio"), "nivel"),
    (lambda d: d["campos"]["bc"]["batch_size"].update(tipo="texto"), "tipo"),
    (lambda d: d["campos"]["bc"]["batch_size"].update(min="uno"), "min"),
    (lambda d: d["campos"]["bc"]["batch_size"].update(otra=1), "solo admite"),
    (lambda d: d.pop("wandb"), "wandb.grupo"),
    (lambda d: d["wandb"].update(grupo="con espacios"), "wandb.grupo"),
    (lambda d: d["wandb"].update(grupo="dashboard-{otro}"), "wandb.grupo"),
    (lambda d: d["wandb"].update(grupo=3), "wandb.grupo"),
])
def test_dashboard_que_no_vale(tmp_path, cambiar, mensaje) -> None:
    with pytest.raises(ValueError, match=mensaje):
        cargar_dashboard(_dashboard_con(tmp_path, cambiar))


def test_comprobar_esquemas_valida_contra_el_modelo_de_la_respuesta(monkeypatch) -> None:
    """El arranque comprueba cada esquema con EsquemaAjustesApi, no solo que se genere:
    lo que daría 500 al pedirlo impide arrancar."""
    from src.app import ajustes

    original = ajustes.esquema_ajustes

    def con_titulo_numerico(*args, **kwargs):
        esquema = original(*args, **kwargs)
        esquema["secciones"][0]["titulo"] = 7
        return esquema

    monkeypatch.setattr(ajustes, "esquema_ajustes", con_titulo_numerico)
    with pytest.raises(ValueError, match="Formulario de experto en cartpole"):
        ajustes.comprobar_esquemas(CONFIG, DASHBOARD)
