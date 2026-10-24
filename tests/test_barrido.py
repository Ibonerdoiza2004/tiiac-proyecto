"""Pruebas de los barridos de BC (scripts/entrenar_bc.py), de su tarea
(entrenar_y_guardar_bc) y de la evaluación final (scripts/evaluar.py).

El reparto en procesos, los fallos y Ctrl+C de main_barrido se prueban con el
script de GAIL en test_gail.py; aquí, lo propio de BC y de la evaluación, con un
flujo pequeño de verdad en CartPole: un experto de pocos pasos, partidas
grabadas con una política fija y BC y GAIL de 2 iteraciones o 3 épocas.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.demostraciones import generar_dataset
from src.training.barrido import cargar_datos
from src.training.bc import cargar_politica_bc, entrenar_bc_subconjunto, entrenar_y_guardar_bc
from src.training.evaluacion import evaluar_final
from src.training.gail import cargar_gail
from src.training.trpo import cargar_experto, entrenar_experto, guardar_experto
from src.utils.configuracion import resolver_configuracion
from src.utils.seed import crear_generador, fijar_seed, semillas_repeticion

RAIZ = Path(__file__).resolve().parent.parent


def _cfg() -> dict:
    """Configuración del proyecto con presupuestos mínimos y W&B desactivado."""
    cfg = resolver_configuracion(RAIZ / "configs" / "default.yaml", "cartpole",
                                 {"wandb": {"modo": "disabled"}})
    cfg["demostraciones"].update(partidas_train=4, partidas_validacion=2, subconjuntos=[1, 4])
    cfg["experimento"].update(repeticiones=2, procesos=1)
    cfg["trpo"].update(pasos_por_iteracion=64, n_entornos=2, batch_size_valor=32,
                       capas_ocultas=[8], pasos_valor=2)
    cfg["experto"].update(pasos_totales=256, evaluar_cada=2, episodios_evaluacion=2)
    cfg["gail"].update(iteraciones=2, evaluar_cada=2, episodios_evaluacion=2)
    cfg["gail"]["discriminador"]["capas_ocultas"] = [8]
    cfg["bc"].update(capas_ocultas=[8], epocas_maximas=3, paciencia=1)
    cfg["evaluacion"]["episodios"] = 2
    return cfg


def _script(nombre: str):
    """Carga un script como módulo (scripts/ no es un paquete)."""
    spec = importlib.util.spec_from_file_location(nombre, RAIZ / "scripts" / f"{nombre}.py")
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def _ejecutar(monkeypatch, nombre: str, argumentos: list[str], cfg: dict | None = None,
              antes=None) -> int:
    """Ejecuta un script con argumentos y la configuración pequeña (o cfg).

    antes(script), si se da, puede sustituir algo del script antes de ejecutarlo.
    """
    script = _script(nombre)

    def resolver(*argumentos_resolver, **opciones):
        return copy.deepcopy(cfg or _cfg())

    # Los de barrido resuelven la configuración en main_barrido; evaluar, en el script.
    if nombre == "evaluar":
        monkeypatch.setattr(script, "resolver_configuracion", resolver)
    else:
        monkeypatch.setattr("src.training.barrido.resolver_configuracion", resolver)
    if antes is not None:
        antes(script)
    monkeypatch.setattr("sys.argv", [f"{nombre}.py", *argumentos])
    return script.main()


@pytest.fixture(scope="module")
def flujo(tmp_path_factory):
    """Experto, datos, BC y GAIL del barrido pequeño, guardados en disco."""
    carpeta = tmp_path_factory.mktemp("flujo")
    cfg = _cfg()
    modelo, resumen = entrenar_experto(cfg)
    guardar_experto(carpeta / "experto", modelo, cfg, resumen)
    generar_dataset(cfg, lambda observacion: np.int64(observacion[2] > 0), carpeta / "datos",
                    {"entorno": "CartPole-v1", "experto": "política fija de prueba",
                     "procedencia": "tests", "configuracion_experto": {}})
    with pytest.MonkeyPatch.context() as monkeypatch:
        for nombre in ("entrenar_bc", "entrenar_gail"):
            metodo = nombre.split("_")[1]
            assert _ejecutar(monkeypatch, nombre, ["--datos", str(carpeta / "datos"),
                                                   "--salida", str(carpeta / metodo)]) == 0
    return carpeta


# --- barrido de BC ---------------------------------------------------------------

def test_entrenar_bc_guarda_cada_cantidad_y_repeticion_con_su_procedencia(flujo):
    cfg = _cfg()
    _, splits, datos = cargar_datos(cfg, flujo / "datos")
    for cantidad in (1, 4):
        for repeticion in (0, 1):
            politica, meta = cargar_politica_bc(
                flujo / "bc" / f"cantidad_{cantidad}" / f"repeticion_{repeticion}.pt")
            semillas = semillas_repeticion(cfg["experimento"]["semilla_repeticiones"], repeticion)
            assert (meta["subconjunto"], meta["semilla"]) == (cantidad, semillas["modelo"])
            assert meta["huella_dataset"] == splits.huella_dataset
            resumen = meta["adicional"]["resumen"]
            assert (resumen["cantidad"], resumen["repeticion"]) == (cantidad, repeticion)
            assert resumen["semillas"] == {"base": cfg["experimento"]["semilla_repeticiones"],
                                           **semillas}
            assert resumen["segundos"] > 0
            assert meta["adicional"]["datos"] == datos
            assert not politica.training


def test_el_bc_del_barrido_es_el_de_entrenarlo_a_mano(flujo):
    """Las semillas de la repetición: fijar_seed con la del modelo y los pares con la
    suya, como GAIL. Así cada BC es reproducible y usa los mismos pares que su GAIL."""
    cfg = _cfg()
    trayectorias, splits, _ = cargar_datos(cfg, flujo / "datos")
    semillas = semillas_repeticion(cfg["experimento"]["semilla_repeticiones"], 1)
    from src.data.entornos import crear_entorno
    entorno = crear_entorno(cfg, None)
    espacios = entorno.observation_space, entorno.action_space
    entorno.close()
    antes = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        fijar_seed(semillas["modelo"])
        a_mano, _ = entrenar_bc_subconjunto(trayectorias, splits, 4, *espacios, cfg,
                                            crear_generador(semillas["pares"]))
    finally:
        torch.set_num_threads(antes)
    guardada, _ = cargar_politica_bc(flujo / "bc" / "cantidad_4" / "repeticion_1.pt")
    estado = guardada.state_dict()
    assert all(torch.equal(valor, estado[clave]) for clave, valor in a_mano.state_dict().items())


def test_bc_en_paralelo_da_lo_mismo_que_en_serie(flujo, tmp_path, monkeypatch):
    cfg = _cfg()
    cfg["experimento"]["procesos"] = 2
    assert _ejecutar(monkeypatch, "entrenar_bc", ["--datos", str(flujo / "datos"), "--salida",
                                                  str(tmp_path / "bc")], cfg) == 0
    for cantidad in (1, 4):
        for repeticion in (0, 1):
            ruta = Path(f"cantidad_{cantidad}") / f"repeticion_{repeticion}.pt"
            serie, _ = cargar_politica_bc(flujo / "bc" / ruta)
            paralelo, _ = cargar_politica_bc(tmp_path / "bc" / ruta)
            estado = paralelo.state_dict()
            assert all(torch.equal(v, estado[k]) for k, v in serie.state_dict().items())


@pytest.mark.parametrize("seccion,clave,valor,mensaje", [
    ("bc", "batch_size", 0, "bc.batch_size"),
    ("bc", "capas_ocultas", [0], "capas_ocultas"),
    ("bc", "activacion", "sigmoide", "activacion"),
    ("bc", "optimizador", "lbfgs", "bc.optimizador"),
    ("bc", "epocas_maximas", 0, "bc.epocas_maximas"),
    ("experimento", "procesos", 0, "experimento.procesos"),
    ("demostraciones", "frecuencia_pares", 0, "frecuencia_pares"),
])
def test_entrenar_bc_rechaza_la_config_antes_de_entrenar(flujo, tmp_path, monkeypatch, capsys,
                                                          seccion, clave, valor, mensaje):
    cfg = _cfg()
    cfg[seccion][clave] = valor
    entrenados = []
    monkeypatch.setattr("src.training.bc.entrenar_bc_subconjunto",
                        lambda *a, **k: entrenados.append(1))
    assert _ejecutar(monkeypatch, "entrenar_bc", ["--datos", str(flujo / "datos"), "--salida",
                                                  str(tmp_path / "bc")], cfg) == 2
    assert not entrenados and not (tmp_path / "bc").exists()
    assert mensaje in capsys.readouterr().err


class _RunFalso:
    def __init__(self) -> None:
        self.codigos: list[int] = []

    def log(self, metricas, step=None) -> None:
        pass

    def finish(self, exit_code: int) -> None:
        self.codigos.append(exit_code)


def test_entrenar_y_guardar_bc_cierra_su_run_y_marca_los_fallos(flujo, tmp_path, monkeypatch):
    cfg = _cfg()
    trayectorias, splits, datos = cargar_datos(cfg, flujo / "datos")
    runs = []
    monkeypatch.setattr("src.utils.tracking.abrir_run",
                        lambda c, **_: runs.append(_RunFalso()) or runs[-1])
    destino = tmp_path / "bc.pt"
    entrenar_y_guardar_bc(cfg, trayectorias, splits, 1, 0, destino, datos)
    with pytest.raises(FileExistsError):  # el guardado falla: destino ya existe
        entrenar_y_guardar_bc(cfg, trayectorias, splits, 1, 0, destino, datos)
    assert [run.codigos for run in runs] == [[0], [1]]


def test_entrenar_y_guardar_bc_rechaza_datos_de_otro_npz(flujo, tmp_path, monkeypatch):
    cfg = _cfg()
    trayectorias, splits, datos = cargar_datos(cfg, flujo / "datos")
    abiertos = []
    monkeypatch.setattr("src.utils.tracking.abrir_run",
                        lambda c, **_: abiertos.append(1))
    with pytest.raises(ValueError, match="huella del NPZ"):
        entrenar_y_guardar_bc(cfg, trayectorias, splits, 1, 0, tmp_path / "bc.pt",
                              {**datos, "huella_dataset": "b" * 64})
    assert not abiertos and not (tmp_path / "bc.pt").exists()


# --- evaluación final -------------------------------------------------------------

def _argumentos(flujo: Path, salida: Path, *extra: str) -> list[str]:
    return ["--experto", str(flujo / "experto"), "--bc", str(flujo / "bc"),
            "--gail", str(flujo / "gail"), "--salida", str(salida), *extra]


def test_evaluar_juega_todos_los_agentes_con_el_mismo_protocolo(flujo, tmp_path, monkeypatch,
                                                                 capsys):
    salida = tmp_path / "evaluacion.json"
    assert _ejecutar(monkeypatch, "evaluar", _argumentos(flujo, salida)) == 0
    contenido = json.loads(salida.read_text(encoding="utf-8"))
    cfg = _cfg()
    assert contenido["protocolo"] == {"episodios": 2, "semilla": cfg["evaluacion"]["semilla"],
                                      "determinista": cfg["evaluacion"]["determinista"]}
    _, splits, _ = cargar_datos(cfg, flujo / "datos")
    assert contenido["huella_dataset"] == splits.huella_dataset
    assert set(contenido) >= {"entorno", "segundos", "codigo", "entorno_ejecucion"}
    filas = {(f["metodo"], f["cantidad"], f["repeticion"]): f for f in contenido["resultados"]}
    esperadas = ({("experto", None, None), ("aleatorio", None, None)}
                 | {(m, c, r) for m in ("bc", "gail") for c in (1, 4) for r in (0, 1)})
    assert set(filas) == esperadas
    assert all(len(f["retornos"]) == len(f["longitudes"]) == 2 for f in filas.values())
    # Cada fila es evaluar_final del modelo guardado.
    agente, _, meta = cargar_gail(flujo / "gail" / "cantidad_4" / "repeticion_1")
    directa = evaluar_final(agente, cfg, meta["resumen"]["normalizacion"])
    assert filas[("gail", 4, 1)]["retornos"] == directa["retornos"]
    experto, meta = cargar_experto(flujo / "experto")
    assert filas[("experto", None, None)]["retornos"] == evaluar_final(
        experto, cfg, meta["resumen"]["normalizacion"])["retornos"]
    assert "10 agentes evaluados con 2 partidas" in capsys.readouterr().out


def test_evaluar_solo_lo_que_se_pide(flujo, tmp_path, monkeypatch):
    salida = tmp_path / "solo_gail.json"
    assert _ejecutar(monkeypatch, "evaluar", ["--gail", str(flujo / "gail"), "--cantidades", "4",
                                              "--repeticiones", "0", "--salida",
                                              str(salida)]) == 0
    filas = json.loads(salida.read_text(encoding="utf-8"))["resultados"]
    assert [(f["metodo"], f["cantidad"], f["repeticion"]) for f in filas] == [
        ("aleatorio", None, None), ("gail", 4, 0)]


@pytest.mark.parametrize("caso,mensaje", [
    ("salida_ocupada", "ya existe"),
    ("falta_un_modelo", "error con los modelos"),
    ("otros_datos", "mismas demostraciones"),
    ("otro_entorno", "configuración pide"),
    ("repeticion", "repeticiones fuera de rango"),
    ("config", "error de configuración"),
])
def test_evaluar_no_evalua_nada_si_algo_no_es_valido(flujo, tmp_path, monkeypatch, capsys,
                                                      caso, mensaje):
    import shutil
    salida = tmp_path / "evaluacion.json"
    extra, cfg, antes = [], None, None
    carpetas = flujo
    if caso == "salida_ocupada":
        salida.write_text("anterior", encoding="utf-8")
    elif caso == "falta_un_modelo":
        carpetas = tmp_path / "copia"
        shutil.copytree(flujo, carpetas)
        (carpetas / "bc" / "cantidad_4" / "repeticion_1.pt").unlink()
    elif caso == "otros_datos":
        def otra_huella(ruta):
            politica, meta = cargar_politica_bc(ruta)
            return politica, {**meta, "huella_dataset": "b" * 64}

        def antes(script):
            monkeypatch.setattr(script, "cargar_politica_bc", otra_huella)
    elif caso == "otro_entorno":
        cfg = _cfg()
        cfg["entorno"]["id"] = "Acrobot-v1"
    elif caso == "repeticion":
        extra = ["--repeticiones", "2"]
    else:
        cfg = _cfg()
        cfg["evaluacion"]["episodios"] = 0
    evaluados = []
    monkeypatch.setattr("src.training.evaluacion.evaluar_politica",
                        lambda *a, **k: evaluados.append(1))
    assert _ejecutar(monkeypatch, "evaluar", _argumentos(carpetas, salida, *extra), cfg,
                     antes) == 2
    assert not evaluados
    assert mensaje in capsys.readouterr().err
    if caso == "salida_ocupada":
        assert salida.read_text(encoding="utf-8") == "anterior"
    else:
        assert not salida.exists()


def test_evaluar_pasa_a_cada_agente_sus_estadisticas_de_normalizacion(monkeypatch, tmp_path):
    """El experto y GAIL se evalúan con la normalización guardada en sus metadatos. En
    CartPole no se normaliza, así que se comprueba con metadatos que sí la tienen."""
    import argparse
    script = _script("evaluar")
    estadisticas = {"experto": {"media": [1.0]}, "gail": {"media": [2.0]}}

    def meta(metodo):
        return {"resumen": {"normalizacion": estadisticas[metodo]},
                "configuracion": {"entorno": {"id": "CartPole-v1"}},
                "datos": {"huella_dataset": "a" * 64}}

    monkeypatch.setattr(script, "cargar_experto", lambda ruta: ("experto", meta("experto")))
    monkeypatch.setattr(script, "cargar_gail", lambda ruta: ("gail", "d", meta("gail")))
    args = argparse.Namespace(experto=tmp_path, bc=None, gail=tmp_path, cantidades=[4],
                              repeticiones=[0])
    agentes, huella = script._cargar_agentes(_cfg(), args)
    assert {a["metodo"]: a["estadisticas"] for a in agentes} == {
        "aleatorio": None, "experto": estadisticas["experto"], "gail": estadisticas["gail"]}
    assert huella == "a" * 64
