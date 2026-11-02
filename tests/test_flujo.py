"""Integración de las etapas de run.py con un experto y datos temporales."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch
import yaml

from src.training.flujo import (comprobar_datos_congelados, ejecutar_experimento,
                                evaluar_carpeta,
                                generar_y_registrar_datos, obtener_datos_congelados,
                                entrenar_y_guardar_experto)
from src.training.trpo import huella_politica
from src.utils.configuracion import DATOS_PENDIENTES, resolver_configuracion

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
    cfg["datos"]["huella_experto"] = huella_politica(experto / "modelo.zip")
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
    cfg["experimento"]["procesos"] = 2
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
    cfg_erronea = copy.deepcopy(cfg)
    cfg_erronea["datos"]["huella_experto"] = "0" * 64
    with pytest.raises(ValueError, match="no es el experto congelado"):
        comprobar_datos_congelados(cfg_erronea, datos, experto)


def test_una_evaluacion_no_valida_falla_antes_de_entrenar(tmp_path):
    """La sección evaluacion se valida antes de iniciar cualquier barrido."""
    from src.training.flujo import ejecutar_experimento

    cfg, experto, datos = _preparar(tmp_path)
    cfg["evaluacion"]["episodios"] = 0
    with pytest.raises(ValueError, match="evaluacion.episodios"):
        ejecutar_experimento(cfg, datos, experto, tmp_path / "run", "test")
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("procesos", [0, -1, True, 1.5, None])
def test_evaluar_y_guardar_rechaza_un_numero_de_procesos_no_valido(tmp_path, procesos):
    """El reparto de evaluación exige experiment.procesos entero positivo."""
    from src.training.flujo import evaluar_y_guardar

    cfg = {"evaluacion": {"episodios": 1, "semilla": 0, "determinista": True},
           "experimento": {"procesos": procesos}}
    with pytest.raises(ValueError, match="experimento.procesos"):
        evaluar_y_guardar(cfg, [], None, tmp_path / "evaluacion.json", "test")


def test_evaluar_y_guardar_rechaza_una_lista_vacia_de_agentes(tmp_path):
    """No se inicia un reparto de cero tareas."""
    from src.training.flujo import evaluar_y_guardar

    cfg = {"evaluacion": {"episodios": 1, "semilla": 0, "determinista": True},
           "experimento": {"procesos": 1}}
    with pytest.raises(ValueError, match="no hay agentes"):
        evaluar_y_guardar(cfg, [], None, tmp_path / "evaluacion.json", "test")


def test_obtener_datos_congelados_descarga_carpetas_vacias(tmp_path, monkeypatch):
    """Descarga ambos artefactos a carpetas vacías y comprueba su procedencia."""
    import shutil

    cfg, experto_origen, datos_origen = _preparar(tmp_path / "origen")
    datos, experto = tmp_path / "descargados" / "datos", tmp_path / "descargados" / "experto"
    datos.mkdir(parents=True)
    experto.mkdir()
    llamadas = []

    def descargar(_cfg, nombre, destino):
        llamadas.append((nombre, destino))
        origen = datos_origen if nombre == "demo" else experto_origen
        shutil.copytree(origen, destino, dirs_exist_ok=True)
        return destino

    cfg["datos"].update(artefacto_demostraciones="demo", artefacto_experto="experto")
    monkeypatch.setattr("src.training.flujo.descargar_artefacto", descargar)
    obtener_datos_congelados(cfg, datos, experto)
    comprobar_datos_congelados(cfg, datos, experto)
    assert [nombre for nombre, _ in llamadas] == ["demo", "experto"]


def test_obtener_datos_congelados_no_descarga_si_ya_estan_completos(tmp_path, monkeypatch):
    """Las carpetas completas se conservan y no llaman a W&B."""
    cfg, experto, datos = _preparar(tmp_path)
    monkeypatch.setattr("src.training.flujo.descargar_artefacto",
                        lambda *a: pytest.fail("no debe descargar carpetas completas"))
    obtener_datos_congelados(cfg, datos, experto)
    comprobar_datos_congelados(cfg, datos, experto)


def test_obtener_datos_congelados_no_sobrescribe_una_carpeta_incompleta(tmp_path,
                                                                        monkeypatch):
    """Una carpeta ocupada no se modifica y la comprobación posterior la rechaza."""
    import shutil

    cfg, experto_origen, datos_origen = _preparar(tmp_path / "origen")
    datos, experto = tmp_path / "local" / "datos", tmp_path / "local" / "experto"
    datos.mkdir(parents=True)
    (datos / "otro.txt").write_text("contenido ajeno", encoding="utf-8")
    llamadas = []

    def descargar(_cfg, nombre, destino):
        llamadas.append(nombre)
        shutil.copytree(experto_origen, destino, dirs_exist_ok=True)
        return destino

    cfg["datos"].update(artefacto_demostraciones="demo", artefacto_experto="experto")
    monkeypatch.setattr("src.training.flujo.descargar_artefacto", descargar)
    obtener_datos_congelados(cfg, datos, experto)
    assert llamadas == ["experto"]
    with pytest.raises(ValueError, match="no están los datos congelados"):
        comprobar_datos_congelados(cfg, datos, experto)
    assert (datos / "otro.txt").read_text(encoding="utf-8") == "contenido ajeno"


def test_una_descarga_cortada_no_deja_la_carpeta_a_medias(tmp_path, monkeypatch):
    """Regresión: una descarga fallida dejaba una carpeta incompleta que luego se saltaba."""
    cfg, _, _ = _preparar(tmp_path / "origen")
    destino = tmp_path / "local"
    datos, experto = destino / "datos", destino / "experto"

    def cortada(_cfg, _nombre, temporal):
        (Path(temporal) / "demostraciones.npz").write_bytes(b"a medias")
        raise RuntimeError("conexión perdida")

    cfg["datos"].update(artefacto_demostraciones="demo", artefacto_experto="experto")
    monkeypatch.setattr("src.training.flujo.descargar_artefacto", cortada)
    with pytest.raises(RuntimeError, match="conexión perdida"):
        obtener_datos_congelados(cfg, datos, experto)
    assert not datos.exists()
    assert not destino.exists() or not any(destino.iterdir())


def test_cada_resultado_es_de_su_agente_en_serie_y_en_paralelo(tmp_path):
    """Regresión: invertir el reparto de resultados entre agentes pasaba todos los tests."""
    from src.training.evaluacion import evaluar_final
    from src.training.flujo import cargar_agentes, evaluar_y_guardar

    cfg, experto, datos = _preparar(tmp_path)
    ejecutar_experimento(cfg, datos, experto, tmp_path / "run", "t")
    agentes, huella = cargar_agentes(cfg, experto, tmp_path / "run" / "bc",
                                     tmp_path / "run" / "gail")
    esperados = [evaluar_final(a["agente"], cfg, a["estadisticas"])["retornos"]
                 for a in agentes]
    for procesos in (1, 2):
        cfg["experimento"]["procesos"] = procesos
        resultados = evaluar_y_guardar(cfg, agentes, huella,
                                       tmp_path / f"eval-{procesos}.json", "t")
        assert [r["metodo"] for r in resultados] == [a["metodo"] for a in agentes]
        assert [r["retornos"] for r in resultados] == esperados


@pytest.mark.parametrize("clave", ["huella_partidas", "huella_experto"])
def test_comprobar_datos_congelados_avisa_del_perfil_pendiente_antes_que_de_las_carpetas(
    tmp_path, clave
):
    """Sin datos todavía, la causa es el perfil pendiente, no las carpetas que faltan."""
    cfg = _cfg()
    cfg["datos"][clave] = DATOS_PENDIENTES
    with pytest.raises(ValueError, match=rf"datos\.{clave} sigue «pendiente»"):
        comprobar_datos_congelados(cfg, tmp_path / "no-existe", tmp_path / "tampoco")


def test_obtener_datos_congelados_no_descarga_un_artefacto_pendiente(tmp_path, monkeypatch):
    """Ni llama a W&B ni crea la carpeta; si ya está en local, no hace falta el nombre."""
    cfg = _cfg()
    cfg["datos"].update(artefacto_demostraciones=DATOS_PENDIENTES,
                        artefacto_experto=DATOS_PENDIENTES)
    llamadas = []
    monkeypatch.setattr("src.training.flujo.descargar_artefacto",
                        lambda *a: llamadas.append(a))
    with pytest.raises(ValueError, match=r"datos\.artefacto_demostraciones sigue"):
        obtener_datos_congelados(cfg, tmp_path / "datos", tmp_path / "experto")
    assert llamadas == [] and not any(tmp_path.iterdir())

    for carpeta in ("datos", "experto"):
        (tmp_path / carpeta).mkdir()
        (tmp_path / carpeta / "algo.txt").write_text("local", encoding="utf-8")
    obtener_datos_congelados(cfg, tmp_path / "datos", tmp_path / "experto")
    assert llamadas == []


def test_marca_de_tiempo_es_la_fecha_y_hora_actuales_ordenables():
    """AAAAMMDD-HHMMSS: las salidas por defecto se ordenan por fecha al listarlas."""
    from datetime import datetime

    from src.utils.archivos import marca_de_tiempo
    antes = datetime.now().replace(microsecond=0)
    marca = marca_de_tiempo()
    despues = datetime.now()
    assert antes <= datetime.strptime(marca, "%Y%m%d-%H%M%S") <= despues
    assert len(marca) == 15 and marca[8] == "-"

