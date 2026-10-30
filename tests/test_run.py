"""Pruebas de la interfaz de línea de comandos del experimento completo."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.data.demostraciones import generar_dataset
from src.utils.configuracion import resolver_configuracion

RAIZ = Path(__file__).resolve().parents[1]


def _ejecutar(*argumentos: str) -> subprocess.CompletedProcess[str]:
    """Ejecuta run.py desde la raíz del repositorio."""
    return subprocess.run([sys.executable, "run.py", *argumentos], cwd=RAIZ, text=True,
                          capture_output=True, check=False)


def _archivos(ruta: Path) -> set[Path]:
    """Devuelve las rutas relativas de los archivos, si el directorio existe."""
    return ({archivo.relative_to(ruta) for archivo in ruta.rglob("*") if archivo.is_file()}
            if ruta.exists() else set())


def test_smoke_llega_a_metricas_y_no_deja_datos_del_experimento(tmp_path):
    antes = {nombre: _archivos(RAIZ / nombre) for nombre in ("checkpoints", "data")}
    inicio = time.perf_counter()
    resultado = _ejecutar("--smoke", "--sin-wandb")
    transcurrido = time.perf_counter() - inicio
    assert resultado.returncode == 0, resultado.stderr
    assert "retorno_medio " in resultado.stdout
    assert transcurrido < 300
    assert {nombre: _archivos(RAIZ / nombre) for nombre in antes} == antes


def test_huella_de_datos_incorrecta_y_solo_evaluar_sin_salida_salen_con_2(tmp_path):
    cfg = resolver_configuracion(RAIZ / "configs" / "default.yaml", "cartpole",
                                 {"wandb": {"modo": "disabled"}})
    cfg["demostraciones"].update(partidas_train=2, partidas_validacion=1, subconjuntos=[1, 2])
    datos = tmp_path / "datos"
    generar_dataset(cfg, lambda observacion: np.int64(observacion[2] > 0), datos,
                    {"entorno": "CartPole-v1", "experto": "política de prueba",
                     "procedencia": "test_run", "configuracion_experto": {}})
    experto = tmp_path / "experto"
    experto.mkdir()
    (experto / "modelo.zip").write_bytes(b"modelo de prueba")
    (experto / "experto.json").write_text("{}", encoding="utf-8")

    distinto = _ejecutar("--datos", str(datos), "--experto", str(experto), "--sin-wandb")
    sin_salida = _ejecutar("--solo-evaluar", "--sin-wandb")
    assert distinto.returncode == 2
    assert "huella de las partidas" in distinto.stderr
    assert sin_salida.returncode == 2
    assert "--solo-evaluar necesita --salida" in sin_salida.stderr


def test_error_al_descargar_datos_sale_con_2(monkeypatch, capsys):
    """Una descarga fallida se comunica como error de configuración, código 2."""
    import run

    monkeypatch.setattr(sys, "argv", ["run.py", "--entorno", "cartpole", "--descargar",
                                       "--sin-wandb"])

    def fallar(*_args):
        raise RuntimeError("descarga de prueba")

    monkeypatch.setattr(run, "obtener_datos_congelados", fallar)
    assert run.main() == 2
    assert "descarga de prueba" in capsys.readouterr().err


def test_un_perfil_con_datos_pendientes_hace_el_smoke_pero_no_el_experimento(tmp_path):
    """El marcador deja generar (el smoke recorre experto y datos) y para el experimento."""
    configs = tmp_path / "configs"
    (configs / "entornos").mkdir(parents=True)
    (configs / "default.yaml").write_bytes((RAIZ / "configs" / "default.yaml").read_bytes())
    perfil = yaml.safe_load((RAIZ / "configs" / "entornos" / "cartpole.yaml").read_text(
        encoding="utf-8"))
    perfil["datos"] = {clave: "pendiente" for clave in perfil["datos"]}
    (configs / "entornos" / "cartpole.yaml").write_text(yaml.safe_dump(perfil),
                                                         encoding="utf-8")
    base = str(configs / "default.yaml")

    smoke = _ejecutar("--config", base, "--smoke", "--sin-wandb")
    experimento = _ejecutar("--config", base, "--sin-wandb")
    reevaluar = _ejecutar("--config", base, "--solo-evaluar", "--salida",
                          str(tmp_path / "run"), "--sin-wandb")
    assert smoke.returncode == 0, smoke.stderr
    for resultado in (experimento, reevaluar):
        assert resultado.returncode == 2
        assert "sigue «pendiente»" in resultado.stderr


def test_una_carpeta_pendiente_se_nombra_aunque_las_huellas_esten(tmp_path):
    """Con solo la ruta pendiente, el error dice cuál y no busca una carpeta «pendiente»."""
    configs = tmp_path / "configs"
    (configs / "entornos").mkdir(parents=True)
    (configs / "default.yaml").write_bytes((RAIZ / "configs" / "default.yaml").read_bytes())
    perfil = yaml.safe_load((RAIZ / "configs" / "entornos" / "cartpole.yaml").read_text(
        encoding="utf-8"))
    perfil["datos"]["demostraciones"] = "pendiente"
    (configs / "entornos" / "cartpole.yaml").write_text(yaml.safe_dump(perfil),
                                                         encoding="utf-8")
    resultado = _ejecutar("--config", str(configs / "default.yaml"), "--sin-wandb")
    assert resultado.returncode == 2
    assert "datos.demostraciones sigue «pendiente»" in resultado.stderr



@pytest.mark.parametrize("entorno", ["mountaincar", "acrobot", "reacher", "halfcheetah"])
def test_los_entornos_nuevos_pasan_el_smoke(entorno):
    """El flujo real en miniatura funciona aunque sus datos sigan pendientes."""
    resultado = _ejecutar("--smoke", "--entorno", entorno, "--sin-wandb")
    assert resultado.returncode == 0, resultado.stderr
    assert resultado.stdout.count("retorno_medio ") == 6


def test_el_experimento_de_un_entorno_pendiente_nombra_su_entorno():
    resultado = _ejecutar("--entorno", "mountaincar", "--sin-wandb")
    assert resultado.returncode == 2
    assert "sigue «pendiente» en el perfil de MountainCar-v0" in resultado.stderr
