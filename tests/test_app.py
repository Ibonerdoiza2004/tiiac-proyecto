"""Tests de la aplicación FastAPI (src/app): estado, frontend, configuración y arranque."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient

from src.app import __main__ as arranque
from src.app.api import SIN_FRONTEND, config_aplicativo, crear_app
from src.utils.configuracion import TipoEntorno, leer_configuracion_base

RAIZ = Path(__file__).resolve().parents[1]
CONFIGS = RAIZ / "configs"
GYM = {"cartpole": "CartPole-v1", "mountaincar": "MountainCar-v0", "acrobot": "Acrobot-v1",
       "reacher": "Reacher-v5", "halfcheetah": "HalfCheetah-v5"}


def _configs(tmp_path: Path, **aplicativo: object) -> Path:
    """Copia configs/ en tmp_path con la sección aplicativo cambiada; devuelve el base."""
    destino = tmp_path / "configs"
    shutil.copytree(CONFIGS, destino)
    base = destino / "default.yaml"
    cfg = yaml.safe_load(base.read_text(encoding="utf-8"))
    cfg["aplicativo"].update(aplicativo)
    base.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return base


def _frontend(carpeta: Path) -> Path:
    """Crea un frontend compilado mínimo: index.html y un recurso."""
    (carpeta / "assets").mkdir(parents=True)
    (carpeta / "index.html").write_text("<!doctype html><title>GAIL lab</title>",
                                        encoding="utf-8")
    (carpeta / "assets" / "app.js").write_text("console.log('hola');", encoding="utf-8")
    return carpeta


@pytest.fixture
def cliente_sin_web(tmp_path: Path) -> TestClient:
    """Aplicación con la config real salvo el frontend, que apunta a una carpeta vacía."""
    return TestClient(crear_app(_configs(tmp_path, web=str(tmp_path / "vacia"))))


# --- Estado -----------------------------------------------------------------

def test_estado_da_los_cinco_entornos_y_los_procesos(cliente_sin_web: TestClient) -> None:
    respuesta = cliente_sin_web.get("/api/estado")
    assert respuesta.status_code == 200
    datos = respuesta.json()
    assert [e["id"] for e in datos["entornos"]] == [t.value for t in TipoEntorno]
    assert {e["id"]: e["gym"] for e in datos["entornos"]} == GYM
    assert datos["procesos"] == leer_configuracion_base(CONFIGS / "default.yaml")[
        "experimento"]["procesos"]


def test_ruta_desconocida_de_la_api_da_404_en_formato_fastapi(cliente_sin_web: TestClient) -> None:
    respuesta = cliente_sin_web.get("/api/no-existe")
    assert respuesta.status_code == 404
    assert respuesta.json() == {"detail": "Not Found"}


def test_la_documentacion_openapi_incluye_las_rutas(cliente_sin_web: TestClient) -> None:
    esquema = cliente_sin_web.get("/openapi.json").json()
    assert "/api/estado" in esquema["paths"]
    assert "EstadoApi" in esquema["components"]["schemas"]


# --- Frontend ---------------------------------------------------------------

def test_sin_frontend_compilado_la_raiz_lo_explica(cliente_sin_web: TestClient) -> None:
    respuesta = cliente_sin_web.get("/")
    assert respuesta.status_code == 404
    assert respuesta.json() == {"detail": SIN_FRONTEND}


def test_una_carpeta_sin_index_cuenta_como_sin_frontend(tmp_path: Path) -> None:
    web = tmp_path / "dist"
    (web / "assets").mkdir(parents=True)
    cliente = TestClient(crear_app(_configs(tmp_path, web=str(web))))
    assert cliente.get("/").json() == {"detail": SIN_FRONTEND}


def test_con_frontend_sirve_la_pagina_y_sus_recursos_sin_tapar_la_api(tmp_path: Path) -> None:
    web = _frontend(tmp_path / "dist")
    cliente = TestClient(crear_app(_configs(tmp_path, web=str(web))))
    pagina = cliente.get("/")
    assert pagina.status_code == 200
    assert "GAIL lab" in pagina.text
    assert cliente.get("/assets/app.js").text == "console.log('hola');"
    assert cliente.get("/api/estado").status_code == 200
    assert cliente.get("/api/no-existe").status_code == 404
    assert cliente.get("/no-existe.js").status_code == 404


def test_la_ruta_del_frontend_es_relativa_a_la_carpeta_de_trabajo(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _frontend(tmp_path / "web" / "dist")
    base = _configs(tmp_path, web="web/dist")
    monkeypatch.chdir(tmp_path)
    assert TestClient(crear_app(base)).get("/").status_code == 200


# --- Configuración ----------------------------------------------------------

def _cfg(**cambios: object) -> dict:
    """Configuración base válida con cambios en aplicativo (None borra la clave)."""
    cfg = {"aplicativo": {"host": "127.0.0.1", "puerto": 8000, "web": "web/dist"},
           "experimento": {"procesos": 5}}
    for clave, valor in cambios.items():
        if valor is None:
            del cfg["aplicativo"][clave]
        else:
            cfg["aplicativo"][clave] = valor
    return cfg


def test_config_aplicativo_valida() -> None:
    assert config_aplicativo(_cfg()) == {"host": "127.0.0.1", "puerto": 8000,
                                         "web": Path("web/dist"), "procesos": 5}


@pytest.mark.parametrize("puerto", [1, 65535, np.int64(8000)])
def test_config_aplicativo_admite_puertos_limite_y_enteros_de_numpy(puerto: object) -> None:
    resultado = config_aplicativo(_cfg(puerto=puerto))
    assert resultado["puerto"] == int(puerto)
    assert type(resultado["puerto"]) is int


@pytest.mark.parametrize("cambio, mensaje", [
    ({"host": ""}, "aplicativo.host"),
    ({"host": 127}, "aplicativo.host"),
    ({"puerto": 0}, "aplicativo.puerto"),
    ({"puerto": 65536}, "aplicativo.puerto"),
    ({"puerto": True}, "aplicativo.puerto"),
    ({"puerto": "8000"}, "aplicativo.puerto"),
    ({"puerto": 8000.0}, "aplicativo.puerto"),
    ({"web": ""}, "aplicativo.web"),
    ({"web": 3}, "aplicativo.web"),
    ({"host": None}, "Falta"),
    ({"puerto": None}, "Falta"),
    ({"web": None}, "Falta"),
])
def test_config_aplicativo_rechaza_valores_no_validos(cambio: dict, mensaje: str) -> None:
    with pytest.raises(ValueError, match=mensaje):
        config_aplicativo(_cfg(**cambio))


@pytest.mark.parametrize("procesos", [0, -1, True, 2.0, "5", None])
def test_config_aplicativo_rechaza_procesos_no_validos(procesos: object) -> None:
    cfg = _cfg()
    cfg["experimento"]["procesos"] = procesos
    with pytest.raises(ValueError, match="experimento.procesos"):
        config_aplicativo(cfg)


@pytest.mark.parametrize("cfg", [{}, {"aplicativo": None, "experimento": {"procesos": 5}},
                                 {"aplicativo": {"host": "h", "puerto": 1, "web": "w"}}])
def test_config_aplicativo_rechaza_secciones_que_faltan(cfg: dict) -> None:
    with pytest.raises(ValueError, match="Falta"):
        config_aplicativo(cfg)


def test_crear_app_rechaza_una_config_no_valida(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="aplicativo.puerto"):
        crear_app(_configs(tmp_path, puerto=0))


def test_leer_configuracion_base_no_resuelve_perfiles(tmp_path: Path) -> None:
    cfg = leer_configuracion_base(CONFIGS / "default.yaml")
    assert cfg["entorno"]["id"] is None
    assert "origen" not in cfg
    with pytest.raises(ValueError, match="No se pudo leer"):
        leer_configuracion_base(tmp_path / "no-existe.yaml")


# --- Arranque ---------------------------------------------------------------

def test_main_sirve_en_la_direccion_de_la_config(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    llamadas = []
    monkeypatch.setattr(arranque.uvicorn, "run",
                        lambda app, host, port: llamadas.append((app, host, port)))
    base = _configs(tmp_path, host="0.0.0.0", puerto=8123, web=str(tmp_path / "vacia"))
    assert arranque.main(["--config", str(base)]) == 0
    (app, host, puerto), = llamadas
    assert (host, puerto) == ("0.0.0.0", 8123)
    assert TestClient(app).get("/api/estado").status_code == 200


def test_main_sale_con_2_si_la_config_no_es_valida(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(arranque.uvicorn, "run",
                        lambda *a, **k: pytest.fail("no debería arrancar"))
    assert arranque.main(["--config", str(_configs(tmp_path, puerto=0))]) == 2
    assert "aplicativo.puerto" in capsys.readouterr().err
    assert arranque.main(["--config", str(tmp_path / "no-existe.yaml")]) == 2


def test_la_api_no_importa_el_entrenamiento() -> None:
    """La API lanza el entrenamiento en procesos aparte: no debe cargarlo en el servidor."""
    codigo = ("import sys, src.app.api, src.app.__main__; "
              "print(sorted(m for m in sys.modules if m.startswith('src.training')))")
    salida = subprocess.run([sys.executable, "-c", codigo], cwd=RAIZ, capture_output=True,
                            text=True, check=True).stdout.strip()
    assert salida == "[]"
