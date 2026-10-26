"""Pruebas unitarias del adaptador de Weights & Biases."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.utils.tracking import (abrir_run, abrir_run_barrido, comprobar_wandb,
                                crear_grupo, registrar, registrar_resumen,
                                subir_artefacto, terminar_run, usar_artefacto)


def _cfg(modo: str = "online") -> dict:
    """Construye la configuración mínima aceptada por tracking."""
    return {"wandb": {"modo": modo, "proyecto": "prueba", "entidad": "equipo"},
            "experimento": {"nombre": "exp"},
            "entorno": {"id": "CartPole-v1"}}


def test_disabled_no_importa_wandb(monkeypatch):
    """El modo disabled devuelve None sin importar W&B."""
    import builtins

    original = builtins.__import__
    imports = []

    def importar(nombre, *args, **kwargs):
        if nombre == "wandb":
            imports.append(nombre)
            raise AssertionError("W&B no debe importarse")
        return original(nombre, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", importar)
    assert comprobar_wandb(_cfg("disabled"), exigir_sesion=True) is None
    assert abrir_run(_cfg("disabled")) is None
    assert imports == []


def test_abrir_run_pasa_grupo_tipo_nombre_etiquetas_y_config(monkeypatch):
    """Los metadatos del run y la configuración resuelta llegan a wandb.init."""
    llamadas = []
    falso = SimpleNamespace(init=lambda **kwargs: llamadas.append(kwargs) or "run")
    monkeypatch.setitem(sys.modules, "wandb", falso)
    cfg = _cfg("offline")
    assert abrir_run(cfg, "grupo-a", "experto", "nombre", ["entorno:CartPole-v1"]) == "run"
    assert llamadas == [{"project": "prueba", "entity": "equipo", "name": "nombre",
                         "config": cfg, "mode": "offline", "group": "grupo-a",
                         "job_type": "experto", "tags": ["entorno:CartPole-v1"]}]


def test_abrir_run_barrido_identifica_grupo_cantidad_repeticion(monkeypatch):
    """Cada entrenamiento del barrido queda filtrable por método y combinación."""
    capturas = []
    monkeypatch.setattr("src.utils.tracking.abrir_run",
                        lambda *a, **kw: capturas.append((a, kw)) or "run")
    cfg = _cfg("offline")
    assert abrir_run_barrido(cfg, "gail", 4, 2, "grupo-a") == "run"
    assert capturas[0][1] == {"grupo": "grupo-a", "job_type": "gail",
                              "etiquetas": ["entorno:CartPole-v1", "metodo:gail",
                                            "cantidad:4", "repeticion:2"]}
    assert cfg["experimento"]["nombre"] == "exp"


def test_comprobar_sesion_online_falla_si_login_devuelve_false(monkeypatch):
    """La ausencia de sesión se detecta en el principal antes de lanzar trabajo."""
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(
        login=lambda **kwargs: False))
    with pytest.raises(RuntimeError, match="wandb login"):
        comprobar_wandb(_cfg("online"), exigir_sesion=True)


def test_comprobar_sesion_online_verifica_credenciales_sin_argumentos_no_compatibles(monkeypatch):
    """El preflight usa solo argumentos disponibles en la versión fijada de W&B."""
    llamadas = []

    def login(anonymous, timeout, verify):
        llamadas.append({"anonymous": anonymous, "timeout": timeout, "verify": verify})
        return True

    falso = SimpleNamespace(login=login)
    monkeypatch.setitem(sys.modules, "wandb", falso)
    assert comprobar_wandb(_cfg("online"), exigir_sesion=True) is falso
    assert llamadas == [{"anonymous": "never", "timeout": 10, "verify": True}]


def test_registrar_solo_envia_metricas_escalares():
    """Solo se envían escalares numéricos de Python y NumPy, no contenedores/tensores."""
    llamadas = []
    run = SimpleNamespace(log=lambda *a, **kw: llamadas.append((a, kw)))
    registrar(run, {"loss": 0.5, "escalar_numpy": np.float32(0.25),
                    "retornos": [1.0, 2.0], "longitudes": (3, 4), "mapa": {"a": 1},
                    "array": np.asarray(2.0), "tensor": torch.tensor(3.0)}, 12)
    assert llamadas == [(({"loss": 0.5, "escalar_numpy": np.float32(0.25)},), {"step": 12})]


def test_registrar_rechaza_un_paso_que_disminuye():
    """Un mismo run no acepta pasos menores que uno ya registrado."""
    run = SimpleNamespace(log=lambda *a, **kw: None)
    registrar(run, {"loss": 1.0}, 12)
    with pytest.raises(ValueError, match="disminuyó"):
        registrar(run, {"loss": 0.5}, 11)


def test_registrar_resumen_subir_artefacto_y_cerrar_run(tmp_path):
    """El resumen y los archivos finales se adjuntan y el cierre indica fallo."""
    archivo = tmp_path / "modelo.pt"
    archivo.write_bytes(b"modelo")
    artefactos = []

    class FakeArtifact:
        def __init__(self, nombre, type, metadata):
            self.nombre, self.type, self.metadata = nombre, type, metadata
            self.archivos = []

        def add_file(self, path, name):
            self.archivos.append((path, name))

    fake_wandb = SimpleNamespace(Artifact=FakeArtifact)
    run = SimpleNamespace(summary={}, finish=lambda **kw: artefactos.append(("finish", kw)),
                          log_artifact=lambda artifact, aliases: artefactos.append(
                              (artifact, aliases)))
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    registrar_resumen(run, {"pasos": 25})
    subir_artefacto(run, "modelo", "bc", [archivo], {"huella": "abc"})
    terminar_run(run, ValueError("fallo"))
    monkeypatch.undo()
    artefacto = artefactos[0][0]
    assert run.summary["pasos"] == 25
    assert {"codigo", "entorno_ejecucion"} <= run.summary.keys()
    assert artefacto.archivos == [(str(archivo), "modelo.pt")]
    assert artefactos[1] == ("finish", {"exit_code": 1})


def test_usar_artefacto_declara_el_linaje_de_entrada(monkeypatch):
    """BC/GAIL y comparación identifican el artefacto de datos o modelo consumido."""
    llamadas = []
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace())
    run = SimpleNamespace(use_artifact=lambda nombre, type: llamadas.append((nombre, type)))
    assert usar_artefacto(run, "demostraciones-a1b2:latest") is None
    assert llamadas == [("demostraciones-a1b2:latest", "demostraciones")]


def test_crear_grupo_devuelve_identificadores_no_vacios_y_distintos():
    """Cada ejecución puede crear un grupo nuevo y breve."""
    primero, segundo = crear_grupo(), crear_grupo()
    assert primero and segundo and primero != segundo
