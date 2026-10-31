"""Pruebas unitarias del adaptador de Weights & Biases."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.utils.configuracion import resolver_configuracion
from src.utils.tracking import (CARACTERES_HUELLA_ARTEFACTO, abrir_run, abrir_run_barrido,
                                comprobar_wandb, con_seccion, crear_grupo, descargar_artefacto,
                                nombre_artefacto, nombre_run, registrar, registrar_resumen,
                                subir_artefacto, terminar_run, usar_artefacto)

RAIZ = Path(__file__).resolve().parent.parent
HUELLA = "0123456789abcdef" * 4


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
    assert abrir_run_barrido(cfg, "gail", np.int64(4), np.int64(2), "grupo-a") == "run"
    assert capturas[0][1] == {"grupo": "grupo-a", "job_type": "gail",
                              "etiquetas": ["entorno:CartPole-v1", "metodo:gail",
                                            "cantidad:4", "repeticion:2"]}
    # En la config para que W&B pueda agrupar por ellos; enteros de Python para el JSON.
    barrido = capturas[0][0][0]["barrido"]
    assert barrido == {"metodo": "gail", "cantidad": 4, "repeticion": 2}
    assert all(type(barrido[k]) is int for k in ("cantidad", "repeticion"))
    assert cfg["experimento"]["nombre"] == "exp" and "barrido" not in cfg


@pytest.mark.parametrize("metricas,esperado", [
    ({}, {}),
    ({"pasos": 1.0}, {"trpo/pasos": 1.0}),
    ({"discriminador/perdida": 0.5}, {"discriminador/perdida": 0.5}),
    ({"pasos": 1.0, "reajuste/perdida_valor": float("inf"), "partidas": 0.0},
     {"trpo/pasos": 1.0, "reajuste/perdida_valor": float("inf"), "trpo/partidas": 0.0}),
    ({"a/b/c": 2}, {"a/b/c": 2}),
])
def test_con_seccion_solo_prefija_las_metricas_sin_seccion(metricas, esperado):
    """Las que ya tienen sección se respetan; los valores no cambian."""
    original = dict(metricas)
    assert con_seccion(metricas, "trpo") == esperado
    assert metricas == original


def test_nombre_run_identifica_metodo_entorno_cantidad_y_repeticion():
    """El nombre distingue método, entorno y, si procede, cada entrenamiento."""
    cfg = {"experimento": {"nombre": "tiiac"}, "entorno": {"id": "CartPole-v1"},
           "origen": {"entorno": "cartpole"}}
    assert nombre_run(cfg, "experto") == "tiiac-experto-cartpole"
    assert nombre_run(cfg, "bc", 4, 2) == "tiiac-bc-cartpole-c4-r2"
    sin_perfil = {"experimento": {"nombre": "tiiac"},
                  "entorno": {"id": "CartPole-v1"}}
    assert nombre_run(sin_perfil, "datos") == "tiiac-datos-CartPole-v1"


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


def test_descargar_artefacto_usa_entidad_proyecto_y_devuelve_destino(monkeypatch, tmp_path):
    """Pide el artefacto latest de la ruta configurada y devuelve el destino."""
    class Error(Exception):
        pass

    rutas, destinos = [], []

    class Artefacto:
        def download(self, root):
            destinos.append(root)
            Path(root).mkdir(parents=True)
            return root

    class Api:
        def artifact(self, ruta):
            rutas.append(ruta)
            return Artefacto()

    falso = SimpleNamespace(Api=Api, errors=SimpleNamespace(Error=Error))
    monkeypatch.setitem(sys.modules, "wandb", falso)
    cfg = _cfg("disabled")
    destino = tmp_path / "descargado"
    assert descargar_artefacto(cfg, "datos-fijos", destino) == destino
    assert rutas == ["equipo/prueba/datos-fijos:latest"]
    assert destinos == [str(destino)]


def test_descargar_artefacto_convierte_artefacto_ausente_a_runtime_error(monkeypatch,
                                                                         tmp_path):
    """La ausencia de un artefacto se traduce a RuntimeError para la CLI."""
    class Error(Exception):
        pass

    class Api:
        def artifact(self, _ruta):
            raise ValueError("no existe")

    falso = SimpleNamespace(Api=Api, errors=SimpleNamespace(Error=Error))
    monkeypatch.setitem(sys.modules, "wandb", falso)
    with pytest.raises(RuntimeError, match="no se pudo descargar"):
        descargar_artefacto(_cfg("disabled"), "ausente", tmp_path / "datos")


def test_descargar_artefacto_sin_sesion_da_runtime_error(monkeypatch, tmp_path):
    """Regresión: sin sesión, wandb lanza UsageError y run.py terminaba con una traza."""
    class Error(Exception):
        pass

    class UsageError(Error):
        pass

    class Api:
        def __init__(self):
            raise UsageError("api_key not configured")

    falso = SimpleNamespace(Api=Api, errors=SimpleNamespace(Error=Error))
    monkeypatch.setitem(sys.modules, "wandb", falso)
    with pytest.raises(RuntimeError, match="wandb login"):
        descargar_artefacto(_cfg("disabled"), "datos", tmp_path / "datos")


@pytest.mark.parametrize("argumentos,esperado", [
    (("experto", HUELLA, "CartPole-v1"), "experto-CartPole-v1-0123456789ab"),
    (("demostraciones", HUELLA), "demostraciones-0123456789ab"),
    (("bc", HUELLA, "CartPole-v1", 4, 2), "bc-CartPole-v1-c4-r2-0123456789ab"),
    (("gail", HUELLA, "CartPole-v1", 10, 0), "gail-CartPole-v1-c10-r0-0123456789ab"),
    (("gail", HUELLA, "CartPole-v1", np.int64(1), np.int64(3)),
     "gail-CartPole-v1-c1-r3-0123456789ab"),
])
def test_nombre_artefacto_de_cada_tipo(argumentos, esperado):
    """Los cuatro tipos de artefacto siguen el formato <tipo>[-entorno][-cN-rM]-<huella corta>."""
    assert nombre_artefacto(*argumentos) == esperado
    assert esperado.endswith("-" + HUELLA[:CARACTERES_HUELLA_ARTEFACTO])


# Huellas de demostraciones.npz congeladas: las imprime get_data.py y las guarda el run de
# datos en W&B (el perfil no las guarda; las de CartPole están también en el README).
@pytest.mark.parametrize("entorno,npz", [
    ("cartpole", "5c5168e7f5f8ad8dc415ff8e13ff0acc3d9fe8e0445e4f9d0298f0817198aff9"),
])
def test_nombre_artefacto_reproduce_los_artefactos_congelados(entorno, npz):
    """Regresión: cambiar el formato o la longitud dejaría sin encontrar los congelados."""
    cfg = resolver_configuracion(RAIZ / "configs" / "default.yaml", entorno)
    assert (nombre_artefacto("experto", cfg["datos"]["huella_experto"], cfg["entorno"]["id"])
            == cfg["datos"]["artefacto_experto"])
    assert nombre_artefacto("demostraciones", npz) == cfg["datos"]["artefacto_demostraciones"]


@pytest.mark.parametrize("huella", [
    None, 123, "", HUELLA[:CARACTERES_HUELLA_ARTEFACTO], HUELLA + "0", HUELLA.upper(),
    "g" * 64,
])
def test_nombre_artefacto_rechaza_huellas_que_no_son_sha256(huella):
    """Una huella recortada, en mayúsculas o que no es hexadecimal no da nombre."""
    with pytest.raises(ValueError, match="SHA-256"):
        nombre_artefacto("experto", huella, "CartPole-v1")


@pytest.mark.parametrize("cantidad,repeticion", [(4, None), (None, 0)])
def test_nombre_artefacto_exige_cantidad_y_repeticion_juntas(cantidad, repeticion):
    """Sin la otra mitad, dos modelos distintos podrían compartir nombre."""
    with pytest.raises(ValueError, match="juntas"):
        nombre_artefacto("bc", HUELLA, "CartPole-v1", cantidad, repeticion)


def test_registrar_solo_envia_metricas_escalares():
    """Solo se envían escalares numéricos de Python y NumPy, no contenedores/tensores."""
    llamadas = []
    run = SimpleNamespace(log=lambda *a, **kw: llamadas.append((a, kw)))
    registrar(run, {"loss": 0.5, "escalar_numpy": np.float32(0.25),
                    "retornos": [1.0, 2.0], "longitudes": (3, 4), "mapa": {"a": 1},
                    "array": np.asarray(2.0), "tensor": torch.tensor(3.0)}, 12)
    assert llamadas == [(({"loss": 0.5, "escalar_numpy": np.float32(0.25)},), {"step": 12})]


def test_registrar_omite_un_paso_que_disminuye(capsys):
    """Un paso W&B decreciente avisa y omite sus métricas sin cortar el entrenamiento."""
    llamadas = []
    run = SimpleNamespace(log=lambda *a, **kw: llamadas.append((a, kw)))
    registrar(run, {"loss": 1.0}, 12)
    registrar(run, {"loss": 0.5}, 11)
    assert len(llamadas) == 1
    assert "paso W&B decreciente" in capsys.readouterr().err


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


@pytest.mark.parametrize("tipo_error", ["ausente", "red"])
def test_usar_artefacto_ausente_avisa_y_no_propaga_el_error(monkeypatch, capsys, tipo_error):
    """La ausencia del artefacto o un error de W&B no detienen el trabajo."""
    class CommError(Exception):
        pass

    falso = SimpleNamespace(errors=SimpleNamespace(CommError=CommError))
    monkeypatch.setitem(sys.modules, "wandb", falso)

    def falla(nombre, type):
        error = ValueError("artifact not found") if tipo_error == "ausente" else CommError("offline")
        raise error

    run = SimpleNamespace(use_artifact=falla)
    assert usar_artefacto(run, "experto-CartPole-v1-abcd1234:latest", "experto") is None
    aviso = capsys.readouterr().err
    assert "experto-CartPole-v1-abcd1234:latest" in aviso
    assert "se sigue sin registrar el linaje" in aviso


def test_crear_grupo_devuelve_identificadores_no_vacios_y_distintos():
    """Cada ejecución puede crear un grupo nuevo y breve."""
    primero, segundo = crear_grupo(), crear_grupo()
    assert primero and segundo and primero != segundo
