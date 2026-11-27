"""Tests de la carpeta de checkpoints de un entrenamiento (src/training/checkpoints.py)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.biblioteca import CARPETA_EN_CURSO, Biblioteca
from src.training.checkpoints import (ARCHIVO_ARGUMENTOS, ARCHIVO_CHECKPOINT, ARCHIVO_EN_USO,
                                      CLAVES_REANUDAR, CarpetaOcupada, borrar_en_curso,
                                      comprobar_misma_config, en_uso, escribir_checkpoint,
                                      leer_checkpoint, preparar_en_curso)

ARGUMENTOS = {"entorno": "cartpole", "nombre": "Mi BC", "ajustes": {"bc": {"paciencia": 3}},
              "semilla": 7, "datos": ["a" * 12], "cantidad": None, "vista": False,
              "wandb": {"activo": False, "grupo": None}, "calidad_jpeg": 80,
              "checkpoint": "x"}


def test_la_primera_vez_crea_la_carpeta_con_los_argumentos(tmp_path):
    carpeta = tmp_path / CARPETA_EN_CURSO / "t1"
    assert preparar_en_curso(carpeta, "bc", {**ARGUMENTOS, "semilla": np.int64(7)}) is None
    guardados = json.loads((carpeta / ARCHIVO_ARGUMENTOS).read_text(encoding="utf-8"))
    assert guardados == {"metodo": "bc", "argumentos": ARGUMENTOS}
    assert sorted(p.name for p in carpeta.iterdir()) == [ARCHIVO_ARGUMENTOS]
    # Cortado antes del primer checkpoint: no hay nada que reanudar.
    assert preparar_en_curso(carpeta, "bc", ARGUMENTOS) is None


def test_con_checkpoint_lo_devuelve(tmp_path):
    preparar_en_curso(tmp_path / "t", "bc", ARGUMENTOS)
    escribir_checkpoint(tmp_path / "t", "bc", {"estado": {"w": torch.arange(3)}, "n": 2})
    leido = preparar_en_curso(tmp_path / "t", "bc", ARGUMENTOS)
    assert torch.equal(leido["estado"]["w"], torch.arange(3)) and leido["n"] == 2
    assert leido["metodo"] == "bc"


@pytest.mark.parametrize("clave", CLAVES_REANUDAR)
def test_con_otros_argumentos_que_definen_el_entrenamiento_no_se_reanuda(tmp_path, clave):
    preparar_en_curso(tmp_path / "t", "bc", ARGUMENTOS)
    with pytest.raises(ValueError, match=clave):
        preparar_en_curso(tmp_path / "t", "bc", {**ARGUMENTOS, clave: "otro"})


@pytest.mark.parametrize("cambio", [{"vista": True}, {"calidad_jpeg": 50},
                                    {"wandb": {"activo": True, "grupo": "g"}},
                                    {"checkpoint": "y"}])
def test_lo_que_no_cambia_el_entrenamiento_puede_cambiar(tmp_path, cambio):
    preparar_en_curso(tmp_path / "t", "bc", ARGUMENTOS)
    assert preparar_en_curso(tmp_path / "t", "bc", {**ARGUMENTOS, **cambio}) is None


def test_una_carpeta_de_otro_metodo_no_se_reanuda(tmp_path):
    preparar_en_curso(tmp_path / "t", "gail", ARGUMENTOS)
    with pytest.raises(ValueError, match="de gail, no de bc"):
        preparar_en_curso(tmp_path / "t", "bc", ARGUMENTOS)


@pytest.mark.parametrize("contenido", ["{", "[]", '{"metodo": "bc"}'])
def test_unos_argumentos_que_no_valen(tmp_path, contenido):
    carpeta = tmp_path / "t"
    carpeta.mkdir()
    (carpeta / ARCHIVO_ARGUMENTOS).write_text(contenido, encoding="utf-8")
    with pytest.raises(ValueError, match=ARCHIVO_ARGUMENTOS):
        preparar_en_curso(carpeta, "bc", ARGUMENTOS)


def test_una_carpeta_sin_argumentos_ni_checkpoint_empieza_de_cero(tmp_path):
    """Regresión: un corte justo al crearla la dejaba bloqueada para siempre."""
    carpeta = tmp_path / "t"
    carpeta.mkdir()
    (carpeta / ARCHIVO_EN_USO).touch()  # la reserva de un intento anterior
    assert preparar_en_curso(carpeta, "bc", ARGUMENTOS) is None
    guardados = json.loads((carpeta / ARCHIVO_ARGUMENTOS).read_text(encoding="utf-8"))
    assert guardados == {"metodo": "bc", "argumentos": ARGUMENTOS}


def test_un_checkpoint_sin_sus_argumentos_no_se_reanuda(tmp_path):
    carpeta = tmp_path / "t"
    carpeta.mkdir()
    escribir_checkpoint(carpeta, "bc", {"n": 1})
    with pytest.raises(ValueError, match="no sus argumentos"):
        preparar_en_curso(carpeta, "bc", ARGUMENTOS)
    assert not (carpeta / ARCHIVO_ARGUMENTOS).exists()


# --- reserva de la carpeta -----------------------------------------------------------------

def test_en_uso_crea_la_carpeta_y_la_reserva(tmp_path):
    carpeta = tmp_path / "a" / "t"
    with en_uso(carpeta):
        assert carpeta.is_dir() and (carpeta / ARCHIVO_EN_USO).is_file()
        with pytest.raises(CarpetaOcupada, match="Ya hay un entrenamiento"):
            with en_uso(carpeta):
                pass
    with en_uso(carpeta):  # al salir se suelta
        pass


def test_en_uso_se_suelta_aunque_falle_el_bloque(tmp_path):
    with pytest.raises(RuntimeError):
        with en_uso(tmp_path / "t"):
            raise RuntimeError("falla dentro")
    with en_uso(tmp_path / "t"):
        pass


def test_carpeta_ocupada_es_un_value_error():
    assert issubclass(CarpetaOcupada, ValueError)


def test_en_uso_entre_procesos(tmp_path):
    """Otro proceso con la carpeta reservada: aquí se rechaza hasta que la suelta."""
    import subprocess
    import sys
    carpeta = tmp_path / "t"
    codigo = ("import sys\n"
              "from src.training.checkpoints import en_uso\n"
              "with en_uso(sys.argv[1]):\n"
              "    print('reservada', flush=True)\n"
              "    sys.stdin.readline()\n")
    otro = subprocess.Popen([sys.executable, "-c", codigo, str(carpeta)], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, text=True,
                            cwd=Path(__file__).resolve().parent.parent)
    try:
        assert otro.stdout.readline().strip() == "reservada"
        with pytest.raises(CarpetaOcupada):
            with en_uso(carpeta):
                pass
    finally:
        otro.communicate("\n", timeout=60)
    with en_uso(carpeta):
        pass



def test_escribir_sustituye_el_anterior_y_no_deja_temporales(tmp_path):
    escribir_checkpoint(tmp_path, "bc", {"n": 1})
    escribir_checkpoint(tmp_path, "bc", {"n": 2})
    assert leer_checkpoint(tmp_path, "bc")["n"] == 2
    assert sorted(p.name for p in tmp_path.iterdir()) == [ARCHIVO_CHECKPOINT]


def test_un_fallo_al_escribir_deja_el_anterior(tmp_path, monkeypatch):
    escribir_checkpoint(tmp_path, "bc", {"n": 1})

    def falla(objeto, ruta, *args, **kwargs):
        Path(ruta).write_bytes(b"a medias")  # escribe algo y se corta
        raise OSError("disco lleno")

    monkeypatch.setattr(torch, "save", falla)
    with pytest.raises(OSError):
        escribir_checkpoint(tmp_path, "bc", {"n": 2})
    monkeypatch.undo()
    assert leer_checkpoint(tmp_path, "bc")["n"] == 1


def test_leer_sin_checkpoint_da_none(tmp_path):
    assert leer_checkpoint(tmp_path, "bc") is None


@pytest.mark.parametrize("contenido, mensaje", [
    (b"no es un checkpoint", "dañado"),
    ("pickle", "dañado"),
    ({"formato": "otro", "version": 1, "metodo": "bc"}, "no es un checkpoint"),
    ({"formato": "tiiac-checkpoint", "version": 2, "metodo": "bc"}, "no es un checkpoint"),
    ([1, 2], "no es un checkpoint"),
])
def test_leer_un_checkpoint_que_no_vale(tmp_path, contenido, mensaje):
    ruta = tmp_path / ARCHIVO_CHECKPOINT
    if isinstance(contenido, bytes):
        ruta.write_bytes(contenido)
    elif contenido == "pickle":
        # Un objeto arbitrario: weights_only no lo carga.
        torch.save({"formato": "tiiac-checkpoint", "objeto": object.__new__(_Clase)}, ruta)
    else:
        torch.save(contenido, ruta)
    with pytest.raises(ValueError, match=mensaje):
        leer_checkpoint(tmp_path, "bc")


class _Clase:
    """Clase que torch.load con weights_only no admite."""


def test_leer_un_checkpoint_de_otro_metodo(tmp_path):
    escribir_checkpoint(tmp_path, "gail", {})
    with pytest.raises(ValueError, match="es de gail, no de bc"):
        leer_checkpoint(tmp_path, "bc")


def test_misma_config_salvo_wandb_y_origen():
    cfg = {"bc": {"paciencia": 3, "capas": (8, 8)}, "wandb": {"modo": "disabled"},
           "origen": {"archivos": ["a"]}}
    comprobar_misma_config({**cfg, "bc": {"paciencia": 3, "capas": [8, 8]},
                            "wandb": {"modo": "online"}, "origen": {}}, cfg)
    with pytest.raises(ValueError, match=r"\['bc'\]"):
        comprobar_misma_config({**cfg, "bc": {"paciencia": 4, "capas": [8, 8]}}, cfg)
    with pytest.raises(ValueError, match=r"\['gail'\]"):
        comprobar_misma_config({**cfg, "gail": {}}, cfg)
    with pytest.raises(ValueError, match="configuración"):
        comprobar_misma_config(None, cfg)


def test_borrar_en_curso(tmp_path):
    preparar_en_curso(tmp_path / "t", "bc", ARGUMENTOS)
    escribir_checkpoint(tmp_path / "t", "bc", {})
    borrar_en_curso(tmp_path / "t")
    assert not (tmp_path / "t").exists()
    borrar_en_curso(tmp_path / "t")  # ya no está: no pasa nada


def test_la_biblioteca_no_lista_la_carpeta_en_curso(tmp_path):
    preparar_en_curso(tmp_path / CARPETA_EN_CURSO / "t", "bc", ARGUMENTOS)
    (tmp_path / CARPETA_EN_CURSO / "cartpole" / "bc" / ("a" * 12)).mkdir(parents=True)
    assert Biblioteca(tmp_path, {"cartpole": "CartPole-v1"}).listar() == []
