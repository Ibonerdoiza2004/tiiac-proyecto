"""Pruebas de la vista en directo: JPEG, ritmo, encendido en marcha y que no cambia nada."""

import io
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image
from stable_baselines3.common.callbacks import BaseCallback

from src.data.entornos import crear_entorno_vectorizado
from src.training.trpo import crear_trpo, entrenar_experto, iteracion_trpo
from src.training.vista import CallbackVista, codificar_jpeg
from src.utils.configuracion import resolver_configuracion
from src.utils.reglas_config import CALIDAD_JPEG_MAXIMA, comprobar_calidad
from src.utils.seed import fijar_seed

RAIZ = Path(__file__).resolve().parent.parent


def _cfg(n_entornos: int = 2) -> dict:
    """CartPole con bloques de 64 pasos y una red pequeña."""
    cfg = resolver_configuracion(RAIZ / "configs" / "default.yaml", "cartpole")
    cfg["trpo"].update(pasos_por_iteracion=64, batch_size_valor=32, n_entornos=n_entornos,
                       capas_ocultas=[16, 16])
    cfg["experto"].update(pasos_totales=256, evaluar_cada=2, episodios_evaluacion=2)
    return cfg


def _imagen(alto: int = 20, ancho: int = 30) -> np.ndarray:
    gradiente = np.linspace(0, 255, alto * ancho * 3).reshape(alto, ancho, 3)
    return gradiente.astype(np.uint8)


# --- JPEG ------------------------------------------------------------------------------

def test_codificar_jpeg_da_un_jpeg_del_mismo_tamano():
    datos = codificar_jpeg(_imagen(), 80)
    assert datos[:2] == b"\xff\xd8"
    with Image.open(io.BytesIO(datos)) as imagen:
        assert imagen.format == "JPEG" and imagen.size == (30, 20) and imagen.mode == "RGB"


def test_mas_calidad_pesa_mas():
    ruido = np.random.default_rng(0).integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    assert len(codificar_jpeg(ruido, 10)) < len(codificar_jpeg(ruido, 90))


def test_codificar_admite_imagenes_no_contiguas():
    """render puede devolver vistas recortadas o traspuestas."""
    recorte = _imagen(40, 40)[::2, ::2]
    assert not recorte.flags["C_CONTIGUOUS"]
    assert codificar_jpeg(recorte, 50)[:2] == b"\xff\xd8"


@pytest.mark.parametrize("imagen", [
    _imagen().astype(np.float32), _imagen()[:, :, 0], np.zeros((4, 4, 4), np.uint8),
    np.zeros((0, 4, 3), np.uint8), [[[0, 0, 0]]], None,
], ids=["float", "gris", "rgba", "vacia", "lista", "none"])
def test_codificar_rechaza_lo_que_no_es_una_imagen_rgb(imagen):
    with pytest.raises(ValueError, match="uint8 RGB"):
        codificar_jpeg(imagen, 80)


@pytest.mark.parametrize("calidad", [0, -1, CALIDAD_JPEG_MAXIMA + 1, 80.0, True, "80", None])
def test_calidades_no_validas(calidad):
    with pytest.raises(ValueError, match="calidad JPEG"):
        comprobar_calidad(calidad)
    with pytest.raises(ValueError, match="calidad JPEG"):
        codificar_jpeg(_imagen(), calidad)


@pytest.mark.parametrize("calidad", [1, 50, CALIDAD_JPEG_MAXIMA, np.int64(70)])
def test_calidades_validas(calidad):
    assert comprobar_calidad(calidad) == int(calidad)


def test_la_calidad_del_dashboard_es_valida():
    datos = yaml.safe_load((RAIZ / "configs" / "dashboard.yaml").read_text(encoding="utf-8"))
    assert comprobar_calidad(datos["vista"]["calidad_jpeg"])


# --- CallbackVista ---------------------------------------------------------------------

class _Reloj:
    """Reloj falso que avanza lo que se le diga en cada lectura."""

    def __init__(self, paso: float) -> None:
        self.ahora = 0.0
        self.paso = paso

    def __call__(self) -> float:
        self.ahora += self.paso
        return self.ahora


@pytest.mark.parametrize("argumentos, mensaje", [
    ({"emitir": None}, "emitir"),
    ({"fps": 0}, "fps"), ({"fps": -5}, "fps"), ({"fps": float("nan")}, "fps"),
    ({"fps": True}, "fps"), ({"fps": "50"}, "fps"),
    ({"calidad": 0}, "calidad JPEG"),
])
def test_callback_rechaza_argumentos_no_validos(argumentos, mensaje):
    base = {"emitir": lambda datos: None, "fps": 50, "calidad": 80}
    with pytest.raises(ValueError, match=mensaje):
        CallbackVista(**{**base, **argumentos})


def _iteraciones(callback: BaseCallback, veces: int = 1, dibujable: bool = True) -> None:
    """Da iteraciones de TRPO en CartPole (2 entornos, 32 pasos de VecEnv por iteración)."""
    cfg = _cfg()
    fijar_seed(0)
    venv = crear_entorno_vectorizado(cfg, 2, 3, dibujable=dibujable)
    try:
        agente = crear_trpo(cfg, venv)
        for _ in range(veces):
            iteracion_trpo(agente, callback)
    finally:
        venv.close()


def test_apagada_no_emite():
    emitidas = []
    _iteraciones(CallbackVista(emitidas.append, 50, 80, reloj=_Reloj(1.0)))
    assert emitidas == []


def test_encendida_emite_jpeg_como_mucho_a_los_fps():
    # 32 llamadas. El reloj avanza 0,01 s por lectura y fps 25 piden 0,04 s entre
    # imágenes: sale una de cada cuatro.
    emitidas = []
    _iteraciones(CallbackVista(emitidas.append, 25, 80, activa=True, reloj=_Reloj(0.01)))
    assert len(emitidas) == 8
    assert all(datos[:2] == b"\xff\xd8" for datos in emitidas)
    with Image.open(io.BytesIO(emitidas[0])) as imagen:
        assert imagen.size == (600, 400)  # el render de CartPole


def test_con_un_reloj_lento_emite_en_cada_paso():
    emitidas = []
    _iteraciones(CallbackVista(emitidas.append, 25, 80, activa=True, reloj=_Reloj(1.0)))
    assert len(emitidas) == 32


def test_al_encenderla_la_primera_imagen_sale_sin_esperar():
    emitidas = []
    # El reloj no avanza: solo sale una imagen tras cada encendido.
    vista = CallbackVista(emitidas.append, 25, 80, activa=True, reloj=_Reloj(0.0))

    class Alternar(BaseCallback):
        """Apaga y vuelve a encender la vista cada 8 pasos, antes de que ella actúe."""

        def _init_callback(self) -> None:
            vista.init_callback(self.model)

        def _on_step(self) -> bool:
            if self.n_calls % 8 == 0:
                vista.activa = False
                vista.activa = True
            return vista.on_step()

    _iteraciones(Alternar())
    # La primera al empezar y una tras cada reencendido: pasos 8, 16, 24 y 32.
    assert len(emitidas) == 5


def test_encender_lo_que_ya_estaba_encendido_no_adelanta_la_siguiente():
    vista = CallbackVista(lambda datos: None, 25, 80, activa=True)
    vista._ultima = 5.0
    vista.activa = True
    assert vista._ultima == 5.0
    vista.activa = False
    assert vista._ultima == 5.0 and not vista.activa


def test_sin_entorno_dibujable_avisa_al_empezar():
    with pytest.raises(ValueError, match="dibujable=True"):
        _iteraciones(CallbackVista(lambda datos: None, 25, 80), dibujable=False)


def test_un_render_sin_imagen_no_emite(monkeypatch):
    emitidas = []
    cfg = _cfg()
    fijar_seed(0)
    venv = crear_entorno_vectorizado(cfg, 2, 3, dibujable=True)
    try:
        agente = crear_trpo(cfg, venv)
        monkeypatch.setattr(venv, "env_method", lambda *args, **kwargs: [None])
        iteracion_trpo(agente, CallbackVista(emitidas.append, 25, 80, activa=True,
                                             reloj=_Reloj(1.0)))
    finally:
        venv.close()
    assert emitidas == []


# --- la vista no cambia el entrenamiento -----------------------------------------------

class _Encender(BaseCallback):
    """Enciende y apaga la vista según el paso, como haría el usuario en marcha."""

    def __init__(self, vista: CallbackVista, cada: int) -> None:
        super().__init__()
        self.vista = vista
        self.cada = cada

    def _init_callback(self) -> None:
        self.vista.init_callback(self.model)

    def _on_step(self) -> bool:
        if self.n_calls % self.cada == 0:
            self.vista.activa = not self.vista.activa
        return self.vista.on_step()


def _pesos(modelo) -> dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in modelo.policy.state_dict().items()}


@pytest.mark.parametrize("caso", ["apagada", "encendida", "alternando"])
def test_el_experto_es_el_mismo_con_y_sin_vista(caso):
    cfg = _cfg()
    sin_modelo, sin_resumen = entrenar_experto(cfg)
    emitidas = []
    vista = CallbackVista(emitidas.append, 50, 80, activa=caso == "encendida", reloj=_Reloj(1.0))
    callback = _Encender(vista, 20) if caso == "alternando" else vista
    con_modelo, con_resumen = entrenar_experto(cfg, callback=callback, dibujable=True)
    pesos_sin, pesos_con = _pesos(sin_modelo), _pesos(con_modelo)
    assert all(torch.equal(pesos_sin[k], pesos_con[k]) for k in pesos_sin)
    assert sin_resumen["evaluaciones"] == con_resumen["evaluaciones"]
    # 256 pasos en 2 entornos: 128 llamadas.
    assert len(emitidas) == {"apagada": 0, "encendida": 128, "alternando": 60}[caso]
