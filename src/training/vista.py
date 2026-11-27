"""Vista en directo del dashboard: dibujar una partida mientras se entrena.

Mientras la vista está encendida, CallbackVista dibuja la primera instancia del VecEnv
de entrenamiento (creado con ``crear_entorno_vectorizado(..., dibujable=True)``) y
manda cada imagen en JPEG a quien la pida, como mucho a los fotogramas por segundo del
entorno (``render_fps``): si el entrenamiento va más rápido, se salta imágenes. Dibujar
no usa ningún generador aleatorio ni cambia el estado del entorno, así que el
entrenamiento es el mismo con la vista encendida, apagada o cambiando en marcha.

EntornoVista hace lo mismo con las partidas de una comparación, pero a velocidad real:
mientras está encendida, cada paso espera lo que falte para ir a los fps del entorno, así
que las partidas se ven como se jugarían; apagada, van a toda velocidad.
"""

from __future__ import annotations

import io
import time
from collections.abc import Callable
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
from PIL import Image
from stable_baselines3.common.callbacks import BaseCallback

from src.utils.reglas_config import comprobar_calidad
from src.utils.validacion import es_real_finito

# Instancia del VecEnv que se dibuja (todas tienen render_mode="rgb_array").
INDICE_DIBUJABLE = 0


def codificar_jpeg(imagen: np.ndarray, calidad: int) -> bytes:
    """Codifica en JPEG una imagen RGB de Gymnasium (``render_mode="rgb_array"``).

    Args:
        imagen: Matriz uint8 de forma (alto, ancho, 3).
        calidad: Calidad JPEG, de 1 a CALIDAD_JPEG_MAXIMA.

    Returns:
        Los bytes del JPEG.

    Raises:
        ValueError: Si la imagen no es RGB uint8 o la calidad no es válida.
    """
    calidad = comprobar_calidad(calidad)
    if (not isinstance(imagen, np.ndarray) or imagen.dtype != np.uint8 or imagen.ndim != 3
            or imagen.shape[2] != 3 or 0 in imagen.shape):
        raise ValueError("La imagen debe ser un array uint8 RGB de forma (alto, ancho, 3)")
    salida = io.BytesIO()
    Image.fromarray(imagen, "RGB").save(salida, "JPEG", quality=calidad)
    return salida.getvalue()


class CallbackVista(BaseCallback):
    """Callback de sb3 que dibuja la partida del entorno 0 mientras se entrena.

    Se pasa como ``callback`` a entrenar_experto o entrenar_gail junto con
    ``dibujable=True``. Con la vista apagada no hace nada; se enciende y se apaga en
    marcha con ``activa``. Cada imagen llega a ``emitir`` en JPEG.
    """

    def __init__(self, emitir: Callable[[bytes], None], fps: float, calidad: int,
                 activa: bool = False, reloj: Callable[[], float] = time.monotonic) -> None:
        """Prepara la vista.

        Args:
            emitir: Recibe cada imagen en JPEG.
            fps: Imágenes por segundo como mucho (render_fps del entorno).
            calidad: Calidad JPEG (vista.calidad_jpeg).
            activa: Si empieza encendida.
            reloj: Segundos de un reloj monótono (los tests pasan uno falso).

        Raises:
            ValueError: Si fps o calidad no son válidos o emitir no es invocable.
        """
        super().__init__()
        if not callable(emitir):
            raise ValueError("emitir debe ser invocable")
        if not es_real_finito(fps) or fps <= 0:
            raise ValueError("fps debe ser un número positivo")
        self.emitir = emitir
        self.intervalo = 1.0 / float(fps)
        self.calidad = comprobar_calidad(calidad)
        self.reloj = reloj
        self._activa = bool(activa)
        self._ultima: float | None = None

    @property
    def activa(self) -> bool:
        """Si la vista está encendida."""
        return self._activa

    @activa.setter
    def activa(self, valor: bool) -> None:
        # Al encenderla, la primera imagen sale en el siguiente paso, sin esperar.
        if valor and not self._activa:
            self._ultima = None
        self._activa = bool(valor)

    def _init_callback(self) -> None:
        """Comprueba que el entorno 0 se puede dibujar antes de empezar."""
        modo = self.training_env.get_attr("render_mode", indices=INDICE_DIBUJABLE)[0]
        if modo != "rgb_array":
            raise ValueError("La vista necesita un VecEnv creado con dibujable=True "
                             "(render_mode='rgb_array')")

    def _on_step(self) -> bool:
        """Tras cada paso, dibuja y emite una imagen si toca."""
        if not self._activa:
            return True
        ahora = self.reloj()
        if self._ultima is not None and ahora - self._ultima < self.intervalo:
            return True
        self._ultima = ahora
        imagen = self.training_env.env_method("render", indices=INDICE_DIBUJABLE)[0]
        if imagen is not None:
            self.emitir(codificar_jpeg(imagen, self.calidad))
        return True


class EntornoVista(gym.Wrapper):
    """Envoltorio que dibuja las partidas a velocidad real mientras está encendido.

    Tras cada ``reset`` y cada ``step``, con la vista encendida, espera hasta que toque la
    siguiente imagen (una cada 1/fps segundos), dibuja el estado y lo emite en JPEG.
    Apagada no espera ni dibuja. Dibujar no cambia el entorno, así que las partidas son
    las mismas con la vista encendida, apagada o cambiando en marcha. Se enciende y se
    apaga con ``activa``; al encenderla, la primera imagen sale sin esperar.
    """

    def __init__(self, entorno: gym.Env, emitir: Callable[[bytes], None], fps: float,
                 calidad: int, activa: bool = False,
                 reloj: Callable[[], float] = time.monotonic,
                 dormir: Callable[[float], None] = time.sleep) -> None:
        """Prepara la vista.

        Args:
            entorno: Entorno creado con ``render_mode="rgb_array"``.
            emitir: Recibe cada imagen en JPEG.
            fps: Imágenes por segundo (render_fps del entorno).
            calidad: Calidad JPEG (vista.calidad_jpeg).
            activa: Si empieza encendida.
            reloj: Segundos de un reloj monótono (los tests pasan uno falso).
            dormir: Espera los segundos que se le digan (los tests pasan una falsa).

        Raises:
            ValueError: Si el entorno no se puede dibujar, fps o calidad no son válidos
                o emitir no es invocable.
        """
        super().__init__(entorno)
        if entorno.render_mode != "rgb_array":
            raise ValueError("La vista necesita un entorno con render_mode='rgb_array'")
        if not callable(emitir):
            raise ValueError("emitir debe ser invocable")
        if not es_real_finito(fps) or fps <= 0:
            raise ValueError("fps debe ser un número positivo")
        self.emitir = emitir
        self.intervalo = 1.0 / float(fps)
        self.calidad = comprobar_calidad(calidad)
        self.reloj, self.dormir = reloj, dormir
        self._activa = bool(activa)
        self._siguiente: float | None = None

    @property
    def activa(self) -> bool:
        """Si la vista está encendida."""
        return self._activa

    @activa.setter
    def activa(self, valor: bool) -> None:
        # Al encenderla, la primera imagen sale sin esperar.
        if valor and not self._activa:
            self._siguiente = None
        self._activa = bool(valor)

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        """Reinicia el entorno y, con la vista encendida, dibuja el estado inicial."""
        resultado = self.env.reset(**kwargs)
        self._mostrar()
        return resultado

    def step(self, accion: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        """Da un paso y, con la vista encendida, lo dibuja a su ritmo."""
        resultado = self.env.step(accion)
        self._mostrar()
        return resultado

    def _mostrar(self) -> None:
        """Espera lo que falte para la siguiente imagen, dibuja y emite."""
        if not self._activa:
            return
        ahora = self.reloj()
        if self._siguiente is not None and ahora < self._siguiente:
            self.dormir(self._siguiente - ahora)
            ahora = self._siguiente
        # Si se va con retraso (o tras una pausa), no se acumula: sigue desde ahora.
        self._siguiente = ahora + self.intervalo
        imagen = self.env.render()
        if imagen is not None:
            self.emitir(codificar_jpeg(imagen, self.calidad))
