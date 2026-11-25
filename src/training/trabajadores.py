"""Trabajadores del dashboard: las tareas que el gestor de trabajos ejecuta en un proceso.

El gestor (src/app/trabajos.py) llama a cada tarea con ``(argumentos, canal)`` en un
proceso propio. La tarea entrena con las mismas funciones que los scripts de la E1
(entrenar_experto…), manda el progreso y las imágenes por el canal, obedece las
órdenes del navegador y, al terminar, guarda el modelo en la biblioteca y devuelve el
resultado (ResultadoEntrenamiento en src/app/contrato.py).

Argumentos de un entrenamiento (los prepara la ruta POST /api/entrenamientos):

- ``ruta_config``: YAML base (configs/default.yaml).
- ``biblioteca``: carpeta de la biblioteca (aplicativo.biblioteca).
- ``calidad_jpeg``: calidad de la vista (vista.calidad_jpeg en configs/dashboard.yaml).
- ``entorno``: clave del entorno (cartpole…).
- ``nombre``: nombre del modelo en la biblioteca.
- ``ajustes``: lo que cambia el usuario, anidado como en configs/.
- ``semilla``: semilla efectiva (la del esquema si el usuario no puso otra).
- ``vista``: si la vista empieza encendida.
- ``wandb``: ``{"activo": bool, "grupo": str | None}``.

Mensajes de progreso: uno por iteración, ``{"paso", "iteracion", ...}`` con ``paso`` en
la unidad de la duración (pasos del entorno en el experto) y las mismas medidas que van
a W&B, con su sección (``trpo/…``); si en esa iteración toca evaluar, también las de la
evaluación (``evaluacion/…``) en el mismo mensaje. Solo números: las listas (como los
retornos de cada partida evaluada) van solo en el resumen.

Órdenes: ``{"tipo": "vista", "activa": bool}`` y ``{"tipo": "finalizar"}``, que termina
tras la iteración en curso y guarda lo que daría el bucle al acabar (el experto, el
mejor evaluado), marcado como incompleto.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback

from src.data.biblioteca import agregar_nuevo
from src.data.entornos import crear_entorno
from src.data.splits import huella_archivo
from src.training.trpo import entrenar_experto, guardar_experto
from src.training.vista import CallbackVista
from src.utils.configuracion import resolver_configuracion
from src.utils.procedencia import a_json
from src.utils.tracking import (abrir_run, nombre_artefacto, registrar, registrar_resumen,
                                subir_artefacto, terminar_run)
from src.utils.validacion import es_semilla

# Claves que tiene que traer argumentos en un entrenamiento.
ARGUMENTOS_ENTRENAMIENTO = frozenset({"ruta_config", "biblioteca", "calidad_jpeg", "entorno",
                                      "nombre", "ajustes", "semilla", "vista", "wandb"})
# Secciones de las medidas que se mandan juntas con las de la iteración en la que ocurren.
SECCIONES_EVALUACION = ("evaluacion/", "validacion/")


def comprobar_argumentos_entrenamiento(argumentos: object) -> dict[str, Any]:
    """Comprueba los argumentos de un entrenamiento antes de crear nada.

    Args:
        argumentos: Lo que recibe la tarea.

    Returns:
        Una copia de los argumentos.

    Raises:
        ValueError: Si falta o sobra alguna clave o algún valor no tiene su tipo.
    """
    if not isinstance(argumentos, dict) or set(argumentos) != ARGUMENTOS_ENTRENAMIENTO:
        claves = sorted(argumentos) if isinstance(argumentos, dict) else argumentos
        raise ValueError("Los argumentos deben tener exactamente "
                         f"{sorted(ARGUMENTOS_ENTRENAMIENTO)}, no {claves}")
    for clave in ("ruta_config", "biblioteca", "entorno", "nombre"):
        if not isinstance(argumentos[clave], str) or not argumentos[clave].strip():
            raise ValueError(f"{clave} debe ser un texto no vacío")
    if not isinstance(argumentos["ajustes"], dict):
        raise ValueError("ajustes debe ser un diccionario anidado como en configs/")
    if not es_semilla(argumentos["semilla"]):
        raise ValueError(f"semilla no válida: {argumentos['semilla']!r}")
    if not isinstance(argumentos["vista"], bool):
        raise ValueError("vista debe ser true o false")
    wandb = argumentos["wandb"]
    if (not isinstance(wandb, dict) or set(wandb) != {"activo", "grupo"}
            or not isinstance(wandb["activo"], bool)
            or (wandb["grupo"] is not None and not isinstance(wandb["grupo"], str))
            or (wandb["activo"] and not wandb["grupo"])):
        raise ValueError("wandb debe ser {\"activo\": bool, \"grupo\": texto o None}, con "
                         "grupo si está activo")
    return dict(argumentos)


class ProgresoTrabajo:
    """Hace de run para los bucles de entrenamiento: reenvía a W&B y manda el progreso.

    Los bucles llaman a ``log(medidas, step)`` una vez por iteración y, si toca evaluar,
    otra con las de la evaluación en el mismo paso. Aquí se juntan en un solo mensaje:
    el de la iteración espera a ver si llega su evaluación y sale con ella, al empezar
    la iteración siguiente (``vaciar``) o al terminar.
    """

    def __init__(self, canal: Any, run: Any | None = None, unidad: str = "pasos") -> None:
        """Prepara el progreso.

        Args:
            canal: CanalTrabajo de la tarea.
            run: Run de W&B o None.
            unidad: Qué es ``paso`` en los mensajes: "pasos" (el step de W&B, pasos del
                entorno) o "iteraciones" (el número de iteración).

        Raises:
            ValueError: Si la unidad no es una de las dos.
        """
        if unidad not in ("pasos", "iteraciones"):
            raise ValueError(f"unidad debe ser pasos o iteraciones, no {unidad!r}")
        self.canal, self.run, self.unidad = canal, run, unidad
        self.iteracion = 0
        self._pendiente: dict[str, Any] | None = None

    def log(self, medidas: dict[str, Any], step: int | None = None) -> None:
        """Recibe las medidas de los bucles, como un run de W&B."""
        registrar(self.run, medidas, step)
        numeros = {clave: float(valor) for clave, valor in medidas.items()
                   if _es_numero(valor)}
        if any(clave.startswith(SECCIONES_EVALUACION) for clave in medidas):
            if self._pendiente is None:  # una evaluación sin iteración antes: va sola
                self._pendiente = self._cabecera(step)
            self._pendiente.update(numeros)
            self.vaciar()
            return
        self.vaciar()
        self.iteracion += 1
        self._pendiente = {**self._cabecera(step), **numeros}

    def _cabecera(self, step: int | None) -> dict[str, Any]:
        """paso e iteración del mensaje."""
        paso = self.iteracion if self.unidad == "iteraciones" else step
        return {"paso": int(paso) if paso is not None else self.iteracion,
                "iteracion": self.iteracion}

    def vaciar(self) -> None:
        """Manda el mensaje pendiente, si hay uno."""
        if self._pendiente is not None:
            pendiente, self._pendiente = self._pendiente, None
            self.canal.progreso(pendiente)


def _es_numero(valor: object) -> bool:
    """Si es un número finito que se puede mandar como medida (no booleanos ni listas)."""
    try:
        return not isinstance(valor, bool) and math.isfinite(float(valor))
    except (TypeError, ValueError):
        return False


class ControlEntrenamiento(BaseCallback):
    """Órdenes del navegador y vista durante un entrenamiento de sb3.

    Se pasa como ``callback`` al bucle y su método ``parar`` como ``parar``. Lee las
    órdenes en cada paso: ``vista`` enciende o apaga la vista al momento y
    ``finalizar`` hace que ``parar`` devuelva True al acabar la iteración en curso.
    """

    def __init__(self, canal: Any, vista: CallbackVista, progreso: ProgresoTrabajo) -> None:
        """Prepara el control.

        Args:
            canal: CanalTrabajo de la tarea.
            vista: Vista del entrenamiento (empieza encendida o apagada).
            progreso: Progreso de la tarea; su mensaje pendiente sale al empezar cada
                iteración.
        """
        super().__init__()
        self.canal, self.vista, self.progreso = canal, vista, progreso
        self.finalizar_pedido = False

    def atender_ordenes(self) -> None:
        """Aplica las órdenes recibidas desde la última vez."""
        for orden in self.canal.ordenes():
            if orden.get("tipo") == "vista":
                self.vista.activa = bool(orden.get("activa"))
            elif orden.get("tipo") == "finalizar":
                self.finalizar_pedido = True

    def parar(self) -> bool:
        """Para ``parar`` del bucle: True si se pidió finalizar."""
        self.atender_ordenes()
        return self.finalizar_pedido

    def _init_callback(self) -> None:
        self.vista.init_callback(self.model)

    def _on_rollout_start(self) -> None:
        # Empieza otra iteración: la anterior ya no tendrá evaluación.
        self.progreso.vaciar()

    def _on_step(self) -> bool:
        self.atender_ordenes()
        return self.vista.on_step()


def fps_entorno(cfg: dict[str, Any]) -> int:
    """Imágenes por segundo del entorno (render_fps), las que usa la vista como máximo.

    Raises:
        ValueError: Si el entorno no declara render_fps.
    """
    entorno = crear_entorno(cfg, None)
    try:
        fps = entorno.metadata.get("render_fps")
    finally:
        entorno.close()
    if isinstance(fps, bool) or not isinstance(fps, int) or fps < 1:
        raise ValueError(f"{cfg['entorno']['id']} no declara render_fps")
    return fps


def tarea_entrenar_experto(argumentos: dict[str, Any], canal: Any) -> dict[str, Any]:
    """Entrena un experto desde el dashboard y lo guarda en la biblioteca.

    Usa entrenar_experto con la configuración del perfil y los ajustes del usuario y la
    semilla pedida en experto.semilla. El modelo se guarda como en la E1 (modelo.zip y
    experto.json) con origen «dashboard». Con W&B, registra las medidas, el resumen y el
    modelo como artefacto en el grupo pedido.

    Args:
        argumentos: Ver el docstring del módulo.
        canal: CanalTrabajo.

    Returns:
        ``{"modelo": {"id", "nombre", "incompleto"}, "semilla", "resumen"}``; incompleto
        si se finalizó antes de llegar al presupuesto o al objetivo.

    Raises:
        ValueError: Si los argumentos o la configuración no son válidos.
    """
    argumentos = comprobar_argumentos_entrenamiento(argumentos)
    cfg = resolver_configuracion(Path(argumentos["ruta_config"]), argumentos["entorno"],
                                 argumentos["ajustes"])
    cfg["experto"]["semilla"] = int(argumentos["semilla"])
    wandb = argumentos["wandb"]
    if not wandb["activo"]:
        cfg["wandb"]["modo"] = "disabled"
    vista = CallbackVista(canal.imagen, fps_entorno(cfg), argumentos["calidad_jpeg"],
                          activa=argumentos["vista"])
    gym = cfg["entorno"]["id"]
    run = abrir_run(cfg, grupo=wandb["grupo"], job_type="experto", nombre=argumentos["nombre"],
                    etiquetas=[f"entorno:{gym}", "metodo:experto", "origen:dashboard"])
    try:
        progreso = ProgresoTrabajo(canal, run, "pasos")
        control = ControlEntrenamiento(canal, vista, progreso)
        modelo, resumen = entrenar_experto(cfg, progreso, parar=control.parar,
                                           callback=control, dibujable=True)
        # La última iteración siempre se evalúa: su mensaje ya salió con la evaluación.
        incompleto = (control.finalizar_pedido and not resumen["objetivo_alcanzado"]
                      and resumen["pasos"] < cfg["experto"]["pasos_totales"])
        resumen = {**resumen, "incompleto": incompleto}
        ficha = agregar_nuevo(Path(argumentos["biblioteca"]), argumentos["entorno"], "experto",
                              argumentos["nombre"].strip(), "dashboard",
                              lambda carpeta: guardar_experto(carpeta, modelo, cfg, resumen))
        if run is not None:
            carpeta = (Path(argumentos["biblioteca"]) / argumentos["entorno"] / "experto"
                       / ficha["id"])
            registrar_resumen(run, resumen)
            huella = huella_archivo(carpeta / "modelo.zip")
            subir_artefacto(run, nombre_artefacto("experto", huella, gym), "experto",
                            [carpeta / "modelo.zip", carpeta / "experto.json"],
                            {"entorno": gym, "huella_modelo": huella, "origen": "dashboard"})
    except BaseException as error:
        terminar_run(run, error)
        raise
    terminar_run(run)
    return {"modelo": {"id": ficha["id"], "nombre": ficha["nombre"], "incompleto": incompleto},
            "semilla": int(argumentos["semilla"]),
            "resumen": a_json_profundo(resumen)}


def a_json_profundo(valor: Any) -> Any:
    """Copia un valor con escalares de NumPy convertidos a tipos de Python (a_json)."""
    if isinstance(valor, dict):
        return {str(clave): a_json_profundo(v) for clave, v in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [a_json_profundo(v) for v in valor]
    if valor is None or isinstance(valor, (bool, int, float, str)):
        return valor
    return a_json(valor)
