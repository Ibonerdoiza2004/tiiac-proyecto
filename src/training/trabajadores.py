"""Trabajadores del dashboard: las tareas que el gestor de trabajos ejecuta en un proceso.

El gestor (src/app/trabajos.py) llama a cada tarea con ``(argumentos, canal)`` en un
proceso propio. La tarea entrena con las mismas funciones que los scripts de la E1
(entrenar_experto…), manda el progreso y las imágenes por el canal, obedece las
órdenes del navegador y, al terminar, guarda el modelo en la biblioteca y devuelve el
resultado (ResultadoEntrenamiento en src/app/contrato.py).

Hay una tarea por método, tarea_entrenar_experto, tarea_entrenar_gail y
tarea_entrenar_bc, y otra para comparar modelos, tarea_comparar (abajo). Argumentos de
un entrenamiento (los prepara la ruta POST /api/entrenamientos):

- ``ruta_config``: YAML base (configs/default.yaml).
- ``biblioteca``: carpeta de la biblioteca (aplicativo.biblioteca).
- ``calidad_jpeg``: calidad de la vista (vista.calidad_jpeg en configs/dashboard.yaml).
- ``entorno``: clave del entorno (cartpole…).
- ``nombre``: nombre del modelo en la biblioteca.
- ``ajustes``: lo que cambia el usuario, anidado como en configs/.
- ``semilla``: semilla efectiva (la del esquema si el usuario no puso otra).
- ``vista``: si la vista empieza encendida.
- ``wandb``: ``{"activo": bool, "grupo": str | None}``.
- Solo BC y GAIL: ``datos`` (ids de conjuntos y subconjuntos de la biblioteca, que se
  preparan con preparar_datos: un conjunto con splits tal cual, si no, un combinado) y
  ``cantidad`` (partidas de train que se usan, las N primeras; None, todas). La semilla
  es la base de las repeticiones (experimento.semilla_repeticiones) y se entrena la
  repetición 0, como en la E1. BC no tiene vista.
- ``checkpoint``: carpeta de sus checkpoints (src/training/checkpoints.py; None, sin
  checkpoints). BC guarda uno al acabar cada época, el experto tras cada evaluación
  (``experto.evaluar_cada``) y GAIL tras cada validación (``gail.evaluar_cada``). Si la
  carpeta ya tiene uno (de una ejecución cortada con los mismos argumentos), se sigue
  desde él (EnCurso): primero se vuelve a mandar el progreso guardado y después se
  entrena el resto. BC sigue exactamente igual que sin corte; el experto y GAIL, con
  entornos nuevos (semilla_reanudacion), así que no salen idénticos. El resumen dice desde qué
  épocas o iteraciones se reanudó (``reanudado_desde``) y, con W&B, se abre un run
  nuevo con la etiqueta ``reanudado`` y el id del anterior en su config. Al terminar
  bien, la carpeta se borra.

Mensajes de progreso: uno por iteración, ``{"paso", "iteracion", ...}`` (en BC, uno por
época, ``{"paso", "epoca", ...}``) con ``paso`` en la unidad de la duración (pasos del
entorno en el experto, iteraciones en GAIL, épocas en BC) y las mismas medidas que van a
W&B, con su sección (``trpo/…``, ``bc/val/…``); si en esa iteración toca evaluar,
también las de la evaluación (``evaluacion/…``, ``validacion/…``) en el mismo mensaje.
En BC, ``bc/train/perdida`` es la media de los batches de la época (a W&B llega la de
cada batch). Solo números: las listas (como los retornos de cada partida evaluada) van
solo en el resumen.

Órdenes: ``{"tipo": "vista", "activa": bool}`` y ``{"tipo": "finalizar"}``, que termina
tras la iteración o época en curso y guarda lo que daría el bucle al acabar (el
experto, el mejor evaluado; GAIL, la última iteración; BC, la mejor época), marcado como
incompleto.

Comparación (tarea_comparar; la prepara la ruta de crear comparaciones): argumentos
``ruta_config``, ``biblioteca``, ``calidad_jpeg``, ``entorno``, ``modelos`` (ids de
expertos, BC y GAIL válidos de la biblioteca, sin repetir), ``aleatorio`` (si se añade el
agente aleatorio), ``partidas`` (por modelo; None, evaluacion.episodios) y ``vista``. Se
juega como la evaluación de la E1 (evaluar_politica): semillas de evaluacion.semilla,
una por partida, la misma para todos, y evaluacion.determinista. Todos los modelos juegan
la partida i antes de la i + 1. Un progreso por partida de cada modelo,
``{"modelo", "partida", "retorno", "pasos"}``, con ``modelo`` el id o ``aleatorio`` y
``partida`` desde 0. La vista dibuja las partidas a velocidad real (los fps del entorno)
mientras está encendida. ``finalizar`` corta la partida en curso y el resultado se queda
solo con las rondas que jugaron todos los modelos. No se guarda nada fuera del trabajo.
"""

from __future__ import annotations

import dataclasses
import math
import time
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

from src.data.biblioteca import Biblioteca, agregar_nuevo
from src.data.demostraciones import jugar_partida, retorno_partida
from src.data.entornos import crear_entorno
from src.data.seleccion import preparar_datos
from src.data.splits import SplitsDemostraciones, huella_archivo, validar_splits
from src.data.tipos import Trayectoria
from src.training.barrido import cargar_datos
from src.training.bc import cargar_politica_bc, entrenar_bc_repeticion, guardar_bc_repeticion
from src.training.checkpoints import (borrar_en_curso, comprobar_misma_config, en_uso,
                                      escribir_checkpoint, preparar_en_curso)
from src.training.evaluacion import config_evaluacion, predecir_accion
from src.training.gail import cargar_gail, entrenar_gail, guardar_gail
from src.training.trpo import cargar_experto, entrenar_experto, guardar_experto
from src.training.vista import CallbackVista, EntornoVista
from src.utils.configuracion import TipoEntorno, resolver_configuracion
from src.utils.procedencia import a_json
from src.utils.tracking import (abrir_run, nombre_artefacto, registrar, registrar_resumen,
                                subir_artefacto, terminar_run)
from src.utils.seed import semillas_independientes
from src.utils.validacion import es_entero_positivo, es_real_finito, es_semilla

# Claves que tiene que traer argumentos en un entrenamiento de cada método.
ARGUMENTOS_ENTRENAMIENTO = frozenset({"ruta_config", "biblioteca", "calidad_jpeg", "entorno",
                                      "nombre", "ajustes", "semilla", "vista", "wandb",
                                      "checkpoint"})
ARGUMENTOS_IMITACION = ARGUMENTOS_ENTRENAMIENTO | {"datos", "cantidad"}
ARGUMENTOS_POR_METODO = {"experto": ARGUMENTOS_ENTRENAMIENTO, "gail": ARGUMENTOS_IMITACION,
                         "bc": ARGUMENTOS_IMITACION}
# Repetición que entrena un BC o GAIL suelto: la 0 de la semilla pedida, como en la E1.
REPETICION_DASHBOARD = 0
# Secciones de las medidas que se mandan juntas con las de la iteración en la que ocurren.
SECCIONES_EVALUACION = ("evaluacion/", "validacion/")
# Claves de los argumentos de una comparación y nombre del agente aleatorio en ella.
ARGUMENTOS_COMPARACION = frozenset({"ruta_config", "biblioteca", "calidad_jpeg", "entorno",
                                    "modelos", "aleatorio", "partidas", "vista"})
ALEATORIO = "aleatorio"
# Tipos de la biblioteca que se pueden comparar.
TIPOS_COMPARABLES = ("experto", "bc", "gail")


def comprobar_argumentos_entrenamiento(argumentos: object,
                                       metodo: str = "experto") -> dict[str, Any]:
    """Comprueba los argumentos de un entrenamiento antes de crear nada.

    Args:
        argumentos: Lo que recibe la tarea.
        metodo: experto, gail o bc (BC y GAIL llevan además datos y cantidad).

    Returns:
        Una copia de los argumentos.

    Raises:
        ValueError: Si falta o sobra alguna clave o algún valor no tiene su tipo.
    """
    esperadas = ARGUMENTOS_POR_METODO[metodo]
    if not isinstance(argumentos, dict) or set(argumentos) != esperadas:
        claves = sorted(argumentos) if isinstance(argumentos, dict) else argumentos
        raise ValueError("Los argumentos deben tener exactamente "
                         f"{sorted(esperadas)}, no {claves}")
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
    if metodo != "experto":
        datos = argumentos["datos"]
        if (not isinstance(datos, list) or not datos
                or not all(isinstance(d, str) and d for d in datos)):
            raise ValueError("datos debe ser una lista no vacía de ids de la biblioteca")
        if argumentos["cantidad"] is not None and not es_entero_positivo(argumentos["cantidad"]):
            raise ValueError(f"cantidad debe ser un entero positivo o None, no "
                             f"{argumentos['cantidad']!r}")
    if metodo == "bc" and argumentos["vista"]:
        raise ValueError("BC no tiene vista: no juega partidas al entrenar")
    checkpoint = argumentos["checkpoint"]
    if checkpoint is not None and (not isinstance(checkpoint, str) or not checkpoint.strip()):
        raise ValueError("checkpoint debe ser la carpeta de los checkpoints (texto) o None")
    return dict(argumentos)


class ProgresoTrabajo:
    """Hace de run para los bucles de entrenamiento: reenvía a W&B y manda el progreso.

    Los bucles llaman a ``log(medidas, step)`` una vez por iteración y, si toca evaluar,
    otra con las de la evaluación en el mismo paso. Aquí se juntan en un solo mensaje:
    el de la iteración espera a ver si llega su evaluación y sale con ella, al empezar
    la iteración siguiente (``vaciar``) o al terminar.
    """

    def __init__(self, canal: Any, run: Any | None = None, unidad: str = "pasos",
                 historial: list[dict[str, Any]] | None = None) -> None:
        """Prepara el progreso.

        Args:
            canal: CanalTrabajo de la tarea.
            run: Run de W&B o None.
            unidad: Qué es ``paso`` en los mensajes: "pasos" (el step de W&B, pasos del
                entorno) o "iteraciones" (el número de iteración).
            historial: Mensajes ya mandados antes de un corte (del checkpoint): se
                vuelven a mandar y se sigue contando desde ellos.

        Raises:
            ValueError: Si la unidad no es una de las dos.
        """
        if unidad not in ("pasos", "iteraciones"):
            raise ValueError(f"unidad debe ser pasos o iteraciones, no {unidad!r}")
        self.canal, self.run, self.unidad = canal, run, unidad
        self.historial: list[dict[str, Any]] = []
        self.iteracion = 0
        self._pendiente: dict[str, Any] | None = None
        for mensaje in historial or []:
            self._mandar(dict(mensaje))

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
            self._mandar(pendiente)

    def _mandar(self, mensaje: dict[str, Any]) -> None:
        """Manda un mensaje y lo apunta en el historial."""
        self.iteracion = int(mensaje["iteracion"])
        self.historial.append(mensaje)
        self.canal.progreso(mensaje)


class ProgresoEpocas:
    """Hace de run para entrenar_bc: reenvía a W&B y manda un progreso por época.

    entrenar_bc registra la pérdida de cada batch (``bc/train/perdida``) y, al validar
    cada época, sus medidas (``bc/val/…``). A W&B llega todo, como en la E1; al canal,
    un mensaje por época con la media de las pérdidas de sus batches y la validación.
    """

    def __init__(self, canal: Any, run: Any | None = None,
                 historial: list[dict[str, Any]] | None = None) -> None:
        """Prepara el progreso.

        Args:
            canal: CanalTrabajo de la tarea.
            run: Run de W&B o None.
            historial: Mensajes ya mandados antes de un corte (del checkpoint): se
                vuelven a mandar y se sigue contando desde ellos.
        """
        self.canal, self.run = canal, run
        self.historial: list[dict[str, Any]] = []
        self.epoca = 0
        self._perdidas: list[float] = []
        for mensaje in historial or []:
            self._mandar(dict(mensaje))

    def log(self, medidas: dict[str, Any], step: int | None = None) -> None:
        """Recibe las medidas de entrenar_bc, como un run de W&B."""
        registrar(self.run, medidas, step)
        if "bc/train/perdida" in medidas:
            self._perdidas.append(float(medidas["bc/train/perdida"]))
            return
        self.epoca += 1
        mensaje: dict[str, Any] = {"paso": self.epoca, "epoca": self.epoca}
        if self._perdidas:
            mensaje["bc/train/perdida"] = math.fsum(self._perdidas) / len(self._perdidas)
        mensaje.update({clave: float(valor) for clave, valor in medidas.items()
                        if _es_numero(valor)})
        self._perdidas = []
        self._mandar(mensaje)

    def _mandar(self, mensaje: dict[str, Any]) -> None:
        """Manda un mensaje y lo apunta en el historial."""
        self.epoca = int(mensaje["epoca"])
        self.historial.append(mensaje)
        self.canal.progreso(mensaje)


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

    def __init__(self, canal: Any, vista: CallbackVista | None = None,
                 progreso: ProgresoTrabajo | None = None) -> None:
        """Prepara el control.

        Args:
            canal: CanalTrabajo de la tarea.
            vista: Vista del entrenamiento (empieza encendida o apagada); None en BC,
                que solo usa ``parar`` y no juega partidas (las órdenes de vista se
                ignoran).
            progreso: Progreso de la tarea; su mensaje pendiente sale al empezar cada
                iteración.
        """
        super().__init__()
        self.canal, self.vista, self.progreso = canal, vista, progreso
        self.finalizar_pedido = False

    def atender_ordenes(self) -> None:
        """Aplica las órdenes recibidas desde la última vez."""
        for orden in self.canal.ordenes():
            if orden.get("tipo") == "vista" and self.vista is not None:
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
        return _fps(entorno, cfg)
    finally:
        entorno.close()


def _fps(entorno: gym.Env, cfg: dict[str, Any]) -> int:
    """render_fps de un entorno ya creado; ValueError si no lo declara."""
    fps = entorno.metadata.get("render_fps")
    if isinstance(fps, bool) or not isinstance(fps, int) or fps < 1:
        raise ValueError(f"{cfg['entorno']['id']} no declara render_fps")
    return fps


def _config(argumentos: dict[str, Any]) -> dict[str, Any]:
    """Perfil del entorno con los ajustes del usuario; sin W&B si no se pidió."""
    cfg = resolver_configuracion(Path(argumentos["ruta_config"]), argumentos["entorno"],
                                 argumentos["ajustes"])
    if not argumentos["wandb"]["activo"]:
        cfg["wandb"]["modo"] = "disabled"
    return cfg


def _abrir_run(cfg: dict[str, Any], argumentos: dict[str, Any], metodo: str,
               reanudado: dict[str, Any] | None = None) -> Any | None:
    """Run de W&B del entrenamiento (None sin W&B), en el grupo pedido.

    Un entrenamiento reanudado abre un run nuevo con la etiqueta ``reanudado`` y
    ``reanudado`` en su config (el run anterior y desde dónde sigue).
    """
    etiquetas = [f"entorno:{cfg['entorno']['id']}", f"metodo:{metodo}", "origen:dashboard"]
    if reanudado is not None:
        cfg = {**cfg, "reanudado": reanudado}
        etiquetas.append("reanudado")
    return abrir_run(cfg, grupo=argumentos["wandb"]["grupo"], job_type=metodo,
                     nombre=argumentos["nombre"], etiquetas=etiquetas)


def _subir_modelo(run: Any | None, metodo: str, carpeta: Path, archivos: list[str],
                  cfg: dict[str, Any], resumen: dict[str, Any],
                  extra: dict[str, Any] | None = None) -> None:
    """Con W&B: el resumen y el modelo guardado en la biblioteca como artefacto."""
    if run is None:
        return
    gym = cfg["entorno"]["id"]
    registrar_resumen(run, resumen)
    huella = huella_archivo(carpeta / archivos[0])
    subir_artefacto(run, nombre_artefacto(metodo, huella, gym), metodo,
                    [carpeta / a for a in archivos],
                    {"entorno": gym, "huella_modelo": huella, "origen": "dashboard",
                     **(extra or {})})


def _resultado(ficha: dict[str, Any], incompleto: bool, semilla: int,
               resumen: dict[str, Any]) -> dict[str, Any]:
    """ResultadoEntrenamiento (src/app/contrato.py)."""
    return {"modelo": {"id": ficha["id"], "nombre": ficha["nombre"], "incompleto": incompleto},
            "semilla": int(semilla), "resumen": a_json_profundo(resumen)}


def entornos_gym(ruta_config: Path) -> dict[str, str]:
    """Id de Gymnasium de cada entorno de TipoEntorno (lo que pide Biblioteca)."""
    return {t.value: resolver_configuracion(Path(ruta_config), t.value)["entorno"]["id"]
            for t in TipoEntorno}


def datos_de_entrenamiento(argumentos: dict[str, Any], cfg: dict[str, Any],
                           ) -> tuple[list[Trayectoria], SplitsDemostraciones, dict[str, str],
                                      int, str]:
    """Partidas, splits y procedencia con los que entrena un BC o GAIL suelto.

    Prepara los datos elegidos con preparar_datos (un conjunto con splits tal cual; si
    no, un combinado de la biblioteca) y los carga con cargar_datos, como la E1. Con
    ``cantidad`` se usan las N primeras partidas de train (por defecto, todas): si los
    splits no tienen ese subconjunto (un congelado solo trae los de la config), se
    añade, porque los subconjuntos son siempre las N primeras de train.

    Returns:
        Partidas, splits (con el subconjunto de la cantidad), procedencia de los datos,
        la cantidad y el id del conjunto usado.

    Raises:
        ValueError: Si los datos no valen o se piden más partidas de las que hay en train.
    """
    biblioteca = Biblioteca(Path(argumentos["biblioteca"]),
                            entornos_gym(argumentos["ruta_config"]))
    id_conjunto = preparar_datos(biblioteca, argumentos["datos"], cfg)
    trayectorias, splits, procedencia = cargar_datos(cfg, biblioteca.ruta(id_conjunto))
    cantidad = argumentos["cantidad"] or len(splits.train)
    if cantidad > len(splits.train):
        raise ValueError(f"Se pidieron {cantidad} partidas de train y los datos tienen "
                         f"{len(splits.train)}")
    if cantidad not in splits.subconjuntos:
        splits = dataclasses.replace(
            splits, subconjuntos={**splits.subconjuntos, cantidad: splits.train[:cantidad]})
        validar_splits(splits, [t.id_episodio for t in trayectorias])
    return trayectorias, splits, procedencia, int(cantidad), id_conjunto


# Lo que guarda EnCurso en cada checkpoint, además del formato, la versión y el método.
CLAVES_CHECKPOINT = frozenset({"estado", "config", "progreso", "reanudado_desde", "run",
                               "segundos"})


class EnCurso:
    """Checkpoints de un entrenamiento del dashboard (src/training/checkpoints.py).

    Se usa como ``with EnCurso(...) as en_curso:`` alrededor de todo el entrenamiento.
    Con ``checkpoint`` en los argumentos, reserva su carpeta mientras dura el bloque
    (en_uso: otro entrenamiento con la misma carpeta se rechaza), la prepara y, si ya
    había un checkpoint de una ejecución cortada, lo lee (con sus claves y la
    configuración comprobadas). Da lo que necesita la tarea para seguir desde él
    (``estado_inicial``, ``historial``, ``reanudado`` para W&B) y la función
    ``guardar_estado`` del bucle, que escribe el estado con el progreso mandado, las
    reanudaciones, el run y los segundos. Sin ``checkpoint``, todo es None y no se
    guarda nada.
    """

    def __init__(self, argumentos: dict[str, Any], metodo: str, cfg: dict[str, Any],
                 unidad: str) -> None:
        """Reserva y prepara la carpeta y lee el checkpoint, si hay.

        Args:
            argumentos: Los de la tarea, ya comprobados.
            metodo: experto, gail o bc.
            cfg: Configuración resuelta con la que se entrena.
            unidad: Clave del estado con la iteración o época del checkpoint
                (``iteracion`` o ``epoca``).

        Raises:
            CarpetaOcupada: Si otro entrenamiento está usando la carpeta.
            ValueError: Si la carpeta es de otro entrenamiento, el checkpoint no está
                completo o la configuración cambió.
        """
        self.metodo, self.cfg = metodo, cfg
        self.carpeta = Path(argumentos["checkpoint"]) if argumentos["checkpoint"] else None
        self._reserva = ExitStack()
        self.previo = None
        if self.carpeta is not None:
            self._reserva.enter_context(en_uso(self.carpeta))
            try:
                self.previo = preparar_en_curso(self.carpeta, metodo, argumentos)
                if self.previo is not None:
                    _comprobar_checkpoint(self.previo, unidad, self.carpeta)
                    comprobar_misma_config(self.previo["config"], cfg)
            except BaseException:
                self._reserva.close()
                raise
        self.reanudado_desde: list[int] = (
            [*self.previo["reanudado_desde"], int(self.previo["estado"][unidad])]
            if self.previo else [])
        self.reanudado = (None if self.previo is None else
                          {"run_anterior": self.previo["run"],
                           f"desde_{unidad}": self.reanudado_desde[-1]})
        self.segundos_previos = float(self.previo["segundos"]) if self.previo else 0.0
        self._inicio = time.perf_counter()

    @property
    def estado_inicial(self) -> dict[str, Any] | None:
        """El estado del bucle desde el que se sigue, o None."""
        return self.previo and self.previo["estado"]

    @property
    def historial(self) -> list[dict[str, Any]] | None:
        """El progreso mandado antes del corte, o None."""
        return self.previo and self.previo["progreso"]

    def guardador(self, progreso: Any, run: Any | None,
                  ) -> Callable[[dict[str, Any]], None] | None:
        """La función guardar_estado del bucle (None sin checkpoint).

        Args:
            progreso: ProgresoTrabajo o ProgresoEpocas, con su historial.
            run: Run de W&B o None.
        """
        if self.carpeta is None:
            return None

        def guardar_estado(estado: dict[str, Any]) -> None:
            escribir_checkpoint(self.carpeta, self.metodo, {
                "estado": estado, "config": self.cfg, "progreso": progreso.historial,
                "reanudado_desde": self.reanudado_desde,
                "run": run.id if run is not None else None,
                "segundos": self.segundos_previos + time.perf_counter() - self._inicio})

        return guardar_estado

    def completar(self, resumen: dict[str, Any]) -> dict[str, Any]:
        """El resumen con los segundos de antes del corte y ``reanudado_desde``."""
        return {**resumen, "segundos": self.segundos_previos + resumen["segundos"],
                "reanudado_desde": self.reanudado_desde}

    def terminar(self) -> None:
        """Borra la carpeta al terminar bien."""
        if self.carpeta is not None:
            borrar_en_curso(self.carpeta)

    def __enter__(self) -> EnCurso:
        """La carpeta ya está reservada desde que se creó."""
        return self

    def __exit__(self, *excepcion: object) -> None:
        """Suelta la reserva de la carpeta, haya ido bien o mal."""
        self._reserva.close()


def _comprobar_checkpoint(previo: dict[str, Any], unidad: str, carpeta: Path) -> None:
    """Comprueba que un checkpoint leído tiene lo que guarda EnCurso.

    Raises:
        ValueError: Si le falta o sobra alguna clave o no tienen su tipo.
    """
    claves = set(previo) - {"formato", "version", "metodo"}
    if claves != CLAVES_CHECKPOINT:
        raise ValueError(f"El checkpoint de {carpeta} no está completo: tiene {sorted(claves)}"
                         f" y debe tener {sorted(CLAVES_CHECKPOINT)}")
    estado, segundos = previo["estado"], previo["segundos"]
    if (not isinstance(estado, dict) or not es_entero_positivo(estado.get(unidad))
            or not isinstance(previo["progreso"], list)
            or not all(isinstance(m, dict) for m in previo["progreso"])
            or not isinstance(previo["reanudado_desde"], list)
            or not all(es_entero_positivo(i) for i in previo["reanudado_desde"])
            or not (previo["run"] is None or isinstance(previo["run"], str))
            or not es_real_finito(segundos) or segundos < 0):
        raise ValueError(f"El checkpoint de {carpeta} no tiene el estado, el progreso, las "
                         "reanudaciones, el run o los segundos de uno de EnCurso")


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
    cfg = _config(argumentos)
    cfg["experto"]["semilla"] = int(argumentos["semilla"])
    vista = CallbackVista(canal.imagen, fps_entorno(cfg), argumentos["calidad_jpeg"],
                          activa=argumentos["vista"])
    with EnCurso(argumentos, "experto", cfg, "iteracion") as en_curso:
        run = _abrir_run(cfg, argumentos, "experto", en_curso.reanudado)
        try:
            progreso = ProgresoTrabajo(canal, run, "pasos", en_curso.historial)
            control = ControlEntrenamiento(canal, vista, progreso)
            modelo, resumen = entrenar_experto(
                cfg, progreso, parar=control.parar, callback=control, dibujable=True,
                estado_inicial=en_curso.estado_inicial,
                guardar_estado=en_curso.guardador(progreso, run))
            # La última iteración siempre se evalúa: su mensaje ya salió con la evaluación.
            incompleto = (control.finalizar_pedido and not resumen["objetivo_alcanzado"]
                          and resumen["pasos"] < cfg["experto"]["pasos_totales"])
            resumen = {**en_curso.completar(resumen), "incompleto": incompleto}
            ficha = agregar_nuevo(Path(argumentos["biblioteca"]), argumentos["entorno"], "experto",
                                  argumentos["nombre"].strip(), "dashboard",
                                  lambda carpeta: guardar_experto(carpeta, modelo, cfg, resumen))
            _subir_modelo(run, "experto", _carpeta(argumentos, "experto", ficha),
                          ["modelo.zip", "experto.json"], cfg, resumen)
        except BaseException as error:
            terminar_run(run, error)
            raise
        terminar_run(run)
        en_curso.terminar()
    return _resultado(ficha, incompleto, argumentos["semilla"], resumen)


def _carpeta(argumentos: dict[str, Any], tipo: str, ficha: dict[str, Any]) -> Path:
    """Carpeta en la biblioteca del elemento recién guardado."""
    return Path(argumentos["biblioteca"]) / argumentos["entorno"] / tipo / ficha["id"]


def _config_imitacion(argumentos: dict[str, Any]) -> dict[str, Any]:
    """Config de un BC o GAIL: la semilla pedida es la base de las repeticiones."""
    cfg = _config(argumentos)
    cfg["experimento"]["semilla_repeticiones"] = int(argumentos["semilla"])
    return cfg


def tarea_entrenar_gail(argumentos: dict[str, Any], canal: Any) -> dict[str, Any]:
    """Entrena un GAIL desde el dashboard y lo guarda en la biblioteca.

    Usa entrenar_gail con la repetición 0 de la semilla pedida y los datos elegidos
    (datos_de_entrenamiento), con la vista y las órdenes del navegador. Se guarda como
    en la E1 (guardar_gail) con origen «dashboard».

    Args:
        argumentos: Ver el docstring del módulo.
        canal: CanalTrabajo.

    Returns:
        ResultadoEntrenamiento; incompleto si se finalizó antes de dar todas las
        iteraciones.

    Raises:
        ValueError: Si los argumentos, la configuración o los datos no son válidos.
    """
    argumentos = comprobar_argumentos_entrenamiento(argumentos, "gail")
    cfg = _config_imitacion(argumentos)
    vista = CallbackVista(canal.imagen, fps_entorno(cfg), argumentos["calidad_jpeg"],
                          activa=argumentos["vista"])
    with EnCurso(argumentos, "gail", cfg, "iteracion") as en_curso:
        run = _abrir_run(cfg, argumentos, "gail", en_curso.reanudado)
        try:
            trayectorias, splits, procedencia, cantidad, id_datos = datos_de_entrenamiento(
                argumentos, cfg)
            progreso = ProgresoTrabajo(canal, run, "iteraciones", en_curso.historial)
            control = ControlEntrenamiento(canal, vista, progreso)
            agente, discriminador, resumen = entrenar_gail(
                cfg, trayectorias, splits, cantidad, REPETICION_DASHBOARD, progreso,
                parar=control.parar, callback=control, dibujable=True,
                estado_inicial=en_curso.estado_inicial,
                guardar_estado=en_curso.guardador(progreso, run))
            # La última iteración siempre se valida: su mensaje ya salió con la validación.
            incompleto = (control.finalizar_pedido
                          and resumen["iteraciones"] < cfg["gail"]["iteraciones"])
            resumen = {**en_curso.completar(resumen), "incompleto": incompleto,
                       "datos_biblioteca": id_datos}
            ficha = agregar_nuevo(Path(argumentos["biblioteca"]), argumentos["entorno"], "gail",
                                  argumentos["nombre"].strip(), "dashboard",
                                  lambda carpeta: guardar_gail(carpeta, agente, discriminador, cfg,
                                                               resumen, procedencia))
            _subir_modelo(run, "gail", _carpeta(argumentos, "gail", ficha),
                          ["modelo.zip", "discriminador.pt", "gail.json"], cfg,
                          {**resumen, **procedencia}, {"cantidad": cantidad, **procedencia})
        except BaseException as error:
            terminar_run(run, error)
            raise
        terminar_run(run)
        en_curso.terminar()
    return _resultado(ficha, incompleto, argumentos["semilla"], resumen)


def tarea_entrenar_bc(argumentos: dict[str, Any], canal: Any) -> dict[str, Any]:
    """Entrena un BC desde el dashboard y lo guarda en la biblioteca.

    Usa entrenar_bc_repeticion con la repetición 0 de la semilla pedida y los datos
    elegidos (datos_de_entrenamiento), con un progreso por época y la orden de
    finalizar. Se guarda como en la E1 (guardar_bc_repeticion, politica.pt) con origen
    «dashboard».

    Con ``checkpoint``, guarda uno al acabar cada época y, si ya había uno, sigue
    desde él (ver el docstring del módulo).

    Args:
        argumentos: Ver el docstring del módulo.
        canal: CanalTrabajo.

    Returns:
        ResultadoEntrenamiento; incompleto si se finalizó antes de llegar a las épocas
        máximas (la parada por paciencia no lo es).

    Raises:
        ValueError: Si los argumentos, la configuración o los datos no son válidos.
    """
    argumentos = comprobar_argumentos_entrenamiento(argumentos, "bc")
    cfg = _config_imitacion(argumentos)
    with EnCurso(argumentos, "bc", cfg, "epoca") as en_curso:
        run = _abrir_run(cfg, argumentos, "bc", en_curso.reanudado)
        try:
            trayectorias, splits, procedencia, cantidad, id_datos = datos_de_entrenamiento(
                argumentos, cfg)
            control = ControlEntrenamiento(canal)
            progreso = ProgresoEpocas(canal, run, en_curso.historial)
            politica, resumen = entrenar_bc_repeticion(
                cfg, trayectorias, splits, cantidad, REPETICION_DASHBOARD, progreso,
                parar=control.parar, estado_inicial=en_curso.estado_inicial,
                guardar_estado=en_curso.guardador(progreso, run))
            # Al agotar la paciencia, entrenar_bc para antes de consultar parar: si se pidió
            # finalizar, paró por la orden, salvo que fuera ya la última época.
            incompleto = (control.finalizar_pedido
                          and resumen["epocas"] < cfg["bc"]["epocas_maximas"])
            resumen = {**en_curso.completar(resumen), "incompleto": incompleto,
                       "datos_biblioteca": id_datos}
            ficha = agregar_nuevo(Path(argumentos["biblioteca"]), argumentos["entorno"], "bc",
                                  argumentos["nombre"].strip(), "dashboard",
                                  lambda carpeta: guardar_bc_repeticion(
                                      carpeta / "politica.pt", politica, cfg, resumen,
                                      procedencia))
            _subir_modelo(run, "bc", _carpeta(argumentos, "bc", ficha), ["politica.pt"], cfg,
                          {**resumen, **procedencia}, {"cantidad": cantidad, **procedencia})
        except BaseException as error:
            terminar_run(run, error)
            raise
        terminar_run(run)
        en_curso.terminar()
    return _resultado(ficha, incompleto, argumentos["semilla"], resumen)


def comprobar_argumentos_comparacion(argumentos: object) -> dict[str, Any]:
    """Comprueba los argumentos de una comparación antes de cargar nada.

    Returns:
        Una copia de los argumentos.

    Raises:
        ValueError: Si falta o sobra alguna clave, algún valor no tiene su tipo, hay
            modelos repetidos o no queda ningún agente que comparar.
    """
    if not isinstance(argumentos, dict) or set(argumentos) != ARGUMENTOS_COMPARACION:
        claves = sorted(argumentos) if isinstance(argumentos, dict) else argumentos
        raise ValueError("Los argumentos deben tener exactamente "
                         f"{sorted(ARGUMENTOS_COMPARACION)}, no {claves}")
    for clave in ("ruta_config", "biblioteca", "entorno"):
        if not isinstance(argumentos[clave], str) or not argumentos[clave].strip():
            raise ValueError(f"{clave} debe ser un texto no vacío")
    modelos = argumentos["modelos"]
    if not isinstance(modelos, list) or not all(isinstance(m, str) and m for m in modelos):
        raise ValueError("modelos debe ser una lista de ids de la biblioteca")
    if len(set(modelos)) != len(modelos):
        raise ValueError("modelos no puede tener ids repetidos")
    for clave in ("aleatorio", "vista"):
        if not isinstance(argumentos[clave], bool):
            raise ValueError(f"{clave} debe ser true o false")
    if not modelos and not argumentos["aleatorio"]:
        raise ValueError("No hay nada que comparar: elige algún modelo o el aleatorio")
    if argumentos["partidas"] is not None and not es_entero_positivo(argumentos["partidas"]):
        raise ValueError(f"partidas debe ser un entero positivo o None, no "
                         f"{argumentos['partidas']!r}")
    return dict(argumentos)


def cargar_agente(biblioteca: Biblioteca, id_modelo: str,
                  entorno: str) -> tuple[Any, dict[str, Any] | None]:
    """Carga un modelo de la biblioteca para jugar con predecir_accion.

    Como la evaluación de la E1: el experto y GAIL con sus estadísticas de
    normalización (None si no normalizan) y BC sin ellas.

    Returns:
        El agente (TRPO o PoliticaBC) y sus estadísticas.

    Raises:
        ValueError: Si el id no existe, no es válido, es de otro entorno o no es un
            experto, BC o GAIL.
    """
    try:
        elemento = biblioteca.obtener(id_modelo)
    except KeyError:
        raise ValueError(f"El modelo {id_modelo} no está en la biblioteca") from None
    if elemento["tipo"] not in TIPOS_COMPARABLES:
        raise ValueError(f"{id_modelo} es {elemento['tipo']}, no un modelo")
    if elemento["entorno"] != entorno:
        raise ValueError(f"{id_modelo} es de {elemento['entorno']}, no de {entorno}")
    if not elemento["valido"]:
        raise ValueError(f"{id_modelo} no es válido: {elemento['motivo']}")
    carpeta = biblioteca.ruta(id_modelo)
    if elemento["tipo"] == "bc":
        politica, _ = cargar_politica_bc(carpeta / "politica.pt")
        politica.eval()
        return politica, None
    if elemento["tipo"] == "experto":
        modelo, metadatos = cargar_experto(carpeta)
    else:
        modelo, _, metadatos = cargar_gail(carpeta)
    return modelo, metadatos["resumen"]["normalizacion"]


class _Finalizada(Exception):
    """Se pidió finalizar la comparación en mitad de una partida."""


def tarea_comparar(argumentos: dict[str, Any], canal: Any) -> dict[str, Any]:
    """Compara modelos de la biblioteca (y el aleatorio) jugando las mismas partidas.

    Cada modelo juega como en evaluar_politica de la E1, así que sus retornos son los
    mismos que daría evaluar_final con esas partidas. Ver el docstring del módulo.

    Args:
        argumentos: Ver el docstring del módulo.
        canal: CanalTrabajo.

    Returns:
        ``{"partidas", "semilla", "incompleta", "modelos": [{"id", "retornos"}]}``, con
        ``partidas`` las rondas completas y los modelos en el orden pedido (el aleatorio,
        el último); incompleta si se finalizó antes de jugarlas todas.

    Raises:
        ValueError: Si los argumentos, la configuración o algún modelo no son válidos.
    """
    argumentos = comprobar_argumentos_comparacion(argumentos)
    cfg = resolver_configuracion(Path(argumentos["ruta_config"]), argumentos["entorno"])
    evaluacion = config_evaluacion(cfg)
    partidas = argumentos["partidas"] or evaluacion["episodios"]
    biblioteca = Biblioteca(Path(argumentos["biblioteca"]),
                            entornos_gym(argumentos["ruta_config"]))
    agentes = [(id_modelo, *cargar_agente(biblioteca, id_modelo, argumentos["entorno"]))
               for id_modelo in argumentos["modelos"]]
    if argumentos["aleatorio"]:
        agentes.append((ALEATORIO, ALEATORIO, None))
    entorno = crear_entorno(cfg, None, render_mode="rgb_array")
    try:
        entorno = EntornoVista(entorno, canal.imagen, _fps(entorno, cfg),
                               argumentos["calidad_jpeg"], activa=argumentos["vista"])
    except BaseException:
        entorno.close()
        raise

    def atender_ordenes() -> None:
        for orden in canal.ordenes():
            if orden.get("tipo") == "vista":
                entorno.activa = bool(orden.get("activa"))
            elif orden.get("tipo") == "finalizar":
                raise _Finalizada

    retornos: dict[str, list[float]] = {id_modelo: [] for id_modelo, _, _ in agentes}
    incompleta = False
    try:
        with torch.no_grad():
            for indice, semilla in enumerate(semillas_independientes(evaluacion["semilla"],
                                                                     partidas)):
                ronda = []
                for id_modelo, agente, estadisticas in agentes:

                    def politica(observacion: np.ndarray, agente: Any = agente,
                                 estadisticas: dict[str, Any] | None = estadisticas,
                                 ) -> int | np.ndarray:
                        atender_ordenes()
                        return predecir_accion(agente, observacion, entorno.action_space,
                                               evaluacion["determinista"], None, estadisticas)

                    partida = jugar_partida(entorno, politica, semilla, indice)
                    retorno = retorno_partida(partida)
                    ronda.append((id_modelo, retorno))
                    canal.progreso({"modelo": id_modelo, "partida": indice, "retorno": retorno,
                                    "pasos": partida.numero_pasos})
                for id_modelo, retorno in ronda:
                    retornos[id_modelo].append(retorno)
    except _Finalizada:
        incompleta = True
    finally:
        entorno.close()
    return {"partidas": len(next(iter(retornos.values()))),
            "semilla": int(evaluacion["semilla"]), "incompleta": incompleta,
            "modelos": [{"id": id_modelo, "retornos": valores}
                        for id_modelo, valores in retornos.items()]}


def a_json_profundo(valor: Any) -> Any:
    """Copia un valor con escalares de NumPy convertidos a tipos de Python (a_json)."""
    if isinstance(valor, dict):
        return {str(clave): a_json_profundo(v) for clave, v in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [a_json_profundo(v) for v in valor]
    if valor is None or isinstance(valor, (bool, int, float, str)):
        return valor
    return a_json(valor)
