"""Trabajadores del dashboard: las tareas que el gestor de trabajos ejecuta en un proceso.

El gestor (src/app/trabajos.py) llama a cada tarea con ``(argumentos, canal)`` en un
proceso propio. La tarea entrena con las mismas funciones que los scripts de la E1
(entrenar_experto…), manda el progreso y las imágenes por el canal, obedece las
órdenes del navegador y, al terminar, guarda el modelo en la biblioteca y devuelve el
resultado (ResultadoEntrenamiento en src/app/contrato.py).

Hay una tarea por método: tarea_entrenar_experto, tarea_entrenar_gail y
tarea_entrenar_bc. Argumentos de un entrenamiento (los prepara la ruta
POST /api/entrenamientos):

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
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback

from src.data.biblioteca import Biblioteca, agregar_nuevo
from src.data.entornos import crear_entorno
from src.data.seleccion import preparar_datos
from src.data.splits import SplitsDemostraciones, huella_archivo, validar_splits
from src.data.tipos import Trayectoria
from src.training.barrido import cargar_datos
from src.training.bc import entrenar_bc_repeticion, guardar_bc_repeticion
from src.training.gail import entrenar_gail, guardar_gail
from src.training.trpo import entrenar_experto, guardar_experto
from src.training.vista import CallbackVista
from src.utils.configuracion import TipoEntorno, resolver_configuracion
from src.utils.procedencia import a_json
from src.utils.tracking import (abrir_run, nombre_artefacto, registrar, registrar_resumen,
                                subir_artefacto, terminar_run)
from src.utils.validacion import es_entero_positivo, es_semilla

# Claves que tiene que traer argumentos en un entrenamiento de cada método.
ARGUMENTOS_ENTRENAMIENTO = frozenset({"ruta_config", "biblioteca", "calidad_jpeg", "entorno",
                                      "nombre", "ajustes", "semilla", "vista", "wandb"})
ARGUMENTOS_IMITACION = ARGUMENTOS_ENTRENAMIENTO | {"datos", "cantidad"}
ARGUMENTOS_POR_METODO = {"experto": ARGUMENTOS_ENTRENAMIENTO, "gail": ARGUMENTOS_IMITACION,
                         "bc": ARGUMENTOS_IMITACION}
# Repetición que entrena un BC o GAIL suelto: la 0 de la semilla pedida, como en la E1.
REPETICION_DASHBOARD = 0
# Secciones de las medidas que se mandan juntas con las de la iteración en la que ocurren.
SECCIONES_EVALUACION = ("evaluacion/", "validacion/")


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


class ProgresoEpocas:
    """Hace de run para entrenar_bc: reenvía a W&B y manda un progreso por época.

    entrenar_bc registra la pérdida de cada batch (``bc/train/perdida``) y, al validar
    cada época, sus medidas (``bc/val/…``). A W&B llega todo, como en la E1; al canal,
    un mensaje por época con la media de las pérdidas de sus batches y la validación.
    """

    def __init__(self, canal: Any, run: Any | None = None) -> None:
        """Prepara el progreso.

        Args:
            canal: CanalTrabajo de la tarea.
            run: Run de W&B o None.
        """
        self.canal, self.run = canal, run
        self.epoca = 0
        self._perdidas: list[float] = []

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
        fps = entorno.metadata.get("render_fps")
    finally:
        entorno.close()
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


def _abrir_run(cfg: dict[str, Any], argumentos: dict[str, Any], metodo: str) -> Any | None:
    """Run de W&B del entrenamiento (None sin W&B), en el grupo pedido."""
    return abrir_run(cfg, grupo=argumentos["wandb"]["grupo"], job_type=metodo,
                     nombre=argumentos["nombre"],
                     etiquetas=[f"entorno:{cfg['entorno']['id']}", f"metodo:{metodo}",
                                "origen:dashboard"])


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
    run = _abrir_run(cfg, argumentos, "experto")
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
        _subir_modelo(run, "experto", _carpeta(argumentos, "experto", ficha),
                      ["modelo.zip", "experto.json"], cfg, resumen)
    except BaseException as error:
        terminar_run(run, error)
        raise
    terminar_run(run)
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
    run = _abrir_run(cfg, argumentos, "gail")
    try:
        trayectorias, splits, procedencia, cantidad, id_datos = datos_de_entrenamiento(
            argumentos, cfg)
        progreso = ProgresoTrabajo(canal, run, "iteraciones")
        control = ControlEntrenamiento(canal, vista, progreso)
        agente, discriminador, resumen = entrenar_gail(
            cfg, trayectorias, splits, cantidad, REPETICION_DASHBOARD, progreso,
            parar=control.parar, callback=control, dibujable=True)
        # La última iteración siempre se valida: su mensaje ya salió con la validación.
        incompleto = (control.finalizar_pedido
                      and resumen["iteraciones"] < cfg["gail"]["iteraciones"])
        resumen = {**resumen, "incompleto": incompleto, "datos_biblioteca": id_datos}
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
    return _resultado(ficha, incompleto, argumentos["semilla"], resumen)


def tarea_entrenar_bc(argumentos: dict[str, Any], canal: Any) -> dict[str, Any]:
    """Entrena un BC desde el dashboard y lo guarda en la biblioteca.

    Usa entrenar_bc_repeticion con la repetición 0 de la semilla pedida y los datos
    elegidos (datos_de_entrenamiento), con un progreso por época y la orden de
    finalizar. Se guarda como en la E1 (guardar_bc_repeticion, politica.pt) con origen
    «dashboard».

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
    run = _abrir_run(cfg, argumentos, "bc")
    try:
        trayectorias, splits, procedencia, cantidad, id_datos = datos_de_entrenamiento(
            argumentos, cfg)
        control = ControlEntrenamiento(canal)
        politica, resumen = entrenar_bc_repeticion(
            cfg, trayectorias, splits, cantidad, REPETICION_DASHBOARD,
            ProgresoEpocas(canal, run), parar=control.parar)
        # Al agotar la paciencia, entrenar_bc para antes de consultar parar: si se pidió
        # finalizar, paró por la orden, salvo que fuera ya la última época.
        incompleto = (control.finalizar_pedido
                      and resumen["epocas"] < cfg["bc"]["epocas_maximas"])
        resumen = {**resumen, "incompleto": incompleto, "datos_biblioteca": id_datos}
        ficha = agregar_nuevo(Path(argumentos["biblioteca"]), argumentos["entorno"], "bc",
                              argumentos["nombre"].strip(), "dashboard",
                              lambda carpeta: guardar_bc_repeticion(
                                  carpeta / "politica.pt", politica, cfg, resumen, procedencia))
        _subir_modelo(run, "bc", _carpeta(argumentos, "bc", ficha), ["politica.pt"], cfg,
                      {**resumen, **procedencia}, {"cantidad": cantidad, **procedencia})
    except BaseException as error:
        terminar_run(run, error)
        raise
    terminar_run(run)
    return _resultado(ficha, incompleto, argumentos["semilla"], resumen)


def a_json_profundo(valor: Any) -> Any:
    """Copia un valor con escalares de NumPy convertidos a tipos de Python (a_json)."""
    if isinstance(valor, dict):
        return {str(clave): a_json_profundo(v) for clave, v in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [a_json_profundo(v) for v in valor]
    if valor is None or isinstance(valor, (bool, int, float, str)):
        return valor
    return a_json(valor)
