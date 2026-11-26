"""TRPO de sb3-contrib para el experto y como generador de GAIL.

- crear_trpo traduce la sección trpo de la configuración a sb3-contrib.
- iteracion_trpo da un bloque: una recogida de n_steps × n_envs pasos y una
  actualización, sin reiniciar el entorno entre bloques. Es lo que la rama gail
  repite intercalando la actualización del discriminador.
- entrenar_experto entrena con la recompensa real, evalúa cada cierto número de
  bloques y se queda con el mejor modelo.
- guardar_experto y cargar_experto guardan el modelo y sus metadatos.
- politica_trpo convierte el modelo en una función de observación a acción, la
  que usa la grabación de demostraciones (jugar_partida); cargar_politica_experto
  la prepara desde un experto guardado. evaluar_retorno delega en el protocolo
  común de evaluación (evaluar_politica, en evaluacion.py).

Las semillas salen de src/utils/seed.py: nunca se pasa seed= a TRPO, porque sb3
volvería a sembrar los generadores globales y los entornos por su cuenta.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import spaces
from sb3_contrib import TRPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv, VecNormalize

from src.data.demostraciones import Politica, comprobar_espacios, config_demostraciones
from src.data.entornos import crear_entorno_vectorizado
from src.data.splits import huella_archivo
from src.models.factory import ACTIVACIONES
from src.training.evaluacion import evaluar_politica, normalizar_observacion
from src.utils.archivos import comprobar_directorio_libre, escribir_carpeta_nueva
from src.utils.procedencia import a_json, estado_git, ruta_registrada, versiones_y_hardware
from src.utils.reglas_config import config_experto as _config_experto
from src.utils.reglas_config import config_trpo, config_trpo_experto
from src.utils.seed import (estado_aleatorio, fijar_seed, restaurar_estado_aleatorio,
                            semilla_reanudacion, semillas_independientes, validar_semilla)
from src.utils.tracking import con_seccion, registrar
from src.utils.validacion import es_entero_positivo

_ARCHIVO_MODELO = "modelo.zip"
_ARCHIVO_POLITICA = "policy.pth"  # pesos de la política dentro del zip de sb3
# Sección de W&B de las medidas de iteracion_trpo, que se devuelven sin prefijo.
SECCION_TRPO = "trpo"
_ARCHIVO_METADATOS = "experto.json"


def _comprobar_espacios(observacion: spaces.Space, accion: spaces.Space) -> None:
    """Admite los espacios de comprobar_espacios, con Discrete solo si empieza en 0."""
    comprobar_espacios(observacion, accion)
    if isinstance(accion, spaces.Discrete) and int(accion.start) != 0:
        raise ValueError("Espacio de acción no compatible: se admite Discrete con start 0 "
                         "(sb3 genera índices 0..n-1) o Box de una dimensión")


def crear_trpo(cfg: dict, venv: VecEnv) -> TRPO:
    """Crea un agente TRPO de sb3-contrib con los hiperparámetros de la configuración.

    La política y la red de valor son perceptrones con trpo.capas_ocultas y
    trpo.activacion (2 × 100 tanh en el paper), con trpo.log_std_inicial (solo
    acciones continuas) y trpo.inicializacion_ortogonal. Cada iteración recoge
    trpo.pasos_por_iteracion pasos repartidos entre los entornos del VecEnv, que
    puede estar envuelto (por EntornoRecompensaGAIL o VecNormalize). No recibe
    semilla: hay que llamar antes a fijar_seed y sembrar los entornos.

    Args:
        cfg: Configuración resuelta con la sección trpo.
        venv: VecEnv con trpo.n_entornos instancias.

    Returns:
        El agente, en CPU: una MLP pequeña on-policy no gana con GPU.

    Raises:
        ValueError: Si la configuración o los espacios no son válidos, o si el
            VecEnv no tiene trpo.n_entornos instancias.
    """
    tcfg = config_trpo(cfg)
    if not isinstance(venv, VecEnv):
        raise ValueError("venv debe ser un VecEnv")
    if venv.num_envs != tcfg["n_entornos"]:
        raise ValueError(f"El VecEnv tiene {venv.num_envs} entornos y trpo.n_entornos "
                         f"pide {tcfg['n_entornos']}")
    _comprobar_espacios(venv.observation_space, venv.action_space)
    capas = list(tcfg["capas_ocultas"])
    return TRPO(
        "MlpPolicy", venv,
        learning_rate=float(tcfg["learning_rate_valor"]),
        n_steps=tcfg["pasos_por_iteracion"] // tcfg["n_entornos"],
        batch_size=tcfg["batch_size_valor"],
        gamma=float(tcfg["gamma"]),
        cg_max_steps=tcfg["pasos_gradiente_conjugado"],
        cg_damping=float(tcfg["cg_damping"]),
        line_search_shrinking_factor=float(tcfg["busqueda_lineal_factor"]),
        line_search_max_iter=tcfg["busqueda_lineal_maximo"],
        n_critic_updates=tcfg["pasos_valor"],
        gae_lambda=float(tcfg["gae_lambda"]),
        normalize_advantage=tcfg["normalizar_ventajas"],
        target_kl=float(tcfg["target_kl"]),
        policy_kwargs={"net_arch": {"pi": capas, "vf": list(capas)},
                       "activation_fn": ACTIVACIONES[tcfg["activacion"]],
                       "log_std_init": float(tcfg["log_std_inicial"]),
                       "ortho_init": tcfg["inicializacion_ortogonal"]},
        device="cpu",
        verbose=0,
    )


def iteracion_trpo(modelo: TRPO, callback: BaseCallback | None = None) -> dict[str, float]:
    """Da una iteración de TRPO: una recogida de n_steps × n_envs pasos y una actualización.

    Usa learn con reset_num_timesteps=False, así que el entorno no se reinicia
    entre iteraciones: las partidas a medias continúan en la siguiente. El
    retorno real sale de los Monitor del VecEnv, aunque un envoltorio cambie la
    recompensa que recibe el agente. Antes de la recogida se vacía el registro
    de partidas de sb3 (ep_info_buffer), que si no acumula las últimas 100: así
    las medidas son solo de las partidas que terminan en esta iteración.

    Args:
        modelo: Agente TRPO de sb3-contrib.
        callback: Callback de sb3 que recibe cada paso de la recogida (lo usa la
            vista del dashboard); no cambia lo que se entrena.

    Returns:
        pasos acumulados, partidas terminadas en la iteración y su
        retorno_real_medio y longitud_media (nan si no terminó ninguna).
    """
    if not isinstance(modelo, TRPO):
        raise ValueError("modelo debe ser un TRPO de sb3-contrib")
    if modelo.ep_info_buffer is not None:
        modelo.ep_info_buffer.clear()
    modelo.learn(total_timesteps=modelo.n_steps * modelo.n_envs, reset_num_timesteps=False,
                 callback=callback)
    partidas = list(modelo.ep_info_buffer)
    return {
        "pasos": float(modelo.num_timesteps),
        "partidas": float(len(partidas)),
        "retorno_real_medio": (float(np.mean([p["r"] for p in partidas]))
                               if partidas else float("nan")),
        "longitud_media": (float(np.mean([p["l"] for p in partidas]))
                           if partidas else float("nan")),
    }


def estadisticas_normalizacion(venv: VecEnv) -> dict[str, Any] | None:
    """Devuelve las estadísticas de VecNormalize del VecEnv, o None si no normaliza.

    Son las que hay que aplicar a las observaciones al evaluar o al usar el
    experto fuera de su VecEnv (normalizar_observacion).
    """
    while isinstance(venv, VecEnv):
        if isinstance(venv, VecNormalize):
            return {"media": venv.obs_rms.mean.tolist(), "varianza": venv.obs_rms.var.tolist(),
                    "epsilon": float(venv.epsilon), "limite": float(venv.clip_obs)}
        venv = getattr(venv, "venv", None)
    return None


def _vecnormalize(venv: VecEnv) -> VecNormalize | None:
    """El VecNormalize de la cadena de envoltorios, o None si no normaliza."""
    while isinstance(venv, VecEnv):
        if isinstance(venv, VecNormalize):
            return venv
        venv = getattr(venv, "venv", None)
    return None


def estado_normalizacion(venv: VecEnv) -> dict[str, Any] | None:
    """Estado completo de VecNormalize para un checkpoint (None si no normaliza).

    A diferencia de estadisticas_normalizacion, lleva también cuántas observaciones
    ha visto (count), que hace falta para seguir actualizándolas igual al reanudar.
    """
    normalizar = _vecnormalize(venv)
    if normalizar is None:
        return None
    return {"media": normalizar.obs_rms.mean.tolist(), "varianza": normalizar.obs_rms.var.tolist(),
            "cuenta": float(normalizar.obs_rms.count)}


def restaurar_normalizacion(venv: VecEnv, estado: dict[str, Any] | None) -> None:
    """Deja el VecNormalize de venv como lo guardó estado_normalizacion.

    Raises:
        ValueError: Si el estado no encaja con el venv (normaliza uno y el otro no, o
            las dimensiones o los valores no valen).
    """
    normalizar = _vecnormalize(venv)
    if (normalizar is None) != (estado is None):
        raise ValueError("El checkpoint y la configuración no coinciden en si se normalizan "
                         "las observaciones")
    if normalizar is None:
        return
    try:
        media = np.asarray(estado["media"], dtype=np.float64)
        varianza = np.asarray(estado["varianza"], dtype=np.float64)
        cuenta = float(estado["cuenta"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Normalización del checkpoint no válida: {error}") from error
    forma = normalizar.obs_rms.mean.shape
    if (media.shape != forma or varianza.shape != forma or not np.isfinite(media).all()
            or not np.isfinite(varianza).all() or (varianza < 0).any()
            or not np.isfinite(cuenta) or cuenta <= 0):
        raise ValueError("Normalización del checkpoint no válida para este entorno")
    normalizar.obs_rms.mean, normalizar.obs_rms.var, normalizar.obs_rms.count = (
        media, varianza, cuenta)


def politica_trpo(modelo: TRPO, determinista: bool,
                  estadisticas: dict[str, Any] | None = None) -> Politica:
    """Convierte el modelo en una función de observación en bruto a acción.

    Aplica la normalización del experto (normalizar_observacion) antes de
    predict, así que recibe las observaciones tal como las da el entorno. Con
    determinista False muestrea la acción con el generador global de PyTorch;
    jugar_partida lo aísla por partida.

    Args:
        modelo: Agente TRPO.
        determinista: True para la acción más probable, False para muestrear.
        estadisticas: De estadisticas_normalizacion si el modelo normaliza.

    Raises:
        ValueError: Si modelo no es un TRPO o determinista no es booleano.
    """
    if not isinstance(modelo, TRPO):
        raise ValueError("modelo debe ser un TRPO de sb3-contrib")
    if not isinstance(determinista, bool):
        raise ValueError(f"determinista debe ser booleano, no {determinista!r}")

    def politica(observacion: np.ndarray) -> np.ndarray:
        """Devuelve la acción del modelo para una observación en bruto."""
        accion, _ = modelo.predict(normalizar_observacion(observacion, estadisticas),
                                   deterministic=determinista)
        return accion

    return politica


def evaluar_retorno(modelo: TRPO, cfg: dict, episodios: int, semilla: int, determinista: bool,
                    estadisticas: dict[str, Any] | None = None) -> dict[str, Any]:
    """Juega episodios completos con el modelo y mide el retorno real.

    Usa el protocolo común de evaluacion-checkpoints. Cada episodio recibe una
    semilla derivada de semilla, compartible con BC y GAIL, y la recompensa es
    la real del entorno incluso si el modelo se entrenó con GAIL.

    Args:
        modelo: Agente TRPO.
        cfg: Configuración resuelta con entorno.id.
        episodios: Número de episodios, entero positivo.
        semilla: Semilla base no negativa de la evaluación.
        determinista: True para la acción más probable, False para muestrear.
        estadisticas: De estadisticas_normalizacion si el modelo normaliza.

    Returns:
        retorno_medio, retorno_desviacion, longitud_media, retornos y
        longitudes (una por episodio, en el orden de las semillas).

    Raises:
        ValueError: Si episodios o semilla no son válidos.
    """
    return evaluar_politica(modelo, cfg, episodios, semilla, determinista, estadisticas)


def entrenar_experto(cfg: dict, run: Any = None, parar: Callable[[], bool] | None = None,
                     callback: BaseCallback | None = None, dibujable: bool = False,
                     estado_inicial: dict[str, Any] | None = None,
                     guardar_estado: Callable[[dict[str, Any]], None] | None = None,
                     ) -> tuple[TRPO, dict[str, Any]]:
    """Entrena el experto TRPO con la recompensa real del entorno.

    Usa la sección trpo con los cambios de experto.trpo (config_trpo_experto).

    De experto.semilla salen tres semillas independientes: la del modelo
    (fijar_seed), la de los entornos de entrenamiento y la de evaluación. Se
    entrena por iteraciones hasta llegar a experto.pasos_totales; cada
    experto.evaluar_cada iteraciones, y al final, se evalúa con
    experto.episodios_evaluacion episodios (política estocástica si
    experto.evaluacion_determinista es false, la que graba las demostraciones).
    Se conserva el mejor modelo y se para en cuanto el retorno medio alcanza
    experto.retorno_objetivo. Al terminar, el modelo tiene los pesos del mejor.

    Args:
        cfg: Configuración resuelta con las secciones entorno, trpo y experto.
        run: Run de W&B o None; recibe las medidas de cada iteración.
        parar: Se consulta tras cada iteración; si devuelve True, se evalúa el
            modelo de esa iteración y se termina como si fuera la última (el
            «finalizar» del dashboard). Con None se entrena hasta el final.
        callback: Callback de sb3 para cada paso de entrenamiento (la vista del
            dashboard); no se usa al evaluar.
        dibujable: Crea el primer entorno de entrenamiento con
            render_mode="rgb_array", el que dibuja la vista (src/training/vista.py).
        estado_inicial: Estado que dio guardar_estado (checkpoint): sigue desde la
            iteración siguiente con los mismos pesos (política y valor), optimizador,
            normalización, mejor modelo, historial y generadores. Los entornos no se
            pueden guardar a mitad de partida, así que se crean nuevos con
            semilla_reanudacion: el resultado no es el mismo que sin cortar, pero
            reanudar dos veces desde el mismo estado da lo mismo.
        guardar_estado: Recibe el estado tras cada evaluación tras la que el
            entrenamiento sigue (no al alcanzar el objetivo, al parar ni al acabar).

    Returns:
        El modelo y un resumen: pasos, iteraciones, mejor retorno y su
        iteración, si se alcanzó el objetivo, historial de evaluaciones,
        segundos medidos (solo los de esta llamada), semillas (con
        entornos_reanudacion si se reanudó), estadísticas de normalización y entorno.

    Cada iteración recoge un bloque completo, así que si experto.pasos_totales no es
    múltiplo de trpo.pasos_por_iteracion se para en la primera iteración que lo
    supera (decisión del usuario): los pasos reales quedan en el resumen.

    Raises:
        ValueError: Si la configuración no es válida o estado_inicial no es de un
            entrenamiento con esta configuración.
    """
    ecfg = _config_experto(cfg)
    cfg_trpo = config_trpo_experto(cfg)
    tcfg = cfg_trpo["trpo"]
    semilla_base = validar_semilla(ecfg["semilla"])
    semilla_modelo, semilla_entornos, semilla_evaluacion = semillas_independientes(
        semilla_base, 3)
    semillas = {"base": semilla_base, "modelo": semilla_modelo, "entornos": semilla_entornos,
                "evaluacion": semilla_evaluacion}
    if estado_inicial is not None:
        _comprobar_estado_experto(estado_inicial, ecfg, tcfg)
        semillas["entornos_reanudacion"] = semilla_reanudacion(semilla_entornos,
                                                               estado_inicial["iteracion"])
    fijar_seed(semilla_modelo)
    venv: VecEnv = crear_entorno_vectorizado(
        cfg, tcfg["n_entornos"], semillas.get("entornos_reanudacion", semilla_entornos),
        dibujable)
    if tcfg["normalizar_observaciones"]:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=False,
                            clip_obs=float(tcfg["limite_normalizacion"]))
    inicio = time.perf_counter()
    try:
        modelo = crear_trpo(cfg_trpo, venv)
        mejor: dict[str, Any] | None = None
        historial = []
        iteracion, alcanzado = 0, False
        if estado_inicial is not None:
            mejor, historial, iteracion = _reanudar_experto(estado_inicial, modelo, venv)
        while modelo.num_timesteps < ecfg["pasos_totales"]:
            iteracion += 1
            medidas = iteracion_trpo(modelo, callback)
            registrar(run, con_seccion(medidas, SECCION_TRPO), paso=modelo.num_timesteps)
            detener = parar is not None and parar()
            if (iteracion % ecfg["evaluar_cada"]
                    and modelo.num_timesteps < ecfg["pasos_totales"] and not detener):
                continue
            estadisticas = estadisticas_normalizacion(venv)
            evaluacion = evaluar_retorno(modelo, cfg, ecfg["episodios_evaluacion"],
                                         semilla_evaluacion, ecfg["evaluacion_determinista"],
                                         estadisticas)
            evaluacion.update(iteracion=iteracion, pasos=modelo.num_timesteps)
            historial.append(evaluacion)
            registrar(run, {f"evaluacion/{k}": v for k, v in evaluacion.items()},
                      paso=modelo.num_timesteps)
            if mejor is None or evaluacion["retorno_medio"] > mejor["retorno_medio"]:
                mejor = {**evaluacion,
                         "pesos": copy.deepcopy(modelo.policy.state_dict()),
                         "estadisticas": copy.deepcopy(estadisticas)}
            if evaluacion["retorno_medio"] >= ecfg["retorno_objetivo"]:
                alcanzado = True
                break
            if detener:
                break
            if guardar_estado is not None and modelo.num_timesteps < ecfg["pasos_totales"]:
                guardar_estado({
                    "iteracion": iteracion, "pasos": int(modelo.num_timesteps),
                    "politica": copy.deepcopy(modelo.policy.state_dict()),
                    "optimizador": copy.deepcopy(modelo.policy.optimizer.state_dict()),
                    # mejor no se modifica después (se sustituye): no hace falta copiarlo.
                    "normalizacion": estado_normalizacion(venv), "mejor": mejor,
                    "historial": copy.deepcopy(historial), "aleatorio": estado_aleatorio()})
        modelo.policy.load_state_dict(mejor["pesos"])
        segundos = time.perf_counter() - inicio
    finally:
        venv.close()
    resumen = {
        "entorno": cfg["entorno"]["id"],
        "pasos": int(modelo.num_timesteps),
        "iteraciones": iteracion,
        "mejor_retorno": mejor["retorno_medio"],
        "mejor_iteracion": mejor["iteracion"],
        "objetivo": ecfg["retorno_objetivo"],
        "objetivo_alcanzado": alcanzado,
        "evaluaciones": historial,
        "segundos": segundos,
        "semillas": semillas,
        "normalizacion": mejor["estadisticas"],
    }
    return modelo, resumen


_CLAVES_ESTADO_EXPERTO = frozenset({"iteracion", "pasos", "politica", "optimizador",
                                    "normalizacion", "mejor", "historial", "aleatorio"})


def _comprobar_estado_experto(estado: object, ecfg: dict[str, Any],
                              tcfg: dict[str, Any]) -> None:
    """Comprueba, antes de crear nada, que el estado encaja con la configuración.

    Raises:
        ValueError: Si le faltan o sobran claves, o la iteración, los pasos, la mejor
            evaluación o el historial no encajan con experto y trpo.
    """
    if not isinstance(estado, dict) or set(estado) != _CLAVES_ESTADO_EXPERTO:
        raise ValueError("El estado del experto no tiene las claves de un checkpoint")
    iteracion, pasos = estado["iteracion"], estado["pasos"]
    if not (es_entero_positivo(iteracion) and es_entero_positivo(pasos)
            and pasos == iteracion * tcfg["pasos_por_iteracion"]
            and pasos < ecfg["pasos_totales"] and iteracion % ecfg["evaluar_cada"] == 0):
        raise ValueError("El estado del experto no encaja con trpo.pasos_por_iteracion, "
                         "experto.pasos_totales y experto.evaluar_cada")
    mejor, historial = estado["mejor"], estado["historial"]
    if (not isinstance(historial, list) or not historial
            or not all(isinstance(e, dict) for e in historial)
            or historial[-1].get("iteracion") != iteracion
            or not isinstance(mejor, dict) or not {"pesos", "estadisticas", "retorno_medio",
                                                   "iteracion"} <= mejor.keys()
            or not isinstance(mejor["retorno_medio"], float)
            or mejor["retorno_medio"] >= ecfg["retorno_objetivo"]):
        raise ValueError("El estado del experto no tiene un historial y una mejor "
                         "evaluación que encajen")


def _reanudar_experto(estado: dict[str, Any], modelo: TRPO,
                      venv: VecEnv) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    """Carga un estado de guardar_estado en el modelo, la normalización y los generadores.

    Returns:
        La mejor evaluación (con sus pesos), el historial y la iteración ya hecha.

    Raises:
        ValueError: Si los pesos, el optimizador o la normalización no son de este
            modelo y entorno.
    """
    try:
        modelo.policy.load_state_dict(estado["politica"])
        modelo.policy.optimizer.load_state_dict(estado["optimizador"])
        # Que los pesos del mejor se puedan cargar, antes de entrenar.
        copy.deepcopy(modelo.policy).load_state_dict(estado["mejor"]["pesos"])
    except (RuntimeError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"El estado del experto no corresponde a este modelo: {error}") from error
    restaurar_normalizacion(venv, estado["normalizacion"])
    modelo.num_timesteps = int(estado["pasos"])
    restaurar_estado_aleatorio(estado["aleatorio"])
    return copy.deepcopy(estado["mejor"]), copy.deepcopy(estado["historial"]), int(
        estado["iteracion"])


def guardar_experto(directorio: Path, modelo: TRPO, cfg: dict,
                    resumen: dict[str, Any]) -> None:
    """Guarda el experto en un directorio nuevo: modelo.zip y experto.json.

    experto.json lleva la configuración resuelta, el resumen del entrenamiento
    (con las semillas, el tiempo medido y la normalización), el commit de git y
    si había cambios sin commit (estado_git), las versiones y el hardware,
    como JSON estricto (sin NaN ni infinitos). No sobrescribe
    (comprobar_directorio_libre).

    Raises:
        FileExistsError: Si el directorio existe con contenido.
        ValueError: Si modelo no es un TRPO o los metadatos tienen NaN o infinitos.
    """
    if not isinstance(modelo, TRPO):
        raise ValueError("modelo debe ser un TRPO de sb3-contrib")
    directorio = comprobar_directorio_libre(directorio)
    metadatos = {"configuracion": cfg, "resumen": resumen, "codigo": estado_git(),
                 "entorno_ejecucion": versiones_y_hardware()}
    # Se serializa antes de crear nada: un NaN no deja una carpeta a medias.
    texto = json.dumps(metadatos, ensure_ascii=False, indent=2, allow_nan=False,
                       default=a_json)

    def escribir(carpeta: Path) -> None:
        """Escribe experto.json y modelo.zip en carpeta."""
        with (carpeta / _ARCHIVO_METADATOS).open("x", encoding="utf-8") as archivo:
            archivo.write(texto)
        modelo.save(carpeta / _ARCHIVO_MODELO)

    escribir_carpeta_nueva(directorio, escribir)


def cargar_experto(directorio: Path) -> tuple[TRPO, dict[str, Any]]:
    """Carga un experto guardado con guardar_experto, en CPU.

    Returns:
        El modelo y los metadatos; las estadísticas de normalización están en
        metadatos["resumen"]["normalizacion"] (None si no normaliza).

    Raises:
        FileNotFoundError: Si falta modelo.zip o experto.json.
    """
    directorio = Path(directorio)
    for nombre in (_ARCHIVO_MODELO, _ARCHIVO_METADATOS):
        if not (directorio / nombre).is_file():
            raise FileNotFoundError(f"Falta {nombre} en {directorio}")
    with (directorio / _ARCHIVO_METADATOS).open(encoding="utf-8") as archivo:
        metadatos = json.load(archivo)
    return TRPO.load(directorio / _ARCHIVO_MODELO, device="cpu"), metadatos


def huella_politica(modelo: Path) -> str:
    """SHA-256 de los pesos de la política (policy.pth dentro del modelo.zip de sb3).

    La huella del zip entero cambia al reentrenar con la misma config, porque sb3
    guarda en él la hora de inicio, los tiempos de las partidas y la fecha de cada
    entrada; los pesos salen idénticos.

    Raises:
        ValueError: Si el archivo no es un zip con policy.pth.
    """
    try:
        with zipfile.ZipFile(modelo) as archivo:
            contenido = archivo.read(_ARCHIVO_POLITICA)
    except (zipfile.BadZipFile, KeyError) as error:
        raise ValueError(f"{modelo} no es un modelo de sb3 con {_ARCHIVO_POLITICA}") from error
    return hashlib.sha256(contenido).hexdigest()


def cargar_politica_experto(directorio: Path, cfg: dict) -> tuple[Politica, dict[str, Any]]:
    """Prepara un experto guardado para grabar demostraciones y describe su procedencia.

    Carga el experto con cargar_experto, comprueba que se entrenó en el
    entorno de la configuración y crea su política con politica_trpo, en el
    modo de demostraciones.determinista y con la normalización guardada.

    Los metadatos son los que exige guardar_demostraciones. No llevan la
    fecha: así regenerar las demostraciones con el mismo código, experto y
    máquina da un archivo idéntico, con la misma huella.

    Args:
        directorio: Carpeta de guardar_experto.
        cfg: Configuración resuelta con entorno.id y la sección demostraciones.

    Returns:
        La política y los metadatos: entorno; experto (carpeta, huellas de
        modelo.zip y experto.json, resultado de su entrenamiento y su commit);
        procedencia (generador, código con el que se graba, versiones,
        hardware y sección demostraciones); y configuracion_experto.

    Raises:
        FileNotFoundError: Si falta modelo.zip o experto.json.
        ValueError: Si la configuración no es válida o el experto es de otro entorno.
    """
    dcfg = config_demostraciones(cfg)
    directorio = Path(directorio)
    modelo, metadatos_experto = cargar_experto(directorio)
    entorno = cfg["entorno"]["id"]
    entorno_experto = metadatos_experto.get("configuracion", {}).get("entorno", {}).get("id")
    if entorno_experto != entorno:
        raise ValueError(f"El experto se entrenó en {entorno_experto!r} y la configuración "
                         f"pide {entorno!r}")
    resumen = metadatos_experto["resumen"]
    politica = politica_trpo(modelo, dcfg["determinista"], resumen["normalizacion"])
    metadatos = {
        "entorno": entorno,
        "experto": {
            "directorio": ruta_registrada(directorio),
            "huella_modelo": huella_archivo(directorio / _ARCHIVO_MODELO),
            "huella_metadatos": huella_archivo(directorio / _ARCHIVO_METADATOS),
            "mejor_retorno": resumen["mejor_retorno"],
            "mejor_iteracion": resumen["mejor_iteracion"],
            "pasos": resumen["pasos"],
            "objetivo_alcanzado": resumen["objetivo_alcanzado"],
            "codigo": metadatos_experto.get("codigo"),
        },
        "procedencia": {
            "generador": "experto TRPO propio (sb3-contrib), sin filtrar partidas",
            "codigo": estado_git(),
            "entorno_ejecucion": versiones_y_hardware(),
            "configuracion_demostraciones": dcfg,
        },
        "configuracion_experto": metadatos_experto["configuracion"],
    }
    return politica, metadatos
