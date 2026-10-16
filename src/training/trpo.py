"""TRPO de sb3-contrib para el experto y como generador de GAIL.

- crear_trpo traduce la sección trpo de la configuración a sb3-contrib.
- iteracion_trpo da un bloque: una recogida de n_steps × n_envs pasos y una
  actualización, sin reiniciar el entorno entre bloques. Es lo que la rama gail
  repite intercalando la actualización del discriminador.
- entrenar_experto entrena con la recompensa real, evalúa cada cierto número de
  bloques y se queda con el mejor modelo.
- guardar_experto y cargar_experto guardan el modelo y sus metadatos.

Las semillas salen de src/utils/seed.py: nunca se pasa seed= a TRPO, porque sb3
volvería a sembrar los generadores globales y los entornos por su cuenta.
"""

from __future__ import annotations

import copy
import json
import math
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import sb3_contrib
import stable_baselines3
import torch
from gymnasium import spaces
from sb3_contrib import TRPO
from stable_baselines3.common.vec_env import VecEnv, VecNormalize

from src.data.entornos import crear_entorno, crear_entorno_vectorizado
from src.models.factory import ACTIVACIONES
from src.utils.seed import (fijar_seed, generador_torch_aislado, reiniciar_con_semilla,
                            semillas_independientes, validar_semilla)
from src.utils.tracking import registrar

_ARCHIVO_MODELO = "modelo.zip"
_ARCHIVO_METADATOS = "experto.json"


def _es_entero_positivo(valor: object) -> bool:
    """Indica si valor es un entero no booleano mayor que cero."""
    return isinstance(valor, int) and not isinstance(valor, bool) and valor > 0


def _es_real(valor: object) -> bool:
    """Indica si valor es un número real finito no booleano."""
    return (isinstance(valor, (int, float)) and not isinstance(valor, bool)
            and math.isfinite(valor))


def _config_trpo(cfg: dict) -> dict:
    """Devuelve la sección trpo comprobada.

    Raises:
        ValueError: Si falta alguna clave o algún valor no es válido.
    """
    tcfg = cfg.get("trpo") if isinstance(cfg, dict) else None
    if not isinstance(tcfg, dict):
        raise ValueError("La configuración necesita la sección trpo")
    reglas = {
        "capas_ocultas": lambda v: (isinstance(v, list) and len(v) > 0
                                    and all(_es_entero_positivo(c) for c in v)),
        "activacion": lambda v: isinstance(v, str) and v in ACTIVACIONES,
        "gamma": lambda v: _es_real(v) and 0 < v <= 1,
        "gae_lambda": lambda v: _es_real(v) and 0 <= v <= 1,
        "target_kl": lambda v: _es_real(v) and v > 0,
        "cg_damping": lambda v: _es_real(v) and v >= 0,
        "pasos_por_iteracion": _es_entero_positivo,
        "n_entornos": _es_entero_positivo,
        "normalizar_observaciones": lambda v: isinstance(v, bool),
        "limite_normalizacion": lambda v: _es_real(v) and v > 0,
        "normalizar_ventajas": lambda v: isinstance(v, bool),
        "pasos_gradiente_conjugado": _es_entero_positivo,
        "busqueda_lineal_factor": lambda v: _es_real(v) and 0 < v < 1,
        "busqueda_lineal_maximo": _es_entero_positivo,
        "learning_rate_valor": lambda v: _es_real(v) and v > 0,
        "pasos_valor": _es_entero_positivo,
        "batch_size_valor": _es_entero_positivo,
    }
    for clave, valida in reglas.items():
        if clave not in tcfg:
            raise ValueError(f"Falta trpo.{clave} en la configuración")
        if not valida(tcfg[clave]):
            raise ValueError(f"trpo.{clave} no es válido: {tcfg[clave]!r}")
    if tcfg["pasos_por_iteracion"] % tcfg["n_entornos"]:
        raise ValueError("trpo.pasos_por_iteracion debe ser divisible por trpo.n_entornos: "
                         f"{tcfg['pasos_por_iteracion']} entre {tcfg['n_entornos']}")
    if tcfg["pasos_por_iteracion"] % tcfg["batch_size_valor"]:
        raise ValueError("trpo.batch_size_valor debe dividir trpo.pasos_por_iteracion: "
                         f"{tcfg['batch_size_valor']} no divide {tcfg['pasos_por_iteracion']}")
    # Estandarizar las ventajas necesita al menos dos valores (sb3 lo exige).
    if tcfg["normalizar_ventajas"] and (tcfg["pasos_por_iteracion"] < 2
                                        or tcfg["batch_size_valor"] < 2):
        raise ValueError("Con trpo.normalizar_ventajas, trpo.pasos_por_iteracion y "
                         "trpo.batch_size_valor deben ser al menos 2")
    return tcfg


def _semilla_valida(valor: object) -> bool:
    """Indica si valor es una semilla válida según validar_semilla, la única comprobación."""
    try:
        validar_semilla(valor)
    except ValueError:
        return False
    return True


def _a_json(valor: object) -> object:
    """Convierte escalares de NumPy a tipos de Python para guardarlos en JSON.

    Raises:
        TypeError: Si el valor no se puede convertir, como hace json por defecto.
    """
    if isinstance(valor, (np.integer, np.floating, np.bool_)):
        return valor.item()
    raise TypeError(f"{type(valor).__name__} no se puede guardar en JSON")


def _config_experto(cfg: dict) -> dict:
    """Devuelve la sección experto comprobada.

    Raises:
        ValueError: Si falta alguna clave o algún valor no es válido.
    """
    ecfg = cfg.get("experto") if isinstance(cfg, dict) else None
    if not isinstance(ecfg, dict):
        raise ValueError("La configuración necesita la sección experto")
    reglas = {
        "pasos_totales": _es_entero_positivo,
        "retorno_objetivo": _es_real,
        "semilla": _semilla_valida,
        "evaluar_cada": _es_entero_positivo,
        "episodios_evaluacion": _es_entero_positivo,
        "evaluacion_determinista": lambda v: isinstance(v, bool),
    }
    for clave, valida in reglas.items():
        if clave not in ecfg:
            raise ValueError(f"Falta experto.{clave} en la configuración")
        if not valida(ecfg[clave]):
            raise ValueError(f"experto.{clave} no es válido: {ecfg[clave]!r}")
    return ecfg


def _comprobar_espacios(observacion: spaces.Space, accion: spaces.Space) -> None:
    """Admite observaciones Box 1-D y acciones Discrete con start 0 o Box 1-D."""
    if not isinstance(observacion, spaces.Box) or len(observacion.shape) != 1:
        raise ValueError("Espacio de observación no compatible: se admite Box de una dimensión")
    discreta = isinstance(accion, spaces.Discrete) and int(accion.start) == 0
    continua = isinstance(accion, spaces.Box) and len(accion.shape) == 1
    if not (discreta or continua):
        raise ValueError("Espacio de acción no compatible: se admite Discrete con start 0 "
                         "(sb3 genera índices 0..n-1) o Box de una dimensión")


def crear_trpo(cfg: dict, venv: VecEnv) -> TRPO:
    """Crea un agente TRPO de sb3-contrib con los hiperparámetros de la configuración.

    La política y la red de valor son perceptrones con trpo.capas_ocultas y
    trpo.activacion (2 × 100 tanh en el paper). Cada iteración recoge
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
    tcfg = _config_trpo(cfg)
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
                       "activation_fn": ACTIVACIONES[tcfg["activacion"]]},
        device="cpu",
        verbose=0,
    )


def iteracion_trpo(modelo: TRPO) -> dict[str, float]:
    """Da una iteración de TRPO: una recogida de n_steps × n_envs pasos y una actualización.

    Usa learn con reset_num_timesteps=False, así que el entorno no se reinicia
    entre iteraciones: las partidas a medias continúan en la siguiente. El
    retorno real sale de los Monitor del VecEnv, aunque un envoltorio cambie la
    recompensa que recibe el agente. Antes de la recogida se vacía el registro
    de partidas de sb3 (ep_info_buffer), que si no acumula las últimas 100: así
    las medidas son solo de las partidas que terminan en esta iteración.

    Returns:
        pasos acumulados, partidas terminadas en la iteración y su
        retorno_real_medio y longitud_media (nan si no terminó ninguna).
    """
    if not isinstance(modelo, TRPO):
        raise ValueError("modelo debe ser un TRPO de sb3-contrib")
    if modelo.ep_info_buffer is not None:
        modelo.ep_info_buffer.clear()
    modelo.learn(total_timesteps=modelo.n_steps * modelo.n_envs, reset_num_timesteps=False)
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


def normalizar_observacion(observacion: np.ndarray,
                           estadisticas: dict[str, Any] | None) -> np.ndarray:
    """Aplica la normalización de VecNormalize con unas estadísticas guardadas.

    Con None devuelve la observación sin cambios. La fórmula es la de sb3:
    recorte de (obs − media) / sqrt(varianza + epsilon) a ±limite.
    """
    if estadisticas is None:
        return observacion
    media = np.asarray(estadisticas["media"])
    escala = np.sqrt(np.asarray(estadisticas["varianza"]) + estadisticas["epsilon"])
    limite = estadisticas["limite"]
    return np.clip((observacion - media) / escala, -limite, limite).astype(np.float32)


def evaluar_retorno(modelo: TRPO, cfg: dict, episodios: int, semilla: int, determinista: bool,
                    estadisticas: dict[str, Any] | None = None) -> dict[str, Any]:
    """Juega episodios completos con el modelo y mide el retorno real.

    Evaluación mínima para elegir el experto; la de la rama
    evaluacion-checkpoints la sustituirá. Cada episodio usa su propia semilla,
    derivada de semilla con semillas_independientes, tanto para el entorno como
    para las acciones muestreadas (generador_torch_aislado). Así la misma semilla
    da siempre el mismo resultado, cada episodio no depende de cuántos haya y
    evaluar no altera el generador global, es decir, el entrenamiento.

    Args:
        modelo: Agente TRPO.
        cfg: Configuración resuelta con entorno.id.
        episodios: Número de episodios, entero positivo.
        semilla: Semilla base no negativa de la evaluación.
        determinista: True para la acción más probable, False para muestrear.
        estadisticas: De estadisticas_normalizacion si el modelo normaliza.

    Returns:
        retorno_medio, retorno_desviacion, longitud_media y retornos (uno por
        episodio, en el orden de las semillas).

    Raises:
        ValueError: Si episodios o semilla no son válidos.
    """
    if not _es_entero_positivo(episodios):
        raise ValueError(f"episodios debe ser un entero positivo, no {episodios!r}")
    semillas = semillas_independientes(validar_semilla(semilla), episodios)
    entorno = crear_entorno(cfg, None)
    retornos, longitudes = [], []
    try:
        for semilla_episodio in semillas:
            with generador_torch_aislado(semilla_episodio):
                observacion, _ = reiniciar_con_semilla(entorno, semilla_episodio)
                retorno, longitud, terminado = 0.0, 0, False
                while not terminado:
                    accion, _ = modelo.predict(normalizar_observacion(observacion, estadisticas),
                                               deterministic=determinista)
                    observacion, recompensa, fin, corte, _ = entorno.step(accion)
                    retorno += float(recompensa)
                    longitud += 1
                    terminado = fin or corte
            retornos.append(retorno)
            longitudes.append(longitud)
    finally:
        entorno.close()
    return {"retorno_medio": float(np.mean(retornos)),
            "retorno_desviacion": float(np.std(retornos)),
            "longitud_media": float(np.mean(longitudes)),
            "retornos": retornos}


def estado_git() -> dict[str, Any]:
    """Devuelve el commit del repositorio y si había cambios sin commit.

    Sirve para saber con qué código se generó un resultado. Si hay cambios sin
    commit (incluidos archivos nuevos no ignorados), el resultado no se puede
    reproducir exactamente desde el repositorio. Si git no está disponible o
    esto no es un repositorio (por ejemplo, una copia sin .git), ambos valores
    son None en vez de fallar.

    Returns:
        {"commit": hash completo o None, "cambios_sin_commit": bool o None}.
    """
    raiz = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=raiz, capture_output=True,
                                text=True, check=True, timeout=10).stdout.strip()
        cambios = subprocess.run(["git", "status", "--porcelain"], cwd=raiz,
                                 capture_output=True, text=True, check=True,
                                 timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "cambios_sin_commit": None}
    return {"commit": commit, "cambios_sin_commit": bool(cambios)}


def _versiones_y_hardware() -> dict[str, Any]:
    """Versiones de las librerías y descripción del hardware, para los metadatos."""
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "stable_baselines3": stable_baselines3.__version__,
        "sb3_contrib": sb3_contrib.__version__,
        "gymnasium": gym.__version__,
        "plataforma": platform.platform(),
        "procesador": platform.processor() or platform.machine(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def entrenar_experto(cfg: dict, run: Any = None) -> tuple[TRPO, dict[str, Any]]:
    """Entrena el experto TRPO con la recompensa real del entorno.

    De experto.semilla salen tres semillas independientes: la del modelo
    (fijar_seed), la de los entornos de entrenamiento y la de evaluación. Se
    entrena por iteraciones hasta experto.pasos_totales; cada
    experto.evaluar_cada iteraciones, y al final, se evalúa con
    experto.episodios_evaluacion episodios (política estocástica si
    experto.evaluacion_determinista es false, la que graba las demostraciones).
    Se conserva el mejor modelo y se para en cuanto el retorno medio alcanza
    experto.retorno_objetivo. Al terminar, el modelo tiene los pesos del mejor.

    Args:
        cfg: Configuración resuelta con las secciones entorno, trpo y experto.
        run: Run de W&B o None; recibe las medidas de cada iteración.

    Returns:
        El modelo y un resumen: pasos, iteraciones, mejor retorno y su
        iteración, si se alcanzó el objetivo, historial de evaluaciones,
        segundos medidos, semillas, estadísticas de normalización y entorno.

    Raises:
        ValueError: Si la configuración no es válida o experto.pasos_totales no es
            múltiplo de trpo.pasos_por_iteracion.
    """
    tcfg, ecfg = _config_trpo(cfg), _config_experto(cfg)
    if ecfg["pasos_totales"] % tcfg["pasos_por_iteracion"]:
        raise ValueError("experto.pasos_totales debe ser múltiplo de trpo.pasos_por_iteracion "
                         f"({ecfg['pasos_totales']} entre {tcfg['pasos_por_iteracion']}): cada "
                         "iteración recoge un bloque completo y se pasaría del presupuesto")
    semilla_base = validar_semilla(ecfg["semilla"])
    semilla_modelo, semilla_entornos, semilla_evaluacion = semillas_independientes(
        semilla_base, 3)
    fijar_seed(semilla_modelo)
    venv: VecEnv = crear_entorno_vectorizado(cfg, tcfg["n_entornos"], semilla_entornos)
    if tcfg["normalizar_observaciones"]:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=False,
                            clip_obs=float(tcfg["limite_normalizacion"]))
    inicio = time.perf_counter()
    try:
        modelo = crear_trpo(cfg, venv)
        mejor: dict[str, Any] | None = None
        historial = []
        iteracion, alcanzado = 0, False
        while modelo.num_timesteps < ecfg["pasos_totales"]:
            iteracion += 1
            medidas = iteracion_trpo(modelo)
            registrar(run, medidas, paso=modelo.num_timesteps)
            if (iteracion % ecfg["evaluar_cada"]
                    and modelo.num_timesteps < ecfg["pasos_totales"]):
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
        "semillas": {"base": semilla_base, "modelo": semilla_modelo,
                     "entornos": semilla_entornos, "evaluacion": semilla_evaluacion},
        "normalizacion": mejor["estadisticas"],
    }
    return modelo, resumen


def comprobar_directorio_libre(directorio: Path) -> Path:
    """Comprueba que se puede guardar en directorio sin sobrescribir nada.

    Vale si no existe o si es una carpeta vacía. Conviene llamarla antes de
    entrenar, para no perder un entrenamiento largo al guardar.

    Returns:
        El directorio como Path.

    Raises:
        FileExistsError: Si existe como archivo o es una carpeta con archivos.
    """
    directorio = Path(directorio)
    if directorio.exists() and (not directorio.is_dir() or any(directorio.iterdir())):
        raise FileExistsError(f"{directorio} ya existe con contenido; no se sobrescribe")
    return directorio


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
                 "entorno_ejecucion": _versiones_y_hardware()}
    # Se serializa antes de crear nada: un NaN no deja una carpeta a medias.
    texto = json.dumps(metadatos, ensure_ascii=False, indent=2, allow_nan=False,
                       default=_a_json)
    directorio.mkdir(parents=True, exist_ok=True)
    with (directorio / _ARCHIVO_METADATOS).open("x", encoding="utf-8") as archivo:
        archivo.write(texto)
    modelo.save(directorio / _ARCHIVO_MODELO)


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
