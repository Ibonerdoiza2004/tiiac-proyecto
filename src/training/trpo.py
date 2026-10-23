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
import json
import shutil
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import spaces
from sb3_contrib import TRPO
from stable_baselines3.common.vec_env import VecEnv, VecNormalize

from src.data.demostraciones import Politica, comprobar_espacios, config_demostraciones
from src.data.entornos import crear_entorno_vectorizado
from src.data.splits import huella_archivo
from src.models.factory import ACTIVACIONES
from src.training.evaluacion import evaluar_politica, normalizar_observacion
from src.utils.procedencia import a_json, estado_git, ruta_registrada, versiones_y_hardware
from src.utils.seed import fijar_seed, semillas_independientes, validar_semilla
from src.utils.tracking import registrar
from src.utils.validacion import es_entero_positivo, es_real_finito

_ARCHIVO_MODELO = "modelo.zip"
_ARCHIVO_METADATOS = "experto.json"


def config_trpo(cfg: dict) -> dict:
    """Devuelve la sección trpo comprobada.

    Raises:
        ValueError: Si falta alguna clave o algún valor no es válido.
    """
    tcfg = cfg.get("trpo") if isinstance(cfg, dict) else None
    if not isinstance(tcfg, dict):
        raise ValueError("La configuración necesita la sección trpo")
    reglas = {
        "capas_ocultas": lambda v: (isinstance(v, list) and len(v) > 0
                                    and all(es_entero_positivo(c) for c in v)),
        "activacion": lambda v: isinstance(v, str) and v in ACTIVACIONES,
        "gamma": lambda v: es_real_finito(v) and 0 < v <= 1,
        "gae_lambda": lambda v: es_real_finito(v) and 0 <= v <= 1,
        "target_kl": lambda v: es_real_finito(v) and v > 0,
        "cg_damping": lambda v: es_real_finito(v) and v >= 0,
        "pasos_por_iteracion": es_entero_positivo,
        "n_entornos": es_entero_positivo,
        "normalizar_observaciones": lambda v: isinstance(v, bool),
        "limite_normalizacion": lambda v: es_real_finito(v) and v > 0,
        "normalizar_ventajas": lambda v: isinstance(v, bool),
        "pasos_gradiente_conjugado": es_entero_positivo,
        "busqueda_lineal_factor": lambda v: es_real_finito(v) and 0 < v < 1,
        "busqueda_lineal_maximo": es_entero_positivo,
        "learning_rate_valor": lambda v: es_real_finito(v) and v > 0,
        "pasos_valor": es_entero_positivo,
        "batch_size_valor": es_entero_positivo,
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


def _config_experto(cfg: dict) -> dict:
    """Devuelve la sección experto comprobada.

    Raises:
        ValueError: Si falta alguna clave o algún valor no es válido.
    """
    ecfg = cfg.get("experto") if isinstance(cfg, dict) else None
    if not isinstance(ecfg, dict):
        raise ValueError("La configuración necesita la sección experto")
    reglas = {
        "pasos_totales": es_entero_positivo,
        "retorno_objetivo": es_real_finito,
        "semilla": _semilla_valida,
        "evaluar_cada": es_entero_positivo,
        "episodios_evaluacion": es_entero_positivo,
        "evaluacion_determinista": lambda v: isinstance(v, bool),
    }
    for clave, valida in reglas.items():
        if clave not in ecfg:
            raise ValueError(f"Falta experto.{clave} en la configuración")
        if not valida(ecfg[clave]):
            raise ValueError(f"experto.{clave} no es válido: {ecfg[clave]!r}")
    return ecfg


def _comprobar_espacios(observacion: spaces.Space, accion: spaces.Space) -> None:
    """Admite los espacios de comprobar_espacios, con Discrete solo si empieza en 0."""
    comprobar_espacios(observacion, accion)
    if isinstance(accion, spaces.Discrete) and int(accion.start) != 0:
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
    tcfg, ecfg = config_trpo(cfg), _config_experto(cfg)
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


def escribir_carpeta_nueva(directorio: Path, escribir: Callable[[Path], None]) -> None:
    """Escribe una carpeta de resultados entera o nada: nunca la deja a medias.

    escribir(temporal) crea los archivos en una carpeta temporal junto a
    directorio, que después se renombra a directorio (en el mismo sistema de
    archivos, el cambio de nombre es inmediato). Si escribir falla, también por
    Ctrl+C, la temporal se borra y directorio queda como estaba, así que se puede
    volver a guardar ahí. Comprueba antes, como comprobar_directorio_libre, que
    directorio no existe o está vacía.

    Raises:
        FileExistsError: Si directorio existe con contenido, antes o al terminar.
        Lo que lance escribir.
    """
    directorio = comprobar_directorio_libre(directorio)
    directorio.parent.mkdir(parents=True, exist_ok=True)
    temporal = Path(tempfile.mkdtemp(prefix=f".{directorio.name}.", suffix=".tmp",
                                     dir=directorio.parent))
    try:
        escribir(temporal)
        # Una carpeta vacía ya creada se sustituye; con contenido, no.
        comprobar_directorio_libre(directorio)
        if directorio.exists():
            directorio.rmdir()
        temporal.rename(directorio)
    except BaseException:
        shutil.rmtree(temporal, ignore_errors=True)
        raise


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

    # Si falla al escribir (disco lleno, Ctrl+C), no queda una carpeta a medias.
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
