"""Reglas de las secciones de entrenamiento de la configuración, sin PyTorch ni sb3.

Son las comprobaciones que hacen los entrenadores (trpo.py, bc.py, gail.py y la
arquitectura de factory.py) antes de crear nada. Viven aquí para que la API las
aplique al pedir un entrenamiento, antes de encolarlo, sin importar src/training:
así un ajuste que el entrenador rechazaría da 422 al momento y no un trabajo que
falla al salir de la cola.

Los nombres de las activaciones y de los optimizadores de BC tienen que coincidir
con las clases de ``ACTIVACIONES`` (factory.py) y ``OPTIMIZADORES`` (bc.py); un
test lo comprueba.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from src.utils.validacion import es_entero_positivo, es_real_finito, es_semilla

ACTIVACIONES = ("tanh", "relu")
OPTIMIZADORES_BC = ("adam", "sgd", "rmsprop")
OPTIMIZADORES_DISCRIMINADOR = ("adam",)
# −log(1 − D) da recompensas >= 0: favorece alargar las partidas, para entornos
# que terminan al fallar (CartPole). log D da recompensas <= 0: favorece
# acabarlas, para entornos que terminan al tener éxito (MountainCar, Acrobot).
# En el original, la segunda es favor_zero_expert_reward (cuts_off_on_success).
MODOS_RECOMPENSA = ("menos_log_uno_menos_d", "log_d")

# Métodos que se entrenan sueltos desde el dashboard.
METODOS = ("experto", "bc", "gail")


def comprobar_arquitectura(capas_ocultas: object, activacion: object) -> None:
    """Comprueba la lista de capas ocultas y el nombre de la activación.

    Es la comprobación que aplican PoliticaBC y Discriminador al crearse; sirve
    también para validar una configuración antes de crear ninguna red.

    Raises:
        ValueError: Si capas_ocultas no es una lista de enteros positivos (puede
            estar vacía) o activacion no es una de ACTIVACIONES.
    """
    if (isinstance(capas_ocultas, (str, bytes)) or not isinstance(capas_ocultas, Sequence)
            or not all(es_entero_positivo(c) for c in capas_ocultas)):
        raise ValueError(
            f"capas_ocultas debe ser una lista de enteros positivos, no {capas_ocultas!r}")
    if not isinstance(activacion, str) or activacion not in ACTIVACIONES:
        raise ValueError(
            f"activacion desconocida: {activacion!r}; opciones: {sorted(ACTIVACIONES)}")


def _seccion(cfg: object, nombre: str) -> dict:
    """La sección de cfg o ValueError si falta."""
    seccion = cfg.get(nombre) if isinstance(cfg, dict) else None
    if not isinstance(seccion, dict):
        raise ValueError(f"La configuración necesita la sección {nombre}")
    return seccion


def _aplicar_reglas(seccion: dict, nombre: str, reglas: dict[str, Any]) -> None:
    """Comprueba que cada clave de reglas esté en la sección y valga."""
    for clave, valida in reglas.items():
        if clave not in seccion:
            raise ValueError(f"Falta {nombre}.{clave} en la configuración")
        if not valida(seccion[clave]):
            raise ValueError(f"{nombre}.{clave} no es válido: {seccion[clave]!r}")


def config_trpo(cfg: dict) -> dict:
    """Devuelve la sección trpo comprobada.

    Raises:
        ValueError: Si falta alguna clave o algún valor no es válido.
    """
    tcfg = _seccion(cfg, "trpo")
    _aplicar_reglas(tcfg, "trpo", {
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
        "log_std_inicial": es_real_finito,
        "inicializacion_ortogonal": lambda v: isinstance(v, bool),
        "pasos_gradiente_conjugado": es_entero_positivo,
        "busqueda_lineal_factor": lambda v: es_real_finito(v) and 0 < v < 1,
        "busqueda_lineal_maximo": es_entero_positivo,
        "learning_rate_valor": lambda v: es_real_finito(v) and v > 0,
        "pasos_valor": es_entero_positivo,
        "batch_size_valor": es_entero_positivo,
    })
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


def config_trpo_experto(cfg: dict) -> dict:
    """Devuelve una copia de cfg con trpo cambiado por experto.trpo, la del experto.

    experto.trpo cambia solo las claves que pone; vacía o ausente, el experto usa la
    sección trpo, como GAIL. La sección resultante se comprueba con config_trpo.

    Args:
        cfg: Configuración resuelta.

    Returns:
        Configuración nueva; cfg no se modifica.

    Raises:
        ValueError: Si experto.trpo no es un diccionario o la sección resultante no
            es válida.
    """
    experto = cfg.get("experto") if isinstance(cfg, dict) else None
    cambios = experto.get("trpo", {}) if isinstance(experto, dict) else {}
    if not isinstance(cambios, dict):
        raise ValueError("experto.trpo debe ser un diccionario")
    tcfg = _seccion(cfg, "trpo")
    combinada = {**cfg, "trpo": {**tcfg, **cambios}}
    config_trpo(combinada)
    return combinada


def config_experto(cfg: dict) -> dict:
    """Devuelve la sección experto comprobada.

    Raises:
        ValueError: Si falta alguna clave o algún valor no es válido.
    """
    ecfg = _seccion(cfg, "experto")
    _aplicar_reglas(ecfg, "experto", {
        "pasos_totales": es_entero_positivo,
        "retorno_objetivo": es_real_finito,
        "semilla": es_semilla,
        "evaluar_cada": es_entero_positivo,
        "episodios_evaluacion": es_entero_positivo,
        "evaluacion_determinista": lambda v: isinstance(v, bool),
    })
    return ecfg


def _entero_positivo_de_cfg(bcfg: dict, clave: str) -> int:
    """Lee bc.clave y comprueba que es un entero positivo."""
    valor = bcfg.get(clave)
    if not es_entero_positivo(valor):
        raise ValueError(f"bc.{clave} debe ser un entero positivo, no {valor!r}")
    return int(valor)


def config_bc(cfg: dict) -> dict:
    """Devuelve la sección bc comprobada: lo que necesita entrenar_bc.

    Raises:
        ValueError: Si falta la sección o epocas_maximas, paciencia,
            optimizador, learning_rate, normalizar_observaciones o (si se normaliza)
            epsilon_normalizacion no son válidos.
    """
    bcfg = _seccion(cfg, "bc")
    _entero_positivo_de_cfg(bcfg, "epocas_maximas")
    _entero_positivo_de_cfg(bcfg, "paciencia")
    optimizador = bcfg.get("optimizador")
    if not isinstance(optimizador, str) or optimizador not in OPTIMIZADORES_BC:
        raise ValueError(f"bc.optimizador desconocido: {optimizador!r}; "
                         f"opciones: {sorted(OPTIMIZADORES_BC)}")
    learning_rate = bcfg.get("learning_rate")
    if not es_real_finito(learning_rate) or learning_rate <= 0:
        raise ValueError(f"bc.learning_rate debe ser un real positivo, no {learning_rate!r}")
    normalizar = bcfg.get("normalizar_observaciones")
    if not isinstance(normalizar, (bool, np.bool_)):
        raise ValueError(f"bc.normalizar_observaciones debe ser booleano, no {normalizar!r}")
    # La misma regla que aplica PoliticaBC al ajustar la normalización con train.
    epsilon = bcfg.get("epsilon_normalizacion")
    if normalizar and (not es_real_finito(epsilon) or epsilon <= 0):
        raise ValueError("bc.epsilon_normalizacion debe ser un real positivo si se normaliza, "
                         f"no {epsilon!r}")
    return bcfg


def config_gail(cfg: dict) -> dict:
    """Devuelve la sección gail comprobada (sin el discriminador, que se comprueba aparte).

    Raises:
        ValueError: Si falta alguna clave o algún valor no es válido.
    """
    gcfg = _seccion(cfg, "gail")
    _aplicar_reglas(gcfg, "gail", {
        "iteraciones": es_entero_positivo,
        "evaluar_cada": es_entero_positivo,
        "episodios_evaluacion": es_entero_positivo,
        "evaluacion_determinista": lambda v: isinstance(v, bool),
        "recompensa": lambda v: v in MODOS_RECOMPENSA,
    })
    return gcfg


def config_discriminador(cfg: dict) -> dict:
    """Devuelve gail.discriminador comprobado.

    Raises:
        ValueError: Si falta alguna clave o algún valor no es válido.
    """
    gcfg = cfg.get("gail") if isinstance(cfg, dict) else None
    dcfg = gcfg.get("discriminador") if isinstance(gcfg, dict) else None
    claves = {"capas_ocultas", "activacion", "optimizador", "learning_rate", "pasos_adam",
              "peso_entropia", "normalizar_entrada", "epsilon_normalizacion"}
    if not isinstance(dcfg, dict) or not claves <= dcfg.keys():
        raise ValueError("La configuración necesita gail.discriminador con "
                         + ", ".join(sorted(claves)))
    # La misma comprobación que al crear la red: así se rechaza antes de entrenar.
    try:
        comprobar_arquitectura(dcfg["capas_ocultas"], dcfg["activacion"])
    except ValueError as error:
        raise ValueError(f"gail.discriminador no es válido: {error}") from error

    def real(clave: str) -> float:
        """Devuelve dcfg[clave] si es un real finito no booleano."""
        valor = dcfg[clave]
        if not es_real_finito(valor):
            raise ValueError(f"gail.discriminador.{clave} debe ser un real finito, no {valor!r}")
        return float(valor)

    if dcfg["optimizador"] not in OPTIMIZADORES_DISCRIMINADOR:
        raise ValueError(f"gail.discriminador.optimizador desconocido: {dcfg['optimizador']!r}; "
                         f"opciones: {list(OPTIMIZADORES_DISCRIMINADOR)}")
    if real("learning_rate") <= 0:
        raise ValueError("gail.discriminador.learning_rate debe ser positivo")
    pasos = dcfg["pasos_adam"]
    if not es_entero_positivo(pasos):
        raise ValueError(
            f"gail.discriminador.pasos_adam debe ser un entero positivo, no {pasos!r}")
    if real("peso_entropia") < 0:
        raise ValueError("gail.discriminador.peso_entropia no puede ser negativo")
    if not isinstance(dcfg["normalizar_entrada"], bool):
        raise ValueError("gail.discriminador.normalizar_entrada debe ser un booleano")
    if real("epsilon_normalizacion") <= 0:
        raise ValueError("gail.discriminador.epsilon_normalizacion debe ser positivo")
    return dcfg


def comprobar_frecuencia_pares(cfg: dict) -> None:
    """Comprueba demostraciones.frecuencia_pares, que usan BC y GAIL.

    Raises:
        ValueError: Si no es un entero positivo.
    """
    frecuencia = _seccion(cfg, "demostraciones").get("frecuencia_pares")
    if not es_entero_positivo(frecuencia):
        raise ValueError("demostraciones.frecuencia_pares debe ser un entero positivo, no "
                         f"{frecuencia!r}")


def comprobar_config_metodo(cfg: dict, metodo: str) -> None:
    """Comprueba las secciones que usa un entrenamiento suelto de ese método.

    - experto: experto y la sección trpo con los cambios de experto.trpo.
    - bc: bc (con su arquitectura y batch_size) y demostraciones.frecuencia_pares.
    - gail: gail, gail.discriminador, trpo y demostraciones.frecuencia_pares.

    Args:
        cfg: Configuración resuelta.
        metodo: Uno de METODOS.

    Raises:
        ValueError: Si el método no existe o alguna sección no es válida.
    """
    if metodo == "experto":
        config_experto(cfg)
        config_trpo_experto(cfg)
    elif metodo == "bc":
        bcfg = config_bc(cfg)
        comprobar_arquitectura(bcfg.get("capas_ocultas"), bcfg.get("activacion"))
        if not es_entero_positivo(bcfg.get("batch_size")):
            raise ValueError("bc.batch_size debe ser un entero positivo, no "
                             f"{bcfg.get('batch_size')!r}")
        comprobar_frecuencia_pares(cfg)
    elif metodo == "gail":
        config_gail(cfg)
        config_discriminador(cfg)
        config_trpo(cfg)
        comprobar_frecuencia_pares(cfg)
    else:
        raise ValueError(f"Método desconocido {metodo!r}; opciones: {', '.join(METODOS)}")
