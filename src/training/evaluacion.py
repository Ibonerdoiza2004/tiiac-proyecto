"""Predicción, evaluación y checkpoints de las políticas del experimento.

Para evaluar agentes guardados: experto con ``cargar_experto(dir)`` y sus
estadísticas en ``metadatos["resumen"]["normalizacion"]``; BC con
``cargar_politica_bc(ruta)`` y sin estadísticas; agente aleatorio con
``"aleatorio"``. GAIL usa ``cargar_gail`` y el mismo campo de estadísticas
que el experto.
"""

from __future__ import annotations

import contextlib
import json
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from gymnasium import spaces
from sb3_contrib import TRPO

from src.data.demostraciones import jugar_partida, retorno_partida
from src.data.entornos import crear_entorno
from src.data.splits import comprobar_huella
from src.models.factory import PoliticaBC
from src.utils.procedencia import a_json, estado_git, versiones_y_hardware
from src.utils.seed import generador_torch_aislado, semillas_independientes, validar_semilla
from src.utils.validacion import es_entero_positivo, es_real_finito

_FORMATO_BC = "tiiac-politica-bc"
_VERSION_BC = 1
_CLAVES_ESTADISTICAS = ("media", "varianza", "epsilon", "limite")


def config_evaluacion(cfg: dict[str, Any]) -> dict[str, Any]:
    """Devuelve la sección evaluacion comprobada.

    Raises:
        ValueError: Si falta o no es válido episodios (entero > 0), semilla
            (validar_semilla) o determinista (booleano).
    """
    ecfg = cfg.get("evaluacion") if isinstance(cfg, dict) else None
    if not isinstance(ecfg, dict):
        raise ValueError("La configuración necesita la sección evaluacion")
    episodios = ecfg.get("episodios")
    if not es_entero_positivo(episodios):
        raise ValueError("evaluacion.episodios debe ser un entero positivo")
    if "semilla" not in ecfg:
        raise ValueError("Falta evaluacion.semilla en la configuración")
    semilla = validar_semilla(ecfg["semilla"])
    determinista = ecfg.get("determinista")
    if not isinstance(determinista, bool):
        raise ValueError("evaluacion.determinista debe ser booleano")
    return {"episodios": int(episodios), "semilla": semilla, "determinista": determinista}


def evaluar_final(
    agente: PoliticaBC | TRPO | str,
    cfg: dict[str, Any],
    estadisticas: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evalúa usando episodios, semilla y modo de la sección evaluacion.

    Raises:
        ValueError: Si la sección evaluacion o algún argumento no son válidos.
    """
    ecfg = config_evaluacion(cfg)
    return evaluar_politica(agente, cfg, ecfg["episodios"], ecfg["semilla"],
                            ecfg["determinista"], estadisticas)


def normalizar_observacion(observacion: np.ndarray,
                           estadisticas: dict[str, Any] | None) -> np.ndarray:
    """Aplica la normalización de VecNormalize con unas estadísticas guardadas.

    Con None devuelve la observación sin cambios. La fórmula es la de sb3:
    recorte de (obs − media) / sqrt(varianza + epsilon) a ±limite. Admite una
    observación [d] o un lote [N, d].

    Args:
        observacion: Observación en bruto, como la da el entorno.
        estadisticas: De estadisticas_normalizacion (media, varianza, epsilon y
            limite), o None si el modelo no normaliza.

    Raises:
        ValueError: Si a las estadísticas les falta alguna clave, media o
            varianza no son vectores finitos de la dimensión de la observación,
            la varianza es negativa, o epsilon o limite no son reales positivos.
    """
    if estadisticas is None:
        return observacion
    if not isinstance(estadisticas, dict) or not set(_CLAVES_ESTADISTICAS) <= estadisticas.keys():
        raise ValueError(f"estadisticas necesita {list(_CLAVES_ESTADISTICAS)}")
    observacion = np.asarray(observacion)
    media = np.asarray(estadisticas["media"], dtype=np.float64)
    varianza = np.asarray(estadisticas["varianza"], dtype=np.float64)
    if (media.ndim != 1 or varianza.shape != media.shape
            or observacion.shape[-1:] != media.shape):
        raise ValueError("media y varianza deben ser vectores de la dimensión de la observación")
    if not (np.isfinite(media).all() and np.isfinite(varianza).all()) or (varianza < 0).any():
        raise ValueError("media y varianza deben ser finitas y la varianza no negativa")
    epsilon, limite = estadisticas["epsilon"], estadisticas["limite"]
    if not (es_real_finito(epsilon) and epsilon > 0 and es_real_finito(limite) and limite > 0):
        raise ValueError("epsilon y limite deben ser reales positivos finitos")
    escala = np.sqrt(varianza + epsilon)
    return np.clip((observacion - media) / escala, -limite, limite).astype(np.float32)


def predecir_accion(
    agente: PoliticaBC | TRPO | str,
    observacion: np.ndarray,
    espacio_accion: spaces.Space,
    determinista: bool,
    semilla: int | None = None,
    estadisticas: dict[str, Any] | None = None,
) -> int | np.ndarray:
    """Predice una acción válida para BC, TRPO o el agente aleatorio.

    ``semilla`` controla el muestreo estocástico de BC cuando se indica; con
    None usa el generador de PyTorch vigente, que el llamador puede aislar por
    episodio (evaluar_politica lo hace). Las acciones aleatorias usan el
    espacio sembrado por ``seed.py``. Solo TRPO admite estadísticas: las
    guardadas por ``VecNormalize`` (normalizar_observacion).

    Raises:
        ValueError: Si el agente, la observación, el espacio o las estadísticas
            no son compatibles; por ejemplo, estadísticas para BC o el agente
            aleatorio, que no normalizan así.
    """
    if semilla is not None:
        semilla = validar_semilla(semilla)
    if not isinstance(determinista, bool):
        raise ValueError("determinista debe ser booleano")
    observacion = np.asarray(observacion)
    if not np.issubdtype(observacion.dtype, np.number) or not np.isfinite(observacion).all():
        raise ValueError("observacion debe contener valores numéricos finitos")
    if not isinstance(espacio_accion, (spaces.Discrete, spaces.Box)):
        raise ValueError("espacio_accion debe ser Discrete o Box")
    if estadisticas is not None and not isinstance(agente, TRPO):
        # BC lleva su normalización en la red; ignorarlas ocultaría un error.
        raise ValueError("Solo TRPO admite estadisticas de normalización")

    if isinstance(agente, str) and agente == "aleatorio":
        accion = espacio_accion.sample()
    elif isinstance(agente, PoliticaBC):
        forma = tuple(espacio_accion.shape) if isinstance(espacio_accion, spaces.Box) else ()
        if isinstance(espacio_accion, spaces.Discrete):
            if agente.continua or agente.dim_accion != espacio_accion.n:
                raise ValueError("La política BC no coincide con el espacio Discrete")
        elif (len(forma) != 1 or not agente.continua
              or agente.dim_accion != espacio_accion.shape[0]):
            raise ValueError("La política BC no coincide con el espacio Box")
        if observacion.shape != (agente.dim_observacion,):
            raise ValueError("observacion no coincide con la dimensión de PoliticaBC")
        agente.eval()
        entrada = torch.as_tensor(observacion, dtype=torch.float32).reshape(1, -1)
        aislado = (generador_torch_aislado(semilla) if semilla is not None
                   else contextlib.nullcontext())
        with torch.no_grad(), aislado:
            if agente.continua:
                salida = (agente(entrada)[0] if determinista
                          else agente.distribucion(entrada).sample()[0])
                accion = salida.cpu().numpy()
            else:
                indice = (int(agente(entrada).argmax(dim=1).item()) if determinista else
                          int(agente.distribucion(entrada).sample().item()))
                accion = indice + int(espacio_accion.start)
    elif isinstance(agente, TRPO):
        obs = normalizar_observacion(observacion, estadisticas)
        accion, _ = agente.predict(obs, deterministic=determinista)
        if isinstance(espacio_accion, spaces.Discrete):
            accion = int(np.asarray(accion).reshape(-1)[0])
        else:
            accion = np.asarray(accion, dtype=np.float32).reshape(espacio_accion.shape)
    else:
        raise ValueError("agente debe ser PoliticaBC, TRPO o 'aleatorio'")

    if isinstance(espacio_accion, spaces.Discrete):
        accion = int(accion)
        if not espacio_accion.contains(accion):
            raise ValueError(f"La acción {accion!r} queda fuera de {espacio_accion}")
        return accion
    accion = np.asarray(accion, dtype=np.float32).reshape(espacio_accion.shape)
    accion = np.clip(accion, espacio_accion.low, espacio_accion.high).astype(np.float32)
    if not espacio_accion.contains(accion):
        raise ValueError("La acción predicha queda fuera del espacio Box")
    return accion


def evaluar_politica(
    agente: PoliticaBC | TRPO | str,
    cfg: dict[str, Any],
    episodios: int,
    semilla: int,
    determinista: bool = True,
    estadisticas: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evalúa episodios completos con las mismas semillas para cada agente.

    Cada episodio es una partida de jugar_partida (la misma que graba las
    demostraciones) con su propia semilla, derivada de ``semilla`` mediante
    ``semillas_independientes``, para el entorno, el espacio de acciones y el
    muestreo de PyTorch, que jugar_partida aísla por episodio. Por ello los
    resultados son pareados y repetibles, cada episodio no depende de cuántos
    haya y evaluar no altera el estado aleatorio global. La recompensa es la
    del entorno y la evaluación no crea gradientes.

    Returns:
        Retorno medio, desviación poblacional, retornos y longitudes por episodio.

    Raises:
        ValueError: Si episodios, semilla, modo, estadísticas o configuración no
            son válidos, o el agente no es compatible con el entorno.
    """
    if not es_entero_positivo(episodios):
        raise ValueError("episodios debe ser un entero positivo")
    semillas = semillas_independientes(validar_semilla(semilla), int(episodios))
    if not isinstance(determinista, bool):
        raise ValueError("determinista debe ser booleano")
    if estadisticas is not None and not isinstance(agente, TRPO):
        raise ValueError("Solo TRPO admite estadisticas de normalización")
    env = crear_entorno(cfg, None)
    if isinstance(agente, PoliticaBC):
        agente.eval()

    def politica(observacion: np.ndarray) -> int | np.ndarray:
        """Acción del agente con el protocolo común (predecir_accion)."""
        return predecir_accion(agente, observacion, env.action_space, determinista, None,
                               estadisticas)

    try:
        with torch.no_grad():
            partidas = [jugar_partida(env, politica, semilla_episodio, indice)
                        for indice, semilla_episodio in enumerate(semillas)]
    finally:
        env.close()
    retornos = [retorno_partida(partida) for partida in partidas]
    longitudes = [partida.numero_pasos for partida in partidas]
    return {
        "retorno_medio": float(np.mean(retornos)),
        "retorno_desviacion": float(np.std(retornos)),
        "retornos": retornos,
        "longitudes": longitudes,
        "longitud_media": float(np.mean(longitudes)),
    }


def _metadatos_checkpoint(cfg: dict[str, Any], metadatos: dict[str, Any]) -> dict[str, Any]:
    """Completa, comprueba y serializa metadatos de procedencia para un checkpoint.

    La semilla pasa por validar_semilla, el subconjunto es un entero positivo y
    la huella es un SHA-256 (comprobar_huella). Los escalares de NumPy se
    guardan como números de Python; nan e infinitos se rechazan.
    """
    requeridos = {"semilla", "subconjunto", "huella_dataset"}
    if not isinstance(metadatos, dict) or not requeridos <= metadatos.keys():
        raise ValueError("metadatos necesita semilla, subconjunto y huella_dataset")
    semilla = validar_semilla(metadatos["semilla"])
    if not es_entero_positivo(metadatos["subconjunto"]):
        raise ValueError("metadatos.subconjunto debe ser un entero positivo (partidas de train), "
                         f"no {metadatos['subconjunto']!r}")
    comprobar_huella(metadatos["huella_dataset"])
    resultado = {
        "configuracion": cfg,
        "entorno": cfg.get("entorno", {}).get("id"),
        "semilla": semilla,
        "subconjunto": int(metadatos["subconjunto"]),
        "huella_dataset": metadatos["huella_dataset"],
        "codigo": estado_git(),
        "versiones": versiones_y_hardware(),
        "adicional": metadatos.get("adicional", {}),
    }
    try:
        return json.loads(json.dumps(resultado, allow_nan=False, default=a_json))
    except (TypeError, ValueError) as error:
        raise ValueError(f"metadatos no son JSON finito: {error}") from error


def guardar_politica_bc(
    ruta: Path,
    politica: PoliticaBC,
    cfg: dict[str, Any],
    metadatos: dict[str, Any],
) -> None:
    """Guarda BC sin sobrescribir, incluyendo arquitectura, buffers y procedencia.

    La arquitectura (capas ocultas y activación) se lee de la propia política,
    y la sección bc de la configuración debe coincidir con ella: la activación
    no tiene pesos y el state_dict no la delata. Se escribe en un archivo
    temporal junto a ruta que se renombra al final, así que un fallo al
    escribir no deja un checkpoint a medias.

    Raises:
        FileExistsError: Si ruta ya existe.
        ValueError: Si la política, la configuración o los metadatos no son válidos.
    """
    ruta = Path(ruta)
    if ruta.exists():
        raise FileExistsError(f"{ruta} ya existe; no se sobrescribe")
    if not isinstance(politica, PoliticaBC):
        raise ValueError("politica debe ser PoliticaBC")
    bc_cfg = cfg.get("bc") if isinstance(cfg, dict) else None
    if not isinstance(bc_cfg, dict):
        raise ValueError("cfg necesita la sección bc")
    capas, activacion = list(politica.capas_ocultas), politica.activacion
    pedidas = bc_cfg.get("capas_ocultas")
    if not isinstance(pedidas, (list, tuple)) or list(pedidas) != capas:
        raise ValueError(f"bc.capas_ocultas no coincide con la política: tiene {capas} y la "
                         f"configuración pide {pedidas!r}")
    if bc_cfg.get("activacion") != activacion:
        raise ValueError(f"bc.activacion no coincide con la política: tiene {activacion!r} y "
                         f"la configuración pide {bc_cfg.get('activacion')!r}")
    if politica.continua and "log_std_inicial" not in bc_cfg:
        raise ValueError("La configuración necesita bc.log_std_inicial")
    arquitectura = {
        "dim_observacion": politica.dim_observacion,
        "dim_accion": politica.dim_accion,
        "continua": politica.continua,
        "capas_ocultas": capas,
        "activacion": activacion,
        "log_std_inicial": (float(bc_cfg["log_std_inicial"])
                            if politica.continua else None),
    }
    payload = {"formato": _FORMATO_BC, "version": _VERSION_BC,
               "arquitectura": arquitectura, "state_dict": politica.state_dict(),
               "metadatos": _metadatos_checkpoint(cfg, metadatos)}
    ruta.parent.mkdir(parents=True, exist_ok=True)
    descriptor, nombre = tempfile.mkstemp(prefix=f".{ruta.name}.", suffix=".tmp", dir=ruta.parent)
    os.close(descriptor)
    temporal = Path(nombre)
    try:
        torch.save(payload, temporal)
        if ruta.exists():
            raise FileExistsError(f"{ruta} ya existe; no se sobrescribe")
        temporal.rename(ruta)
    except BaseException:
        # También con Ctrl+C: no queda ni el temporal ni un checkpoint a medias.
        temporal.unlink(missing_ok=True)
        raise


def cargar_politica_bc(ruta: Path) -> tuple[PoliticaBC, dict[str, Any]]:
    """Carga un checkpoint BC usando deserialización restringida de PyTorch.

    Raises:
        FileNotFoundError: Si ruta no existe.
        ValueError: Si el archivo está dañado o no es un checkpoint BC compatible.
    """
    ruta = Path(ruta)
    if not ruta.is_file():
        raise FileNotFoundError(ruta)
    try:
        payload = torch.load(ruta, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, EOFError, pickle.UnpicklingError) as error:
        raise ValueError(f"Checkpoint BC dañado o no válido: {ruta}") from error
    if (not isinstance(payload, dict) or payload.get("formato") != _FORMATO_BC
            or payload.get("version") != _VERSION_BC):
        raise ValueError("Formato o versión de checkpoint BC no compatibles")
    arquitectura = payload.get("arquitectura")
    if not isinstance(arquitectura, dict):
        raise ValueError("Checkpoint BC sin arquitectura")
    try:
        politica = PoliticaBC(**arquitectura)
        politica.load_state_dict(payload["state_dict"], strict=True)
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise ValueError("Checkpoint BC con arquitectura o pesos incompatibles") from error
    metadatos = payload.get("metadatos")
    if not isinstance(metadatos, dict):
        raise ValueError("Checkpoint BC sin metadatos")
    politica.eval()
    return politica, metadatos
