"""Predicción, evaluación y checkpoints de las políticas del experimento.

Para evaluar agentes guardados: experto con ``cargar_experto(dir)`` y sus
estadísticas en ``metadatos["resumen"]["normalizacion"]``; BC con
``cargar_politica_bc(ruta)`` y sin estadísticas; agente aleatorio con
``"aleatorio"``. GAIL usará ``cargar_gail`` y el mismo campo de estadísticas
que el experto.
"""

from __future__ import annotations

import contextlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from gymnasium import spaces
from sb3_contrib import TRPO

from src.data.entornos import crear_entorno
from src.models.factory import PoliticaBC
from src.utils.procedencia import estado_git, versiones_y_hardware
from src.utils.seed import (generador_torch_aislado, reiniciar_con_semilla,
                            semillas_independientes, validar_semilla)

_FORMATO_BC = "tiiac-politica-bc"
_VERSION_BC = 1


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
    if (isinstance(episodios, bool) or not isinstance(episodios, (int, np.integer))
            or episodios <= 0):
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
    """Evalúa usando episodios, semilla y modo de la sección evaluacion."""
    ecfg = config_evaluacion(cfg)
    return evaluar_politica(agente, cfg, ecfg["episodios"], ecfg["semilla"],
                            ecfg["determinista"], estadisticas)


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
    episodio. Las acciones aleatorias usan el espacio sembrado por ``seed.py``.
    Para TRPO, las estadísticas opcionales son las guardadas por ``VecNormalize``.

    Raises:
        ValueError: Si el agente, la observación o el espacio no son compatibles.
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
                distribucion = agente.distribucion(entrada)
                salida = agente(entrada)[0] if determinista else distribucion.sample()[0]
                accion = salida.cpu().numpy()
            else:
                logits = agente(entrada)
                indice = (int(logits.argmax(dim=1).item()) if determinista else
                          int(agente.distribucion(entrada).sample().item()))
                accion = indice + int(espacio_accion.start)
    elif isinstance(agente, TRPO):
        if not isinstance(espacio_accion, (spaces.Discrete, spaces.Box)):
            raise ValueError("TRPO requiere espacio Discrete o Box")
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

    La recompensa es la del entorno y cada episodio tiene una semilla derivada
    de ``semilla`` mediante ``semillas_independientes``. Por ello los resultados
    son pareados y repetibles. La evaluación no crea gradientes y restaura el
    estado aleatorio global de PyTorch tras cada episodio.

    Returns:
        Retorno medio, desviación poblacional, retornos y longitudes por episodio.

    Raises:
        ValueError: Si episodios, semilla o configuración no son válidos.
    """
    if (isinstance(episodios, bool) or not isinstance(episodios, (int, np.integer))
            or episodios <= 0):
        raise ValueError("episodios debe ser un entero positivo")
    semillas = semillas_independientes(validar_semilla(semilla), int(episodios))
    env = crear_entorno(cfg, None)
    retornos: list[float] = []
    longitudes: list[int] = []
    try:
        if isinstance(agente, PoliticaBC):
            agente.eval()
        for semilla_episodio in semillas:
            observacion, _ = reiniciar_con_semilla(env, semilla_episodio)
            retorno = 0.0
            longitud = 0
            with torch.no_grad(), generador_torch_aislado(semilla_episodio):
                terminado = truncado = False
                while not (terminado or truncado):
                    accion = predecir_accion(agente, observacion, env.action_space,
                                             determinista, None, estadisticas)
                    observacion, recompensa, terminado, truncado, _ = env.step(accion)
                    retorno += float(recompensa)
                    longitud += 1
            retornos.append(retorno)
            longitudes.append(longitud)
    finally:
        env.close()
    return {
        "retorno_medio": float(np.mean(retornos)),
        "retorno_desviacion": float(np.std(retornos)),
        "retornos": retornos,
        "longitudes": longitudes,
        "longitud_media": float(np.mean(longitudes)),
    }


def _metadatos_checkpoint(cfg: dict[str, Any], metadatos: dict[str, Any]) -> dict[str, Any]:
    """Completa y serializa metadatos de procedencia para un checkpoint."""
    requeridos = {"semilla", "subconjunto", "huella_dataset"}
    if not isinstance(metadatos, dict) or not requeridos <= metadatos.keys():
        raise ValueError("metadatos necesita semilla, subconjunto y huella_dataset")
    validar_semilla(metadatos["semilla"])
    resultado = {
        "configuracion": cfg,
        "entorno": cfg.get("entorno", {}).get("id"),
        "semilla": int(metadatos["semilla"]),
        "subconjunto": metadatos["subconjunto"],
        "huella_dataset": metadatos["huella_dataset"],
        "codigo": estado_git(),
        "versiones": versiones_y_hardware(),
        "adicional": metadatos.get("adicional", {}),
    }
    try:
        return json.loads(json.dumps(resultado, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError(f"metadatos no son JSON finito: {error}") from error


def normalizar_observacion(observacion: np.ndarray,
                           estadisticas: dict[str, Any] | None) -> np.ndarray:
    """Aplica estadísticas de VecNormalize guardadas, si están disponibles."""
    if estadisticas is None:
        return observacion
    media = np.asarray(estadisticas["media"])
    escala = np.sqrt(np.asarray(estadisticas["varianza"]) + estadisticas["epsilon"])
    return np.clip((observacion - media) / escala, -estadisticas["limite"],
                   estadisticas["limite"]).astype(np.float32)
def guardar_politica_bc(
    ruta: Path,
    politica: PoliticaBC,
    cfg: dict[str, Any],
    metadatos: dict[str, Any],
) -> None:
    """Guarda BC sin sobrescribir, incluyendo arquitectura, buffers y procedencia."""
    ruta = Path(ruta)
    if ruta.exists():
        raise FileExistsError(f"{ruta} ya existe; no se sobrescribe")
    if not isinstance(politica, PoliticaBC):
        raise ValueError("politica debe ser PoliticaBC")
    bc_cfg = cfg.get("bc") if isinstance(cfg, dict) else None
    if not isinstance(bc_cfg, dict):
        raise ValueError("cfg necesita la sección bc")
    capas_modelo = [c.out_features for c in politica.red
                    if isinstance(c, torch.nn.Linear)][:-1]
    if bc_cfg.get("capas_ocultas") != capas_modelo:
        raise ValueError("bc.capas_ocultas no coincide con la arquitectura de la política")
    if politica.continua and "log_std_inicial" not in bc_cfg:
        raise ValueError("La configuración necesita bc.log_std_inicial")
    arquitectura = {
        "dim_observacion": politica.dim_observacion,
        "dim_accion": politica.dim_accion,
        "continua": politica.continua,
        "capas_ocultas": capas_modelo,
        "activacion": bc_cfg["activacion"],
        "log_std_inicial": (float(bc_cfg["log_std_inicial"])
                            if politica.continua else None),
    }
    payload = {"formato": _FORMATO_BC, "version": _VERSION_BC,
               "arquitectura": arquitectura, "state_dict": politica.state_dict(),
               "metadatos": _metadatos_checkpoint(cfg, metadatos)}
    ruta.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, ruta)


def cargar_politica_bc(ruta: Path) -> tuple[PoliticaBC, dict[str, Any]]:
    """Carga un checkpoint BC usando deserialización restringida de PyTorch."""
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
