"""Behavioral Cloning: entrenamiento, guardado y la tarea de su barrido.

- config_bc comprueba la sección bc.
- entrenar_bc entrena una PoliticaBC con parada temprana sobre la pérdida de
  validación; entrenar_bc_subconjunto lo hace para un subconjunto de partidas.
- guardar_politica_bc y cargar_politica_bc guardan la política con su
  arquitectura y su procedencia (cargada con weights_only=True).
- entrenar_y_guardar_bc es un entrenamiento del barrido (scripts/entrenar_bc.py):
  las semillas de su repetición, las mismas que GAIL, así que BC y GAIL de la
  misma repetición entrenan con los mismos pares expertos.
"""

from __future__ import annotations

import copy
import json
import math
import pickle
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from gymnasium import spaces
from torch.utils.data import (
    BatchSampler, DataLoader, IterableDataset, RandomSampler, SequentialSampler,
    SubsetRandomSampler,
)

from src.data.entornos import espacios_entorno
from src.data.loaders import construir_loaders_bc
from src.data.splits import SplitsDemostraciones, comprobar_huella
from src.data.tipos import Trayectoria
from src.models.factory import PoliticaBC, comprobar_arquitectura, construir_politica_bc
from src.training.barrido import comprobar_config_comun
from src.utils.archivos import escribir_archivo_nuevo
from src.utils.procedencia import a_json, estado_git, versiones_y_hardware
from src.utils.seed import crear_generador, fijar_seed, semillas_repeticion, validar_semilla
from src.utils.tracking import (abrir_run_barrido, registrar, registrar_resumen,
                                subir_artefacto, terminar_run, usar_artefacto)
from src.utils.validacion import es_entero_positivo, es_real_finito

OPTIMIZADORES = {
    "adam": torch.optim.Adam,
    "sgd": torch.optim.SGD,
    "rmsprop": torch.optim.RMSprop,
}

_FORMATO_BC = "tiiac-politica-bc"
_VERSION_BC = 1


def _entero_positivo_de_cfg(bcfg: dict, clave: str) -> int:
    """Lee bc.clave y comprueba que es un entero positivo."""
    valor = bcfg.get(clave)
    if not es_entero_positivo(valor):
        raise ValueError(f"bc.{clave} debe ser un entero positivo, no {valor!r}")
    return int(valor)


def _indices_de_train(loader: DataLoader) -> list[int]:
    """Índices del dataset que recorre el loader de train, sin gastar números aleatorios.

    Recorrer el loader sacaría un número del generador global de PyTorch, y leer
    todo el dataset ignoraría su sampler: si train y validación comparten
    dataset, metería estados de validación. Por eso se leen los índices del
    propio sampler. Se admiten SequentialSampler y RandomSampler sin reemplazo y
    con su tamaño por defecto, que recorren todo el dataset, y
    SubsetRandomSampler, que recorre sus índices.

    Raises:
        ValueError: Si el loader usa otro sampler, un batch_sampler propio o un
            dataset iterable, porque entonces no se sabe qué pares recorre.
    """
    muestreador = loader.sampler
    if (isinstance(loader.dataset, IterableDataset)
            or not isinstance(loader.batch_sampler, BatchSampler)
            or loader.batch_sampler.sampler is not muestreador):
        raise ValueError("No se puede saber qué pares recorre el loader de train para normalizar")
    total = len(loader.dataset)
    if isinstance(muestreador, SequentialSampler) or (
            isinstance(muestreador, RandomSampler) and not muestreador.replacement
            and muestreador.num_samples == total):
        return list(range(total))
    if isinstance(muestreador, SubsetRandomSampler):
        return [int(i) for i in muestreador.indices]
    raise ValueError(f"No se puede normalizar con el sampler {type(muestreador).__name__}: "
                     "usa SequentialSampler, RandomSampler o SubsetRandomSampler")


def config_bc(cfg: dict) -> dict:
    """Devuelve la sección bc comprobada: lo que necesita entrenar_bc.

    Raises:
        ValueError: Si falta la sección o epocas_maximas, paciencia,
            optimizador, learning_rate o normalizar_observaciones no son válidos.
    """
    bcfg = cfg.get("bc") if isinstance(cfg, dict) else None
    if not isinstance(bcfg, dict):
        raise ValueError("La configuración necesita la sección bc")
    _entero_positivo_de_cfg(bcfg, "epocas_maximas")
    _entero_positivo_de_cfg(bcfg, "paciencia")
    optimizador = bcfg.get("optimizador")
    if not isinstance(optimizador, str) or optimizador not in OPTIMIZADORES:
        raise ValueError(f"bc.optimizador desconocido: {optimizador!r}; "
                         f"opciones: {sorted(OPTIMIZADORES)}")
    learning_rate = bcfg.get("learning_rate")
    if not es_real_finito(learning_rate) or learning_rate <= 0:
        raise ValueError(f"bc.learning_rate debe ser un real positivo, no {learning_rate!r}")
    normalizar = bcfg.get("normalizar_observaciones")
    if not isinstance(normalizar, (bool, np.bool_)):
        raise ValueError(f"bc.normalizar_observaciones debe ser booleano, no {normalizar!r}")
    return bcfg


@torch.no_grad()
def evaluar_bc(politica: PoliticaBC, loader: DataLoader) -> dict[str, float]:
    """Evalúa la política de BC en modo evaluación y sin gradientes.

    Devuelve la pérdida (log-verosimilitud negativa media por par) y, según el
    tipo de acción, el acierto (fracción de pares cuya acción más probable es la
    del experto) o el error cuadrático entre la media y la acción del experto,
    sumado por componentes y promediado por par, como el código original de
    GAIL. Las medias son por par, aunque los batches tengan tamaños distintos.

    Raises:
        ValueError: Si el loader no tiene pares.
    """
    politica.eval()
    perdida_total, coincidencia_total, total = 0.0, 0.0, 0
    for estados, acciones in loader:
        perdida_total -= politica.log_verosimilitud(estados, acciones).sum().item()
        salida = politica(estados)
        if politica.continua:
            coincidencia_total += ((salida - acciones) ** 2).sum().item()
        else:
            # dim=1 son las acciones; dim=0 seria el batch.
            coincidencia_total += (salida.argmax(dim=1) == acciones).sum().item()
        total += len(acciones)
    if total == 0:
        raise ValueError("El loader de evaluación no tiene pares")
    clave = "error_cuadratico" if politica.continua else "acierto"
    return {"perdida": perdida_total / total, clave: coincidencia_total / total}


def entrenar_bc(politica: PoliticaBC, train_loader: DataLoader, val_loader: DataLoader,
                cfg: dict, run: Any = None) -> dict[str, float | int]:
    """Entrena la política de BC con parada temprana y deja los mejores pesos.

    Si bc.normalizar_observaciones es true, antes de entrenar ajusta la
    normalización de la política con los estados que recorre train_loader
    (los índices de su sampler) y bc.epsilon_normalizacion; nunca con
    validación ni gastando números aleatorios. Si es false, la deja como esté.

    Minimiza la log-verosimilitud negativa de las acciones de train. Tras cada
    época evalúa en validación con el criterio de selección: la pérdida con
    acciones discretas y el error cuadrático de la media con continuas, como el
    código original de GAIL, que evalúa la acción media y así no depende de
    log_std. Si el criterio no baja durante bc.paciencia épocas seguidas, o se
    llega a bc.epocas_maximas, para y restaura los pesos de la mejor época. El
    paper solo dice que se entrena hasta que el error de validación deja de
    bajar: la paciencia es una elección propia.

    Args:
        politica: Política a entrenar; se modifica en el sitio.
        train_loader: Pares de train.
        val_loader: Pares de validación, de partidas distintas a las de train.
        cfg: Configuración resuelta con bc.optimizador, bc.learning_rate,
            bc.epocas_maximas, bc.paciencia y bc.normalizar_observaciones, y
            bc.epsilon_normalizacion si se normaliza.
        run: Run de W&B o None.

    Returns:
        mejor_epoca (desde 1), epocas entrenadas y las métricas de validación
        de la mejor época, con el prefijo val_.

    Raises:
        ValueError: Configuración no válida, train_loader sin pares o con un
            sampler del que no se sabe qué pares recorre (al normalizar).
        FloatingPointError: Si la pérdida de train o alguna métrica de
            validación no es finita.
    """
    bcfg = config_bc(cfg)
    epocas_maximas, paciencia = int(bcfg["epocas_maximas"]), int(bcfg["paciencia"])
    optimizador, learning_rate = bcfg["optimizador"], bcfg["learning_rate"]
    if bcfg["normalizar_observaciones"]:
        indices = _indices_de_train(train_loader)
        if not indices:
            raise ValueError("El loader de train no tiene pares")
        estados_train = torch.stack([train_loader.dataset[i][0] for i in indices])
        politica.ajustar_normalizacion(estados_train, bcfg.get("epsilon_normalizacion"))
    opt = OPTIMIZADORES[optimizador](politica.parameters(), lr=float(learning_rate))

    criterio = "error_cuadratico" if politica.continua else "perdida"
    mejores: dict[str, float] = {}
    mejor_estado = copy.deepcopy(politica.state_dict())
    mejor_epoca, sin_mejora, paso, epoca = 0, 0, 0, 0
    for epoca in range(1, epocas_maximas + 1):
        politica.train()
        for estados, acciones in train_loader:
            opt.zero_grad()
            perdida = -politica.log_verosimilitud(estados, acciones).mean()
            if not torch.isfinite(perdida):
                raise FloatingPointError(f"Pérdida de train no finita en la época {epoca}")
            perdida.backward()
            opt.step()
            paso += 1
            # .item() rompe el grafo: sin esto se acumula memoria hasta reventar
            registrar(run, {"bc/train/perdida": perdida.item()}, paso)
        if paso == 0:
            raise ValueError("El loader de train no tiene pares")

        metricas = evaluar_bc(politica, val_loader)
        registrar(run, {f"bc/val/{k}": v for k, v in metricas.items()}, paso)
        if not all(math.isfinite(v) for v in metricas.values()):
            raise FloatingPointError(
                f"Métricas de validación no finitas en la época {epoca}: {metricas}")
        if not mejores or metricas[criterio] < mejores[criterio]:
            mejores, mejor_epoca, sin_mejora = metricas, epoca, 0
            mejor_estado = copy.deepcopy(politica.state_dict())
        else:
            sin_mejora += 1
            if sin_mejora >= paciencia:
                break

    politica.load_state_dict(mejor_estado)
    return {"mejor_epoca": mejor_epoca, "epocas": epoca,
            **{f"val_{k}": v for k, v in mejores.items()}}


def entrenar_bc_subconjunto(
    trayectorias: Sequence[Trayectoria], splits: SplitsDemostraciones, cantidad: int,
    espacio_observacion: spaces.Space, espacio_accion: spaces.Space, cfg: dict,
    generador: np.random.Generator, run: Any = None,
) -> tuple[PoliticaBC, dict[str, float | int]]:
    """Recorrido completo de BC para un subconjunto de partidas de train.

    Construye los loaders del subconjunto cantidad y de validación, la política
    según los espacios del entorno y la entrena con entrenar_bc. El generador
    decide qué pares se toman de cada partida; la inicialización de los pesos y
    el orden de los batches dependen de la semilla global, que quien llama fija
    con src.utils.seed.fijar_seed antes de cada ejecución.

    Args:
        trayectorias: Todas las partidas del dataset para el que se crearon los splits.
        splits: Splits de esas partidas.
        cantidad: Número de partidas de train que se usan, por ejemplo 1, 4, 7 o 10.
            Tiene que haber un subconjunto de ese tamaño: splits.subconjuntos
            asocia cada tamaño con los IDs de sus partidas.
        espacio_observacion: Espacio de observación del entorno.
        espacio_accion: Espacio de acción del entorno.
        cfg: Configuración resuelta con la sección bc completa.
        generador: Generador de src.utils.seed.crear_generador.
        run: Run de W&B o None.

    Returns:
        La política con los pesos de la mejor época y un resumen con cantidad,
        pares_train, pares_validacion y lo que devuelve entrenar_bc.

    Raises:
        ValueError: Datos, splits, espacios o configuración no válidos.
        FloatingPointError: Si la pérdida de train no es finita.
    """
    train_loader, val_loader = construir_loaders_bc(
        trayectorias, splits, cantidad, espacio_accion, cfg, generador)
    politica = construir_politica_bc(cfg, espacio_observacion, espacio_accion)
    resultado = entrenar_bc(politica, train_loader, val_loader, cfg, run)
    return politica, {"cantidad": int(cantidad), "pares_train": len(train_loader.dataset),
                      "pares_validacion": len(val_loader.dataset), **resultado}


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
    # Si falla al escribir, también por Ctrl+C, no queda un checkpoint a medias.
    escribir_archivo_nuevo(ruta, lambda temporal: torch.save(payload, temporal))


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


def comprobar_config_barrido_bc(cfg: dict) -> None:
    """Comprueba la configuración del barrido de BC antes de entrenar nada.

    La sección bc (config_bc, más la arquitectura y bc.batch_size) y lo común a
    los barridos (comprobar_config_comun): así scripts/entrenar_bc.py rechaza
    una configuración no válida una sola vez, en lugar de fallar en cada
    entrenamiento.

    Raises:
        ValueError: Si falta alguna clave o algún valor no es válido.
    """
    bcfg = config_bc(cfg)
    comprobar_arquitectura(bcfg.get("capas_ocultas"), bcfg.get("activacion"))
    if not es_entero_positivo(bcfg.get("batch_size")):
        raise ValueError("bc.batch_size debe ser un entero positivo, no "
                         f"{bcfg.get('batch_size')!r}")
    comprobar_config_comun(cfg)


def entrenar_y_guardar_bc(cfg: dict, trayectorias: Sequence[Trayectoria],
                          splits: SplitsDemostraciones, cantidad: int, repeticion: int,
                          destino: Path, datos: dict[str, str],
                          grupo: str | None = None) -> dict[str, Any]:
    """Entrena un BC del barrido con las semillas de su repetición y lo guarda.

    Las semillas salen de semillas_repeticion, como en GAIL: fijar_seed con la
    del modelo (pesos y orden de los batches) y un generador con la de los
    pares, así que BC y GAIL de la misma repetición entrenan con los mismos
    pares expertos. Abre su propio run de W&B (según wandb.modo) y lo cierra
    con terminar_run, como fallido si falla. Solo depende de sus argumentos, así
    que da lo mismo en serie que en paralelo, en otro proceso.

    Args:
        cfg: Configuración resuelta.
        trayectorias: Todas las partidas del dataset.
        splits: Sus splits.
        cantidad: Tamaño del subconjunto de train.
        repeticion: Número de repetición.
        destino: Archivo nuevo donde guardar la política (guardar_politica_bc).
        datos: Procedencia de los datos (procedencia_datos).

    Returns:
        El resumen de entrenar_bc_subconjunto con la repetición, las semillas y
        los segundos medidos.

    Raises:
        ValueError: Si algo no es válido o los datos no son los de los splits.
        FileExistsError: Si destino ya existe.
    """
    if datos.get("huella_dataset") != splits.huella_dataset:
        raise ValueError("Las demostraciones no son las de los splits: la huella del NPZ no "
                         "coincide")
    semilla_base = cfg["experimento"]["semilla_repeticiones"]
    semillas = semillas_repeticion(semilla_base, repeticion)
    espacios = espacios_entorno(cfg)
    run = abrir_run_barrido(cfg, "bc", cantidad, repeticion, grupo)
    try:
        if cfg.get("wandb", {}).get("modo") == "online":
            usar_artefacto(run, f"demostraciones-{datos['huella_dataset'][:12]}:latest")
        inicio = time.perf_counter()
        fijar_seed(semillas["modelo"])
        politica, resumen = entrenar_bc_subconjunto(trayectorias, splits, cantidad, *espacios,
                                                    cfg, crear_generador(semillas["pares"]), run)
        resumen = {**resumen, "repeticion": int(repeticion),
                   "semillas": {"base": int(semilla_base), **semillas},
                   "segundos": time.perf_counter() - inicio}
        guardar_politica_bc(destino, politica, cfg,
                            {"semilla": semillas["modelo"], "subconjunto": cantidad,
                             "huella_dataset": datos["huella_dataset"],
                             "adicional": {"resumen": resumen, "datos": datos}})
        registrar_resumen(run, {**resumen, **datos})
        subir_artefacto(run, f"bc-{cfg['entorno']['id']}-c{cantidad}-r{repeticion}-"
                        f"{grupo or 'independiente'}", "bc",
                        [destino], {"cantidad": cantidad, "repeticion": repeticion,
                                    **datos})
    except BaseException as error:
        # Si falla el entrenamiento o el guardado, W&B lo marca como fallido.
        terminar_run(run, error)
        raise
    terminar_run(run)
    return resumen
