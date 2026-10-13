"""Bucle de entrenamiento y evaluacion."""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from torch.utils.data import (
    BatchSampler, DataLoader, IterableDataset, RandomSampler, SequentialSampler,
    SubsetRandomSampler,
)

from src.data.loaders import construir_loaders_bc
from src.data.splits import SplitsDemostraciones
from src.data.tipos import Trayectoria
from src.models.factory import PoliticaBC, construir_politica_bc
from src.utils.tracking import registrar

OPTIMIZADORES = {
    "adam": torch.optim.Adam,
    "sgd": torch.optim.SGD,
    "rmsprop": torch.optim.RMSprop,
}


def entrenar(modelo: nn.Module, train_loader: DataLoader,
             val_loader: DataLoader, cfg: dict, run=None) -> nn.Module:
    tcfg = cfg["entrenamiento"]
    opt = OPTIMIZADORES[tcfg["optimizador"]](modelo.parameters(),
                                             lr=tcfg["learning_rate"])
    criterio = nn.CrossEntropyLoss()
    max_steps = tcfg.get("max_steps")

    paso = 0
    for epoca in range(tcfg["epochs"]):
        modelo.train()
        for x, y in train_loader:
            opt.zero_grad()
            perdida = criterio(modelo(x), y)
            perdida.backward()
            opt.step()

            paso += 1
            # .item() rompe el grafo: sin esto se acumula memoria hasta reventar
            registrar(run, {"train/loss": perdida.item()}, paso)
            if max_steps and paso >= max_steps:
                return modelo

        metricas = evaluar(modelo, val_loader, cfg)
        registrar(run, {f"val/{k}": v for k, v in metricas.items()}, paso)
        print(f"epoca {epoca + 1}/{tcfg['epochs']}  "
              + "  ".join(f"{k}={v:.4f}" for k, v in metricas.items()))
    return modelo


@torch.no_grad()
def evaluar(modelo: nn.Module, loader: DataLoader, cfg: dict) -> dict[str, float]:
    """Evaluacion. El decorador no_grad no es opcional: sin el, la evaluacion
    construye grafo y agota la memoria."""
    modelo.eval()
    criterio = nn.CrossEntropyLoss()
    perdida_total, aciertos, total = 0.0, 0, 0

    for x, y in loader:
        logits = modelo(x)
        perdida_total += criterio(logits, y).item() * y.size(0)
        # dim=1 son las clases. dim=0 seria el batch: no da error y esta MAL.
        aciertos += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)

    return {"loss": perdida_total / max(total, 1),
            "accuracy": aciertos / max(total, 1)}


def _entero_positivo_de_cfg(bcfg: dict, clave: str) -> int:
    """Lee bc.clave y comprueba que es un entero positivo."""
    valor = bcfg.get(clave)
    if (isinstance(valor, (bool, np.bool_)) or not isinstance(valor, (int, np.integer))
            or valor < 1):
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
                cfg: dict, run=None) -> dict[str, float | int]:
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
    bcfg = cfg.get("bc") if isinstance(cfg, dict) else None
    if not isinstance(bcfg, dict):
        raise ValueError("La configuración necesita la sección bc")
    epocas_maximas = _entero_positivo_de_cfg(bcfg, "epocas_maximas")
    paciencia = _entero_positivo_de_cfg(bcfg, "paciencia")
    optimizador = bcfg.get("optimizador")
    if not isinstance(optimizador, str) or optimizador not in OPTIMIZADORES:
        raise ValueError(f"bc.optimizador desconocido: {optimizador!r}; "
                         f"opciones: {sorted(OPTIMIZADORES)}")
    learning_rate = bcfg.get("learning_rate")
    if (isinstance(learning_rate, (bool, np.bool_))
            or not isinstance(learning_rate, (int, float, np.integer, np.floating))
            or not math.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError(f"bc.learning_rate debe ser un real positivo, no {learning_rate!r}")
    normalizar = bcfg.get("normalizar_observaciones")
    if not isinstance(normalizar, (bool, np.bool_)):
        raise ValueError(f"bc.normalizar_observaciones debe ser booleano, no {normalizar!r}")
    if normalizar:
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
    generador: np.random.Generator, run=None,
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
