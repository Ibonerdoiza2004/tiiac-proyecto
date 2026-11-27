"""Ruta de la pestaña Comparar: crear una comparación de modelos de la biblioteca.

La comparación la juega el trabajador tarea_comparar (src/training/trabajadores.py) en
su propio proceso, como la evaluación de la E1: las partidas y la semilla de
``evaluacion`` en configs/ (las partidas, editables hasta ``laboratorio.partidas_maximas``
de configs/dashboard.yaml) y todos los modelos la partida i antes de la i + 1. Seguirla,
pausarla, reanudarla y finalizarla es común a todos los trabajos (rutas_trabajos.py),
con las órdenes ``vista`` y ``finalizar``. El resultado se queda en el trabajo: no se
guarda en la biblioteca ni en W&B.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from src.app.contrato import ArgumentosComparacionApi, ComparacionCrear, TrabajoApi
from src.app.rutas_laboratorio import comprobar_referencia
from src.app.trabajos import GestorTrabajos
from src.data.biblioteca import Biblioteca
from src.utils.configuracion import resolver_configuracion

# La tarea del gestor, como texto: la API no importa src/training.
TAREA_COMPARAR = "src.training.trabajadores:tarea_comparar"
# Tipos de la biblioteca que se pueden comparar.
TIPOS_COMPARABLES = ("experto", "bc", "gail")


def crear_router_comparar(ruta_config: Path, dashboard: dict[str, Any],
                          gestor: GestorTrabajos, biblioteca: Biblioteca) -> APIRouter:
    """Rutas de Comparar.

    Args:
        ruta_config: Ruta al YAML base.
        dashboard: Contenido de configs/dashboard.yaml (cargar_dashboard).
        gestor: Gestor de trabajos.
        biblioteca: Índice de la biblioteca (para comprobar los modelos).

    Returns:
        Un router con POST /comparaciones.
    """
    router = APIRouter()

    @router.post("/comparaciones", response_model=TrabajoApi, status_code=201,
                 tags=["comparar"])
    def crear_comparacion(peticion: ComparacionCrear) -> dict[str, Any]:
        """Compara modelos de la biblioteca (y el aleatorio) con las mismas partidas.

        Espera en la cola si no hay hueco. Se sigue por el WebSocket /api/trabajos/{id}:
        un progreso por partida de cada modelo (ProgresoComparacionApi) y, al final, el
        resultado (ResultadoComparacion). Sus ``argumentos`` llevan las partidas y la
        semilla que se usan (ArgumentosComparacionApi).
        """
        maximo = dashboard["laboratorio"]["partidas_maximas"]
        if peticion.partidas is not None and peticion.partidas > maximo:
            raise HTTPException(status_code=422, detail=f"Como mucho {maximo} partidas")
        for id_modelo in peticion.modelos:
            try:
                comprobar_referencia(biblioteca, id_modelo, TIPOS_COMPARABLES,
                                     peticion.entorno)
            except ValueError as error:
                raise HTTPException(status_code=422, detail=f"modelos: {error}") from error
        evaluacion = resolver_configuracion(ruta_config, peticion.entorno)["evaluacion"]
        publicos = ArgumentosComparacionApi(
            entorno=peticion.entorno, modelos=peticion.modelos, aleatorio=peticion.aleatorio,
            partidas=peticion.partidas or evaluacion["episodios"],
            semilla=evaluacion["semilla"], vista=peticion.vista).model_dump(mode="json")
        # Lo que recibe el trabajador, con las rutas del servidor: no sale de él.
        argumentos = {"ruta_config": str(ruta_config), "biblioteca": str(biblioteca.raiz),
                      "calidad_jpeg": dashboard["vista"]["calidad_jpeg"],
                      "entorno": peticion.entorno.value, "modelos": list(peticion.modelos),
                      "aleatorio": peticion.aleatorio, "partidas": publicos["partidas"],
                      "vista": peticion.vista}
        return gestor.crear("comparacion", TAREA_COMPARAR, argumentos, publicos=publicos)

    return router
