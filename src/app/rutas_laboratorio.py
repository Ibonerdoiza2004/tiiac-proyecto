"""Rutas del Laboratorio: configuración del juego, esquema de hiperparámetros y crear
sesiones y entrenamientos.

Las que solo leen la configuración funcionan ya. Las de crear sesiones y
entrenamientos validan la petición entera (con los mismos errores que tendrán) y
responden 501 hasta que lleguen sus trabajadores (ramas laboratorio-jugar-api y
laboratorio-entrenar-api). Seguir y controlar lo creado es común a todos los trabajos
(rutas_trabajos.py).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from src.app.ajustes import comprobar_ajustes, esquema_ajustes
from src.app.contrato import (ConfiguracionJuegoApi, EntrenamientoCrear, EsquemaAjustesApi,
                              MetodoEntrenamiento, ResultadoGuardadoApi, SesionCrear,
                              SesionGuardar, TrabajoApi)
from src.app.trabajos import GestorTrabajos, OperacionNoValida
from src.data.biblioteca import nombre_por_defecto
from src.utils.configuracion import TipoEntorno, resolver_configuracion

PENDIENTE_JUGAR = ("Todavía no implementado: las sesiones llegan con la rama "
                   "laboratorio-jugar-api.")
PENDIENTE_ENTRENAR = ("Todavía no implementado: los entrenamientos llegan con la rama "
                      "laboratorio-entrenar-api.")


@lru_cache(maxsize=None)
def fps_entorno(gym_id: str) -> int:
    """Imágenes por segundo del entorno a velocidad ×1 (``render_fps`` de Gymnasium).

    En los de MuJoCo solo se conoce al crear el entorno (depende de su paso de
    tiempo), así que se crea una vez sin dibujar y se guarda el resultado.

    Args:
        gym_id: Id de Gymnasium, como CartPole-v1.

    Returns:
        Las imágenes por segundo.

    Raises:
        ValueError: Si el entorno no declara render_fps.
    """
    import gymnasium  # noqa: PLC0415 - solo al pedirlo por primera vez

    entorno = gymnasium.make(gym_id)
    try:
        fps = entorno.metadata.get("render_fps")
    finally:
        entorno.close()
    if not isinstance(fps, int) or fps < 1:
        raise ValueError(f"{gym_id} no declara render_fps")
    return fps


def nombre_modelo(peticion: EntrenamientoCrear, gym: str) -> str:
    """Nombre con el que se guardará el modelo: el del usuario, sin espacios a los lados,
    o uno generado (nombre_por_defecto) si no puso ninguno.

    Args:
        peticion: Petición de entrenamiento ya validada.
        gym: Id de Gymnasium del entorno.
    """
    return peticion.nombre.strip() if peticion.nombre else nombre_por_defecto(peticion.metodo,
                                                                               gym)


def crear_router_laboratorio(ruta_config: Path, dashboard: dict[str, Any],
                             gestor: GestorTrabajos) -> APIRouter:
    """Rutas del Laboratorio.

    Args:
        ruta_config: Ruta al YAML base.
        dashboard: Contenido de configs/dashboard.yaml (cargar_dashboard).
        gestor: Gestor de trabajos (para comprobar las sesiones al guardar).

    Returns:
        Un router con las rutas de /laboratorio, /sesiones y /entrenamientos.
    """
    router = APIRouter()
    laboratorio = dashboard["laboratorio"]

    def no_valida(motivo: str) -> HTTPException:
        """Error 422 con el formato de FastAPI."""
        return HTTPException(status_code=422, detail=motivo)

    @router.get("/laboratorio/{entorno}", response_model=ConfiguracionJuegoApi,
                tags=["laboratorio"])
    def configuracion_juego(entorno: TipoEntorno) -> dict[str, Any]:
        """Velocidad del entorno, si se juega a mano y las partidas por defecto."""
        cfg = resolver_configuracion(ruta_config, entorno)
        demostraciones = cfg["demostraciones"]
        return {"entorno": entorno, "fps": fps_entorno(cfg["entorno"]["id"]),
                "jugar_a_mano": entorno.value in laboratorio["jugar_a_mano"],
                "partidas_experto": demostraciones["partidas_train"]
                + demostraciones["partidas_validacion"],
                "partidas_maximas": laboratorio["partidas_maximas"],
                "semilla": demostraciones["semilla"]}

    @router.get("/laboratorio/{entorno}/ajustes/{metodo}", response_model=EsquemaAjustesApi,
                tags=["laboratorio"])
    def ajustes(entorno: TipoEntorno, metodo: MetodoEntrenamiento) -> dict[str, Any]:
        """Hiperparámetros del método con los valores del perfil, para el formulario."""
        return esquema_ajustes(ruta_config, entorno, metodo, dashboard)

    @router.post("/sesiones", response_model=TrabajoApi, status_code=201, tags=["sesiones"])
    def crear_sesion(peticion: SesionCrear) -> dict[str, Any]:
        """Jugar a mano o ver al experto. Se sigue por el WebSocket /api/trabajos/{id}."""
        if peticion.modo == "manual" and peticion.entorno.value not in laboratorio["jugar_a_mano"]:
            raise no_valida(f"{peticion.entorno.value} no se puede jugar a mano: solo "
                            f"{', '.join(laboratorio['jugar_a_mano'])}")
        if peticion.partidas is not None and peticion.partidas > laboratorio["partidas_maximas"]:
            raise no_valida(f"Como mucho {laboratorio['partidas_maximas']} partidas")
        raise HTTPException(status_code=501, detail=PENDIENTE_JUGAR)

    @router.post("/sesiones/{id_trabajo}/guardar", response_model=ResultadoGuardadoApi,
                 tags=["sesiones"])
    def guardar_sesion(id_trabajo: str, peticion: SesionGuardar) -> dict[str, Any]:
        """Guarda en la biblioteca las partidas y tramos elegidos y cierra la sesión."""
        if gestor.obtener(id_trabajo)["tipo"] != "sesion":  # 404 si no existe
            raise OperacionNoValida(f"El trabajo {id_trabajo} no es una sesión")
        raise HTTPException(status_code=501, detail=PENDIENTE_JUGAR)

    @router.post("/entrenamientos", response_model=TrabajoApi, status_code=201,
                 tags=["entrenamientos"])
    def crear_entrenamiento(peticion: EntrenamientoCrear) -> dict[str, Any]:
        """Entrena un experto, un BC o un GAIL. Se sigue por /api/trabajos/{id}."""
        try:
            cfg = comprobar_ajustes(ruta_config, peticion.entorno, peticion.metodo,
                                    peticion.ajustes, dashboard)
        except ValueError as error:
            raise no_valida(str(error)) from error
        nombre_modelo(peticion, cfg["entorno"]["id"])  # lo recibirá el trabajador
        raise HTTPException(status_code=501, detail=PENDIENTE_ENTRENAR)

    return router
