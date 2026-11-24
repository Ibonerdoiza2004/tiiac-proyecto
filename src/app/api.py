"""API del dashboard: rutas bajo /api y el frontend compilado en la raíz.

El contrato de la API está en el propio código: los modelos Pydantic de las rutas
generan la documentación HTTP en /docs (OpenAPI). Los errores siguen el formato de
FastAPI, ``{"detail": ...}``, que es el que lee el cliente del frontend.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from src.app.ajustes import cargar_dashboard, comprobar_esquemas
from src.app.contrato import EntornoApi, EstadoApi, WandbApi
from src.app import estado_wandb as modulo_wandb
from src.app.estado_wandb import Comprobacion, EstadoWandb
from src.app.rutas_biblioteca import crear_router_biblioteca
from src.app.rutas_laboratorio import crear_router_laboratorio
from src.app.rutas_trabajos import crear_router_trabajos
from src.app.trabajos import GestorTrabajos, OperacionNoValida, SinHueco, TrabajoDesconocido
from src.data.biblioteca import Biblioteca
from src.utils.configuracion import (TipoEntorno, leer_configuracion_base,
                                     resolver_configuracion)
from src.utils.validacion import es_entero, es_entero_positivo

PREFIJO = "/api"
CONFIG_POR_DEFECTO = Path("configs/default.yaml")
SIN_FRONTEND = ("El frontend no está compilado: ejecuta `npm ci` y `npm run build` "
                "en web/, o usa `npm run dev` con la API en marcha.")
PUERTO_MAXIMO = 65535
# Textos, niveles y rangos del formulario, junto al YAML base.
ARCHIVO_DASHBOARD = "dashboard.yaml"


def config_aplicativo(cfg: dict[str, Any]) -> dict[str, Any]:
    """Comprueba y devuelve lo que la API lee de la configuración base.

    Args:
        cfg: YAML base (leer_configuracion_base).

    Returns:
        ``{"host", "puerto", "web", "biblioteca", "procesos"}``, con las rutas como Path.

    Raises:
        ValueError: Si falta alguna clave o su valor no es válido.
    """
    try:
        seccion = cfg["aplicativo"]
        host, puerto, web = seccion["host"], seccion["puerto"], seccion["web"]
        biblioteca = seccion["biblioteca"]
        procesos = cfg["experimento"]["procesos"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"Falta en la configuración del aplicativo: {error}") from error
    if not isinstance(host, str) or not host:
        raise ValueError(f"aplicativo.host debe ser un texto no vacío, no {host!r}")
    if not es_entero(puerto) or not 1 <= puerto <= PUERTO_MAXIMO:
        raise ValueError(f"aplicativo.puerto debe ser un entero de 1 a {PUERTO_MAXIMO}, "
                         f"no {puerto!r}")
    if not isinstance(web, str) or not web:
        raise ValueError(f"aplicativo.web debe ser una ruta, no {web!r}")
    if not isinstance(biblioteca, str) or not biblioteca:
        raise ValueError(f"aplicativo.biblioteca debe ser una ruta, no {biblioteca!r}")
    if not es_entero_positivo(procesos):
        raise ValueError(f"experimento.procesos debe ser un entero positivo, no {procesos!r}")
    return {"host": host, "puerto": int(puerto), "web": Path(web),
            "biblioteca": Path(biblioteca), "procesos": int(procesos)}


def describir_entornos(ruta_config: Path) -> list[EntornoApi]:
    """Resuelve el perfil de cada entorno y devuelve su clave y su id de Gymnasium.

    Args:
        ruta_config: Ruta al YAML base.

    Returns:
        Un EntornoApi por entorno, en el orden de TipoEntorno.

    Raises:
        ValueError: Si algún perfil no se puede resolver.
    """
    return [EntornoApi(id=tipo.value,
                       gym=resolver_configuracion(ruta_config, tipo)["entorno"]["id"])
            for tipo in TipoEntorno]


def _respuesta_error(codigo: int) -> Any:
    """Manejador que convierte una excepción en ``{"detail": mensaje}`` con ese código."""
    async def manejar(request: Request, error: Exception) -> JSONResponse:
        """Responde con el mensaje de la excepción."""
        mensaje = error.args[0] if isinstance(error, KeyError) and error.args else str(error)
        if isinstance(error, TrabajoDesconocido):
            mensaje = f"No existe el trabajo {mensaje}"
        return JSONResponse(status_code=codigo, content={"detail": mensaje})
    return manejar


def crear_app(ruta_config: Path = CONFIG_POR_DEFECTO,
              gestor: GestorTrabajos | None = None,
              comprobar_wandb: Comprobacion | None = None) -> FastAPI:
    """Crea la aplicación: la API bajo /api y, si está compilado, el frontend en /.

    La configuración (y configs/dashboard.yaml, junto al YAML base) se lee y se
    comprueba aquí, una vez, con el formulario de cada entorno y método: si algo no
    vale, la aplicación no se crea. El gestor de trabajos se cierra al apagar el
    servidor, así que no quedan procesos sueltos.

    Args:
        ruta_config: Ruta al YAML base.
        gestor: Gestor de trabajos (por defecto, uno nuevo con experimento.procesos).
        comprobar_wandb: Comprueba si W&B se puede usar (None si sí; si no, el motivo);
            por defecto, comprobar_sesion_wandb. Se llama al arrancar, en segundo plano, y
            al pedirlo por la API.

    Returns:
        La aplicación FastAPI.

    Raises:
        ValueError: Si la configuración base o algún perfil no son válidos.
    """
    aplicativo = config_aplicativo(leer_configuracion_base(ruta_config))
    entornos = describir_entornos(ruta_config)
    dashboard = cargar_dashboard(Path(ruta_config).parent / ARCHIVO_DASHBOARD)
    comprobar_esquemas(Path(ruta_config), dashboard)
    gestor = gestor or GestorTrabajos(aplicativo["procesos"])
    biblioteca = Biblioteca(aplicativo["biblioteca"], {e.id: e.gym for e in entornos})
    estado_wandb = EstadoWandb(comprobar_wandb or modulo_wandb.comprobar_sesion_wandb)

    @asynccontextmanager
    async def ciclo_de_vida(app: FastAPI) -> AsyncIterator[None]:
        """Al arrancar, comprueba W&B en segundo plano; al apagar, cancela los trabajos."""
        comprobacion = asyncio.create_task(asyncio.to_thread(estado_wandb.comprobar))
        yield
        await asyncio.to_thread(gestor.cerrar)
        if not comprobacion.done():
            comprobacion.cancel()

    app = FastAPI(title="GAIL lab", description="API del dashboard de TIIAC.",
                  lifespan=ciclo_de_vida)
    app.state.gestor = gestor
    app.state.biblioteca = biblioteca
    app.state.estado_wandb = estado_wandb
    for excepcion, codigo in ((TrabajoDesconocido, 404), (OperacionNoValida, 409),
                              (SinHueco, 503)):
        app.add_exception_handler(excepcion, _respuesta_error(codigo))

    api = APIRouter(prefix=PREFIJO)

    @api.get("/estado", response_model=EstadoApi)
    def leer_estado() -> EstadoApi:
        """Entornos, trabajos a la vez, huecos libres y si se puede usar W&B."""
        return EstadoApi(entornos=entornos, procesos=gestor.procesos, libres=gestor.libres(),
                         wandb=WandbApi(**estado_wandb.describir()))

    @api.post("/wandb/comprobar", response_model=WandbApi, tags=["wandb"])
    async def comprobar_wandb_ahora() -> dict[str, Any]:
        """Vuelve a comprobar la sesión de W&B (por ejemplo, tras hacer wandb login)."""
        return await asyncio.to_thread(estado_wandb.comprobar)

    api.include_router(crear_router_trabajos(gestor))
    api.include_router(crear_router_laboratorio(Path(ruta_config), dashboard, gestor,
                                                estado_wandb, biblioteca))
    api.include_router(crear_router_biblioteca(biblioteca))
    app.include_router(api)

    web = aplicativo["web"]
    if (web / "index.html").is_file():
        # Después de las rutas de /api: solo recibe lo que ellas no atienden.
        app.mount("/", StaticFiles(directory=web, html=True), name="web")
    else:
        @app.get("/", include_in_schema=False)
        def sin_frontend() -> None:
            """Avisa de que falta compilar el frontend."""
            raise HTTPException(status_code=404, detail=SIN_FRONTEND)

    return app
