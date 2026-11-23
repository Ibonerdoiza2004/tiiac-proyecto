"""API del dashboard: rutas bajo /api y el frontend compilado en la raíz.

El contrato de la API está en el propio código: los modelos Pydantic de las rutas
generan la documentación HTTP en /docs (OpenAPI). Los errores siguen el formato de
FastAPI, ``{"detail": ...}``, que es el que lee el cliente del frontend.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.utils.configuracion import (TipoEntorno, leer_configuracion_base,
                                     resolver_configuracion)
from src.utils.validacion import es_entero, es_entero_positivo

PREFIJO = "/api"
CONFIG_POR_DEFECTO = Path("configs/default.yaml")
SIN_FRONTEND = ("El frontend no está compilado: ejecuta `npm ci` y `npm run build` "
                "en web/, o usa `npm run dev` con la API en marcha.")
PUERTO_MAXIMO = 65535


class EntornoApi(BaseModel):
    """Un entorno del dashboard: la clave de su perfil y su id de Gymnasium."""

    id: str
    gym: str


class EstadoApi(BaseModel):
    """Respuesta de GET /api/estado."""

    entornos: list[EntornoApi]
    procesos: int


def config_aplicativo(cfg: dict[str, Any]) -> dict[str, Any]:
    """Comprueba y devuelve lo que la API lee de la configuración base.

    Args:
        cfg: YAML base (leer_configuracion_base).

    Returns:
        ``{"host", "puerto", "web", "procesos"}``, con ``web`` como Path.

    Raises:
        ValueError: Si falta alguna clave o su valor no es válido.
    """
    try:
        seccion = cfg["aplicativo"]
        host, puerto, web = seccion["host"], seccion["puerto"], seccion["web"]
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
    if not es_entero_positivo(procesos):
        raise ValueError(f"experimento.procesos debe ser un entero positivo, no {procesos!r}")
    return {"host": host, "puerto": int(puerto), "web": Path(web), "procesos": int(procesos)}


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


def crear_app(ruta_config: Path = CONFIG_POR_DEFECTO) -> FastAPI:
    """Crea la aplicación: la API bajo /api y, si está compilado, el frontend en /.

    La configuración se lee y se comprueba aquí, una vez: si no es válida, la
    aplicación no se crea.

    Args:
        ruta_config: Ruta al YAML base.

    Returns:
        La aplicación FastAPI.

    Raises:
        ValueError: Si la configuración base o algún perfil no son válidos.
    """
    aplicativo = config_aplicativo(leer_configuracion_base(ruta_config))
    estado = EstadoApi(entornos=describir_entornos(ruta_config),
                       procesos=aplicativo["procesos"])

    app = FastAPI(title="GAIL lab", description="API del dashboard de TIIAC.")
    api = APIRouter(prefix=PREFIJO)

    @api.get("/estado", response_model=EstadoApi)
    def leer_estado() -> EstadoApi:
        """Entornos disponibles y cuántos trabajos pueden correr a la vez."""
        return estado

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
