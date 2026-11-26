"""Rutas de la biblioteca: listar y consultar modelos y demostraciones.

- ``GET /biblioteca/modelos?entorno=&tipo=``  Expertos, BC y GAIL.
- ``GET /biblioteca/demostraciones?entorno=``  Conjuntos y subconjuntos de demostraciones.
- ``GET /biblioteca/{id}``                     Un elemento cualquiera (404 si no existe).

Los elementos que no se pueden leer salen con ``valido: false`` y el motivo, también
si sus metadatos no encajan con el modelo de la respuesta: un elemento raro nunca
convierte el listado en un error 500.
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ValidationError

from src.app.contrato import DemostracionApi, ElementoBibliotecaApi
from src.data.biblioteca import TIPOS_MODELO, Biblioteca
from src.utils.configuracion import TipoEntorno


def como_demostracion(elemento: dict[str, Any]) -> dict[str, Any]:
    """Pasa un elemento de demostraciones del índice a la forma de DemostracionApi."""
    detalles = elemento["detalles"]
    return {"id": elemento["id"], "entorno": elemento["entorno"], "nombre": elemento["nombre"],
            "tipo": detalles.get("tipo", "conjunto"), "partidas": detalles.get("partidas"),
            "pares": detalles.get("pares"), "fuente": elemento["fuente"],
            "origen": elemento["origen"], "valido": elemento["valido"],
            "motivo": elemento["motivo"]}


def como_respuesta(modelo: type[BaseModel], datos: dict[str, Any]) -> dict[str, Any]:
    """Los datos si encajan con el modelo de la respuesta; si no, el elemento como no válido.

    Se queda con el id, el entorno y el tipo, que salen de la carpeta, y pone a None lo
    que viene de los metadatos.

    Args:
        modelo: ElementoBibliotecaApi o DemostracionApi.
        datos: El elemento con la forma del modelo.

    Returns:
        Un diccionario que el modelo acepta.
    """
    try:
        return modelo.model_validate(datos).model_dump()
    except ValidationError as error:
        motivo = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}"
                           for e in error.errors())
        vacio = {campo: None for campo in modelo.model_fields}
        respaldo = {**vacio, "id": datos["id"], "entorno": datos["entorno"],
                    "nombre": datos["id"], "valido": False,
                    "motivo": f"metadatos con una forma no válida: {motivo}"}
        respaldo.update({k: datos[k] for k in ("tipo",) if k in vacio})
        if "detalles" in vacio:
            respaldo["detalles"] = {}
        return modelo.model_validate(respaldo).model_dump()


def crear_router_biblioteca(biblioteca: Biblioteca) -> APIRouter:
    """Rutas de la biblioteca.

    Args:
        biblioteca: Índice de la biblioteca.

    Returns:
        Un router con las rutas bajo /biblioteca (se monta con el prefijo /api).
    """
    router = APIRouter(prefix="/biblioteca", tags=["biblioteca"])

    @router.get("/modelos", response_model=list[ElementoBibliotecaApi])
    def modelos(entorno: TipoEntorno | None = None,
                tipo: Literal["experto", "bc", "gail"] | None = None) -> list[dict[str, Any]]:
        """Expertos, BC y GAIL, por entorno, tipo y fecha; con filtros opcionales."""
        tipos = (tipo,) if tipo else TIPOS_MODELO
        return [como_respuesta(ElementoBibliotecaApi, e)
                for e in biblioteca.listar(entorno.value if entorno else None, tipos)]

    @router.get("/demostraciones", response_model=list[DemostracionApi])
    def demostraciones(entorno: TipoEntorno | None = None) -> list[dict[str, Any]]:
        """Conjuntos de demostraciones, con su número de partidas y de pares."""
        return [como_respuesta(DemostracionApi, como_demostracion(e)) for e in
                biblioteca.listar(entorno.value if entorno else None, ("demostraciones",))]

    @router.get("/{id_elemento}", response_model=ElementoBibliotecaApi)
    def elemento(id_elemento: str) -> dict[str, Any]:
        """Un modelo o unas demostraciones por su id."""
        try:
            return como_respuesta(ElementoBibliotecaApi, biblioteca.obtener(id_elemento))
        except KeyError:
            raise HTTPException(status_code=404,
                                detail=f"No existe el elemento {id_elemento}") from None

    return router
