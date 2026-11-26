"""Rutas de la biblioteca: listar y consultar modelos y demostraciones.

- ``GET /biblioteca/modelos?entorno=&tipo=``  Expertos, BC y GAIL.
- ``GET /biblioteca/demostraciones?entorno=``  Conjuntos y subconjuntos de demostraciones.
- ``GET /biblioteca/{id}``                     Un elemento cualquiera (404 si no existe).

Los elementos que no se pueden leer salen con ``valido: false`` y el motivo.
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException

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
        return biblioteca.listar(entorno.value if entorno else None, tipos)

    @router.get("/demostraciones", response_model=list[DemostracionApi])
    def demostraciones(entorno: TipoEntorno | None = None) -> list[dict[str, Any]]:
        """Conjuntos de demostraciones, con su número de partidas y de pares."""
        return [como_demostracion(e) for e in
                biblioteca.listar(entorno.value if entorno else None, ("demostraciones",))]

    @router.get("/{id_elemento}", response_model=ElementoBibliotecaApi)
    def elemento(id_elemento: str) -> dict[str, Any]:
        """Un modelo o unas demostraciones por su id."""
        try:
            return biblioteca.obtener(id_elemento)
        except KeyError:
            raise HTTPException(status_code=404,
                                detail=f"No existe el elemento {id_elemento}") from None

    return router
