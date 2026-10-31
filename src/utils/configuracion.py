"""Carga y resuelve la configuración común y el perfil de entorno."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Any

import yaml


class TipoEntorno(str, Enum):
    """Entornos activos en el proyecto y el nombre de su perfil YAML."""

    CARTPOLE = "cartpole"
    MOUNTAINCAR = "mountaincar"
    ACROBOT = "acrobot"
    REACHER = "reacher"
    HALFCHEETAH = "halfcheetah"


# Valor provisional de datos.* en el perfil de un entorno cuyo experto y demostraciones
# todavía no están congelados: la configuración se resuelve y se pueden generar, pero el
# experimento completo se niega a empezar hasta que se sustituya por los valores reales.
DATOS_PENDIENTES = "pendiente"


def dato_congelado(cfg: dict, clave: str) -> Any:
    """Devuelve cfg["datos"][clave] si el perfil ya tiene su valor congelado.

    Args:
        cfg: Configuración resuelta.
        clave: Clave de la sección datos (rutas, huellas o artefactos).

    Returns:
        El valor del perfil.

    Raises:
        ValueError: Si el valor sigue siendo DATOS_PENDIENTES.
    """
    valor = cfg["datos"][clave]
    if valor == DATOS_PENDIENTES:
        raise ValueError(
            f"datos.{clave} sigue «{DATOS_PENDIENTES}» en el perfil de "
            f"{cfg['entorno']['id']}: genera y congela el experto y las demostraciones "
            "(ver «Datos» en el README) y pon sus valores en el perfil")
    return valor


def _leer_yaml(ruta: Path) -> dict[str, Any]:
    """Lee un YAML que debe contener un diccionario en su nivel superior."""
    try:
        with ruta.open(encoding="utf-8") as archivo:
            contenido = yaml.safe_load(archivo)
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"No se pudo leer la configuración {ruta}: {error}") from error
    if not isinstance(contenido, dict):
        raise ValueError(f"La configuración {ruta} debe contener un diccionario YAML")
    return contenido


# Secciones que cambian claves de otra solo para una parte del flujo: admiten las claves
# de la sección de referencia (experto.trpo, las de trpo, solo al entrenar el experto).
SOBRESCRITURAS = {"experto.trpo": "trpo"}


def _combinar(base: dict[str, Any], capa: Mapping[str, Any], ruta: str = "",
              raiz: dict[str, Any] | None = None) -> None:
    """Combina una capa en base y rechaza claves que no declare la base.

    En las secciones de SOBRESCRITURAS, las claves válidas son las de su sección de
    referencia en raiz (por defecto, base).
    """
    raiz = base if raiz is None else raiz
    for clave, valor in capa.items():
        nombre = f"{ruta}.{clave}" if ruta else str(clave)
        if clave not in base:
            raise ValueError(f"Clave de configuración desconocida: {nombre}")
        if nombre in SOBRESCRITURAS:
            if not isinstance(valor, Mapping):
                raise ValueError(f"Se esperaba un diccionario en {nombre}")
            referencia = raiz[SOBRESCRITURAS[nombre]]
            for subclave, subvalor in valor.items():
                if subclave not in referencia:
                    raise ValueError(f"Clave de configuración desconocida: {nombre}.{subclave}")
                if isinstance(subvalor, Mapping):
                    raise ValueError("Se esperaba un valor escalar o lista en "
                                     f"{nombre}.{subclave}")
                base[clave][subclave] = deepcopy(subvalor)
            continue
        if isinstance(base[clave], dict):
            if not isinstance(valor, Mapping):
                raise ValueError(f"Se esperaba un diccionario en {nombre}")
            _combinar(base[clave], valor, nombre, raiz)
        elif isinstance(valor, Mapping):
            raise ValueError(f"Se esperaba un valor escalar o lista en {nombre}")
        else:
            base[clave] = deepcopy(valor)


def _buscar_nulos(valor: Any, ruta: str = "") -> list[str]:
    """Devuelve rutas con valores null, recorriendo diccionarios y listas."""
    if isinstance(valor, dict):
        return [ruta_hija for clave, hijo in valor.items()
                for ruta_hija in _buscar_nulos(hijo, f"{ruta}.{clave}" if ruta else str(clave))]
    if isinstance(valor, list):
        return [ruta_hija for indice, hijo in enumerate(valor)
                for ruta_hija in _buscar_nulos(hijo, f"{ruta}[{indice}]")]
    return [ruta] if valor is None else []


def resolver_configuracion(
    ruta_config: Path,
    entorno: TipoEntorno | str,
    ajustes: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Combina base, perfil y ajustes, validando claves y valores pendientes.

    Los ajustes tienen precedencia sobre el perfil. No se modifica ninguno de
    los diccionarios recibidos. El perfil se busca en ``entornos/`` junto al
    archivo base y las rutas relativas de procedencia se guardan en ``origen``.

    Args:
        ruta_config: Ruta al YAML base.
        entorno: Miembro de TipoEntorno o su valor, como ``"cartpole"``.
        ajustes: Capa opcional, por ejemplo las reducciones del modo smoke.

    Returns:
        Configuración completa e independiente de las entradas.

    Raises:
        ValueError: Si el entorno, los archivos, las capas o los valores no
            cumplen el esquema declarado por el YAML base.
    """
    try:
        tipo = entorno if isinstance(entorno, TipoEntorno) else TipoEntorno(entorno)
    except (TypeError, ValueError) as error:
        opciones = ", ".join(miembro.value for miembro in TipoEntorno)
        raise ValueError(f"Entorno desconocido {entorno!r}; Opciones: {opciones}") from error

    ruta_base = Path(ruta_config)
    ruta_perfil = ruta_base.parent / "entornos" / f"{tipo.value}.yaml"
    cfg = _leer_yaml(ruta_base)
    perfil = _leer_yaml(ruta_perfil)
    _combinar(cfg, perfil)
    if ajustes is not None:
        if not isinstance(ajustes, Mapping):
            raise ValueError("ajustes debe ser un mapeo o None")
        _combinar(cfg, ajustes)

    pendientes = _buscar_nulos(cfg)
    if pendientes:
        raise ValueError("Faltan valores de configuración: " + ", ".join(pendientes))
    try:
        ruta_base_registrada = ruta_base.resolve().relative_to(Path.cwd().resolve()).as_posix()
        ruta_perfil_registrada = ruta_perfil.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        ruta_base_registrada = ruta_base.name
        ruta_perfil_registrada = ruta_perfil.name
    cfg["origen"] = {
        "entorno": tipo.value,
        "archivos": [ruta_base_registrada, ruta_perfil_registrada],
    }
    return cfg
