"""Hiperparámetros que el dashboard deja cambiar: su esquema y la validación de ajustes.

El esquema sale de la configuración resuelta del entorno (valores del perfil) y de
``configs/dashboard.yaml`` (textos, nivel, rangos y opciones). Los ajustes que manda
el formulario son una capa más sobre el perfil, anidada como en ``configs/``: aquí se
comprueba que solo toquen campos del método y que cada valor tenga su tipo y su
rango; después se resuelve la configuración con ellos (``resolver_configuracion``) y
se le aplican las reglas del entrenador (``comprobar_config_metodo``), las mismas que
comprueban los scripts antes de entrenar, como que los entornos repartan los pasos de
cada iteración.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from src.app.contrato import EsquemaAjustesApi
from src.utils.configuracion import TipoEntorno, leer_configuracion_base, resolver_configuracion
from src.utils.reglas_config import METODOS, comprobar_calidad, comprobar_config_metodo
from src.utils.validacion import es_entero, es_real_finito

NIVELES = ("basico", "avanzado")
TIPOS = ("entero", "real", "booleano", "opcion", "lista_enteros")
# Claves de cada sección que no son hiperparámetros del método o se ajustan aparte
# (la semilla va en su propio campo; experto.trpo y gail.discriminador, en su sección).
EXCLUIDAS = {"experto": {"semilla", "trpo"}, "gail": {"discriminador"}}
DURACION = {"experto": ("pasos", ("experto", "pasos_totales")),
            "bc": ("epocas", ("bc", "epocas_maximas")),
            "gail": ("iteraciones", ("gail", "iteraciones"))}
# Campo del formulario al que se redondea la duración (pasos_reales_experto).
BLOQUE_EXPERTO = "experto.trpo.pasos_por_iteracion"
DESCRIPCION_CAMPO = {"etiqueta", "ayuda", "nivel", "tipo", "min", "max", "paso"}


def cargar_dashboard(ruta: Path) -> dict[str, Any]:
    """Lee y comprueba ``configs/dashboard.yaml``.

    Args:
        ruta: Ruta al YAML.

    Returns:
        Su contenido: laboratorio, opciones, secciones y campos.

    Raises:
        ValueError: Si falta algo o alguna descripción no es válida.
    """
    datos = leer_configuracion_base(Path(ruta))
    for clave in ("laboratorio", "vista", "opciones", "secciones", "campos"):
        if not isinstance(datos.get(clave), dict):
            raise ValueError(f"{ruta}: falta la sección {clave}")
    laboratorio = datos["laboratorio"]
    validos = {t.value for t in TipoEntorno}
    jugar_a_mano = laboratorio.get("jugar_a_mano")
    if (not isinstance(jugar_a_mano, list)
            or not all(isinstance(e, str) and e in validos for e in jugar_a_mano)):
        raise ValueError(f"{ruta}: laboratorio.jugar_a_mano solo admite {sorted(validos)}")
    if not es_entero(laboratorio.get("partidas_maximas")) or laboratorio["partidas_maximas"] < 1:
        raise ValueError(f"{ruta}: laboratorio.partidas_maximas debe ser un entero positivo")
    try:
        comprobar_calidad(datos["vista"].get("calidad_jpeg"))
    except ValueError as error:
        raise ValueError(f"{ruta}: vista.calidad_jpeg: {error}") from None
    grupo = datos.get("wandb", {}).get("grupo") if isinstance(datos.get("wandb"), dict) else None
    try:
        valido = isinstance(grupo, str) and bool(re.fullmatch(r"[A-Za-z0-9_.-]+",
                                                              grupo.format(entorno="x")))
    except (KeyError, IndexError, ValueError):
        valido = False
    if not valido:
        raise ValueError(f"{ruta}: wandb.grupo debe ser un nombre de grupo de W&B, con "
                         "{entorno} si se quiere el del entorno")
    for seccion, titulo in datos["secciones"].items():
        if not isinstance(titulo, str) or not titulo.strip():
            raise ValueError(f"{ruta}: el título de la sección {seccion} debe ser un texto")
    for seccion, campos in datos["campos"].items():
        if seccion not in datos["secciones"]:
            raise ValueError(f"{ruta}: la sección {seccion} no tiene título en secciones")
        for clave, descripcion in campos.items():
            _comprobar_descripcion(f"{ruta}: {seccion}.{clave}", descripcion)
    return datos


def _comprobar_descripcion(donde: str, descripcion: Any) -> None:
    """Comprueba la descripción de un campo de dashboard.yaml."""
    if not isinstance(descripcion, dict) or not descripcion.keys() <= DESCRIPCION_CAMPO:
        raise ValueError(f"{donde}: solo admite {sorted(DESCRIPCION_CAMPO)}")
    for texto in ("etiqueta", "ayuda"):
        if not isinstance(descripcion.get(texto), str) or not descripcion[texto].strip():
            raise ValueError(f"{donde}: falta {texto}")
    if descripcion.get("nivel") not in NIVELES:
        raise ValueError(f"{donde}: nivel debe ser uno de {NIVELES}")
    if "tipo" in descripcion and descripcion["tipo"] not in TIPOS:
        raise ValueError(f"{donde}: tipo debe ser uno de {TIPOS}")
    for limite in ("min", "max", "paso"):
        if limite in descripcion and not es_real_finito(descripcion[limite]):
            raise ValueError(f"{donde}: {limite} debe ser un número")


def secciones_de_metodo(cfg: dict[str, Any], metodo: str) -> list[tuple[str, dict[str, Any]]]:
    """Secciones de la configuración que el formulario muestra para un método.

    Args:
        cfg: Configuración resuelta del entorno.
        metodo: experto, bc o gail.

    Returns:
        Pares (clave de la sección como en configs/, valores del perfil). En el
        experto, ``experto.trpo`` lleva todos los valores de ``trpo`` con los suyos
        encima, porque son los que usa al entrenar.

    Raises:
        ValueError: Si el método no existe.
    """
    def sin(seccion: str) -> dict[str, Any]:
        """La sección sin sus claves excluidas."""
        return {k: v for k, v in cfg[seccion].items() if k not in EXCLUIDAS.get(seccion, set())}

    frecuencia = {"frecuencia_pares": cfg["demostraciones"]["frecuencia_pares"]}
    if metodo == "experto":
        return [("experto", sin("experto")),
                ("experto.trpo", {**cfg["trpo"], **cfg["experto"]["trpo"]})]
    if metodo == "bc":
        return [("bc", cfg["bc"]), ("demostraciones", frecuencia)]
    if metodo == "gail":
        return [("gail", sin("gail")), ("gail.discriminador", cfg["gail"]["discriminador"]),
                ("trpo", cfg["trpo"]), ("demostraciones", frecuencia)]
    raise ValueError(f"Método desconocido {metodo!r}; opciones: {', '.join(METODOS)}")


def _descripcion(dashboard: dict[str, Any], seccion: str, clave: str) -> dict[str, Any]:
    """Descripción de un campo; experto.trpo usa las de trpo."""
    campos = dashboard["campos"].get("trpo" if seccion == "experto.trpo" else seccion, {})
    if clave not in campos:
        raise ValueError(f"configs/dashboard.yaml no describe {seccion}.{clave}")
    return campos[clave]


def _opciones(dashboard: dict[str, Any], seccion: str, clave: str) -> list[str] | None:
    """Valores admitidos de un campo de texto (por sección y clave, o solo por clave)."""
    opciones = dashboard["opciones"]
    return opciones.get(f"{seccion}.{clave}", opciones.get(clave))


def _tipo(valor: Any, descripcion: dict[str, Any], opciones: list[str] | None) -> str:
    """Tipo del campo: el de la descripción o el que se deduce del valor del perfil."""
    if "tipo" in descripcion:
        return descripcion["tipo"]
    if isinstance(valor, bool):
        return "booleano"
    if es_entero(valor):
        return "entero"
    if isinstance(valor, float):
        return "real"
    if isinstance(valor, list) and all(es_entero(v) for v in valor):
        return "lista_enteros"
    if isinstance(valor, str) and opciones:
        return "opcion"
    raise ValueError(f"No se puede deducir el tipo del valor {valor!r}")


def campo(dashboard: dict[str, Any], seccion: str, clave: str, valor: Any) -> dict[str, Any]:
    """Describe un hiperparámetro para el formulario.

    Args:
        dashboard: Contenido de dashboard.yaml (cargar_dashboard).
        seccion: Clave de la sección como en configs/ (por ejemplo gail.discriminador).
        clave: Nombre del hiperparámetro.
        valor: Su valor en el perfil.

    Returns:
        Un diccionario con la forma de CampoAjusteApi.

    Raises:
        ValueError: Si dashboard.yaml no lo describe o no se puede saber su tipo.
    """
    descripcion = _descripcion(dashboard, seccion, clave)
    opciones = _opciones(dashboard, seccion, clave)
    tipo = _tipo(valor, descripcion, opciones)
    return {"clave": clave, "tipo": tipo, "valor": float(valor) if tipo == "real" else valor,
            "etiqueta": descripcion["etiqueta"], "ayuda": descripcion["ayuda"],
            "nivel": descripcion["nivel"], "min": descripcion.get("min"),
            "max": descripcion.get("max"), "paso": descripcion.get("paso"),
            "opciones": opciones if tipo == "opcion" else None}


def semilla_por_defecto(cfg: dict[str, Any], metodo: str) -> int:
    """Semilla de un entrenamiento suelto si el usuario no la cambia.

    El experto usa ``experto.semilla``; BC y GAIL, la base de las repeticiones
    (``experimento.semilla_repeticiones``), de la que el trabajador sacará las de la
    repetición 0 con ``semillas_repeticion``, como en la E1.
    """
    return int(cfg["experto"]["semilla"] if metodo == "experto"
               else cfg["experimento"]["semilla_repeticiones"])


def pasos_reales_experto(cfg: dict[str, Any]) -> int:
    """Pasos que da como mucho el experto: bloques completos hasta pasar el presupuesto.

    TRPO recoge cada iteración un bloque entero de ``trpo.pasos_por_iteracion`` (con los
    cambios de ``experto.trpo``) y entrenar_experto para en el primero que llega a
    ``experto.pasos_totales``: con 25 000 000 y bloques de 2048, 25 001 984.

    Args:
        cfg: Configuración resuelta del entorno.

    Returns:
        El múltiplo de pasos_por_iteracion igual o inmediatamente superior al presupuesto.
    """
    presupuesto = int(cfg["experto"]["pasos_totales"])
    bloque = int({**cfg["trpo"], **cfg["experto"].get("trpo", {})}["pasos_por_iteracion"])
    return -(-presupuesto // bloque) * bloque


def esquema_ajustes(ruta_config: Path, entorno: TipoEntorno | str, metodo: str,
                    dashboard: dict[str, Any], ajustes: dict[str, Any] | None = None
                    ) -> dict[str, Any]:
    """Esquema del formulario de un método en un entorno, con los valores del perfil.

    Args:
        ruta_config: Ruta al YAML base.
        entorno: Entorno.
        metodo: experto, bc o gail.
        dashboard: Contenido de dashboard.yaml.
        ajustes: Capa opcional sobre el perfil (para calcular la duración pedida).

    Returns:
        Un diccionario con la forma de EsquemaAjustesApi.

    Raises:
        ValueError: Si el método, la configuración o dashboard.yaml no valen.
    """
    cfg = resolver_configuracion(ruta_config, entorno, ajustes)
    secciones = [{"clave": seccion, "titulo": dashboard["secciones"][seccion],
                  "campos": [campo(dashboard, seccion, clave, valor)
                             for clave, valor in valores.items()]}
                 for seccion, valores in secciones_de_metodo(cfg, metodo)]
    unidad, (seccion, clave) = DURACION[metodo]
    duracion = {"unidad": unidad, "total": int(cfg[seccion][clave]),
                "campo": f"{seccion}.{clave}"}
    if metodo == "experto":
        duracion.update(total=pasos_reales_experto(cfg), bloque=BLOQUE_EXPERTO)
    return {"metodo": metodo, "duracion": duracion,
            "semilla": semilla_por_defecto(cfg, metodo), "secciones": secciones}


def _hojas(ajustes: Any, prefijo: str = "") -> list[tuple[str, Any]]:
    """Pares (ruta con puntos, valor) de las hojas de unos ajustes anidados."""
    if not isinstance(ajustes, dict):
        raise ValueError(f"Los ajustes{' de ' + prefijo if prefijo else ''} deben ser un "
                         "diccionario anidado como en configs/")
    hojas = []
    for clave, valor in ajustes.items():
        ruta = f"{prefijo}.{clave}" if prefijo else str(clave)
        hojas.extend(_hojas(valor, ruta) if isinstance(valor, dict) else [(ruta, valor)])
    return hojas


def _valor_valido(campo_esquema: dict[str, Any], valor: Any) -> str | None:
    """Motivo por el que el valor no vale para el campo, o None si vale."""
    tipo, minimo, maximo = campo_esquema["tipo"], campo_esquema["min"], campo_esquema["max"]
    if tipo == "booleano":
        return None if isinstance(valor, bool) else "tiene que ser true o false"
    if tipo == "opcion":
        opciones = campo_esquema["opciones"]
        return None if valor in opciones else f"tiene que ser una de {opciones}"
    if tipo == "lista_enteros":
        if not isinstance(valor, list) or not valor or not all(es_entero(v) for v in valor):
            return "tiene que ser una lista de enteros no vacía"
        numeros = valor
    elif tipo == "entero":
        if not es_entero(valor):
            return "tiene que ser un entero"
        numeros = [valor]
    else:
        if not es_real_finito(valor):
            return "tiene que ser un número finito"
        numeros = [valor]
    if minimo is not None and any(n < minimo for n in numeros):
        return f"tiene que ser como mínimo {minimo:g}"
    if maximo is not None and any(n > maximo for n in numeros):
        return f"tiene que ser como mucho {maximo:g}"
    return None


def comprobar_ajustes(ruta_config: Path, entorno: TipoEntorno | str, metodo: str,
                      ajustes: dict[str, Any], dashboard: dict[str, Any]) -> dict[str, Any]:
    """Comprueba los ajustes de un entrenamiento y devuelve la configuración resuelta.

    Solo se admiten los campos que el formulario ofrece para ese método, con un valor
    de su tipo dentro de su rango; después la configuración se resuelve con ellos y se
    comprueba con las reglas del entrenador (comprobar_config_metodo), que también
    miran combinaciones de campos.

    Args:
        ruta_config: Ruta al YAML base.
        entorno: Entorno.
        metodo: experto, bc o gail.
        ajustes: Lo que cambia el usuario, anidado como en configs/.
        dashboard: Contenido de dashboard.yaml.

    Returns:
        La configuración resuelta con los ajustes.

    Raises:
        ValueError: Con todos los motivos si algún ajuste no vale.
    """
    esquema = esquema_ajustes(ruta_config, entorno, metodo, dashboard)
    campos = {f"{s['clave']}.{c['clave']}": c for s in esquema["secciones"] for c in s["campos"]}
    errores = []
    for ruta, valor in _hojas(ajustes):
        if ruta not in campos:
            errores.append(f"{ruta}: no es un ajuste de {metodo}")
        elif (motivo := _valor_valido(campos[ruta], valor)) is not None:
            errores.append(f"{ruta}: {motivo}")
    if errores:
        raise ValueError("Ajustes no válidos: " + "; ".join(errores))
    cfg = resolver_configuracion(ruta_config, entorno, ajustes)
    try:
        comprobar_config_metodo(cfg, metodo)
    except ValueError as error:
        raise ValueError(f"Ajustes no válidos: {error}") from error
    return cfg


def comprobar_esquemas(ruta_config: Path, dashboard: dict[str, Any]) -> None:
    """Genera el esquema de cada entorno y método, para no descubrir un fallo al pedirlo.

    Lo llama la API al crearse: si dashboard.yaml no describe algún campo, un perfil
    no vale o el esquema no encaja con el modelo de la respuesta (EsquemaAjustesApi),
    no arranca (en lugar de responder 500 al abrir ese formulario).

    Raises:
        ValueError: Con el entorno, el método y el motivo del primero que falle.
    """
    for entorno in TipoEntorno:
        for metodo in METODOS:
            try:
                EsquemaAjustesApi.model_validate(
                    esquema_ajustes(ruta_config, entorno, metodo, dashboard))
            except (ValueError, KeyError, TypeError) as error:
                raise ValueError(f"Formulario de {metodo} en {entorno.value}: {error}") from error
