"""Biblioteca del dashboard: modelos y demostraciones guardados en disco, y su índice.

Cada elemento es una carpeta ``<raiz>/<entorno>/<tipo>/<id>/`` con sus archivos de
siempre y una ficha (``ficha.json``) con lo que esos archivos no dicen: el nombre
para el usuario, de dónde viene (congelado, entrega o dashboard) y cuándo se creó.

| Tipo             | Archivos                                          |
|------------------|---------------------------------------------------|
| ``experto``      | ``modelo.zip``, ``experto.json``                  |
| ``gail``         | ``modelo.zip``, ``discriminador.pt``, ``gail.json`` |
| ``bc``           | ``politica.pt`` (con sus metadatos dentro)        |
| ``demostraciones`` | ``demostraciones.npz`` y, si los hay, ``splits.json`` |

Un subconjunto de demostraciones (src/data/seleccion.py) no tiene NPZ: su carpeta
lleva ``subconjunto.json``, que dice de qué conjunto sale y qué partidas toma.

El índice no se guarda en ningún archivo común: se reconstruye leyendo las
carpetas cada vez, así que dos trabajos que terminan a la vez no se pisan (cada uno
escribe su carpeta nueva y la renombra al final). Una carpeta que no se puede leer
sale en el índice como no válida, con el motivo.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.archivos import escribir_carpeta_nueva
from src.utils.configuracion import TipoEntorno

FORMATO_FICHA = "tiiac-biblioteca"
VERSION_FICHA = 1
ARCHIVO_FICHA = "ficha.json"
# Un subconjunto no tiene NPZ: este archivo dice de qué conjunto sale y qué toma.
ARCHIVO_SUBCONJUNTO = "subconjunto.json"
TIPOS_MODELO = ("experto", "bc", "gail")
TIPOS = (*TIPOS_MODELO, "demostraciones")
ORIGENES = ("congelado", "entrega", "dashboard")
FUENTES = ("experto", "humano", "combinado")
ARCHIVOS = {"experto": ("modelo.zip", "experto.json"),
            "gail": ("modelo.zip", "discriminador.pt", "gail.json"),
            "bc": ("politica.pt",),
            "demostraciones": ("demostraciones.npz",)}
CARACTERES_ID = 12
_ID = re.compile(rf"^[0-9a-f]{{{CARACTERES_ID}}}$")
# Reservas de ids: un archivo vacío por id en uso. Crearlo es atómico (O_EXCL), así que
# dos procesos nunca se quedan con el mismo id, aunque sean de entornos o tipos distintos.
CARPETA_IDS = ".ids"
# Ids nuevos que prueba agregar_nuevo antes de rendirse si salen repetidos.
INTENTOS_ID = 5
# Checkpoints de los entrenamientos en curso (src/training/checkpoints.py), una carpeta
# por entrenamiento. Es oculta, así que el índice no la lee.
CARPETA_EN_CURSO = ".en_curso"


class IdRepetido(FileExistsError):
    """El id ya lo usa otro elemento de la biblioteca (de cualquier entorno o tipo)."""


def nuevo_id() -> str:
    """Identificador nuevo de un elemento: 12 caracteres hexadecimales al azar.

    No depende de ninguna semilla a propósito: solo tiene que ser único, no
    reproducible, y no toca el estado aleatorio de los experimentos.
    """
    return secrets.token_hex(CARACTERES_ID // 2)


def nombre_por_defecto(tipo: str, gym: str, momento: datetime | None = None) -> str:
    """Nombre de un modelo cuando el usuario no pone ninguno.

    Args:
        tipo: experto, bc o gail.
        gym: Id de Gymnasium del entorno (CartPole-v1).
        momento: Fecha y hora (por defecto, ahora, en la hora local).

    Returns:
        Por ejemplo «GAIL · CartPole · 24/11 18:30».
    """
    momento = momento or datetime.now().astimezone()
    etiqueta = "Experto" if tipo == "experto" else tipo.upper()
    return f"{etiqueta} · {gym.split('-v')[0]} · {momento:%d/%m %H:%M}"


def es_id(valor: object) -> bool:
    """Indica si el valor tiene la forma de un id de la biblioteca."""
    return isinstance(valor, str) and bool(_ID.match(valor))


def _ahora() -> str:
    """Fecha y hora actual en UTC, en ISO 8601."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def crear_ficha(id_elemento: str, entorno: str, tipo: str, nombre: str, origen: str,
                **extra: Any) -> dict[str, Any]:
    """Ficha de un elemento nueva y comprobada.

    Args:
        id_elemento: Id (nuevo_id).
        entorno: Clave del entorno (cartpole, reacher…).
        tipo: Uno de TIPOS.
        nombre: Nombre para el usuario.
        origen: Uno de ORIGENES.
        **extra: Campos opcionales: descripcion, fuente (demostraciones), artefacto
            (nombre y versión en W&B) y cualquier dato JSON más.

    Returns:
        La ficha.

    Raises:
        ValueError: Si algún campo no vale.
    """
    ficha = {"formato": FORMATO_FICHA, "version": VERSION_FICHA, "id": id_elemento,
             "entorno": entorno, "tipo": tipo, "nombre": nombre, "origen": origen,
             "creado": extra.pop("creado", _ahora()), **extra}
    comprobar_ficha(ficha)
    return ficha


def comprobar_ficha(ficha: Any, entorno: str | None = None, tipo: str | None = None,
                    id_elemento: str | None = None) -> None:
    """Comprueba una ficha y, si se dan, que coincida con su carpeta.

    Raises:
        ValueError: Con el motivo si no vale.
    """
    if not isinstance(ficha, dict):
        raise ValueError("la ficha no es un objeto JSON")
    if ficha.get("formato") != FORMATO_FICHA or ficha.get("version") != VERSION_FICHA:
        raise ValueError(f"la ficha no es de formato {FORMATO_FICHA} versión {VERSION_FICHA}")
    if not es_id(ficha.get("id")):
        raise ValueError(f"id no válido: {ficha.get('id')!r}")
    if ficha.get("tipo") not in TIPOS:
        raise ValueError(f"tipo no válido: {ficha.get('tipo')!r}")
    if ficha.get("origen") not in ORIGENES:
        raise ValueError(f"origen no válido: {ficha.get('origen')!r}")
    for campo in ("entorno", "nombre", "creado"):
        if not isinstance(ficha.get(campo), str) or not ficha[campo].strip():
            raise ValueError(f"falta {campo}")
    entornos = [t.value for t in TipoEntorno]
    if ficha["entorno"] not in entornos:
        raise ValueError(f"entorno no válido: {ficha['entorno']!r}; opciones: {entornos}")
    if "fuente" in ficha and ficha["fuente"] not in FUENTES:
        raise ValueError(f"fuente no válida: {ficha['fuente']!r}")
    for campo in ("descripcion", "artefacto"):
        if ficha.get(campo) is not None and not isinstance(ficha[campo], str):
            raise ValueError(f"{campo} debe ser un texto, no {ficha[campo]!r}")
    for campo, esperado in (("entorno", entorno), ("tipo", tipo), ("id", id_elemento)):
        if esperado is not None and ficha[campo] != esperado:
            raise ValueError(f"la ficha dice {campo} {ficha[campo]!r} pero la carpeta es "
                             f"de {esperado!r}")


def _id_en_uso(raiz: Path, id_elemento: str) -> bool:
    """Si alguna carpeta de la biblioteca, de cualquier entorno o tipo, ya usa el id."""
    if not raiz.is_dir():
        return False
    return any((entorno / tipo / id_elemento).exists()
               for entorno in raiz.iterdir()
               if entorno.is_dir() and not entorno.name.startswith(".")
               for tipo in TIPOS)


def _reservar_id(raiz: Path, id_elemento: str) -> None:
    """Reserva el id en toda la biblioteca, de forma atómica.

    Raises:
        IdRepetido: Si ya está reservado o alguna carpeta lo usa.
    """
    reservas = raiz / CARPETA_IDS
    reservas.mkdir(parents=True, exist_ok=True)
    try:
        os.close(os.open(reservas / id_elemento, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:
        raise IdRepetido(f"El id {id_elemento} ya está en uso en la biblioteca") from None
    # También los elementos que se añadieron sin reserva (a mano o antes de existir esta).
    if _id_en_uso(raiz, id_elemento):
        _liberar_id(raiz, id_elemento)
        raise IdRepetido(f"El id {id_elemento} ya está en uso en la biblioteca")


def _liberar_id(raiz: Path, id_elemento: str) -> None:
    """Quita la reserva de un id (por ejemplo, si no se llegó a guardar el elemento)."""
    (raiz / CARPETA_IDS / id_elemento).unlink(missing_ok=True)


def agregar(raiz: Path, ficha: Mapping[str, Any], escribir: Callable[[Path], None]) -> Path:
    """Añade un elemento a la biblioteca de forma atómica.

    Primero reserva su id en toda la biblioteca (ningún otro elemento, de ningún
    entorno ni tipo, puede tenerlo). Después ``escribir(carpeta)`` deja los archivos
    del elemento en una carpeta temporal; si termina bien, se añade la ficha y la
    carpeta se renombra a su sitio. Si algo falla, no queda nada a medias y el id
    vuelve a quedar libre.

    Args:
        raiz: Carpeta de la biblioteca.
        ficha: Ficha del elemento (crear_ficha).
        escribir: Escribe los archivos del elemento en la carpeta que recibe.

    Returns:
        La carpeta del elemento.

    Raises:
        ValueError: Si la ficha no vale o faltan los archivos de su tipo.
        IdRepetido: Si otro elemento de la biblioteca ya usa ese id.
    """
    comprobar_ficha(dict(ficha))
    raiz = Path(raiz)
    destino = raiz / ficha["entorno"] / ficha["tipo"] / ficha["id"]

    def escribir_con_ficha(carpeta: Path) -> None:
        """Escribe los archivos, comprueba que estén y añade la ficha."""
        escribir(carpeta)
        faltan = [a for a in archivos_de(ficha["tipo"], carpeta) if not (carpeta / a).is_file()]
        if faltan:
            raise ValueError(f"Faltan archivos de un elemento {ficha['tipo']}: {faltan}")
        (carpeta / ARCHIVO_FICHA).write_text(
            json.dumps(dict(ficha), ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8")

    _reservar_id(raiz, ficha["id"])
    try:
        escribir_carpeta_nueva(destino, escribir_con_ficha)
    except BaseException:
        _liberar_id(raiz, ficha["id"])
        raise
    return destino


def agregar_nuevo(raiz: Path, entorno: str, tipo: str, nombre: str, origen: str,
                  escribir: Callable[[Path], None], **extra: Any) -> dict[str, Any]:
    """Crea la ficha con un id nuevo y añade el elemento; si el id sale repetido, prueba otro.

    Args:
        raiz: Carpeta de la biblioteca.
        entorno: Clave del entorno.
        tipo: Uno de TIPOS.
        nombre: Nombre para el usuario.
        origen: Uno de ORIGENES.
        escribir: Escribe los archivos del elemento (como en agregar).
        **extra: Campos opcionales de la ficha (crear_ficha).

    Returns:
        La ficha del elemento añadido.

    Raises:
        IdRepetido: Si salen repetidos los INTENTOS_ID ids probados.
        Lo mismo que agregar.
    """
    for _ in range(INTENTOS_ID):
        ficha = crear_ficha(nuevo_id(), entorno, tipo, nombre, origen, **extra)
        try:
            agregar(raiz, ficha, escribir)
        except IdRepetido:
            continue
        return ficha
    raise IdRepetido(f"Salieron repetidos {INTENTOS_ID} ids seguidos")


def _leer_json(ruta: Path) -> Any:
    """JSON estricto de un archivo (ValueError si no se puede leer)."""
    try:
        return json.loads(ruta.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"no se pudo leer {ruta.name}: {error}") from error


def _objeto(valor: Any, donde: str) -> dict[str, Any]:
    """El valor si es un objeto JSON (diccionario); si no, ValueError con dónde está."""
    if not isinstance(valor, dict):
        raise ValueError(f"{donde} no es un objeto JSON")
    return valor


def _detalles_experto(carpeta: Path, gym: str) -> dict[str, Any]:
    """Lo principal de experto.json."""
    datos = _objeto(_leer_json(carpeta / "experto.json"), "experto.json")
    resumen = _objeto(datos.get("resumen", {}), "el resumen de experto.json")
    if resumen.get("entorno") != gym:
        raise ValueError(f"experto.json es de {resumen.get('entorno')!r}, no de {gym}")
    return {clave: resumen.get(clave) for clave in
            ("pasos", "iteraciones", "mejor_retorno", "objetivo", "objetivo_alcanzado",
             "segundos")}


def _detalles_gail(carpeta: Path, gym: str) -> dict[str, Any]:
    """Lo principal de gail.json."""
    datos = _objeto(_leer_json(carpeta / "gail.json"), "gail.json")
    resumen = _objeto(datos.get("resumen", {}), "el resumen de gail.json")
    procedencia = _objeto(datos.get("datos", {}), "los datos de gail.json")
    if resumen.get("entorno") != gym:
        raise ValueError(f"gail.json es de {resumen.get('entorno')!r}, no de {gym}")
    return {"cantidad": resumen.get("cantidad"), "repeticion": resumen.get("repeticion"),
            "iteraciones": resumen.get("iteraciones"), "pasos": resumen.get("pasos"),
            "segundos": resumen.get("segundos"),
            "huella_dataset": procedencia.get("huella_dataset")}


def _detalles_bc(carpeta: Path, gym: str) -> dict[str, Any]:
    """Lo principal de los metadatos guardados dentro de politica.pt."""
    import torch  # noqa: PLC0415 - solo si hay algún BC que leer

    try:
        carga = torch.load(carpeta / "politica.pt", map_location="cpu", weights_only=True)
    except Exception as error:  # noqa: BLE001 - torch lanza varios tipos según el fallo
        raise ValueError(f"no se pudo leer politica.pt: {error}") from error
    metadatos = carga.get("metadatos") if isinstance(carga, dict) else None
    if not isinstance(metadatos, dict):
        raise ValueError("politica.pt no tiene metadatos")
    if metadatos.get("entorno") != gym:
        raise ValueError(f"politica.pt es de {metadatos.get('entorno')!r}, no de {gym}")
    adicional = _objeto(metadatos.get("adicional", {}), "adicional de politica.pt")
    resumen = _objeto(adicional.get("resumen", {}), "el resumen de politica.pt")
    return {"subconjunto": metadatos.get("subconjunto"), "semilla": metadatos.get("semilla"),
            "repeticion": resumen.get("repeticion"), "segundos": resumen.get("segundos"),
            "huella_dataset": metadatos.get("huella_dataset")}


def _detalles_demostraciones(carpeta: Path, gym: str) -> dict[str, Any]:
    """Partidas, pares y fuente del manifiesto del NPZ; sus splits, si los tiene, tienen
    que ser de ese NPZ y de sus partidas (si no, el conjunto no es válido: entrenar con él
    fallaría)."""
    from src.data.demostraciones import (fuente_demostraciones,  # noqa: PLC0415
                                         resumen_demostraciones)
    from src.data.splits import cargar_splits, huella_archivo  # noqa: PLC0415

    resumen = resumen_demostraciones(carpeta / "demostraciones.npz")
    entorno = resumen["metadatos"].get("entorno")
    if entorno != gym:
        raise ValueError(f"demostraciones.npz es de {entorno!r}, no de {gym}")
    for clave in ("partidas", "pares"):
        if not isinstance(resumen[clave], int) or resumen[clave] < 1:
            raise ValueError(f"demostraciones.npz no tiene {clave}: {resumen[clave]!r}")
    splits = carpeta / "splits.json"
    if splits.is_file():
        try:
            cargar_splits(splits, huella_archivo(carpeta / "demostraciones.npz"),
                          list(resumen["pasos_por_partida"]))
        except (OSError, ValueError) as error:
            raise ValueError(f"splits.json no corresponde a su NPZ: {error}") from None
    return {"tipo": "conjunto", "partidas": resumen["partidas"], "pares": resumen["pares"],
            "fuente": fuente_demostraciones(resumen["metadatos"]), "splits": splits.is_file()}


def es_subconjunto(carpeta: Path) -> bool:
    """Si la carpeta de unas demostraciones es un subconjunto (tiene subconjunto.json)."""
    return (Path(carpeta) / ARCHIVO_SUBCONJUNTO).is_file()


def archivos_de(tipo: str, carpeta: Path) -> tuple[str, ...]:
    """Archivos que tiene que tener un elemento: los de su tipo, salvo un subconjunto."""
    if tipo == "demostraciones" and es_subconjunto(carpeta):
        return (ARCHIVO_SUBCONJUNTO,)
    return ARCHIVOS[tipo]


DETALLES: dict[str, Callable[[Path, str], dict[str, Any]]] = {
    "experto": _detalles_experto, "gail": _detalles_gail, "bc": _detalles_bc,
    "demostraciones": _detalles_demostraciones}


class Biblioteca:
    """Índice de la biblioteca, leído de las carpetas cada vez que se pide."""

    def __init__(self, raiz: Path, entornos: Mapping[str, str]):
        """Prepara el índice.

        Args:
            raiz: Carpeta de la biblioteca (puede no existir todavía).
            entornos: Clave de cada entorno → su id de Gymnasium, para comprobar que
                los metadatos de cada elemento son de su entorno.
        """
        self.raiz = Path(raiz)
        self.entornos = dict(entornos)

    def _leer(self, carpeta: Path, entorno: str, tipo: str) -> dict[str, Any]:
        """Elemento de una carpeta; si algo falla, no válido con el motivo."""
        elemento: dict[str, Any] = {"id": carpeta.name, "entorno": entorno, "tipo": tipo,
                                    "nombre": carpeta.name, "origen": None, "creado": None,
                                    "descripcion": None, "fuente": None, "artefacto": None,
                                    "valido": False,
                                    "motivo": None, "detalles": {}}
        try:
            ficha = _leer_json(carpeta / ARCHIVO_FICHA)
            comprobar_ficha(ficha, entorno, tipo, carpeta.name)
            elemento.update({clave: ficha.get(clave) for clave in
                             ("nombre", "origen", "creado", "descripcion", "fuente",
                              "artefacto")})
            faltan = [a for a in archivos_de(tipo, carpeta) if not (carpeta / a).is_file()]
            if faltan:
                raise ValueError(f"faltan {', '.join(faltan)}")
            subconjunto = tipo == "demostraciones" and es_subconjunto(carpeta)
            if subconjunto:
                elemento["detalles"] = {"tipo": "subconjunto"}
                elemento["detalles"] = self._detalles_subconjunto(carpeta, entorno)
            else:
                elemento["detalles"] = DETALLES[tipo](carpeta, self.entornos[entorno])
            if tipo == "demostraciones":
                # La fuente la dicen los datos (el NPZ o el de su conjunto); la ficha, si
                # la da, tiene que coincidir.
                fuente = elemento["detalles"]["fuente"]
                if elemento["fuente"] not in (None, fuente):
                    datos = "su conjunto" if subconjunto else "el NPZ"
                    raise ValueError(f"la ficha dice fuente {elemento['fuente']!r} y {datos}, "
                                     f"{fuente!r}")
                elemento["fuente"] = fuente
            elemento["valido"] = True
        except ValueError as error:
            elemento["motivo"] = str(error)
        except Exception as error:  # noqa: BLE001 - un elemento roto no tumba el índice
            elemento["motivo"] = f"no se pudo leer: {error!r}"
        return elemento

    def _detalles_subconjunto(self, carpeta: Path, entorno: str) -> dict[str, Any]:
        """Detalles de un subconjunto, comprobado contra su conjunto tal como está ahora.

        Raises:
            ValueError: Si su conjunto no existe, es de otro entorno, es otro subconjunto,
                no es válido o ha cambiado, o no tiene sus partidas.
        """
        from src.data.demostraciones import resumen_demostraciones  # noqa: PLC0415
        from src.data.seleccion import (comprobar_contra_conjunto,  # noqa: PLC0415
                                        leer_subconjunto)
        from src.data.splits import huella_archivo  # noqa: PLC0415

        datos = leer_subconjunto(carpeta)
        id_conjunto = datos["conjunto"]
        try:
            origen = self.ruta(id_conjunto)
        except KeyError:
            raise ValueError(f"no existe su conjunto {id_conjunto}") from None
        if origen.parent.name != "demostraciones" or origen.parent.parent.name != entorno:
            raise ValueError(f"{id_conjunto} no son demostraciones de {entorno}")
        # Antes de leerlo: un subconjunto de otro (o de sí mismo) no se sigue.
        if es_subconjunto(origen):
            raise ValueError(f"{id_conjunto} es un subconjunto: solo se saca de un conjunto")
        conjunto = self._leer(origen, entorno, "demostraciones")
        if not conjunto["valido"]:
            raise ValueError(f"su conjunto {id_conjunto} no es válido: {conjunto['motivo']}")
        npz = origen / "demostraciones.npz"
        pasos = resumen_demostraciones(npz)["pasos_por_partida"]
        comprobar_contra_conjunto(datos, huella_archivo(npz), pasos)
        return {"tipo": "subconjunto", "conjunto": id_conjunto,
                "partidas": len(datos["partidas"]),
                "pares": sum(pasos[i] for i in datos["partidas"]),
                "fuente": conjunto["fuente"], "splits": False}

    def listar(self, entorno: str | None = None,
               tipos: tuple[str, ...] = TIPOS) -> list[dict[str, Any]]:
        """Elementos de la biblioteca, válidos o no, ordenados por entorno, tipo y fecha.

        Solo mira las carpetas de entornos y tipos conocidos; dentro, ignora las
        ocultas (las temporales de un elemento que se está escribiendo).

        Args:
            entorno: Solo los de ese entorno (None: todos).
            tipos: Solo los de esos tipos.

        Returns:
            Un diccionario por elemento: id, entorno, tipo, nombre, origen, creado,
            descripcion, fuente, artefacto, valido, motivo y detalles.
        """
        elementos = []
        for clave in self.entornos:
            if entorno is not None and clave != entorno:
                continue
            for tipo in tipos:
                carpeta_tipo = self.raiz / clave / tipo
                if not carpeta_tipo.is_dir():
                    continue
                for carpeta in sorted(carpeta_tipo.iterdir()):
                    if carpeta.is_dir() and not carpeta.name.startswith("."):
                        elementos.append(self._leer(carpeta, clave, tipo))
        return sorted(elementos, key=lambda e: (e["entorno"], TIPOS.index(e["tipo"]),
                                                e["creado"] or "", e["id"]))

    def obtener(self, id_elemento: str) -> dict[str, Any]:
        """Un elemento por su id, sea cual sea su entorno y tipo.

        Raises:
            KeyError: Si no existe.
        """
        carpeta = self.ruta(id_elemento)
        return self._leer(carpeta, carpeta.parent.parent.name, carpeta.parent.name)

    def ruta(self, id_elemento: str) -> Path:
        """Carpeta de un elemento.

        Raises:
            KeyError: Si no existe (o el id no tiene la forma de uno).
        """
        if es_id(id_elemento):
            for clave in self.entornos:
                for tipo in TIPOS:
                    carpeta = self.raiz / clave / tipo / id_elemento
                    if carpeta.is_dir():
                        return carpeta
        raise KeyError(id_elemento)

    def quitar(self, id_elemento: str) -> None:
        """Borra un elemento de la biblioteca y libera su id.

        Primero lo renombra a una carpeta oculta, así deja de verse de golpe aunque el
        borrado tarde o falle a medias.

        Raises:
            KeyError: Si no existe.
        """
        carpeta = self.ruta(id_elemento)
        oculta = carpeta.with_name(f".borrando-{carpeta.name}")
        carpeta.rename(oculta)
        shutil.rmtree(oculta)
        _liberar_id(self.raiz, id_elemento)

    def experto_por_defecto(self, entorno: str) -> dict[str, Any] | None:
        """El experto congelado válido del entorno (el más antiguo si hay varios), o None."""
        congelados = [e for e in self.listar(entorno, ("experto",))
                      if e["valido"] and e["origen"] == "congelado"]
        return congelados[0] if congelados else None
