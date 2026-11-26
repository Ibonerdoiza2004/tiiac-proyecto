"""Precarga de la biblioteca desde W&B: los congelados y los BC y GAIL de entrega.

La usan ``scripts/preparar_biblioteca.py`` y, al arrancar, el contenedor de Docker.
Es idempotente: cada elemento guarda en su ficha el artefacto del que salió y no se
vuelve a descargar si ya está y es válido (uno roto se descarga de nuevo). Dos
precargas a la vez sobre la misma biblioteca se turnan con un cerrojo de archivo, así
que no añaden dos veces el mismo artefacto.

- *Congelados*: el experto y las demostraciones del perfil (``datos.artefacto_*``),
  comprobados con sus huellas (``comprobar_datos_congelados``), como hace
  ``run.py --descargar``.
- *Entrega*: los BC y GAIL que evaluó el run de comparación de la entrega de cada
  entorno (``precarga.comparaciones`` en ``configs/dashboard.yaml``). Se toman los
  que usó la comparación, no todos los del grupo, porque un grupo puede tener
  intentos interrumpidos. Solo se añaden si se entrenaron con las demostraciones
  congeladas del perfil que ya están en la biblioteca: la huella del NPZ tiene que
  coincidir en los metadatos del artefacto y en los del propio modelo.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from src.data.biblioteca import DETALLES, Biblioteca, agregar_nuevo
from src.data.splits import huella_archivo
from src.training.barrido import ARCHIVO_DEMOSTRACIONES
from src.training.flujo import comprobar_datos_congelados
from src.utils.archivos import turno_exclusivo
from src.utils.configuracion import dato_congelado
from src.utils.tracking import comprobar_wandb, descargar_artefacto

Descargar = Callable[[str, Path], Path]
Registro = Callable[[str], None]
# Cerrojo de las precargas, en la carpeta de la biblioteca (oculto para el índice).
ARCHIVO_CERROJO = ".precarga.lock"


@contextmanager
def _turno(biblioteca: Biblioteca) -> Iterator[None]:
    """Espera a que ninguna otra precarga (de este u otro proceso) use la biblioteca."""
    with turno_exclusivo(_carpeta_temporal(biblioteca) / ARCHIVO_CERROJO):
        yield


def _artefactos_presentes(biblioteca: Biblioteca, entorno: str) -> set[str]:
    """Artefactos de W&B que ya están en la biblioteca para ese entorno, en un elemento válido."""
    return {e["artefacto"] for e in biblioteca.listar(entorno) if e["artefacto"] and e["valido"]}


def _copiar(origen: Path, archivos: dict[str, str]) -> Callable[[Path], None]:
    """Escritor para agregar: copia cada archivo de origen con su nombre de destino."""
    def escribir(destino: Path) -> None:
        """Copia los archivos."""
        for de, a in archivos.items():
            shutil.copy2(origen / de, destino / a)
    return escribir


def precargar_congelados(cfg: dict[str, Any], biblioteca: Biblioteca,
                         descargar: Descargar | None = None,
                         informar: Registro = print) -> list[str]:
    """Descarga y añade el experto y las demostraciones congelados de un entorno.

    Args:
        cfg: Configuración resuelta del entorno.
        biblioteca: Índice de la biblioteca donde se añaden.
        descargar: ``descargar(nombre, carpeta)`` (por defecto, de W&B).
        informar: Recibe una línea por elemento añadido u omitido.

    Returns:
        Los ids añadidos (vacío si ya estaban los dos).

    Raises:
        ValueError: Si el perfil tiene los datos pendientes o las huellas no coinciden.
        RuntimeError: Si no se puede descargar algún artefacto.
    """
    descargar = descargar or (lambda nombre, carpeta: descargar_artefacto(cfg, nombre, carpeta))
    nombres = {"experto": dato_congelado(cfg, "artefacto_experto"),
               "demostraciones": dato_congelado(cfg, "artefacto_demostraciones")}
    with _turno(biblioteca):
        return _precargar_congelados(cfg, biblioteca, nombres, descargar, informar)


def _precargar_congelados(cfg: dict[str, Any], biblioteca: Biblioteca, nombres: dict[str, str],
                          descargar: Descargar, informar: Registro) -> list[str]:
    """Cuerpo de precargar_congelados, con el turno de la biblioteca tomado."""
    entorno = cfg["origen"]["entorno"]
    presentes = _artefactos_presentes(biblioteca, entorno)
    if all(nombre in presentes for nombre in nombres.values()):
        informar(f"{entorno}: los congelados ya están en la biblioteca")
        return []
    with tempfile.TemporaryDirectory(prefix=".precarga-",
                                     dir=_carpeta_temporal(biblioteca)) as tmp:
        carpetas = {tipo: Path(tmp) / tipo for tipo in nombres}
        for tipo, nombre in nombres.items():
            carpetas[tipo].mkdir()
            descargar(nombre, carpetas[tipo])
        # Las dos a la vez: la huella del experto y la de las partidas, como run.py --descargar.
        comprobar_datos_congelados(cfg, carpetas["demostraciones"], carpetas["experto"])
        anadidos = []
        textos = {"experto": "Experto congelado", "demostraciones": "Demostraciones congeladas"}
        archivos = {"experto": {"modelo.zip": "modelo.zip", "experto.json": "experto.json"},
                    "demostraciones": {ARCHIVO_DEMOSTRACIONES: ARCHIVO_DEMOSTRACIONES,
                                       "splits.json": "splits.json"}}
        for tipo, nombre in nombres.items():
            if nombre in presentes:
                informar(f"{entorno}: {nombre} ya está")
                continue
            extra = {"fuente": "experto"} if tipo == "demostraciones" else {}
            ficha = agregar_nuevo(biblioteca.raiz, entorno, tipo, textos[tipo], "congelado",
                                  _copiar(carpetas[tipo], archivos[tipo]), artefacto=nombre,
                                  **extra)
            informar(f"{entorno}: añadido {nombre} ({ficha['id']})")
            anadidos.append(ficha["id"])
    return anadidos


def _carpeta_temporal(biblioteca: Biblioteca) -> Path:
    """Carpeta de la biblioteca para las descargas temporales (oculta para el índice)."""
    biblioteca.raiz.mkdir(parents=True, exist_ok=True)
    return biblioteca.raiz


def huella_demostraciones_congeladas(cfg: dict[str, Any], biblioteca: Biblioteca) -> str | None:
    """Huella del NPZ de las demostraciones congeladas del perfil, o None si no están.

    Se busca el elemento válido que salió del artefacto del perfil
    (``datos.artefacto_demostraciones``), no cualquier congelado del entorno: si
    quedan otros de un congelado anterior, no cuentan.
    """
    artefacto = dato_congelado(cfg, "artefacto_demostraciones")
    for elemento in biblioteca.listar(cfg["origen"]["entorno"], ("demostraciones",)):
        if (elemento["valido"] and elemento["origen"] == "congelado"
                and elemento["artefacto"] == artefacto):
            return huella_archivo(biblioteca.ruta(elemento["id"]) / ARCHIVO_DEMOSTRACIONES)
    return None


def artefactos_de_comparacion(cfg: dict[str, Any], id_run: str) -> list[Any]:
    """Artefactos de BC y GAIL que usó un run de comparación de W&B.

    Raises:
        RuntimeError: Si no se puede consultar W&B.
    """
    wandb = comprobar_wandb({"wandb": {"modo": "offline"}})
    ruta = f"{cfg['wandb']['entidad']}/{cfg['wandb']['proyecto']}/{id_run}"
    try:
        return [a for a in wandb.Api().run(ruta).used_artifacts() if a.type in ("bc", "gail")]
    except (ValueError, wandb.errors.Error) as error:
        raise RuntimeError(f"no se pudo consultar el run {ruta} (¿has iniciado sesión con "
                           f"wandb login?): {error}") from error


def precargar_entrega(cfg: dict[str, Any], biblioteca: Biblioteca, artefactos: list[Any],
                      informar: Registro = print) -> list[str]:
    """Añade los BC y GAIL de entrega de un entorno.

    Args:
        cfg: Configuración resuelta del entorno.
        biblioteca: Índice de la biblioteca (ya con las demostraciones congeladas).
        artefactos: Artefactos de W&B (artefactos_de_comparacion): cada uno con
            ``name``, ``type``, ``metadata`` y ``download(root=...)``.
        informar: Recibe una línea por elemento añadido u omitido.

    Returns:
        Los ids añadidos.

    Raises:
        ValueError: Si no están las demostraciones congeladas en la biblioteca.
    """
    with _turno(biblioteca):
        return _precargar_entrega(cfg, biblioteca, artefactos, informar)


def _comprobar_huella_modelo(tipo: str, gym: str, huella: str) -> Callable[[Path], None]:
    """Comprueba en la carpeta escrita que el modelo dice haberse entrenado con huella.

    Lee los metadatos del propio modelo (gail.json o los de politica.pt) con los
    mismos lectores que el índice de la biblioteca.

    Raises:
        ValueError: Si el modelo no se puede leer, es de otro entorno o cita otra huella.
    """
    def comprobar(carpeta: Path) -> None:
        """Lanza ValueError si la huella interna no es la esperada."""
        interna = DETALLES[tipo](carpeta, gym).get("huella_dataset")
        if interna != huella:
            raise ValueError(f"el modelo se entrenó con {interna!r}, no con las demostraciones "
                             "congeladas")
    return comprobar


def _precargar_entrega(cfg: dict[str, Any], biblioteca: Biblioteca, artefactos: list[Any],
                       informar: Registro) -> list[str]:
    """Cuerpo de precargar_entrega, con el turno de la biblioteca tomado."""
    entorno = cfg["origen"]["entorno"]
    huella = huella_demostraciones_congeladas(cfg, biblioteca)
    if huella is None:
        raise ValueError(f"{entorno}: faltan las demostraciones congeladas en la biblioteca; "
                         "precarga antes los congelados")
    presentes = _artefactos_presentes(biblioteca, entorno)
    anadidos = []
    for artefacto in artefactos:
        if artefacto.name in presentes:
            informar(f"{entorno}: {artefacto.name} ya está")
            continue
        metadatos = artefacto.metadata or {}
        if metadatos.get("huella_dataset") != huella:
            informar(f"{entorno}: se omite {artefacto.name}: no se entrenó con las "
                     "demostraciones congeladas")
            continue
        tipo, cantidad, repeticion = artefacto.type, metadatos.get("cantidad"), metadatos.get(
            "repeticion")
        with tempfile.TemporaryDirectory(prefix=".precarga-",
                                         dir=_carpeta_temporal(biblioteca)) as tmp:
            carpeta = Path(artefacto.download(root=tmp))
            if tipo == "bc":
                pesos = sorted(carpeta.glob("*.pt"))
                if len(pesos) != 1:
                    informar(f"{entorno}: se omite {artefacto.name}: no tiene un único .pt")
                    continue
                archivos = {pesos[0].name: "politica.pt"}
            else:
                archivos = {a: a for a in ("modelo.zip", "discriminador.pt", "gail.json")}
            copiar = _copiar(carpeta, archivos)
            comprobar = _comprobar_huella_modelo(tipo, cfg["entorno"]["id"], huella)

            def escribir(destino: Path) -> None:
                """Copia los archivos y comprueba la huella que cita el propio modelo."""
                copiar(destino)
                comprobar(destino)

            try:
                ficha = agregar_nuevo(
                    biblioteca.raiz, entorno, tipo,
                    f"{tipo.upper()} · {cantidad} partidas · repetición {repeticion}",
                    "entrega", escribir, artefacto=artefacto.name)
            except ValueError as error:
                informar(f"{entorno}: se omite {artefacto.name}: {error}")
                continue
        informar(f"{entorno}: añadido {artefacto.name} ({ficha['id']})")
        anadidos.append(ficha["id"])
    return anadidos
