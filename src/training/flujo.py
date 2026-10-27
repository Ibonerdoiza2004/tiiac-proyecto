"""Etapas del experimento, que encadenan run.py y los scripts.

Experto y datos solo los usa el smoke (y sus scripts): el experimento real
parte de los datos congelados de la configuración (datos.*).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from src.data.demostraciones import (generar_dataset, huella_partidas,
                                     cargar_demostraciones)
from src.data.entornos import espacios_entorno
from src.data.splits import cargar_splits, huella_archivo
from src.data.tipos import Trayectoria
from src.training.barrido import ARCHIVO_DEMOSTRACIONES, ARCHIVO_SPLITS, MetodoBarrido, ResultadoTarea
from src.training.bc import METODO_BC, cargar_politica_bc
from src.training.evaluacion import config_evaluacion, evaluar_final
from src.training.gail import METODO_GAIL, cargar_gail
from src.training.trpo import (cargar_experto, cargar_politica_experto, entrenar_experto,
                               guardar_experto)
from src.utils.archivos import comprobar_directorio_libre, escribir_archivo_nuevo
from src.utils.procedencia import (a_json, avisar_estado_git, estado_git, ruta_registrada,
                                   versiones_y_hardware)
from src.utils.tracking import (abrir_run, abrir_run_barrido, nombre_run, registrar,
                                registrar_resumen, subir_artefacto, terminar_run,
                                usar_artefacto)


def entrenar_y_guardar_experto(cfg: dict, salida: Path, grupo: str) -> dict[str, Any]:
    """Entrena, registra y guarda el experto TRPO.

    Args:
        cfg: Configuración resuelta del entorno.
        salida: Carpeta nueva para modelo.zip y experto.json.
        grupo: Grupo compartido de W&B.

    Returns:
        Resumen del entrenamiento del experto.

    Raises:
        FileExistsError: Si salida ya existe.
        ValueError: Si la configuración o el entrenamiento no son válidos.
    """
    salida = Path(salida)
    comprobar_directorio_libre(salida)
    run = abrir_run(cfg, grupo=grupo, job_type="experto", nombre=nombre_run(cfg, "experto"),
                    etiquetas=[f"entorno:{cfg['entorno']['id']}", "metodo:experto"])
    try:
        modelo, resumen = entrenar_experto(cfg, run)
        guardar_experto(salida, modelo, cfg, resumen)
        registrar_resumen(run, resumen)
        if run is not None:
            huella_modelo = huella_archivo(salida / "modelo.zip")
            subir_artefacto(run, f"experto-{cfg['entorno']['id']}-{huella_modelo[:12]}",
                            "experto", [salida / "modelo.zip", salida / "experto.json"],
                            {"entorno": cfg["entorno"]["id"], "huella_modelo": huella_modelo})
    except BaseException as error:
        terminar_run(run, error)
        raise
    terminar_run(run)
    avisar_estado_git(estado_git())
    return resumen


def generar_y_registrar_datos(cfg: dict, experto: Path, salida: Path, grupo: str,
                              huella_esperada: str | None = None) -> dict[str, Any]:
    """Genera las demostraciones, registra su linaje y comprueba su huella opcional.

    Args:
        cfg: Configuración resuelta del entorno.
        experto: Carpeta del experto guardado.
        salida: Carpeta nueva para NPZ y splits.
        grupo: Grupo compartido de W&B.
        huella_esperada: Huella opcional del contenido de las partidas.

    Returns:
        Resultado de generar_dataset y metadatos del experto, más ``coincide``;
        cuando coincide, incluye ``metadatos``.

    Raises:
        FileExistsError: Si salida ya existe.
        OSError: Si no se puede cargar el experto o guardar los datos.
        ValueError: Si el experto o la configuración no son válidos.
    """
    salida = Path(salida)
    comprobar_directorio_libre(salida)
    politica, metadatos = cargar_politica_experto(Path(experto), cfg)
    run = abrir_run(cfg, grupo=grupo, job_type="datos", nombre=nombre_run(cfg, "datos"),
                    etiquetas=[f"entorno:{cfg['entorno']['id']}", "metodo:datos"])
    try:
        if run is not None and cfg.get("wandb", {}).get("modo") == "online":
            huella_modelo = metadatos["experto"]["huella_modelo"]
            usar_artefacto(run, f"experto-{cfg['entorno']['id']}-{huella_modelo[:12]}:latest",
                           "experto")
        inicio = time.perf_counter()
        resultado = generar_dataset(cfg, politica, salida, metadatos)
        resultado["segundos"] = time.perf_counter() - inicio
        coincide = (huella_esperada is None
                    or huella_esperada == resultado["huella_partidas"])
        if not coincide:
            terminar_run(run, ValueError("la huella de las partidas no coincide"))
            return {**resultado, "coincide": False}
        retornos = resultado["retornos"]
        registrar_resumen(run, {"pasos": resultado["pasos"], "semillas": resultado["semillas"],
                                "segundos": resultado["segundos"], "partidas": len(retornos),
                                "retorno_medio": sum(retornos) / len(retornos),
                                "huella_dataset": resultado["huella_dataset"],
                                "huella_splits": resultado["huella_splits"],
                                "huella_partidas": resultado["huella_partidas"]})
        subir_artefacto(run, f"demostraciones-{resultado['huella_dataset'][:12]}",
                        "demostraciones", [resultado["demostraciones"], resultado["splits"]],
                        {k: resultado[k] for k in
                         ("huella_dataset", "huella_splits", "huella_partidas")})
    except BaseException as error:
        terminar_run(run, error)
        raise
    terminar_run(run)
    resultado["coincide"] = True
    resultado["metadatos"] = metadatos
    avisar_estado_git(metadatos["procedencia"]["codigo"])
    return resultado


def barrer(metodo: MetodoBarrido, cfg: dict, datos: Path, salida: Path, grupo: str,
           informar: Callable[[ResultadoTarea], None] | None = None,
           cantidades: Sequence[int] | None = None,
           repeticiones: Sequence[int] | None = None) -> list[tuple[int, int]]:
    """Ejecuta el barrido de un método y devuelve las tareas fallidas.

    Args:
        metodo: Método BC o GAIL.
        cfg: Configuración resuelta.
        datos: Carpeta con las demostraciones y los splits.
        salida: Carpeta nueva para los modelos.
        grupo: Grupo compartido de W&B.
        informar: Callback opcional por tarea terminada.
        cantidades: Subconjuntos a entrenar; por defecto los de la config.
        repeticiones: Repeticiones a entrenar; por defecto todas.

    Returns:
        Lista de pares (cantidad, repetición) que fallaron.

    Raises:
        FileExistsError: Si salida ya existe.
        OSError: Si faltan o no se pueden leer los datos.
        ValueError: Si la configuración, cantidades o repeticiones no son válidas.
    """
    salida = Path(salida)
    comprobar_directorio_libre(salida)
    from src.training import barrido as modulo_barrido

    trayectorias, splits, procedencia = modulo_barrido.cargar_datos(cfg, Path(datos))
    tareas = modulo_barrido.elegir_tareas(cfg, splits, cantidades, repeticiones)
    reportar = informar or (lambda _: None)
    inicio = time.perf_counter()
    fallos = modulo_barrido.ejecutar_barrido(
        tareas, min(int(cfg["experimento"]["procesos"]), len(tareas)), metodo.tarea,
        lambda c, r: (cfg, trayectorias, splits, c, r, metodo.destino(salida, c, r),
                      procedencia, grupo), reportar)
    _ = time.perf_counter() - inicio
    return fallos


def cargar_agentes(cfg: dict, experto: Path | None, bc: Path | None, gail: Path | None,
                   cantidades: Sequence[int] | None = None,
                   repeticiones: Sequence[int] | None = None
                   ) -> tuple[list[dict[str, Any]], str | None]:
    """Carga expertos, políticas BC y agentes GAIL y comprueba su procedencia.

    Args:
        cfg: Configuración resuelta del entorno.
        experto: Carpeta del experto o None para omitirlo.
        bc: Raíz de los modelos BC o None para omitirlos.
        gail: Raíz de los modelos GAIL o None para omitirlos.
        cantidades: Subconjuntos que se cargan; por defecto los configurados.
        repeticiones: Repeticiones que se cargan; por defecto todas.

    Returns:
        Agentes incluyendo el aleatorio y huella del NPZ compartido (None si no hay BC/GAIL).

    Raises:
        OSError: Si falta algún checkpoint.
        ValueError: Si los agentes no corresponden al entorno o usan datasets distintos.
    """
    entorno = cfg["entorno"]["id"]
    cantidades_lista = list(cantidades or cfg["demostraciones"]["subconjuntos"])
    total = cfg["experimento"]["repeticiones"]
    repeticiones_lista = (list(repeticiones) if repeticiones is not None else list(range(total)))
    if any(not 0 <= r < total for r in repeticiones_lista):
        raise ValueError(f"repeticiones fuera de rango: {repeticiones_lista}; van de 0 a {total - 1}")
    agentes: list[dict[str, Any]] = [{"metodo": "aleatorio", "cantidad": None,
                                      "repeticion": None, "ruta": None, "agente": "aleatorio",
                                      "estadisticas": None}]
    huellas: set[str] = set()

    def anotar(metodo: str, cantidad: int | None, repeticion: int | None, ruta: Path,
               agente: Any, estadisticas: dict[str, Any] | None, entorno_modelo: str) -> None:
        """Añade un agente solo si pertenece al entorno pedido."""
        if entorno_modelo != entorno:
            raise ValueError(f"{ruta} es de {entorno_modelo!r} y la configuración pide {entorno!r}")
        agentes.append({"metodo": metodo, "cantidad": cantidad, "repeticion": repeticion,
                        "ruta": ruta, "agente": agente, "estadisticas": estadisticas})

    if experto is not None:
        modelo, meta = cargar_experto(Path(experto))
        anotar("experto", None, None, Path(experto), modelo,
               meta["resumen"]["normalizacion"], meta["configuracion"]["entorno"]["id"])
    for cantidad in cantidades_lista:
        for repeticion in repeticiones_lista:
            if bc is not None:
                ruta = Path(bc) / f"cantidad_{cantidad}" / f"repeticion_{repeticion}.pt"
                politica, meta = cargar_politica_bc(ruta)
                huellas.add(meta["huella_dataset"])
                anotar("bc", cantidad, repeticion, ruta, politica, None, meta["entorno"])
            if gail is not None:
                ruta = Path(gail) / f"cantidad_{cantidad}" / f"repeticion_{repeticion}"
                agente, _, meta = cargar_gail(ruta)
                huellas.add(meta["datos"]["huella_dataset"])
                anotar("gail", cantidad, repeticion, ruta, agente,
                       meta["resumen"]["normalizacion"], meta["configuracion"]["entorno"]["id"])
    if len(huellas) > 1:
        raise ValueError("Los BC y GAIL no se entrenaron con las mismas demostraciones: "
                         f"huellas {sorted(huellas)}")
    return agentes, next(iter(huellas), None)


def evaluar_y_guardar(cfg: dict, agentes: Sequence[dict[str, Any]], huella: str | None,
                      salida: Path, grupo: str) -> list[dict[str, Any]]:
    """Evalúa agentes con el protocolo común, guarda JSON y registra comparación en W&B.

    Args:
        cfg: Configuración resuelta.
        agentes: Entradas devueltas por cargar_agentes.
        huella: Huella del NPZ compartido por BC y GAIL.
        salida: Archivo JSON nuevo para la evaluación.
        grupo: Grupo compartido de W&B.

    Returns:
        Una fila de resultados por agente, con ruta, retornos y longitudes.

    Raises:
        FileExistsError: Si salida ya existe.
        OSError: Si no se puede escribir el resultado.
        ValueError: Si la configuración o un agente no son válidos.
    """
    salida = Path(salida)
    protocolo = config_evaluacion(cfg)
    inicio = time.perf_counter()
    run = abrir_run(cfg, grupo=grupo, job_type="comparacion",
                    nombre=nombre_run(cfg, "comparacion"),
                    etiquetas=[f"entorno:{cfg['entorno']['id']}", "metodo:comparacion"])
    try:
        resultados: list[dict[str, Any]] = []
        for entrada in agentes:
            resultado = evaluar_final(entrada["agente"], cfg, entrada["estadisticas"])
            ruta = entrada["ruta"]
            resultados.append({"metodo": entrada["metodo"], "cantidad": entrada["cantidad"],
                                "repeticion": entrada["repeticion"],
                                "ruta": None if ruta is None else ruta_registrada(ruta),
                                **resultado})
        segundos = time.perf_counter() - inicio
        texto = json.dumps({"entorno": cfg["entorno"]["id"], "protocolo": protocolo,
                            "huella_dataset": huella, "segundos": segundos,
                            "codigo": estado_git(), "entorno_ejecucion": versiones_y_hardware(),
                            "resultados": resultados}, ensure_ascii=False, indent=2,
                           allow_nan=False, default=a_json)
        escribir_archivo_nuevo(salida, lambda temporal: temporal.write_text(texto, encoding="utf-8"))
        if run is not None:
            wandb = __import__("wandb")
            if cfg.get("wandb", {}).get("modo") == "online":
                for entrada in agentes:
                    metodo = entrada["metodo"]
                    if metodo == "experto":
                        huella_modelo = huella_archivo(Path(entrada["ruta"]) / "modelo.zip")
                        usar_artefacto(run, f"experto-{cfg['entorno']['id']}-{huella_modelo[:12]}:latest",
                                       "experto")
                    elif metodo in ("bc", "gail"):
                        ruta_modelo = (Path(entrada["ruta"]) if metodo == "bc"
                                       else Path(entrada["ruta"]) / "modelo.zip")
                        huella_modelo = huella_archivo(ruta_modelo)
                        nombre = (f"{metodo}-{cfg['entorno']['id']}-c{entrada['cantidad']}-"
                                  f"r{entrada['repeticion']}-{huella_modelo[:12]}:latest")
                        usar_artefacto(run, nombre, metodo)
            columnas = ["metodo", "cantidad", "repeticion", "retorno_medio",
                        "retorno_desviacion", "longitud_media"]
            tabla = wandb.Table(columns=columnas)
            episodios = wandb.Table(columns=["metodo", "cantidad", "repeticion", "episodio",
                                             "retorno", "longitud"])
            for fila in resultados:
                tabla.add_data(*(fila.get(k) for k in columnas))
                for indice, (retorno, longitud) in enumerate(zip(fila["retornos"], fila["longitudes"])):
                    episodios.add_data(fila["metodo"], fila["cantidad"], fila["repeticion"],
                                       indice, retorno, longitud)
            run.log({"comparacion": tabla, "episodios": episodios})
        registrar_resumen(run, {"entorno": cfg["entorno"]["id"],
                                "episodios": protocolo["episodios"],
                                "semilla": protocolo["semilla"], "segundos": segundos,
                                "huella_dataset": huella})
        subir_artefacto(run, f"comparacion-{cfg['entorno']['id']}", "comparacion",
                        [salida], {"grupo": grupo, "huella_dataset": huella})
    except BaseException as error:
        terminar_run(run, error)
        raise
    terminar_run(run)
    avisar_estado_git(estado_git())
    return resultados


def comprobar_datos_congelados(cfg: dict, datos: Path, experto: Path) -> None:
    """Comprueba que datos y experto son los congelados de la configuración.

    Args:
        cfg: Configuración que declara datos.huella_partidas.
        datos: Carpeta de demostraciones.
        experto: Carpeta del experto.

    Raises:
        ValueError: Si faltan los datos o experto, los splits no citan ese NPZ,
            o la huella de partidas no coincide con la declarada.

    Returns:
        None si ambas carpetas corresponden al congelado.
    """
    datos, experto = Path(datos), Path(experto)
    if not (experto / "modelo.zip").is_file() or not (experto / "experto.json").is_file():
        raise ValueError(f"no está el experto congelado en {experto}; ver «Datos» en el README")
    try:
        trayectorias, _ = cargar_demostraciones(datos / ARCHIVO_DEMOSTRACIONES,
                                                *espacios_entorno(cfg))
        cargar_splits(datos / ARCHIVO_SPLITS, huella_archivo(datos / ARCHIVO_DEMOSTRACIONES),
                      [t.id_episodio for t in trayectorias])
    except OSError as error:
        raise ValueError(f"no están los datos congelados en {datos}; ver «Datos» en el README") from error
    if huella_partidas(trayectorias) != cfg["datos"]["huella_partidas"]:
        raise ValueError(f"{datos} no son las demostraciones congeladas: la huella de las "
                         "partidas no coincide con datos.huella_partidas")


def ejecutar_experimento(cfg: dict, datos: Path, experto: Path, carpeta: Path,
                         grupo: str) -> dict[str, Any]:
    """Ejecuta los barridos BC y GAIL y evalúa todos los agentes.

    Args:
        cfg: Configuración resuelta del entorno.
        datos: Carpeta con NPZ y splits.
        experto: Carpeta del experto guardado.
        carpeta: Carpeta nueva de salida.
        grupo: Grupo compartido de W&B.

    Returns:
        Diccionario con resultados y fallos. Si falla un modelo, no compara agentes.

    Raises:
        FileExistsError: Si la carpeta de evaluación existe.
        OSError: Si faltan datos o modelos.
        ValueError: Si la configuración o los splits no son válidos.
    """
    METODO_BC.comprobar_config(cfg)
    METODO_GAIL.comprobar_config(cfg)
    carpeta = Path(carpeta)
    fallos = [("bc", *t) for t in barrer(METODO_BC, cfg, datos, carpeta / "bc", grupo)]
    fallos += [("gail", *t) for t in barrer(METODO_GAIL, cfg, datos, carpeta / "gail", grupo)]
    if fallos:
        return {"resultados": [], "fallos": fallos}
    agentes, huella = cargar_agentes(cfg, experto, carpeta / "bc", carpeta / "gail")
    resultados = evaluar_y_guardar(cfg, agentes, huella, carpeta / "evaluacion.json", grupo)
    return {"resultados": resultados, "fallos": []}


def evaluar_carpeta(cfg: dict, carpeta: Path, experto: Path, grupo: str) -> dict[str, Any]:
    """Vuelve a evaluar una ejecución previa de ejecutar_experimento.

    Args:
        cfg: Configuración usada en la evaluación.
        carpeta: Carpeta que contiene bc/ y gail/.
        experto: Carpeta del experto guardado.
        grupo: Grupo compartido de W&B.

    Returns:
        Diccionario con resultados y sin fallos.

    Raises:
        OSError: Si falta algún checkpoint.
        ValueError: Si los modelos son incompatibles.
    """
    carpeta = Path(carpeta)
    agentes, huella = cargar_agentes(cfg, experto, carpeta / "bc", carpeta / "gail")
    salida = carpeta / f"evaluacion-{datetime.now():%Y%m%d-%H%M%S}.json"
    return {"resultados": evaluar_y_guardar(cfg, agentes, huella, salida, grupo), "fallos": []}
