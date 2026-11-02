"""Etapas del experimento, que encadenan run.py y los scripts.

Experto y datos solo los usa el smoke (y sus scripts): el experimento real
parte de los datos congelados de la configuración (datos.*).
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from src.data.demostraciones import (generar_dataset, huella_partidas,
                                     cargar_demostraciones)
from src.data.entornos import espacios_entorno
from src.data.splits import cargar_splits, huella_archivo
from src.training.barrido import (ARCHIVO_DEMOSTRACIONES, ARCHIVO_SPLITS, ResultadoTarea,
                                  barrer, ejecutar_barrido)
from src.training.bc import METODO_BC, cargar_politica_bc
from src.training.evaluacion import config_evaluacion, evaluar_final
from src.training.gail import METODO_GAIL, cargar_gail
from src.training.trpo import (cargar_experto, cargar_politica_experto, entrenar_experto,
                               guardar_experto)
from src.utils.archivos import (comprobar_directorio_libre, escribir_archivo_nuevo,
                                escribir_carpeta_nueva, marca_de_tiempo)
from src.utils.configuracion import dato_congelado
from src.utils.procedencia import (a_json, avisar_estado_git, estado_git, ruta_registrada,
                                   versiones_y_hardware)
from src.utils.tracking import (abrir_run, comprobar_wandb, descargar_artefacto,
                                nombre_artefacto, nombre_run, registrar_resumen,
                                subir_artefacto, terminar_run, usar_artefacto)
from src.utils.validacion import es_entero_positivo


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
            subir_artefacto(run, nombre_artefacto("experto", huella_modelo, cfg["entorno"]["id"]),
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
            usar_artefacto(run, nombre_artefacto("experto", huella_modelo, cfg["entorno"]["id"])
                           + ":latest", "experto")
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
        subir_artefacto(run, nombre_artefacto("demostraciones", resultado["huella_dataset"]),
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
        raise ValueError(f"repeticiones fuera de rango: {repeticiones_lista}; "
                         f"van de 0 a {total - 1}")
    agentes: list[dict[str, Any]] = [{"metodo": "aleatorio", "cantidad": None,
                                      "repeticion": None, "ruta": None, "agente": "aleatorio",
                                      "estadisticas": None}]
    huellas: set[str] = set()

    def anotar(metodo: str, cantidad: int | None, repeticion: int | None, ruta: Path,
               agente: Any, estadisticas: dict[str, Any] | None, entorno_modelo: str) -> None:
        """Añade un agente solo si pertenece al entorno pedido."""
        if entorno_modelo != entorno:
            raise ValueError(f"{ruta} es de {entorno_modelo!r} y la configuración "
                             f"pide {entorno!r}")
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


def _evaluar_en_procesos(cfg: dict, agentes: Sequence[dict[str, Any]],
                         procesos: int) -> list[dict[str, Any]]:
    """Evalúa cada agente desde disco en procesos y devuelve sus filas en el orden de entrada.

    Raises:
        ValueError: Si falla la evaluación de algún agente.
    """
    por_indice: dict[int, dict[str, Any]] = {}

    def guardar(resultado: ResultadoTarea) -> None:
        """Guarda el resultado de cada agente por su índice de entrada."""
        indice, _, resumen, error = resultado
        if error is None and resumen is not None:
            por_indice[indice] = resumen

    tareas = [(indice, 0) for indice in range(len(agentes))]
    fallidos = ejecutar_barrido(
        tareas, min(procesos, len(tareas)), evaluar_agente_guardado,
        lambda indice, _: (cfg, agentes[indice]["metodo"], agentes[indice]["ruta"]),
        guardar)
    if fallidos:
        raise ValueError(f"falló la evaluación de los agentes {[i for i, _ in fallidos]}")
    return [{"metodo": entrada["metodo"], "cantidad": entrada["cantidad"],
             "repeticion": entrada["repeticion"],
             "ruta": None if entrada["ruta"] is None else ruta_registrada(entrada["ruta"]),
             **por_indice[indice]}
            for indice, entrada in enumerate(agentes)]


def _artefacto_de_agente(cfg: dict, entrada: dict[str, Any]) -> tuple[str, str] | None:
    """Nombre (última versión) y tipo del artefacto W&B de un agente; None si no tiene."""
    metodo = entrada["metodo"]
    if metodo not in ("experto", "bc", "gail"):
        return None
    ruta = Path(entrada["ruta"])
    huella = huella_archivo(ruta if metodo == "bc" else ruta / "modelo.zip")
    nombre = nombre_artefacto(metodo, huella, cfg["entorno"]["id"], entrada["cantidad"],
                              entrada["repeticion"])
    return f"{nombre}:latest", metodo


def _registrar_comparacion(run: Any, cfg: dict, agentes: Sequence[dict[str, Any]],
                           resultados: Sequence[dict[str, Any]]) -> None:
    """Declara en W&B los modelos evaluados y registra las tablas de comparación y episodios."""
    wandb = comprobar_wandb(cfg)
    if cfg.get("wandb", {}).get("modo") == "online":
        for entrada in agentes:
            artefacto = _artefacto_de_agente(cfg, entrada)
            if artefacto is not None:
                usar_artefacto(run, *artefacto)
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


def evaluar_y_guardar(cfg: dict, agentes: Sequence[dict[str, Any]], huella: str | None,
                      salida: Path, grupo: str) -> list[dict[str, Any]]:
    """Evalúa agentes en procesos y registra JSON y comparación con el protocolo común.

    La cantidad de procesos es experimento.procesos, igual que en los barridos;
    cada agente se recarga desde disco dentro del proceso que lo evalúa.

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
    procesos = cfg.get("experimento", {}).get("procesos")
    if not es_entero_positivo(procesos):
        raise ValueError("experimento.procesos debe ser un entero positivo")
    if not agentes:
        raise ValueError("no hay agentes para evaluar")
    inicio = time.perf_counter()
    run = abrir_run(cfg, grupo=grupo, job_type="comparacion",
                    nombre=nombre_run(cfg, "comparacion"),
                    etiquetas=[f"entorno:{cfg['entorno']['id']}", "metodo:comparacion"])
    try:
        resultados = _evaluar_en_procesos(cfg, agentes, int(procesos))
        segundos = time.perf_counter() - inicio
        texto = json.dumps({"entorno": cfg["entorno"]["id"], "protocolo": protocolo,
                            "huella_dataset": huella, "segundos": segundos,
                            "codigo": estado_git(), "entorno_ejecucion": versiones_y_hardware(),
                            "resultados": resultados}, ensure_ascii=False, indent=2,
                           allow_nan=False, default=a_json)
        escribir_archivo_nuevo(
            salida, lambda temporal: temporal.write_text(texto, encoding="utf-8"))
        if run is not None:
            _registrar_comparacion(run, cfg, agentes, resultados)
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
        ValueError: Si el perfil aún tiene las huellas pendientes (DATOS_PENDIENTES),
            faltan los datos o experto, los splits no citan ese NPZ, o las huellas
            de partidas o experto no coinciden con las declaradas.

    Returns:
        None si ambas carpetas corresponden al congelado.
    """
    datos, experto = Path(datos), Path(experto)
    # Antes que los archivos: con el perfil pendiente, la causa es esa, no su ausencia.
    huella_esperada_partidas = dato_congelado(cfg, "huella_partidas")
    huella_esperada_experto = dato_congelado(cfg, "huella_experto")
    if not (experto / "modelo.zip").is_file() or not (experto / "experto.json").is_file():
        raise ValueError(f"no está el experto congelado en {experto}; ver «Datos» en el README")
    try:
        trayectorias, _ = cargar_demostraciones(datos / ARCHIVO_DEMOSTRACIONES,
                                                *espacios_entorno(cfg))
        cargar_splits(datos / ARCHIVO_SPLITS, huella_archivo(datos / ARCHIVO_DEMOSTRACIONES),
                      [t.id_episodio for t in trayectorias])
    except OSError as error:
        raise ValueError(
            f"no están los datos congelados en {datos}; ver «Datos» en el README") from error
    if huella_partidas(trayectorias) != huella_esperada_partidas:
        raise ValueError(f"{datos} no son las demostraciones congeladas: la huella de las "
                         "partidas no coincide con datos.huella_partidas")
    if huella_archivo(experto / "modelo.zip") != huella_esperada_experto:
        raise ValueError(f"{experto} no es el experto congelado: la huella de modelo.zip "
                         "no coincide con datos.huella_experto")


def obtener_datos_congelados(cfg: dict, datos: Path, experto: Path) -> None:
    """Descarga de W&B los datos y el experto congelados que falten localmente.

    Solo descarga en carpetas que no existen o están vacías; nunca sobrescribe.
    Después hay que comprobarlos con comprobar_datos_congelados.

    Args:
        cfg: Configuración que declara los nombres de artefacto de W&B.
        datos: Carpeta de demostraciones congeladas.
        experto: Carpeta del experto congelado.

    Raises:
        FileExistsError: Si una carpeta existe con otro contenido.
        RuntimeError: Si no se puede descargar algún artefacto.
        ValueError: Si hay que descargar un artefacto que el perfil aún tiene
            pendiente (DATOS_PENDIENTES).

    La descarga va a una carpeta temporal que se renombra al final; si falla, no queda nada.
    """
    for carpeta, clave in ((Path(datos), "artefacto_demostraciones"),
                           (Path(experto), "artefacto_experto")):
        if carpeta.is_dir() and any(carpeta.iterdir()):
            continue
        # Se descarga en una temporal que solo se renombra si termina bien: una
        # descarga cortada no deja una carpeta a medias que luego se saltaría.
        escribir_carpeta_nueva(carpeta, lambda temporal, nombre=dato_congelado(cfg, clave):
                               descargar_artefacto(cfg, nombre, temporal))


def evaluar_agente_guardado(cfg: dict, metodo: str, ruta: Path | None) -> dict[str, Any]:
    """Carga un agente desde su ruta y lo evalúa con evaluar_final (una tarea del reparto).

    Args:
        cfg: Configuración resuelta con la sección evaluacion.
        metodo: aleatorio, experto, bc o gail.
        ruta: Carpeta o archivo del agente; None para el aleatorio.

    Returns:
        El resultado de evaluar_final.

    Raises:
        OSError: Si falta el checkpoint.
        ValueError: Si el agente o la configuración no son válidos.
    """
    if metodo == "aleatorio":
        return evaluar_final("aleatorio", cfg)
    if ruta is None:
        raise ValueError(f"falta la ruta del agente {metodo}")
    if metodo == "experto":
        modelo, meta = cargar_experto(Path(ruta))
        estadisticas = meta["resumen"]["normalizacion"]
    elif metodo == "bc":
        modelo, _ = cargar_politica_bc(Path(ruta))
        estadisticas = None
    elif metodo == "gail":
        modelo, _, meta = cargar_gail(Path(ruta))
        estadisticas = meta["resumen"]["normalizacion"]
    else:
        raise ValueError(f"método de evaluación desconocido: {metodo!r}")
    return evaluar_final(modelo, cfg, estadisticas)


def imprimir_resultados(resultados: Sequence[dict[str, Any]]) -> None:
    """Imprime una línea con las métricas de cada agente evaluado (el smoke las necesita)."""
    print("\n=== métricas ===")
    for resultado in resultados:
        nombre = " ".join(str(x) for x in (resultado["metodo"], resultado.get("cantidad"),
                                           resultado.get("repeticion")) if x is not None)
        print(f"{nombre}: retorno_medio {resultado['retorno_medio']:.1f} "
              f"± {resultado['retorno_desviacion']:.1f} "
              f"(longitud {resultado['longitud_media']:.1f})")


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
        ValueError: Si la configuración (también evaluacion) o los splits no son válidos.
    """
    METODO_BC.comprobar_config(cfg)
    METODO_GAIL.comprobar_config(cfg)
    config_evaluacion(cfg)
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
    salida = carpeta / f"evaluacion-{marca_de_tiempo()}.json"
    return {"resultados": evaluar_y_guardar(cfg, agentes, huella, salida, grupo), "fallos": []}
