"""Barridos de entrenamientos: cada cantidad de partidas expertas × cada repetición.

Es lo común a los barridos de BC y de GAIL (scripts/entrenar_bc.py y
scripts/entrenar_gail.py), que solo se diferencian en qué entrena cada tarea:

- procedencia_datos, cargar_datos y elegir_tareas preparan los datos y las
  combinaciones (cantidad, repetición) que se entrenan.
- comprobar_config_comun valida lo que comparten los dos barridos.
- ejecutar_barrido reparte las tareas en serie o en procesos spawn, siempre con
  un hilo de PyTorch, y aísla los fallos: uno que falla no detiene los demás.
- main_barrido es el flujo de línea de comandos de los dos scripts, con los
  mismos argumentos, códigos de salida y avisos.

Cada tarea depende solo de sus argumentos (cantidad y repetición fijan las
semillas con semillas_repeticion), así que en serie y en paralelo, en la misma
máquina, dan el mismo resultado.
"""

from __future__ import annotations

import argparse
import multiprocessing
import signal
import sys
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from multiprocessing.connection import Connection, wait
from pathlib import Path
from typing import Any

import torch

from src.data.demostraciones import cargar_demostraciones, config_demostraciones, huella_partidas
from src.data.entornos import espacios_entorno
from src.data.splits import SplitsDemostraciones, cargar_splits, huella_archivo
from src.data.tipos import Trayectoria
from src.utils.archivos import comprobar_directorio_libre
from src.utils.configuracion import TipoEntorno, resolver_configuracion
from src.utils.procedencia import avisar_estado_git, estado_git, ruta_registrada
from src.utils.seed import validar_semilla
from src.utils.tracking import comprobar_wandb, crear_grupo
from src.utils.validacion import es_entero_positivo

ARCHIVO_DEMOSTRACIONES = "demostraciones.npz"
ARCHIVO_SPLITS = "splits.json"

ResultadoTarea = tuple[int, int, dict[str, Any] | None, str | None]
"""(cantidad, repeticion, resumen, error): resumen si fue bien, error si falló."""


def procedencia_datos(ruta_demostraciones: Path, ruta_splits: Path,
                      trayectorias: Sequence[Trayectoria]) -> dict[str, str]:
    """Describe con qué datos se entrenó, para los metadatos de los modelos guardados.

    Args:
        ruta_demostraciones: NPZ de demostraciones del que salen las partidas.
        ruta_splits: JSON de splits usado.
        trayectorias: Las partidas cargadas de ese NPZ.

    Returns:
        Las rutas (relativas a la carpeta de trabajo, o solo el nombre), la
        huella de cada archivo (huella_dataset, huella_splits) y la del
        contenido de las partidas (huella_partidas), que no cambia con los
        metadatos del NPZ.

    Raises:
        FileNotFoundError: Si falta alguno de los archivos.
        ValueError: Si trayectorias no es una secuencia de Trayectoria.
    """
    ruta_demostraciones, ruta_splits = Path(ruta_demostraciones), Path(ruta_splits)
    return {
        "demostraciones": ruta_registrada(ruta_demostraciones),
        "splits": ruta_registrada(ruta_splits),
        "huella_dataset": huella_archivo(ruta_demostraciones),
        "huella_splits": huella_archivo(ruta_splits),
        "huella_partidas": huella_partidas(trayectorias),
    }


def comprobar_config_comun(cfg: dict) -> None:
    """Comprueba lo que comparten los barridos de BC y GAIL antes de entrenar nada.

    La sección demostraciones con frecuencia_pares (los pares de cada partida
    salen de pares_de_subconjunto en los dos), experimento.repeticiones y
    experimento.procesos (enteros positivos) y experimento.semilla_repeticiones.

    Raises:
        ValueError: Si falta alguna clave o algún valor no es válido.
    """
    dcfg = config_demostraciones(cfg)
    if not es_entero_positivo(dcfg.get("frecuencia_pares")):
        raise ValueError("demostraciones.frecuencia_pares debe ser un entero positivo, no "
                         f"{dcfg.get('frecuencia_pares')!r}")
    ecfg = cfg.get("experimento")
    if not isinstance(ecfg, dict):
        raise ValueError("La configuración necesita la sección experimento")
    for clave in ("repeticiones", "procesos"):
        if not es_entero_positivo(ecfg.get(clave)):
            raise ValueError(f"experimento.{clave} debe ser un entero positivo, "
                             f"no {ecfg.get(clave)!r}")
    try:
        validar_semilla(ecfg.get("semilla_repeticiones"))
    except ValueError as error:
        raise ValueError(f"experimento.semilla_repeticiones no es válida: {error}") from error


def cargar_datos(cfg: dict, carpeta: Path) -> tuple[list[Trayectoria], SplitsDemostraciones,
                                                    dict[str, str]]:
    """Carga las demostraciones y los splits de una carpeta de get_data.py.

    Valida las partidas frente a los espacios del entorno de la configuración y
    los splits frente a la huella del NPZ, que tienen que citar.

    Returns:
        Las partidas, los splits y su procedencia (procedencia_datos).

    Raises:
        OSError: Si falta o no se puede leer algún archivo.
        ValueError: Si los datos no son válidos o los splits son de otro NPZ.
    """
    ruta_demostraciones = Path(carpeta) / ARCHIVO_DEMOSTRACIONES
    ruta_splits = Path(carpeta) / ARCHIVO_SPLITS
    trayectorias, _ = cargar_demostraciones(ruta_demostraciones, *espacios_entorno(cfg))
    splits = cargar_splits(ruta_splits, huella_archivo(ruta_demostraciones),
                           [t.id_episodio for t in trayectorias])
    return trayectorias, splits, procedencia_datos(ruta_demostraciones, ruta_splits, trayectorias)


def elegir_tareas(cfg: dict, splits: SplitsDemostraciones, cantidades: Sequence[int] | None,
                  repeticiones: Sequence[int] | None) -> list[tuple[int, int]]:
    """Combina cantidades y repeticiones en las tareas del barrido, cantidad a cantidad.

    Sin cantidades, las de demostraciones.subconjuntos de la configuración (los
    splits pueden tener más); sin repeticiones, de 0 a experimento.repeticiones − 1.

    Raises:
        ValueError: Si hay cantidades que no están en los splits o repeticiones
            fuera de rango, o alguna repetida.
    """
    cantidades = list(cantidades or cfg["demostraciones"]["subconjuntos"])
    total = cfg["experimento"]["repeticiones"]
    repeticiones = list(repeticiones) if repeticiones is not None else list(range(total))
    no_validas = [c for c in cantidades if c not in splits.subconjuntos]
    if no_validas or len(set(cantidades)) != len(cantidades):
        raise ValueError(f"cantidades no válidas o repetidas: {cantidades}; los splits tienen "
                         f"subconjuntos de {sorted(splits.subconjuntos)}")
    if (any(not 0 <= r < total for r in repeticiones)
            or len(set(repeticiones)) != len(repeticiones)):
        raise ValueError(f"repeticiones no válidas o repetidas: {repeticiones}; van de 0 a "
                         f"{total - 1}")
    return [(cantidad, repeticion) for cantidad in cantidades for repeticion in repeticiones]


def preparar_proceso_barrido() -> None:
    """Prepara un proceso del barrido en paralelo: un hilo de PyTorch y sin Ctrl+C.

    Con un solo hilo, el resultado es el mismo que en serie (con varios cambia el
    orden de las sumas y el redondeo) y los procesos no se reparten los núcleos.
    Ctrl+C lo recoge solo el proceso principal, que termina los demás; si no, cada
    proceso imprimiría su propia traza de error.
    """
    torch.set_num_threads(1)
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _ejecutar(tarea: Callable[..., dict[str, Any]],
              argumentos: tuple[Any, ...]) -> tuple[dict[str, Any] | None, str | None]:
    """Ejecuta tarea(*argumentos) y devuelve (resumen, None) o (None, error) si falla."""
    try:
        return tarea(*argumentos), None
    except Exception as error:  # noqa: BLE001 - se informa y sigue el barrido
        return None, repr(error)


def ejecutar_tarea_barrido(conexion: Connection, tarea: Callable[..., dict[str, Any]],
                           argumentos: tuple[Any, ...]) -> None:
    """Cuerpo de cada proceso del barrido en paralelo: prepara el proceso, ejecuta y envía.

    Prepara el proceso con preparar_proceso_barrido, ejecuta tarea(*argumentos)
    y envía (resumen, error) por conexion. Si el proceso muere antes de
    enviarlo, por ejemplo porque el sistema lo mata por falta de memoria, la
    conexión se cierra sin datos y quien espera recibe EOFError en lugar de
    quedarse esperando para siempre.

    Args:
        conexion: Extremo de envío de un Pipe; se cierra al terminar.
        tarea: Función de nivel de módulo (el proceso nuevo la importa).
        argumentos: Lo que recibe tarea.
    """
    preparar_proceso_barrido()
    try:
        conexion.send(_ejecutar(tarea, argumentos))
    finally:
        conexion.close()


def ejecutar_barrido(tareas: Sequence[tuple[int, int]], procesos: int,
                     tarea: Callable[..., dict[str, Any]],
                     argumentos: Callable[[int, int], tuple[Any, ...]],
                     informar: Callable[[ResultadoTarea], None]) -> list[tuple[int, int]]:
    """Ejecuta tarea(*argumentos(cantidad, repeticion)) para cada tarea, en serie o en procesos.

    Siempre con un solo hilo de PyTorch: así el resultado no depende de cuántos
    procesos se usen (con varios hilos cambia el redondeo) y, en paralelo, los
    procesos no se reparten los núcleos. Con más de un proceso, cada tarea va
    en un proceso nuevo (spawn, ejecutar_tarea_barrido), con como mucho procesos
    a la vez: cuando uno acaba, empieza el siguiente. Cada proceso devuelve su
    resultado por un Pipe; si muere sin enviarlo o a mitad del envío (por
    ejemplo, el sistema lo mata por falta de memoria), su tarea cuenta como
    fallida y el resto sigue: no se queda esperando para siempre, como le
    pasaba a Pool. Un fallo se informa y no detiene las demás. Ctrl+C termina al
    momento todos los procesos que no hayan acabado, también el que se esté
    recibiendo, y se propaga como KeyboardInterrupt.

    Args:
        tareas: Pares (cantidad, repeticion).
        procesos: Cuántas tareas a la vez (1 = en serie, en este proceso).
        tarea: Función de nivel de módulo que entrena y guarda; devuelve un resumen.
        argumentos: Construye los argumentos de tarea para cada (cantidad, repeticion).
        informar: Recibe (cantidad, repeticion, resumen, error) de cada tarea al acabar.

    Returns:
        Las tareas que fallaron.
    """
    fallos = []

    def anotar(cantidad: int, repeticion: int, resumen: dict[str, Any] | None,
               error: str | None) -> None:
        """Pasa el resultado a informar y anota los fallos."""
        if error is not None:
            fallos.append((cantidad, repeticion))
        informar((cantidad, repeticion, resumen, error))

    if procesos == 1:
        hilos_previos = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            for cantidad, repeticion in tareas:
                anotar(cantidad, repeticion, *_ejecutar(tarea, argumentos(cantidad, repeticion)))
        finally:
            torch.set_num_threads(hilos_previos)
        return fallos
    contexto = multiprocessing.get_context("spawn")
    pendientes = deque(tareas)
    activos: dict[Any, tuple[Any, int, int]] = {}  # extremo de lectura -> (proceso, c, r)
    try:
        while pendientes or activos:
            while pendientes and len(activos) < procesos:
                cantidad, repeticion = pendientes.popleft()
                lectura, escritura = contexto.Pipe(duplex=False)
                proceso = contexto.Process(target=ejecutar_tarea_barrido,
                                           args=(escritura, tarea,
                                                 argumentos(cantidad, repeticion)))
                proceso.start()
                # En activos nada más arrancar: un Ctrl+C a partir de aquí lo termina.
                activos[lectura] = (proceso, cantidad, repeticion)
                # Solo el proceso escribe: si muere, la lectura da EOFError.
                escritura.close()
            for lectura in wait(list(activos)):
                # Sigue en activos hasta que termina del todo: si llega un Ctrl+C
                # mientras se recibe o se espera, el finally también lo termina.
                proceso, cantidad, repeticion = activos[lectura]
                try:
                    resumen, error = lectura.recv()
                except (EOFError, OSError) as fallo:
                    # EOFError: murió sin enviar nada; OSError: murió a mitad del
                    # envío (mensaje truncado). En los dos, su tarea falla y las
                    # demás siguen.
                    proceso.join()
                    resumen, error = None, (
                        f"el proceso terminó sin resultado (código de salida "
                        f"{proceso.exitcode}; negativo si lo mató una señal, por "
                        f"ejemplo por falta de memoria): {fallo!r}")
                lectura.close()
                proceso.join()
                del activos[lectura]
                anotar(cantidad, repeticion, resumen, error)
    finally:
        # También por Ctrl+C: no quedan procesos entrenando por detrás.
        for proceso, _, _ in activos.values():
            proceso.terminate()
        for proceso, _, _ in activos.values():
            proceso.join()
    return fallos


@dataclass(frozen=True)
class MetodoBarrido:
    """Lo que distingue el barrido de un método (BC o GAIL) en main_barrido.

    Attributes:
        nombre: Nombre corto, que da la carpeta por defecto checkpoints/<nombre>/...
        comprobar_config: Valida la configuración del barrido antes de entrenar.
        tarea: Función de nivel de módulo (cfg, trayectorias, splits, cantidad,
            repeticion, destino, datos) que entrena, guarda y devuelve un resumen.
        destino: Dónde se guarda cada (cantidad, repeticion) dentro de la salida.
        describir: Una línea con el resultado de un resumen, para la terminal.
    """

    nombre: str
    comprobar_config: Callable[[dict], None]
    tarea: Callable[..., dict[str, Any]]
    destino: Callable[[Path, int, int], Path]
    describir: Callable[[dict[str, Any]], str]


def main_barrido(metodo: MetodoBarrido, descripcion: str,
                 argv: Sequence[str] | None = None) -> int:
    """Flujo de línea de comandos de un barrido; devuelve el código de salida.

    Resuelve y comprueba la configuración, comprueba que la salida está libre,
    carga los datos, elige las tareas, las ejecuta (ejecutar_barrido, con
    experimento.procesos a la vez) e informa de cada una y del total.

    Códigos: 0 si todo va bien; 1 si algún entrenamiento falló (los demás se
    guardan); 2 si la configuración, los datos, las cantidades, las
    repeticiones o la carpeta de salida no son válidos, sin entrenar nada; 130
    si se interrumpe con Ctrl+C.

    Args:
        metodo: Qué se entrena.
        descripcion: Ayuda del script (su docstring).
        argv: Argumentos; None para los de la línea de comandos.
    """
    ap = argparse.ArgumentParser(description=descripcion,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"),
                    help="configuración base")
    ap.add_argument("--entorno", choices=[e.value for e in TipoEntorno],
                    default=TipoEntorno.CARTPOLE.value, help="perfil de entorno")
    ap.add_argument("--datos", type=Path, required=True,
                    help=f"carpeta de get_data.py ({ARCHIVO_DEMOSTRACIONES} y {ARCHIVO_SPLITS})")
    ap.add_argument("--cantidades", type=int, nargs="+", default=None,
                    help="tamaños de subconjunto, que deben estar en los splits (por defecto, "
                         "demostraciones.subconjuntos de la configuración)")
    ap.add_argument("--repeticiones", type=int, nargs="+", default=None,
                    help="repeticiones, de 0 a experimento.repeticiones − 1 (por defecto, todas)")
    ap.add_argument("--salida", type=Path, default=None,
                    help=f"carpeta nueva para los modelos (por defecto "
                         f"checkpoints/{metodo.nombre}/<entorno>/<fecha>)")
    ap.add_argument("--sin-wandb", action="store_true", help="no registrar en W&B")
    ap.add_argument("--grupo", default=None, help="grupo W&B compartido con otros scripts")
    args = ap.parse_args(argv)

    ajustes = {"wandb": {"modo": "disabled"}} if args.sin_wandb else None
    try:
        cfg = resolver_configuracion(args.config, args.entorno, ajustes)
        # Toda la del barrido, antes de entrenar: si no, cada entrenamiento fallaría
        # por separado y el script saldría con 1 en vez de 2.
        metodo.comprobar_config(cfg)
    except (OSError, ValueError) as error:
        print(f"error de configuración: {error}", file=sys.stderr)
        return 2
    salida = args.salida or (Path("checkpoints") / metodo.nombre / args.entorno
                             / datetime.now().strftime("%Y%m%d-%H%M%S"))
    # Antes de entrenar: si la salida está ocupada, el entrenamiento se perdería.
    try:
        comprobar_directorio_libre(salida)
    except FileExistsError as error:
        print(f"error: {error}; elige otra --salida", file=sys.stderr)
        return 2
    try:
        trayectorias, splits, datos = cargar_datos(cfg, args.datos)
    except (OSError, ValueError) as error:
        print(f"error con los datos: {error}", file=sys.stderr)
        return 2
    try:
        tareas = elegir_tareas(cfg, splits, args.cantidades, args.repeticiones)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    try:
        comprobar_wandb(cfg, exigir_sesion=True)
    except RuntimeError as error:
        print(f"error de W&B: {error}", file=sys.stderr)
        return 2
    grupo = args.grupo or crear_grupo()
    print(f"grupo W&B: {grupo}")

    def informar(resultado: ResultadoTarea) -> None:
        """Imprime el resultado de un entrenamiento."""
        cantidad, repeticion, resumen, error = resultado
        if error is not None:
            print(f"error en cantidad {cantidad}, repetición {repeticion}: {error}",
                  file=sys.stderr, flush=True)
        else:
            print(f"cantidad {cantidad}, repetición {repeticion}: {metodo.describir(resumen)}",
                  flush=True)

    inicio = time.perf_counter()
    try:
        fallos = ejecutar_barrido(
            tareas, min(int(cfg["experimento"]["procesos"]), len(tareas)), metodo.tarea,
            lambda c, r: (cfg, trayectorias, splits, c, r, metodo.destino(salida, c, r),
                          datos, grupo),
            informar)
    except KeyboardInterrupt:
        # Los guardados se escriben enteros o nada (src/utils/archivos.py); un proceso
        # terminado a mitad de guardar puede dejar un temporal oculto, nunca uno a medias.
        print(f"interrumpido: los entrenamientos que terminaron están en {salida}, completos; "
              "si alguno se estaba guardando, puede quedar un temporal oculto (.*.tmp) que "
              "se puede borrar", file=sys.stderr)
        return 130
    print(f"{len(tareas) - len(fallos)} de {len(tareas)} entrenamientos guardados en {salida}; "
          f"tiempo total medido: {time.perf_counter() - inicio:.0f} s")
    avisar_estado_git(estado_git())
    return 1 if fallos else 0
