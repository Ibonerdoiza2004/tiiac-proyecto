"""Gestor de trabajos del dashboard: cola, límite de procesos, estados y canales.

Cada trabajo (una sesión de juego, un entrenamiento, una comparación) corre en un
proceso nuevo (``spawn``), como el barrido de la E1, con un hilo de PyTorch y de
BLAS: el resultado no depende de cuántos trabajos haya a la vez y cada uno usa un
núcleo. Como mucho corren ``procesos`` trabajos a la vez (``experimento.procesos``);
los que están en pausa siguen ocupando su hueco, porque conservan su memoria y
continúan al reanudar.

- Los trabajos interactivos (sesiones de jugar y ver al experto) no esperan: si no
  hay hueco, ``crear`` lanza ``SinHueco`` (la API responde 503).
- Los demás esperan en una cola por orden de llegada.

La tarea se indica como texto, ``"modulo:funcion"``, y se importa dentro del proceso
hijo: así la API no importa ``src/training``. La función recibe
``(argumentos, canal)`` y devuelve un diccionario con el resultado. Por ``canal``
(``CanalTrabajo``) manda el progreso y las imágenes y lee las órdenes.

Comunicación con cada proceso, por tres tuberías de un solo sentido:

- *Mensajes* (hijo → API): el progreso y, al final, el resultado o el error. Si el
  proceso muere sin enviar el resultado (por ejemplo, por falta de memoria), la
  tubería se cierra y el trabajo cuenta como fallido; los demás siguen.
- *Imágenes* (hijo → API): JPEG en bytes, en una tubería propia para que no retrasen
  el progreso. La API guarda solo la última.
- *Órdenes* (API → hijo): diccionarios que la tarea lee entre pasos.

Pausar y reanudar envían ``SIGSTOP`` y ``SIGCONT`` al proceso: es exacto, el
proceso sigue después donde estaba. Cancelar lo termina sin guardar nada.

Quien quiera seguir los trabajos (el WebSocket de la API) se registra con
``observar``: recibe cada evento (estado, progreso, imagen) desde un único hilo
repartidor y en el orden en que ocurrieron.
"""

from __future__ import annotations

import importlib
import multiprocessing
import os
import queue
import signal
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from multiprocessing.connection import Connection
from typing import Any

# Este módulo no importa NumPy ni PyTorch arriba: el proceso hijo lo importa antes de
# fijar VARIABLES_UN_HILO, y NumPy lee esas variables al importarse.

# Variables que limitan a un hilo las bibliotecas numéricas del proceso hijo. Se fijan
# antes de importar la tarea, que es cuando esas bibliotecas leen su valor.
VARIABLES_UN_HILO = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
# Segundos que se espera a que un proceso termine tras pedírselo antes de matarlo.
ESPERA_TERMINAR = 5.0


class EstadoTrabajo(str, Enum):
    """Estados de un trabajo."""

    EN_COLA = "en_cola"
    EN_MARCHA = "en_marcha"
    PAUSADO = "pausado"
    TERMINADO = "terminado"
    FALLIDO = "fallido"
    CANCELADO = "cancelado"


# Los que ocupan un hueco del límite y los que ya no cambian.
ESTADOS_VIVOS = frozenset({EstadoTrabajo.EN_MARCHA, EstadoTrabajo.PAUSADO})
ESTADOS_FINALES = frozenset({EstadoTrabajo.TERMINADO, EstadoTrabajo.FALLIDO,
                             EstadoTrabajo.CANCELADO})


class SinHueco(RuntimeError):
    """Un trabajo interactivo no puede empezar porque el límite de procesos está lleno."""


class TrabajoDesconocido(KeyError):
    """No hay ningún trabajo con ese identificador."""


class OperacionNoValida(RuntimeError):
    """La operación no se puede hacer en el estado actual del trabajo."""


Evento = dict[str, Any]
Observador = Callable[[str, Evento], None]


class CanalTrabajo:
    """Lo que recibe la tarea en el proceso hijo para hablar con la API."""

    def __init__(self, mensajes: Connection, imagenes: Connection, ordenes: Connection):
        """Guarda los extremos del hijo de las tres tuberías.

        Args:
            mensajes: Envío de progreso, resultado y error.
            imagenes: Envío de imágenes.
            ordenes: Recepción de órdenes.
        """
        self._mensajes = mensajes
        self._imagenes = imagenes
        self._ordenes = ordenes

    def progreso(self, medidas: Mapping[str, Any]) -> None:
        """Envía una medida de progreso (por ejemplo, tras cada iteración o época).

        Args:
            medidas: Diccionario que se pueda pasar por una tubería (pickle).
        """
        self._mensajes.send(("progreso", dict(medidas)))

    def imagen(self, datos: bytes) -> None:
        """Envía una imagen del entorno (JPEG). La API se queda solo con la última.

        Args:
            datos: Bytes de la imagen.
        """
        self._imagenes.send_bytes(bytes(datos))

    def ordenes(self) -> list[dict[str, Any]]:
        """Devuelve las órdenes recibidas desde la última llamada, sin esperar.

        Returns:
            Las órdenes en el orden en que llegaron (lista vacía si no hay).
        """
        recibidas = []
        while self._ordenes.poll():
            recibidas.append(self._ordenes.recv())
        return recibidas


def resolver_tarea(ruta: str) -> Callable[[dict[str, Any], CanalTrabajo], dict[str, Any]]:
    """Importa y devuelve la función de una ruta ``"modulo:funcion"``.

    Args:
        ruta: Módulo importable y nombre de la función, separados por dos puntos.

    Returns:
        La función.

    Raises:
        ValueError: Si la ruta no tiene el formato, el módulo no existe o lo que
            nombra no es una función.
    """
    comprobar_ruta_tarea(ruta)
    modulo, nombre = ruta.split(":")
    try:
        funcion = getattr(importlib.import_module(modulo), nombre)
    except (ImportError, AttributeError) as error:
        raise ValueError(f"No se encuentra la tarea {ruta!r}: {error!r}") from error
    if not callable(funcion):
        raise ValueError(f"La tarea {ruta!r} no es una función")
    return funcion


def comprobar_ruta_tarea(ruta: object) -> None:
    """Comprueba que la ruta tenga el formato ``"modulo:funcion"``, sin importarla.

    Raises:
        ValueError: Si no es un texto con un módulo y un nombre separados por «:».
    """
    partes = ruta.split(":") if isinstance(ruta, str) else []
    if len(partes) != 2 or not all(parte.strip() == parte and parte for parte in partes):
        raise ValueError(f"La tarea debe ser 'modulo:funcion', no {ruta!r}")


def ejecutar_en_proceso(ruta: str, argumentos: dict[str, Any], mensajes: Connection,
                        imagenes: Connection, ordenes: Connection) -> None:
    """Cuerpo del proceso hijo: prepara el proceso, ejecuta la tarea y envía el final.

    Limita las bibliotecas numéricas a un hilo antes de importar la tarea, ignora
    Ctrl+C (lo gestiona la API) y envía ``("resultado", dict)`` o
    ``("error", texto)`` por la tubería de mensajes.

    Args:
        ruta: Tarea ``"modulo:funcion"``.
        argumentos: Lo que recibe la tarea.
        mensajes: Extremo de envío de los mensajes.
        imagenes: Extremo de envío de las imágenes.
        ordenes: Extremo de recepción de las órdenes.
    """
    for variable in VARIABLES_UN_HILO:
        os.environ[variable] = "1"
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        # PyTorch, NumPy y BLAS se importan aquí, ya con VARIABLES_UN_HILO: arrancan con
        # un hilo, y así el resultado no depende de cuántos trabajos haya a la vez.
        tarea = resolver_tarea(ruta)
        resultado = tarea(argumentos, CanalTrabajo(mensajes, imagenes, ordenes))
        if not isinstance(resultado, dict):
            raise TypeError(f"La tarea debe devolver un diccionario, no {type(resultado).__name__}")
        mensajes.send(("resultado", resultado))
    except Exception as error:  # noqa: BLE001 - se informa a la API, que sigue
        mensajes.send(("error", repr(error)))
    finally:
        for conexion in (mensajes, imagenes, ordenes):
            conexion.close()


@dataclass
class _Trabajo:
    """Un trabajo y sus recursos. Solo lo toca el gestor, con su cerrojo."""

    id: str
    tipo: str
    tarea: str
    argumentos: dict[str, Any]
    interactivo: bool
    creado: float
    estado: EstadoTrabajo = EstadoTrabajo.EN_COLA
    iniciado: float | None = None
    terminado: float | None = None
    progreso: list[dict[str, Any]] = field(default_factory=list)
    resultado: dict[str, Any] | None = None
    error: str | None = None
    imagen: bytes | None = None
    cancelado: bool = False
    proceso: Any = None
    ordenes: Connection | None = None

    def describir(self) -> dict[str, Any]:
        """Datos públicos del trabajo (sin el historial de progreso ni la imagen)."""
        return {"id": self.id, "tipo": self.tipo, "estado": self.estado.value,
                "argumentos": self.argumentos, "interactivo": self.interactivo,
                "creado": self.creado, "iniciado": self.iniciado,
                "terminado": self.terminado, "progresos": len(self.progreso),
                "resultado": self.resultado, "error": self.error}


class GestorTrabajos:
    """Cola y procesos de los trabajos, con como mucho ``procesos`` a la vez."""

    def __init__(self, procesos: int, contexto: Any = None):
        """Crea el gestor vacío.

        Args:
            procesos: Trabajos que pueden correr a la vez (experimento.procesos).
            contexto: Contexto de multiprocessing (por defecto, spawn).

        Raises:
            ValueError: Si procesos no es un entero positivo.
        """
        # Aquí y no arriba: importa NumPy, y el proceso hijo no debe cargarlo antes de
        # fijar los hilos (solo el proceso de la API crea gestores).
        from src.utils.validacion import es_entero_positivo

        if not es_entero_positivo(procesos):
            raise ValueError(f"procesos debe ser un entero positivo, no {procesos!r}")
        self.procesos = int(procesos)
        self._contexto = contexto or multiprocessing.get_context("spawn")
        self._cerrojo = threading.Condition()
        self._trabajos: dict[str, _Trabajo] = {}
        self._cola: deque[str] = deque()
        self._observadores: list[Observador] = []
        self._cerrado = False
        # Los eventos se encolan con el cerrojo tomado (así quedan en el orden en que
        # ocurren) y los reparte un solo hilo, sin el cerrojo.
        self._eventos: queue.SimpleQueue[tuple[str, Evento] | None] = queue.SimpleQueue()
        threading.Thread(target=self._repartir, name="repartidor-trabajos",
                         daemon=True).start()

    # --- Consultas ----------------------------------------------------------

    def _buscar(self, id_trabajo: str) -> _Trabajo:
        """Devuelve el trabajo o lanza TrabajoDesconocido. Con el cerrojo tomado."""
        try:
            return self._trabajos[id_trabajo]
        except KeyError:
            raise TrabajoDesconocido(id_trabajo) from None

    def _vivos(self) -> int:
        """Trabajos que ocupan hueco (en marcha o en pausa). Con el cerrojo tomado."""
        return sum(t.estado in ESTADOS_VIVOS for t in self._trabajos.values())

    def obtener(self, id_trabajo: str) -> dict[str, Any]:
        """Datos públicos de un trabajo.

        Raises:
            TrabajoDesconocido: Si no existe.
        """
        with self._cerrojo:
            return self._buscar(id_trabajo).describir()

    def listar(self) -> list[dict[str, Any]]:
        """Datos públicos de todos los trabajos, por orden de creación."""
        with self._cerrojo:
            return [t.describir() for t in self._trabajos.values()]

    def historial(self, id_trabajo: str) -> list[dict[str, Any]]:
        """Todos los mensajes de progreso del trabajo, en orden (para reconectarse).

        Raises:
            TrabajoDesconocido: Si no existe.
        """
        with self._cerrojo:
            return list(self._buscar(id_trabajo).progreso)

    def ultima_imagen(self, id_trabajo: str) -> bytes | None:
        """La última imagen que envió el trabajo, o None.

        Raises:
            TrabajoDesconocido: Si no existe.
        """
        with self._cerrojo:
            return self._buscar(id_trabajo).imagen

    def libres(self) -> int:
        """Huecos libres: procesos menos los trabajos en marcha o en pausa."""
        with self._cerrojo:
            return self.procesos - self._vivos()

    def esperar(self, id_trabajo: str, timeout: float | None = None) -> dict[str, Any]:
        """Espera a que el trabajo termine, falle o se cancele.

        Args:
            id_trabajo: Trabajo.
            timeout: Segundos como mucho (None, sin límite).

        Returns:
            Sus datos públicos.

        Raises:
            TrabajoDesconocido: Si no existe.
            TimeoutError: Si no acaba a tiempo.
        """
        with self._cerrojo:
            trabajo = self._buscar(id_trabajo)
            if not self._cerrojo.wait_for(lambda: trabajo.estado in ESTADOS_FINALES, timeout):
                raise TimeoutError(f"El trabajo {id_trabajo} sigue {trabajo.estado.value}")
            return trabajo.describir()

    # --- Observadores -------------------------------------------------------

    def observar(self, observador: Observador) -> Callable[[], None]:
        """Registra una función que recibe (id, evento) de todos los trabajos.

        Los eventos son ``{"tipo": "estado", "trabajo": datos públicos}``,
        ``{"tipo": "progreso", "medidas": ...}`` e ``{"tipo": "imagen", "datos": bytes}``.
        Se llaman desde un único hilo repartidor, en el orden en que ocurrieron y sin
        el cerrojo tomado: tienen que ser rápidos (si lanzan una excepción, se ignora).

        Returns:
            Una función que deja de observar.
        """
        with self._cerrojo:
            self._observadores.append(observador)

        def dejar() -> None:
            """Quita el observador (no falla si ya no estaba)."""
            with self._cerrojo:
                if observador in self._observadores:
                    self._observadores.remove(observador)
        return dejar

    def _encolar(self, id_trabajo: str, evento: Evento) -> None:
        """Deja el evento para el repartidor. Con el cerrojo tomado."""
        self._eventos.put((id_trabajo, evento))

    def _encolar_estado(self, trabajo: _Trabajo) -> None:
        """Deja un evento con el estado actual del trabajo. Con el cerrojo tomado."""
        self._encolar(trabajo.id, {"tipo": "estado", "trabajo": trabajo.describir()})

    def _repartir(self) -> None:
        """Hilo repartidor: pasa cada evento a los observadores, en orden, sin el cerrojo."""
        while (elemento := self._eventos.get()) is not None:
            id_trabajo, evento = elemento
            with self._cerrojo:
                observadores = list(self._observadores)
            for observador in observadores:
                try:
                    observador(id_trabajo, evento)
                except Exception:  # noqa: BLE001 - un observador roto no para el gestor
                    pass

    # --- Crear y lanzar -----------------------------------------------------

    def crear(self, tipo: str, tarea: str, argumentos: Mapping[str, Any] | None = None,
              interactivo: bool = False) -> dict[str, Any]:
        """Crea un trabajo: empieza si hay hueco y, si no, espera en la cola.

        Args:
            tipo: Qué es (por ejemplo "sesion", "entrenamiento" o "comparacion").
            tarea: Función ``"modulo:funcion"`` que se ejecuta en el proceso hijo.
            argumentos: Lo que recibe la tarea (tiene que poder pasarse por pickle).
            interactivo: Si es una sesión que no puede esperar en la cola.

        Returns:
            Los datos públicos del trabajo nuevo.

        Raises:
            ValueError: Si el tipo, la tarea o los argumentos no son válidos.
            SinHueco: Si es interactivo y no queda ningún hueco.
            RuntimeError: Si el gestor ya está cerrado.
        """
        if not isinstance(tipo, str) or not tipo:
            raise ValueError(f"tipo debe ser un texto no vacío, no {tipo!r}")
        comprobar_ruta_tarea(tarea)
        if argumentos is not None and not isinstance(argumentos, Mapping):
            raise ValueError("argumentos debe ser un diccionario o None")
        trabajo = _Trabajo(id=uuid.uuid4().hex, tipo=tipo, tarea=tarea,
                           argumentos=dict(argumentos or {}), interactivo=bool(interactivo),
                           creado=time.time())
        with self._cerrojo:
            if self._cerrado:
                raise RuntimeError("El gestor de trabajos está cerrado")
            if trabajo.interactivo and self._vivos() >= self.procesos:
                raise SinHueco(f"No hay núcleos libres: ya hay {self.procesos} trabajos en "
                               "marcha. Espera a que acabe alguno o finaliza otro.")
            self._trabajos[trabajo.id] = trabajo
            self._cola.append(trabajo.id)
            self._lanzar_pendientes()
            if trabajo.estado is EstadoTrabajo.EN_COLA:
                self._encolar_estado(trabajo)
            return trabajo.describir()

    def _lanzar_pendientes(self) -> None:
        """Arranca los de la cola mientras haya hueco y avisa. Con el cerrojo tomado."""
        while self._cola and self._vivos() < self.procesos:
            trabajo = self._trabajos[self._cola.popleft()]
            self._arrancar(trabajo)
            self._encolar_estado(trabajo)

    def _arrancar(self, trabajo: _Trabajo) -> None:
        """Crea las tuberías y el proceso del trabajo y sus hilos de lectura."""
        lectura_mensajes, escritura_mensajes = self._contexto.Pipe(duplex=False)
        lectura_imagenes, escritura_imagenes = self._contexto.Pipe(duplex=False)
        lectura_ordenes, escritura_ordenes = self._contexto.Pipe(duplex=False)
        proceso = self._contexto.Process(
            target=ejecutar_en_proceso, name=f"trabajo-{trabajo.id}",
            args=(trabajo.tarea, trabajo.argumentos, escritura_mensajes,
                  escritura_imagenes, lectura_ordenes))
        proceso.start()
        # Los extremos del hijo se cierran aquí: si el hijo muere, la lectura da EOFError.
        for extremo in (escritura_mensajes, escritura_imagenes, lectura_ordenes):
            extremo.close()
        trabajo.proceso = proceso
        trabajo.ordenes = escritura_ordenes
        trabajo.estado = EstadoTrabajo.EN_MARCHA
        trabajo.iniciado = time.time()
        threading.Thread(target=self._leer_imagenes, args=(trabajo, lectura_imagenes),
                         name=f"imagenes-{trabajo.id}", daemon=True).start()
        threading.Thread(target=self._leer_mensajes, args=(trabajo, lectura_mensajes),
                         name=f"mensajes-{trabajo.id}", daemon=True).start()

    # --- Hilos de lectura ---------------------------------------------------

    def _leer_imagenes(self, trabajo: _Trabajo, lectura: Connection) -> None:
        """Guarda la última imagen del trabajo y avisa, hasta que el hijo cierra."""
        try:
            while True:
                try:
                    datos = lectura.recv_bytes()
                except (EOFError, OSError):
                    return
                with self._cerrojo:
                    trabajo.imagen = datos
                    self._encolar(trabajo.id, {"tipo": "imagen", "datos": datos})
        finally:
            lectura.close()

    def _leer_mensajes(self, trabajo: _Trabajo, lectura: Connection) -> None:
        """Recoge progreso y final del trabajo; al acabar, fija su estado y libera el hueco."""
        resultado: dict[str, Any] | None = None
        error: str | None = None
        try:
            while True:
                try:
                    clase, contenido = lectura.recv()
                except (EOFError, OSError):
                    break
                if clase == "progreso":
                    with self._cerrojo:
                        trabajo.progreso.append(contenido)
                        self._encolar(trabajo.id, {"tipo": "progreso", "medidas": contenido})
                elif clase == "resultado":
                    resultado = contenido
                elif clase == "error":
                    error = contenido
        finally:
            lectura.close()
        proceso = trabajo.proceso
        proceso.join(ESPERA_TERMINAR)
        if proceso.is_alive():
            proceso.kill()
            proceso.join()
        with self._cerrojo:
            if trabajo.cancelado:
                trabajo.estado = EstadoTrabajo.CANCELADO
            elif resultado is not None:
                trabajo.estado, trabajo.resultado = EstadoTrabajo.TERMINADO, resultado
            else:
                trabajo.estado = EstadoTrabajo.FALLIDO
                trabajo.error = error or (
                    f"El proceso terminó sin resultado (código de salida {proceso.exitcode}; "
                    "negativo si lo mató una señal, por ejemplo por falta de memoria)")
            trabajo.terminado = time.time()
            trabajo.ordenes.close()
            self._encolar_estado(trabajo)
            if not self._cerrado:
                self._lanzar_pendientes()
            self._cerrojo.notify_all()

    # --- Control ------------------------------------------------------------

    def pausar(self, id_trabajo: str) -> dict[str, Any]:
        """Detiene el proceso (SIGSTOP). Sigue ocupando su hueco.

        Raises:
            TrabajoDesconocido: Si no existe.
            OperacionNoValida: Si no está en marcha.
        """
        return self._senal(id_trabajo, EstadoTrabajo.EN_MARCHA, signal.SIGSTOP,
                           EstadoTrabajo.PAUSADO, "pausar")

    def reanudar(self, id_trabajo: str) -> dict[str, Any]:
        """Reanuda un proceso en pausa (SIGCONT), donde estaba.

        Raises:
            TrabajoDesconocido: Si no existe.
            OperacionNoValida: Si no está en pausa.
        """
        return self._senal(id_trabajo, EstadoTrabajo.PAUSADO, signal.SIGCONT,
                           EstadoTrabajo.EN_MARCHA, "reanudar")

    def _senal(self, id_trabajo: str, desde: EstadoTrabajo, senal: int,
               hacia: EstadoTrabajo, accion: str) -> dict[str, Any]:
        """Envía la señal si el trabajo está en el estado desde y pasa a hacia."""
        with self._cerrojo:
            trabajo = self._buscar(id_trabajo)
            if trabajo.estado is not desde or trabajo.cancelado:
                raise OperacionNoValida(f"No se puede {accion} un trabajo "
                                        f"{trabajo.estado.value}")
            try:
                os.kill(trabajo.proceso.pid, senal)
            except ProcessLookupError:
                raise OperacionNoValida(f"No se puede {accion}: el proceso ya ha "
                                        "terminado") from None
            trabajo.estado = hacia
            self._encolar_estado(trabajo)
            return trabajo.describir()

    def enviar_orden(self, id_trabajo: str, orden: Mapping[str, Any]) -> None:
        """Manda una orden al trabajo, que la lee entre pasos (CanalTrabajo.ordenes).

        Si está en pausa, la orden espera en la tubería hasta que se reanude.

        Raises:
            TrabajoDesconocido: Si no existe.
            OperacionNoValida: Si no está en marcha ni en pausa.
            ValueError: Si la orden no es un diccionario.
        """
        if not isinstance(orden, Mapping):
            raise ValueError("La orden debe ser un diccionario")
        with self._cerrojo:
            trabajo = self._buscar(id_trabajo)
            if trabajo.estado not in ESTADOS_VIVOS or trabajo.cancelado:
                raise OperacionNoValida(f"No se puede mandar una orden a un trabajo "
                                        f"{trabajo.estado.value}")
            try:
                trabajo.ordenes.send(dict(orden))
            except OSError as error:
                raise OperacionNoValida(f"El trabajo ya no recibe órdenes: {error!r}") from error

    def cancelar(self, id_trabajo: str) -> dict[str, Any]:
        """Cancela el trabajo sin guardar nada: lo quita de la cola o termina su proceso.

        Raises:
            TrabajoDesconocido: Si no existe.
            OperacionNoValida: Si ya había terminado, fallado o se había cancelado.
        """
        with self._cerrojo:
            trabajo = self._buscar(id_trabajo)
            if trabajo.estado in ESTADOS_FINALES or trabajo.cancelado:
                raise OperacionNoValida(f"No se puede cancelar un trabajo "
                                        f"{trabajo.estado.value}")
            trabajo.cancelado = True
            if trabajo.estado is EstadoTrabajo.EN_COLA:
                self._cola.remove(trabajo.id)
                trabajo.estado = EstadoTrabajo.CANCELADO
                trabajo.terminado = time.time()
                self._encolar_estado(trabajo)
                self._cerrojo.notify_all()
            else:
                # El hilo de mensajes fija CANCELADO (y avisa) al ver morir el proceso.
                self._terminar_proceso(trabajo)
            return trabajo.describir()

    @staticmethod
    def _terminar_proceso(trabajo: _Trabajo) -> None:
        """Termina el proceso: si está en pausa, primero lo reanuda para que reciba SIGTERM."""
        try:
            if trabajo.estado is EstadoTrabajo.PAUSADO:
                os.kill(trabajo.proceso.pid, signal.SIGCONT)
            trabajo.proceso.terminate()
        except ProcessLookupError:
            pass

    def cerrar(self, timeout: float = ESPERA_TERMINAR * 2) -> None:
        """Cancela todo (cola y procesos) y espera a que terminen. Al parar la API.

        Args:
            timeout: Segundos como mucho para esperar a los procesos.
        """
        with self._cerrojo:
            self._cerrado = True
            pendientes = [t for t in self._trabajos.values() if t.estado not in ESTADOS_FINALES]
            for trabajo in pendientes:
                trabajo.cancelado = True
                if trabajo.estado is EstadoTrabajo.EN_COLA:
                    trabajo.estado = EstadoTrabajo.CANCELADO
                    trabajo.terminado = time.time()
                    self._encolar_estado(trabajo)
                else:
                    self._terminar_proceso(trabajo)
            self._cola.clear()
            self._cerrojo.notify_all()
            self._cerrojo.wait_for(
                lambda: all(t.estado in ESTADOS_FINALES for t in pendientes), timeout)
            # El repartidor termina después de repartir lo que ya estaba en la cola.
            self._eventos.put(None)
