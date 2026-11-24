"""Rutas de los trabajos: listarlos, consultarlos, controlarlos y seguirlos en directo.

Es el protocolo único de todos los trabajos (ver contrato.py). No hay una ruta
genérica para crearlos: los crean las rutas de cada funcionalidad (sesiones,
entrenamientos, comparaciones), que deciden qué función ejecuta cada uno. Así el
navegador nunca elige qué código importa el servidor.

HTTP (prefijo /api):

- ``GET  /trabajos``                 Todos, por orden de creación (``?tipo=`` filtra).
- ``GET  /trabajos/{id}``            Uno.
- ``POST /trabajos/{id}/pausar``     Lo detiene; sigue ocupando su hueco.
- ``POST /trabajos/{id}/reanudar``   Sigue donde estaba.
- ``POST /trabajos/{id}/cancelar``   Lo quita de la cola o lo termina, sin guardar.

Errores: 404 si no existe, 409 si la operación no vale en su estado y 503 si una
sesión no encuentra hueco.

WebSocket ``/api/trabajos/{id}``, para seguir un trabajo y reconectarse a mitad:

- Al conectar, el servidor manda ``{"tipo": "estado", "trabajo": ...}``, después
  ``{"tipo": "historial", "progreso": [...], "eventos": [...]}`` con todo el progreso
  y los eventos propios hasta ese momento y, si la hay, la última imagen. Con eso el
  navegador redibuja todo.
- Después, en directo: ``{"tipo": "progreso", "medidas": {...}}``, cada imagen como
  mensaje binario (JPEG; si el navegador va lento, solo la más reciente),
  ``{"tipo": "estado", "trabajo": ...}`` en cada cambio de estado y los eventos propios
  de la funcionalidad tal cual (por ejemplo ``{"tipo": "partida_terminada", ...}``).
- Cuando el trabajo termina, falla o se cancela, manda ese último estado y cierra
  la conexión normalmente (código 1000).
- Del navegador, mensajes de texto JSON: ``{"tipo": "pausar"}``,
  ``{"tipo": "reanudar"}`` y ``{"tipo": "cancelar"}`` para cualquier trabajo, y las
  órdenes de su tipo (``ORDENES_POR_TIPO`` en contrato.py: ``accion`` y ``finalizar``
  en una sesión, ``vista`` y ``finalizar`` en un entrenamiento), que se validan y se
  pasan al trabajo. ``finalizar`` en pausa reanuda el trabajo para que pueda terminar.
- Un mensaje que no vale (también uno binario) recibe
  ``{"tipo": "error", "mensaje": ...}`` y la conexión sigue. Si el trabajo no existe,
  se manda ese error y se cierra con el código 4404.
- Cerrar la conexión no afecta a un entrenamiento, que sigue y admite reconectarse;
  una sesión interactiva (jugar o ver al experto), en cambio, se cancela cuando se
  cierra la última conexión que la seguía (otra pestaña abierta la mantiene).

Los números que no son finitos (NaN, infinito) se mandan como ``null``, también en
las respuestas HTTP, porque JSON no los admite.
"""

from __future__ import annotations

import asyncio
import json
import math
import threading
from collections import Counter
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.app.contrato import CONTROLES, ORDENES_QUE_REANUDAN, TrabajoApi, validar_orden
from src.app.trabajos import ESTADOS_FINALES, GestorTrabajos, OperacionNoValida, TrabajoDesconocido

# Código de cierre del WebSocket cuando el trabajo no existe (4000-4999: de la aplicación).
CIERRE_NO_EXISTE = 4404
_FINALES = {estado.value for estado in ESTADOS_FINALES}


class _Imagen:
    """Hueco de una imagen en la cola de envío; guarda la más reciente de su tanda."""

    def __init__(self, datos: bytes):
        """Hueco con su primera imagen."""
        self.datos = datos


class ColaEnvio:
    """Lo que falta por enviar por un WebSocket, en orden; de las imágenes, solo la última.

    Si llegan varias imágenes seguidas antes de enviar la primera, se envía una sola
    vez la más reciente: el navegador nunca recibe imágenes repetidas ni atrasadas.
    Una imagen nunca adelanta a un evento que llegó antes que ella: si entre dos
    imágenes hay otro evento, van las tres en su orden.
    """

    def __init__(self) -> None:
        """Crea la cola vacía. Se usa desde un único bucle de asyncio."""
        self._cola: asyncio.Queue[Any] = asyncio.Queue()
        # El hueco de imagen al final de la cola, si lo último que entró fue una imagen.
        self._ultima: _Imagen | None = None

    def poner(self, evento: dict[str, Any]) -> None:
        """Añade un evento del gestor (estado, progreso o imagen)."""
        if evento["tipo"] == "imagen":
            if self._ultima is not None:
                self._ultima.datos = evento["datos"]  # sustituye a la de su tanda
            else:
                self._ultima = _Imagen(evento["datos"])
                self._cola.put_nowait(self._ultima)
        else:
            self._ultima = None
            self._cola.put_nowait(evento)

    def vacia(self) -> bool:
        """Si no queda nada por enviar."""
        return self._cola.empty()

    async def sacar(self) -> dict[str, Any] | bytes:
        """Espera y devuelve lo siguiente: un evento o los bytes de una imagen."""
        evento = await self._cola.get()
        if isinstance(evento, _Imagen):
            if evento is self._ultima:
                self._ultima = None  # ya salió: la siguiente imagen va en otro hueco
            return evento.datos
        return evento


def a_json_finito(valor: Any) -> Any:
    """Copia el valor para JSON: NaN e infinito pasan a None y los escalares de NumPy,
    a números de Python.

    Args:
        valor: Diccionarios, listas, tuplas, números, textos, booleanos o None.

    Returns:
        Una copia que ``json.dumps(..., allow_nan=False)`` acepta.
    """
    if isinstance(valor, dict):
        return {str(clave): a_json_finito(v) for clave, v in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [a_json_finito(v) for v in valor]
    if hasattr(valor, "item") and not isinstance(valor, (str, bytes)):
        valor = valor.item()  # escalar de NumPy
    if isinstance(valor, float) and not math.isfinite(valor):
        return None
    return valor


def crear_router_trabajos(gestor: GestorTrabajos) -> APIRouter:
    """Rutas HTTP y WebSocket de los trabajos del gestor.

    Args:
        gestor: El gestor de trabajos de la aplicación.

    Returns:
        Un router con las rutas bajo /trabajos (se monta con el prefijo /api).
    """
    router = APIRouter(prefix="/trabajos", tags=["trabajos"])
    conexiones = Conexiones()

    @router.get("", response_model=list[TrabajoApi])
    def listar(tipo: str | None = None) -> list[dict[str, Any]]:
        """Todos los trabajos, por orden de creación; con ``tipo``, solo los de ese tipo."""
        return [a_json_finito(t) for t in gestor.listar() if tipo is None or t["tipo"] == tipo]

    @router.get("/{id_trabajo}", response_model=TrabajoApi)
    def obtener(id_trabajo: str) -> dict[str, Any]:
        """Un trabajo."""
        return a_json_finito(gestor.obtener(id_trabajo))

    @router.post("/{id_trabajo}/pausar", response_model=TrabajoApi)
    def pausar(id_trabajo: str) -> dict[str, Any]:
        """Detiene el trabajo; sigue ocupando su hueco."""
        return a_json_finito(gestor.pausar(id_trabajo))

    @router.post("/{id_trabajo}/reanudar", response_model=TrabajoApi)
    def reanudar(id_trabajo: str) -> dict[str, Any]:
        """Reanuda un trabajo en pausa donde estaba."""
        return a_json_finito(gestor.reanudar(id_trabajo))

    @router.post("/{id_trabajo}/cancelar", response_model=TrabajoApi)
    def cancelar(id_trabajo: str) -> dict[str, Any]:
        """Lo quita de la cola o termina su proceso, sin guardar nada."""
        return a_json_finito(gestor.cancelar(id_trabajo))

    @router.websocket("/{id_trabajo}")
    async def seguir(websocket: WebSocket, id_trabajo: str) -> None:
        """Sigue un trabajo: foto inicial, eventos en directo y controles."""
        await seguir_trabajo(websocket, gestor, id_trabajo, conexiones)

    return router


class Conexiones:
    """Cuántos WebSockets siguen cada trabajo, para cancelar una sesión con el último."""

    def __init__(self) -> None:
        """Sin conexiones."""
        self._cerrojo = threading.Lock()
        self._abiertas: Counter[str] = Counter()

    def abrir(self, id_trabajo: str) -> None:
        """Cuenta una conexión más al trabajo."""
        with self._cerrojo:
            self._abiertas[id_trabajo] += 1

    def cerrar(self, id_trabajo: str) -> bool:
        """Descuenta una conexión; devuelve si era la última."""
        with self._cerrojo:
            self._abiertas[id_trabajo] -= 1
            if self._abiertas[id_trabajo] > 0:
                return False
            del self._abiertas[id_trabajo]
            return True


async def seguir_trabajo(websocket: WebSocket, gestor: GestorTrabajos, id_trabajo: str,
                         conexiones: Conexiones | None = None) -> None:
    """Cuerpo del WebSocket de un trabajo (protocolo en el docstring del módulo).

    Los eventos llegan desde el hilo repartidor del gestor y pasan al bucle de asyncio
    con ``call_soon_threadsafe``. De las imágenes solo se guarda la última: si el
    navegador va más lento que el trabajo, las intermedias se descartan (ColaEnvio).

    Args:
        websocket: Conexión sin aceptar todavía.
        gestor: Gestor de trabajos.
        id_trabajo: Trabajo que se sigue.
        conexiones: Conexiones abiertas por trabajo (por defecto, solo esta).
    """
    conexiones = conexiones or Conexiones()
    await websocket.accept()
    bucle = asyncio.get_running_loop()
    cola = ColaEnvio()

    def al_evento(evento: dict[str, Any]) -> None:
        """Llamado desde el hilo repartidor: pasa el evento al bucle de asyncio."""
        try:
            bucle.call_soon_threadsafe(cola.poner, evento)
        except RuntimeError:
            pass  # el bucle ya se ha cerrado: la conexión terminó

    try:
        foto, dejar = gestor.suscribir(id_trabajo, al_evento)
    except TrabajoDesconocido:
        await websocket.send_json({"tipo": "error",
                                   "mensaje": f"No existe el trabajo {id_trabajo}"})
        await websocket.close(code=CIERRE_NO_EXISTE)
        return

    # Si la última conexión de una sesión interactiva acaba por cualquier otro motivo
    # que el final del trabajo (el navegador cierra, se corta la red o el servidor
    # interrumpe esta función), la sesión se descarta: no sigue sin nadie mirando. Por
    # eso va en el finally y no después: una interrupción (CancelledError) se saltaría
    # el resto.
    acabado = False
    conexiones.abrir(id_trabajo)
    try:
        await _enviar_json(websocket, {"tipo": "estado", "trabajo": foto["trabajo"]})
        await _enviar_json(websocket, {"tipo": "historial", "progreso": foto["historial"],
                                       "eventos": foto["eventos"]})
        if foto["imagen"] is not None:
            await websocket.send_bytes(foto["imagen"])
        if foto["trabajo"]["estado"] in _FINALES:
            acabado = True
            await websocket.close()
            return
        enviar = asyncio.create_task(_enviar_eventos(websocket, cola))
        recibir = asyncio.create_task(_recibir_controles(websocket, gestor, id_trabajo,
                                                         foto["trabajo"]["tipo"]))
        try:
            hechas, _ = await asyncio.wait({enviar, recibir},
                                           return_when=asyncio.FIRST_COMPLETED)
        finally:
            for tarea in (enviar, recibir):
                tarea.cancel()
            await asyncio.gather(enviar, recibir, return_exceptions=True)
        if enviar in hechas and enviar.exception() is None:
            acabado = True
            await websocket.close()
        for tarea in (enviar, recibir):  # un error inesperado de cualquiera de las dos
            if tarea in hechas and not isinstance(tarea.exception(),
                                                  (WebSocketDisconnect, type(None))):
                raise tarea.exception()
    except WebSocketDisconnect:
        pass
    finally:
        dejar()
        ultima = conexiones.cerrar(id_trabajo)
        if not acabado and ultima and foto["trabajo"]["interactivo"]:
            try:
                gestor.cancelar(id_trabajo)
            except OperacionNoValida:
                pass  # ya había terminado o se estaba cancelando


async def _enviar_json(websocket: WebSocket, mensaje: dict[str, Any]) -> None:
    """Envía un mensaje JSON estricto (los números no finitos, como null)."""
    await websocket.send_text(json.dumps(a_json_finito(mensaje), allow_nan=False))


async def _enviar_eventos(websocket: WebSocket, cola: ColaEnvio) -> None:
    """Envía los eventos de la cola hasta el estado final del trabajo."""
    while True:
        evento = await cola.sacar()
        if isinstance(evento, bytes):
            await websocket.send_bytes(evento)
        elif evento["tipo"] == "progreso":
            await _enviar_json(websocket, {"tipo": "progreso", "medidas": evento["medidas"]})
        elif evento["tipo"] == "estado":
            await _enviar_json(websocket, {"tipo": "estado", "trabajo": evento["trabajo"]})
            if evento["trabajo"]["estado"] in _FINALES:
                return
        else:  # evento propio de la funcionalidad, tal cual (sin su número interno)
            await _enviar_json(websocket, {k: v for k, v in evento.items() if k != "n"})


async def _recibir_controles(websocket: WebSocket, gestor: GestorTrabajos,
                             id_trabajo: str, tipo_trabajo: str) -> None:
    """Atiende los mensajes del navegador hasta que cierra la conexión.

    Los controles comunes van al gestor; las órdenes se validan con las de su tipo de
    trabajo (validar_orden) y se pasan al trabajo.
    """
    while True:
        recibido = await websocket.receive()
        if recibido["type"] == "websocket.disconnect":
            return
        texto = recibido.get("text")
        if texto is None:
            await _enviar_json(websocket, {"tipo": "error",
                                           "mensaje": "Los mensajes tienen que ser texto JSON"})
            continue
        try:
            mensaje = json.loads(texto)
        except json.JSONDecodeError:
            await _enviar_json(websocket, {"tipo": "error", "mensaje": "El mensaje no es JSON"})
            continue
        tipo = mensaje.get("tipo") if isinstance(mensaje, dict) else None
        if not isinstance(tipo, str) or not tipo:
            await _enviar_json(websocket, {"tipo": "error",
                                           "mensaje": "El mensaje necesita un «tipo» de texto"})
            continue
        try:
            if tipo in CONTROLES:
                getattr(gestor, tipo)(id_trabajo)
            else:
                gestor.enviar_orden(id_trabajo, validar_orden(tipo_trabajo, mensaje),
                                    reanudar=tipo in ORDENES_QUE_REANUDAN)
        except (OperacionNoValida, TrabajoDesconocido, ValueError) as error:
            await _enviar_json(websocket, {"tipo": "error", "mensaje": str(error)})
