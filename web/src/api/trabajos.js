/**
 * Cliente del protocolo único de trabajos de la API (src/app/contrato.py y
 * src/app/rutas_trabajos.py): todo lo pesado (jugar, entrenar, comparar) es un *trabajo*
 * que se crea con la ruta de su funcionalidad y después se consulta, controla y sigue
 * igual, sea del tipo que sea.
 *
 * - HTTP: GET /api/trabajos (?tipo=), GET /api/trabajos/{id} y POST
 *   /api/trabajos/{id}/pausar, /reanudar y /cancelar.
 * - WebSocket /api/trabajos/{id}. Al conectar: {tipo: 'estado', trabajo}, {tipo:
 *   'historial', progreso, eventos} con todo lo anterior y la última imagen. En directo:
 *   {tipo: 'progreso', medidas}, imágenes JPEG en binario, {tipo: 'estado', trabajo} y los
 *   eventos propios de cada funcionalidad. Al terminar, el último estado (con el
 *   resultado) y cierre normal; 4404 si el trabajo no existe. Del navegador: pausar,
 *   reanudar, cancelar y las órdenes de cada tipo de trabajo.
 *
 * Cerrar la conexión no para un entrenamiento ni una comparación: se puede volver a
 * seguir y el historial trae todo lo anterior, sin perder ni repetir nada.
 */

import { crearClienteHttp } from './http.js';
import { conectarWebSocket } from './websocket.js';

/** Estados de un trabajo (EstadoTrabajo en src/app/trabajos.py). */
export const ESTADOS_TRABAJO = Object.freeze({
  EN_COLA: 'en_cola',
  EN_MARCHA: 'en_marcha',
  PAUSADO: 'pausado',
  TERMINADO: 'terminado',
  FALLIDO: 'fallido',
  CANCELADO: 'cancelado',
});

/** Código con el que la API cierra el WebSocket de un trabajo que no existe. */
export const CIERRE_NO_EXISTE = 4404;

/** Estados en los que el trabajo ya no cambia. */
export const ESTADOS_FINALES = Object.freeze(new Set([ESTADOS_TRABAJO.TERMINADO,
  ESTADOS_TRABAJO.FALLIDO, ESTADOS_TRABAJO.CANCELADO]));

/**
 * @typedef {object} Trabajo
 * @property {string} id
 * @property {string} tipo sesion, entrenamiento o comparacion.
 * @property {string} estado ESTADOS_TRABAJO.
 * @property {Record<string, unknown>} argumentos Lo que se pidió al crearlo.
 * @property {boolean} interactivo
 * @property {number} creado Segundos desde 1970.
 * @property {number | null} iniciado
 * @property {number | null} terminado
 * @property {number} progresos Mensajes de progreso enviados hasta ahora.
 * @property {Record<string, unknown> | null} resultado Solo al terminar bien.
 * @property {string | null} error Solo si falló.
 */

/**
 * @typedef {object} CallbacksTrabajo
 * @property {(trabajo: Trabajo) => void} [alEstado] Al conectar y en cada cambio.
 * @property {(historial: {progreso: object[], eventos: object[]}) => void} [alHistorial]
 *   Al conectar: todo lo anterior, para rehacer la vista desde cero.
 * @property {(medidas: Record<string, unknown>) => void} [alProgreso]
 * @property {(evento: object) => void} [alEvento] Eventos propios de la funcionalidad.
 * @property {(imagen: Blob) => void} [alImagen]
 * @property {(error: Error) => void} [alError]
 * @property {(estado: string, cierre?: import('./websocket.js').Cierre) => void} [alCambiarEstado]
 *   Cambios de la conexión y, al cerrarse, cómo terminó
 *   (ESTADOS_CONEXION de ./websocket.js).
 */

/**
 * @typedef {object} ConexionTrabajo
 * @property {() => boolean} pausar
 * @property {() => boolean} reanudar
 * @property {() => boolean} cancelar
 * @property {(orden: {tipo: string}) => boolean} ordenar Orden propia del tipo de trabajo,
 *   como {tipo: 'vista', activa: true} o {tipo: 'finalizar'}.
 * @property {() => void} cerrar Deja de seguirlo sin pararlo.
 */

const MENSAJES_COMUNES = new Set(['estado', 'historial', 'progreso', 'error']);

/**
 * Reparte un mensaje JSON del WebSocket de un trabajo entre los callbacks.
 *
 * @param {unknown} mensaje Mensaje ya decodificado.
 * @param {CallbacksTrabajo} callbacks
 */
export function repartirMensajeTrabajo(mensaje, callbacks) {
  const tipo = mensaje && typeof mensaje === 'object' && !Array.isArray(mensaje)
    ? mensaje.tipo : undefined;
  if (tipo === 'estado') {
    callbacks.alEstado?.(mensaje.trabajo);
  } else if (tipo === 'historial') {
    callbacks.alHistorial?.({ progreso: mensaje.progreso ?? [], eventos: mensaje.eventos ?? [] });
  } else if (tipo === 'progreso') {
    callbacks.alProgreso?.(mensaje.medidas ?? {});
  } else if (tipo === 'error') {
    callbacks.alError?.(new Error(String(mensaje.mensaje ?? 'Error del servidor')));
  } else if (typeof tipo === 'string' && tipo && !MENSAJES_COMUNES.has(tipo)) {
    callbacks.alEvento?.(mensaje);
  } else {
    callbacks.alError?.(new Error(`Mensaje desconocido del servidor: ${JSON.stringify(mensaje)}`));
  }
}

/**
 * @typedef {object} ApiTrabajos
 * @property {(opciones?: {tipo?: string, senal?: AbortSignal}) => Promise<Trabajo[]>} listar
 * @property {(id: string) => Promise<Trabajo>} obtener
 * @property {(id: string, callbacks: CallbacksTrabajo) => ConexionTrabajo} seguir
 */

/**
 * Crea el cliente de trabajos sobre el backend real.
 *
 * @param {object} [opciones]
 * @param {import('./http.js').ClienteHttp} [opciones.http] Cliente HTTP.
 * @param {typeof conectarWebSocket} [opciones.conectar] Abre un WebSocket (los tests pasan
 *   uno simulado).
 * @returns {ApiTrabajos}
 */
export function crearApiTrabajos({ http = crearClienteHttp(), conectar = conectarWebSocket } = {}) {
  const ruta = (id) => `/trabajos/${encodeURIComponent(id)}`;
  return {
    listar: ({ tipo, senal } = {}) => http.obtener('/trabajos', { parametros: { tipo }, senal }),
    obtener: (id) => http.obtener(ruta(id)),
    seguir(id, callbacks = {}) {
      const conexion = conectar(ruta(id), {
        alMensaje: (mensaje) => repartirMensajeTrabajo(mensaje, callbacks),
        alImagen: (imagen) => callbacks.alImagen?.(imagen),
        alError: (error) => callbacks.alError?.(error),
        alCambiarEstado: (estado, ...cierre) => callbacks.alCambiarEstado?.(estado, ...cierre),
      });
      return {
        pausar: () => conexion.enviar({ tipo: 'pausar' }),
        reanudar: () => conexion.enviar({ tipo: 'reanudar' }),
        cancelar: () => conexion.enviar({ tipo: 'cancelar' }),
        ordenar: (orden) => conexion.enviar(orden),
        cerrar: () => conexion.cerrar(),
        conectando: () => conexion.conectando?.() ?? false,
      };
    },
  };
}
