/**
 * Cliente de WebSocket de la API.
 *
 * Por el mismo socket llegan dos tipos de mensaje: texto con JSON (progreso, estado,
 * partidas grabadas…) e imágenes binarias (JPEG del entorno). Los mensajes que se envían
 * (órdenes y teclas) van siempre como JSON. El formato concreto de cada mensaje lo fija el
 * contrato de la API; este cliente solo los separa y los decodifica.
 */

import { BASE_API, comprobarRuta } from './http.js';

// Valores de `WebSocket.CONNECTING` y `WebSocket.OPEN` en el estándar.
const CONECTANDO = 0;
const ABIERTO = 1;

/** Estados por los que pasa una conexión. */
export const ESTADOS_CONEXION = Object.freeze({
  CONECTANDO: 'conectando',
  ABIERTA: 'abierta',
  CERRADA: 'cerrada',
});

/**
 * @typedef {object} Cierre Cómo terminó una conexión.
 * @property {number | null} codigo Código de cierre (1000 normal; la API usa 4404 si el
 *   trabajo no existe); null si el navegador no lo dio.
 * @property {string | null} error Mensaje del último `{tipo: 'error'}` del servidor si fue
 *   lo último que llegó antes de cerrar (el motivo del cierre); si no, null.
 */

/**
 * Construye la URL `ws://` o `wss://` de una ruta de la API.
 *
 * @param {string} ruta Ruta que empieza por «/», como `/trabajos/3/progreso`.
 * @param {object} [opciones]
 * @param {string} [opciones.base] Prefijo de la API (por defecto `/api`), relativo a la
 *   página o una URL absoluta `http(s)://servidor/prefijo`.
 * @param {{ protocol: string, host: string }} [opciones.ubicacion] Origen de la página
 *   (por defecto `window.location`); con una base absoluta no se usa.
 * @returns {string} URL absoluta del WebSocket.
 * @throws {TypeError} Si la ruta no es válida o el protocolo no es http ni https.
 */
export function urlWebSocket(ruta, { base = BASE_API, ubicacion = globalThis.location } = {}) {
  comprobarRuta(ruta);
  const protocolos = { 'http:': 'ws:', 'https:': 'wss:' };
  let origen = ubicacion;
  let prefijo = base;
  if (/^[a-z][a-z\d+.-]*:/i.test(base)) {
    const url = new URL(base);
    origen = url;
    prefijo = url.pathname;
  }
  const protocolo = protocolos[origen?.protocol];
  if (!protocolo || !origen.host) {
    throw new TypeError(`No se puede abrir un WebSocket desde ${String(origen?.protocol)}`);
  }
  return `${protocolo}//${origen.host}${prefijo.replace(/\/+$/, '')}${ruta}`;
}

/** Mensaje de un `{tipo: 'error', mensaje}` del servidor, o null si no lo es. */
function errorDelServidor(mensaje) {
  if (!mensaje || typeof mensaje !== 'object' || mensaje.tipo !== 'error') {
    return null;
  }
  return String(mensaje.mensaje ?? 'Error del servidor');
}

/**
 * @typedef {object} ConexionWebSocket
 * @property {(mensaje: unknown) => boolean} enviar Envía un objeto como JSON; devuelve
 *   `false` sin enviar nada si la conexión no está abierta.
 * @property {() => void} cerrar Cierra la conexión; después no se llama a ningún callback.
 * @property {() => string} estado Estado actual (`ESTADOS_CONEXION`).
 * @property {() => boolean} conectando Si el socket aún no ha abierto: lo que no se pudo
 *   enviar se puede volver a intentar, a diferencia de una conexión que se cierra.
 */

/**
 * Abre un WebSocket con la API.
 *
 * @param {string} ruta Ruta que empieza por «/».
 * @param {object} [opciones]
 * @param {(mensaje: unknown) => void} [opciones.alMensaje] Recibe cada mensaje JSON.
 * @param {(imagen: Blob) => void} [opciones.alImagen] Recibe cada imagen binaria.
 * @param {(estado: string, cierre?: Cierre) => void} [opciones.alCambiarEstado] Recibe cada
 *   cambio de estado; al cerrarse, también cómo terminó.
 * @param {(error: Error) => void} [opciones.alError] Recibe errores de conexión y mensajes
 *   de texto que no son JSON.
 * @param {string} [opciones.base] Prefijo de la API.
 * @param {{ protocol: string, host: string }} [opciones.ubicacion] Origen de la página.
 * @param {typeof globalThis.WebSocket} [opciones.WebSocket] Implementación (los tests pasan
 *   una simulada).
 * @returns {ConexionWebSocket} Control de la conexión.
 */
export function conectarWebSocket(
  ruta,
  {
    alMensaje = () => {},
    alImagen = () => {},
    alCambiarEstado = () => {},
    alError = () => {},
    base,
    ubicacion,
    WebSocket: ClaseWebSocket = globalThis.WebSocket,
  } = {},
) {
  const socket = new ClaseWebSocket(urlWebSocket(ruta, { base, ubicacion }));
  socket.binaryType = 'blob';
  let estado = ESTADOS_CONEXION.CONECTANDO;
  let activa = true;
  // El servidor avisa con {tipo: 'error'} antes de cerrar por un motivo propio: se guarda
  // mientras sea lo último que llegó, para decir por qué se cerró.
  let ultimoError = null;

  const cambiarEstado = (nuevo, ...detalle) => {
    if (estado !== nuevo) {
      estado = nuevo;
      if (activa) {
        alCambiarEstado(nuevo, ...detalle);
      }
    }
  };

  socket.onopen = () => cambiarEstado(ESTADOS_CONEXION.ABIERTA);
  socket.onclose = (evento) => cambiarEstado(ESTADOS_CONEXION.CERRADA,
    { codigo: typeof evento?.code === 'number' ? evento.code : null, error: ultimoError });
  socket.onerror = () => {
    if (activa) {
      alError(new Error('Error en la conexión con la API'));
    }
  };
  socket.onmessage = ({ data }) => {
    if (!activa) {
      return;
    }
    if (typeof data !== 'string') {
      ultimoError = null;
      alImagen(data instanceof Blob ? data : new Blob([data], { type: 'image/jpeg' }));
      return;
    }
    let mensaje;
    try {
      mensaje = JSON.parse(data);
    } catch (error) {
      ultimoError = null;
      alError(new Error('La API envió un mensaje que no es JSON', { cause: error }));
      return;
    }
    ultimoError = errorDelServidor(mensaje);
    alMensaje(mensaje);
  };

  alCambiarEstado(estado);

  return {
    enviar(mensaje) {
      if (!activa || socket.readyState !== (ClaseWebSocket.OPEN ?? ABIERTO)) {
        return false;
      }
      socket.send(JSON.stringify(mensaje));
      return true;
    },
    cerrar() {
      if (!activa) {
        return;
      }
      activa = false;
      estado = ESTADOS_CONEXION.CERRADA;
      socket.close();
    },
    estado: () => estado,
    conectando: () => activa && socket.readyState === CONECTANDO,
  };
}
