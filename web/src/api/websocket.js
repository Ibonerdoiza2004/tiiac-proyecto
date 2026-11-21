/**
 * Cliente de WebSocket de la API.
 *
 * Por el mismo socket llegan dos tipos de mensaje: texto con JSON (progreso, estado,
 * partidas grabadas…) e imágenes binarias (JPEG del entorno). Los mensajes que se envían
 * (órdenes y teclas) van siempre como JSON. El formato concreto de cada mensaje lo fija el
 * contrato de la API; este cliente solo los separa y los decodifica.
 */

import { BASE_API, comprobarRuta } from './http.js';

// Valor de `WebSocket.OPEN` en el estándar, por si la implementación no lo expone.
const ABIERTO = 1;

/** Estados por los que pasa una conexión. */
export const ESTADOS_CONEXION = Object.freeze({
  CONECTANDO: 'conectando',
  ABIERTA: 'abierta',
  CERRADA: 'cerrada',
});

/**
 * Construye la URL `ws://` o `wss://` de una ruta de la API.
 *
 * @param {string} ruta Ruta que empieza por «/», como `/trabajos/3/progreso`.
 * @param {object} [opciones]
 * @param {string} [opciones.base] Prefijo de la API (por defecto `/api`).
 * @param {{ protocol: string, host: string }} [opciones.ubicacion] Origen de la página
 *   (por defecto `window.location`).
 * @returns {string} URL absoluta del WebSocket.
 * @throws {TypeError} Si la ruta no es válida o el protocolo no es http ni https.
 */
export function urlWebSocket(ruta, { base = BASE_API, ubicacion = globalThis.location } = {}) {
  comprobarRuta(ruta);
  const protocolos = { 'http:': 'ws:', 'https:': 'wss:' };
  const protocolo = protocolos[ubicacion?.protocol];
  if (!protocolo || !ubicacion.host) {
    throw new TypeError(`No se puede abrir un WebSocket desde ${String(ubicacion?.protocol)}`);
  }
  return `${protocolo}//${ubicacion.host}${base.replace(/\/+$/, '')}${ruta}`;
}

/**
 * @typedef {object} ConexionWebSocket
 * @property {(mensaje: unknown) => boolean} enviar Envía un objeto como JSON; devuelve
 *   `false` sin enviar nada si la conexión no está abierta.
 * @property {() => void} cerrar Cierra la conexión; después no se llama a ningún callback.
 * @property {() => string} estado Estado actual (`ESTADOS_CONEXION`).
 */

/**
 * Abre un WebSocket con la API.
 *
 * @param {string} ruta Ruta que empieza por «/».
 * @param {object} [opciones]
 * @param {(mensaje: unknown) => void} [opciones.alMensaje] Recibe cada mensaje JSON.
 * @param {(imagen: Blob) => void} [opciones.alImagen] Recibe cada imagen binaria.
 * @param {(estado: string) => void} [opciones.alCambiarEstado] Recibe cada cambio de estado.
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

  const cambiarEstado = (nuevo) => {
    if (estado !== nuevo) {
      estado = nuevo;
      if (activa) {
        alCambiarEstado(nuevo);
      }
    }
  };

  socket.onopen = () => cambiarEstado(ESTADOS_CONEXION.ABIERTA);
  socket.onclose = () => cambiarEstado(ESTADOS_CONEXION.CERRADA);
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
      alImagen(data instanceof Blob ? data : new Blob([data], { type: 'image/jpeg' }));
      return;
    }
    let mensaje;
    try {
      mensaje = JSON.parse(data);
    } catch (error) {
      alError(new Error('La API envió un mensaje que no es JSON', { cause: error }));
      return;
    }
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
  };
}
