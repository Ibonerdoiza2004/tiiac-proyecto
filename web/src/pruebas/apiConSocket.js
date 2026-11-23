import { vi } from 'vitest';

import { crearApiEntrenamiento } from '../api/entrenamiento.js';
import { crearApiLaboratorio } from '../api/laboratorio.js';
import { conectarWebSocket } from '../api/websocket.js';

/** Clase WebSocket falsa: guarda lo enviado y deja abrir, recibir y cerrar a mano. */
function claseSocketFalso(sockets) {
  return class SocketFalso {
    static OPEN = 1;

    constructor() {
      this.readyState = 0;
      this.enviados = [];
      sockets.push(this);
    }

    send(datos) {
      this.enviados.push(JSON.parse(datos));
    }

    close() {
      this.readyState = 3;
    }

    abrir() {
      this.readyState = 1;
      this.onopen?.({});
    }

    recibir(mensaje) {
      this.onmessage?.({ data: mensaje instanceof Blob ? mensaje : JSON.stringify(mensaje) });
    }

    caer() {
      this.readyState = 3;
      this.onclose?.({});
    }
  };
}

/**
 * API real del Laboratorio (contrato y cliente de WebSocket incluidos) en la que solo son
 * falsos HTTP y el socket, para provocar cierres y silencios que el simulado no tiene.
 *
 * @returns {{api: import('../api/laboratorio.js').ApiLaboratorio, http: object, sockets: object[]}}
 */
export function apiConSocket() {
  const sockets = [];
  const SocketFalso = claseSocketFalso(sockets);
  const http = {
    obtener: vi.fn(async () => ({ fps: 50, partidas_experto: 15 })),
    enviar: vi.fn(async (ruta, datos) => (ruta === '/sesiones'
      ? { id: 's1', fps: 50, partidas: null, ...datos }
      : { nombre: datos.nombre, partidas: datos.partidas.length, pares: 10 })),
    borrar: vi.fn(async () => null),
  };
  const conectar = (ruta, opciones) => conectarWebSocket(ruta, { ...opciones, WebSocket: SocketFalso });
  return { api: crearApiLaboratorio({ http, conectar }), http, sockets };
}

/**
 * Igual que apiConSocket, para la API de entrenamiento.
 *
 * @returns {{api: import('../api/entrenamiento.js').ApiEntrenamiento, http: object, sockets: object[]}}
 */
export function apiEntrenamientoConSocket() {
  const sockets = [];
  const SocketFalso = claseSocketFalso(sockets);
  const esquema = { metodo: 'gail', duracion: { unidad: 'iteraciones', total: 20 }, secciones: [
    { clave: 'gail', campos: [{ clave: 'iteraciones', tipo: 'entero', min: 1, valor: 20 }] }] };
  const http = {
    obtener: vi.fn(async (ruta) => (ruta.startsWith('/biblioteca')
      ? [{ id: 'd1', nombre: 'Demos', tipo: 'conjunto', partidas: 3, pares: 30, fuente: 'humano' }]
      : esquema)),
    enviar: vi.fn(async (_, datos) => ({ id: 'e1', entorno: datos.entorno, metodo: datos.metodo,
      duracion: { unidad: 'iteraciones', total: 20 } })),
  };
  const conectar = (ruta, opciones) => conectarWebSocket(ruta, { ...opciones, WebSocket: SocketFalso });
  return { api: crearApiEntrenamiento({ http, conectar }), http, sockets };
}
