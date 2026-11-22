import { vi } from 'vitest';

import { crearApiLaboratorio } from '../api/laboratorio.js';
import { conectarWebSocket } from '../api/websocket.js';

/**
 * API real del Laboratorio (contrato y cliente de WebSocket incluidos) en la que solo son
 * falsos HTTP y el socket, para provocar cierres y silencios que el simulado no tiene.
 *
 * @returns {{api: import('../api/laboratorio.js').ApiLaboratorio, http: object, sockets: object[]}}
 */
export function apiConSocket() {
  const sockets = [];
  class SocketFalso {
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
  }
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
