import { vi } from 'vitest';

import { crearApiComparar } from '../api/comparar.js';
import { crearApiEntrenamiento } from '../api/entrenamiento.js';
import { crearApiLaboratorio } from '../api/laboratorio.js';
import { crearApiTrabajos } from '../api/trabajos.js';
import { conectarWebSocket } from '../api/websocket.js';

/** Clase WebSocket falsa: guarda lo enviado y deja abrir, recibir y cerrar a mano. */
function claseSocketFalso(sockets) {
  return class SocketFalso {
    static OPEN = 1;

    constructor(url) {
      this.url = url;
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

    caer(code) {
      this.readyState = 3;
      this.onclose?.(code === undefined ? {} : { code });
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

/** Modelos de la biblioteca de prueba de Comparar: dos válidos y uno roto. */
export const MODELOS_PRUEBA = Object.freeze([
  { id: 'exp', entorno: 'cartpole', tipo: 'experto', nombre: 'Experto congelado', origen: 'congelado',
    creado: null, valido: true, motivo: null, detalles: { pasos: 60000, segundos: 25 } },
  { id: 'g1', entorno: 'cartpole', tipo: 'gail', nombre: 'GAIL 1', origen: 'entrega', creado: null,
    valido: true, motivo: null, detalles: { cantidad: 1, repeticion: 0, iteraciones: 20, segundos: 300 } },
  { id: 'roto', entorno: 'cartpole', tipo: 'bc', nombre: 'BC roto', origen: 'dashboard', creado: null,
    valido: false, motivo: 'falta politica.pt', detalles: {} },
]);

/**
 * API real de Comparar (cliente de trabajos y WebSocket incluidos) con HTTP y socket
 * falsos. Crear devuelve un trabajo «c1» en cola con 2 partidas y semilla 9000.
 *
 * @returns {{api: import('../api/comparar.js').ApiComparar, http: object, sockets: object[]}}
 */
export function apiCompararConSocket() {
  const sockets = [];
  const SocketFalso = claseSocketFalso(sockets);
  const http = {
    obtener: vi.fn(async (ruta) => (ruta === '/trabajos' ? [] : MODELOS_PRUEBA.map((m) => ({ ...m })))),
    enviar: vi.fn(async (_, datos) => ({ id: 'c1', tipo: 'comparacion', estado: 'en_cola', creado: 1,
      argumentos: { ...datos, partidas: datos.partidas ?? 2, semilla: 9000 }, resultado: null, error: null })),
  };
  const conectar = (ruta, opciones) => conectarWebSocket(ruta, { ...opciones, WebSocket: SocketFalso });
  return { api: crearApiComparar({ http, trabajos: crearApiTrabajos({ http, conectar }) }), http, sockets };
}
