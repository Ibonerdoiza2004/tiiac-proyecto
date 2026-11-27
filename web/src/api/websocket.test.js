import { describe, expect, it, vi } from 'vitest';

import { ESTADOS_CONEXION, conectarWebSocket, urlWebSocket } from './websocket.js';

const LOCAL = { protocol: 'http:', host: '127.0.0.1:5173' };

/** WebSocket simulado: guarda lo enviado y deja disparar los eventos a mano. */
class WebSocketSimulado {
  static OPEN = 1;
  static creados = [];

  constructor(url) {
    this.url = url;
    this.readyState = 0;
    this.enviados = [];
    this.cerrado = false;
    WebSocketSimulado.creados.push(this);
  }

  send(datos) {
    this.enviados.push(datos);
  }

  close() {
    this.cerrado = true;
    this.readyState = 3;
    this.onclose?.({});
  }

  abrir() {
    this.readyState = 1;
    this.onopen?.({});
  }

  recibir(data) {
    this.onmessage?.({ data });
  }

  fallar() {
    this.onerror?.({});
  }

  cerrarDesdeServidor(code) {
    this.readyState = 3;
    this.onclose?.(code === undefined ? {} : { code });
  }
}

function conectar(opciones = {}) {
  WebSocketSimulado.creados = [];
  const callbacks = {
    alMensaje: vi.fn(),
    alImagen: vi.fn(),
    alCambiarEstado: vi.fn(),
    alError: vi.fn(),
  };
  const conexion = conectarWebSocket('/trabajos/3', {
    ...callbacks,
    ubicacion: LOCAL,
    WebSocket: WebSocketSimulado,
    ...opciones,
  });
  return { conexion, socket: WebSocketSimulado.creados[0], ...callbacks };
}

describe('urlWebSocket', () => {
  it.each([
    [{ protocol: 'http:', host: '127.0.0.1:5173' }, 'ws://127.0.0.1:5173/api/x'],
    [{ protocol: 'https:', host: 'gail.example' }, 'wss://gail.example/api/x'],
  ])('desde %j da %s', (ubicacion, esperada) => {
    expect(urlWebSocket('/x', { ubicacion })).toBe(esperada);
  });

  it.each([
    ['/api/', 'ws://127.0.0.1:5173/api/x'],
    ['', 'ws://127.0.0.1:5173/x'],
    ['/v2/api', 'ws://127.0.0.1:5173/v2/api/x'],
  ])('con la base %j da %s', (base, esperada) => {
    expect(urlWebSocket('/x', { base, ubicacion: LOCAL })).toBe(esperada);
  });

  it('por defecto usa la ubicación de la página', () => {
    expect(urlWebSocket('/x')).toBe(`ws://${globalThis.location.host}/api/x`);
  });

  it.each([
    [{ protocol: 'file:', host: '' }],
    [{ protocol: 'ftp:', host: 'x' }],
    [{ protocol: 'http:', host: '' }],
    [undefined],
  ])('rechaza la ubicación %j', (ubicacion) => {
    expect(() => urlWebSocket('/x', { ubicacion: ubicacion ?? null })).toThrow(TypeError);
  });

  it.each(['x', '', '//x', null])('rechaza la ruta %j', (ruta) => {
    expect(() => urlWebSocket(ruta, { ubicacion: LOCAL })).toThrow(TypeError);
  });
});

describe('conectarWebSocket', () => {
  it('abre la URL de la ruta y recibe binarios como Blob', () => {
    const { socket } = conectar();
    expect(socket.url).toBe('ws://127.0.0.1:5173/api/trabajos/3');
    expect(socket.binaryType).toBe('blob');
  });

  it('pasa por conectando, abierta y cerrada', () => {
    const { conexion, socket, alCambiarEstado } = conectar();
    expect(conexion.estado()).toBe(ESTADOS_CONEXION.CONECTANDO);
    socket.abrir();
    expect(conexion.estado()).toBe(ESTADOS_CONEXION.ABIERTA);
    socket.cerrarDesdeServidor();
    expect(conexion.estado()).toBe(ESTADOS_CONEXION.CERRADA);
    expect(alCambiarEstado.mock.calls.map(([estado]) => estado)).toEqual([
      'conectando',
      'abierta',
      'cerrada',
    ]);
  });

  it('no repite un estado que no cambia', () => {
    const { socket, alCambiarEstado } = conectar();
    socket.abrir();
    socket.abrir();
    expect(alCambiarEstado).toHaveBeenCalledTimes(2);
  });

  it.each([
    ['{"tipo":"progreso","iteracion":3,"retorno":412.5}', { tipo: 'progreso', iteracion: 3, retorno: 412.5 }],
    ['[1,2]', [1, 2]],
    ['null', null],
    ['"texto"', 'texto'],
  ])('decodifica el JSON %s', (texto, esperado) => {
    const { socket, alMensaje, alError } = conectar();
    socket.recibir(texto);
    expect(alMensaje).toHaveBeenCalledExactlyOnceWith(esperado);
    expect(alError).not.toHaveBeenCalled();
  });

  it.each(['{roto', '', 'hola'])('un texto que no es JSON (%j) va a alError', (texto) => {
    const { socket, alMensaje, alError } = conectar();
    socket.recibir(texto);
    expect(alMensaje).not.toHaveBeenCalled();
    expect(alError).toHaveBeenCalledOnce();
    expect(alError.mock.calls[0][0].cause).toBeInstanceOf(SyntaxError);
  });

  it('entrega las imágenes Blob tal cual', () => {
    const { socket, alImagen, alMensaje } = conectar();
    const imagen = new Blob([new Uint8Array([0xff, 0xd8, 0xff])], { type: 'image/jpeg' });
    socket.recibir(imagen);
    expect(alImagen).toHaveBeenCalledExactlyOnceWith(imagen);
    expect(alMensaje).not.toHaveBeenCalled();
  });

  it('convierte un ArrayBuffer en un Blob JPEG', async () => {
    const { socket, alImagen } = conectar();
    socket.recibir(new Uint8Array([0xff, 0xd8, 0xff]).buffer);
    const [imagen] = alImagen.mock.calls[0];
    expect(imagen).toBeInstanceOf(Blob);
    expect(imagen.type).toBe('image/jpeg');
    expect([...new Uint8Array(await imagen.arrayBuffer())]).toEqual([0xff, 0xd8, 0xff]);
  });

  it('avisa de los errores de conexión', () => {
    const { socket, alError } = conectar();
    socket.fallar();
    expect(alError).toHaveBeenCalledOnce();
    expect(alError.mock.calls[0][0]).toBeInstanceOf(Error);
  });

  it('enviar manda JSON solo con la conexión abierta', () => {
    const { conexion, socket } = conectar();
    expect(conexion.enviar({ orden: 'vista', activa: true })).toBe(false);
    expect(socket.enviados).toEqual([]);
    socket.abrir();
    expect(conexion.enviar({ orden: 'vista', activa: true })).toBe(true);
    expect(socket.enviados).toEqual(['{"orden":"vista","activa":true}']);
    socket.cerrarDesdeServidor();
    expect(conexion.enviar({ orden: 'finalizar' })).toBe(false);
    expect(socket.enviados).toHaveLength(1);
  });

  it('enviar usa readyState 1 si la clase no define OPEN', () => {
    class SinConstantes extends WebSocketSimulado {}
    Object.defineProperty(SinConstantes, 'OPEN', { value: undefined });
    const { conexion, socket } = conectar({ WebSocket: SinConstantes });
    socket.abrir();
    expect(conexion.enviar({ tecla: 'izquierda' })).toBe(true);
  });

  it('tras cerrar no llama a ningún callback ni envía', () => {
    const { conexion, socket, alMensaje, alImagen, alCambiarEstado, alError } = conectar();
    socket.abrir();
    alCambiarEstado.mockClear();
    conexion.cerrar();
    expect(socket.cerrado).toBe(true);
    expect(conexion.estado()).toBe(ESTADOS_CONEXION.CERRADA);

    socket.recibir('{"tipo":"progreso"}');
    socket.recibir(new Blob(['x']));
    socket.fallar();
    socket.readyState = 1;
    expect(conexion.enviar({ orden: 'x' })).toBe(false);
    for (const callback of [alMensaje, alImagen, alCambiarEstado, alError]) {
      expect(callback).not.toHaveBeenCalled();
    }
  });

  it('cerrar dos veces cierra el socket una sola vez', () => {
    const { conexion, socket } = conectar();
    const cerrar = vi.spyOn(socket, 'close');
    conexion.cerrar();
    conexion.cerrar();
    expect(cerrar).toHaveBeenCalledOnce();
  });

  it('funciona sin callbacks', () => {
    WebSocketSimulado.creados = [];
    conectarWebSocket('/x', { ubicacion: LOCAL, WebSocket: WebSocketSimulado });
    const [socket] = WebSocketSimulado.creados;
    expect(() => {
      socket.abrir();
      socket.recibir('{}');
      socket.recibir('{roto');
      socket.recibir(new Blob(['x']));
      socket.fallar();
      socket.cerrarDesdeServidor();
    }).not.toThrow();
  });

  it('rechaza una ruta no válida sin abrir ningún socket', () => {
    WebSocketSimulado.creados = [];
    expect(() =>
      conectarWebSocket('trabajos', { ubicacion: LOCAL, WebSocket: WebSocketSimulado }),
    ).toThrow(TypeError);
    expect(WebSocketSimulado.creados).toEqual([]);
  });
});

describe('cómo terminó la conexión', () => {
  const cierre = (alCambiarEstado) => alCambiarEstado.mock.calls.at(-1);

  it('da el código de cierre y el error del servidor si fue lo último que llegó', () => {
    // Regresión: sin esto, «Se perdió la conexión» tapaba el motivo (por ejemplo, 4404).
    const { socket, alCambiarEstado } = conectar();
    socket.abrir();
    socket.recibir(JSON.stringify({ tipo: 'error', mensaje: 'No existe el trabajo 3' }));
    socket.cerrarDesdeServidor(4404);
    expect(cierre(alCambiarEstado)).toEqual(['cerrada', { codigo: 4404, error: 'No existe el trabajo 3' }]);
  });

  it.each([
    ['otro mensaje JSON', (s) => s.recibir(JSON.stringify({ tipo: 'progreso', medidas: {} }))],
    ['una imagen', (s) => s.recibir(new Blob(['x']))],
    ['un texto que no es JSON', (s) => s.recibir('no json')],
  ])('un error del servidor seguido de %s no es el motivo del cierre', (_, despues) => {
    const { socket, alCambiarEstado } = conectar();
    socket.abrir();
    socket.recibir(JSON.stringify({ tipo: 'error', mensaje: 'orden no válida' }));
    despues(socket);
    socket.cerrarDesdeServidor(1006);
    expect(cierre(alCambiarEstado)).toEqual(['cerrada', { codigo: 1006, error: null }]);
  });

  it('un error sin mensaje cuenta con un texto genérico', () => {
    const { socket, alCambiarEstado } = conectar();
    socket.recibir(JSON.stringify({ tipo: 'error' }));
    socket.cerrarDesdeServidor(1000);
    expect(cierre(alCambiarEstado)).toEqual(['cerrada', { codigo: 1000, error: 'Error del servidor' }]);
  });

  it('sin código de cierre da null', () => {
    const { socket, alCambiarEstado } = conectar();
    socket.cerrarDesdeServidor();
    expect(cierre(alCambiarEstado)).toEqual(['cerrada', { codigo: null, error: null }]);
  });

  it('los demás cambios de estado no llevan detalle', () => {
    const { socket, alCambiarEstado } = conectar();
    socket.abrir();
    expect(alCambiarEstado.mock.calls).toEqual([['conectando'], ['abierta']]);
  });
});

describe('conectando', () => {
  it.each([[0, true], [1, false], [2, false], [3, false]])('con readyState %i es %s', (estado, esperado) => {
    const { conexion, socket } = conectar();
    socket.readyState = estado;
    expect(conexion.conectando()).toBe(esperado);
  });

  it('tras cerrar desde aquí ya no está conectando', () => {
    const { conexion } = conectar();
    conexion.cerrar();
    expect(conexion.conectando()).toBe(false);
  });
});

describe('urlWebSocket con una base absoluta', () => {
  it.each([
    ['http://api.local:8000/api', 'ws://api.local:8000/api/x'],
    ['https://gail.example/v2/api/', 'wss://gail.example/v2/api/x'],
    ['http://api.local:8000', 'ws://api.local:8000/x'],
  ])('%s da %s, sin mezclarla con la página', (base, esperada) => {
    // Regresión: salía «ws://127.0.0.1:5173http://api.local:8000/api/x».
    expect(urlWebSocket('/x', { base, ubicacion: LOCAL })).toBe(esperada);
  });

  it('con una base absoluta no necesita la ubicación de la página', () => {
    expect(urlWebSocket('/x', { base: 'http://api.local/api', ubicacion: null })).toBe('ws://api.local/api/x');
  });

  it.each(['ftp://api.local/api', 'file:///api'])('rechaza la base %s', (base) => {
    expect(() => urlWebSocket('/x', { base, ubicacion: LOCAL })).toThrow(TypeError);
  });
});
