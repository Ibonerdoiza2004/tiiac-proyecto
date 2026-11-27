import { describe, expect, it, vi } from 'vitest';

import { MENSAJES_SERVIDOR, crearApiLaboratorio, repartirMensaje } from './laboratorio.js';

function http() {
  return {
    obtener: vi.fn().mockResolvedValue({ fps: 50, partidas_experto: 15 }),
    enviar: vi.fn().mockResolvedValue({ id: 's1' }),
    borrar: vi.fn().mockResolvedValue(null),
  };
}

function conexionFalsa() {
  const conexion = { enviar: vi.fn().mockReturnValue(true), cerrar: vi.fn() };
  const conectar = vi.fn().mockReturnValue(conexion);
  return { conexion, conectar };
}

describe('crearApiLaboratorio', () => {
  it('usa las rutas HTTP del contrato', async () => {
    const cliente = http();
    const api = crearApiLaboratorio({ http: cliente, conectar: vi.fn() });
    await api.configuracion('cartpole');
    await api.crearSesion({ entorno: 'cartpole', modo: 'manual' });
    await api.guardar('s/1', { nombre: 'x', partidas: [{ indice: 0, tramo: [0, 3] }] });
    await expect(api.descartar('s1')).resolves.toBeUndefined();
    expect(cliente.obtener).toHaveBeenCalledWith('/laboratorio/cartpole');
    expect(cliente.enviar.mock.calls).toEqual([
      ['/sesiones', { entorno: 'cartpole', modo: 'manual' }],
      ['/sesiones/s%2F1/guardar', { nombre: 'x', partidas: [{ indice: 0, tramo: [0, 3] }] }],
    ]);
    expect(cliente.borrar).toHaveBeenCalledWith('/sesiones/s1');
  });

  it('los errores de la API llegan tal cual', async () => {
    const cliente = http();
    cliente.enviar.mockRejectedValue(new Error('422'));
    const api = crearApiLaboratorio({ http: cliente, conectar: vi.fn() });
    await expect(api.crearSesion({})).rejects.toThrow('422');
  });

  it('envía la acción mantenida y la orden de finalizar por el WebSocket', () => {
    const { conexion, conectar } = conexionFalsa();
    const juego = crearApiLaboratorio({ http: http(), conectar }).conectar('s1', {});
    expect(conectar.mock.calls[0][0]).toBe('/sesiones/s1');
    expect(juego.enviarAccion(1)).toBe(true);
    juego.enviarAccion(null);
    juego.finalizar();
    juego.cerrar();
    expect(conexion.enviar.mock.calls).toEqual([
      [{ tipo: 'accion', accion: 1 }], [{ tipo: 'accion', accion: null }], [{ tipo: 'finalizar' }],
    ]);
    expect(conexion.cerrar).toHaveBeenCalledOnce();
  });

  it('pasa las imágenes, los mensajes y los errores a los callbacks', () => {
    const { conectar } = conexionFalsa();
    const callbacks = { alImagen: vi.fn(), alEstado: vi.fn(), alError: vi.fn() };
    crearApiLaboratorio({ http: http(), conectar }).conectar('s1', callbacks);
    const { alMensaje, alImagen, alError } = conectar.mock.calls[0][1];
    const imagen = new Blob(['x']);
    alImagen(imagen);
    alMensaje({ tipo: 'estado', partida: 1, paso: 2, retorno: 2, pares: 2, esperando: false });
    alError(new Error('red'));
    expect(callbacks.alImagen).toHaveBeenCalledWith(imagen);
    expect(callbacks.alEstado).toHaveBeenCalledWith({ partida: 1, paso: 2, retorno: 2, pares: 2,
      esperando: false });
    expect(callbacks.alError.mock.calls[0][0].message).toBe('red');
  });

  it('pasa los cambios de estado de la conexión', () => {
    const { conectar } = conexionFalsa();
    const alCambiarEstado = vi.fn();
    crearApiLaboratorio({ http: http(), conectar }).conectar('s1', { alCambiarEstado });
    for (const estado of ['conectando', 'abierta']) {
      conectar.mock.calls[0][1].alCambiarEstado(estado);
    }
    conectar.mock.calls[0][1].alCambiarEstado('cerrada', { codigo: 4404, error: 'No existe' });
    expect(alCambiarEstado.mock.calls).toEqual([['conectando'], ['abierta'],
      ['cerrada', { codigo: 4404, error: 'No existe' }]]);
  });

  it('funciona sin callbacks', () => {
    const { conectar } = conexionFalsa();
    crearApiLaboratorio({ http: http(), conectar }).conectar('s1');
    const { alMensaje, alImagen, alError, alCambiarEstado } = conectar.mock.calls[0][1];
    expect(() => {
      alCambiarEstado('cerrada');
      alMensaje({ tipo: 'estado' });
      alImagen(new Blob(['x']));
      alError(new Error('x'));
    }).not.toThrow();
  });
});

describe('repartirMensaje', () => {
  const callbacks = () => ({ alEstado: vi.fn(), alTerminarPartida: vi.fn(),
    alTerminarSesion: vi.fn(), alError: vi.fn() });
  const partida = { indice: 0, pasos: 9, retorno: 9, semilla: 3 };

  it.each([
    [{ tipo: MENSAJES_SERVIDOR.ESTADO, partida: 2, paso: 1 }, 'alEstado', { partida: 2, paso: 1 }],
    [{ tipo: MENSAJES_SERVIDOR.PARTIDA_TERMINADA, partida }, 'alTerminarPartida', partida],
    [{ tipo: MENSAJES_SERVIDOR.SESION_TERMINADA, partidas: [partida] }, 'alTerminarSesion', [partida]],
  ])('%j va a %s', (mensaje, destino, argumento) => {
    const c = callbacks();
    repartirMensaje(mensaje, c);
    expect(c[destino]).toHaveBeenCalledExactlyOnceWith(argumento);
    for (const [nombre, callback] of Object.entries(c)) {
      if (nombre !== destino) {
        expect(callback).not.toHaveBeenCalled();
      }
    }
  });

  it.each([
    [{ tipo: 'error', mensaje: 'Sesión caducada' }, 'Sesión caducada'],
    [{ tipo: 'error' }, 'Error del servidor'],
    [{ tipo: 'otro' }, 'Mensaje desconocido del servidor: {"tipo":"otro"}'],
    [null, 'Mensaje desconocido del servidor: null'],
    [[1], 'Mensaje desconocido del servidor: [1]'],
    ['texto', 'Mensaje desconocido del servidor: "texto"'],
  ])('%j es un error', (mensaje, texto) => {
    const c = callbacks();
    repartirMensaje(mensaje, c);
    expect(c.alError.mock.calls[0][0].message).toBe(texto);
  });
});
