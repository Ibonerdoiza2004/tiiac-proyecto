import { describe, expect, it, vi } from 'vitest';

import { ESTADOS_FINALES, ESTADOS_TRABAJO, crearApiTrabajos, repartirMensajeTrabajo } from './trabajos.js';

function callbacks() {
  return { alEstado: vi.fn(), alHistorial: vi.fn(), alProgreso: vi.fn(), alEvento: vi.fn(), alError: vi.fn() };
}

describe('estados de trabajo', () => {
  it('son los de src/app/trabajos.py y solo tres son finales', () => {
    expect(Object.values(ESTADOS_TRABAJO)).toEqual(['en_cola', 'en_marcha', 'pausado', 'terminado',
      'fallido', 'cancelado']);
    expect([...ESTADOS_FINALES]).toEqual(['terminado', 'fallido', 'cancelado']);
  });
});

describe('repartirMensajeTrabajo', () => {
  it('reparte estado, historial, progreso, error y eventos propios', () => {
    const c = callbacks();
    const trabajo = { id: 't1', estado: 'en_marcha' };
    repartirMensajeTrabajo({ tipo: 'estado', trabajo }, c);
    repartirMensajeTrabajo({ tipo: 'historial', progreso: [{ a: 1 }], eventos: [{ tipo: 'x' }] }, c);
    repartirMensajeTrabajo({ tipo: 'progreso', medidas: { modelo: 'a' } }, c);
    repartirMensajeTrabajo({ tipo: 'error', mensaje: 'mal' }, c);
    repartirMensajeTrabajo({ tipo: 'partida_terminada', n: 3 }, c);
    expect(c.alEstado).toHaveBeenCalledWith(trabajo);
    expect(c.alHistorial).toHaveBeenCalledWith({ progreso: [{ a: 1 }], eventos: [{ tipo: 'x' }] });
    expect(c.alProgreso).toHaveBeenCalledWith({ modelo: 'a' });
    expect(c.alError.mock.calls[0][0].message).toBe('mal');
    expect(c.alEvento).toHaveBeenCalledWith({ tipo: 'partida_terminada', n: 3 });
  });

  it('completa lo que falta con valores vacíos', () => {
    const c = callbacks();
    repartirMensajeTrabajo({ tipo: 'historial' }, c);
    repartirMensajeTrabajo({ tipo: 'progreso' }, c);
    repartirMensajeTrabajo({ tipo: 'error' }, c);
    expect(c.alHistorial).toHaveBeenCalledWith({ progreso: [], eventos: [] });
    expect(c.alProgreso).toHaveBeenCalledWith({});
    expect(c.alError.mock.calls[0][0].message).toBe('Error del servidor');
  });

  it.each([[null], [[]], ['texto'], [3], [{}], [{ tipo: '' }], [{ tipo: 4 }], [[{ tipo: 'estado' }]]])(
    'avisa de un mensaje desconocido: %j', (mensaje) => {
      const c = callbacks();
      repartirMensajeTrabajo(mensaje, c);
      expect(c.alError.mock.calls[0][0].message).toMatch(/^Mensaje desconocido del servidor/);
      for (const otro of ['alEstado', 'alHistorial', 'alProgreso', 'alEvento']) {
        expect(c[otro]).not.toHaveBeenCalled();
      }
    },
  );

  it('no falla si falta el callback', () => {
    expect(() => repartirMensajeTrabajo({ tipo: 'estado', trabajo: {} }, {})).not.toThrow();
    expect(() => repartirMensajeTrabajo(null, {})).not.toThrow();
  });
});

describe('crearApiTrabajos', () => {
  function dobles() {
    const http = { obtener: vi.fn().mockResolvedValue([]) };
    const conexion = { enviar: vi.fn().mockReturnValue(true), cerrar: vi.fn() };
    const conectar = vi.fn().mockReturnValue(conexion);
    return { http, conexion, conectar, api: crearApiTrabajos({ http, conectar }) };
  }

  it('lista y obtiene por HTTP, codificando el id', async () => {
    const { http, api } = dobles();
    const senal = new AbortController().signal;
    await api.listar({ tipo: 'comparacion', senal });
    await api.listar();
    await api.obtener('a/b');
    expect(http.obtener.mock.calls).toEqual([
      ['/trabajos', { parametros: { tipo: 'comparacion' }, senal }],
      ['/trabajos', { parametros: { tipo: undefined }, senal: undefined }],
      ['/trabajos/a%2Fb'],
    ]);
  });

  it('sigue por el WebSocket del trabajo y manda las órdenes comunes y propias', () => {
    const { conectar, conexion, api } = dobles();
    const seguido = api.seguir('t?1', {});
    expect(conectar.mock.calls[0][0]).toBe('/trabajos/t%3F1');
    seguido.pausar();
    seguido.reanudar();
    seguido.cancelar();
    expect(seguido.ordenar({ tipo: 'vista', activa: true })).toBe(true);
    seguido.cerrar();
    expect(conexion.enviar.mock.calls.map(([m]) => m)).toEqual([{ tipo: 'pausar' }, { tipo: 'reanudar' },
      { tipo: 'cancelar' }, { tipo: 'vista', activa: true }]);
    expect(conexion.cerrar).toHaveBeenCalledOnce();
  });

  it('devuelve false si la orden no sale', () => {
    const { conexion, api } = dobles();
    conexion.enviar.mockReturnValue(false);
    expect(api.seguir('t1', {}).pausar()).toBe(false);
  });

  it('pasa mensajes, imágenes, errores y cambios de conexión a los callbacks', () => {
    const { conectar, api } = dobles();
    const c = { ...callbacks(), alImagen: vi.fn(), alCambiarEstado: vi.fn() };
    api.seguir('t1', c);
    const opciones = conectar.mock.calls[0][1];
    const imagen = new Blob(['x']);
    opciones.alMensaje({ tipo: 'progreso', medidas: { m: 1 } });
    opciones.alImagen(imagen);
    opciones.alError(new Error('red'));
    opciones.alCambiarEstado('cerrada', { codigo: 4404, error: 'No existe' });
    expect(c.alProgreso).toHaveBeenCalledWith({ m: 1 });
    expect(c.alImagen).toHaveBeenCalledWith(imagen);
    expect(c.alError.mock.calls[0][0].message).toBe('red');
    expect(c.alCambiarEstado).toHaveBeenCalledWith('cerrada', { codigo: 4404, error: 'No existe' });
  });

  it('sin callbacks no falla', () => {
    const { conectar, api } = dobles();
    api.seguir('t1');
    const opciones = conectar.mock.calls[0][1];
    expect(() => {
      opciones.alMensaje({ tipo: 'estado', trabajo: {} });
      opciones.alImagen(new Blob(['x']));
      opciones.alError(new Error('x'));
      opciones.alCambiarEstado('abierta');
    }).not.toThrow();
  });
});
