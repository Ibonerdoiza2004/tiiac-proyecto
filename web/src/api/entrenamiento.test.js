import { describe, expect, it, vi } from 'vitest';

import { MENSAJES_ENTRENAMIENTO, METODOS, METODOS_CON_DATOS, METODOS_CON_VISTA,
  crearApiEntrenamiento, repartirMensajeEntrenamiento } from './entrenamiento.js';

function dobles() {
  const http = { obtener: vi.fn().mockResolvedValue([]), enviar: vi.fn().mockResolvedValue({ id: 'e1' }) };
  const conexion = { enviar: vi.fn().mockReturnValue(true), cerrar: vi.fn() };
  const conectar = vi.fn().mockReturnValue(conexion);
  return { http, conexion, conectar, api: crearApiEntrenamiento({ http, conectar }) };
}

describe('crearApiEntrenamiento', () => {
  it('usa las rutas HTTP del contrato y pasa la señal de cancelación', async () => {
    const { http, api } = dobles();
    const senal = new AbortController().signal;
    await api.ajustes('cartpole', 'gail', { senal });
    await api.demostraciones('mountain car');
    await api.crear({ entorno: 'cartpole', metodo: 'bc' }, { senal });
    expect(http.obtener.mock.calls).toEqual([
      ['/laboratorio/cartpole/ajustes/gail', { senal }],
      ['/biblioteca/demostraciones', { parametros: { entorno: 'mountain car' }, senal: undefined }],
    ]);
    expect(http.enviar).toHaveBeenCalledWith('/entrenamientos', { entorno: 'cartpole', metodo: 'bc' }, { senal });
  });

  it('codifica los identificadores en las rutas', async () => {
    const { http, conectar, api } = dobles();
    await api.ajustes('a/b', 'c?d');
    api.conectar('e/1', {});
    expect(http.obtener.mock.calls[0][0]).toBe('/laboratorio/a%2Fb/ajustes/c%3Fd');
    expect(conectar.mock.calls[0][0]).toBe('/entrenamientos/e%2F1');
  });

  it('envía las órdenes por el WebSocket', () => {
    const { conexion, api } = dobles();
    const entreno = api.conectar('e1', {});
    expect(entreno.pausar()).toBe(true);
    entreno.reanudar();
    entreno.cambiarVista(1);
    entreno.cambiarVista(false);
    entreno.finalizar();
    entreno.cerrar();
    expect(conexion.enviar.mock.calls.map(([m]) => m)).toEqual([
      { tipo: 'pausar' }, { tipo: 'reanudar' }, { tipo: 'vista', activa: true },
      { tipo: 'vista', activa: false }, { tipo: 'finalizar' }]);
    expect(conexion.cerrar).toHaveBeenCalledOnce();
  });

  it('pasa imágenes, mensajes, errores y cambios de conexión a los callbacks', () => {
    const { conectar, api } = dobles();
    const c = { alImagen: vi.fn(), alProgreso: vi.fn(), alError: vi.fn(), alCambiarEstado: vi.fn() };
    api.conectar('e1', c);
    const opciones = conectar.mock.calls[0][1];
    const imagen = new Blob(['x']);
    opciones.alImagen(imagen);
    opciones.alMensaje({ tipo: 'progreso', paso: 3, medidas: { m: 1 } });
    opciones.alError(new Error('red'));
    opciones.alCambiarEstado('cerrada');
    expect(c.alImagen).toHaveBeenCalledWith(imagen);
    expect(c.alProgreso).toHaveBeenCalledWith({ paso: 3, medidas: { m: 1 } });
    expect(c.alError.mock.calls[0][0].message).toBe('red');
    expect(c.alCambiarEstado).toHaveBeenCalledWith('cerrada');
  });

  it('funciona sin callbacks', () => {
    const { conectar, api } = dobles();
    api.conectar('e1');
    const { alMensaje, alImagen, alError, alCambiarEstado } = conectar.mock.calls[0][1];
    expect(() => {
      alMensaje({ tipo: 'estado', estado: 'pausado' });
      alImagen(new Blob(['x']));
      alError(new Error('x'));
      alCambiarEstado('cerrada');
    }).not.toThrow();
  });
});

describe('repartirMensajeEntrenamiento', () => {
  const callbacks = () => ({ alProgreso: vi.fn(), alEstado: vi.fn(), alTerminar: vi.fn(), alError: vi.fn() });
  const modelo = { nombre: 'gail_cartpole_1', incompleto: false };

  it.each([
    [{ tipo: MENSAJES_ENTRENAMIENTO.PROGRESO, paso: 4, medidas: { m: 2 } }, 'alProgreso', { paso: 4, medidas: { m: 2 } }],
    [{ tipo: 'progreso', paso: 4 }, 'alProgreso', { paso: 4, medidas: {} }],
    [{ tipo: 'estado', estado: 'pausado' }, 'alEstado', 'pausado'],
    [{ tipo: 'terminado', modelo, resumen: { r: 1 } }, 'alTerminar', { modelo, resumen: { r: 1 } }],
    [{ tipo: 'terminado', modelo }, 'alTerminar', { modelo, resumen: {} }],
  ])('%j va a %s', (mensaje, destino, argumento) => {
    const c = callbacks();
    repartirMensajeEntrenamiento(mensaje, c);
    expect(c[destino]).toHaveBeenCalledExactlyOnceWith(argumento);
    expect(Object.entries(c).filter(([n, f]) => n !== destino && f.mock.calls.length)).toEqual([]);
  });

  it.each([
    [{ tipo: 'error', mensaje: 'Sin memoria' }, 'Sin memoria'],
    [{ tipo: 'error' }, 'Error del servidor'],
    [{ tipo: 'otro' }, 'Mensaje desconocido del servidor: {"tipo":"otro"}'],
    [null, 'Mensaje desconocido del servidor: null'],
    ['x', 'Mensaje desconocido del servidor: "x"'],
  ])('%j es un error', (mensaje, texto) => {
    const c = callbacks();
    repartirMensajeEntrenamiento(mensaje, c);
    expect(c.alError.mock.calls[0][0].message).toBe(texto);
  });
});

describe('métodos', () => {
  it('BC no tiene vista y el experto no usa datos', () => {
    expect([...METODOS_CON_VISTA].sort()).toEqual([METODOS.EXPERTO, METODOS.GAIL]);
    expect([...METODOS_CON_DATOS].sort()).toEqual([METODOS.BC, METODOS.GAIL]);
  });
});
