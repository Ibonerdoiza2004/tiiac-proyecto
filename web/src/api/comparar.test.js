import { describe, expect, it, vi } from 'vitest';

import { ALEATORIO, MAXIMO_PARTIDAS, TIPOS_MODELO, comparacionesDe, crearApiComparar } from './comparar.js';

describe('constantes', () => {
  it('compara expertos, GAIL y BC, con el aleatorio aparte', () => {
    expect(TIPOS_MODELO).toEqual(['experto', 'gail', 'bc']);
    expect(ALEATORIO).toBe('aleatorio');
    expect(MAXIMO_PARTIDAS).toBe(1000);
  });
});

describe('comparacionesDe', () => {
  it('se queda con las comparaciones del entorno, de la más reciente a la más antigua', () => {
    const trabajos = [
      { id: '1', tipo: 'comparacion', creado: 10, argumentos: { entorno: 'cartpole' } },
      { id: '2', tipo: 'entrenamiento', creado: 30, argumentos: { entorno: 'cartpole' } },
      { id: '3', tipo: 'comparacion', creado: 20, argumentos: { entorno: 'acrobot' } },
      { id: '4', tipo: 'comparacion', creado: 40, argumentos: { entorno: 'cartpole' } },
      { id: '5', tipo: 'comparacion', creado: 50 },
    ];
    expect(comparacionesDe(trabajos, 'cartpole').map((t) => t.id)).toEqual(['4', '1']);
    expect(comparacionesDe([], 'cartpole')).toEqual([]);
  });
});

describe('crearApiComparar', () => {
  function dobles() {
    const http = { obtener: vi.fn().mockResolvedValue([]), enviar: vi.fn().mockResolvedValue({ id: 'c1' }) };
    const trabajos = { listar: vi.fn().mockResolvedValue([
      { id: 'a', tipo: 'comparacion', creado: 1, argumentos: { entorno: 'cartpole' } },
      { id: 'b', tipo: 'comparacion', creado: 2, argumentos: { entorno: 'acrobot' } }]),
    seguir: vi.fn().mockReturnValue('conexion') };
    return { http, trabajos, api: crearApiComparar({ http, trabajos }) };
  }

  it('pide los modelos del entorno a la biblioteca real', async () => {
    const { http, api } = dobles();
    await api.modelos('cartpole');
    expect(http.obtener).toHaveBeenCalledWith('/biblioteca/modelos', { parametros: { entorno: 'cartpole' } });
  });

  it('lista las comparaciones del entorno con el protocolo de trabajos', async () => {
    const { trabajos, api } = dobles();
    expect((await api.comparaciones('cartpole')).map((t) => t.id)).toEqual(['a']);
    expect(trabajos.listar).toHaveBeenCalledWith({ tipo: 'comparacion' });
  });

  it('crea con POST /comparaciones y sigue con el WebSocket de trabajos', async () => {
    const { http, trabajos, api } = dobles();
    const senal = new AbortController().signal;
    const peticion = { entorno: 'cartpole', modelos: ['x'], aleatorio: true, vista: false };
    await api.crear(peticion, { senal });
    await api.crear(peticion);
    expect(http.enviar.mock.calls).toEqual([['/comparaciones', peticion, { senal }],
      ['/comparaciones', peticion, { senal: undefined }]]);
    const callbacks = {};
    expect(api.seguir('c1', callbacks)).toBe('conexion');
    expect(trabajos.seguir).toHaveBeenCalledWith('c1', callbacks);
  });
});
