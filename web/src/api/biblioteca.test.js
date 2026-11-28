import { describe, expect, it, vi } from 'vitest';

import { LARGO_NOMBRE, TIPOS_MODELO, crearApiBiblioteca } from './biblioteca.js';
import { TIPOS_MODELO as TIPOS_COMPARAR } from './comparar.js';

function dobles() {
  const http = { obtener: vi.fn().mockResolvedValue([]), enviar: vi.fn().mockResolvedValue({ id: 's1' }) };
  return { http, api: crearApiBiblioteca({ http }) };
}

describe('constantes', () => {
  it('son las del contrato y Comparar usa los mismos tipos', () => {
    expect(TIPOS_MODELO).toEqual(['experto', 'gail', 'bc']);
    expect(TIPOS_COMPARAR).toBe(TIPOS_MODELO);
    expect(LARGO_NOMBRE).toBe(100);
  });
});

describe('crearApiBiblioteca', () => {
  it('lista modelos y demostraciones del entorno con las rutas que ya existen', async () => {
    const { http, api } = dobles();
    await api.modelos('cartpole');
    await api.demostraciones('mountaincar');
    expect(http.obtener.mock.calls).toEqual([
      ['/biblioteca/modelos', { parametros: { entorno: 'cartpole' } }],
      ['/biblioteca/demostraciones', { parametros: { entorno: 'mountaincar' } }],
    ]);
  });

  it('pide un elemento y sus partidas codificando el id', async () => {
    const { http, api } = dobles();
    await api.elemento('a/b?c');
    await api.partidas('a/b?c');
    expect(http.obtener.mock.calls).toEqual([['/biblioteca/a%2Fb%3Fc'], ['/biblioteca/a%2Fb%3Fc/partidas']]);
  });

  it('crea un subconjunto con POST y pasa la señal de cancelación', async () => {
    const { http, api } = dobles();
    const senal = new AbortController().signal;
    const peticion = { conjunto: 'c1', partidas: [0, 2], nombre: 'mío' };
    expect(await api.crearSubconjunto(peticion, { senal })).toEqual({ id: 's1' });
    await api.crearSubconjunto(peticion);
    expect(http.enviar.mock.calls).toEqual([['/biblioteca/subconjuntos', peticion, { senal }],
      ['/biblioteca/subconjuntos', peticion, { senal: undefined }]]);
  });

  it('deja pasar los errores del servidor', async () => {
    const { http, api } = dobles();
    http.obtener.mockRejectedValueOnce(new Error('404'));
    await expect(api.elemento('x')).rejects.toThrow('404');
  });
});
