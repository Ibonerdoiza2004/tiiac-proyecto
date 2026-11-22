import { describe, expect, it } from 'vitest';

import { VISTAS, VISTA_INICIAL, buscarVista } from './vistas.js';

describe('VISTAS', () => {
  it('son Laboratorio, Comparar y Biblioteca, en ese orden', () => {
    expect(VISTAS.map((vista) => vista.etiqueta)).toEqual(['Laboratorio', 'Comparar', 'Biblioteca']);
  });

  it('tienen claves, títulos y subtítulos distintos y no vacíos', () => {
    for (const campo of ['id', 'titulo', 'subtitulo']) {
      const valores = VISTAS.map((vista) => vista[campo]);
      expect(valores.every((valor) => typeof valor === 'string' && valor.trim())).toBe(true);
      expect(new Set(valores).size).toBe(VISTAS.length);
    }
  });

  it('arranca en el Laboratorio', () => {
    expect(VISTA_INICIAL).toBe('laboratorio');
  });
});

describe('buscarVista', () => {
  it.each(VISTAS.map((vista) => vista.id))('encuentra %s', (id) => {
    expect(buscarVista(id).id).toBe(id);
  });

  it.each([['Laboratorio'], [''], [undefined], [null]])('rechaza %j', (id) => {
    expect(() => buscarVista(id)).toThrow(RangeError);
  });
});
