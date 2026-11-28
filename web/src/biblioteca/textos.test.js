import { describe, expect, it } from 'vitest';

import { ETIQUETAS_DETALLES, ORDENES, errorNombre, filasDetalles, formatearCifra, formatearDuracion,
  formatearFecha, ordenarPartidas, resumenEleccion, textoTipo, valorDetalle } from './textos.js';

describe('formatearDuracion', () => {
  it.each([[0, '0 s'], [59.4, '59 s'], [119.4, '119 s'], [120, '2 min'], [7199, '120 min'], [7200, '2 h'],
    [23532, '6,54 h'], [-1, '—'], [Number.NaN, '—'], [Infinity, '—'], [null, '—'], ['10', '—']])(
    '%j → %s', (segundos, texto) => {
      expect(formatearDuracion(segundos)).toBe(texto);
    },
  );
});

describe('formatearFecha', () => {
  it('da día, mes, año y hora', () => {
    expect(formatearFecha('2026-11-01T10:05:00')).toBe('01/11/2026, 10:05');
  });

  it.each([[null, '—'], [undefined, '—'], ['', '—'], [3, '—'], ['ayer', 'ayer']])('%j → %s', (iso, texto) => {
    expect(formatearFecha(iso)).toBe(texto);
  });
});

describe('formatearCifra', () => {
  it.each([[7500, '7500'], [25001984, '25.001.984'], [-90, '-90'], [0, '0'], [3.14159, '3,14'], [-3.504, '-3,5'],
    [Number.NaN, '—'], [null, '—'], ['5', '—']])('%j → %s', (valor, texto) => {
    expect(formatearCifra(valor)).toBe(texto);
  });
});

describe('valorDetalle', () => {
  it.each([
    ['objetivo_alcanzado', true, 'Sí'],
    ['splits', false, 'No'],
    ['mejor_retorno', null, '—'],
    ['pasos', undefined, '—'],
    ['segundos', 3390, '57 min'],
    ['pasos', 100000, '100.000'],
    ['mejor_retorno', 499.876, '500'],
    ['huella_dataset', '5c5168e7f5f8a1b2c3d4', '5c5168e7f5f8…'],
    ['huella_dataset', 'corta', 'corta'],
    ['conjunto', 'abc123', 'abc123'],
    ['raro', { a: 1 }, '{"a":1}'],
    ['lista', [1, 2], '[1,2]'],
  ])('%s = %j → %s', (clave, valor, texto) => {
    expect(valorDetalle(clave, valor)).toBe(texto);
  });
});

describe('filasDetalles', () => {
  it('sigue el orden de la API, pone las etiquetas y se salta tipo y fuente', () => {
    const filas = filasDetalles({ tipo: 'conjunto', partidas: 15, pares: 7500, fuente: 'experto', splits: true,
      nuevo: 3 });
    expect(filas.map((f) => [f.etiqueta, f.texto])).toEqual([['Partidas', '15'], ['Pares estado-acción', '7500'],
      ['Reparto en train y validación', 'Sí'], ['nuevo', '3']]);
  });

  it('da el valor entero cuando el texto lo acorta', () => {
    const [fila] = filasDetalles({ huella_dataset: 'a'.repeat(64) });
    expect(fila.completo).toBe('a'.repeat(64));
    expect(filasDetalles({ conjunto: 'c1' })[0].completo).toBeNull();
    expect(filasDetalles({ pasos: 5 })[0].completo).toBeNull();
  });

  it.each([[null], [undefined], [{}]])('sin detalles no da filas: %j', (detalles) => {
    expect(filasDetalles(detalles)).toEqual([]);
  });

  it('tiene etiqueta para todo lo que mandan los detalles de src/data/biblioteca.py', () => {
    // _detalles_experto, _detalles_gail, _detalles_bc, _detalles_demostraciones y de un subconjunto.
    for (const clave of ['pasos', 'iteraciones', 'mejor_retorno', 'objetivo', 'objetivo_alcanzado', 'segundos',
      'cantidad', 'repeticion', 'huella_dataset', 'subconjunto', 'semilla', 'partidas', 'pares', 'splits',
      'conjunto']) {
      expect(ETIQUETAS_DETALLES[clave], clave).toBeTruthy();
    }
  });
});

describe('textoTipo', () => {
  it.each([
    [{ tipo: 'experto' }, 'Experto (TRPO)'],
    [{ tipo: 'gail' }, 'GAIL'],
    [{ tipo: 'bc' }, 'BC'],
    [{ tipo: 'demostraciones', detalles: { tipo: 'conjunto' } }, 'Conjunto'],
    [{ tipo: 'demostraciones', detalles: { tipo: 'subconjunto' } }, 'Subconjunto'],
    [{ tipo: 'demostraciones', detalles: {} }, 'Demostraciones'],
    [{ tipo: 'otro' }, 'otro'],
  ])('%j → %s', (elemento, texto) => {
    expect(textoTipo(elemento)).toBe(texto);
  });
});

describe('errorNombre', () => {
  it.each([['mío', null], [' a ', null], ['x'.repeat(100), null], [` ${'x'.repeat(100)} `, null],
    ['', 'Escribe un nombre.'], ['   ', 'Escribe un nombre.'], ['\t\n', 'Escribe un nombre.'],
    ['x'.repeat(101), 'Como mucho 100 caracteres.']])('%j → %j', (nombre, error) => {
    expect(errorNombre(nombre)).toBe(error);
  });
});

const PARTIDAS = [
  { id_episodio: 0, pasos: 10, retorno: 5, split: 'validacion', orden_train: null },
  { id_episodio: 1, pasos: 20, retorno: 9, split: 'train', orden_train: 1 },
  { id_episodio: 2, pasos: 30, retorno: 9, split: 'train', orden_train: 0 },
  { id_episodio: 3, pasos: 40, retorno: -1, split: null, orden_train: null },
];

describe('resumenEleccion', () => {
  it.each([[[], 0, 0], [[1], 1, 20], [[0, 3], 2, 50], [[0, 1, 2, 3], 4, 100], [[9], 0, 0]])(
    'elegidas %j → %i partidas y %i pares', (elegidas, partidas, pares) => {
      expect(resumenEleccion(PARTIDAS, new Set(elegidas))).toEqual({ partidas, pares });
    },
  );
});

describe('ordenarPartidas', () => {
  const ids = (orden) => ordenarPartidas([...PARTIDAS].reverse(), orden).map((p) => p.id_episodio);

  it('por número, por retorno (empates por número) y en el orden de train (el resto al final)', () => {
    expect(ids('id')).toEqual([0, 1, 2, 3]);
    expect(ids('retorno')).toEqual([1, 2, 0, 3]);
    expect(ids('train')).toEqual([2, 1, 0, 3]);
    expect(ids('desconocido')).toEqual([0, 1, 2, 3]);
  });

  it('no modifica la lista', () => {
    const lista = [...PARTIDAS].reverse();
    ordenarPartidas(lista, 'id');
    expect(lista.map((p) => p.id_episodio)).toEqual([3, 2, 1, 0]);
    expect(Object.keys(ORDENES)).toEqual(['id', 'retorno', 'train']);
  });
});
