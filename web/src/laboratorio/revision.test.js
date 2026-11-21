import { describe, expect, it } from 'vitest';

import { contar, eleccionInicial, validarRevision } from './revision.js';

const PARTIDAS = [
  { indice: 0, pasos: 20, retorno: 20, semilla: 1 },
  { indice: 1, pasos: 5, retorno: 5, semilla: 2 },
];

describe('eleccionInicial', () => {
  it('elige todas las partidas enteras', () => {
    expect(eleccionInicial(PARTIDAS)).toEqual({
      0: { incluir: true, desde: 1, hasta: 20 },
      1: { incluir: true, desde: 1, hasta: 5 },
    });
  });

  it('sin partidas no elige nada', () => {
    expect(eleccionInicial([])).toEqual({});
  });
});

describe('validarRevision', () => {
  it('pasa los tramos de 1–n incluido a [n0, fin) desde 0', () => {
    const elecciones = { 0: { incluir: true, desde: 3, hasta: 10 }, 1: { incluir: true, desde: '1', hasta: '5' } };
    expect(validarRevision(PARTIDAS, elecciones, ' humanas ')).toEqual({
      valida: true,
      errores: { partidas: {} },
      seleccion: [{ indice: 0, tramo: [2, 10] }, { indice: 1, tramo: [0, 5] }],
      totales: { partidas: 2, pares: 13 },
    });
  });

  it('las partidas sin marcar no se guardan ni se comprueban', () => {
    const elecciones = { 0: { incluir: false, desde: 'x', hasta: 99 }, 1: { incluir: true, desde: 2, hasta: 2 } };
    const revision = validarRevision(PARTIDAS, elecciones, 'a');
    expect(revision.valida).toBe(true);
    expect(revision.seleccion).toEqual([{ indice: 1, tramo: [1, 2] }]);
    expect(revision.totales).toEqual({ partidas: 1, pares: 1 });
  });

  it('en una partida de 5 pasos acepta exactamente los 15 tramos posibles', () => {
    // Dominio finito: todos los pares (desde, hasta) entre 0 y 6.
    let validos = 0;
    for (let desde = 0; desde <= 6; desde += 1) {
      for (let hasta = 0; hasta <= 6; hasta += 1) {
        const revision = validarRevision([PARTIDAS[1]], { 1: { incluir: true, desde, hasta } }, 'a');
        const esperado = desde >= 1 && hasta <= 5 && desde <= hasta;
        expect(revision.valida).toBe(esperado);
        if (esperado) {
          validos += 1;
          expect(revision.seleccion).toEqual([{ indice: 1, tramo: [desde - 1, hasta] }]);
          expect(revision.totales.pares).toBe(hasta - desde + 1);
        } else {
          expect(revision.errores.partidas[1]).toMatch(/entre 1 y 5/);
        }
      }
    }
    expect(validos).toBe(15);
  });

  it.each(['', ' ', '1.5', '-1', '2e1', 'uno', null, undefined])('rechaza «%s» como extremo',
    (valor) => {
      const revision = validarRevision(PARTIDAS, { 0: { incluir: true, desde: valor, hasta: 3 },
        1: { incluir: false } }, 'a');
      expect(revision.valida).toBe(false);
      expect(revision.errores.partidas[0]).toBe('Escribe números enteros.');
      expect(revision.seleccion).toEqual([]);
    });

  it.each(['', '   ', null, undefined, 3])('exige un nombre (%j)', (nombre) => {
    const revision = validarRevision(PARTIDAS, eleccionInicial(PARTIDAS), nombre);
    expect(revision.valida).toBe(false);
    expect(revision.errores.nombre).toBe('Escribe un nombre para el conjunto.');
  });

  it('exige al menos una partida', () => {
    for (const elecciones of [{}, { 0: { incluir: false }, 1: { incluir: false } }]) {
      const revision = validarRevision(PARTIDAS, elecciones, 'a');
      expect(revision.valida).toBe(false);
      expect(revision.errores.general).toBe('Elige al menos una partida.');
      expect(revision.totales).toEqual({ partidas: 0, pares: 0 });
    }
    expect(validarRevision([], {}, 'a').errores.general).toBe('Elige al menos una partida.');
  });
});

describe('contar', () => {
  it.each([[0, '0 partidas'], [1, '1 partida'], [2, '2 partidas']])('%i da «%s»', (n, texto) => {
    expect(contar(n, 'partida', 'partidas')).toBe(texto);
  });

  it('usa el plural que se le da', () => {
    expect(contar(1, 'par', 'pares')).toBe('1 par');
    expect(contar(3, 'par', 'pares')).toBe('3 pares');
  });
});
