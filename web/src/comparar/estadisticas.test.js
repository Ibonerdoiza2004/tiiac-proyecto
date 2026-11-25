import { describe, expect, it } from 'vitest';

import { acumular, acumularTodo, clasificar, desdeResultado, partidasCompletas, resumir,
  resumirModelo } from './estadisticas.js';

describe('acumular', () => {
  it('guarda el retorno por modelo y partida sin modificar lo anterior', () => {
    const vacio = {};
    const uno = acumular(vacio, { modelo: 'a', partida: 0, retorno: 5, pasos: 9 });
    const dos = acumular(uno, { modelo: 'a', partida: 1, retorno: -2.5 });
    expect(vacio).toEqual({});
    expect(uno).toEqual({ a: { 0: 5 } });
    expect(dos).toEqual({ a: { 0: 5, 1: -2.5 } });
  });

  it('no cuenta dos veces la misma partida (historial y directo a la vez)', () => {
    const uno = acumular({}, { modelo: 'a', partida: 0, retorno: 5 });
    expect(acumular(uno, { modelo: 'a', partida: 0, retorno: 99 })).toBe(uno);
  });

  it.each([
    ['sin medidas', undefined],
    ['null', null],
    ['modelo vacío', { modelo: '', partida: 0, retorno: 1 }],
    ['modelo no texto', { modelo: 3, partida: 0, retorno: 1 }],
    ['partida negativa', { modelo: 'a', partida: -1, retorno: 1 }],
    ['partida decimal', { modelo: 'a', partida: 0.5, retorno: 1 }],
    ['partida texto', { modelo: 'a', partida: '0', retorno: 1 }],
    ['retorno texto', { modelo: 'a', partida: 0, retorno: '1' }],
    ['retorno NaN', { modelo: 'a', partida: 0, retorno: Number.NaN }],
    ['retorno infinito', { modelo: 'a', partida: 0, retorno: Infinity }],
    ['retorno null (NaN del servidor)', { modelo: 'a', partida: 0, retorno: null }],
  ])('ignora medidas no válidas: %s', (_, medidas) => {
    const antes = { b: { 0: 1 } };
    expect(acumular(antes, medidas)).toBe(antes);
  });

  it('acumularTodo rehace todo desde el historial', () => {
    expect(acumularTodo([])).toEqual({});
    expect(acumularTodo([{ modelo: 'a', partida: 0, retorno: 1 }, { modelo: 'b', partida: 0, retorno: 2 },
      { modelo: 'a', partida: 0, retorno: 3 }])).toEqual({ a: { 0: 1 }, b: { 0: 2 } });
  });
});

describe('desdeResultado', () => {
  it('convierte los retornos guardados al formato del progreso', () => {
    expect(desdeResultado({ modelos: [{ id: 'a', retornos: [1, 2] }, { id: 'aleatorio', retornos: [] }] }))
      .toEqual({ a: { 0: 1, 1: 2 } });
  });

  it.each([[undefined], [null], [{}], [{ modelos: 'x' }]])('sin modelos da null: %j', (resultado) => {
    expect(desdeResultado(resultado)).toBeNull();
  });

  it('salta elementos y retornos que no valen', () => {
    expect(desdeResultado({ modelos: [null, { id: 'a' }, { id: 'b', retornos: [1, null, 3] }] }))
      .toEqual({ b: { 0: 1, 2: 3 } });
  });
});

describe('resumirModelo', () => {
  it('sin partidas no tiene media ni desviación', () => {
    expect(resumirModelo('a')).toEqual({ modelo: 'a', partidas: 0, media: null, desviacion: null,
      ultima: null, retornos: [], porPartida: {} });
  });

  it('con una partida hay media pero no desviación', () => {
    expect(resumirModelo('a', { 0: 7 })).toMatchObject({ partidas: 1, media: 7, desviacion: null, ultima: 7 });
  });

  it('calcula la media y la desviación típica muestral (n − 1), en orden de partida', () => {
    // Llegan desordenadas: la última es la de mayor índice.
    const r = resumirModelo('a', { 2: 6, 0: 2, 1: 4 });
    expect(r.retornos).toEqual([2, 4, 6]);
    expect(r.media).toBe(4);
    expect(r.desviacion).toBe(2);
    expect(r.ultima).toBe(6);
  });

  it('ordena las partidas como números, no como texto', () => {
    const porPartida = Object.fromEntries(Array.from({ length: 11 }, (_, i) => [i, i]));
    expect(resumirModelo('a', porPartida).ultima).toBe(10);
  });

  it('con retornos iguales la desviación es 0', () => {
    expect(resumirModelo('a', { 0: 500, 1: 500, 2: 500 }).desviacion).toBe(0);
  });
});

describe('resumir, partidasCompletas y clasificar', () => {
  const acumulado = { a: { 0: 1, 1: 3 }, b: { 0: 10 } };

  it('resume en el orden pedido, también modelos sin partidas', () => {
    expect(resumir(acumulado, ['b', 'c', 'a']).map((r) => [r.modelo, r.partidas, r.media]))
      .toEqual([['b', 1, 10], ['c', 0, null], ['a', 2, 2]]);
  });

  it('cuenta las partidas que han jugado todos', () => {
    expect(partidasCompletas(acumulado, ['a', 'b'])).toBe(1);
    expect(partidasCompletas(acumulado, ['a'])).toBe(2);
    expect(partidasCompletas(acumulado, ['a', 'c'])).toBe(0);
    expect(partidasCompletas(acumulado, [])).toBe(0);
  });

  it('clasifica de mayor a menor media, sin partidas al final, sin modificar la lista', () => {
    const resumenes = resumir({ a: { 0: -5 }, b: { 0: 3 }, d: { 0: 3 } }, ['c', 'a', 'b', 'e', 'd']);
    const orden = clasificar(resumenes).map((r) => r.modelo);
    expect(orden.slice(0, 2).sort()).toEqual(['b', 'd']);
    expect(orden.slice(2)).toEqual(['a', 'c', 'e']);
    expect(resumenes.map((r) => r.modelo)).toEqual(['c', 'a', 'b', 'e', 'd']);
  });
});
