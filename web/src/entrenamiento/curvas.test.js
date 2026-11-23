import { describe, expect, it } from 'vitest';

import { agregarProgreso, formatearNumero, nombreMedida, trazo } from './curvas.js';

describe('agregarProgreso', () => {
  it('añade un punto por medida sin tocar las series que recibe', () => {
    const antes = { perdida: [[1, 0.5]] };
    const despues = agregarProgreso(antes, { paso: 2, medidas: { perdida: 0.4, otra: 3 } });
    expect(despues).toEqual({ perdida: [[1, 0.5], [2, 0.4]], otra: [[2, 3]] });
    expect(antes).toEqual({ perdida: [[1, 0.5]] });
  });

  it.each([NaN, Infinity, null, '1', undefined, {}])('ignora el valor %j', (valor) => {
    expect(agregarProgreso({}, { paso: 1, medidas: { m: valor } })).toEqual({});
  });

  it.each([NaN, undefined, '3'])('ignora un paso %j', (paso) => {
    const series = { m: [[1, 1]] };
    expect(agregarProgreso(series, { paso, medidas: { m: 2 } })).toBe(series);
  });

  it('sin medidas no cambia nada', () => {
    expect(agregarProgreso({ m: [[1, 1]] }, { paso: 2 })).toEqual({ m: [[1, 1]] });
  });

  it('al pasar del máximo se queda con uno de cada dos y siempre con el último', () => {
    let series = {};
    for (let paso = 1; paso <= 9; paso += 1) {
      series = agregarProgreso(series, { paso, medidas: { m: paso } }, 4);
      expect(series.m.length).toBeLessThanOrEqual(4);
      expect(series.m.at(-1)).toEqual([paso, paso]);
    }
  });
});

describe('trazo', () => {
  const tamano = { ancho: 100, alto: 50 };

  it('escala los puntos al rectángulo, con el máximo arriba', () => {
    expect(trazo([[0, 0], [5, 10], [10, 5]], tamano)).toEqual({
      d: 'M0.0 50.0 L50.0 0.0 L100.0 25.0', minimo: 0, maximo: 10, desde: 0, hasta: 10 });
  });

  it('un punto o un valor constante quedan en el centro', () => {
    expect(trazo([[3, 7]], tamano).d).toBe('M50.0 25.0');
    expect(trazo([[1, 2], [3, 2]], tamano).d).toBe('M0.0 25.0 L100.0 25.0');
  });

  it('sin puntos no hay trazo', () => {
    expect(trazo([], tamano)).toBeNull();
  });
});

describe('textos', () => {
  it.each([
    ['retorno_medio', 'Retorno medio'], ['perdida_validacion', 'Pérdida en validación'],
    ['discriminador/aciertos', 'Discriminador · aciertos'], ['kl_media', 'Kl media'],
    ['constructor', 'Constructor'],
  ])('%s se llama «%s»', (medida, nombre) => {
    expect(nombreMedida(medida)).toBe(nombre);
  });

  it.each([[0.123456, '0,123'], [12345, '12.300'], [-3.5, '-3,5'], [NaN, '—'], [Infinity, '—']])(
    '%f se escribe %s', (valor, texto) => {
      expect(formatearNumero(valor)).toBe(texto);
    });
});
