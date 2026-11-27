import { render, screen, within } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { resumir } from './estadisticas.js';
import Resultados, { textoDatos, textoHistorial } from './Resultados.jsx';

describe('textoHistorial', () => {
  it.each([
    [{ detalles: { pasos: 60000, segundos: 25 } }, '60.000 pasos · 25 s'],
    [{ detalles: { iteraciones: 20, pasos: 100000, segundos: 300 } }, '100.000 pasos · 20 iteraciones · 5 min'],
    [{ detalles: { epocas: 3, incompleto: true } }, '3 épocas · incompleto'],
    [{ detalles: { segundos: 119.6 } }, '120 s'],
    [{ detalles: { segundos: 120 } }, '2 min'],
    [{ detalles: { pasos: 'muchos' } }, '—'],
    [{ detalles: {} }, '—'],
    [{}, '—'],
    [undefined, '—'],
  ])('%j → %s', (modelo, texto) => {
    expect(textoHistorial(modelo)).toBe(texto);
  });
});

describe('textoDatos', () => {
  it.each([
    [{ tipo: 'gail', detalles: { cantidad: 1, repeticion: 0 } }, '1 partida · repetición 0'],
    [{ tipo: 'bc', detalles: { subconjunto: 10, repeticion: 4 } }, '10 partidas · repetición 4'],
    [{ tipo: 'bc', detalles: { subconjunto: 4 } }, '4 partidas'],
    [{ tipo: 'experto', detalles: { pasos: 1 } }, 'refuerzo (TRPO)'],
    [{ tipo: 'gail', detalles: {} }, '—'],
    [undefined, '—'],
  ])('%j → %s', (modelo, texto) => {
    expect(textoDatos(modelo)).toBe(texto);
  });
});

describe('Resultados', () => {
  const nombres = { a: 'Modelo A', b: 'Modelo B', c: 'Modelo C' };
  const pintar = (acumulado, ids) => render(<Resultados resumenes={resumir(acumulado, ids)}
    nombre={(id) => nombres[id]} modelo={(id) => (id === 'a' ? { tipo: 'experto', detalles: { pasos: 10 } } : undefined)} />);

  it('sin partidas avisa de que los resultados llegarán', () => {
    pintar({}, ['a', 'b']);
    expect(screen.getByText('Los resultados aparecerán con la primera partida.')).toBeInTheDocument();
    expect(screen.queryByText('Puntuación de cada partida')).toBeNull();
  });

  it('barras de mejor a peor, con la mejor al 100 % y la peor al 0 % si es negativa', () => {
    const { container } = pintar({ a: { 0: -10 }, b: { 0: 30 }, c: { 0: 10 } }, ['a', 'b', 'c']);
    const barras = [...container.querySelectorAll('.barra-fila')];
    expect(barras.map((b) => b.firstChild.textContent)).toEqual(['Modelo B', 'Modelo C', 'Modelo A']);
    expect(barras.map((b) => b.querySelector('.barra-pista span').style.width)).toEqual(['100%', '50%', '0%']);
  });

  it('con todo positivo la escala empieza en 0', () => {
    const { container } = pintar({ a: { 0: 50 }, b: { 0: 100 } }, ['a', 'b']);
    expect([...container.querySelectorAll('.barra-pista span')].map((s) => s.style.width)).toEqual(['100%', '50%']);
  });

  it('un modelo sin partidas aún sale con guiones y al final', () => {
    pintar({ a: { 0: 5, 1: 7 } }, ['b', 'a']);
    const tabla = screen.getByRole('table', { name: 'Resultados por modelo, de mejor a peor media' });
    const filas = within(tabla).getAllByRole('row').slice(1).map((f) => f.textContent);
    expect(filas).toEqual(['Modelo A26 ± 1,417', 'Modelo B0——']);
  });

  it('la tabla de partidas deja huecos donde un modelo aún no ha jugado', () => {
    const { container } = pintar({ a: { 0: 5, 1: 7 }, b: { 0: 1 } }, ['a', 'b']);
    const filas = [...container.querySelector('.detalle-partidas tbody').rows].map((f) => f.textContent);
    expect(filas).toEqual(['151', '27—']);
  });

  it('la tabla de partidas va por número de partida aunque falte alguna', () => {
    // Regresión: iba por posición y, si faltaba la partida 2, la 3 subía a su fila.
    const { container } = pintar({ a: { 0: 1, 2: 3 }, b: { 0: 5, 1: 6, 2: 7 } }, ['a', 'b']);
    const filas = [...container.querySelector('.detalle-partidas tbody').rows]
      .map((f) => [...f.cells].map((c) => c.textContent));
    expect(filas).toEqual([['1', '1', '5'], ['2', '—', '6'], ['3', '3', '7']]);
  });

  it('el historial solo se rellena para modelos de la biblioteca', () => {
    const { container } = pintar({ a: { 0: 1 }, b: { 0: 1 } }, ['a', 'b']);
    const historial = [...container.querySelectorAll('.resultados-bloque')[1].querySelectorAll('tbody tr')]
      .map((f) => [...f.cells].map((c) => c.textContent));
    expect(historial).toEqual([['Modelo A', '10 pasos', 'refuerzo (TRPO)'], ['Modelo B', 'No aplica', 'No aplica']]);
  });
});
