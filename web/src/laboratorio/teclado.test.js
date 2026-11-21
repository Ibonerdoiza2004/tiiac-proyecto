import { describe, expect, it } from 'vitest';

import { ENTORNOS } from '../entornos.js';
import { accionDeTeclas, accionesBotones, ayudaTeclado, esTeclaDeJuego } from './teclado.js';

const DE_JUEGO = ['ArrowLeft', 'ArrowRight', 'a', 'A', 'd', 'D'];

describe('accionDeTeclas', () => {
  it.each([
    ['cartpole', 'ArrowLeft', 0], ['cartpole', 'a', 0], ['cartpole', 'A', 0],
    ['cartpole', 'ArrowRight', 1], ['cartpole', 'd', 1], ['cartpole', 'D', 1],
    ['mountaincar', 'ArrowLeft', 0], ['mountaincar', 'a', 0],
    ['mountaincar', 'ArrowRight', 2], ['mountaincar', 'D', 2],
  ])('en %s, %s da la acción %i', (entorno, tecla, accion) => {
    expect(accionDeTeclas(entorno, [tecla])).toBe(accion);
  });

  it.each(['cartpole', 'mountaincar'])('en %s, sin teclas no hay acción', (entorno) => {
    expect(accionDeTeclas(entorno, [])).toBeNull();
    expect(accionDeTeclas(entorno, ['Shift', ' ', 'ArrowUp', 'Enter'])).toBeNull();
  });

  it('con varias teclas manda la última de juego que se pulsó', () => {
    expect(accionDeTeclas('cartpole', ['ArrowLeft', 'ArrowRight'])).toBe(1);
    expect(accionDeTeclas('cartpole', ['ArrowRight', 'ArrowLeft'])).toBe(0);
    expect(accionDeTeclas('cartpole', ['ArrowRight', 'Shift'])).toBe(1);
    expect(accionDeTeclas('mountaincar', ['d', 'a', 'x'])).toBe(0);
  });

  it.each(['acrobot', 'reacher', 'halfcheetah', 'otro'])('%s no se juega con el teclado', (entorno) => {
    for (const tecla of DE_JUEGO) {
      expect(accionDeTeclas(entorno, [tecla])).toBeNull();
      expect(esTeclaDeJuego(entorno, tecla)).toBe(false);
    }
    expect(ayudaTeclado(entorno)).toBeNull();
    expect(accionesBotones(entorno)).toBeNull();
  });
});

describe('teclado y entornos', () => {
  it('solo tienen teclas los entornos que se juegan a mano', () => {
    for (const entorno of ENTORNOS) {
      expect(esTeclaDeJuego(entorno.id, 'ArrowLeft')).toBe(entorno.jugarAMano);
      expect(ayudaTeclado(entorno.id) !== null).toBe(entorno.jugarAMano);
    }
  });

  it('los botones en pantalla dan las mismas acciones que las flechas', () => {
    for (const entorno of ['cartpole', 'mountaincar']) {
      expect(accionesBotones(entorno)).toEqual({
        izquierda: accionDeTeclas(entorno, ['ArrowLeft']),
        derecha: accionDeTeclas(entorno, ['ArrowRight']),
      });
    }
  });

  it('no confunde las propiedades de los objetos con teclas', () => {
    expect(esTeclaDeJuego('cartpole', 'toString')).toBe(false);
    expect(accionDeTeclas('cartpole', ['constructor'])).toBeNull();
  });
});
