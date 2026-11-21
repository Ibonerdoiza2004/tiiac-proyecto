import { readFileSync, readdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vitest';

import { ENTORNOS, ENTORNO_INICIAL, buscarEntorno } from './entornos.js';

const PERFILES = join(dirname(fileURLToPath(import.meta.url)), '..', '..', 'configs', 'entornos');

// Lee `entorno.id` del perfil YAML sin depender de un parser: es la primera clave `id`
// sangrada dentro de la sección `entorno:`.
function idGymnasium(perfil) {
  const texto = readFileSync(join(PERFILES, `${perfil}.yaml`), 'utf-8');
  const seccion = texto.split(/^entorno:\s*$/m)[1];
  return seccion.match(/^\s+id:\s*([^\s#]+)/m)[1];
}

describe('ENTORNOS', () => {
  it('tiene un entorno por cada perfil de configs/entornos', () => {
    const perfiles = readdirSync(PERFILES)
      .filter((archivo) => archivo.endsWith('.yaml'))
      .map((archivo) => archivo.replace(/\.yaml$/, ''))
      .sort();
    expect(ENTORNOS.map((entorno) => entorno.id).sort()).toEqual(perfiles);
  });

  it.each(ENTORNOS.map((entorno) => [entorno.id, entorno.gymnasium]))(
    'el id de Gymnasium de %s coincide con su perfil',
    (id, gymnasium) => {
      expect(idGymnasium(id)).toBe(gymnasium);
    },
  );

  it('sigue el orden de las etapas, de 1 a 5', () => {
    expect(ENTORNOS.map((entorno) => entorno.etapa)).toEqual([1, 2, 3, 4, 5]);
  });

  it('solo deja jugar a mano en CartPole y MountainCar (plan de la E2)', () => {
    const aMano = ENTORNOS.filter((entorno) => entorno.jugarAMano).map((entorno) => entorno.id);
    expect(aMano).toEqual(['cartpole', 'mountaincar']);
  });

  it('marca como continuas las acciones de los entornos MuJoCo', () => {
    const tipos = Object.fromEntries(ENTORNOS.map((entorno) => [entorno.id, entorno.acciones]));
    expect(tipos).toEqual({
      cartpole: 'discretas',
      mountaincar: 'discretas',
      acrobot: 'discretas',
      reacher: 'continuas',
      halfcheetah: 'continuas',
    });
  });

  it('no se puede modificar', () => {
    expect(Object.isFrozen(ENTORNOS)).toBe(true);
    expect(ENTORNOS.every((entorno) => Object.isFrozen(entorno))).toBe(true);
  });

  it('arranca con un entorno que existe', () => {
    expect(buscarEntorno(ENTORNO_INICIAL).id).toBe(ENTORNO_INICIAL);
  });
});

describe('buscarEntorno', () => {
  it.each(ENTORNOS.map((entorno) => entorno.id))('encuentra %s', (id) => {
    expect(buscarEntorno(id).id).toBe(id);
  });

  it.each([['CartPole'], ['CartPole-v1'], [''], [undefined], [null], [1]])(
    'rechaza %j',
    (id) => {
      expect(() => buscarEntorno(id)).toThrow(RangeError);
    },
  );
});
