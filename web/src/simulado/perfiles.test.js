import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vitest';

import { OPCIONES, generarPerfiles, textoPerfiles } from '../../scripts/perfiles.mjs';
import { ENTORNOS } from '../entornos.js';
import PERFILES from './perfiles.json';

const RAIZ = join(dirname(fileURLToPath(import.meta.url)), '..', '..', '..');
const leer = (ruta) => readFileSync(join(RAIZ, ruta), 'utf-8');
const campo = (entorno, metodo, seccion, clave) => PERFILES[entorno][metodo].secciones
  .find((s) => s.clave === seccion).campos.find((c) => c.clave === clave);

describe('perfiles.json', () => {
  it('está al día con configs/ (si falla: npm run perfiles)', () => {
    expect(JSON.stringify(PERFILES, null, 2) + '\n').toBe(textoPerfiles(generarPerfiles()));
  });

  it('tiene los tres métodos de cada entorno', () => {
    expect(Object.keys(PERFILES).sort()).toEqual(ENTORNOS.map((e) => e.id).sort());
    for (const entorno of Object.values(PERFILES)) {
      expect(Object.keys(entorno).sort()).toEqual(['bc', 'experto', 'gail']);
    }
  });

  // Valores leídos con src/utils/configuracion.py el 2026-11-23 (resolver_configuracion).
  it.each([
    ['cartpole', 'gail', 'gail', 'iteraciones', 20, 'entero'],
    ['halfcheetah', 'gail', 'gail', 'iteraciones', 500, 'entero'],
    ['cartpole', 'gail', 'trpo', 'pasos_por_iteracion', 5000, 'entero'],
    ['halfcheetah', 'gail', 'trpo', 'pasos_por_iteracion', 50000, 'entero'],
    ['halfcheetah', 'experto', 'experto.trpo', 'capas_ocultas', [64, 64], 'lista_enteros'],
    ['halfcheetah', 'experto', 'experto.trpo', 'pasos_por_iteracion', 2048, 'entero'],
    ['cartpole', 'experto', 'experto.trpo', 'pasos_por_iteracion', 5000, 'entero'],
    ['reacher', 'experto', 'experto', 'retorno_objetivo', -3.5, 'real'],
    ['cartpole', 'experto', 'experto', 'retorno_objetivo', 500, 'real'],
    ['halfcheetah', 'experto', 'experto', 'pasos_totales', 25000000, 'entero'],
    ['cartpole', 'bc', 'demostraciones', 'frecuencia_pares', 10, 'entero'],
    ['halfcheetah', 'bc', 'demostraciones', 'frecuencia_pares', 20, 'entero'],
    ['cartpole', 'bc', 'bc', 'log_std_inicial', 0, 'real'],
    ['cartpole', 'gail', 'trpo', 'limite_normalizacion', 10, 'real'],
    ['cartpole', 'gail', 'gail', 'recompensa', 'menos_log_uno_menos_d', 'opcion'],
    ['cartpole', 'gail', 'gail.discriminador', 'learning_rate', 0.01, 'real'],
  ])('%s %s: %s.%s vale %j (%s)', (entorno, metodo, seccion, clave, valor, tipo) => {
    expect(campo(entorno, metodo, seccion, clave)).toMatchObject({ valor, tipo });
  });

  it('no ofrece semillas ni secciones anidadas como campos', () => {
    for (const metodos of Object.values(PERFILES)) {
      for (const { secciones } of Object.values(metodos)) {
        for (const s of secciones) {
          expect(s.campos.map((c) => c.clave)).not.toContain('semilla');
          expect(s.campos.every((c) => c.valor !== null && typeof c.valor !== 'object'
            || Array.isArray(c.valor))).toBe(true);
        }
      }
    }
  });

  it('la duración es la del perfil', () => {
    expect(PERFILES.cartpole.experto.duracion).toEqual({ unidad: 'pasos', total: 100000 });
    expect(PERFILES.cartpole.bc.duracion).toEqual({ unidad: 'epocas', total: 500 });
    expect(PERFILES.cartpole.gail.duracion).toEqual({ unidad: 'iteraciones', total: 20 });
  });

  it('guarda de dónde sale cada valor cuando el YAML lo comenta', () => {
    expect(campo('cartpole', 'gail', 'trpo', 'gamma').ayuda).toBe('Paper Ho y Ermon (2016), apéndice B.');
    // El de un valor que cambia el perfil es el del perfil, no el de la base («tabla 2»).
    // (configs/entornos/cartpole.yaml frente a configs/default.yaml).
    const iteraciones = campo('cartpole', 'gail', 'gail', 'iteraciones').ayuda;
    expect(iteraciones).toMatch(/^Elección propia por regla 5\/20/);
    expect(iteraciones).not.toMatch(/^Iteraciones de GAIL/);
  });

  it('las opciones coinciden con las del código de Python', () => {
    const claves = (texto, nombre) => [...texto.slice(texto.indexOf(`${nombre} = `)).split('\n\n')[0]
      .matchAll(/"([a-z_]+)"/g)].map((m) => m[1]).sort();
    expect(OPCIONES.activacion).toEqual(claves(leer('src/models/factory.py'), 'ACTIVACIONES'));
    expect(OPCIONES['bc.optimizador']).toEqual(claves(leer('src/training/bc.py'), 'OPTIMIZADORES'));
    expect(OPCIONES.recompensa).toEqual(claves(leer('src/training/gail.py'), 'MODOS_RECOMPENSA'));
    expect(leer('src/training/gail.py')).toContain('if dcfg["optimizador"] != "adam":');
    expect(OPCIONES['gail.discriminador.optimizador']).toEqual(['adam']);
    expect(campo('cartpole', 'bc', 'bc', 'optimizador').opciones).toEqual(['adam', 'rmsprop', 'sgd']);
    expect(campo('cartpole', 'gail', 'gail.discriminador', 'optimizador').opciones).toEqual(['adam']);
  });
});
