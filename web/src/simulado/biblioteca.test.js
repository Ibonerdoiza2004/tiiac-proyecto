import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { ErrorApi } from '../api/http.js';
import { ENTORNOS } from '../entornos.js';
import { crearAlmacen, partidasCongeladas } from './almacen.js';
import { crearApiBibliotecaSimulada, modelosIniciales } from './biblioteca.js';
import DEMOSTRACIONES from './demostraciones.json';
import { crearApiEntrenamientoSimulada } from './entrenamiento.js';
import { crearApiLaboratorioSimulada } from './laboratorio.js';

const dibujar = async () => new Blob(['imagen'], { type: 'image/jpeg' });

beforeEach(() => {
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
});

describe('partidasCongeladas', () => {
  it.each(ENTORNOS.map((e) => [e.id]))('%s: reparto de configs/, con train barajado y sin huecos', (entorno) => {
    const partidas = partidasCongeladas(entorno);
    const { partidas_train: train, partidas_validacion: validacion } = DEMOSTRACIONES[entorno];
    expect(partidas.map((p) => p.id_episodio)).toEqual([...Array(train + validacion).keys()]);
    const deTrain = partidas.filter((p) => p.split === 'train');
    expect(deTrain).toHaveLength(train);
    expect(partidas.filter((p) => p.split === 'validacion')).toHaveLength(validacion);
    expect(deTrain.map((p) => p.orden_train).sort((a, b) => a - b)).toEqual([...Array(train).keys()]);
    expect(partidas.filter((p) => p.split !== 'train').every((p) => p.orden_train === null)).toBe(true);
    // Barajado: train no son las primeras en orden.
    expect(deTrain.map((p) => p.id_episodio)).not.toEqual([...Array(train).keys()]);
    for (const p of partidas) {
      expect(Number.isInteger(p.pasos) && p.pasos > 0).toBe(true);
      expect(Number.isFinite(p.retorno)).toBe(true);
    }
  });

  it('es siempre igual y cuadra con los pasos de cada entorno', () => {
    expect(partidasCongeladas('acrobot')).toEqual(partidasCongeladas('acrobot'));
    expect(partidasCongeladas('cartpole').every((p) => p.pasos === 500 && p.retorno === 500)).toBe(true);
    expect(partidasCongeladas('reacher').every((p) => p.pasos === 50 && p.retorno <= -3.5)).toBe(true);
    expect(partidasCongeladas('mountaincar').every((p) => p.retorno === -p.pasos)).toBe(true);
    expect(partidasCongeladas('halfcheetah').every((p) => p.retorno <= 5000 && p.retorno >= 4500)).toBe(true);
  });

  it('el almacén cuenta partidas y pares de ellas', () => {
    for (const d of crearAlmacen().demostraciones) {
      expect(d.partidas).toBe(d.detallePartidas.length);
      expect(d.pares).toBe(d.detallePartidas.reduce((suma, p) => suma + p.pasos, 0));
    }
  });
});

describe('listas y detalle', () => {
  it.each(ENTORNOS.map((e) => [e.id]))('%s: modelos y demostraciones con la forma de la API', async (entorno) => {
    const api = crearApiBibliotecaSimulada();
    expect(await api.modelos(entorno)).toEqual(modelosIniciales(entorno));
    expect(await api.demostraciones(entorno)).toEqual([{ id: `congelado-${entorno}`, entorno,
      nombre: expect.stringContaining('experto congelado'), tipo: 'conjunto', partidas: expect.any(Number),
      pares: expect.any(Number), fuente: 'experto', origen: 'congelado', valido: true, motivo: null }]);
  });

  it('rechaza un entorno desconocido', async () => {
    const api = crearApiBibliotecaSimulada();
    await expect(api.modelos('pong')).rejects.toMatchObject({ estado: 422 });
    await expect(api.demostraciones('pong')).rejects.toMatchObject({ estado: 422 });
  });

  it('da el detalle de modelos y de demostraciones, y 404 si no existe', async () => {
    const api = crearApiBibliotecaSimulada();
    expect(await api.elemento('gail-acrobot-10')).toEqual(modelosIniciales('acrobot')[3]);
    expect(await api.elemento('congelado-cartpole')).toEqual({ id: 'congelado-cartpole', entorno: 'cartpole',
      tipo: 'demostraciones', nombre: 'CartPole · experto congelado', origen: 'congelado',
      creado: '2026-11-01T00:00:00Z', descripcion: null, fuente: 'experto', artefacto: null, valido: true,
      motivo: null, detalles: { tipo: 'conjunto', partidas: 15, pares: 7500, fuente: 'experto', splits: true } });
    const error = await api.elemento('nada').catch((e) => e);
    expect(error).toBeInstanceOf(ErrorApi);
    expect(error.estado).toBe(404);
  });

  it('da las partidas de un conjunto; de un modelo, 422, y de lo que no existe, 404', async () => {
    const api = crearApiBibliotecaSimulada();
    expect(await api.partidas('congelado-reacher')).toEqual(partidasCongeladas('reacher'));
    await expect(api.partidas('experto-reacher')).rejects.toMatchObject({ estado: 422 });
    await expect(api.partidas('nada')).rejects.toMatchObject({ estado: 404 });
  });

  it('devuelve copias: cambiar lo recibido no cambia la biblioteca', async () => {
    const api = crearApiBibliotecaSimulada();
    (await api.partidas('congelado-cartpole'))[0].pasos = 1;
    (await api.demostraciones('cartpole'))[0].nombre = 'x';
    expect((await api.partidas('congelado-cartpole'))[0].pasos).toBe(500);
    expect((await api.demostraciones('cartpole'))[0].nombre).not.toBe('x');
  });
});

describe('crearSubconjunto', () => {
  const peticion = { conjunto: 'congelado-acrobot', partidas: [4, 0, 2], nombre: 'Tres' };

  it('guarda las partidas elegidas, enteras, y aparece en la lista y para entrenar', async () => {
    const almacen = crearAlmacen();
    const api = crearApiBibliotecaSimulada({ almacen });
    const nuevo = await api.crearSubconjunto(peticion);
    const pasos = partidasCongeladas('acrobot');
    const pares = pasos[0].pasos + pasos[2].pasos + pasos[4].pasos;
    expect(nuevo).toMatchObject({ entorno: 'acrobot', tipo: 'demostraciones', nombre: 'Tres', origen: 'dashboard',
      fuente: 'experto', valido: true, detalles: { tipo: 'subconjunto', partidas: 3, pares, splits: false,
        conjunto: 'congelado-acrobot' } });
    expect(await api.elemento(nuevo.id)).toEqual(nuevo);
    // En orden de id y sin reparto: se reparten al entrenar.
    expect(await api.partidas(nuevo.id)).toEqual([0, 2, 4].map((i) => ({ ...pasos[i], split: null,
      orden_train: null })));
    expect((await api.demostraciones('acrobot')).map((d) => d.tipo)).toEqual(['conjunto', 'subconjunto']);
    const entrenar = crearApiEntrenamientoSimulada({ almacen, dibujar });
    expect((await entrenar.demostraciones('acrobot'))[1]).toEqual({ id: nuevo.id, nombre: 'Tres',
      tipo: 'subconjunto', partidas: 3, pares, fuente: 'experto' });
  });

  it.each([
    ['conjunto que no existe', { ...peticion, conjunto: 'nada' }, /No existe el conjunto nada/],
    ['sin nombre', { ...peticion, nombre: '' }, /nombre/],
    ['nombre en blanco', { ...peticion, nombre: '   ' }, /nombre/],
    ['nombre largo', { ...peticion, nombre: 'x'.repeat(101) }, /nombre/],
    ['nombre no texto', { ...peticion, nombre: 3 }, /nombre/],
    ['sin partidas', { ...peticion, partidas: [] }, /al menos una partida/],
    ['partidas no lista', { ...peticion, partidas: 3 }, /al menos una partida/],
    ['repetidas', { ...peticion, partidas: [1, 1] }, /repetidas/],
    ['de fuera', { ...peticion, partidas: [0, 15] }, /no son del conjunto: 15/],
    ['negativa', { ...peticion, partidas: [-1] }, /no son del conjunto: -1/],
    ['decimal', { ...peticion, partidas: [0.5] }, /no son del conjunto: 0.5/],
  ])('rechaza con 422: %s', async (_, datos, mensaje) => {
    const api = crearApiBibliotecaSimulada();
    const error = await api.crearSubconjunto(datos).catch((e) => e);
    expect(error.estado).toBe(422);
    expect(error.message).toMatch(mensaje);
    expect(await api.demostraciones('acrobot')).toHaveLength(1);
  });

  it('acepta 100 caracteres y una sola partida', async () => {
    const api = crearApiBibliotecaSimulada();
    expect((await api.crearSubconjunto({ ...peticion, nombre: 'x'.repeat(100), partidas: [14] })).detalles.partidas)
      .toBe(1);
  });

  it('no saca un subconjunto de otro subconjunto', async () => {
    const api = crearApiBibliotecaSimulada();
    const { id } = await api.crearSubconjunto(peticion);
    const error = await api.crearSubconjunto({ ...peticion, conjunto: id }).catch((e) => e);
    expect(error.estado).toBe(422);
    expect(error.message).toMatch('no es un conjunto');
  });

  it('lo guardado al jugar es un conjunto sin reparto con sus partidas', async () => {
    const almacen = crearAlmacen();
    const laboratorio = crearApiLaboratorioSimulada({ almacen, dibujar });
    const sesion = await laboratorio.crearSesion({ entorno: 'reacher', modo: 'experto', partidas: 2 });
    laboratorio.conectar(sesion.id, {});
    await vi.advanceTimersByTimeAsync(10_000);
    await laboratorio.guardar(sesion.id, { nombre: 'mías', partidas: [{ indice: 1, tramo: [0, 50] },
      { indice: 0, tramo: [10, 30] }] });
    const api = crearApiBibliotecaSimulada({ almacen });
    const [, mias] = await api.demostraciones('reacher');
    expect(mias).toMatchObject({ nombre: 'mías', tipo: 'conjunto', partidas: 2, pares: 70, origen: 'dashboard' });
    const partidas = await api.partidas(mias.id);
    expect(partidas.map((p) => [p.id_episodio, p.pasos, p.split, p.orden_train])).toEqual([[0, 50, null, null],
      [1, 20, null, null]]);
    expect((await api.elemento(mias.id)).detalles.splits).toBe(false);
    expect((await api.crearSubconjunto({ conjunto: mias.id, partidas: [1], nombre: 'una' })).detalles.pares).toBe(20);
  });
});
