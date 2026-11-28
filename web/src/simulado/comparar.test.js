import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { ALEATORIO } from '../api/comparar.js';
import { ErrorApi } from '../api/http.js';
import { ENTORNOS } from '../entornos.js';
import { crearAlmacen } from './almacen.js';
import { modelosIniciales } from './biblioteca.js';
import { crearApiCompararSimulada } from './comparar.js';
import EVALUACION from './evaluacion.json';
import { crearServidorTrabajosSimulado } from './trabajos.js';

const INTERVALO = 50;
const dibujar = vi.fn(async () => new Blob(['imagen'], { type: 'image/jpeg' }));

beforeEach(() => {
  vi.useFakeTimers();
  dibujar.mockClear();
});

afterEach(() => {
  vi.useRealTimers();
});

function crear(almacen = crearAlmacen()) {
  const servidor = crearServidorTrabajosSimulado({ intervalo: INTERVALO });
  return { almacen, servidor, api: crearApiCompararSimulada({ almacen, servidor, dibujar }) };
}

async function seguirHastaElFinal(api, id, pasos) {
  const c = { alEstado: vi.fn(), alProgreso: vi.fn(), alImagen: vi.fn(), alHistorial: vi.fn() };
  const conexion = api.seguir(id, c);
  await vi.advanceTimersByTimeAsync(INTERVALO * pasos);
  return { c, conexion, final: c.alEstado.mock.calls.at(-1)?.[0] };
}

describe('biblioteca simulada', () => {
  it.each(ENTORNOS.map((e) => [e.id]))('%s tiene experto, GAIL y BC válidos con su historial', async (entorno) => {
    const { api } = crear();
    const modelos = await api.modelos(entorno);
    expect(modelos).toEqual(modelosIniciales(entorno));
    expect(modelos.map((m) => m.tipo)).toEqual(['experto', 'gail', 'bc', 'gail', 'bc']);
    expect(modelos.every((m) => m.valido && m.entorno === entorno)).toBe(true);
    expect(new Set(modelos.map((m) => m.id)).size).toBe(modelos.length);
    // Historial con la duración del perfil de cada método (configs/).
    expect(modelos[0].detalles.pasos).toBeGreaterThan(0);
    expect(modelos[1].detalles.iteraciones).toBeGreaterThan(0);
    expect(modelos[2].detalles.epocas).toBeGreaterThan(0);
  });

  it('añade lo entrenado en el Laboratorio simulado de ese entorno', async () => {
    const almacen = crearAlmacen();
    almacen.modelos.push({ id: 'm1', nombre: 'Mi GAIL', entorno: 'cartpole', metodo: 'gail', incompleto: true,
      resumen: { unidad: 'iteraciones', paso: 4 }, creado: '2026-11-25T10:00:00Z' });
    almacen.modelos.push({ id: 'm2', nombre: 'Otro', entorno: 'acrobot', metodo: 'bc', incompleto: false,
      resumen: { unidad: 'epocas', paso: 2 } });
    const { api } = crear(almacen);
    const propios = (await api.modelos('cartpole')).filter((m) => m.origen === 'dashboard');
    expect(propios).toEqual([{ id: 'm1', entorno: 'cartpole', tipo: 'gail', nombre: 'Mi GAIL',
      origen: 'dashboard', creado: '2026-11-25T10:00:00Z', descripcion: null, fuente: null, artefacto: null,
      valido: true, motivo: null,
      detalles: { iteraciones: 4, incompleto: true } }]);
  });

  it('rechaza un entorno desconocido', async () => {
    const { api } = crear();
    await expect(api.modelos('pong')).rejects.toMatchObject({ estado: 422 });
  });

  it('devuelve copias: cambiar la lista no cambia la biblioteca', async () => {
    const { api } = crear();
    (await api.modelos('cartpole'))[0].nombre = 'x';
    expect((await api.modelos('cartpole'))[0].nombre).not.toBe('x');
  });
});

describe('crear comparación simulada', () => {
  const base = { entorno: 'cartpole', modelos: ['experto-cartpole', 'bc-cartpole-1'], aleatorio: true, vista: false };

  it.each([
    ['entorno desconocido', { ...base, entorno: 'pong' }, /Entorno desconocido/],
    ['modelos no lista', { ...base, modelos: 'x' }, /lista de ids/],
    ['modelos repetidos', { ...base, modelos: ['bc-cartpole-1', 'bc-cartpole-1'] }, /sin repetir/],
    ['aleatorio no booleano', { ...base, aleatorio: 1 }, /booleanos/],
    ['vista no booleana', { ...base, vista: 'si' }, /booleanos/],
    ['un solo participante', { ...base, modelos: ['bc-cartpole-1'], aleatorio: false }, /al menos dos/],
    ['ninguno', { ...base, modelos: [], aleatorio: false }, /al menos dos/],
    ['partidas 0', { ...base, partidas: 0 }, /entre 1 y 1000/],
    ['partidas 1001', { ...base, partidas: 1001 }, /entre 1 y 1000/],
    ['partidas decimales', { ...base, partidas: 2.5 }, /entre 1 y 1000/],
    ['partidas texto', { ...base, partidas: '5' }, /entre 1 y 1000/],
    ['modelo de otro entorno', { ...base, modelos: ['experto-acrobot'] }, /no existen.*experto-acrobot/],
    ['modelo inexistente', { ...base, modelos: ['nada', 'bc-cartpole-1'] }, /no existen.*nada/],
  ])('rechaza con 422: %s', async (_, peticion, mensaje) => {
    const { api } = crear();
    const error = await api.crear(peticion).catch((e) => e);
    expect(error).toBeInstanceOf(ErrorApi);
    expect(error.estado).toBe(422);
    expect(error.message).toMatch(mensaje);
  });

  it.each([[1], [1000]])('acepta %i partidas', async (partidas) => {
    const { api } = crear();
    expect((await api.crear({ ...base, partidas })).argumentos.partidas).toBe(partidas);
  });

  it('sin partidas usa las de la evaluación de configs/ y su semilla', async () => {
    const { api } = crear();
    const trabajo = await api.crear(base);
    expect(trabajo).toMatchObject({ tipo: 'comparacion', estado: 'en_marcha', argumentos: { ...base,
      partidas: EVALUACION.cartpole.episodios, semilla: EVALUACION.cartpole.semilla } });
    expect(await api.comparaciones('cartpole')).toHaveLength(1);
    expect(await api.comparaciones('acrobot')).toEqual([]);
  });

  it('juega ronda a ronda: todos los modelos la partida i antes de la i + 1', async () => {
    const { api } = crear();
    const { id } = await api.crear({ ...base, partidas: 3 });
    const { c, final } = await seguirHastaElFinal(api, id, 5);
    const orden = c.alProgreso.mock.calls.map(([m]) => [m.partida, m.modelo]);
    expect(orden).toEqual([0, 1, 2].flatMap((p) => [[p, 'experto-cartpole'], [p, 'bc-cartpole-1'], [p, ALEATORIO]]));
    for (const [m] of c.alProgreso.mock.calls) {
      expect(Number.isFinite(m.retorno)).toBe(true);
      expect(Number.isInteger(m.pasos) && m.pasos >= 1).toBe(true);
    }
    expect(final).toMatchObject({ estado: 'terminado', resultado: { partidas: 3, semilla: EVALUACION.cartpole.semilla,
      incompleta: false } });
    expect(final.resultado.modelos.map((m) => [m.id, m.retornos.length]))
      .toEqual([['experto-cartpole', 3], ['bc-cartpole-1', 3], [ALEATORIO, 3]]);
  });

  it('nadie supera el objetivo del perfil', async () => {
    const { api } = crear();
    const { id } = await api.crear({ ...base, partidas: 30 });
    const { final } = await seguirHastaElFinal(api, id, 35);
    expect(Math.max(...final.resultado.modelos.flatMap((m) => m.retornos))).toBeLessThanOrEqual(500);
  });

  it('el experto puntúa más que BC con 1 partida, y este más que el aleatorio', async () => {
    const { api } = crear();
    const { id } = await api.crear({ ...base, partidas: 20 });
    const { final } = await seguirHastaElFinal(api, id, 25);
    const media = (i) => final.resultado.modelos[i].retornos.reduce((a, b) => a + b, 0) / 20;
    expect(media(0)).toBeGreaterThan(media(1));
    expect(media(1)).toBeGreaterThan(media(2));
  });

  it('las mismas partidas dan los mismos retornos en dos comparaciones', async () => {
    const { api } = crear();
    const a = await api.crear({ ...base, partidas: 2 });
    await vi.advanceTimersByTimeAsync(INTERVALO * 3);
    const b = await api.crear({ ...base, partidas: 2 });
    await vi.advanceTimersByTimeAsync(INTERVALO * 3);
    const [ra, rb] = [a, b].map(({ id }) => api.comparaciones('cartpole')
      .then((lista) => lista.find((t) => t.id === id).resultado));
    expect(await ra).not.toBeNull();
    expect(await rb).toEqual(await ra);
  });

  it('finalizar termina con lo jugado y marca incompleta', async () => {
    const { api } = crear();
    const { id } = await api.crear({ ...base, partidas: 50 });
    const { c, conexion } = await seguirHastaElFinal(api, id, 2);
    conexion.ordenar({ tipo: 'finalizar' });
    const final = c.alEstado.mock.calls.at(-1)[0];
    expect(final.resultado).toMatchObject({ partidas: 2, incompleta: true });
    expect(final.resultado.modelos.every((m) => m.retornos.length === 2)).toBe(true);
  });

  it('dibuja solo con la vista activa, y la orden vista la cambia en marcha', async () => {
    const { api } = crear();
    const { id } = await api.crear({ ...base, partidas: 50, vista: true });
    const { c, conexion } = await seguirHastaElFinal(api, id, 2);
    expect(dibujar).toHaveBeenCalledTimes(2);
    expect(c.alImagen).toHaveBeenCalled();
    conexion.ordenar({ tipo: 'vista', activa: false });
    await vi.advanceTimersByTimeAsync(INTERVALO * 3);
    expect(dibujar).toHaveBeenCalledTimes(2);
    conexion.ordenar({ tipo: 'vista', activa: true });
    await vi.advanceTimersByTimeAsync(INTERVALO);
    expect(dibujar).toHaveBeenCalledTimes(3);
    expect(dibujar.mock.calls[0][0]).toBe('cartpole');
  });

  it.each(ENTORNOS.map((e) => [e.id]))('%s compara y dibuja sin errores', async (entorno) => {
    const { api } = crear();
    const ids = modelosIniciales(entorno).map((m) => m.id);
    const { id } = await api.crear({ entorno, modelos: ids, aleatorio: true, partidas: 2, vista: true });
    const { final } = await seguirHastaElFinal(api, id, 3);
    expect(final.estado).toBe('terminado');
    expect(final.resultado.modelos).toHaveLength(ids.length + 1);
  });
});
