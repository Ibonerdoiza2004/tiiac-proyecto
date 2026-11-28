import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { ErrorApi } from '../api/http.js';
import { crearAlmacen } from './almacen.js';
import { ACTUALIZACIONES, INTERVALO_SIMULADO_MS, crearApiEntrenamientoSimulada } from './entrenamiento.js';
import { crearApiLaboratorioSimulada } from './laboratorio.js';

const dibujar = vi.fn(async () => new Blob(['imagen'], { type: 'image/jpeg' }));
const DATOS = ['congelado-cartpole'];

beforeEach(() => {
  vi.useFakeTimers();
  dibujar.mockClear();
});

afterEach(() => {
  vi.useRealTimers();
});

function callbacks() {
  return { alProgreso: vi.fn(), alImagen: vi.fn(), alEstado: vi.fn(), alTerminar: vi.fn(), alError: vi.fn() };
}

async function entrenar(api, peticion) {
  const creado = await api.crear(peticion);
  const c = callbacks();
  return { creado, c, entreno: api.conectar(creado.id, c) };
}

const hastaElFinal = () => vi.advanceTimersByTimeAsync(INTERVALO_SIMULADO_MS * (ACTUALIZACIONES + 5));

describe('ajustes y demostraciones', () => {
  it('ofrece los hiperparámetros del perfil y copias que no se pueden estropear', async () => {
    const api = crearApiEntrenamientoSimulada({ dibujar });
    const esquema = await api.ajustes('cartpole', 'gail');
    expect(esquema.duracion).toEqual({ unidad: 'iteraciones', total: 20, campo: 'gail.iteraciones' });
    esquema.secciones.length = 0;
    expect((await api.ajustes('cartpole', 'gail')).secciones.length).toBeGreaterThan(0);
    await expect(api.ajustes('pendulum', 'gail')).rejects.toMatchObject({ estado: 404 });
    await expect(api.ajustes('cartpole', 'ppo')).rejects.toMatchObject({ estado: 422 });
  });

  it('lista las demostraciones del entorno, también las que se guardan al jugar', async () => {
    const almacen = crearAlmacen();
    const api = crearApiEntrenamientoSimulada({ almacen, dibujar });
    expect(await api.demostraciones('cartpole')).toEqual([{ id: 'congelado-cartpole',
      nombre: 'CartPole · experto congelado', tipo: 'conjunto', partidas: 15, pares: 7500, fuente: 'experto' }]);
    // Las partidas de MountainCar duran distinto: los pares son la suma de las inventadas.
    const [montana] = await api.demostraciones('mountaincar');
    expect(montana).toEqual({ id: 'congelado-mountaincar', nombre: 'MountainCar · experto congelado',
      tipo: 'conjunto', partidas: 15, pares: expect.any(Number), fuente: 'experto' });
    expect(montana.pares).toBeGreaterThan(15);

    const laboratorio = crearApiLaboratorioSimulada({ almacen, dibujar });
    const sesion = await laboratorio.crearSesion({ entorno: 'reacher', modo: 'experto', partidas: 1 });
    laboratorio.conectar(sesion.id, {});
    await vi.advanceTimersByTimeAsync(5000);
    await laboratorio.guardar(sesion.id, { nombre: 'mias', partidas: [{ indice: 0, tramo: [0, 50] }] });
    expect((await api.demostraciones('reacher')).map((d) => [d.nombre, d.fuente, d.pares]))
      .toEqual([['Reacher · experto congelado', 'experto', 1500], ['mias', 'experto', 50]]);
    expect(await api.demostraciones('cartpole')).toHaveLength(1);
    await expect(api.demostraciones('pendulum')).rejects.toMatchObject({ estado: 404 });
  });
});

describe('crear', () => {
  it('devuelve la duración con los ajustes pedidos', async () => {
    const api = crearApiEntrenamientoSimulada({ dibujar });
    expect(await api.crear({ entorno: 'cartpole', metodo: 'gail', datos: DATOS,
      ajustes: { gail: { iteraciones: 7 } } })).toMatchObject({ metodo: 'gail',
      duracion: { unidad: 'iteraciones', total: 7 } });
    expect((await api.crear({ entorno: 'cartpole', metodo: 'experto' })).duracion)
      .toEqual({ unidad: 'pasos', total: 100000 });
    // Como el real: en el experto, bloques completos de pasos_por_iteracion.
    expect((await api.crear({ entorno: 'cartpole', metodo: 'experto',
      ajustes: { experto: { pasos_totales: 12001, trpo: { pasos_por_iteracion: 1000 } } } })).duracion)
      .toEqual({ unidad: 'pasos', total: 13000 });
    expect((await api.crear({ entorno: 'cartpole', metodo: 'bc', datos: DATOS })).duracion)
      .toEqual({ unidad: 'epocas', total: 500 });
  });

  it.each([
    [{ entorno: 'pendulum', metodo: 'bc', datos: DATOS }, 'Entorno desconocido'],
    [{ entorno: 'cartpole', metodo: 'ppo' }, 'Método desconocido'],
    [{ entorno: 'cartpole', metodo: 'bc', datos: [] }, 'al menos un conjunto'],
    [{ entorno: 'cartpole', metodo: 'gail' }, 'al menos un conjunto'],
    [{ entorno: 'cartpole', metodo: 'bc', datos: ['no-existe'] }, 'no son de cartpole'],
    [{ entorno: 'cartpole', metodo: 'bc', datos: ['congelado-reacher'] }, 'no son de cartpole'],
    [{ entorno: 'cartpole', metodo: 'bc', datos: [...DATOS, ...DATOS] }, 'sin repetir'],
    [{ entorno: 'cartpole', metodo: 'bc', datos: 'congelado-cartpole' }, 'sin repetir'],
    [{ entorno: 'cartpole', metodo: 'experto', datos: DATOS }, 'no usa demostraciones'],
    [{ entorno: 'cartpole', metodo: 'bc', datos: DATOS, vista: true }, 'no tiene vista'],
    [{ entorno: 'cartpole', metodo: 'gail', datos: DATOS, vista: 'sí' }, 'booleanos'],
    [{ entorno: 'cartpole', metodo: 'gail', datos: DATOS, wandb: 1 }, 'booleanos'],
    [{ entorno: 'cartpole', metodo: 'gail', datos: DATOS, ajustes: { gail: { pasos: 3 } } }, 'desconocido'],
    [{ entorno: 'cartpole', metodo: 'gail', datos: DATOS, ajustes: { bc: { batch_size: 3 } } }, 'desconocido'],
    [{ entorno: 'cartpole', metodo: 'gail', datos: DATOS, ajustes: { gail: { iteraciones: 0 } } }, 'no válido'],
    [{ entorno: 'cartpole', metodo: 'gail', datos: DATOS, ajustes: { gail: { iteraciones: 2.5 } } }, 'no válido'],
    [{ entorno: 'cartpole', metodo: 'gail', datos: DATOS, ajustes: { gail: { iteraciones: '20' } } }, 'no válido'],
    [{ entorno: 'cartpole', metodo: 'gail', datos: DATOS, ajustes: { trpo: { activacion: 'sigmoid' } } }, 'no válido'],
    [{ entorno: 'cartpole', metodo: 'gail', datos: DATOS, ajustes: { trpo: { capas_ocultas: [0] } } }, 'no válido'],
    [{ entorno: 'cartpole', metodo: 'gail', datos: DATOS, ajustes: { trpo: { normalizar_ventajas: 'no' } } }, 'no válido'],
    [{ entorno: 'cartpole', metodo: 'gail', datos: DATOS, ajustes: [] }, 'anidado'],
  ])('rechaza %j con un 422', async (peticion, mensaje) => {
    const error = await crearApiEntrenamientoSimulada({ dibujar }).crear(peticion).catch((e) => e);
    expect(error).toBeInstanceOf(ErrorApi);
    expect(error.estado).toBe(422);
    expect(error.message).toMatch(mensaje);
  });

  it('conectar a un entrenamiento que no existe es un 404', () => {
    expect(() => crearApiEntrenamientoSimulada({ dibujar }).conectar('x', {})).toThrow(ErrorApi);
  });
});

describe('entrenar', () => {
  it.each([
    ['gail', { datos: DATOS }, 'iteraciones', 'recompensa_gail_media'],
    ['bc', { datos: DATOS }, 'epocas', 'perdida_validacion'],
    ['experto', {}, 'pasos', 'retorno_medio'],
  ])('%s avanza, envía sus medidas y guarda el modelo al acabar', async (metodo, extra, unidad, medida) => {
    const almacen = crearAlmacen();
    const api = crearApiEntrenamientoSimulada({ almacen, dibujar });
    const { creado, c } = await entrenar(api, { entorno: 'cartpole', metodo, ...extra });
    await vi.advanceTimersByTimeAsync(0);
    expect(c.alEstado).toHaveBeenCalledWith('en_marcha');
    await hastaElFinal();
    const pasos = c.alProgreso.mock.calls.map(([p]) => p.paso);
    expect(pasos.length).toBeGreaterThan(1);
    expect(pasos.every((p, i) => i === 0 || p > pasos[i - 1])).toBe(true);
    expect(pasos.at(-1)).toBeLessThanOrEqual(creado.duracion.total);
    expect(c.alProgreso.mock.calls[0][0].medidas).toHaveProperty(medida);
    expect(c.alTerminar).toHaveBeenCalledOnce();
    const [{ modelo, resumen }] = c.alTerminar.mock.calls[0];
    expect(modelo).toEqual({ nombre: `${metodo}_cartpole_1`, incompleto: false });
    expect(resumen).toMatchObject({ unidad, paso: pasos.at(-1), total: creado.duracion.total });
    expect(almacen.modelos).toHaveLength(1);
    expect(c.alImagen).not.toHaveBeenCalled();
  });

  it('GAIL valida en cada evaluar_cada aunque una actualización salte varias iteraciones', async () => {
    const api = crearApiEntrenamientoSimulada({ dibujar });
    const { c } = await entrenar(api, { entorno: 'halfcheetah', metodo: 'gail', datos: ['congelado-halfcheetah'] });
    await hastaElFinal();
    const pasos = c.alProgreso.mock.calls.map(([p]) => p.paso);
    const con = c.alProgreso.mock.calls.filter(([p]) => 'retorno_validacion' in p.medidas).length;
    expect(pasos.at(-1)).toBe(500);
    expect(con).toBeGreaterThan(pasos.length / 2);
  });

  it('el experto para al llegar al objetivo, también si es negativo', async () => {
    for (const entorno of ['cartpole', 'mountaincar']) {
      const api = crearApiEntrenamientoSimulada({ dibujar });
      const { creado, c } = await entrenar(api, { entorno, metodo: 'experto' });
      await hastaElFinal();
      const [{ resumen }] = c.alTerminar.mock.calls[0];
      expect(resumen.motivo).toBe('objetivo alcanzado');
      expect(resumen.paso).toBeLessThan(creado.duracion.total);
    }
  });

  it('pausar para el progreso y reanudar lo sigue, avisando del estado', async () => {
    const api = crearApiEntrenamientoSimulada({ dibujar });
    const { c, entreno } = await entrenar(api, { entorno: 'cartpole', metodo: 'gail', datos: DATOS });
    await vi.advanceTimersByTimeAsync(INTERVALO_SIMULADO_MS * 3);
    expect(entreno.pausar()).toBe(true);
    await vi.advanceTimersByTimeAsync(0);
    expect(c.alEstado).toHaveBeenLastCalledWith('pausado');
    const antes = c.alProgreso.mock.calls.length;
    await vi.advanceTimersByTimeAsync(INTERVALO_SIMULADO_MS * 10);
    expect(c.alProgreso.mock.calls.length).toBe(antes);
    expect(entreno.pausar()).toBe(true);
    await vi.advanceTimersByTimeAsync(0);
    expect(c.alEstado).toHaveBeenCalledTimes(2);
    expect(entreno.reanudar()).toBe(true);
    await vi.advanceTimersByTimeAsync(INTERVALO_SIMULADO_MS * 2);
    expect(c.alEstado).toHaveBeenLastCalledWith('en_marcha');
    expect(c.alProgreso.mock.calls.length).toBe(antes + 2);
  });

  it('finalizar guarda el modelo incompleto y después nada más llega', async () => {
    const almacen = crearAlmacen();
    const api = crearApiEntrenamientoSimulada({ almacen, dibujar });
    const { c, entreno } = await entrenar(api, { entorno: 'cartpole', metodo: 'gail', datos: DATOS });
    await vi.advanceTimersByTimeAsync(INTERVALO_SIMULADO_MS * 3);
    expect(entreno.finalizar()).toBe(true);
    expect(c.alTerminar).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(0);
    expect(c.alTerminar.mock.calls[0][0].modelo).toEqual({ nombre: 'gail_cartpole_1', incompleto: true });
    expect(c.alTerminar.mock.calls[0][0].resumen.motivo).toBe('finalizado por el usuario');
    const progresos = c.alProgreso.mock.calls.length;
    await vi.advanceTimersByTimeAsync(INTERVALO_SIMULADO_MS * 10);
    expect(c.alProgreso.mock.calls.length).toBe(progresos);
    for (const orden of ['pausar', 'reanudar', 'finalizar']) {
      expect(entreno[orden]()).toBe(false);
    }
    expect(entreno.cambiarVista(true)).toBe(false);
    expect(almacen.modelos).toHaveLength(1);
  });

  it('finalizar justo al acabar no pierde el resultado ni lo duplica', async () => {
    const api = crearApiEntrenamientoSimulada({ dibujar });
    const { c, entreno } = await entrenar(api, { entorno: 'cartpole', metodo: 'gail', datos: DATOS,
      ajustes: { gail: { iteraciones: 1 } } });
    vi.advanceTimersByTime(INTERVALO_SIMULADO_MS);
    expect(c.alTerminar).not.toHaveBeenCalled();
    expect(entreno.finalizar()).toBe(true);
    await vi.advanceTimersByTimeAsync(1);
    expect(c.alTerminar).toHaveBeenCalledOnce();
    expect(c.alTerminar.mock.calls[0][0].modelo.incompleto).toBe(false);
  });

  it('la vista manda imágenes solo mientras está encendida', async () => {
    const api = crearApiEntrenamientoSimulada({ dibujar });
    const { c, entreno } = await entrenar(api, { entorno: 'cartpole', metodo: 'gail', datos: DATOS, vista: true });
    await vi.advanceTimersByTimeAsync(INTERVALO_SIMULADO_MS * 3);
    expect(c.alImagen).toHaveBeenCalledTimes(3);
    entreno.cambiarVista(false);
    await vi.advanceTimersByTimeAsync(INTERVALO_SIMULADO_MS * 3);
    expect(c.alImagen).toHaveBeenCalledTimes(3);
    entreno.cambiarVista(true);
    await vi.advanceTimersByTimeAsync(INTERVALO_SIMULADO_MS);
    expect(c.alImagen).toHaveBeenCalledTimes(4);
  });

  it('cerrar corta los avisos sin terminar el entrenamiento', async () => {
    const almacen = crearAlmacen();
    const api = crearApiEntrenamientoSimulada({ almacen, dibujar });
    const { c, entreno } = await entrenar(api, { entorno: 'cartpole', metodo: 'gail', datos: DATOS });
    await vi.advanceTimersByTimeAsync(INTERVALO_SIMULADO_MS);
    entreno.cerrar();
    const progresos = c.alProgreso.mock.calls.length;
    await hastaElFinal();
    expect(c.alProgreso.mock.calls.length).toBe(progresos);
    expect(c.alTerminar).not.toHaveBeenCalled();
    expect(almacen.modelos).toHaveLength(0);
  });

  it('cada entrenamiento se conecta una sola vez', async () => {
    const api = crearApiEntrenamientoSimulada({ dibujar });
    const { creado } = await entrenar(api, { entorno: 'cartpole', metodo: 'experto' });
    expect(() => api.conectar(creado.id, {})).toThrow(ErrorApi);
  });

  it('la misma semilla da las mismas curvas', async () => {
    const curvas = [];
    for (let i = 0; i < 2; i += 1) {
      const api = crearApiEntrenamientoSimulada({ dibujar, semilla: 3 });
      const { c } = await entrenar(api, { entorno: 'cartpole', metodo: 'bc', datos: DATOS });
      await hastaElFinal();
      curvas.push(c.alProgreso.mock.calls);
    }
    expect(curvas[0]).toEqual(curvas[1]);
  });

  it('un fallo al dibujar llega como error', async () => {
    const api = crearApiEntrenamientoSimulada({ dibujar: () => Promise.reject(new Error('sin canvas')) });
    const { c } = await entrenar(api, { entorno: 'cartpole', metodo: 'experto', vista: true });
    await vi.advanceTimersByTimeAsync(INTERVALO_SIMULADO_MS);
    expect(c.alError.mock.calls[0][0].message).toBe('sin canvas');
  });
});
