import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { ErrorApi } from '../api/http.js';
import { crearApiLaboratorioSimulada } from './laboratorio.js';

const dibujar = vi.fn(async () => new Blob(['imagen'], { type: 'image/jpeg' }));

beforeEach(() => {
  vi.useFakeTimers();
  dibujar.mockClear();
});

afterEach(() => {
  vi.useRealTimers();
});

function callbacks() {
  return { alEstado: vi.fn(), alImagen: vi.fn(), alTerminarPartida: vi.fn(),
    alTerminarSesion: vi.fn(), alError: vi.fn() };
}

const ultimoEstado = (c) => c.alEstado.mock.calls.at(-1)?.[0];

async function sesion(api, datos) {
  const creada = await api.crearSesion(datos);
  const c = callbacks();
  return { creada, c, juego: api.conectar(creada.id, c) };
}

describe('configuracion y crearSesion', () => {
  it('da los fps y las partidas del perfil de cada entorno', async () => {
    const api = crearApiLaboratorioSimulada({ dibujar });
    await expect(api.configuracion('cartpole')).resolves.toEqual({ fps: 50, partidas_experto: 15 });
    await expect(api.configuracion('halfcheetah')).resolves.toEqual({ fps: 20, partidas_experto: 30 });
    await expect(api.configuracion('pendulum')).rejects.toMatchObject({ estado: 404 });
  });

  it('crea sesiones con IDs distintos; el experto usa las partidas del perfil por defecto', async () => {
    const api = crearApiLaboratorioSimulada({ dibujar });
    const a = await api.crearSesion({ entorno: 'cartpole', modo: 'manual' });
    const b = await api.crearSesion({ entorno: 'reacher', modo: 'experto' });
    const c = await api.crearSesion({ entorno: 'acrobot', modo: 'experto', partidas: 3 });
    expect(a).toEqual({ id: a.id, entorno: 'cartpole', modo: 'manual', fps: 50, partidas: null });
    expect(b.partidas).toBe(30);
    expect(c.partidas).toBe(3);
    expect(new Set([a.id, b.id, c.id]).size).toBe(3);
  });

  it.each([
    [{ entorno: 'reacher', modo: 'manual' }, 'no se puede jugar a mano'],
    [{ entorno: 'pendulum', modo: 'manual' }, 'Entorno desconocido'],
    [{ entorno: 'cartpole', modo: 'mirar' }, 'Modo desconocido'],
    [{ entorno: 'cartpole', modo: 'experto', partidas: 0 }, 'entre 1 y 1000'],
    [{ entorno: 'cartpole', modo: 'experto', partidas: 1001 }, 'entre 1 y 1000'],
    [{ entorno: 'cartpole', modo: 'experto', partidas: 2.5 }, 'entre 1 y 1000'],
    [{ entorno: 'cartpole', modo: 'experto', partidas: '3' }, 'entre 1 y 1000'],
    [undefined, 'Entorno desconocido'],
  ])('rechaza %j con un 422', async (datos, mensaje) => {
    const error = await crearApiLaboratorioSimulada({ dibujar }).crearSesion(datos).catch((e) => e);
    expect(error).toBeInstanceOf(ErrorApi);
    expect(error.estado).toBe(422);
    expect(error.message).toMatch(mensaje);
  });

  it('conectar a una sesión que no existe es un 404', () => {
    expect(() => crearApiLaboratorioSimulada({ dibujar }).conectar('no', {})).toThrow(ErrorApi);
  });
});

describe('jugar a mano', () => {
  it('CartPole espera sin tecla y avanza a 50 pasos por segundo mientras se pulsa', async () => {
    const api = crearApiLaboratorioSimulada({ dibujar });
    const { c, juego } = await sesion(api, { entorno: 'cartpole', modo: 'manual' });
    await vi.advanceTimersByTimeAsync(200);
    expect(ultimoEstado(c)).toMatchObject({ partida: 1, paso: 0, esperando: true });
    expect(c.alImagen).toHaveBeenCalled();

    juego.enviarAccion(1);
    await vi.advanceTimersByTimeAsync(100);
    expect(ultimoEstado(c)).toMatchObject({ paso: 5, retorno: 5, pares: 5, esperando: false });
    juego.enviarAccion(0);
    await vi.advanceTimersByTimeAsync(100);
    juego.enviarAccion(null);
    await vi.advanceTimersByTimeAsync(200);
    expect(ultimoEstado(c)).toMatchObject({ paso: 10, esperando: true });
  });

  it('al caer el palo empieza otra partida y cuenta sus pares', async () => {
    const api = crearApiLaboratorioSimulada({ dibujar });
    const { c, juego } = await sesion(api, { entorno: 'cartpole', modo: 'manual' });
    juego.enviarAccion(1);
    await vi.advanceTimersByTimeAsync(5000);
    expect(c.alTerminarPartida.mock.calls.length).toBeGreaterThanOrEqual(2);
    const [primera, segunda] = c.alTerminarPartida.mock.calls.map(([p]) => p);
    expect(primera).toMatchObject({ indice: 0, retorno: primera.pasos });
    expect(segunda.indice).toBe(1);
    expect(segunda.semilla).not.toBe(primera.semilla);
    const estado = ultimoEstado(c);
    const terminadas = c.alTerminarPartida.mock.calls.reduce((n, [p]) => n + p.pasos, 0);
    expect(estado.pares).toBe(terminadas + estado.paso);
    expect(estado.partida).toBe(c.alTerminarPartida.mock.calls.length + 1);
  });

  it('MountainCar sin tecla no acelera pero sigue avanzando', async () => {
    const api = crearApiLaboratorioSimulada({ dibujar });
    const { c } = await sesion(api, { entorno: 'mountaincar', modo: 'manual' });
    await vi.advanceTimersByTimeAsync(1000);
    expect(ultimoEstado(c)).toMatchObject({ paso: 30, retorno: -30, esperando: false });
  });

  it('finalizar descarta la partida en curso y devuelve las terminadas', async () => {
    const api = crearApiLaboratorioSimulada({ dibujar });
    const { c, juego } = await sesion(api, { entorno: 'cartpole', modo: 'manual' });
    juego.enviarAccion(1);
    await vi.advanceTimersByTimeAsync(1500);
    const terminadas = c.alTerminarPartida.mock.calls.map(([p]) => p);
    expect(juego.finalizar()).toBe(true);
    expect(c.alTerminarSesion).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(0);
    expect(c.alTerminarSesion).toHaveBeenCalledExactlyOnceWith(terminadas);
    const llamadas = c.alEstado.mock.calls.length;
    await vi.advanceTimersByTimeAsync(1000);
    expect(c.alEstado.mock.calls.length).toBe(llamadas);
    expect(juego.finalizar()).toBe(false);
    expect(juego.enviarAccion(1)).toBe(false);
  });

  it('cerrar corta los avisos sin terminar la sesión', async () => {
    const api = crearApiLaboratorioSimulada({ dibujar });
    const { c, juego } = await sesion(api, { entorno: 'mountaincar', modo: 'manual' });
    await vi.advanceTimersByTimeAsync(100);
    juego.cerrar();
    const llamadas = c.alEstado.mock.calls.length;
    await vi.advanceTimersByTimeAsync(1000);
    expect(c.alEstado.mock.calls.length).toBe(llamadas);
    expect(c.alTerminarSesion).not.toHaveBeenCalled();
  });

  it('un fallo al dibujar llega como error', async () => {
    const api = crearApiLaboratorioSimulada({ dibujar: () => Promise.reject(new Error('sin canvas')) });
    const { c } = await sesion(api, { entorno: 'mountaincar', modo: 'manual' });
    await vi.advanceTimersByTimeAsync(100);
    expect(c.alError.mock.calls[0][0].message).toBe('sin canvas');
  });
});

describe('ver al experto', () => {
  it.each([['acrobot', 2, 120], ['reacher', 3, 50]])(
    'en %s juega %i partidas y termina solo', async (entorno, partidas, pasos) => {
      const api = crearApiLaboratorioSimulada({ dibujar });
      const { c } = await sesion(api, { entorno, modo: 'experto', partidas });
      await vi.advanceTimersByTimeAsync(60_000);
      expect(c.alTerminarSesion).toHaveBeenCalledOnce();
      const [terminadas] = c.alTerminarSesion.mock.calls[0];
      expect(terminadas.map((p) => p.indice)).toEqual([...Array(partidas).keys()]);
      expect(terminadas.every((p) => p.pasos === pasos)).toBe(true);
      expect(new Set(terminadas.map((p) => p.semilla)).size).toBe(partidas);
    });

  it('finalizar justo cuando el experto acaba no pierde sus partidas', async () => {
    const api = crearApiLaboratorioSimulada({ dibujar });
    const { c, juego } = await sesion(api, { entorno: 'reacher', modo: 'experto', partidas: 1 });
    // La última partida ya ha acabado y sesion_terminada sale en el siguiente temporizador.
    vi.advanceTimersByTime(999);
    while (!c.alTerminarPartida.mock.calls.length) {
      vi.advanceTimersToNextTimer();
    }
    expect(c.alTerminarSesion).not.toHaveBeenCalled();
    expect(juego.finalizar()).toBe(true);
    await vi.advanceTimersByTimeAsync(1);
    expect(c.alTerminarSesion).toHaveBeenCalledOnce();
    expect(c.alTerminarSesion.mock.calls[0][0]).toHaveLength(1);
  });

  it('la misma semilla base da las mismas partidas', async () => {
    const resultados = [];
    for (let i = 0; i < 2; i += 1) {
      const api = crearApiLaboratorioSimulada({ dibujar, semilla: 7 });
      const { c } = await sesion(api, { entorno: 'cartpole', modo: 'experto', partidas: 2 });
      await vi.advanceTimersByTimeAsync(60_000);
      resultados.push(c.alTerminarSesion.mock.calls[0][0]);
    }
    expect(resultados[0]).toEqual(resultados[1]);
  });
});

describe('guardar y descartar', () => {
  async function terminada(api) {
    const { creada, c } = await sesion(api, { entorno: 'reacher', modo: 'experto', partidas: 2 });
    await vi.advanceTimersByTimeAsync(10_000);
    expect(c.alTerminarSesion).toHaveBeenCalled();
    return creada.id;
  }

  it('guarda las partidas y los tramos elegidos', async () => {
    const api = crearApiLaboratorioSimulada({ dibujar });
    const id = await terminada(api);
    await expect(api.guardar(id, { nombre: ' reacher ', partidas: [
      { indice: 0, tramo: [0, 50] }, { indice: 1, tramo: [10, 20] }] }))
      .resolves.toEqual({ nombre: 'reacher', partidas: 2, pares: 60 });
    await expect(api.guardar(id, { nombre: 'x', partidas: [{ indice: 0, tramo: [0, 1] }] }))
      .rejects.toMatchObject({ estado: 404 });
  });

  it.each([
    [{ nombre: '', partidas: [{ indice: 0, tramo: [0, 1] }] }, 'nombre'],
    [{ nombre: 'x', partidas: [] }, 'al menos una partida'],
    [{ nombre: 'x' }, 'al menos una partida'],
    [{ nombre: 'x', partidas: [{ indice: 5, tramo: [0, 1] }] }, 'no válido'],
    [{ nombre: 'x', partidas: [{ indice: 0, tramo: [3, 3] }] }, 'no válido'],
    [{ nombre: 'x', partidas: [{ indice: 0, tramo: [-1, 3] }] }, 'no válido'],
    [{ nombre: 'x', partidas: [{ indice: 0, tramo: [0, 51] }] }, 'no válido'],
    [{ nombre: 'x', partidas: [{ indice: 0, tramo: [0.5, 3] }] }, 'no válido'],
    [{ nombre: 'x', partidas: [{ indice: 0 }] }, 'no válido'],
    [{ nombre: 'x', partidas: [{ indice: 0, tramo: [0, 1] }, { indice: 0, tramo: [2, 3] }] },
      'no válido'],
  ])('rechaza guardar %j', async (datos, mensaje) => {
    const api = crearApiLaboratorioSimulada({ dibujar });
    const id = await terminada(api);
    const error = await api.guardar(id, datos).catch((e) => e);
    expect(error.estado).toBe(422);
    expect(error.message).toMatch(mensaje);
  });

  it('no guarda una sesión que no ha terminado', async () => {
    const api = crearApiLaboratorioSimulada({ dibujar });
    const { creada } = await sesion(api, { entorno: 'cartpole', modo: 'manual' });
    await expect(api.guardar(creada.id, { nombre: 'x', partidas: [] }))
      .rejects.toThrow('todavía no ha terminado');
  });

  it('descartar borra la sesión', async () => {
    const api = crearApiLaboratorioSimulada({ dibujar });
    const id = await terminada(api);
    await expect(api.descartar(id)).resolves.toBeUndefined();
    await expect(api.descartar(id)).rejects.toMatchObject({ estado: 404 });
  });
});
