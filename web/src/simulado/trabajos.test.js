import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { ESTADOS_CONEXION } from '../api/websocket.js';
import { crearServidorTrabajosSimulado } from './trabajos.js';

const INTERVALO = 100;

beforeEach(() => {
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
});

/** Tarea de juguete: un progreso por paso y termina en el paso `pasos`. */
function contador(pasos = 3) {
  const ordenes = [];
  const crearTarea = (canal) => {
    let n = 0;
    return {
      paso() {
        n += 1;
        canal.progreso({ n });
        canal.imagen(new Blob([String(n)]));
        if (n >= pasos) {
          canal.terminar({ total: n });
        }
      },
      orden: (orden) => ordenes.push(orden),
    };
  };
  return { crearTarea, ordenes };
}

function seguidor() {
  return { alEstado: vi.fn(), alHistorial: vi.fn(), alProgreso: vi.fn(), alImagen: vi.fn(),
    alError: vi.fn(), alCambiarEstado: vi.fn() };
}

const estados = (c) => c.alEstado.mock.calls.map(([t]) => t.estado);
const conexion = (c) => c.alCambiarEstado.mock.calls.map(([e]) => e);

describe('servidor de trabajos simulado', () => {
  it('crea un trabajo en marcha con sus argumentos copiados y lo lista por tipo', () => {
    const servidor = crearServidorTrabajosSimulado({ intervalo: INTERVALO, reloj: () => 7 });
    const argumentos = { entorno: 'cartpole', lista: [1] };
    const t = servidor.crear('comparacion', argumentos, contador().crearTarea);
    argumentos.lista.push(2);
    expect(t).toMatchObject({ id: 'simulado-1', tipo: 'comparacion', estado: 'en_marcha', creado: 7,
      iniciado: 7, terminado: null, progresos: 0, resultado: null, error: null, interactivo: false,
      argumentos: { entorno: 'cartpole', lista: [1] } });
    servidor.crear('entrenamiento', {}, contador().crearTarea);
    expect(servidor.listar({ tipo: 'comparacion' }).map((x) => x.id)).toEqual(['simulado-1']);
    expect(servidor.listar().map((x) => x.id)).toEqual(['simulado-1', 'simulado-2']);
    expect(servidor.obtener('simulado-2').tipo).toBe('entrenamiento');
    expect(servidor.obtener('otro')).toBeNull();
  });

  it('al seguir manda primero el estado, el historial y la última imagen, nunca dentro de la llamada', async () => {
    const servidor = crearServidorTrabajosSimulado({ intervalo: INTERVALO });
    const { id } = servidor.crear('comparacion', {}, contador(10).crearTarea);
    await vi.advanceTimersByTimeAsync(INTERVALO * 2);
    const c = seguidor();
    servidor.seguir(id, c);
    expect(conexion(c)).toEqual([ESTADOS_CONEXION.CONECTANDO]);
    expect(c.alEstado).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(0);
    expect(conexion(c)).toEqual([ESTADOS_CONEXION.CONECTANDO, ESTADOS_CONEXION.ABIERTA]);
    expect(c.alEstado.mock.calls[0][0]).toMatchObject({ estado: 'en_marcha', progresos: 2 });
    expect(c.alHistorial).toHaveBeenCalledWith({ progreso: [{ n: 1 }, { n: 2 }], eventos: [] });
    expect(await c.alImagen.mock.calls[0][0].text()).toBe('2');
    await vi.advanceTimersByTimeAsync(INTERVALO);
    expect(c.alProgreso).toHaveBeenCalledWith({ n: 3 });
  });

  it('al terminar manda el estado final con el resultado y cierra', async () => {
    const servidor = crearServidorTrabajosSimulado({ intervalo: INTERVALO, reloj: () => 9 });
    const { id } = servidor.crear('comparacion', {}, contador(2).crearTarea);
    const c = seguidor();
    servidor.seguir(id, c);
    await vi.advanceTimersByTimeAsync(INTERVALO * 5);
    expect(c.alProgreso).toHaveBeenCalledTimes(2);
    expect(c.alEstado.mock.calls.at(-1)[0]).toMatchObject({ estado: 'terminado', resultado: { total: 2 }, terminado: 9 });
    expect(conexion(c).at(-1)).toBe(ESTADOS_CONEXION.CERRADA);
  });

  it('seguir uno terminado manda su estado e historial y cierra; uno que no existe, error', async () => {
    const servidor = crearServidorTrabajosSimulado({ intervalo: INTERVALO });
    const { id } = servidor.crear('comparacion', {}, contador(1).crearTarea);
    await vi.advanceTimersByTimeAsync(INTERVALO);
    const c = seguidor();
    const seguido = servidor.seguir(id, c);
    await vi.advanceTimersByTimeAsync(0);
    expect(estados(c)).toEqual(['terminado']);
    expect(c.alHistorial.mock.calls[0][0].progreso).toEqual([{ n: 1 }]);
    expect(conexion(c).at(-1)).toBe(ESTADOS_CONEXION.CERRADA);
    expect(seguido.pausar()).toBe(false);

    const nadie = seguidor();
    const otro = servidor.seguir('no', nadie);
    await vi.advanceTimersByTimeAsync(0);
    expect(nadie.alError.mock.calls[0][0].message).toBe('No existe el trabajo no');
    expect(conexion(nadie).at(-1)).toBe(ESTADOS_CONEXION.CERRADA);
    expect(otro.ordenar({ tipo: 'finalizar' })).toBe(false);
    expect(() => otro.cerrar()).not.toThrow();
  });

  it('pausa y reanuda: en pausa no avanza', async () => {
    const servidor = crearServidorTrabajosSimulado({ intervalo: INTERVALO });
    const { id } = servidor.crear('comparacion', {}, contador(10).crearTarea);
    const c = seguidor();
    const seguido = servidor.seguir(id, c);
    await vi.advanceTimersByTimeAsync(0);
    expect(seguido.pausar()).toBe(true);
    seguido.pausar();
    await vi.advanceTimersByTimeAsync(INTERVALO * 5);
    expect(c.alProgreso).not.toHaveBeenCalled();
    expect(seguido.reanudar()).toBe(true);
    seguido.reanudar();
    await vi.advanceTimersByTimeAsync(INTERVALO);
    expect(c.alProgreso).toHaveBeenCalledOnce();
    expect(estados(c)).toEqual(['en_marcha', 'pausado', 'en_marcha']);
  });

  it('finalizar en pausa reanuda y pasa la orden a la tarea; cancelar termina', async () => {
    const servidor = crearServidorTrabajosSimulado({ intervalo: INTERVALO, reloj: () => 3 });
    const { crearTarea, ordenes } = contador(10);
    const { id } = servidor.crear('comparacion', {}, crearTarea);
    const c = seguidor();
    const seguido = servidor.seguir(id, c);
    await vi.advanceTimersByTimeAsync(0);
    seguido.pausar();
    expect(seguido.ordenar({ tipo: 'finalizar' })).toBe(true);
    expect(ordenes).toEqual([{ tipo: 'finalizar' }]);
    expect(estados(c)).toEqual(['en_marcha', 'pausado', 'en_marcha']);
    seguido.ordenar({ tipo: 'vista', activa: true });
    expect(ordenes.at(-1)).toEqual({ tipo: 'vista', activa: true });
    expect(seguido.cancelar()).toBe(true);
    expect(c.alEstado.mock.calls.at(-1)[0]).toMatchObject({ estado: 'cancelado', terminado: 3 });
    expect(conexion(c).at(-1)).toBe(ESTADOS_CONEXION.CERRADA);
    await vi.advanceTimersByTimeAsync(INTERVALO * 5);
    expect(c.alProgreso).not.toHaveBeenCalled();
  });

  it('cerrar deja de seguirlo sin pararlo, y otro seguidor lo recupera entero', async () => {
    const servidor = crearServidorTrabajosSimulado({ intervalo: INTERVALO });
    const { id } = servidor.crear('comparacion', {}, contador(4).crearTarea);
    const c = seguidor();
    const seguido = servidor.seguir(id, c);
    await vi.advanceTimersByTimeAsync(INTERVALO);
    seguido.cerrar();
    expect(seguido.pausar()).toBe(false);
    await vi.advanceTimersByTimeAsync(INTERVALO * 2);
    expect(c.alProgreso).toHaveBeenCalledOnce();
    const otro = seguidor();
    servidor.seguir(id, otro);
    await vi.advanceTimersByTimeAsync(0);
    expect(otro.alHistorial.mock.calls[0][0].progreso).toEqual([{ n: 1 }, { n: 2 }, { n: 3 }]);
  });

  it('cerrar antes de abrir no manda nada', async () => {
    const servidor = crearServidorTrabajosSimulado({ intervalo: INTERVALO });
    const { id } = servidor.crear('comparacion', {}, contador().crearTarea);
    const c = seguidor();
    servidor.seguir(id, c).cerrar();
    await vi.advanceTimersByTimeAsync(INTERVALO * 5);
    expect(conexion(c)).toEqual([ESTADOS_CONEXION.CONECTANDO]);
    expect(c.alEstado).not.toHaveBeenCalled();
  });

  it('una tarea que falla deja el trabajo fallido con el error', async () => {
    const servidor = crearServidorTrabajosSimulado({ intervalo: INTERVALO });
    const { id } = servidor.crear('comparacion', {}, (canal) => ({ paso: () => canal.fallar('se rompió') }));
    const c = seguidor();
    servidor.seguir(id, c);
    await vi.advanceTimersByTimeAsync(INTERVALO);
    expect(c.alEstado.mock.calls.at(-1)[0]).toMatchObject({ estado: 'fallido', error: 'se rompió' });
    expect(servidor.obtener(id).estado).toBe('fallido');
  });
});
