import { act, fireEvent, render, screen, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { ErrorApi } from '../api/http.js';
import { MODELOS_PRUEBA, apiCompararConSocket } from '../pruebas/apiConSocket.js';
import { crearAlmacen } from '../simulado/almacen.js';
import { crearApiCompararSimulada } from '../simulado/comparar.js';
import EVALUACION from '../simulado/evaluacion.json';
import { crearServidorTrabajosSimulado } from '../simulado/trabajos.js';
import PanelComparar, { leerPartidas } from './PanelComparar.jsx';
import { ESPERA_CREAR_MS, ESPERA_FIN_MS } from './useComparacion.js';

const INTERVALO = 100;
const dibujar = async () => new Blob(['imagen'], { type: 'image/jpeg' });
const avanzar = (ms) => act(() => vi.advanceTimersByTimeAsync(ms));
// Con los relojes falsos de Vitest, userEvent se queda esperando; fireEvent es síncrono.
const pulsar = async (elemento) => {
  fireEvent.click(elemento);
  await avanzar(0);
};
const boton = (nombre, dentro = screen) => dentro.getByRole('button', { name: nombre });
const casilla = (nombre) => screen.getByRole('checkbox', { name: nombre });
const campoPartidas = () => screen.getByRole('spinbutton', { name: 'Partidas por modelo' });
const estado = () => screen.getByRole('status');
// Filas de la primera tabla del bloque con ese título, sin la cabecera.
const filas = (titulo) => [...screen.getByText(titulo).closest('.resultados-bloque').querySelector('table')
  .querySelectorAll('tbody tr')].map((fila) => [...fila.querySelectorAll('th, td')].map((c) => c.textContent));

beforeEach(() => {
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
});

function simulada() {
  const api = crearApiCompararSimulada({ almacen: crearAlmacen(),
    servidor: crearServidorTrabajosSimulado({ intervalo: INTERVALO }), dibujar });
  for (const metodo of ['modelos', 'comparaciones', 'crear']) {
    api[metodo] = vi.fn(api[metodo]);
  }
  return api;
}

async function montar(api, entorno = 'cartpole') {
  const ocupado = vi.fn();
  const vista = render(<PanelComparar entorno={entorno} api={api} alCambiarOcupado={ocupado} />);
  await avanzar(0);
  return { ocupado, ...vista };
}

async function comparar(api, { partidas = '3', vista = false } = {}) {
  const montado = await montar(api);
  fireEvent.click(casilla(/Experto congelado/));
  fireEvent.click(casilla(/BC · 1 partidas/));
  fireEvent.change(campoPartidas(), { target: { value: partidas } });
  if (vista) {
    fireEvent.click(casilla(/Ver las partidas mientras compara/));
  }
  await pulsar(boton('Iniciar comparación'));
  return montado;
}

describe('leerPartidas', () => {
  it.each([['', undefined], ['  ', undefined], ['1', 1], [' 50 ', 50], ['1000', 1000], ['0', null],
    ['1001', null], ['2.5', null], ['-3', null], ['1e2', null], ['abc', null]])('%j → %j', (texto, valor) => {
    expect(leerPartidas(texto)).toBe(valor);
  });
});

describe('PanelComparar con el simulado', () => {
  it('lista los modelos por tipo, con el aleatorio marcado como referencia', async () => {
    const api = simulada();
    await montar(api);
    expect(api.modelos).toHaveBeenCalledWith('cartpole');
    const grupos = screen.getAllByRole('heading', { level: 3 }).map((h) => h.textContent);
    expect(grupos.slice(0, 4)).toEqual(['Expertos', 'GAIL', 'BC', 'Referencia']);
    expect(casilla(/Aleatorio/)).toBeChecked();
    expect(casilla(/Experto congelado/)).not.toBeChecked();
    expect(screen.getByText('Todavía no hay ninguna.')).toBeInTheDocument();
  });

  it('pide al menos dos participantes', async () => {
    const api = simulada();
    await montar(api);
    fireEvent.click(casilla(/Aleatorio/));
    fireEvent.click(casilla(/Experto congelado/));
    await pulsar(boton('Iniciar comparación'));
    expect(screen.getByRole('alert')).toHaveTextContent('Elige al menos dos participantes');
    expect(api.crear).not.toHaveBeenCalled();
    // Uno y el aleatorio ya valen.
    fireEvent.click(casilla(/Aleatorio/));
    await pulsar(boton('Iniciar comparación'));
    expect(api.crear).toHaveBeenCalledOnce();
  });

  it.each(['0', '1001', '2.5', '-1'])('rechaza %j partidas sin llamar al servidor', async (texto) => {
    const api = simulada();
    await montar(api);
    fireEvent.click(casilla(/Experto congelado/));
    fireEvent.change(campoPartidas(), { target: { value: texto } });
    await pulsar(boton('Iniciar comparación'));
    expect(screen.getByRole('alert')).toHaveTextContent('entre 1 y 1000');
    expect(campoPartidas()).toHaveAttribute('aria-invalid', 'true');
    expect(api.crear).not.toHaveBeenCalled();
  });

  it('sin partidas no las manda y usa las de la evaluación de la E1', async () => {
    const api = simulada();
    await montar(api);
    fireEvent.click(casilla(/GAIL · 10 partidas/));
    fireEvent.click(casilla(/Experto congelado/));
    await pulsar(boton('Iniciar comparación'));
    // Los ids van en el orden de la biblioteca, no en el de los clics.
    expect(api.crear).toHaveBeenCalledWith({ entorno: 'cartpole', modelos: ['experto-cartpole', 'gail-cartpole-10'],
      aleatorio: true, vista: false }, expect.anything());
    expect(screen.getByText(new RegExp(`${EVALUACION.cartpole.episodios} partidas · semilla ${EVALUACION.cartpole.semilla}`)))
      .toBeInTheDocument();
  });

  it('compara de principio a fin: progreso, tabla, partidas e historial', async () => {
    const api = simulada();
    const { ocupado } = await comparar(api);
    expect(ocupado).toHaveBeenLastCalledWith(true);
    expect(estado()).toHaveTextContent('Comparando');
    expect(screen.getByText(/^0 de 3 partidas/)).toBeInTheDocument();

    await avanzar(INTERVALO);
    expect(screen.getByText(/^1 de 3 partidas/)).toBeInTheDocument();
    // Con una partida no hay desviación todavía.
    expect(filas('Recompensa media por partida')[0][2]).not.toContain('±');

    await avanzar(INTERVALO * 2);
    expect(estado()).toHaveTextContent('Terminada');
    expect(ocupado).toHaveBeenLastCalledWith(false);
    expect(screen.getByRole('progressbar')).toHaveAttribute('aria-valuenow', '100');
    const tabla = filas('Recompensa media por partida');
    expect(tabla.map((f) => f[0])).toEqual(['Experto congelado · CartPole', 'BC · 1 partidas', 'Aleatorio']);
    expect(tabla.every((f) => f[1] === '3' && f[2].includes('±'))).toBe(true);

    const detalle = within(screen.getByText('Puntuación de cada partida').closest('details'));
    expect(detalle.getAllByRole('row')).toHaveLength(4);
    const historial = filas('Historial de entrenamiento');
    expect(historial[0][1]).toMatch(/pasos/);
    expect(historial[1][2]).toBe('1 partida · repetición 0');
    expect(historial[2]).toEqual(['Aleatorio', 'No aplica', 'No aplica']);

    expect(screen.queryByRole('button', { name: 'Pausar' })).toBeNull();
    expect(screen.getAllByRole('button', { name: 'Ver' })).toHaveLength(1);
    await pulsar(boton('Nueva comparación'));
    expect(screen.getByRole('form', { name: 'Nueva comparación' })).toBeInTheDocument();
  });

  it('una comparación anterior se vuelve a ver con su resultado guardado', async () => {
    const api = simulada();
    await comparar(api, { partidas: '2' });
    await avanzar(INTERVALO * 3);
    const antes = filas('Recompensa media por partida');
    await pulsar(boton('Nueva comparación'));
    await pulsar(boton('Ver'));
    expect(estado()).toHaveTextContent('Terminada');
    expect(filas('Recompensa media por partida')).toEqual(antes);
    expect(boton('Ver')).toBeDisabled();
  });

  it('pausa, reanuda y finaliza con las partidas jugadas', async () => {
    const api = simulada();
    await comparar(api, { partidas: '50' });
    await avanzar(INTERVALO * 2);
    await pulsar(boton('Pausar'));
    expect(estado()).toHaveTextContent('Pausada');
    await avanzar(INTERVALO * 5);
    expect(screen.getByText(/^2 de 50 partidas/)).toBeInTheDocument();
    await pulsar(boton('Reanudar'));
    await avanzar(INTERVALO);
    expect(screen.getByText(/^3 de 50 partidas/)).toBeInTheDocument();
    await pulsar(boton('Finalizar'));
    expect(estado()).toHaveTextContent('Terminada');
    expect(screen.getByText(/finalizada antes de tiempo/)).toBeInTheDocument();
    expect(filas('Recompensa media por partida').every((f) => f[1] === '3')).toBe(true);
    // Regresión: el final llegaba durante el envío y aun así se esperaba la confirmación.
    await avanzar(ESPERA_FIN_MS);
    expect(screen.queryByRole('alert')).toBeNull();
  });

  it('finalizar en pausa también termina', async () => {
    const api = simulada();
    await comparar(api, { partidas: '50' });
    await avanzar(INTERVALO);
    await pulsar(boton('Pausar'));
    await pulsar(boton('Finalizar'));
    expect(estado()).toHaveTextContent('Terminada');
  });

  it('cancelar la descarta y lo avisa', async () => {
    const api = simulada();
    const { ocupado } = await comparar(api, { partidas: '50' });
    await avanzar(INTERVALO);
    await pulsar(boton('Cancelar'));
    expect(estado()).toHaveTextContent('Cancelada');
    expect(screen.getByRole('alert')).toHaveTextContent('Comparación cancelada.');
    expect(ocupado).toHaveBeenLastCalledWith(false);
  });

  it('con vista muestra las partidas y se puede quitar en marcha', async () => {
    const api = simulada();
    await comparar(api, { partidas: '50', vista: true });
    await avanzar(INTERVALO);
    expect(screen.getByRole('img')).toBeInTheDocument();
    fireEvent.click(casilla('Ver las partidas'));
    await avanzar(0);
    expect(screen.queryByRole('img')).toBeNull();
    fireEvent.click(casilla('Ver las partidas'));
    await avanzar(0);
    expect(screen.getByText('Esperando la primera imagen…')).toBeInTheDocument();
    await avanzar(INTERVALO);
    expect(screen.getByRole('img')).toBeInTheDocument();
  });

  it('sin vista no hay imagen', async () => {
    const api = simulada();
    await comparar(api, { partidas: '50' });
    await avanzar(INTERVALO * 2);
    expect(screen.queryByRole('img')).toBeNull();
    expect(casilla('Ver las partidas')).not.toBeChecked();
  });

  it('al cambiar de entorno vuelve al formulario y olvida la elección', async () => {
    const api = simulada();
    const { rerender } = await comparar(api, { partidas: '1' });
    await avanzar(INTERVALO);
    expect(estado()).toHaveTextContent('Terminada');
    rerender(<PanelComparar entorno="acrobot" api={api} />);
    await avanzar(0);
    expect(api.modelos).toHaveBeenLastCalledWith('acrobot');
    expect(casilla(/Experto congelado · Acrobot/)).not.toBeChecked();
    expect(screen.getByText('Todavía no hay ninguna.')).toBeInTheDocument();
  });

  it('muestra los errores del servidor al crear y deja volver a intentarlo', async () => {
    const api = simulada();
    api.crear.mockRejectedValueOnce(new ErrorApi('Modelos que no existen', { estado: 422 }));
    const { ocupado } = await comparar(api);
    expect(screen.getByRole('alert')).toHaveTextContent('Modelos que no existen');
    expect(screen.getByRole('form', { name: 'Nueva comparación' })).toBeInTheDocument();
    expect(ocupado).toHaveBeenLastCalledWith(false);
  });

  it('avisa si no carga la biblioteca y deja reintentar', async () => {
    const api = simulada();
    api.modelos.mockRejectedValueOnce(new Error('sin red'));
    await montar(api);
    expect(screen.getByRole('alert')).toHaveTextContent('No se pudo cargar la biblioteca (sin red)');
    expect(boton('Iniciar comparación')).toBeDisabled();
    await pulsar(boton('Actualizar la biblioteca'));
    expect(casilla(/Experto congelado/)).toBeEnabled();
  });
});

describe('PanelComparar con el protocolo de trabajos', () => {
  async function enMarcha({ vista = false } = {}) {
    const doble = apiCompararConSocket();
    const { ocupado, unmount } = await montar(doble.api);
    fireEvent.click(casilla(/Experto congelado/));
    if (vista) {
      fireEvent.click(casilla(/Ver las partidas mientras compara/));
    }
    await pulsar(boton('Iniciar comparación'));
    const socket = doble.sockets.at(-1);
    socket.abrir();
    socket.recibir({ tipo: 'estado', trabajo: { id: 'c1', estado: 'en_marcha', argumentos:
      { entorno: 'cartpole', modelos: ['exp'], aleatorio: true, partidas: 2, semilla: 9000 } } });
    socket.recibir({ tipo: 'historial', progreso: [], eventos: [] });
    await avanzar(0);
    return { ...doble, socket, ocupado, unmount };
  }
  const progreso = (modelo, partida, retorno) => ({ tipo: 'progreso', medidas: { modelo, partida, retorno, pasos: 10 } });
  const PERDIDA = 'Se perdió la conexión con el servidor. La comparación puede seguir allí: reconecta para verla.';

  it('no cuenta un modelo elegido que ya no está al actualizar la biblioteca', async () => {
    const { api, http } = apiCompararConSocket();
    await montar(api);
    fireEvent.click(casilla(/Aleatorio/));
    fireEvent.click(casilla(/Experto congelado/));
    fireEvent.click(casilla(/GAIL 1/));
    http.obtener.mockImplementation(async (ruta) => (ruta === '/trabajos' ? [] : [MODELOS_PRUEBA[0]]));
    await pulsar(boton('Actualizar la biblioteca'));
    await pulsar(boton('Iniciar comparación'));
    expect(screen.getByRole('alert')).toHaveTextContent('Elige al menos dos participantes');
    expect(http.enviar).not.toHaveBeenCalled();
  });

  it('muestra desactivados los modelos que no son válidos, con el motivo', async () => {
    const { api } = apiCompararConSocket();
    await montar(api);
    expect(casilla(/BC roto/)).toBeDisabled();
    expect(screen.getByText('no válido: falta politica.pt')).toBeInTheDocument();
  });

  it('sigue el trabajo por /trabajos/{id} y manda la petición propuesta', async () => {
    const { http, socket } = await enMarcha();
    expect(http.enviar).toHaveBeenCalledWith('/comparaciones', { entorno: 'cartpole', modelos: ['exp'],
      aleatorio: true, vista: false }, expect.objectContaining({ senal: expect.any(AbortSignal) }));
    expect(socket.url).toMatch(/\/api\/trabajos\/c1$/);
    expect(estado()).toHaveTextContent('Comparando');
  });

  it('reconectar rehace los resultados con el historial, sin perder ni repetir partidas', async () => {
    const { socket, sockets, ocupado } = await enMarcha();
    socket.recibir(progreso('exp', 0, 500));
    socket.recibir(progreso('aleatorio', 0, 20));
    socket.caer();
    await avanzar(0);
    expect(screen.getByRole('alert')).toHaveTextContent(PERDIDA);
    expect(estado()).toHaveTextContent('Sin conexión');
    expect(ocupado).toHaveBeenLastCalledWith(false);

    await pulsar(boton('Reconectar'));
    const nuevo = sockets.at(-1);
    expect(nuevo).not.toBe(socket);
    nuevo.abrir();
    nuevo.recibir({ tipo: 'estado', trabajo: { id: 'c1', estado: 'en_marcha', argumentos:
      { entorno: 'cartpole', modelos: ['exp'], aleatorio: true, partidas: 2, semilla: 9000 } } });
    nuevo.recibir({ tipo: 'historial', progreso: [progreso('exp', 0, 500).medidas, progreso('aleatorio', 0, 20).medidas,
      progreso('exp', 1, 300).medidas], eventos: [] });
    await avanzar(0);
    expect(screen.queryByRole('alert')).toBeNull();
    expect(ocupado).toHaveBeenLastCalledWith(true);
    expect(filas('Recompensa media por partida')[0]).toEqual(['Experto congelado', '2', '400 ± 141', '300']);
    // Lo que ya estaba en el historial no cuenta dos veces si llega también en directo.
    nuevo.recibir(progreso('exp', 1, 300));
    nuevo.recibir(progreso('aleatorio', 1, 40));
    await avanzar(0);
    expect(filas('Recompensa media por partida').map((f) => f[1])).toEqual(['2', '2']);
    expect(screen.getByText(/^2 de 2 partidas/)).toBeInTheDocument();
  });

  it('el estado final con el resultado y el cierre normal no son una desconexión', async () => {
    const { socket, ocupado } = await enMarcha();
    socket.recibir(progreso('exp', 0, 500));
    socket.recibir({ tipo: 'estado', trabajo: { id: 'c1', estado: 'terminado', argumentos:
      { entorno: 'cartpole', modelos: ['exp'], aleatorio: true, partidas: 2, semilla: 9000 },
    resultado: { partidas: 2, semilla: 9000, incompleta: false,
      modelos: [{ id: 'exp', retornos: [500, 480] }, { id: 'aleatorio', retornos: [20, 10] }] } } });
    socket.caer();
    await avanzar(0);
    expect(screen.queryByRole('alert')).toBeNull();
    expect(estado()).toHaveTextContent('Terminada');
    expect(ocupado).toHaveBeenLastCalledWith(false);
    // El resultado guardado completa lo que no llegó por progreso.
    expect(filas('Recompensa media por partida')).toEqual([['Experto congelado', '2', '490 ± 14,1', '480'],
      ['Aleatorio', '2', '15 ± 7,07', '10']]);
  });

  it.each([
    ['falla', { estado: 'fallido', error: 'Se acabó la memoria' }, 'Se acabó la memoria'],
    ['falla sin motivo', { estado: 'fallido', error: null }, 'La comparación falló.'],
  ])('si %s lo muestra', async (_, trabajo, texto) => {
    const { socket } = await enMarcha();
    socket.recibir({ tipo: 'estado', trabajo: { id: 'c1', argumentos: {}, ...trabajo } });
    socket.caer();
    await avanzar(0);
    expect(screen.getByRole('alert')).toHaveTextContent(texto);
    expect(estado()).toHaveTextContent('Falló');
  });

  it('manda pausar, reanudar, vista, finalizar y cancelar por el WebSocket', async () => {
    const { socket } = await enMarcha();
    await pulsar(boton('Pausar'));
    socket.recibir({ tipo: 'estado', trabajo: { id: 'c1', estado: 'pausado', argumentos: {} } });
    await avanzar(0);
    await pulsar(boton('Reanudar'));
    socket.recibir({ tipo: 'estado', trabajo: { id: 'c1', estado: 'en_marcha', argumentos: {} } });
    await avanzar(0);
    fireEvent.click(casilla('Ver las partidas'));
    await pulsar(boton('Cancelar'));
    await pulsar(boton('Finalizar'));
    expect(socket.enviados).toEqual([{ tipo: 'pausar' }, { tipo: 'reanudar' }, { tipo: 'vista', activa: true },
      { tipo: 'cancelar' }, { tipo: 'finalizar' }]);
    expect(estado()).toHaveTextContent('Terminando…');
  });

  it('en cola no se puede pausar ni finalizar, pero sí cancelar', async () => {
    const { socket } = await enMarcha();
    socket.recibir({ tipo: 'estado', trabajo: { id: 'c1', estado: 'en_cola', argumentos: {} } });
    await avanzar(0);
    expect(estado()).toHaveTextContent('En cola');
    expect(boton('Pausar')).toBeDisabled();
    expect(boton('Finalizar')).toBeDisabled();
    expect(boton('Cancelar')).toBeEnabled();
    expect(casilla('Ver las partidas')).toBeDisabled();
  });

  it('si el final no llega a tiempo se libera con un aviso', async () => {
    const { socket, ocupado } = await enMarcha();
    await pulsar(boton('Finalizar'));
    await avanzar(ESPERA_FIN_MS - 1);
    expect(screen.queryByRole('alert')).toBeNull();
    expect(ocupado).toHaveBeenLastCalledWith(true);
    await avanzar(1);
    expect(screen.getByRole('alert')).toHaveTextContent('El servidor no confirmó el final de la comparación.');
    expect(ocupado).toHaveBeenLastCalledWith(false);
    expect(socket.readyState).toBe(3);
  });

  it('el final que llega a tiempo cancela la espera', async () => {
    const { socket } = await enMarcha();
    await pulsar(boton('Finalizar'));
    socket.recibir({ tipo: 'estado', trabajo: { id: 'c1', estado: 'terminado', argumentos: {}, resultado: null } });
    socket.caer();
    await avanzar(ESPERA_FIN_MS * 2);
    expect(screen.queryByRole('alert')).toBeNull();
    expect(estado()).toHaveTextContent('Terminada');
  });

  it('una orden que no sale porque el socket ya está cerrado cuenta como desconexión', async () => {
    const { socket } = await enMarcha();
    socket.readyState = 3;
    await pulsar(boton('Pausar'));
    expect(screen.getByRole('alert')).toHaveTextContent(PERDIDA);
  });

  it('si el servidor no responde al crear, vuelve al formulario', async () => {
    const doble = apiCompararConSocket();
    doble.http.enviar.mockImplementationOnce((_, __, { senal }) => new Promise((_resolver, rechazar) => {
      senal.addEventListener('abort', () => rechazar(new DOMException('abortado', 'AbortError')));
    }));
    const { ocupado } = await montar(doble.api);
    fireEvent.click(casilla(/Experto congelado/));
    await pulsar(boton('Iniciar comparación'));
    expect(estado()).toHaveTextContent('Preparando…');
    expect(ocupado).toHaveBeenLastCalledWith(true);
    await avanzar(ESPERA_CREAR_MS);
    expect(screen.getByRole('alert')).toHaveTextContent('El servidor no respondió al crear la comparación.');
    expect(ocupado).toHaveBeenLastCalledWith(false);
    expect(doble.sockets).toHaveLength(0);
  });

  it('las imágenes solo se muestran con la vista activa', async () => {
    const { socket } = await enMarcha();
    socket.recibir(new Blob(['x'], { type: 'image/jpeg' }));
    await avanzar(0);
    expect(screen.queryByRole('img')).toBeNull();
    // Al encenderla no aparece la imagen vieja que llegó con la vista apagada.
    fireEvent.click(casilla('Ver las partidas'));
    await avanzar(0);
    expect(screen.getByText('Esperando la primera imagen…')).toBeInTheDocument();
    expect(screen.queryByRole('img')).toBeNull();
  });

  it('con vista desde el principio muestra las imágenes que llegan', async () => {
    const { socket } = await enMarcha({ vista: true });
    expect(screen.getByText('Esperando la primera imagen…')).toBeInTheDocument();
    socket.recibir(new Blob(['x'], { type: 'image/jpeg' }));
    await avanzar(0);
    expect(screen.getByRole('img')).toBeInTheDocument();
  });

  it('un mensaje desconocido se avisa sin cortar la comparación', async () => {
    const { socket } = await enMarcha();
    socket.recibir({ nada: 1 });
    await avanzar(0);
    expect(screen.getByRole('alert')).toHaveTextContent('Mensaje desconocido del servidor');
    expect(estado()).toHaveTextContent('Comparando');
  });

  it('al desmontar cierra la conexión sin parar la comparación', async () => {
    const { socket, unmount } = await enMarcha();
    unmount();
    expect(socket.readyState).toBe(3);
    expect(socket.enviados).toEqual([]);
  });

  it('lista las comparaciones anteriores del entorno y abre una en marcha', async () => {
    const doble = apiCompararConSocket();
    doble.http.obtener.mockImplementation(async (ruta) => (ruta === '/trabajos'
      ? [{ id: 'v1', tipo: 'comparacion', estado: 'en_marcha', creado: 1_764_000_000,
        argumentos: { entorno: 'cartpole', modelos: ['exp', 'g1'], aleatorio: true, partidas: 50 } },
      { id: 'v2', tipo: 'comparacion', estado: 'terminado', creado: 1, argumentos: { entorno: 'acrobot' } },
      { id: 'v3', tipo: 'comparacion', estado: 'terminado', creado: 2, argumentos: { entorno: 'cartpole' } }]
      : []));
    const { ocupado } = await montar(doble.api);
    expect(doble.http.obtener).toHaveBeenCalledWith('/trabajos', { parametros: { tipo: 'comparacion' }, senal: undefined });
    const lista = within(screen.getByText(/Comparaciones anteriores de CartPole/).closest('.anteriores'));
    expect(lista.getAllByRole('listitem')).toHaveLength(2);
    expect(lista.getByText(/3 participantes · 50 partidas/)).toBeInTheDocument();
    await pulsar(lista.getAllByRole('button', { name: 'Ver' })[0]);
    expect(doble.sockets).toHaveLength(1);
    expect(ocupado).toHaveBeenLastCalledWith(true);
    // Mientras sigue una en marcha no se abre otra.
    expect(lista.getAllByRole('button', { name: 'Ver' }).map((b) => b.disabled)).toEqual([true, true]);
  });
});
