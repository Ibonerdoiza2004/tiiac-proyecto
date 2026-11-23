import { act, fireEvent, render, screen, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { ENTORNOS } from '../entornos.js';
import { apiConSocket } from '../pruebas/apiConSocket.js';
import { crearAlmacen } from '../simulado/almacen.js';
import { INTERVALO_SIMULADO_MS, crearApiEntrenamientoSimulada } from '../simulado/entrenamiento.js';
import Laboratorio from './Laboratorio.jsx';

const IDS = ENTORNOS.map(({ id }) => id);
// Dominio finito: los 20 cambios posibles entre los cinco entornos.
const CAMBIOS = IDS.flatMap((origen) => IDS.filter((d) => d !== origen).map((destino) => [origen, destino]));
const PARTIDA = { indice: 0, pasos: 10, retorno: 10, semilla: 1 };
const ENTRENAMIENTO = crearApiEntrenamientoSimulada({ dibujar: async () => new Blob(['x']) });

const valor = (nombre) => screen.getByText(nombre).nextElementSibling;
const pulsar = (nombre) => act(async () => {
  fireEvent.click(screen.getByText(nombre, { selector: 'button' }));
});

async function jugarUnaPartida(origen) {
  const doble = apiConSocket();
  const vista = render(<Laboratorio entorno={origen} api={doble.api} apiEntrenamiento={ENTRENAMIENTO} />);
  await act(async () => {});
  await pulsar('Ver agente');
  await pulsar('Ver al experto');
  const socket = doble.sockets[0];
  act(() => {
    socket.abrir();
    socket.recibir(new Blob(['imagen']));
    socket.recibir({ tipo: 'estado', partida: 2, paso: 4, retorno: 4, pares: 14, esperando: false });
    socket.recibir({ tipo: 'partida_terminada', partida: PARTIDA });
  });
  return { ...doble, socket, vista };
}

function esperarPanelLimpio() {
  expect(screen.queryByAltText(/última imagen/)).not.toBeInTheDocument();
  expect(screen.getByText('Preparado')).toBeInTheDocument();
  expect(valor('Partida')).toHaveTextContent(/^1$/);
  for (const nombre of ['Puntuación', 'Pares recogidos', 'Partidas terminadas']) {
    expect(valor(nombre)).toHaveTextContent(/^0$/);
  }
}

describe('cambio de entorno', () => {
  it.each(CAMBIOS)('tras guardar, de %s a %s no queda nada del anterior', async (origen, destino) => {
    const { api, socket, vista } = await jugarUnaPartida(origen);
    act(() => socket.recibir({ tipo: 'sesion_terminada', partidas: [PARTIDA] }));
    fireEvent.change(screen.getByLabelText('Nombre del conjunto'), { target: { value: 'mio' } });
    await pulsar('Guardar en la biblioteca');
    expect(screen.getByText(/Guardado «mio»/)).toBeInTheDocument();
    expect(screen.getByAltText(/última imagen/)).toBeInTheDocument();

    vista.rerender(<Laboratorio entorno={destino} api={api} apiEntrenamiento={ENTRENAMIENTO} />);
    await act(async () => {});
    esperarPanelLimpio();
    expect(screen.queryByText(/Guardado «mio»/)).not.toBeInTheDocument();
  });

  it('tras perder la conexión tampoco queda el error del anterior', async () => {
    const { api, socket, vista } = await jugarUnaPartida('cartpole');
    act(() => socket.caer());
    expect(screen.getByRole('alert')).toHaveTextContent('Se perdió la conexión');

    vista.rerender(<Laboratorio entorno="reacher" api={api} apiEntrenamiento={ENTRENAMIENTO} />);
    await act(async () => {});
    esperarPanelLimpio();
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });

  it('si falla la configuración del nuevo, no hereda las partidas del anterior', async () => {
    const { api, http } = apiConSocket();
    const vista = render(<Laboratorio entorno="cartpole" api={api} apiEntrenamiento={ENTRENAMIENTO} />);
    await act(async () => {});
    await pulsar('Ver agente');
    expect(screen.getByRole('spinbutton')).toHaveValue(15);

    http.obtener.mockRejectedValueOnce(new Error('Perfil ilegible'));
    vista.rerender(<Laboratorio entorno="reacher" api={api} apiEntrenamiento={ENTRENAMIENTO} />);
    await act(async () => {});
    expect(screen.getByRole('spinbutton')).toHaveValue(null);
    expect(screen.getByRole('alert')).toHaveTextContent('Perfil ilegible');
  });

  it('una configuración que llega tarde no pisa la del entorno nuevo', async () => {
    const { api, http } = apiConSocket();
    let responder;
    http.obtener.mockReturnValueOnce(new Promise((resolver) => {
      responder = resolver;
    }));
    http.obtener.mockResolvedValueOnce({ fps: 50, partidas_experto: 30 });
    const vista = render(<Laboratorio entorno="cartpole" api={api} apiEntrenamiento={ENTRENAMIENTO} />);
    vista.rerender(<Laboratorio entorno="reacher" api={api} apiEntrenamiento={ENTRENAMIENTO} />);
    await act(async () => {});
    await act(async () => responder({ fps: 50, partidas_experto: 15 }));
    expect(screen.getByRole('spinbutton')).toHaveValue(30);
  });
});

describe('entrenar en los dos paneles', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  const avanzar = (ms) => act(() => vi.advanceTimersByTimeAsync(ms));
  const clic = async (elemento) => {
    fireEvent.click(elemento);
    await avanzar(0);
  };
  const panel = (titulo) => screen.getByRole('heading', { name: titulo }).closest('article');

  it('el entorno sigue bloqueado mientras cualquiera de los dos entrena', async () => {
    const almacen = crearAlmacen();
    const entrenamiento = crearApiEntrenamientoSimulada({ almacen, dibujar: async () => new Blob(['x']) });
    const ocupado = vi.fn();
    render(<Laboratorio entorno="cartpole" api={apiConSocket().api} apiEntrenamiento={entrenamiento}
      alCambiarOcupado={ocupado} />);
    await avanzar(0);
    const explorar = panel('Explorar y generar');
    const imitar = panel('Entrenar e imitar');

    await clic(within(explorar).getByRole('button', { name: 'Entrenar experto' }));
    expect(within(explorar).getByRole('form', { name: 'Entrenar Experto (TRPO)' })).toBeInTheDocument();
    await clic(within(explorar).getByRole('button', { name: 'Entrenar Experto (TRPO)' }));
    expect(ocupado).toHaveBeenLastCalledWith(true);
    for (const pestana of ['Jugar tú', 'Entrenar experto', 'Ver agente']) {
      expect(within(explorar).getByRole('button', { name: pestana })).toBeDisabled();
    }

    // GAIL más corto que el experto: dos iteraciones.
    fireEvent.click(within(imitar).getByRole('checkbox', { name: /experto congelado/ }));
    fireEvent.change(within(imitar).getByLabelText('iteraciones'), { target: { value: '2' } });
    await clic(within(imitar).getByRole('button', { name: 'Entrenar GAIL' }));
    await avanzar(INTERVALO_SIMULADO_MS * 3);
    expect(within(imitar).getByText(/gail_cartpole_1/)).toBeInTheDocument();
    expect(within(explorar).getByRole('button', { name: 'Finalizar' })).toBeEnabled();
    expect(ocupado).toHaveBeenLastCalledWith(true);

    await clic(within(explorar).getByRole('button', { name: 'Finalizar' }));
    expect(within(explorar).getByText(/experto_cartpole_1/)).toBeInTheDocument();
    expect(ocupado).toHaveBeenLastCalledWith(false);
    expect(almacen.modelos.map((m) => m.nombre).sort()).toEqual(['experto_cartpole_1', 'gail_cartpole_1']);
  });
});
