import { act, fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { ENTORNOS } from '../entornos.js';
import { apiConSocket } from '../pruebas/apiConSocket.js';
import Laboratorio from './Laboratorio.jsx';

const IDS = ENTORNOS.map(({ id }) => id);
// Dominio finito: los 20 cambios posibles entre los cinco entornos.
const CAMBIOS = IDS.flatMap((origen) => IDS.filter((d) => d !== origen).map((destino) => [origen, destino]));
const PARTIDA = { indice: 0, pasos: 10, retorno: 10, semilla: 1 };

const valor = (nombre) => screen.getByText(nombre).nextElementSibling;
const pulsar = (nombre) => act(async () => {
  fireEvent.click(screen.getByText(nombre, { selector: 'button' }));
});

async function jugarUnaPartida(origen) {
  const doble = apiConSocket();
  const vista = render(<Laboratorio entorno={origen} api={doble.api} />);
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

    vista.rerender(<Laboratorio entorno={destino} api={api} />);
    await act(async () => {});
    esperarPanelLimpio();
    expect(screen.queryByText(/Guardado «mio»/)).not.toBeInTheDocument();
  });

  it('tras perder la conexión tampoco queda el error del anterior', async () => {
    const { api, socket, vista } = await jugarUnaPartida('cartpole');
    act(() => socket.caer());
    expect(screen.getByRole('alert')).toHaveTextContent('Se perdió la conexión');

    vista.rerender(<Laboratorio entorno="reacher" api={api} />);
    await act(async () => {});
    esperarPanelLimpio();
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });

  it('si falla la configuración del nuevo, no hereda las partidas del anterior', async () => {
    const { api, http } = apiConSocket();
    const vista = render(<Laboratorio entorno="cartpole" api={api} />);
    await act(async () => {});
    await pulsar('Ver agente');
    expect(screen.getByRole('spinbutton')).toHaveValue(15);

    http.obtener.mockRejectedValueOnce(new Error('Perfil ilegible'));
    vista.rerender(<Laboratorio entorno="reacher" api={api} />);
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
    const vista = render(<Laboratorio entorno="cartpole" api={api} />);
    vista.rerender(<Laboratorio entorno="reacher" api={api} />);
    await act(async () => {});
    await act(async () => responder({ fps: 50, partidas_experto: 15 }));
    expect(screen.getByRole('spinbutton')).toHaveValue(30);
  });
});
