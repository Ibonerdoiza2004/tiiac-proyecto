import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { apiConSocket } from '../pruebas/apiConSocket.js';
import { crearApiLaboratorioSimulada } from '../simulado/laboratorio.js';
import ExplorarGenerar from './ExplorarGenerar.jsx';
import { ESPERA_FIN_MS } from './useSesionJuego.js';

const dibujar = async () => new Blob(['imagen'], { type: 'image/jpeg' });

/** API simulada que además apunta las acciones que manda la interfaz. */
function apiEspia(base = crearApiLaboratorioSimulada({ dibujar })) {
  const acciones = [];
  const api = {
    ...base,
    crearSesion: vi.fn(base.crearSesion),
    guardar: vi.fn(base.guardar),
    descartar: vi.fn(base.descartar),
    conectar(id, callbacks) {
      const conexion = base.conectar(id, callbacks);
      return { ...conexion, enviarAccion: (accion) => {
        acciones.push(accion);
        return conexion.enviarAccion(accion);
      } };
    },
  };
  return { api, acciones };
}

const avanzar = (ms) => act(() => vi.advanceTimersByTimeAsync(ms));
const boton = (nombre) => screen.getByRole('button', { name: nombre });
const contador = (nombre) => Number(screen.getByText(nombre).nextElementSibling.textContent);

// Con los relojes falsos de Vitest, userEvent se queda esperando un temporizador que
// nunca avanza; fireEvent es síncrono y no tiene ese problema.
const usuario = {
  click: async (elemento) => {
    fireEvent.click(elemento);
    await avanzar(0);
  },
  clear: async (campo) => fireEvent.change(campo, { target: { value: '' } }),
  type: async (campo, texto) => fireEvent.change(campo, { target: { value: `${campo.value}${texto}` } }),
};

async function montar(entorno, api, alCambiarOcupado) {
  render(<ExplorarGenerar entorno={entorno} api={api} alCambiarOcupado={alCambiarOcupado} />);
  await avanzar(0);
}

beforeEach(() => {
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
});

describe('Jugar tú', () => {
  it('juega a CartPole con el teclado, revisa y guarda las partidas', async () => {
    const { api, acciones } = apiEspia();
    const ocupado = vi.fn();
    await montar('cartpole', api, ocupado);
    expect(boton('Jugar tú')).toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByText('Inicia una sesión para ver el entorno.')).toBeInTheDocument();

    await usuario.click(boton('Iniciar sesión'));
    await avanzar(100);
    expect(ocupado).toHaveBeenLastCalledWith(true);
    expect(screen.getByRole('status')).toHaveTextContent('Esperando tecla');
    expect(screen.getByText(/Mantén ← o →/)).toBeInTheDocument();
    expect(screen.getByRole('img', { name: 'CartPole: última imagen de la partida' }))
      .toHaveAttribute('src', expect.stringMatching(/^blob:/));
    expect(boton('Jugar tú')).toBeDisabled();

    fireEvent.keyDown(window, { key: 'ArrowRight' });
    await avanzar(4000);
    expect(screen.getByRole('status')).toHaveTextContent('En marcha');
    const terminadas = contador('Partidas terminadas');
    expect(terminadas).toBeGreaterThanOrEqual(2);
    expect(contador('Pares recogidos')).toBeGreaterThan(terminadas);
    expect(contador('Partida')).toBe(terminadas + 1);

    fireEvent.keyUp(window, { key: 'ArrowRight' });
    await avanzar(100);
    expect(screen.getByRole('status')).toHaveTextContent('Esperando tecla');
    expect(acciones).toEqual([1, null]);

    await usuario.click(boton('Finalizar'));
    await avanzar(0);
    const revision = screen.getByRole('form', { name: 'Revisar partidas' });
    expect(within(revision).getAllByRole('checkbox')).toHaveLength(terminadas);

    await usuario.click(within(revision).getByRole('button', { name: 'Guardar en la biblioteca' }));
    expect(screen.getByText('Escribe un nombre para el conjunto.')).toBeInTheDocument();
    expect(api.guardar).not.toHaveBeenCalled();

    const [desde, hasta] = within(revision).getAllByRole('spinbutton');
    await usuario.clear(hasta);
    await usuario.type(hasta, '999');
    await usuario.type(within(revision).getByRole('textbox', { name: 'Nombre del conjunto' }), 'mis partidas');
    await usuario.click(within(revision).getByRole('button', { name: 'Guardar en la biblioteca' }));
    expect(within(revision).getByText(/Elige un tramo entre 1 y/)).toBeInTheDocument();
    expect(api.guardar).not.toHaveBeenCalled();

    await usuario.clear(desde);
    await usuario.type(desde, '2');
    await usuario.clear(hasta);
    await usuario.type(hasta, '3');
    await usuario.click(within(revision).getByRole('button', { name: 'Guardar en la biblioteca' }));
    await avanzar(0);
    const [, datos] = api.guardar.mock.calls[0];
    expect(datos.nombre).toBe('mis partidas');
    expect(datos.partidas[0]).toEqual({ indice: 0, tramo: [1, 3] });
    expect(datos.partidas).toHaveLength(terminadas);
    expect(screen.getByText(new RegExp(`Guardado «mis partidas»: ${terminadas} partidas y \\d+ pares`)))
      .toBeInTheDocument();
    expect(ocupado).toHaveBeenLastCalledWith(false);
    expect(boton('Iniciar sesión')).toBeEnabled();
  });

  it('manda la última flecha pulsada, no repite y suelta al perder el foco', async () => {
    const { api, acciones } = apiEspia();
    await montar('mountaincar', api);
    await usuario.click(boton('Iniciar sesión'));
    await avanzar(50);
    fireEvent.keyDown(window, { key: 'ArrowLeft' });
    fireEvent.keyDown(window, { key: 'ArrowLeft', repeat: true });
    fireEvent.keyDown(window, { key: 'd' });
    fireEvent.keyDown(window, { key: 'Shift' });
    fireEvent.keyUp(window, { key: 'd' });
    fireEvent.keyUp(window, { key: 'Shift' });
    fireEvent.blur(window);
    fireEvent.blur(window);
    expect(acciones).toEqual([0, 2, 0, null]);
  });

  it('las flechas no hacen nada al escribir en un campo', async () => {
    const { api, acciones } = apiEspia();
    await montar('cartpole', api);
    await usuario.click(boton('Iniciar sesión'));
    await avanzar(50);
    const campo = document.createElement('input');
    document.body.append(campo);
    fireEvent.keyDown(campo, { key: 'ArrowRight' });
    campo.remove();
    expect(acciones).toEqual([]);
  });

  it('los botones en pantalla empujan mientras se mantienen', async () => {
    const { api, acciones } = apiEspia();
    await montar('cartpole', api);
    await usuario.click(boton('Iniciar sesión'));
    await avanzar(50);
    fireEvent.pointerDown(boton('Derecha'));
    fireEvent.pointerUp(boton('Derecha'));
    fireEvent.pointerDown(boton('Izquierda'));
    fireEvent.pointerLeave(boton('Izquierda'));
    expect(acciones).toEqual([1, null, 0, null]);
  });

  it('finalizar sin partidas terminadas avisa de que no hay nada que guardar', async () => {
    const { api } = apiEspia();
    await montar('cartpole', api);
    await usuario.click(boton('Iniciar sesión'));
    await avanzar(50);
    await usuario.click(boton('Finalizar'));
    await avanzar(0);
    expect(screen.getByRole('alert')).toHaveTextContent('sin ninguna partida completa');
    expect(screen.queryByRole('form')).not.toBeInTheDocument();
    expect(boton('Iniciar sesión')).toBeEnabled();
  });

  it('las teclas dejan de mandar acciones al terminar la sesión', async () => {
    const { api, acciones } = apiEspia();
    await montar('cartpole', api);
    await usuario.click(boton('Iniciar sesión'));
    await avanzar(50);
    // Mientras se juega, las flechas no desplazan la página (fireEvent da false)…
    expect(fireEvent.keyDown(window, { key: 'ArrowLeft' })).toBe(false);
    fireEvent.keyUp(window, { key: 'ArrowLeft' });
    await usuario.click(boton('Finalizar'));
    await avanzar(0);
    // …y al terminar vuelven a funcionar con normalidad.
    expect(fireEvent.keyDown(window, { key: 'ArrowRight' })).toBe(true);
    expect(acciones).toEqual([0, null]);
  });
});

describe('Ver agente', () => {
  it('en Reacher solo se puede ver al experto, con las partidas del perfil', async () => {
    const { api } = apiEspia();
    await montar('reacher', api);
    expect(boton('Jugar tú')).toBeDisabled();
    expect(boton('Ver agente')).toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByText('En Reacher solo se puede ver al experto y grabar partidas con él.'))
      .toBeInTheDocument();
    expect(screen.getByRole('spinbutton', { name: 'Partidas del experto' })).toHaveValue(30);
  });

  it('el experto juega las partidas pedidas, se revisan y se descartan', async () => {
    const { api } = apiEspia();
    await montar('acrobot', api);
    const campo = screen.getByRole('spinbutton', { name: 'Partidas del experto' });
    await usuario.clear(campo);
    await usuario.type(campo, '2');
    await usuario.click(boton('Ver al experto'));
    await avanzar(100);
    expect(api.crearSesion).toHaveBeenCalledWith({ entorno: 'acrobot', modo: 'experto', partidas: 2 });
    expect(screen.getByText('1 / 2')).toBeInTheDocument();
    expect(campo).toBeDisabled();
    await avanzar(20_000);
    const revision = screen.getByRole('form', { name: 'Revisar partidas' });
    expect(within(revision).getAllByRole('checkbox')).toHaveLength(2);
    expect(within(revision).getByText('Se guardarán 2 partidas y 240 pares.')).toBeInTheDocument();
    await usuario.click(within(revision).getAllByRole('checkbox')[0]);
    expect(within(revision).getByText('Se guardarán 1 partida y 120 pares.')).toBeInTheDocument();
    await usuario.click(boton('Descartar sesión'));
    await avanzar(0);
    expect(api.descartar).toHaveBeenCalledOnce();
    expect(boton('Ver al experto')).toBeEnabled();
  });

  it.each(['0', '1001', '2.5', ''])('no empieza con %j partidas', async (valor) => {
    const { api } = apiEspia();
    await montar('cartpole', api);
    await usuario.click(boton('Ver agente'));
    const campo = screen.getByRole('spinbutton', { name: 'Partidas del experto' });
    await usuario.clear(campo);
    if (valor) {
      await usuario.type(campo, valor);
    }
    await usuario.click(boton('Ver al experto'));
    expect(screen.getByText('Escribe un número entero de partidas entre 1 y 1000.')).toBeInTheDocument();
    expect(campo).toHaveAttribute('aria-invalid', 'true');
    expect(api.crearSesion).not.toHaveBeenCalled();
    await usuario.type(campo, '1');
    expect(screen.queryByText(/Escribe un número entero/)).not.toBeInTheDocument();
  });
});

describe('errores del backend', () => {
  it('si no se puede crear la sesión lo dice y deja empezar otra vez', async () => {
    const { api } = apiEspia();
    api.crearSesion.mockRejectedValueOnce(new Error('No hay procesos libres'));
    const ocupado = vi.fn();
    await montar('cartpole', api, ocupado);
    await usuario.click(boton('Iniciar sesión'));
    await avanzar(0);
    expect(screen.getByRole('alert')).toHaveTextContent('No hay procesos libres');
    expect(boton('Iniciar sesión')).toBeEnabled();
    expect(ocupado).toHaveBeenLastCalledWith(false);
  });

  it('si falla el guardado sigue en la revisión con el error', async () => {
    const { api } = apiEspia();
    api.guardar.mockRejectedValueOnce(new Error('Ya existe un conjunto con ese nombre'));
    await montar('reacher', api);
    const campo = screen.getByRole('spinbutton', { name: 'Partidas del experto' });
    await usuario.clear(campo);
    await usuario.type(campo, '1');
    await usuario.click(boton('Ver al experto'));
    await avanzar(5000);
    await usuario.type(screen.getByRole('textbox', { name: 'Nombre del conjunto' }), 'repetido');
    await usuario.click(boton('Guardar en la biblioteca'));
    await avanzar(0);
    expect(screen.getByRole('alert')).toHaveTextContent('Ya existe un conjunto con ese nombre');
    expect(screen.getByRole('form', { name: 'Revisar partidas' })).toBeInTheDocument();
    await usuario.click(boton('Guardar en la biblioteca'));
    await avanzar(0);
    expect(screen.getByText(/Guardado «repetido»: 1 partida y 50 pares/)).toBeInTheDocument();
  });

  it('un error de la conexión se muestra', async () => {
    const api = crearApiLaboratorioSimulada({ dibujar: () => Promise.reject(new Error('Conexión perdida')) });
    await montar('mountaincar', api);
    await usuario.click(boton('Iniciar sesión'));
    await avanzar(100);
    expect(screen.getByRole('alert')).toHaveTextContent('Conexión perdida');
  });
});

describe('conexión perdida', () => {
  const PERDIDA = 'Se perdió la conexión con el servidor';
  const SIN_RESPUESTA = 'El servidor no respondió al finalizar';
  const PARTIDA = { indice: 0, pasos: 10, retorno: 10, semilla: 1 };
  const MODOS_INICIO = [['manual', 'cartpole', 'Iniciar sesión'], ['experto', 'reacher', 'Ver al experto']];

  async function enSesion(entorno = 'cartpole', inicio = 'Iniciar sesión', abrir = true) {
    const doble = apiConSocket();
    const ocupado = vi.fn();
    await montar(entorno, doble.api, ocupado);
    await usuario.click(boton(inicio));
    const socket = doble.sockets[0];
    if (abrir) {
      act(() => socket.abrir());
    }
    return { ...doble, socket, ocupado };
  }

  function esperarLiberado(ocupado, texto, inicio = 'Iniciar sesión') {
    expect(screen.getByRole('alert')).toHaveTextContent(texto);
    expect(screen.getByRole('status')).toHaveTextContent('Preparado');
    expect(boton(inicio)).toBeEnabled();
    expect(boton('Ver agente')).toBeEnabled();
    expect(screen.queryByRole('button', { name: 'Finalizar' })).not.toBeInTheDocument();
    expect(ocupado).toHaveBeenLastCalledWith(false);
  }

  // Los dos modos por los tres momentos en que se puede cerrar el socket.
  describe.each(MODOS_INICIO)('en modo %s', (_, entorno, inicio) => {
    it('si se cierra antes de abrirse, libera la sesión', async () => {
      const { socket, ocupado } = await enSesion(entorno, inicio, false);
      act(() => socket.caer());
      esperarLiberado(ocupado, PERDIDA, inicio);
    });

    it('si se cierra jugando, libera la sesión', async () => {
      const { socket, ocupado } = await enSesion(entorno, inicio);
      act(() => socket.recibir({ tipo: 'partida_terminada', partida: PARTIDA }));
      act(() => socket.caer());
      esperarLiberado(ocupado, PERDIDA, inicio);
      expect(screen.queryByRole('form')).not.toBeInTheDocument();
    });

    it('si se cierra terminando, libera la sesión', async () => {
      const { socket, ocupado } = await enSesion(entorno, inicio);
      await usuario.click(boton('Finalizar'));
      expect(screen.getByRole('status')).toHaveTextContent('Terminando…');
      act(() => socket.caer());
      esperarLiberado(ocupado, PERDIDA, inicio);
    });

    it('si el servidor no responde al finalizar, espera el plazo justo y libera', async () => {
      const { socket, ocupado } = await enSesion(entorno, inicio);
      await usuario.click(boton('Finalizar'));
      expect(socket.enviados).toEqual([{ tipo: 'finalizar' }]);
      await avanzar(ESPERA_FIN_MS - 1);
      expect(screen.getByRole('status')).toHaveTextContent('Terminando…');
      expect(screen.queryByRole('alert')).not.toBeInTheDocument();
      await avanzar(1);
      esperarLiberado(ocupado, SIN_RESPUESTA, inicio);
      expect(socket.readyState).toBe(3);
    });
  });

  it.each([2, 3])('finalizar con el socket en readyState %i libera la sesión', async (estado) => {
    const { socket, ocupado } = await enSesion();
    socket.readyState = estado;
    await usuario.click(boton('Finalizar'));
    esperarLiberado(ocupado, PERDIDA);
  });

  it('finalizar mientras el socket aún conecta avisa sin perder la sesión', async () => {
    // Regresión: con readyState 0 la orden no sale y se daba la sesión por perdida.
    const { socket, ocupado } = await enSesion();
    socket.readyState = 0;
    await usuario.click(boton('Finalizar'));
    expect(screen.getByRole('alert')).toHaveTextContent('Todavía se está conectando');
    expect(ocupado).toHaveBeenLastCalledWith(true);
    expect(socket.enviados).toEqual([]);
    socket.readyState = 1;
    await usuario.click(boton('Finalizar'));
    expect(socket.enviados).toEqual([{ tipo: 'finalizar' }]);
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });

  it('si el servidor avisa de un error y cierra, se ve su motivo', async () => {
    // Regresión: «Se perdió la conexión» sustituía al mensaje del servidor.
    const { socket, ocupado } = await enSesion();
    act(() => {
      socket.recibir({ tipo: 'error', mensaje: 'El entorno falló.' });
      socket.caer();
    });
    esperarLiberado(ocupado, 'El entorno falló. Las partidas de esta sesión no se han guardado.');
  });

  it('el error del socket deja paso al de conexión perdida', async () => {
    const { socket, ocupado } = await enSesion();
    act(() => {
      socket.onerror({});
      socket.caer();
    });
    esperarLiberado(ocupado, PERDIDA);
  });

  it('un mensaje de error con la conexión abierta no termina la sesión', async () => {
    const { socket } = await enSesion();
    act(() => socket.recibir({ tipo: 'error', mensaje: 'Aviso del servidor' }));
    expect(screen.getByRole('alert')).toHaveTextContent('Aviso del servidor');
    expect(boton('Finalizar')).toBeEnabled();
  });

  it('el cierre normal tras sesion_terminada no es una conexión perdida', async () => {
    const { socket } = await enSesion();
    act(() => {
      socket.recibir({ tipo: 'sesion_terminada', partidas: [PARTIDA] });
      socket.caer();
    });
    expect(screen.getByRole('form', { name: 'Revisar partidas' })).toBeInTheDocument();
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });

  it('una respuesta dentro del plazo anula la espera', async () => {
    const { socket } = await enSesion();
    await usuario.click(boton('Finalizar'));
    await avanzar(ESPERA_FIN_MS - 1);
    act(() => socket.recibir({ tipo: 'sesion_terminada', partidas: [PARTIDA] }));
    await avanzar(ESPERA_FIN_MS);
    expect(screen.getByRole('form', { name: 'Revisar partidas' })).toBeInTheDocument();
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });

  it('una respuesta fuera de plazo se ignora', async () => {
    const { socket, ocupado } = await enSesion();
    await usuario.click(boton('Finalizar'));
    await avanzar(ESPERA_FIN_MS);
    act(() => socket.recibir({ tipo: 'sesion_terminada', partidas: [PARTIDA] }));
    esperarLiberado(ocupado, SIN_RESPUESTA);
    expect(screen.queryByRole('form')).not.toBeInTheDocument();
  });

  it('se puede empezar otra y ni el socket ni la espera de la anterior la afectan', async () => {
    const { socket, sockets } = await enSesion();
    await usuario.click(boton('Finalizar'));
    act(() => socket.caer());
    await usuario.click(boton('Iniciar sesión'));
    act(() => sockets[1].abrir());
    act(() => socket.recibir({ tipo: 'sesion_terminada', partidas: [PARTIDA] }));
    await avanzar(ESPERA_FIN_MS * 2);
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    expect(boton('Finalizar')).toBeEnabled();
  });

  it('las flechas dejan de bloquearse al perder la conexión', async () => {
    const { socket } = await enSesion();
    act(() => socket.caer());
    expect(fireEvent.keyDown(window, { key: 'ArrowRight' })).toBe(true);
  });

  it('desmontar cancela la espera', async () => {
    await enSesion();
    await usuario.click(boton('Finalizar'));
    expect(vi.getTimerCount()).toBeGreaterThan(0);
    cleanup();
    expect(vi.getTimerCount()).toBe(0);
  });
});

describe('configuración que falla', () => {
  it('lo dice al ver al agente y deja escribir las partidas', async () => {
    const { api, http } = apiConSocket();
    http.obtener.mockRejectedValueOnce(new Error('Perfil ilegible'));
    await montar('cartpole', api);
    // Jugar a mano no necesita ese número.
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    await usuario.click(boton('Ver agente'));
    const campo = screen.getByRole('spinbutton', { name: 'Partidas del experto' });
    expect(campo).toHaveValue(null);
    expect(screen.getByRole('alert')).toHaveTextContent(
      'No se pudieron cargar las partidas por defecto (Perfil ilegible). Escribe cuántas quieres.');
    await usuario.type(campo, '3');
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    await usuario.click(boton('Ver al experto'));
    expect(http.enviar).toHaveBeenCalledWith('/sesiones',
      { entorno: 'cartpole', modo: 'experto', partidas: 3 });
  });
});

describe('configuración que llega tarde', () => {
  it('no pisa la del entorno que se muestra', async () => {
    const { api, http } = apiConSocket();
    let responder;
    http.obtener.mockReturnValueOnce(new Promise((resolver) => {
      responder = resolver;
    }));
    http.obtener.mockResolvedValueOnce({ fps: 50, partidas_experto: 30 });
    const vista = render(<ExplorarGenerar entorno="cartpole" api={api} />);
    vista.rerender(<ExplorarGenerar entorno="reacher" api={api} />);
    await avanzar(0);
    await act(async () => responder({ fps: 50, partidas_experto: 15 }));
    expect(screen.getByRole('spinbutton')).toHaveValue(30);
  });
});

describe('pares recogidos', () => {
  it('suman solo las partidas terminadas, no la que está en curso', async () => {
    const { api, sockets } = apiConSocket();
    await montar('cartpole', api);
    await usuario.click(boton('Iniciar sesión'));
    const estado = (partida, paso, pares) => act(() => sockets[0].recibir(
      { tipo: 'estado', partida, paso, retorno: paso, pares, esperando: false }));
    const terminar = (indice, pasos) => act(() => sockets[0].recibir(
      { tipo: 'partida_terminada', partida: { indice, pasos, retorno: pasos, semilla: indice } }));

    estado(1, 7, 7);
    expect(contador('Pares recogidos')).toBe(0);
    terminar(0, 10);
    estado(2, 3, 13);
    expect(contador('Pares recogidos')).toBe(10);
    terminar(1, 5);
    expect(contador('Pares recogidos')).toBe(15);
  });
});
