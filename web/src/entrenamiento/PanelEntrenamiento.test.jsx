import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { apiEntrenamientoConSocket } from '../pruebas/apiConSocket.js';
import { crearAlmacen } from '../simulado/almacen.js';
import { INTERVALO_SIMULADO_MS, crearApiEntrenamientoSimulada } from '../simulado/entrenamiento.js';
import PanelEntrenamiento from './PanelEntrenamiento.jsx';
import { ESPERA_CREAR_MS, ESPERA_FIN_ENTRENO_MS } from './useEntrenamiento.js';

const dibujar = async () => new Blob(['imagen'], { type: 'image/jpeg' });
const avanzar = (ms) => act(() => vi.advanceTimersByTimeAsync(ms));
// Con los relojes falsos de Vitest, userEvent se queda esperando; fireEvent es síncrono.
const pulsar = async (elemento) => {
  fireEvent.click(elemento);
  await avanzar(0);
};
const escribir = (campo, texto) => fireEvent.change(campo, { target: { value: texto } });
const boton = (nombre, dentro = screen) => dentro.getByRole('button', { name: nombre });
const formulario = (metodo) => screen.getByRole('form', { name: `Entrenar ${metodo}` });

function simulada(opciones) {
  const api = crearApiEntrenamientoSimulada({ dibujar, almacen: crearAlmacen(), ...opciones });
  for (const metodo of ['ajustes', 'demostraciones', 'crear']) {
    api[metodo] = vi.fn(api[metodo]);
  }
  return api;
}

async function montar(api, metodos = ['gail', 'bc'], entorno = 'cartpole') {
  const ocupado = vi.fn();
  render(<PanelEntrenamiento entorno={entorno} api={api} metodos={metodos} alCambiarOcupado={ocupado} />);
  await avanzar(0);
  return ocupado;
}

async function empezarGail(api, { vista = false } = {}) {
  const ocupado = await montar(api);
  fireEvent.click(screen.getByRole('checkbox', { name: /experto congelado/ }));
  if (vista) {
    fireEvent.click(screen.getByRole('checkbox', { name: /Ver las partidas mientras entrena/ }));
  }
  await pulsar(boton('Entrenar GAIL'));
  return ocupado;
}

beforeEach(() => {
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
});

describe('formulario', () => {
  it('muestra los hiperparámetros del perfil, por secciones, y los datos', async () => {
    const api = simulada();
    await montar(api);
    expect(api.ajustes).toHaveBeenCalledWith('cartpole', 'gail');
    const form = formulario('GAIL');
    expect(within(form).getByText('Dura 20 iteraciones.', { exact: false })).toBeInTheDocument();
    expect([...form.querySelectorAll('summary code')].map((c) => c.textContent))
      .toEqual(['gail', 'gail.discriminador', 'trpo', 'demostraciones']);
    expect(within(form).getByLabelText('iteraciones')).toHaveValue('20');
    expect(within(form).getByLabelText('gamma')).toHaveValue('0.995');
    expect(within(form).getByLabelText('gamma')).toHaveAccessibleDescription('Paper Ho y Ermon (2016), apéndice B.');
    expect(within(form).getByRole('checkbox', { name: /experto congelado/ })).not.toBeChecked();
    expect(within(form).getByText('Sin cambios respecto al perfil')).toBeInTheDocument();
    expect(within(form).getByRole('checkbox', { name: /Ver las partidas/ })).not.toBeChecked();
    expect(within(form).getByRole('checkbox', { name: /W&B/ })).not.toBeChecked();
  });

  it('BC no tiene vista y el experto no tiene datos', async () => {
    const api = simulada();
    await montar(api);
    await pulsar(boton('BC'));
    expect(api.ajustes).toHaveBeenLastCalledWith('cartpole', 'bc');
    expect(within(formulario('BC')).queryByRole('checkbox', { name: /Ver las partidas/ })).toBeNull();
    expect(within(formulario('BC')).getByLabelText('epocas_maximas')).toHaveValue('500');
    cleanup();
    await montar(api, ['experto']);
    expect(screen.queryByRole('group', { name: 'Método de entrenamiento' })).toBeNull();
    expect(screen.queryByText('Demostraciones')).toBeNull();
    expect(within(formulario('Experto (TRPO)')).getByRole('checkbox', { name: /Ver las partidas/ }))
      .toBeInTheDocument();
  });

  it('envía solo lo que cambia, con los datos, la vista y W&B elegidos', async () => {
    const api = simulada();
    await montar(api);
    const form = formulario('GAIL');
    fireEvent.click(within(form).getByRole('checkbox', { name: /experto congelado/ }));
    escribir(within(form).getByLabelText('learning_rate'), '0,02');
    escribir(within(form).getByLabelText('capas_ocultas', { selector: '[id*="gail.discriminador"]' }), '64, 64');
    fireEvent.click(within(form).getByLabelText('normalizar_observaciones'));
    expect(within(form).getByText('3 cambiados respecto al perfil')).toBeInTheDocument();
    expect(within(form).getByText('Perfil: 0.01')).toBeInTheDocument();
    fireEvent.click(within(form).getByRole('checkbox', { name: /Ver las partidas/ }));
    fireEvent.click(within(form).getByRole('checkbox', { name: /W&B/ }));
    await pulsar(boton('Entrenar GAIL'));
    expect(api.crear).toHaveBeenCalledExactlyOnceWith({
      entorno: 'cartpole', metodo: 'gail', datos: ['congelado-cartpole'], vista: true, wandb: true,
      ajustes: { gail: { discriminador: { learning_rate: 0.02, capas_ocultas: [64, 64] } },
        trpo: { normalizar_observaciones: true } },
    }, expect.objectContaining({ senal: expect.any(AbortSignal) }));
  });

  it('la primera sección empieza abierta y las demás se abren a mano', async () => {
    const api = simulada();
    await montar(api);
    const secciones = () => [...formulario('GAIL').querySelectorAll('details')];
    expect(secciones().map((d) => d.open)).toEqual([true, false, false, false]);
    const trpo = secciones()[2];
    trpo.open = true;
    fireEvent(trpo, new Event('toggle'));
    // Cambiar un valor y deshacerlo no la cierra.
    escribir(within(trpo).getByLabelText('gamma'), '0.9');
    escribir(within(trpo).getByLabelText('gamma'), '0.995');
    expect(trpo.open).toBe(true);
    secciones()[0].open = false;
    fireEvent(secciones()[0], new Event('toggle'));
    escribir(screen.getByLabelText('iteraciones'), '30');
    expect(secciones()[0].open).toBe(false);
  });

  it('una sección con errores se abre sola al intentar empezar', async () => {
    const api = simulada();
    await montar(api);
    const discriminador = formulario('GAIL').querySelectorAll('details')[1];
    escribir(within(discriminador).getByLabelText('learning_rate'), 'x');
    expect(discriminador.open).toBe(false);
    await pulsar(boton('Entrenar GAIL'));
    expect(discriminador.open).toBe(true);
    expect(within(discriminador).getByText('Escribe un número, como 0.01 o 1e-3.')).toBeVisible();
  });

  it('al cambiar de método el formulario empieza con la primera sección del nuevo', async () => {
    const api = simulada();
    await montar(api);
    const trpo = formulario('GAIL').querySelectorAll('details')[2];
    trpo.open = true;
    fireEvent(trpo, new Event('toggle'));
    await pulsar(boton('BC'));
    expect([...formulario('BC').querySelectorAll('details')].map((d) => d.open)).toEqual([true, false]);
  });

  it('restablecer vuelve a los valores del perfil', async () => {
    const api = simulada();
    await montar(api);
    const campo = screen.getByLabelText('iteraciones');
    escribir(campo, '7');
    fireEvent.click(boton('Restablecer valores del perfil'));
    expect(campo).toHaveValue('20');
    expect(boton('Restablecer valores del perfil')).toBeDisabled();
  });

  it('sin datos o con valores que no valen no empieza y dice por qué', async () => {
    const api = simulada();
    await montar(api);
    escribir(screen.getByLabelText('iteraciones'), '0');
    escribir(screen.getByLabelText('gamma'), 'mucho');
    await pulsar(boton('Entrenar GAIL'));
    expect(api.crear).not.toHaveBeenCalled();
    expect(screen.getByText('Elige al menos un conjunto de demostraciones.')).toBeInTheDocument();
    expect(screen.getByText('Tiene que ser como mínimo 1.')).toBeInTheDocument();
    expect(screen.getByText('Escribe un número, como 0.01 o 1e-3.')).toBeInTheDocument();
    expect(screen.getByLabelText('gamma')).toHaveAttribute('aria-invalid', 'true');
    expect(screen.getByRole('alert')).toHaveTextContent('Hay valores que no son válidos');
  });

  it('con datos elegidos, un valor que no vale sigue sin dejar empezar', async () => {
    const api = simulada();
    await montar(api);
    fireEvent.click(screen.getByRole('checkbox', { name: /experto congelado/ }));
    escribir(screen.getByLabelText('iteraciones'), '-3');
    await pulsar(boton('Entrenar GAIL'));
    expect(api.crear).not.toHaveBeenCalled();
    expect(screen.queryByText(/Elige al menos un conjunto/)).toBeNull();
  });

  it('pedir la vista en GAIL y pasar a BC no la envía', async () => {
    const api = simulada();
    await montar(api);
    fireEvent.click(screen.getByRole('checkbox', { name: /Ver las partidas mientras entrena/ }));
    fireEvent.click(screen.getByRole('checkbox', { name: /experto congelado/ }));
    await pulsar(boton('BC'));
    await pulsar(boton('Entrenar BC'));
    expect(api.crear.mock.calls[0][0]).toMatchObject({ metodo: 'bc', vista: false });
    expect(screen.queryByRole('alert')).toBeNull();
  });

  it('los errores solo se muestran tras intentar empezar', async () => {
    const api = simulada();
    await montar(api);
    escribir(screen.getByLabelText('iteraciones'), '0');
    expect(screen.queryByText('Tiene que ser como mínimo 1.')).toBeNull();
    expect(screen.queryByText(/Elige al menos un conjunto/)).toBeNull();
  });

  it('si no carga el perfil lo dice y deja reintentar', async () => {
    const api = simulada();
    api.ajustes.mockRejectedValueOnce(new Error('Perfil ilegible'));
    await montar(api);
    expect(screen.getByRole('alert')).toHaveTextContent('No se pudieron cargar los valores del perfil (Perfil ilegible)');
    expect(boton('Entrenar GAIL')).toBeDisabled();
    await pulsar(boton('Reintentar'));
    expect(screen.getByLabelText('iteraciones')).toHaveValue('20');
    expect(screen.queryByRole('alert')).toBeNull();
  });

  it('la lista de datos avisa si falla o está vacía y se puede actualizar', async () => {
    const api = simulada();
    api.demostraciones.mockRejectedValueOnce(new Error('Sin biblioteca')).mockResolvedValueOnce([]);
    await montar(api);
    expect(screen.getByText('No se pudo cargar la lista (Sin biblioteca).')).toBeInTheDocument();
    await pulsar(boton('Actualizar lista'));
    expect(screen.getByText(/No hay demostraciones de este entorno/)).toBeInTheDocument();
    await pulsar(boton('Actualizar lista'));
    expect(screen.getByRole('checkbox', { name: /experto congelado/ })).toBeInTheDocument();
    expect(api.demostraciones).toHaveBeenCalledTimes(3);
  });
});

describe('en marcha', () => {
  it('muestra el progreso y la curva, se pausa, se reanuda y termina solo', async () => {
    const api = simulada();
    const ocupado = await empezarGail(api);
    expect(ocupado).toHaveBeenLastCalledWith(true);
    await avanzar(INTERVALO_SIMULADO_MS * 5);
    expect(screen.getByRole('status')).toHaveTextContent('Entrenando');
    expect(screen.getByText('5 / 20 iteraciones · 25 %')).toBeInTheDocument();
    expect(screen.getByRole('progressbar')).toHaveAttribute('aria-valuenow', '25');
    expect(screen.getByRole('img', { name: /Recompensa de GAIL: 5 puntos/ })).toBeInTheDocument();
    expect(boton('BC')).toBeDisabled();

    fireEvent.click(boton('Pausar'));
    expect(screen.getByRole('status')).toHaveTextContent('Pausando…');
    expect(boton('Pausar')).toBeDisabled();
    await avanzar(0);
    expect(screen.getByRole('status')).toHaveTextContent('Pausado');
    await avanzar(INTERVALO_SIMULADO_MS * 10);
    expect(screen.getByText('5 / 20 iteraciones · 25 %')).toBeInTheDocument();
    await pulsar(boton('Reanudar'));
    expect(screen.getByRole('status')).toHaveTextContent('Entrenando');

    await avanzar(INTERVALO_SIMULADO_MS * 20);
    expect(screen.getByText(/Modelo guardado en la biblioteca: «gail_cartpole_1»\./)).toBeInTheDocument();
    expect(screen.getByText('20 / 20 iteraciones · 100 %')).toBeInTheDocument();
    expect(ocupado).toHaveBeenLastCalledWith(false);
    expect(boton('BC')).toBeDisabled();

    await pulsar(boton('Nuevo entrenamiento'));
    expect(formulario('GAIL')).toBeInTheDocument();
    expect(boton('BC')).toBeEnabled();
  });

  it('finalizar guarda el modelo marcado como incompleto', async () => {
    const api = simulada();
    await empezarGail(api);
    await avanzar(INTERVALO_SIMULADO_MS * 2);
    fireEvent.click(boton('Finalizar'));
    expect(screen.getByRole('status')).toHaveTextContent('Terminando…');
    await avanzar(0);
    expect(screen.getByText(/«gail_cartpole_1» \(incompleto: se finalizó antes de tiempo\)/)).toBeInTheDocument();
  });

  it('la vista se enciende y se apaga en marcha', async () => {
    const api = simulada();
    await empezarGail(api, { vista: true });
    await avanzar(INTERVALO_SIMULADO_MS);
    expect(screen.getByRole('img', { name: /CartPole: última imagen/ })).toBeInTheDocument();
    fireEvent.click(screen.getByRole('checkbox', { name: 'Ver las partidas del entrenamiento' }));
    await avanzar(INTERVALO_SIMULADO_MS * 2);
    expect(screen.queryByRole('img', { name: /CartPole: última imagen/ })).toBeNull();
    fireEvent.click(screen.getByRole('checkbox', { name: 'Ver las partidas del entrenamiento' }));
    await avanzar(INTERVALO_SIMULADO_MS);
    expect(screen.getByRole('img', { name: /CartPole: última imagen/ })).toBeInTheDocument();
  });

  it('BC entrena sin vista y muestra sus pérdidas', async () => {
    const api = simulada();
    await montar(api);
    await pulsar(boton('BC'));
    fireEvent.click(screen.getByRole('checkbox', { name: /experto congelado/ }));
    await pulsar(boton('Entrenar BC'));
    await avanzar(INTERVALO_SIMULADO_MS * 2);
    expect(screen.queryByRole('checkbox', { name: /Ver las partidas/ })).toBeNull();
    const medida = screen.getByRole('combobox', { name: 'Medida' });
    expect(within(medida).getAllByRole('option').map((o) => o.textContent))
      .toEqual(['Pérdida en train', 'Pérdida en validación']);
    fireEvent.change(medida, { target: { value: 'perdida_validacion' } });
    expect(screen.getByRole('img', { name: /Pérdida en validación/ })).toBeInTheDocument();
  });

  it('un 422 al crear se muestra y deja corregir', async () => {
    const api = simulada();
    api.crear.mockRejectedValueOnce(new Error('Ajuste gail.iteraciones no válido'));
    const ocupado = await empezarGail(api);
    expect(screen.getByRole('alert')).toHaveTextContent('Ajuste gail.iteraciones no válido');
    expect(formulario('GAIL')).toBeInTheDocument();
    expect(ocupado).toHaveBeenLastCalledWith(false);
  });
});

describe('conexión perdida', () => {
  const PERDIDA = 'Se perdió la conexión con el servidor';

  async function enMarcha() {
    const doble = apiEntrenamientoConSocket();
    const ocupado = await montar(doble.api);
    fireEvent.click(screen.getByRole('checkbox', { name: /Demos/ }));
    await pulsar(boton('Entrenar GAIL'));
    const socket = doble.sockets[0];
    act(() => socket.abrir());
    return { ...doble, socket, ocupado };
  }

  function liberado(ocupado, texto) {
    expect(screen.getByRole('alert')).toHaveTextContent(texto);
    expect(screen.getByRole('status')).toHaveTextContent('Terminado');
    expect(boton('Nuevo entrenamiento')).toBeEnabled();
    expect(screen.queryByRole('button', { name: 'Finalizar' })).toBeNull();
    expect(ocupado).toHaveBeenLastCalledWith(false);
  }

  it.each([['entrenando', false], ['terminando', true]])('si se cierra %s, libera el panel',
    async (_, terminando) => {
      const { socket, ocupado } = await enMarcha();
      act(() => socket.recibir({ tipo: 'progreso', paso: 3, medidas: { m: 1 } }));
      if (terminando) {
        await pulsar(boton('Finalizar'));
      }
      act(() => socket.caer());
      liberado(ocupado, PERDIDA);
      // La curva de lo que llegó se conserva.
      expect(screen.getByRole('img', { name: /M: 1 puntos/ })).toBeInTheDocument();
    });

  it.each([['Pausar'], ['Finalizar']])('si la orden %s no se puede enviar, libera el panel', async (orden) => {
    const { socket, ocupado } = await enMarcha();
    socket.readyState = 3;
    await pulsar(boton(orden));
    liberado(ocupado, PERDIDA);
  });

  it('si la vista no se puede cambiar, libera el panel', async () => {
    const { socket, ocupado } = await enMarcha();
    socket.readyState = 2;
    await pulsar(screen.getByRole('checkbox', { name: 'Ver las partidas del entrenamiento' }));
    liberado(ocupado, PERDIDA);
  });

  it('si no confirma el final, espera el plazo justo y libera', async () => {
    const { socket, ocupado } = await enMarcha();
    await pulsar(boton('Finalizar'));
    expect(socket.enviados).toEqual([{ tipo: 'finalizar' }]);
    await avanzar(ESPERA_FIN_ENTRENO_MS - 1);
    expect(screen.getByRole('status')).toHaveTextContent('Terminando…');
    await avanzar(1);
    liberado(ocupado, 'El servidor no confirmó el final del entrenamiento.');
    act(() => socket.recibir({ tipo: 'terminado', modelo: { nombre: 'tarde', incompleto: true } }));
    expect(screen.queryByText(/tarde/)).toBeNull();
  });

  it('el cierre normal tras terminado no es una conexión perdida', async () => {
    const { socket } = await enMarcha();
    act(() => {
      socket.recibir({ tipo: 'terminado', modelo: { nombre: 'gail_1', incompleto: false } });
      socket.caer();
    });
    expect(screen.getByText(/«gail_1»/)).toBeInTheDocument();
    expect(screen.queryByRole('alert')).toBeNull();
  });

  it('un mensaje de error no termina el entrenamiento', async () => {
    const { socket } = await enMarcha();
    act(() => socket.recibir({ tipo: 'error', mensaje: 'Aviso del servidor' }));
    expect(screen.getByRole('alert')).toHaveTextContent('Aviso del servidor');
    expect(boton('Finalizar')).toBeEnabled();
  });

  it('una imagen que llega con la vista apagada no se muestra, ni queda al volver a encenderla', async () => {
    const { socket } = await enMarcha();
    const vista = () => screen.getByRole('checkbox', { name: 'Ver las partidas del entrenamiento' });
    await pulsar(vista());
    act(() => socket.recibir(new Blob(['1'])));
    expect(screen.getByRole('img', { name: /última imagen/ })).toBeInTheDocument();
    await pulsar(vista());
    act(() => socket.recibir(new Blob(['tarde'])));
    expect(screen.queryByRole('img', { name: /última imagen/ })).toBeNull();
    await pulsar(vista());
    expect(screen.queryByRole('img', { name: /última imagen/ })).toBeNull();
    expect(screen.getByText('Esperando la primera imagen…')).toBeInTheDocument();
  });

  it('las órdenes van por el socket y el estado lo confirma el servidor', async () => {
    const { socket } = await enMarcha();
    await pulsar(boton('Pausar'));
    expect(screen.getByRole('status')).toHaveTextContent('Pausando…');
    act(() => socket.recibir({ tipo: 'estado', estado: 'pausado' }));
    await pulsar(boton('Reanudar'));
    act(() => socket.recibir({ tipo: 'estado', estado: 'en_marcha' }));
    await pulsar(screen.getByRole('checkbox', { name: 'Ver las partidas del entrenamiento' }));
    expect(socket.enviados).toEqual([{ tipo: 'pausar' }, { tipo: 'reanudar' }, { tipo: 'vista', activa: true }]);
    // Con la vista encendida sigue habiendo un solo estado anunciado.
    expect(screen.getByRole('status')).toHaveTextContent('Entrenando');
  });

  it('si crear no responde, vuelve al formulario con un error', async () => {
    const doble = apiEntrenamientoConSocket();
    let senal;
    doble.http.enviar.mockImplementationOnce((_, __, opciones) => new Promise((___, rechazar) => {
      senal = opciones.senal;
      senal.addEventListener('abort', () => rechazar(new DOMException('cancelada', 'AbortError')));
    }));
    const ocupado = await montar(doble.api);
    fireEvent.click(screen.getByRole('checkbox', { name: /Demos/ }));
    await pulsar(boton('Entrenar GAIL'));
    expect(screen.getByRole('status')).toHaveTextContent('Preparando…');
    await avanzar(ESPERA_CREAR_MS - 1);
    expect(senal.aborted).toBe(false);
    await avanzar(1);
    expect(senal.aborted).toBe(true);
    expect(screen.getByRole('alert')).toHaveTextContent('El servidor no respondió al crear el entrenamiento.');
    expect(formulario('GAIL')).toBeInTheDocument();
    expect(ocupado).toHaveBeenLastCalledWith(false);
    expect(doble.sockets).toHaveLength(0);
  });

  it('desmontar a mitad de crear no se conecta ni deja temporizadores', async () => {
    const doble = apiEntrenamientoConSocket();
    let responder;
    doble.http.enviar.mockImplementationOnce(() => new Promise((resolver) => {
      responder = resolver;
    }));
    await montar(doble.api);
    fireEvent.click(screen.getByRole('checkbox', { name: /Demos/ }));
    await pulsar(boton('Entrenar GAIL'));
    cleanup();
    await act(async () => responder({ id: 'e1', duracion: { unidad: 'iteraciones', total: 20 } }));
    expect(doble.sockets).toHaveLength(0);
    expect(vi.getTimerCount()).toBe(0);
  });
});
