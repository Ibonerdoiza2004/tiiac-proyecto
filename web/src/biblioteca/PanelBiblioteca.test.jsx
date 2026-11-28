import { act, fireEvent, render, screen, within } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { ErrorApi } from '../api/http.js';
import { crearAlmacen } from '../simulado/almacen.js';
import { crearApiBibliotecaSimulada } from '../simulado/biblioteca.js';
import { ESPERA_CREAR_MS } from './CrearSubconjunto.jsx';
import PanelBiblioteca from './PanelBiblioteca.jsx';

const boton = (nombre, dentro = screen) => dentro.getByRole('button', { name: nombre });
const lista = () => screen.getByRole('table', { name: /de CartPole$|de Acrobot$/ });
const filas = () => within(lista()).getAllByRole('row').slice(1).map((f) => [...f.cells].map((c) => c.textContent));
const ficha = () => Object.fromEntries([...document.querySelectorAll('.ficha > div')]
  .map((d) => [d.querySelector('dt').textContent, d.querySelector('dd').textContent]));
const casillasPartidas = () => screen.getAllByRole('checkbox', { name: /^Partida \d+$/ });

function simulada(almacen = crearAlmacen()) {
  const api = crearApiBibliotecaSimulada({ almacen });
  for (const metodo of Object.keys(api)) {
    api[metodo] = vi.fn(api[metodo]);
  }
  return api;
}

async function montar(api = simulada(), entorno = 'cartpole') {
  const vista = render(<PanelBiblioteca entorno={entorno} api={api} />);
  await screen.findByRole('table');
  return { api, ...vista };
}

async function abrirConjunto(api) {
  const montado = await montar(api);
  fireEvent.click(boton(/^Demostraciones/));
  fireEvent.click(await screen.findByRole('button', { name: 'CartPole · experto congelado' }));
  await screen.findByRole('button', { name: 'Crear un subconjunto' });
  return montado;
}

afterEach(() => {
  vi.useRealTimers();
});

describe('PanelBiblioteca: modelos', () => {
  it('lista los modelos del entorno con tipo, origen, fecha y estado', async () => {
    const { api } = await montar();
    expect(api.modelos).toHaveBeenCalledWith('cartpole');
    expect(boton('Modelos (5)')).toHaveAttribute('aria-pressed', 'true');
    expect(boton('Demostraciones (1)')).toHaveAttribute('aria-pressed', 'false');
    expect(filas()[0]).toEqual(['Experto congelado · CartPole', 'Experto (TRPO)', 'Congelado (E1)',
      expect.stringMatching(/^01\/11\/2026/), 'Válido']);
    expect(filas().map((f) => f[1])).toEqual(['Experto (TRPO)', 'GAIL', 'BC', 'GAIL', 'BC']);
    expect(screen.getByText('Elige un elemento de la lista para ver su detalle.')).toBeInTheDocument();
  });

  it('filtra por tipo y avisa si no hay ninguno', async () => {
    await montar();
    fireEvent.change(screen.getByRole('combobox', { name: 'Tipo' }), { target: { value: 'gail' } });
    expect(filas().map((f) => f[0])).toEqual(['GAIL · 1 partidas', 'GAIL · 10 partidas']);
    fireEvent.change(screen.getByRole('combobox', { name: 'Tipo' }), { target: { value: '' } });
    expect(filas()).toHaveLength(5);
  });

  it('si no hay de ese tipo lo dice', async () => {
    const api = simulada();
    api.modelos.mockResolvedValue((await crearApiBibliotecaSimulada().modelos('cartpole')).slice(0, 1));
    await montar(api);
    fireEvent.change(screen.getByRole('combobox', { name: 'Tipo' }), { target: { value: 'bc' } });
    expect(screen.getByText('No hay modelos de tipo BC de CartPole. Entrénalos en el Laboratorio.')).toBeInTheDocument();
    expect(screen.queryByRole('table')).toBeNull();
  });

  it('abre el detalle con los datos del entrenamiento', async () => {
    const { api } = await montar();
    fireEvent.click(boton('GAIL · 10 partidas'));
    expect(await screen.findByRole('heading', { name: 'GAIL · 10 partidas' })).toBeInTheDocument();
    expect(api.elemento).toHaveBeenCalledWith('gail-cartpole-10');
    expect(boton('GAIL · 10 partidas')).toHaveAttribute('aria-pressed', 'true');
    expect(ficha()).toEqual({ Tipo: 'GAIL', Origen: 'Entrega de la E1', Creado: expect.stringMatching(/^01\/11\/2026/),
      'Partidas de entrenamiento': '10', Repetición: '0', Iteraciones: '20' });
    // Los modelos no tienen partidas que listar.
    expect(api.partidas).not.toHaveBeenCalled();
    expect(screen.queryByRole('button', { name: 'Crear un subconjunto' })).toBeNull();
  });

  it('muestra los no válidos con su motivo, la huella acortada, la descripción y el artefacto', async () => {
    const api = simulada();
    const roto = { id: 'roto', entorno: 'cartpole', tipo: 'bc', nombre: 'BC roto', origen: 'dashboard', creado: null,
      descripcion: 'Probando', fuente: null, artefacto: 'bc-abc:v2', valido: false, motivo: 'falta politica.pt',
      detalles: { huella_dataset: 'f'.repeat(64), segundos: 12 } };
    api.modelos.mockResolvedValue([roto]);
    api.elemento.mockResolvedValue(roto);
    await montar(api);
    expect(filas()[0][4]).toBe('No válido');
    expect(screen.getByText('No válido')).toHaveAttribute('title', 'falta politica.pt');
    fireEvent.click(boton('BC roto'));
    expect(await screen.findByText('No válido: falta politica.pt')).toBeInTheDocument();
    expect(screen.getByText('Probando')).toBeInTheDocument();
    expect(ficha()).toMatchObject({ Creado: '—', 'Artefacto de W&B': 'bc-abc:v2',
      'Huella de los datos': 'ffffffffffff…', 'Duración del entrenamiento': '12 s' });
    expect(screen.getByText('ffffffffffff…')).toHaveAttribute('title', 'f'.repeat(64));
  });

  it('sin modelos lo dice', async () => {
    const api = simulada();
    api.modelos.mockResolvedValue([]);
    render(<PanelBiblioteca entorno="cartpole" api={api} />);
    expect(await screen.findByText('No hay modelos de CartPole. Entrénalos en el Laboratorio.')).toBeInTheDocument();
  });

  it('al cambiar de entorno no enseña la lista del anterior mientras llega la nueva', async () => {
    const { api, rerender } = await montar();
    let responder;
    api.modelos.mockImplementationOnce(() => new Promise((resolver) => {
      responder = resolver;
    }));
    rerender(<PanelBiblioteca entorno="acrobot" api={api} />);
    expect(screen.queryByRole('table')).toBeNull();
    expect(screen.getByText('Cargando…')).toBeInTheDocument();
    await act(async () => responder(await crearApiBibliotecaSimulada().modelos('acrobot')));
    expect(filas()[0][0]).toBe('Experto congelado · Acrobot');
  });

  it('si falla la lista lo dice y Actualizar la vuelve a pedir', async () => {
    const api = simulada();
    api.modelos.mockRejectedValueOnce(new Error('sin red'));
    render(<PanelBiblioteca entorno="cartpole" api={api} />);
    expect(await screen.findByRole('alert')).toHaveTextContent('No se pudo cargar la lista (sin red).');
    fireEvent.click(boton('Actualizar'));
    expect(await screen.findByRole('table')).toBeInTheDocument();
    expect(screen.queryByRole('alert')).toBeNull();
  });

  it('si falla el detalle lo dice y deja reintentar', async () => {
    const api = simulada();
    api.elemento.mockRejectedValueOnce(new ErrorApi('No existe el elemento', { estado: 404 }));
    await montar(api);
    fireEvent.click(boton('BC · 1 partidas'));
    expect(await screen.findByRole('alert')).toHaveTextContent('No se pudo cargar el elemento (No existe el elemento).');
    fireEvent.click(boton('Reintentar'));
    expect(await screen.findByRole('heading', { name: 'BC · 1 partidas' })).toBeInTheDocument();
  });

  it('una respuesta que llega tarde de otro elemento no se muestra', async () => {
    const api = simulada();
    let responder;
    api.elemento.mockImplementationOnce(() => new Promise((resolver) => {
      responder = resolver;
    }));
    await montar(api);
    fireEvent.click(boton('BC · 1 partidas'));
    fireEvent.click(boton('GAIL · 1 partidas'));
    expect(await screen.findByRole('heading', { name: 'GAIL · 1 partidas' })).toBeInTheDocument();
    await act(async () => responder({ id: 'bc-cartpole-1', tipo: 'bc', nombre: 'BC tardío', detalles: {} }));
    expect(screen.queryByText('BC tardío')).toBeNull();
  });

  it('al cambiar de entorno pide las listas de ese entorno y cierra el detalle', async () => {
    const { api, rerender } = await montar();
    fireEvent.click(boton('BC · 1 partidas'));
    await screen.findByRole('heading', { name: 'BC · 1 partidas' });
    rerender(<PanelBiblioteca entorno="acrobot" api={api} />);
    expect(await screen.findByRole('button', { name: 'Experto congelado · Acrobot' })).toBeInTheDocument();
    expect(api.modelos).toHaveBeenLastCalledWith('acrobot');
    expect(api.demostraciones).toHaveBeenLastCalledWith('acrobot');
    expect(screen.queryByRole('heading', { level: 3 })).toBeNull();
  });
});

describe('PanelBiblioteca: demostraciones', () => {
  it('lista conjuntos con partidas, pares y quién jugó, y muestra sus partidas', async () => {
    const { api } = await abrirConjunto(simulada());
    expect(filas()).toEqual([['CartPole · experto congelado', 'Conjunto', '15', '7500', 'Experto', 'Válido']]);
    expect(ficha()).toMatchObject({ Tipo: 'Conjunto', 'Quién jugó': 'Experto', Partidas: '15',
      'Pares estado-acción': '7500', 'Reparto en train y validación': 'Sí' });
    expect(api.partidas).toHaveBeenCalledWith('congelado-cartpole');
    const tabla = within(document.querySelector('.tabla-partidas table'));
    expect(tabla.getAllByRole('row')).toHaveLength(16);
    expect(tabla.getAllByText(/^Train · \d+\.ª$/)).toHaveLength(10);
    expect(tabla.getAllByText('Validación')).toHaveLength(5);
    // Solo lectura hasta que se crea un subconjunto.
    expect(screen.queryAllByRole('checkbox')).toHaveLength(0);
  });

  it('ordena las partidas por retorno o en el orden de train', async () => {
    const api = simulada();
    api.partidas.mockResolvedValue([
      { id_episodio: 0, pasos: 10, retorno: 1, split: 'validacion', orden_train: null },
      { id_episodio: 1, pasos: 10, retorno: 3, split: 'train', orden_train: 1 },
      { id_episodio: 2, pasos: 10, retorno: 2, split: 'train', orden_train: 0 },
    ]);
    await abrirConjunto(api);
    const orden = () => [...document.querySelectorAll('.tabla-partidas tbody th')].map((c) => c.textContent);
    expect(orden()).toEqual(['0', '1', '2']);
    fireEvent.change(screen.getByRole('combobox', { name: 'Ordenar' }), { target: { value: 'retorno' } });
    expect(orden()).toEqual(['1', '2', '0']);
    fireEvent.change(screen.getByRole('combobox', { name: 'Ordenar' }), { target: { value: 'train' } });
    expect(orden()).toEqual(['2', '1', '0']);
  });

  it('sin reparto no ofrece el orden de train ni «Solo las de train»', async () => {
    const api = simulada();
    api.partidas.mockResolvedValue([{ id_episodio: 0, pasos: 10, retorno: 1, split: null, orden_train: null }]);
    await abrirConjunto(api);
    expect(within(screen.getByRole('combobox', { name: 'Ordenar' })).getAllByRole('option')).toHaveLength(2);
    fireEvent.click(boton('Crear un subconjunto'));
    expect(screen.queryByRole('button', { name: 'Solo las de train' })).toBeNull();
  });

  it('si fallan las partidas lo dice y deja reintentar', async () => {
    const api = simulada();
    api.partidas.mockRejectedValueOnce(new ErrorApi('Not Found', { estado: 404 }));
    await montar(api);
    fireEvent.click(boton(/^Demostraciones/));
    fireEvent.click(await screen.findByRole('button', { name: 'CartPole · experto congelado' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('No se pudieron cargar las partidas (Not Found).');
    fireEvent.click(boton('Reintentar'));
    expect(await screen.findByRole('button', { name: 'Crear un subconjunto' })).toBeInTheDocument();
  });

  it('crea un subconjunto con las partidas elegidas, lo abre y lo añade a la lista', async () => {
    const { api } = await abrirConjunto(simulada());
    fireEvent.click(boton('Crear un subconjunto'));
    const formulario = screen.getByRole('form', { name: 'Crear subconjunto' });
    expect(within(formulario).getByRole('status')).toHaveTextContent('0 partidas elegidas · 0 pares');
    fireEvent.click(within(formulario).getByRole('checkbox', { name: 'Partida 3' }));
    fireEvent.click(within(formulario).getByRole('checkbox', { name: 'Partida 0' }));
    expect(within(formulario).getByRole('status')).toHaveTextContent('2 partidas elegidas · 1000 pares');
    fireEvent.change(screen.getByRole('textbox', { name: 'Nombre' }), { target: { value: '  Dos buenas  ' } });
    fireEvent.click(boton('Guardar subconjunto'));

    expect(await screen.findByText('Subconjunto «Dos buenas» guardado en la biblioteca.')).toBeInTheDocument();
    expect(api.crearSubconjunto).toHaveBeenCalledWith({ conjunto: 'congelado-cartpole', partidas: [0, 3],
      nombre: 'Dos buenas' }, { senal: expect.any(AbortSignal) });
    expect(await screen.findByRole('heading', { name: 'Dos buenas' })).toBeInTheDocument();
    expect(ficha()).toMatchObject({ Tipo: 'Subconjunto', Partidas: '2', 'Sale del conjunto': 'CartPole · experto congelado' });
    expect(await screen.findByRole('button', { name: 'Dos buenas' })).toHaveAttribute('aria-pressed', 'true');
    expect(boton('Demostraciones (2)')).toBeInTheDocument();
    // De un subconjunto no se saca otro.
    expect(screen.queryByRole('button', { name: 'Crear un subconjunto' })).toBeNull();

    // «Sale del conjunto» abre el conjunto.
    fireEvent.click(within(document.querySelector('.ficha')).getByRole('button', { name: 'CartPole · experto congelado' }));
    expect(await screen.findByRole('heading', { name: 'CartPole · experto congelado' })).toBeInTheDocument();
    expect(screen.queryByText(/guardado en la biblioteca/)).toBeNull();
  });

  it('Todas, Ninguna y Solo las de train cambian la elección', async () => {
    await abrirConjunto(simulada());
    fireEvent.click(boton('Crear un subconjunto'));
    const elegidas = () => casillasPartidas().filter((c) => c.checked).length;
    fireEvent.click(boton('Todas'));
    expect(elegidas()).toBe(15);
    fireEvent.click(boton('Ninguna'));
    expect(elegidas()).toBe(0);
    fireEvent.click(boton('Solo las de train'));
    expect(elegidas()).toBe(10);
    const deTrain = [...document.querySelectorAll('.tabla-partidas tbody tr')]
      .filter((f) => f.textContent.includes('Train')).map((f) => f.querySelector('input').checked);
    expect(deTrain.every(Boolean)).toBe(true);
    fireEvent.click(casillasPartidas().find((c) => c.checked));
    expect(elegidas()).toBe(9);
  });

  it('no deja guardar sin partidas o sin nombre', async () => {
    const { api } = await abrirConjunto(simulada());
    fireEvent.click(boton('Crear un subconjunto'));
    expect(screen.getByRole('textbox', { name: 'Nombre' })).toHaveValue('Subconjunto de CartPole · experto congelado');
    fireEvent.change(screen.getByRole('textbox', { name: 'Nombre' }), { target: { value: '   ' } });
    fireEvent.click(boton('Guardar subconjunto'));
    const avisos = screen.getAllByRole('alert').map((a) => a.textContent);
    expect(avisos).toEqual(['Elige al menos una partida.', 'Escribe un nombre.']);
    expect(screen.getByRole('textbox', { name: 'Nombre' })).toHaveAttribute('aria-invalid', 'true');
    expect(api.crearSubconjunto).not.toHaveBeenCalled();
  });

  it('con nombre pero sin partidas no llama al servidor', async () => {
    const { api } = await abrirConjunto(simulada());
    fireEvent.click(boton('Crear un subconjunto'));
    fireEvent.click(boton('Guardar subconjunto'));
    expect(screen.getAllByRole('alert').map((a) => a.textContent)).toEqual(['Elige al menos una partida.']);
    expect(api.crearSubconjunto).not.toHaveBeenCalled();
  });

  it('de unas demostraciones no válidas no pide las partidas ni deja crear', async () => {
    const api = simulada();
    const rotas = { id: 'rotas', entorno: 'cartpole', tipo: 'demostraciones', nombre: 'Rotas', origen: 'dashboard',
      creado: null, descripcion: null, fuente: null, artefacto: null, valido: false, motivo: 'faltan splits',
      detalles: { tipo: 'conjunto' } };
    api.demostraciones.mockResolvedValue([{ ...rotas, tipo: 'conjunto', partidas: null, pares: null },
      { ...rotas, id: 'raras', nombre: 'Raras', tipo: null, partidas: null, pares: null }]);
    api.elemento.mockResolvedValue(rotas);
    await montar(api);
    fireEvent.click(boton(/^Demostraciones/));
    fireEvent.click(await screen.findByRole('button', { name: 'Rotas' }));
    expect(await screen.findByText('No válido: faltan splits')).toBeInTheDocument();
    expect(api.partidas).not.toHaveBeenCalled();
    expect(screen.queryByRole('button', { name: 'Crear un subconjunto' })).toBeNull();
    expect(filas()).toEqual([['Rotas', 'Conjunto', '—', '—', '—', 'No válido'],
      ['Raras', '—', '—', '—', '—', 'No válido']]);
  });

  it('muestra el error del servidor y deja volver a intentarlo', async () => {
    const api = simulada();
    api.crearSubconjunto.mockRejectedValueOnce(new ErrorApi('Hay partidas repetidas', { estado: 422 }));
    await abrirConjunto(api);
    fireEvent.click(boton('Crear un subconjunto'));
    fireEvent.click(casillasPartidas()[0]);
    fireEvent.click(boton('Guardar subconjunto'));
    expect(await screen.findByRole('alert')).toHaveTextContent('Hay partidas repetidas');
    expect(boton('Guardar subconjunto')).toBeEnabled();
    fireEvent.click(boton('Guardar subconjunto'));
    expect(await screen.findByText(/guardado en la biblioteca/)).toBeInTheDocument();
  });

  it('mientras guarda no deja cambiar nada, y si no responde lo dice', async () => {
    vi.useFakeTimers();
    const api = simulada();
    api.crearSubconjunto.mockImplementationOnce((_, { senal }) => new Promise((_resolver, rechazar) => {
      senal.addEventListener('abort', () => rechazar(new DOMException('abortado', 'AbortError')));
    }));
    render(<PanelBiblioteca entorno="cartpole" api={api} />);
    await act(() => vi.advanceTimersByTimeAsync(0));
    fireEvent.click(boton(/^Demostraciones/));
    await act(() => vi.advanceTimersByTimeAsync(0));
    fireEvent.click(boton('CartPole · experto congelado'));
    await act(() => vi.advanceTimersByTimeAsync(0));
    fireEvent.click(boton('Crear un subconjunto'));
    fireEvent.click(casillasPartidas()[0]);
    fireEvent.click(boton('Guardar subconjunto'));
    await act(() => vi.advanceTimersByTimeAsync(0));
    expect(boton('Guardando…')).toBeDisabled();
    expect(casillasPartidas()[1]).toBeDisabled();
    expect(boton('Todas')).toBeDisabled();
    expect(screen.getByRole('textbox', { name: 'Nombre' })).toBeDisabled();
    await act(() => vi.advanceTimersByTimeAsync(ESPERA_CREAR_MS - 1));
    expect(screen.queryByRole('alert')).toBeNull();
    await act(() => vi.advanceTimersByTimeAsync(1));
    expect(screen.getByRole('alert')).toHaveTextContent('El servidor no respondió al crear el subconjunto.');
    expect(boton('Guardar subconjunto')).toBeEnabled();
  });

  it('Cancelar vuelve a las partidas sin crear nada', async () => {
    const { api } = await abrirConjunto(simulada());
    fireEvent.click(boton('Crear un subconjunto'));
    fireEvent.click(boton('Cancelar'));
    expect(screen.queryByRole('form', { name: 'Crear subconjunto' })).toBeNull();
    expect(boton('Crear un subconjunto')).toBeInTheDocument();
    expect(api.crearSubconjunto).not.toHaveBeenCalled();
  });

  it('elegir otro elemento cierra el formulario', async () => {
    const almacen = crearAlmacen();
    const api = simulada(almacen);
    await api.crearSubconjunto({ conjunto: 'congelado-cartpole', partidas: [1], nombre: 'Otro' });
    await abrirConjunto(api);
    fireEvent.click(boton('Crear un subconjunto'));
    fireEvent.click(boton('Otro'));
    expect(await screen.findByRole('heading', { name: 'Otro' })).toBeInTheDocument();
    expect(screen.queryByRole('form', { name: 'Crear subconjunto' })).toBeNull();
  });

  it('una respuesta que llega después de cambiar de elemento no abre nada', async () => {
    const api = simulada();
    let responder;
    api.crearSubconjunto.mockImplementationOnce(() => new Promise((resolver) => {
      responder = resolver;
    }));
    await abrirConjunto(api);
    fireEvent.click(boton('Crear un subconjunto'));
    fireEvent.click(casillasPartidas()[0]);
    fireEvent.click(boton('Guardar subconjunto'));
    fireEvent.click(boton(/^Modelos/));
    await act(async () => responder({ id: 'tarde', nombre: 'Tarde' }));
    expect(screen.queryByText(/Tarde/)).toBeNull();
    expect(screen.getByText('Elige un elemento de la lista para ver su detalle.')).toBeInTheDocument();
  });
});
