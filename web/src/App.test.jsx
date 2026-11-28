import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';

import App from './App.jsx';
import { ENTORNOS } from './entornos.js';
import { modelosIniciales } from './simulado/biblioteca.js';
import { crearApiLaboratorioSimulada } from './simulado/laboratorio.js';
import { VISTAS } from './vistas.js';

const navegacion = () => screen.getByRole('navigation', { name: 'Navegación principal' });
const pestana = (nombre) => within(navegacion()).getByRole('button', { name: nombre });
const selector = () => screen.getByRole('combobox', { name: 'Entorno' });
const dibujoFalso = async () => new Blob(['imagen'], { type: 'image/jpeg' });

describe('App', () => {
  it('arranca en el Laboratorio con CartPole', () => {
    render(<App />);
    expect(pestana('Laboratorio')).toHaveAttribute('aria-pressed', 'true');
    expect(pestana('Comparar')).toHaveAttribute('aria-pressed', 'false');
    expect(pestana('Biblioteca')).toHaveAttribute('aria-pressed', 'false');
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('Laboratorio de imitación');
    expect(selector()).toHaveValue('cartpole');
    expect(screen.getByRole('region', { name: 'Laboratorio' })).toBeInTheDocument();
  });

  it.each(VISTAS.map((vista) => [vista.etiqueta, vista]))(
    'la pestaña %s muestra su título y su contenido, y solo ese',
    async (_, vista) => {
      const usuario = userEvent.setup();
      render(<App vistaInicial={VISTAS.find((otra) => otra.id !== vista.id).id} />);
      await usuario.click(pestana(vista.etiqueta));

      expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent(vista.titulo);
      expect(screen.getByText(vista.subtitulo)).toBeInTheDocument();
      for (const otra of VISTAS) {
        expect(pestana(otra.etiqueta)).toHaveAttribute('aria-pressed', String(otra.id === vista.id));
      }
      expect(screen.getAllByRole('region')).toHaveLength(1);
    },
  );

  it('ofrece los cinco entornos en orden', () => {
    render(<App />);
    const opciones = within(selector()).getAllByRole('option');
    expect(opciones.map((opcion) => opcion.value)).toEqual(ENTORNOS.map((entorno) => entorno.id));
    expect(opciones[0]).toHaveTextContent('1 · CartPole');
  });

  it.each(ENTORNOS.map((entorno) => [entorno.nombre, entorno]))(
    'al elegir %s cambia la descripción del entorno',
    async (_, entorno) => {
      const usuario = userEvent.setup();
      render(<App entornoInicial={entorno.id === 'cartpole' ? 'reacher' : 'cartpole'} />);
      await usuario.selectOptions(selector(), entorno.id);

      expect(selector()).toHaveValue(entorno.id);
      const contexto = screen.getByText(entorno.objetivo).closest('#contexto-entorno');
      expect(contexto).toHaveTextContent(`Etapa ${entorno.etapa} de 5`);
      expect(contexto).toHaveTextContent(`${entorno.gymnasium} · acciones ${entorno.acciones}`);
      expect(contexto.textContent.includes('se puede jugar a mano')).toBe(entorno.jugarAMano);
    },
  );

  it('conserva el entorno al cambiar de pestaña', async () => {
    const usuario = userEvent.setup();
    render(<App />);
    await usuario.selectOptions(selector(), 'acrobot');
    for (const vista of ['Comparar', 'Biblioteca', 'Laboratorio']) {
      await usuario.click(pestana(vista));
      expect(selector()).toHaveValue('acrobot');
      expect(screen.getByText(/Acrobot-v1/)).toBeInTheDocument();
    }
  });

  it('volver a pulsar la pestaña activa no cambia nada', async () => {
    const usuario = userEvent.setup();
    render(<App vistaInicial="comparar" />);
    await usuario.click(pestana('Comparar'));
    expect(pestana('Comparar')).toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByRole('region', { name: 'Comparación de modelos' })).toBeInTheDocument();
  });

  it('se maneja con el teclado', async () => {
    const usuario = userEvent.setup();
    render(<App />);
    await usuario.tab();
    expect(pestana('Laboratorio')).toHaveFocus();
    await usuario.tab();
    await usuario.keyboard('{Enter}');
    expect(pestana('Comparar')).toHaveAttribute('aria-pressed', 'true');
  });

  it.each([
    ['una pestaña', { vistaInicial: 'resultados' }],
    ['un entorno', { entornoInicial: 'pendulum' }],
  ])('no arranca con %s desconocido', (_, props) => {
    // React informa del error por la consola además de lanzarlo.
    const consola = console.error;
    console.error = () => {};
    try {
      expect(() => render(<App {...props} />)).toThrow(RangeError);
    } finally {
      console.error = consola;
    }
  });

  it('el Laboratorio solo deja jugar a mano en CartPole y MountainCar', async () => {
    const usuario = userEvent.setup();
    render(<App apiLaboratorio={crearApiLaboratorioSimulada({ dibujar: dibujoFalso })} />);
    const jugar = () => screen.getByRole('button', { name: 'Jugar tú' });
    expect(jugar()).toBeEnabled();
    await usuario.selectOptions(selector(), 'halfcheetah');
    expect(jugar()).toBeDisabled();
    expect(screen.getByText('En HalfCheetah solo se puede ver al experto y grabar partidas con él.'))
      .toBeInTheDocument();
    await usuario.selectOptions(selector(), 'mountaincar');
    expect(jugar()).toBeEnabled();
  });

  it('con una sesión abierta no se cambia de entorno y la sesión sigue al cambiar de pestaña',
    async () => {
      const usuario = userEvent.setup();
      render(<App apiLaboratorio={crearApiLaboratorioSimulada({ dibujar: dibujoFalso })} />);
      expect(selector()).toBeEnabled();
      await usuario.click(screen.getByRole('button', { name: 'Iniciar sesión' }));
      await waitFor(() => expect(selector()).toBeDisabled());
      expect(selector()).toHaveAttribute('title', 'Termina las ejecuciones en marcha para cambiarlo');

      await usuario.click(pestana('Comparar'));
      await usuario.click(pestana('Laboratorio'));
      expect(screen.getByRole('button', { name: 'Finalizar' })).toBeEnabled();

      await usuario.click(screen.getByRole('button', { name: 'Finalizar' }));
      // Sin ninguna partida terminada no hay nada que revisar y el entorno se libera.
      await waitFor(() => expect(selector()).toBeEnabled());
      expect(screen.getByRole('alert')).toHaveTextContent('sin ninguna partida completa');
    });

  it('el entorno sigue bloqueado mientras quede algo en marcha en cualquier pestaña', async () => {
    const usuario = userEvent.setup();
    // Una comparación que se queda preparando: bloquea hasta que responda.
    const apiComparar = { modelos: async (entorno) => modelosIniciales(entorno), comparaciones: async () => [],
      crear: vi.fn(() => new Promise(() => {})), seguir: vi.fn() };
    render(<App apiLaboratorio={crearApiLaboratorioSimulada({ dibujar: dibujoFalso })} apiComparar={apiComparar} />);
    await usuario.click(screen.getByRole('button', { name: 'Iniciar sesión' }));
    await waitFor(() => expect(selector()).toBeDisabled());

    await usuario.click(pestana('Comparar'));
    await usuario.click(await screen.findByRole('checkbox', { name: /Experto congelado/ }));
    await usuario.click(screen.getByRole('button', { name: 'Iniciar comparación' }));
    expect(apiComparar.crear).toHaveBeenCalledOnce();

    await usuario.click(pestana('Laboratorio'));
    await usuario.click(screen.getByRole('button', { name: 'Finalizar' }));
    await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent('sin ninguna partida completa'));
    // La sesión ya no bloquea, pero la comparación sí.
    expect(selector()).toBeDisabled();
  });

  it('la pestaña Comparar usa la API que recibe', async () => {
    const usuario = userEvent.setup();
    const apiComparar = { modelos: vi.fn(async () => []), comparaciones: vi.fn(async () => []), crear: vi.fn(),
      seguir: vi.fn() };
    render(<App vistaInicial="comparar" entornoInicial="acrobot" apiComparar={apiComparar} />);
    expect(await screen.findByText(/No hay modelos de este entorno/)).toBeInTheDocument();
    expect(apiComparar.modelos).toHaveBeenCalledWith('acrobot');
    await usuario.selectOptions(selector(), 'reacher');
    await waitFor(() => expect(apiComparar.comparaciones).toHaveBeenLastCalledWith('reacher'));
  });

  it('la pestaña Biblioteca usa la API que recibe', async () => {
    const usuario = userEvent.setup();
    const apiBiblioteca = { modelos: vi.fn(async () => []), demostraciones: vi.fn(async () => []), elemento: vi.fn(),
      partidas: vi.fn(), crearSubconjunto: vi.fn() };
    render(<App vistaInicial="biblioteca" entornoInicial="mountaincar" apiBiblioteca={apiBiblioteca} />);
    expect(await screen.findByText('No hay modelos de MountainCar. Entrénalos en el Laboratorio.')).toBeInTheDocument();
    expect(apiBiblioteca.modelos).toHaveBeenCalledWith('mountaincar');
    await usuario.selectOptions(selector(), 'halfcheetah');
    await waitFor(() => expect(apiBiblioteca.demostraciones).toHaveBeenLastCalledWith('halfcheetah'));
  });
});
