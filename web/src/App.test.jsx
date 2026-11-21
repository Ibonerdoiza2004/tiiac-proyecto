import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';

import App from './App.jsx';
import { ENTORNOS } from './entornos.js';
import { VISTAS } from './vistas.js';

const navegacion = () => screen.getByRole('navigation', { name: 'Navegación principal' });
const pestana = (nombre) => within(navegacion()).getByRole('button', { name: nombre });
const selector = () => screen.getByRole('combobox', { name: 'Entorno' });

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

  it('el Laboratorio explica cómo se graban las demostraciones según el entorno', async () => {
    const usuario = userEvent.setup();
    render(<App />);
    expect(screen.getByText('Juega a CartPole, mira al experto y graba demostraciones.')).toBeInTheDocument();
    await usuario.selectOptions(selector(), 'halfcheetah');
    expect(
      screen.getByText('En HalfCheetah, las demostraciones se graban con el experto.'),
    ).toBeInTheDocument();
  });
});
