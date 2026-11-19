import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';

import SelectorEntorno from './SelectorEntorno.jsx';

describe('SelectorEntorno', () => {
  it('avisa con la clave del entorno elegido', async () => {
    const usuario = userEvent.setup();
    const alCambiarEntorno = vi.fn();
    render(<SelectorEntorno entorno="cartpole" alCambiarEntorno={alCambiarEntorno} />);
    await usuario.selectOptions(screen.getByRole('combobox', { name: 'Entorno' }), 'reacher');
    expect(alCambiarEntorno).toHaveBeenCalledExactlyOnceWith('reacher');
  });

  it('bloqueado, no deja cambiar el entorno y explica por qué', async () => {
    const usuario = userEvent.setup();
    const alCambiarEntorno = vi.fn();
    render(<SelectorEntorno entorno="cartpole" alCambiarEntorno={alCambiarEntorno} bloqueado />);
    const campo = screen.getByRole('combobox', { name: 'Entorno' });
    expect(campo).toBeDisabled();
    expect(campo).toHaveAttribute('title', 'Termina las ejecuciones en marcha para cambiarlo');
    await usuario.selectOptions(campo, 'reacher').catch(() => {});
    expect(alCambiarEntorno).not.toHaveBeenCalled();
  });

  it('sin bloquear, no lleva aviso', () => {
    render(<SelectorEntorno entorno="mountaincar" alCambiarEntorno={() => {}} />);
    const campo = screen.getByRole('combobox', { name: 'Entorno' });
    expect(campo).toBeEnabled();
    expect(campo).not.toHaveAttribute('title');
    expect(campo).toHaveValue('mountaincar');
  });
});
