import { expect, test } from '@playwright/test';

test.beforeEach(async ({ page }) => {
  await page.goto('/');
});

test('arranca en el Laboratorio con CartPole', async ({ page }) => {
  await expect(page).toHaveTitle('GAIL lab');
  await expect(page.getByRole('heading', { level: 1 })).toHaveText('Laboratorio de imitación');
  await expect(page.getByRole('combobox', { name: 'Entorno' })).toHaveValue('cartpole');
  await expect(page.getByText('CartPole-v1 · acciones discretas')).toBeVisible();
});

test('cambia de pestaña y conserva el entorno elegido', async ({ page }) => {
  const navegacion = page.getByRole('navigation', { name: 'Navegación principal' });
  await page.getByRole('combobox', { name: 'Entorno' }).selectOption('reacher');

  for (const [pestana, titulo] of [
    ['Comparar', 'Compara lo que han aprendido.'],
    ['Biblioteca', 'Cada experimento, en su sitio.'],
    ['Laboratorio', 'Laboratorio de imitación'],
  ]) {
    await navegacion.getByRole('button', { name: pestana }).click();
    await expect(navegacion.getByRole('button', { name: pestana })).toHaveAttribute(
      'aria-pressed',
      'true',
    );
    await expect(page.getByRole('heading', { level: 1 })).toHaveText(titulo);
    await expect(page.getByText('Reacher-v5 · acciones continuas')).toBeVisible();
  }
});

test('en un móvil no hay desplazamiento horizontal', async ({ page }) => {
  await page.setViewportSize({ width: 360, height: 740 });
  const sobra = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
  expect(sobra).toBeLessThanOrEqual(0);
});
