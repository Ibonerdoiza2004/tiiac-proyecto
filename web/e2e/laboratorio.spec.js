import { expect, test } from '@playwright/test';

// El servidor de desarrollo usa el backend simulado (VITE_API sin definir): estas pruebas
// recorren la interfaz de verdad en Chromium, con imágenes dibujadas en un canvas.

test.beforeEach(async ({ page }) => {
  await page.goto('/');
});

const contador = (page, nombre) => page.locator('dt', { hasText: new RegExp(`^${nombre}$`) })
  .locator('xpath=following-sibling::dd');

test('jugar a CartPole con el teclado, revisar y guardar', async ({ page }) => {
  const selector = page.getByRole('combobox', { name: 'Entorno' });
  await page.getByRole('button', { name: 'Iniciar sesión' }).click();
  await expect(page.getByRole('status').first()).toHaveText('Esperando tecla');
  await expect(selector).toBeDisabled();

  const imagen = page.getByRole('img', { name: 'CartPole: última imagen de la partida' });
  await expect(imagen).toBeVisible();
  expect(await imagen.evaluate((img) => img.naturalWidth)).toBeGreaterThan(0);

  await page.keyboard.down('ArrowRight');
  await expect(page.getByRole('status').first()).toHaveText('En marcha');
  await expect(contador(page, 'Partidas terminadas')).not.toHaveText('0', { timeout: 10_000 });
  await page.keyboard.up('ArrowRight');
  await expect(page.getByRole('status').first()).toHaveText('Esperando tecla');

  await page.getByRole('button', { name: 'Finalizar' }).click();
  const revision = page.getByRole('form', { name: 'Revisar partidas' });
  await expect(revision).toBeVisible();
  await revision.getByRole('textbox', { name: 'Nombre del conjunto' }).fill('cartpole_e2e');
  await revision.getByRole('button', { name: 'Guardar en la biblioteca' }).click();
  await expect(page.getByText(/Guardado «cartpole_e2e»: \d+ partidas? y \d+ pares?/)).toBeVisible();
  await expect(selector).toBeEnabled();
});

test('ver al experto de Reacher, revisar y descartar', async ({ page }) => {
  await page.getByRole('combobox', { name: 'Entorno' }).selectOption('reacher');
  await expect(page.getByRole('button', { name: 'Jugar tú' })).toBeDisabled();
  const partidas = page.getByRole('spinbutton', { name: 'Partidas del experto' });
  await expect(partidas).toHaveValue('30');
  await partidas.fill('2');
  await page.getByRole('button', { name: 'Ver al experto' }).click();
  const revision = page.getByRole('form', { name: 'Revisar partidas' });
  await expect(revision).toBeVisible({ timeout: 10_000 });
  await expect(revision.getByRole('checkbox')).toHaveCount(2);
  await expect(revision.getByText('Se guardarán 2 partidas y 100 pares.')).toBeVisible();
  await revision.getByRole('button', { name: 'Descartar sesión' }).click();
  await expect(page.getByRole('button', { name: 'Ver al experto' })).toBeEnabled();
});

test('la sesión sigue al cambiar de pestaña', async ({ page }) => {
  await page.getByRole('button', { name: 'Iniciar sesión' }).click();
  await expect(page.getByRole('status').first()).toHaveText('Esperando tecla');
  const navegacion = page.getByRole('navigation', { name: 'Navegación principal' });
  await navegacion.getByRole('button', { name: 'Comparar' }).click();
  await navegacion.getByRole('button', { name: 'Laboratorio' }).click();
  await expect(page.getByRole('button', { name: 'Finalizar' })).toBeEnabled();
});
