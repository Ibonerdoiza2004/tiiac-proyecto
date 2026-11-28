import { expect, test } from '@playwright/test';

// Con el backend simulado (VITE_API sin definir): los modelos y las partidas son inventados,
// pero el reparto en train y validación sale de configs/ y el recorrido es el real.

const navegacion = (page) => page.getByRole('navigation', { name: 'Navegación principal' });
const region = (page) => page.getByRole('region', { name: 'Biblioteca' });

test.beforeEach(async ({ page }) => {
  await page.goto('/');
  await navegacion(page).getByRole('button', { name: 'Biblioteca' }).click();
});

test('ver los modelos del entorno y el detalle de uno', async ({ page }) => {
  await page.getByRole('combobox', { name: 'Entorno' }).selectOption('acrobot');
  const biblioteca = region(page);
  await biblioteca.getByRole('combobox', { name: 'Tipo' }).selectOption('bc');
  await expect(biblioteca.getByRole('table').getByRole('row')).toHaveCount(3);
  await biblioteca.getByRole('button', { name: 'BC · 10 partidas' }).click();
  await expect(biblioteca.getByRole('heading', { name: 'BC · 10 partidas' })).toBeVisible();
  await expect(biblioteca.locator('.ficha')).toContainText('Partidas de entrenamiento10');
  await expect(biblioteca.locator('.ficha')).toContainText('OrigenEntrega de la E1');
});

test('crear un subconjunto y entrenar con él en el Laboratorio', async ({ page }) => {
  const biblioteca = region(page);
  await biblioteca.getByRole('button', { name: /^Demostraciones/ }).click();
  await biblioteca.getByRole('button', { name: 'CartPole · experto congelado' }).click();
  await biblioteca.getByRole('button', { name: 'Crear un subconjunto' }).click();
  await biblioteca.getByRole('combobox', { name: 'Ordenar' }).selectOption('train');
  for (const partida of await biblioteca.getByRole('checkbox', { name: /^Partida \d+$/ }).all().then((c) => c.slice(0, 3))) {
    await partida.check();
  }
  await expect(biblioteca.getByText('3 partidas elegidas · 1500 pares')).toBeVisible();
  await biblioteca.getByRole('textbox', { name: 'Nombre' }).fill('Tres primeras de train');
  await biblioteca.getByRole('button', { name: 'Guardar subconjunto' }).click();
  await expect(biblioteca.getByText('Subconjunto «Tres primeras de train» guardado en la biblioteca.')).toBeVisible();
  await expect(biblioteca.getByRole('heading', { name: 'Tres primeras de train' })).toBeVisible();
  await expect(biblioteca.locator('.tabla-partidas tbody tr')).toHaveCount(3);

  await navegacion(page).getByRole('button', { name: 'Laboratorio' }).click();
  const imitar = page.getByRole('article').filter({ has: page.getByRole('heading', { name: 'Entrenar e imitar' }) });
  await imitar.getByRole('button', { name: 'Actualizar lista' }).click();
  await expect(imitar.getByRole('checkbox', { name: /Tres primeras de train/ })).toBeVisible();
  await expect(imitar.getByText(/subconjunto · 3 partidas · 1500 pares/)).toBeVisible();
});

test('no deja guardar un subconjunto sin partidas', async ({ page }) => {
  const biblioteca = region(page);
  await biblioteca.getByRole('button', { name: /^Demostraciones/ }).click();
  await biblioteca.getByRole('button', { name: 'CartPole · experto congelado' }).click();
  await biblioteca.getByRole('button', { name: 'Crear un subconjunto' }).click();
  await biblioteca.getByRole('button', { name: 'Guardar subconjunto' }).click();
  await expect(biblioteca.getByText('Elige al menos una partida.')).toBeVisible();
  await expect(biblioteca.getByRole('button', { name: /^Demostraciones \(1\)/ })).toBeVisible();
});

test('se ve bien a ancho de móvil', async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 800 });
  const biblioteca = region(page);
  await biblioteca.getByRole('button', { name: /^Demostraciones/ }).click();
  await biblioteca.getByRole('button', { name: 'CartPole · experto congelado' }).click();
  await biblioteca.getByRole('button', { name: 'Crear un subconjunto' }).click();
  await expect(biblioteca.getByRole('form', { name: 'Crear subconjunto' })).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(375);
});
