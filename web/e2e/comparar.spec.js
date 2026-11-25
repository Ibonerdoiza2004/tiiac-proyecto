import { expect, test } from '@playwright/test';

// Con el backend simulado (VITE_API sin definir): los retornos son inventados, pero las
// partidas y la semilla por defecto son las de configs/ y el recorrido de la interfaz es
// el real, con el mismo protocolo de trabajos que la API.

test.beforeEach(async ({ page }) => {
  await page.goto('/');
  await page.getByRole('navigation', { name: 'Navegación principal' }).getByRole('button', { name: 'Comparar' }).click();
});

const region = (page) => page.getByRole('region', { name: 'Comparación de modelos' });

test('comparar el experto, un BC y el aleatorio de principio a fin', async ({ page }) => {
  const comparar = region(page);
  await comparar.getByRole('checkbox', { name: /Experto congelado/ }).check();
  await comparar.getByRole('checkbox', { name: /BC · 1 partidas/ }).check();
  await comparar.getByRole('spinbutton', { name: 'Partidas por modelo' }).fill('5');
  await comparar.getByRole('button', { name: 'Iniciar comparación' }).click();

  const selector = page.getByRole('combobox', { name: 'Entorno' });
  await expect(selector).toBeDisabled();
  await expect(comparar.getByRole('status')).toHaveText('Terminada', { timeout: 10_000 });
  await expect(selector).toBeEnabled();
  await expect(comparar.getByText(/^5 de 5 partidas/)).toBeVisible();

  const tabla = comparar.getByRole('table', { name: 'Resultados por modelo, de mejor a peor media' });
  await expect(tabla.getByRole('row')).toHaveCount(4);
  await expect(tabla.getByRole('row').nth(1)).toContainText('Experto congelado');
  await expect(tabla.getByRole('row').nth(3)).toContainText('Aleatorio');
  await comparar.getByText('Puntuación de cada partida').click();
  await expect(comparar.locator('.detalle-partidas tbody tr')).toHaveCount(5);

  await comparar.getByRole('button', { name: 'Nueva comparación' }).click();
  await comparar.getByRole('button', { name: 'Ver' }).click();
  await expect(tabla.getByRole('row').nth(1)).toContainText('Experto congelado');
});

test('ver las partidas, pausar y finalizar antes de tiempo', async ({ page }) => {
  await page.getByRole('combobox', { name: 'Entorno' }).selectOption('mountaincar');
  const comparar = region(page);
  await comparar.getByRole('checkbox', { name: /GAIL · 10 partidas/ }).check();
  await comparar.getByRole('checkbox', { name: /Ver las partidas mientras compara/ }).check();
  await comparar.getByRole('button', { name: 'Iniciar comparación' }).click();
  await expect(comparar.getByText(/50 partidas · semilla 9000/)).toBeVisible();

  const imagen = comparar.getByRole('img', { name: 'MountainCar: última imagen de la partida' });
  await expect(imagen).toBeVisible();
  expect(await imagen.evaluate((img) => img.naturalWidth)).toBeGreaterThan(0);
  await comparar.getByRole('button', { name: 'Pausar' }).click();
  await expect(comparar.getByRole('status')).toHaveText('Pausada');
  await comparar.getByRole('button', { name: 'Finalizar' }).click();
  await expect(comparar.getByRole('status')).toHaveText('Terminada');
  await expect(comparar.getByText(/finalizada antes de tiempo/)).toBeVisible();
  await expect(imagen).toHaveCount(0);
});

test('lo entrenado en el Laboratorio aparece para comparar', async ({ page }) => {
  await page.getByRole('navigation', { name: 'Navegación principal' }).getByRole('button', { name: 'Laboratorio' }).click();
  const explorar = page.getByRole('article').filter({ has: page.getByRole('heading', { name: 'Explorar y generar' }) });
  await explorar.getByRole('button', { name: 'Entrenar experto' }).click();
  await explorar.getByRole('button', { name: 'Entrenar Experto (TRPO)' }).click();
  await explorar.getByRole('button', { name: 'Finalizar' }).click();
  await expect(explorar.getByText(/Modelo guardado en la biblioteca: «experto_cartpole_1»/)).toBeVisible();

  await page.getByRole('navigation', { name: 'Navegación principal' }).getByRole('button', { name: 'Comparar' }).click();
  const comparar = region(page);
  await comparar.getByRole('button', { name: 'Actualizar la biblioteca' }).click();
  await expect(comparar.getByRole('checkbox', { name: /experto_cartpole_1.*del dashboard/ })).toBeEnabled();
});

test('no deja comparar un solo modelo ni partidas que no valen', async ({ page }) => {
  const comparar = region(page);
  await comparar.getByRole('checkbox', { name: /Aleatorio/ }).uncheck();
  await comparar.getByRole('checkbox', { name: /Experto congelado/ }).check();
  await comparar.getByRole('spinbutton', { name: 'Partidas por modelo' }).fill('0');
  await comparar.getByRole('button', { name: 'Iniciar comparación' }).click();
  await expect(comparar.getByText('Elige al menos dos participantes', { exact: false })).toBeVisible();
  await expect(comparar.getByText(/entre 1 y 1000/)).toBeVisible();
  await expect(comparar.getByRole('progressbar')).toHaveCount(0);
});

test('se ve bien a ancho de móvil', async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 800 });
  const comparar = region(page);
  await comparar.getByRole('checkbox', { name: /Experto congelado/ }).check();
  await comparar.getByRole('spinbutton', { name: 'Partidas por modelo' }).fill('2');
  await comparar.getByRole('button', { name: 'Iniciar comparación' }).click();
  await expect(comparar.getByRole('status')).toHaveText('Terminada', { timeout: 10_000 });
  const ancho = await page.evaluate(() => document.documentElement.scrollWidth);
  expect(ancho).toBeLessThanOrEqual(375);
});
