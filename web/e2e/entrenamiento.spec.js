import { expect, test } from '@playwright/test';

// Con el backend simulado (VITE_API sin definir): las curvas son inventadas, pero los
// hiperparámetros son los de configs/ y el recorrido de la interfaz es el real.

test.beforeEach(async ({ page }) => {
  await page.goto('/');
});

const panel = (page, titulo) => page.getByRole('article').filter({
  has: page.getByRole('heading', { name: titulo }) });

test('entrenar BC con lo grabado jugando, pausar, reanudar y finalizar', async ({ page }) => {
  const explorar = panel(page, 'Explorar y generar');
  const imitar = panel(page, 'Entrenar e imitar');

  await explorar.getByRole('button', { name: 'Iniciar sesión' }).click();
  await page.keyboard.down('ArrowRight');
  await expect(explorar.locator('dt', { hasText: /^Partidas terminadas$/ })
    .locator('xpath=following-sibling::dd')).not.toHaveText('0', { timeout: 10_000 });
  await page.keyboard.up('ArrowRight');
  await explorar.getByRole('button', { name: 'Finalizar' }).click();
  await explorar.getByRole('textbox', { name: 'Nombre del conjunto' }).fill('mis_partidas');
  await explorar.getByRole('button', { name: 'Guardar en la biblioteca' }).click();
  await expect(explorar.getByText(/Guardado «mis_partidas»/)).toBeVisible();

  await imitar.getByRole('button', { name: 'BC', exact: true }).click();
  await imitar.getByRole('button', { name: 'Actualizar lista' }).click();
  await imitar.getByRole('checkbox', { name: /mis_partidas/ }).check();
  await expect(imitar.getByText(/jugadas a mano/)).toBeVisible();
  await imitar.getByLabel('learning_rate').fill('0,002');
  await expect(imitar.getByText('1 cambiados respecto al perfil')).toBeVisible();
  await imitar.getByRole('button', { name: 'Entrenar BC' }).click();

  const selector = page.getByRole('combobox', { name: 'Entorno' });
  await expect(selector).toBeDisabled();
  await expect(imitar.getByRole('img', { name: /Pérdida en train/ })).toBeVisible();
  await imitar.getByRole('button', { name: 'Pausar' }).click();
  await expect(imitar.getByRole('status')).toHaveText('Pausado');
  await imitar.getByRole('button', { name: 'Reanudar' }).click();
  await expect(imitar.getByRole('status')).toHaveText('Entrenando');
  await imitar.getByRole('button', { name: 'Finalizar' }).click();
  await expect(imitar.getByText(/Modelo guardado en la biblioteca: «bc_cartpole_1» \(incompleto/)).toBeVisible();
  await expect(selector).toBeEnabled();
});

test('entrenar el experto con la vista encendida y apagarla en marcha', async ({ page }) => {
  await page.getByRole('combobox', { name: 'Entorno' }).selectOption('mountaincar');
  const explorar = panel(page, 'Explorar y generar');
  await explorar.getByRole('button', { name: 'Entrenar experto' }).click();
  await expect(explorar.getByLabel('retorno_objetivo')).toHaveValue('-90');
  await explorar.getByRole('checkbox', { name: /Ver las partidas mientras entrena/ }).check();
  await explorar.getByRole('button', { name: 'Entrenar Experto (TRPO)' }).click();

  const imagen = explorar.getByRole('img', { name: 'MountainCar: última imagen de la partida' });
  await expect(imagen).toBeVisible();
  expect(await imagen.evaluate((img) => img.naturalWidth)).toBeGreaterThan(0);
  await explorar.getByRole('checkbox', { name: 'Ver las partidas del entrenamiento' }).uncheck();
  await expect(imagen).toHaveCount(0);
  await expect(explorar.getByText(/Modelo guardado en la biblioteca: «experto_mountaincar_1»/))
    .toBeVisible({ timeout: 15_000 });
});

test('no deja entrenar sin datos ni con valores que no valen', async ({ page }) => {
  const imitar = panel(page, 'Entrenar e imitar');
  await imitar.getByLabel('iteraciones').fill('0');
  await imitar.getByRole('button', { name: 'Entrenar GAIL' }).click();
  await expect(imitar.getByText('Elige al menos un conjunto de demostraciones.')).toBeVisible();
  await expect(imitar.getByText('Tiene que ser como mínimo 1.')).toBeVisible();
  await expect(imitar.getByRole('progressbar')).toHaveCount(0);
});
