// Preparación común de Vitest: matchers de jest-dom y limpieza del DOM entre tests.
import '@testing-library/jest-dom/vitest';
import { cleanup } from '@testing-library/react';
import { afterEach } from 'vitest';

// jsdom no implementa URL.createObjectURL, que usa la vista del entorno con cada imagen.
if (!URL.createObjectURL) {
  let siguiente = 0;
  URL.createObjectURL = () => {
    siguiente += 1;
    return `blob:prueba/${siguiente}`;
  };
  URL.revokeObjectURL = () => {};
}

afterEach(() => {
  cleanup();
});
