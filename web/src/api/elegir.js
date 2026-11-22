/**
 * Elige el backend del Laboratorio: el real (FastAPI) o el simulado en el navegador.
 *
 * Mientras no exista laboratorio-jugar-api, el simulado es el de por defecto. Con
 * VITE_API=real (por ejemplo `VITE_API=real npm run dev`) se usa el real.
 */

import { crearApiLaboratorioSimulada } from '../simulado/laboratorio.js';
import { crearApiLaboratorio } from './laboratorio.js';

/**
 * Crea la API del Laboratorio según la variable VITE_API.
 *
 * @param {string | undefined} [modo] 'real' o 'simulada' (por defecto, VITE_API).
 * @returns {import('./laboratorio.js').ApiLaboratorio}
 */
export function crearApiPorDefecto(modo = import.meta.env.VITE_API) {
  return modo === 'real' ? crearApiLaboratorio() : crearApiLaboratorioSimulada();
}
