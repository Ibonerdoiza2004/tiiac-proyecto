/**
 * Elige el backend del Laboratorio: el real (FastAPI) o el simulado en el navegador.
 *
 * Mientras no existan laboratorio-jugar-api y laboratorio-entrenar-api, el simulado es el
 * de por defecto. Con VITE_API=real (por ejemplo `VITE_API=real npm run dev`) se usa el real.
 */

import { crearAlmacen } from '../simulado/almacen.js';
import { crearApiEntrenamientoSimulada } from '../simulado/entrenamiento.js';
import { crearApiLaboratorioSimulada } from '../simulado/laboratorio.js';
import { crearApiEntrenamiento } from './entrenamiento.js';
import { crearApiLaboratorio } from './laboratorio.js';

/**
 * Crea las APIs del Laboratorio según la variable VITE_API. Las simuladas comparten
 * biblioteca: lo que se guarda al jugar aparece en los datos para entrenar.
 *
 * @param {string | undefined} [modo] 'real' o 'simulada' (por defecto, VITE_API).
 * @returns {{laboratorio: import('./laboratorio.js').ApiLaboratorio,
 *   entrenamiento: import('./entrenamiento.js').ApiEntrenamiento}}
 */
export function crearApisPorDefecto(modo = import.meta.env.VITE_API) {
  if (modo === 'real') {
    return { laboratorio: crearApiLaboratorio(), entrenamiento: crearApiEntrenamiento() };
  }
  const almacen = crearAlmacen();
  return {
    laboratorio: crearApiLaboratorioSimulada({ almacen }),
    entrenamiento: crearApiEntrenamientoSimulada({ almacen }),
  };
}
