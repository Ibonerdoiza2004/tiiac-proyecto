/**
 * Elige el backend del dashboard: el real (FastAPI) o el simulado en el navegador.
 *
 * Mientras no existan laboratorio-jugar-api, laboratorio-entrenar-api y comparar-api, el
 * simulado es el de por defecto. Con VITE_API=real (por ejemplo `VITE_API=real npm run dev`)
 * se usa el real.
 */

import { crearAlmacen } from '../simulado/almacen.js';
import { crearApiCompararSimulada } from '../simulado/comparar.js';
import { crearApiEntrenamientoSimulada } from '../simulado/entrenamiento.js';
import { crearApiLaboratorioSimulada } from '../simulado/laboratorio.js';
import { crearApiComparar } from './comparar.js';
import { crearApiEntrenamiento } from './entrenamiento.js';
import { crearApiLaboratorio } from './laboratorio.js';

/**
 * Crea las APIs del dashboard según la variable VITE_API. Las simuladas comparten
 * biblioteca: lo que se guarda al jugar aparece en los datos para entrenar, y lo que se
 * entrena, en los modelos para comparar.
 *
 * @param {string | undefined} [modo] 'real' o 'simulada' (por defecto, VITE_API).
 * @returns {{laboratorio: import('./laboratorio.js').ApiLaboratorio,
 *   entrenamiento: import('./entrenamiento.js').ApiEntrenamiento,
 *   comparar: import('./comparar.js').ApiComparar}}
 */
export function crearApisPorDefecto(modo = import.meta.env.VITE_API) {
  if (modo === 'real') {
    return { laboratorio: crearApiLaboratorio(), entrenamiento: crearApiEntrenamiento(),
      comparar: crearApiComparar() };
  }
  const almacen = crearAlmacen();
  return {
    laboratorio: crearApiLaboratorioSimulada({ almacen }),
    entrenamiento: crearApiEntrenamientoSimulada({ almacen }),
    comparar: crearApiCompararSimulada({ almacen }),
  };
}
