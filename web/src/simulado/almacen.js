/**
 * Biblioteca del backend simulado: lo que guarda «Jugar tú» o «Ver agente» y los modelos
 * que se entrenan, compartido entre las APIs simuladas para que se vean unas a otras.
 * Solo vive en la memoria del navegador: al recargar la página empieza de cero.
 */

import { ENTORNOS } from '../entornos.js';
import { CONFIGURACION_SIMULADA } from './entornos.js';

// Pares de las demostraciones congeladas cuando todas las partidas duran lo mismo:
// CartPole (15 × 500, README), Reacher (30 × 50) y HalfCheetah (30 × 1000).
const PARES_CONGELADOS = Object.freeze({ cartpole: 7500, reacher: 1500, halfcheetah: 30000 });

/**
 * @typedef {object} Almacen
 * @property {(import('../api/entrenamiento.js').Demostracion & {entorno: string})[]} demostraciones
 * @property {object[]} modelos
 * @property {(prefijo: string) => string} nuevoId
 */

/**
 * Crea la biblioteca simulada, con el experto congelado de cada entorno.
 *
 * @returns {Almacen}
 */
export function crearAlmacen() {
  let contador = 0;
  return {
    demostraciones: ENTORNOS.map(({ id, nombre }) => ({
      id: `congelado-${id}`,
      entorno: id,
      nombre: `${nombre} · experto congelado`,
      tipo: 'conjunto',
      partidas: CONFIGURACION_SIMULADA[id].partidas_experto,
      pares: PARES_CONGELADOS[id] ?? null,
      fuente: 'experto',
    })),
    modelos: [],
    nuevoId(prefijo) {
      contador += 1;
      return `${prefijo}-${contador}`;
    },
  };
}
