/**
 * Cliente de la pestaña Comparar.
 *
 * Usa lo que ya existe en la API (src/app): los modelos de la biblioteca con su
 * historial de entrenamiento (GET /api/biblioteca/modelos), las comparaciones anteriores
 * (GET /api/trabajos?tipo=comparacion) y el seguimiento en directo por el protocolo
 * único de trabajos (./trabajos.js), con las órdenes vista y finalizar que el contrato ya
 * reserva para el tipo comparacion.
 *
 * Lo único que todavía no existe es crear la comparación, que llega con comparar-api.
 * Lo que espera esta interfaz, pendiente de acordar con el backend:
 *
 * - POST /api/comparaciones {entorno, modelos, aleatorio, partidas?, vista} → Trabajo de
 *   tipo comparacion, con los valores efectivos en argumentos (partidas y semilla). Sin
 *   partidas, las de la evaluación de la E1 (evaluacion.episodios); semilla y
 *   determinismo, los de evaluacion, para que todas las comparaciones sean comparables.
 * - Progreso, una vez por partida terminada de cada modelo:
 *   {medidas: {modelo, partida, retorno, pasos}}, con modelo = id de la biblioteca o
 *   «aleatorio» y partida desde 0. Todos los modelos juegan la partida i con la misma
 *   semilla antes de pasar a la i + 1, para que los resultados provisionales sean
 *   comparables en todo momento.
 * - resultado al terminar: {partidas, semilla, incompleta, modelos: [{id, retornos}]}.
 *   La interfaz calcula media y desviación a partir de los retornos.
 */

import { crearClienteHttp } from './http.js';
import { crearApiTrabajos } from './trabajos.js';

/** Id con el que se identifica el agente aleatorio en el progreso y el resultado. */
export const ALEATORIO = 'aleatorio';

/** Máximo de partidas por modelo que acepta la interfaz (propuesta, como al grabar). */
export const MAXIMO_PARTIDAS = 1000;

/** Tipos de modelo que se pueden comparar: los de la biblioteca. */
export { TIPOS_MODELO } from './biblioteca.js';

/**
 * @typedef {object} ElementoBiblioteca
 * @property {string} id
 * @property {string} entorno
 * @property {'experto' | 'bc' | 'gail'} tipo
 * @property {string} nombre
 * @property {'congelado' | 'entrega' | 'dashboard' | null} origen
 * @property {string | null} creado Fecha ISO.
 * @property {boolean} valido
 * @property {string | null} motivo Por qué no es válido.
 * @property {Record<string, unknown>} detalles Lo principal de su entrenamiento (pasos,
 *   iteraciones, segundos, subconjunto o cantidad, repetición…).
 */

/**
 * @typedef {object} PeticionComparacion
 * @property {string} entorno
 * @property {string[]} modelos Ids de la biblioteca.
 * @property {boolean} aleatorio Si se añade el agente aleatorio como referencia.
 * @property {number} [partidas] Por modelo; sin ella, las de la evaluación de la E1.
 * @property {boolean} vista Dibujar las partidas.
 */

/**
 * @typedef {object} ApiComparar
 * @property {(entorno: string) => Promise<ElementoBiblioteca[]>} modelos
 * @property {(entorno: string) => Promise<import('./trabajos.js').Trabajo[]>} comparaciones
 *   Las de este entorno, de la más reciente a la más antigua.
 * @property {(peticion: PeticionComparacion, opciones?: {senal?: AbortSignal})
 *   => Promise<import('./trabajos.js').Trabajo>} crear
 * @property {(id: string, callbacks: import('./trabajos.js').CallbacksTrabajo)
 *   => import('./trabajos.js').ConexionTrabajo} seguir
 */

/**
 * Comparaciones de un entorno, de la más reciente a la más antigua.
 *
 * @param {import('./trabajos.js').Trabajo[]} trabajos
 * @param {string} entorno
 * @returns {import('./trabajos.js').Trabajo[]}
 */
export function comparacionesDe(trabajos, entorno) {
  return trabajos
    .filter((t) => t.tipo === 'comparacion' && t.argumentos?.entorno === entorno)
    .sort((a, b) => b.creado - a.creado);
}

/**
 * Crea la API de Comparar sobre el backend real.
 *
 * @param {object} [opciones]
 * @param {import('./http.js').ClienteHttp} [opciones.http]
 * @param {import('./trabajos.js').ApiTrabajos} [opciones.trabajos]
 * @returns {ApiComparar}
 */
export function crearApiComparar({ http = crearClienteHttp(), trabajos = crearApiTrabajos({ http }) } = {}) {
  return {
    modelos: (entorno) => http.obtener('/biblioteca/modelos', { parametros: { entorno } }),
    comparaciones: async (entorno) => comparacionesDe(await trabajos.listar({ tipo: 'comparacion' }), entorno),
    crear: (peticion, { senal } = {}) => http.enviar('/comparaciones', peticion, { senal }),
    seguir: (id, callbacks) => trabajos.seguir(id, callbacks),
  };
}
