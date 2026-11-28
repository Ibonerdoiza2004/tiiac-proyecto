/**
 * Cliente de la pestaña Biblioteca.
 *
 * Usa lo que ya existe en la API (src/app/rutas_biblioteca.py):
 *
 * - GET /api/biblioteca/modelos?entorno=   expertos, BC y GAIL (ElementoBibliotecaApi).
 * - GET /api/biblioteca/demostraciones?entorno=   conjuntos y subconjuntos (DemostracionApi).
 * - GET /api/biblioteca/{id}   un elemento cualquiera con sus detalles; 404 si no existe.
 *
 * Lo que todavía no existe llega con biblioteca-api. Lo que espera esta interfaz,
 * pendiente de acordar con el backend:
 *
 * - GET /api/biblioteca/{id}/partidas   partidas de unas demostraciones (conjunto o
 *   subconjunto), en orden de id_episodio: [{id_episodio, pasos, retorno, split,
 *   orden_train}]. split es 'train', 'validacion' o null si el conjunto no tiene splits
 *   (en un subconjunto, siempre null); orden_train, la posición de la partida en train
 *   (desde 0; BC y GAIL con N partidas usan las N primeras) o null. Sale de
 *   resumen_demostraciones, retornos_por_partida y los splits del conjunto.
 * - POST /api/biblioteca/subconjuntos {conjunto, partidas, nombre}   crea un subconjunto
 *   con esas partidas enteras (id_episodio, sin repetir) de un conjunto válido (no de otro
 *   subconjunto), con crear_subconjunto de src/data/seleccion.py; responde 201 con el
 *   elemento nuevo (ElementoBibliotecaApi) y 422 si algo no vale. nombre como
 *   NombreUsuario: de 1 a 100 caracteres, alguno visible.
 */

import { crearClienteHttp } from './http.js';

/** Tipos de modelo de la biblioteca, en el orden en que se muestran. */
export const TIPOS_MODELO = Object.freeze(['experto', 'gail', 'bc']);

/** Largo máximo del nombre de un subconjunto (NombreUsuario en src/app/contrato.py). */
export const LARGO_NOMBRE = 100;

/**
 * @typedef {object} ElementoBiblioteca ElementoBibliotecaApi de src/app/contrato.py.
 * @property {string} id
 * @property {string} entorno
 * @property {'experto' | 'bc' | 'gail' | 'demostraciones'} tipo
 * @property {string} nombre
 * @property {'congelado' | 'entrega' | 'dashboard' | null} origen
 * @property {string | null} creado Fecha ISO.
 * @property {string | null} descripcion
 * @property {'experto' | 'humano' | 'combinado' | null} fuente Solo en demostraciones.
 * @property {string | null} artefacto Artefacto de W&B (nombre:versión).
 * @property {boolean} valido
 * @property {string | null} motivo
 * @property {Record<string, unknown>} detalles
 */

/**
 * @typedef {object} Demostracion DemostracionApi de src/app/contrato.py.
 * @property {string} id
 * @property {string} entorno
 * @property {string} nombre
 * @property {'conjunto' | 'subconjunto'} tipo
 * @property {number | null} partidas
 * @property {number | null} pares
 * @property {'experto' | 'humano' | 'combinado' | null} fuente
 * @property {'congelado' | 'entrega' | 'dashboard' | null} origen
 * @property {boolean} valido
 * @property {string | null} motivo
 */

/**
 * @typedef {object} Partida
 * @property {number} id_episodio
 * @property {number} pasos
 * @property {number} retorno
 * @property {'train' | 'validacion' | null} split
 * @property {number | null} orden_train
 */

/**
 * @typedef {object} ApiBiblioteca
 * @property {(entorno: string) => Promise<ElementoBiblioteca[]>} modelos
 * @property {(entorno: string) => Promise<Demostracion[]>} demostraciones
 * @property {(id: string) => Promise<ElementoBiblioteca>} elemento
 * @property {(id: string) => Promise<Partida[]>} partidas
 * @property {(peticion: {conjunto: string, partidas: number[], nombre: string},
 *   opciones?: {senal?: AbortSignal}) => Promise<ElementoBiblioteca>} crearSubconjunto
 */

/**
 * Crea la API de la Biblioteca sobre el backend real.
 *
 * @param {object} [opciones]
 * @param {import('./http.js').ClienteHttp} [opciones.http]
 * @returns {ApiBiblioteca}
 */
export function crearApiBiblioteca({ http = crearClienteHttp() } = {}) {
  const ruta = (id) => `/biblioteca/${encodeURIComponent(id)}`;
  return {
    modelos: (entorno) => http.obtener('/biblioteca/modelos', { parametros: { entorno } }),
    demostraciones: (entorno) => http.obtener('/biblioteca/demostraciones', { parametros: { entorno } }),
    elemento: (id) => http.obtener(ruta(id)),
    partidas: (id) => http.obtener(`${ruta(id)}/partidas`),
    crearSubconjunto: (peticion, { senal } = {}) => http.enviar('/biblioteca/subconjuntos', peticion, { senal }),
  };
}
