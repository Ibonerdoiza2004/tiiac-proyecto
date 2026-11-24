/**
 * Pestañas principales del dashboard, con el título y el subtítulo de cada una
 * (textos del prototipo del dashboard).
 *
 * @typedef {object} Vista
 * @property {'laboratorio' | 'comparar' | 'biblioteca'} id Clave de la pestaña.
 * @property {string} etiqueta Texto del botón de la pestaña.
 * @property {string} titulo Encabezado de la página.
 * @property {string} subtitulo Frase bajo el encabezado.
 */

/** @type {readonly Vista[]} */
export const VISTAS = Object.freeze(
  [
    {
      id: 'laboratorio',
      etiqueta: 'Laboratorio',
      titulo: 'Laboratorio de imitación',
      subtitulo: 'Juega, crea demostraciones y observa cómo aprende tu agente.',
    },
    {
      id: 'comparar',
      etiqueta: 'Comparar',
      titulo: 'Compara lo que han aprendido.',
      subtitulo: 'Observa sus decisiones y pon los resultados en contexto.',
    },
    {
      id: 'biblioteca',
      etiqueta: 'Biblioteca',
      titulo: 'Cada experimento, en su sitio.',
      subtitulo: 'Recupera modelos, selecciona demostraciones y prepara tus datos.',
    },
  ].map((vista) => Object.freeze(vista)),
);

/** Pestaña con la que arranca el dashboard. */
export const VISTA_INICIAL = 'laboratorio';

/**
 * Busca una pestaña por su clave.
 *
 * @param {string} id Clave de la pestaña.
 * @returns {Vista} La pestaña.
 * @throws {RangeError} Si no existe.
 */
export function buscarVista(id) {
  const vista = VISTAS.find((candidata) => candidata.id === id);
  if (!vista) {
    throw new RangeError(`Vista desconocida: ${String(id)}`);
  }
  return vista;
}
