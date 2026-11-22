/**
 * Revisión de las partidas antes de guardarlas: qué partidas y qué tramo de pares de
 * cada una. La interfaz cuenta los pares desde 1 e incluye el último («del 1 al 20»);
 * el contrato los envía desde 0 y sin incluir el final ([0, 20]).
 */

/**
 * @typedef {object} EleccionPartida
 * @property {boolean} incluir Si la partida se guarda.
 * @property {string | number} desde Primer par, desde 1.
 * @property {string | number} hasta Último par, incluido.
 */

/**
 * Cantidad con su palabra en singular o plural: «1 partida», «3 partidas».
 *
 * @param {number} cantidad
 * @param {string} singular
 * @param {string} plural
 * @returns {string}
 */
export const contar = (cantidad, singular, plural) => `${cantidad} ${cantidad === 1 ? singular : plural}`;

/**
 * Elección inicial: todas las partidas enteras.
 *
 * @param {import('../api/laboratorio.js').Partida[]} partidas
 * @returns {Record<number, EleccionPartida>}
 */
export function eleccionInicial(partidas) {
  return Object.fromEntries(partidas.map((p) => [p.indice, { incluir: true, desde: 1, hasta: p.pasos }]));
}

const aEntero = (valor) => {
  const texto = String(valor).trim();
  return /^\d+$/.test(texto) ? Number(texto) : null;
};

/**
 * Comprueba la revisión y la convierte en lo que pide guardar.
 *
 * @param {import('../api/laboratorio.js').Partida[]} partidas Partidas de la sesión.
 * @param {Record<number, EleccionPartida>} elecciones Elección de cada partida.
 * @param {string} nombre Nombre del conjunto.
 * @returns {{valida: boolean, errores: {nombre?: string, general?: string,
 *   partidas: Record<number, string>}, seleccion: {indice: number, tramo: [number, number]}[],
 *   totales: {partidas: number, pares: number}}}
 */
export function validarRevision(partidas, elecciones, nombre) {
  const errores = { partidas: {} };
  const seleccion = [];
  for (const partida of partidas) {
    const eleccion = elecciones[partida.indice];
    if (!eleccion?.incluir) {
      continue;
    }
    const desde = aEntero(eleccion.desde);
    const hasta = aEntero(eleccion.hasta);
    if (desde === null || hasta === null) {
      errores.partidas[partida.indice] = 'Escribe números enteros.';
    } else if (desde < 1 || hasta > partida.pasos || desde > hasta) {
      errores.partidas[partida.indice] = `Elige un tramo entre 1 y ${partida.pasos}, con «desde» menor o igual que «hasta».`;
    } else {
      seleccion.push({ indice: partida.indice, tramo: [desde - 1, hasta] });
    }
  }
  if (typeof nombre !== 'string' || !nombre.trim()) {
    errores.nombre = 'Escribe un nombre para el conjunto.';
  }
  if (partidas.every((p) => !elecciones[p.indice]?.incluir)) {
    errores.general = 'Elige al menos una partida.';
  }
  const pares = seleccion.reduce((total, { tramo: [inicio, fin] }) => total + fin - inicio, 0);
  const valida = !errores.nombre && !errores.general && Object.keys(errores.partidas).length === 0;
  return { valida, errores, seleccion, totales: { partidas: seleccion.length, pares } };
}
