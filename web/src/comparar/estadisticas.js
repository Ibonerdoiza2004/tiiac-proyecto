/**
 * Resultados de una comparación a partir de su progreso: un mensaje por partida terminada
 * de cada modelo, {modelo, partida, retorno}. Sirve igual para el historial que llega al
 * reconectar que para el progreso en directo: una partida repetida no cuenta dos veces.
 */

/**
 * Añade un mensaje de progreso a lo acumulado, sin modificarlo.
 *
 * @param {Record<string, Record<number, number>>} acumulado Retorno por modelo y partida.
 * @param {Record<string, unknown>} medidas {modelo, partida, retorno}.
 * @returns {Record<string, Record<number, number>>} Lo acumulado, nuevo si cambia.
 */
export function acumular(acumulado, medidas) {
  const { modelo, partida, retorno } = medidas ?? {};
  if (typeof modelo !== 'string' || !modelo || !Number.isInteger(partida) || partida < 0
      || typeof retorno !== 'number' || !Number.isFinite(retorno)) {
    return acumulado;
  }
  if (acumulado[modelo]?.[partida] !== undefined) {
    return acumulado;
  }
  return { ...acumulado, [modelo]: { ...acumulado[modelo], [partida]: retorno } };
}

/**
 * Acumula una lista de mensajes de progreso desde cero (el historial al conectar).
 *
 * @param {Record<string, unknown>[]} progreso
 * @returns {Record<string, Record<number, number>>}
 */
export const acumularTodo = (progreso) => progreso.reduce(acumular, {});

/**
 * Retornos de un resultado final ({modelos: [{id, retornos}]}) con la misma forma que el
 * acumulado del progreso. Si el resultado no trae modelos, devuelve null.
 *
 * @param {{modelos?: {id: string, retornos: number[]}[]} | null | undefined} resultado
 * @returns {Record<string, Record<number, number>> | null}
 */
export function desdeResultado(resultado) {
  if (!Array.isArray(resultado?.modelos)) {
    return null;
  }
  return resultado.modelos.reduce((acumulado, elemento) => {
    const { id, retornos } = elemento ?? {};
    return Array.isArray(retornos)
      ? retornos.reduce((a, retorno, partida) => acumular(a, { modelo: id, partida, retorno }), acumulado)
      : acumulado;
  }, {});
}

/**
 * @typedef {object} ResumenModelo
 * @property {string} modelo
 * @property {number} partidas Partidas jugadas.
 * @property {number | null} media
 * @property {number | null} desviacion Desviación típica muestral; null con menos de dos.
 * @property {number | null} ultima Retorno de la última partida jugada.
 * @property {number[]} retornos En orden de partida.
 * @property {Record<number, number>} porPartida Retorno por número de partida (desde 0),
 *   para no desplazar las filas si falta alguna.
 */

/**
 * Media, desviación y última partida de un modelo.
 *
 * @param {string} modelo
 * @param {Record<number, number> | undefined} porPartida
 * @returns {ResumenModelo}
 */
export function resumirModelo(modelo, porPartida = {}) {
  const indices = Object.keys(porPartida).map(Number).sort((a, b) => a - b);
  const retornos = indices.map((i) => porPartida[i]);
  const n = retornos.length;
  if (!n) {
    return { modelo, partidas: 0, media: null, desviacion: null, ultima: null, retornos, porPartida: {} };
  }
  const media = retornos.reduce((suma, r) => suma + r, 0) / n;
  const desviacion = n > 1
    ? Math.sqrt(retornos.reduce((suma, r) => suma + (r - media) ** 2, 0) / (n - 1)) : null;
  return { modelo, partidas: n, media, desviacion, ultima: retornos.at(-1), retornos,
    porPartida: { ...porPartida } };
}

/**
 * Resumen de cada modelo, en el orden dado.
 *
 * @param {Record<string, Record<number, number>>} acumulado
 * @param {string[]} modelos
 * @returns {ResumenModelo[]}
 */
export function resumir(acumulado, modelos) {
  return modelos.map((modelo) => resumirModelo(modelo, acumulado[modelo]));
}

/**
 * Partidas que han terminado todos los modelos (la menor de sus cuentas).
 *
 * @param {Record<string, Record<number, number>>} acumulado
 * @param {string[]} modelos
 * @returns {number}
 */
export function partidasCompletas(acumulado, modelos) {
  if (!modelos.length) {
    return 0;
  }
  return Math.min(...modelos.map((m) => Object.keys(acumulado[m] ?? {}).length));
}

/**
 * Ordena los resúmenes de mejor a peor media; los que aún no han jugado, al final.
 *
 * @param {ResumenModelo[]} resumenes
 * @returns {ResumenModelo[]}
 */
export function clasificar(resumenes) {
  return [...resumenes].sort((a, b) => {
    if (a.media === null || b.media === null) {
      return (a.media === null) - (b.media === null);
    }
    return b.media - a.media;
  });
}
