/**
 * Series de las curvas de entrenamiento: se acumulan con cada mensaje de progreso y se
 * convierten en trazos SVG.
 */

/** Puntos que se guardan como mucho por serie; al pasar, se queda uno de cada dos. */
export const MAXIMO_PUNTOS = 1000;

const NOMBRES = Object.freeze({
  retorno_medio: 'Retorno medio',
  retorno_validacion: 'Retorno de validación',
  perdida_train: 'Pérdida en train',
  perdida_validacion: 'Pérdida en validación',
  recompensa_gail_media: 'Recompensa de GAIL',
  discriminador_aciertos: 'Aciertos del discriminador',
});

/**
 * Nombre legible de una medida; las desconocidas se muestran con espacios.
 *
 * @param {string} medida Clave que envía el servidor.
 * @returns {string}
 */
export function nombreMedida(medida) {
  if (Object.hasOwn(NOMBRES, medida)) {
    return NOMBRES[medida];
  }
  const texto = String(medida).replaceAll('_', ' ').replaceAll('/', ' · ');
  return texto.charAt(0).toUpperCase() + texto.slice(1);
}

/**
 * Añade un mensaje de progreso a las series, sin modificar las que recibe.
 *
 * Solo se guardan medidas numéricas finitas. Si una serie supera maximo puntos se queda
 * con uno de cada dos, conservando siempre el último.
 *
 * @param {Record<string, [number, number][]>} series Puntos [paso, valor] por medida.
 * @param {{paso: number, medidas: Record<string, unknown>}} progreso
 * @param {number} [maximo]
 * @returns {Record<string, [number, number][]>}
 */
export function agregarProgreso(series, { paso, medidas }, maximo = MAXIMO_PUNTOS) {
  if (!Number.isFinite(paso)) {
    return series;
  }
  const nuevas = { ...series };
  for (const [medida, valor] of Object.entries(medidas ?? {})) {
    if (typeof valor !== 'number' || !Number.isFinite(valor)) {
      continue;
    }
    let puntos = [...(series[medida] ?? []), [paso, valor]];
    if (puntos.length > maximo) {
      puntos = puntos.filter((_, i) => i % 2 === 1 || i === puntos.length - 1);
    }
    nuevas[medida] = puntos;
  }
  return nuevas;
}

/**
 * Trazo SVG de una serie dentro de un rectángulo.
 *
 * @param {[number, number][]} puntos
 * @param {{ancho: number, alto: number}} tamano
 * @returns {{d: string, minimo: number, maximo: number, desde: number, hasta: number} | null}
 *   null si no hay puntos.
 */
export function trazo(puntos, { ancho, alto }) {
  if (!puntos.length) {
    return null;
  }
  const xs = puntos.map(([x]) => x);
  const ys = puntos.map(([, y]) => y);
  const desde = Math.min(...xs);
  const hasta = Math.max(...xs);
  const minimo = Math.min(...ys);
  const maximo = Math.max(...ys);
  // Con un solo paso o un valor constante, la línea queda en el centro.
  const x = (v) => (hasta === desde ? ancho / 2 : ((v - desde) / (hasta - desde)) * ancho);
  const y = (v) => (maximo === minimo ? alto / 2 : alto - ((v - minimo) / (maximo - minimo)) * alto);
  const d = puntos.map(([px, py], i) => `${i ? 'L' : 'M'}${x(px).toFixed(1)} ${y(py).toFixed(1)}`).join(' ');
  return { d, minimo, maximo, desde, hasta };
}

/**
 * Número con como mucho tres cifras significativas, en español.
 *
 * @param {number} valor
 * @returns {string}
 */
export function formatearNumero(valor) {
  if (!Number.isFinite(valor)) {
    return '—';
  }
  return valor.toLocaleString('es-ES', { maximumSignificantDigits: 3 });
}
