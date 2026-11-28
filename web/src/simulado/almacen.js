/**
 * Biblioteca del backend simulado: lo que guarda «Jugar tú» o «Ver agente» y los modelos
 * que se entrenan, compartido entre las APIs simuladas para que se vean unas a otras.
 * Solo vive en la memoria del navegador: al recargar la página empieza de cero.
 */

import { ENTORNOS } from '../entornos.js';
import DEMOSTRACIONES from './demostraciones.json';
import { crearAleatorio } from './entornos.js';
import PERFILES from './perfiles.json';

// Pasos de cada partida congelada donde todas duran lo mismo (README): CartPole llega a
// 500 y Reacher y HalfCheetah se truncan en 50 y 1000. En los demás, inventados.
const PASOS_FIJOS = Object.freeze({ cartpole: 500, reacher: 50, halfcheetah: 1000 });
const PASOS_INVENTADOS = Object.freeze({ mountaincar: [85, 125], acrobot: [60, 95] });
// Recompensa por paso donde Gymnasium la da constante: el retorno sale de los pasos.
const RECOMPENSA_POR_PASO = Object.freeze({ cartpole: 1, mountaincar: -1, acrobot: -1 });

/**
 * Partidas de las demostraciones congeladas simuladas de un entorno, con la forma de
 * GET /api/biblioteca/{id}/partidas: pasos y retornos inventados (cerca del objetivo del
 * perfil), y el reparto en train y validación de configs/, con train barajado como en la E1.
 *
 * @param {string} entorno
 * @returns {import('../api/biblioteca.js').Partida[]}
 */
export function partidasCongeladas(entorno) {
  const { partidas_train: train, partidas_validacion: validacion } = DEMOSTRACIONES[entorno];
  const azar = crearAleatorio(1000 + ENTORNOS.findIndex((e) => e.id === entorno));
  const objetivo = PERFILES[entorno].experto.secciones.find((s) => s.clave === 'experto')
    .campos.find((c) => c.clave === 'retorno_objetivo').valor;
  const partidas = Array.from({ length: train + validacion }, (_, id) => {
    const [minimo, maximo] = PASOS_INVENTADOS[entorno] ?? [PASOS_FIJOS[entorno], PASOS_FIJOS[entorno]];
    const pasos = minimo + Math.round(azar() * (maximo - minimo));
    const retorno = entorno in RECOMPENSA_POR_PASO ? RECOMPENSA_POR_PASO[entorno] * pasos
      : objetivo - Math.abs(objetivo) * 0.1 * azar();
    return { id_episodio: id, pasos, retorno, split: 'validacion', orden_train: null };
  });
  // Barajado de Fisher-Yates: las «train» primeras del orden barajado van a train.
  const orden = partidas.map((p) => p.id_episodio);
  for (let i = orden.length - 1; i > 0; i -= 1) {
    const j = Math.floor(azar() * (i + 1));
    [orden[i], orden[j]] = [orden[j], orden[i]];
  }
  orden.slice(0, train).forEach((id, posicion) => Object.assign(partidas[id], { split: 'train', orden_train: posicion }));
  return partidas;
}

/**
 * @typedef {object} Almacen
 * @property {object[]} demostraciones Conjuntos y subconjuntos, con su entorno, origen,
 *   fecha y las partidas (detallePartidas en un conjunto; conjunto e ids en un subconjunto).
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
    demostraciones: ENTORNOS.map(({ id, nombre }) => {
      const detallePartidas = partidasCongeladas(id);
      return {
        id: `congelado-${id}`,
        entorno: id,
        nombre: `${nombre} · experto congelado`,
        tipo: 'conjunto',
        partidas: detallePartidas.length,
        pares: detallePartidas.reduce((suma, p) => suma + p.pasos, 0),
        fuente: 'experto',
        origen: 'congelado',
        creado: '2026-11-01T00:00:00Z',
        detallePartidas,
      };
    }),
    modelos: [],
    nuevoId(prefijo) {
      contador += 1;
      return `${prefijo}-${contador}`;
    },
  };
}
