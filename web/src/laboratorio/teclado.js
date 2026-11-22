/**
 * Teclado de «Jugar tú»: qué acción del entorno corresponde a las teclas pulsadas.
 *
 * Se envía la acción mientras se mantiene la tecla. Con las dos flechas pulsadas manda
 * la última que se pulsó. Sin ninguna, la acción es null y el backend decide: CartPole se
 * para hasta que se vuelva a pulsar y MountainCar no acelera.
 */

const TECLAS = Object.freeze({
  cartpole: Object.freeze({ arrowleft: 0, a: 0, arrowright: 1, d: 1 }),
  mountaincar: Object.freeze({ arrowleft: 0, a: 0, arrowright: 2, d: 2 }),
});

const AYUDA = Object.freeze({
  cartpole: 'Mantén ← o → (o A y D) para empujar el carro; sin tecla, la partida espera.',
  mountaincar: 'Mantén ← o → (o A y D) para acelerar; sin tecla, el coche no acelera.',
});

/**
 * Normaliza KeyboardEvent.key: minúsculas, para que A y a sean la misma tecla.
 *
 * @param {string} tecla Valor de KeyboardEvent.key.
 * @returns {string}
 */
export const normalizarTecla = (tecla) => String(tecla).toLowerCase();

/**
 * Indica si la tecla controla el entorno.
 *
 * @param {string} entorno Clave del entorno.
 * @param {string} tecla Valor de KeyboardEvent.key.
 * @returns {boolean}
 */
export function esTeclaDeJuego(entorno, tecla) {
  return Object.hasOwn(TECLAS[entorno] ?? {}, normalizarTecla(tecla));
}

/**
 * Acción que corresponde a las teclas pulsadas, en el orden en que se pulsaron.
 *
 * @param {string} entorno Clave del entorno.
 * @param {string[]} pulsadas Teclas mantenidas, la última pulsada al final.
 * @returns {number | null} La acción de la última tecla de juego, o null si no hay.
 */
export function accionDeTeclas(entorno, pulsadas) {
  const mapa = TECLAS[entorno] ?? {};
  for (let i = pulsadas.length - 1; i >= 0; i -= 1) {
    const tecla = normalizarTecla(pulsadas[i]);
    if (Object.hasOwn(mapa, tecla)) {
      return mapa[tecla];
    }
  }
  return null;
}

/**
 * Texto de ayuda de los controles del entorno.
 *
 * @param {string} entorno Clave del entorno.
 * @returns {string | null} null si el entorno no se juega a mano.
 */
export function ayudaTeclado(entorno) {
  return AYUDA[entorno] ?? null;
}

/**
 * Acciones de los botones en pantalla (para ratón o pantalla táctil).
 *
 * @param {string} entorno Clave del entorno.
 * @returns {{izquierda: number, derecha: number} | null}
 */
export function accionesBotones(entorno) {
  const mapa = TECLAS[entorno];
  return mapa ? { izquierda: mapa.arrowleft, derecha: mapa.arrowright } : null;
}
