/**
 * Entornos simulados en el navegador para desarrollar el Laboratorio sin backend.
 *
 * CartPole y MountainCar siguen las ecuaciones de Gymnasium 1.3 (CartPole-v1 y
 * MountainCar-v0) para que jugar a mano se parezca a lo real. Acrobot, Reacher y
 * HalfCheetah son solo animaciones de ejemplo con partidas cortas: sirven para probar
 * «Ver agente», no imitan su física. Nada de esto se usa para entrenar ni evaluar.
 */

/**
 * Generador pseudoaleatorio con semilla (mulberry32), para que la simulación se repita.
 *
 * @param {number} semilla Entero.
 * @returns {() => number} Devuelve números en [0, 1).
 */
export function crearAleatorio(semilla) {
  let a = semilla >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const uniforme = (aleatorio, min, max) => min + (max - min) * aleatorio();

// Constantes de CartPole-v1 en Gymnasium.
const CP = {
  gravedad: 9.8, masaCarro: 1, masaPalo: 0.1, mitadPalo: 0.5, fuerza: 10, tau: 0.02,
  limiteAngulo: (12 * 2 * Math.PI) / 360, limiteX: 2.4, maxPasos: 500,
};

/**
 * Un paso de CartPole-v1 (integración de Euler, como Gymnasium).
 *
 * @param {number[]} estado [x, velocidad, ángulo, velocidad angular].
 * @param {0 | 1} accion 0 empuja a la izquierda y 1 a la derecha.
 * @returns {number[]} El estado siguiente.
 */
export function pasoCartPole([x, v, theta, w], accion) {
  const total = CP.masaCarro + CP.masaPalo;
  const momento = CP.masaPalo * CP.mitadPalo;
  const fuerza = accion === 1 ? CP.fuerza : -CP.fuerza;
  const cos = Math.cos(theta);
  const sin = Math.sin(theta);
  const temp = (fuerza + momento * w * w * sin) / total;
  const aceleracionAngular = (CP.gravedad * sin - cos * temp)
    / (CP.mitadPalo * (4 / 3 - (CP.masaPalo * cos * cos) / total));
  const aceleracion = temp - (momento * aceleracionAngular * cos) / total;
  return [x + CP.tau * v, v + CP.tau * aceleracion, theta + CP.tau * w,
    w + CP.tau * aceleracionAngular];
}

// Constantes de MountainCar-v0 en Gymnasium.
const MC = {
  minimo: -1.2, maximo: 0.6, velocidadMaxima: 0.07, meta: 0.5, fuerza: 0.001,
  gravedad: 0.0025, maxPasos: 200,
};

/**
 * Un paso de MountainCar-v0.
 *
 * @param {number[]} estado [posición, velocidad].
 * @param {0 | 1 | 2} accion 0 acelera a la izquierda, 1 no acelera y 2 a la derecha.
 * @returns {number[]} El estado siguiente.
 */
export function pasoMountainCar([posicion, velocidad], accion) {
  let v = velocidad + (accion - 1) * MC.fuerza + Math.cos(3 * posicion) * -MC.gravedad;
  v = Math.min(Math.max(v, -MC.velocidadMaxima), MC.velocidadMaxima);
  let p = Math.min(Math.max(posicion + v, MC.minimo), MC.maximo);
  if (p === MC.minimo && v < 0) {
    v = 0;
  }
  return [p, v];
}

/**
 * @typedef {object} Simulacion
 * @property {() => void} reiniciar Empieza una partida nueva.
 * @property {(accion: number) => {recompensa: number, terminado: boolean,
 *   truncado: boolean}} paso Avanza un paso.
 * @property {(aleatorio: () => number) => number} accionExperto Acción del «experto».
 * @property {number | null} accionSinTecla Acción si no hay tecla pulsada, o null si
 *   la partida se para (CartPole, que no tiene acción de no hacer nada).
 * @property {() => {estado: number[], paso: number}} foto Lo necesario para dibujar.
 */

const ESPECIFICACIONES = {
  cartpole: {
    accionSinTecla: null,
    inicial: (a) => Array.from({ length: 4 }, () => uniforme(a, -0.05, 0.05)),
    paso: (estado, accion) => {
      const siguiente = pasoCartPole(estado, accion);
      const terminado = Math.abs(siguiente[0]) > CP.limiteX
        || Math.abs(siguiente[2]) > CP.limiteAngulo;
      return { siguiente, recompensa: 1, terminado };
    },
    maxPasos: CP.maxPasos,
    experto: ([, , theta, w], a) => (a() < 0.05 ? (a() < 0.5 ? 0 : 1) : (theta + 0.3 * w > 0 ? 1 : 0)),
  },
  mountaincar: {
    // Sin tecla, el coche no acelera (acción 1): pendiente de confirmar en el backend.
    accionSinTecla: 1,
    inicial: (a) => [uniforme(a, -0.6, -0.4), 0],
    paso: (estado, accion) => {
      const siguiente = pasoMountainCar(estado, accion);
      return { siguiente, recompensa: -1, terminado: siguiente[0] >= MC.meta && siguiente[1] >= 0 };
    },
    maxPasos: MC.maxPasos,
    experto: ([, v], a) => (a() < 0.05 ? 1 : (v >= 0 ? 2 : 0)),
  },
};

// Animaciones de ejemplo: una fase que avanza y una partida de longitud fija.
const EJEMPLO = { acrobot: 120, reacher: 50, halfcheetah: 150 };

function especificacionEjemplo(entorno) {
  const pasos = EJEMPLO[entorno];
  return {
    accionSinTecla: null,
    inicial: (a) => [uniforme(a, 0, 2 * Math.PI)],
    paso: ([fase]) => ({ siguiente: [fase + 0.15], recompensa: entorno === 'acrobot' ? -1 : 1,
      terminado: false }),
    maxPasos: pasos,
    experto: () => 0,
  };
}

/**
 * Crea la simulación de un entorno.
 *
 * @param {string} entorno Clave del entorno (cartpole, mountaincar…).
 * @param {() => number} aleatorio Generador de crearAleatorio.
 * @returns {Simulacion}
 * @throws {RangeError} Si el entorno no existe.
 */
export function crearSimulacion(entorno, aleatorio) {
  const especificacion = ESPECIFICACIONES[entorno]
    ?? (entorno in EJEMPLO ? especificacionEjemplo(entorno) : null);
  if (!especificacion) {
    throw new RangeError(`Entorno desconocido: ${String(entorno)}`);
  }
  let estado = especificacion.inicial(aleatorio);
  let pasos = 0;
  return {
    reiniciar() {
      estado = especificacion.inicial(aleatorio);
      pasos = 0;
    },
    paso(accion) {
      const { siguiente, recompensa, terminado } = especificacion.paso(estado, accion);
      estado = siguiente;
      pasos += 1;
      return { recompensa, terminado, truncado: !terminado && pasos >= especificacion.maxPasos };
    },
    accionExperto: (a) => especificacion.experto(estado, a),
    accionSinTecla: especificacion.accionSinTecla,
    foto: () => ({ estado: [...estado], paso: pasos }),
  };
}

/** Imágenes por segundo (render_fps de Gymnasium) y partidas del perfil por entorno. */
export const CONFIGURACION_SIMULADA = Object.freeze({
  cartpole: { fps: 50, partidas_experto: 15 },
  mountaincar: { fps: 30, partidas_experto: 15 },
  acrobot: { fps: 15, partidas_experto: 15 },
  reacher: { fps: 50, partidas_experto: 30 },
  halfcheetah: { fps: 20, partidas_experto: 30 },
});
