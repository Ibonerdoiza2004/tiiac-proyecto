import { describe, expect, it } from 'vitest';

import { ENTORNOS } from '../entornos.js';
import { CONFIGURACION_SIMULADA, crearAleatorio, crearSimulacion, pasoCartPole,
  pasoMountainCar } from './entornos.js';

// Estados que da Gymnasium 1.3 con las mismas acciones desde el mismo estado
// (obtenidos con gymnasium.make(...).unwrapped; Gymnasium guarda float32).
const CARTPOLE_GYMNASIUM = [
  [0.009600000455975533, 0.17467918992042542, 0.030799999833106995, -0.24306872487068176],
  [0.013093584217131138, 0.3693479895591736, 0.025938626378774643, -0.5258796215057373],
  [0.02048054337501526, 0.17387081682682037, 0.015421032905578613, -0.22513741254806519],
  [0.023957960307598114, 0.3687690198421478, 0.010918284766376019, -0.5129163265228271],
  [0.03133333846926689, 0.17349499464035034, 0.0006599578191526234, -0.216812863945961],
  [0.03480324149131775, -0.021636387333273888, -0.0036762994714081287, 0.07607816159725189],
  [0.03437051177024841, 0.17353807389736176, -0.002154736313968897, -0.21776238083839417],
];
const MOUNTAINCAR_GYMNASIUM = [
  [-0.49917683005332947, 0.0008231570245698094], [-0.49753668904304504, 0.001640156377106905],
  [-0.49709179997444153, 0.00044488973799161613], [-0.4968455135822296, 0.0002462968695908785],
  [-0.49579963088035583, 0.001045862678438425], [-0.4959620237350464, -0.0001623892894713208],
  [-0.497331440448761, -0.0013694275403395295],
];

function cercanos(obtenido, esperado) {
  obtenido.forEach((valor, i) => expect(valor).toBeCloseTo(esperado[i], 5));
}

describe('dinámica', () => {
  it('CartPole coincide con Gymnasium', () => {
    let estado = [0.01, -0.02, 0.03, 0.04];
    [1, 1, 0, 1, 0, 0, 1].forEach((accion, i) => {
      estado = pasoCartPole(estado, accion);
      cercanos(estado, CARTPOLE_GYMNASIUM[i]);
    });
  });

  it('MountainCar coincide con Gymnasium', () => {
    let estado = [-0.5, 0];
    [2, 2, 0, 1, 2, 0, 0].forEach((accion, i) => {
      estado = pasoMountainCar(estado, accion);
      cercanos(estado, MOUNTAINCAR_GYMNASIUM[i]);
    });
  });

  it('MountainCar se para contra la pared izquierda', () => {
    expect(pasoMountainCar([-1.19, -0.02], 0)).toEqual([-1.2, 0]);
  });
});

describe('crearAleatorio', () => {
  it('se repite con la misma semilla y cambia con otra', () => {
    const serie = (semilla) => {
      const a = crearAleatorio(semilla);
      return Array.from({ length: 5 }, a);
    };
    expect(serie(3)).toEqual(serie(3));
    expect(serie(3)).not.toEqual(serie(4));
    expect(serie(3).every((x) => x >= 0 && x < 1)).toBe(true);
  });
});

function jugar(entorno, politica, semilla = 1) {
  const simulacion = crearSimulacion(entorno, crearAleatorio(semilla));
  let pasos = 0;
  let resultado;
  do {
    resultado = simulacion.paso(politica(simulacion));
    pasos += 1;
  } while (!resultado.terminado && !resultado.truncado && pasos < 10000);
  return { pasos, ...resultado };
}

describe('crearSimulacion', () => {
  it('tiene una simulación y una configuración por cada entorno', () => {
    for (const { id } of ENTORNOS) {
      expect(() => crearSimulacion(id, crearAleatorio(0))).not.toThrow();
      expect(CONFIGURACION_SIMULADA[id].fps).toBeGreaterThan(0);
    }
    expect(() => crearSimulacion('pendulum', crearAleatorio(0))).toThrow(RangeError);
  });

  it('CartPole termina si el palo cae y se trunca en 500 pasos', () => {
    const empujarSiempre = jugar('cartpole', () => 1);
    expect(empujarSiempre.terminado).toBe(true);
    expect(empujarSiempre.pasos).toBeLessThan(50);
    const experto = jugar('cartpole', (s) => s.accionExperto(crearAleatorio(9)));
    expect(experto).toMatchObject({ pasos: 500, truncado: true, terminado: false });
  });

  it('MountainCar sin acelerar se trunca en 200 pasos y el experto llega a la meta', () => {
    expect(jugar('mountaincar', () => 1)).toMatchObject({ pasos: 200, truncado: true });
    const experto = jugar('mountaincar', (s) => s.accionExperto(crearAleatorio(2)));
    expect(experto.terminado).toBe(true);
    expect(experto.pasos).toBeLessThan(200);
  });

  it.each([['acrobot', 120], ['reacher', 50], ['halfcheetah', 150]])(
    '%s es una animación de ejemplo de %i pasos', (entorno, pasos) => {
      expect(jugar(entorno, (s) => s.accionExperto(crearAleatorio(0)))).toMatchObject({
        pasos, truncado: true, terminado: false });
    });

  it('sin tecla CartPole se para y MountainCar no acelera', () => {
    expect(crearSimulacion('cartpole', crearAleatorio(0)).accionSinTecla).toBeNull();
    expect(crearSimulacion('mountaincar', crearAleatorio(0)).accionSinTecla).toBe(1);
  });

  it('reiniciar empieza otra partida y la misma semilla da las mismas partidas', () => {
    const a = crearSimulacion('cartpole', crearAleatorio(5));
    const b = crearSimulacion('cartpole', crearAleatorio(5));
    expect(a.foto()).toEqual(b.foto());
    a.paso(1);
    expect(a.foto().paso).toBe(1);
    a.reiniciar();
    expect(a.foto().paso).toBe(0);
    expect(a.foto().estado.every((x) => Math.abs(x) <= 0.05)).toBe(true);
    expect(a.foto().estado).not.toEqual(b.foto().estado);
  });
});
