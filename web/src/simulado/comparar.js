/**
 * Comparar simulado: la misma ApiComparar que src/api/comparar.js, en el navegador.
 *
 * Los modelos son los de la biblioteca simulada (./biblioteca.js): por entorno, el experto
 * congelado y unos BC y GAIL de entrega inventados, más los que se entrenen en el
 * Laboratorio simulado. La comparación corre en el servidor de trabajos simulado con el
 * protocolo único: un progreso por partida de cada modelo, ronda a ronda, y el resultado
 * al terminar. Los retornos son
 * inventados (cerca del objetivo del perfil para el experto, más bajos cuanto peor es el
 * modelo); las partidas, la semilla y el determinismo por defecto son los de configs/.
 */

import { ALEATORIO, MAXIMO_PARTIDAS, comparacionesDe } from '../api/comparar.js';
import { ErrorApi } from '../api/http.js';
import { ENTORNOS } from '../entornos.js';
import { crearAlmacen } from './almacen.js';
import { modelosSimulados } from './biblioteca.js';
import { crearAleatorio, crearSimulacion } from './entornos.js';
import EVALUACION from './evaluacion.json';
import PERFILES from './perfiles.json';
import { crearServidorTrabajosSimulado } from './trabajos.js';

function rechazar(estado, mensaje) {
  return Promise.reject(new ErrorApi(mensaje, { estado, detalle: { detail: mensaje } }));
}

const campoExperto = (entorno, clave) => PERFILES[entorno].experto.secciones
  .find((s) => s.clave === 'experto').campos.find((c) => c.clave === clave).valor;

/** Calidad inventada de un modelo, de 0 (aleatorio) a 1 (experto). */
function calidad(modelo) {
  if (modelo.tipo === 'experto') {
    return 1;
  }
  const partidas = modelo.detalles?.cantidad ?? modelo.detalles?.subconjunto ?? 1;
  const tope = modelo.tipo === 'gail' ? 0.97 : 0.8;
  return tope * (1 - Math.exp(-partidas / 3));
}

/**
 * Crea la API de Comparar simulada.
 *
 * @param {object} [opciones]
 * @param {import('./almacen.js').Almacen} [opciones.almacen] Biblioteca simulada (la misma
 *   que el entrenamiento simulado, para comparar lo que se entrena).
 * @param {ReturnType<typeof crearServidorTrabajosSimulado>} [opciones.servidor]
 * @param {(entorno: string, foto: object) => Promise<Blob>} [opciones.dibujar]
 * @returns {import('../api/comparar.js').ApiComparar}
 */
export function crearApiCompararSimulada({ almacen = crearAlmacen(), servidor = crearServidorTrabajosSimulado(),
  dibujar } = {}) {
  const pintar = dibujar
    ?? ((entorno, foto) => import('./dibujo.js').then((m) => m.dibujarFotograma(entorno, foto)));

  const modelos = (entorno) => modelosSimulados(almacen, entorno);

  return {
    modelos(entorno) {
      if (!ENTORNOS.some((e) => e.id === entorno)) {
        return rechazar(422, `Entorno desconocido: ${entorno}`);
      }
      return Promise.resolve(JSON.parse(JSON.stringify(modelos(entorno))));
    },

    comparaciones: (entorno) => Promise.resolve(comparacionesDe(servidor.listar({ tipo: 'comparacion' }), entorno)),

    crear({ entorno, modelos: ids, aleatorio = false, partidas, vista = false } = {}) {
      if (!ENTORNOS.some((e) => e.id === entorno)) {
        return rechazar(422, `Entorno desconocido: ${entorno}`);
      }
      if (!Array.isArray(ids) || new Set(ids).size !== ids.length) {
        return rechazar(422, 'modelos debe ser una lista de ids sin repetir');
      }
      if (typeof aleatorio !== 'boolean' || typeof vista !== 'boolean') {
        return rechazar(422, 'aleatorio y vista deben ser booleanos');
      }
      if (ids.length + Number(aleatorio) < 2) {
        return rechazar(422, 'Hacen falta al menos dos modelos para comparar');
      }
      if (partidas !== undefined && (!Number.isInteger(partidas) || partidas < 1 || partidas > MAXIMO_PARTIDAS)) {
        return rechazar(422, `Las partidas deben ser un entero entre 1 y ${MAXIMO_PARTIDAS}`);
      }
      const disponibles = new Map(modelos(entorno).map((m) => [m.id, m]));
      const ajenos = ids.filter((id) => !disponibles.get(id)?.valido);
      if (ajenos.length) {
        return rechazar(422, `Modelos que no existen en la biblioteca de ${entorno}: ${ajenos.join(', ')}`);
      }
      const evaluacion = EVALUACION[entorno];
      const argumentos = { entorno, modelos: ids, aleatorio, partidas: partidas ?? evaluacion.episodios,
        semilla: evaluacion.semilla, vista };
      const participantes = [...ids.map((id) => disponibles.get(id)), ...(aleatorio ? [{ id: ALEATORIO }] : [])];
      const objetivo = campoExperto(entorno, 'retorno_objetivo');
      const azar = objetivo > 0 ? objetivo * 0.044 : objetivo * 2.2;

      const trabajo = servidor.crear('comparacion', argumentos, (canal) => {
        let ronda = 0;
        let verVista = vista;
        const simulacion = crearSimulacion(entorno, crearAleatorio(evaluacion.semilla));
        const retornos = Object.fromEntries(participantes.map((p) => [p.id, []]));
        const terminar = (incompleta) => canal.terminar({ partidas: ronda, semilla: evaluacion.semilla,
          incompleta, modelos: participantes.map((p) => ({ id: p.id, retornos: retornos[p.id] })) });
        return {
          paso() {
            // Todos los modelos juegan la partida «ronda» con la misma semilla.
            const ruido = crearAleatorio(evaluacion.semilla + ronda);
            for (const p of participantes) {
              const q = p.id === ALEATORIO ? 0 : calidad(p);
              // El ruido solo resta: nadie supera el objetivo (500 es el máximo de CartPole).
              const retorno = azar + q * (objetivo - azar) - ruido() * Math.abs(objetivo) * 0.08;
              retornos[p.id].push(retorno);
              canal.progreso({ modelo: p.id, partida: ronda, retorno, pasos: 1 + Math.round(ruido() * 499) });
            }
            if (verVista) {
              for (let i = 0; i < 5; i += 1) {
                const fin = simulacion.paso(simulacion.accionExperto(ruido));
                if (fin.terminado || fin.truncado) {
                  simulacion.reiniciar();
                }
              }
              pintar(entorno, simulacion.foto()).then(canal.imagen).catch(() => {});
            }
            ronda += 1;
            if (ronda >= argumentos.partidas) {
              terminar(false);
            }
          },
          orden(orden) {
            if (orden.tipo === 'vista') {
              verVista = Boolean(orden.activa);
            } else if (orden.tipo === 'finalizar') {
              terminar(true);
            }
          },
        };
      });
      return Promise.resolve(trabajo);
    },

    seguir: (id, callbacks) => servidor.seguir(id, callbacks),
  };
}

