/**
 * Backend simulado del Laboratorio: implementa la misma ApiLaboratorio que
 * src/api/laboratorio.js, pero en el navegador, para trabajar la interfaz mientras no
 * existe laboratorio-jugar-api. Sigue el contrato propuesto: mismas respuestas, mismos
 * mensajes y mismos errores (ErrorApi con código 404 o 422).
 */

import { ErrorApi } from '../api/http.js';
import { MODOS } from '../api/laboratorio.js';
import { buscarEntorno } from '../entornos.js';
import { CONFIGURACION_SIMULADA, crearAleatorio, crearSimulacion } from './entornos.js';

const MAXIMO_PARTIDAS = 1000;

const esEnteroPositivo = (valor) => Number.isInteger(valor) && valor > 0;

function rechazar(estado, mensaje) {
  return Promise.reject(new ErrorApi(mensaje, { estado, detalle: { detail: mensaje } }));
}

/**
 * Crea la API simulada.
 *
 * @param {object} [opciones]
 * @param {(entorno: string, foto: object) => Promise<Blob>} [opciones.dibujar] Dibuja
 *   cada imagen (por defecto, dibujarFotograma de ./dibujo.js, que necesita un navegador).
 * @param {number} [opciones.semilla] Semilla base de las partidas.
 * @returns {import('../api/laboratorio.js').ApiLaboratorio}
 */
export function crearApiLaboratorioSimulada({ dibujar, semilla = 1000 } = {}) {
  const sesiones = new Map();
  let contador = 0;
  const pintar = dibujar
    ?? ((entorno, foto) => import('./dibujo.js').then((m) => m.dibujarFotograma(entorno, foto)));

  return {
    configuracion(entorno) {
      const configuracion = CONFIGURACION_SIMULADA[entorno];
      return configuracion ? Promise.resolve({ ...configuracion })
        : rechazar(404, `Entorno desconocido: ${entorno}`);
    },

    crearSesion({ entorno, modo, partidas } = {}) {
      const configuracion = CONFIGURACION_SIMULADA[entorno];
      if (!configuracion) {
        return rechazar(422, `Entorno desconocido: ${entorno}`);
      }
      if (modo !== MODOS.MANUAL && modo !== MODOS.EXPERTO) {
        return rechazar(422, `Modo desconocido: ${modo}`);
      }
      if (modo === MODOS.MANUAL && !buscarEntorno(entorno).jugarAMano) {
        return rechazar(422, `En ${buscarEntorno(entorno).nombre} no se puede jugar a mano`);
      }
      let objetivo = null;
      if (modo === MODOS.EXPERTO) {
        objetivo = partidas ?? configuracion.partidas_experto;
        if (!esEnteroPositivo(objetivo) || objetivo > MAXIMO_PARTIDAS) {
          return rechazar(422, `Las partidas deben ser un entero entre 1 y ${MAXIMO_PARTIDAS}`);
        }
      }
      contador += 1;
      const id = `simulada-${contador}`;
      sesiones.set(id, { entorno, modo, objetivo, partidas: [], terminada: false,
        semilla: semilla + 1000 * contador });
      return Promise.resolve({ id, entorno, modo, fps: configuracion.fps, partidas: objetivo });
    },

    conectar(id, callbacks = {}) {
      const sesion = sesiones.get(id);
      if (!sesion) {
        throw new ErrorApi(`Sesión desconocida: ${id}`, { estado: 404 });
      }
      const { entorno, modo, objetivo } = sesion;
      const aleatorio = crearAleatorio(sesion.semilla);
      const simulacion = crearSimulacion(entorno, crearAleatorio(sesion.semilla + 1));
      let activa = true;
      let pulsada = null;
      let paso = 0;
      let retorno = 0;
      let paresTerminados = 0;
      let esperando = false;
      const semillaPartida = () => sesion.semilla + sesion.partidas.length;

      const avisar = (callback, ...argumentos) => {
        if (activa) {
          callback?.(...argumentos);
        }
      };
      const enviarEstado = () => avisar(callbacks.alEstado, {
        partida: sesion.partidas.length + 1, paso, retorno, pares: paresTerminados + paso, esperando,
      });
      const enviarImagen = () => {
        pintar(entorno, simulacion.foto())
          .then((imagen) => avisar(callbacks.alImagen, imagen))
          .catch((error) => avisar(callbacks.alError, error));
      };
      const terminar = () => {
        sesion.terminada = true;
        clearInterval(reloj);
        const partidas = sesion.partidas.map((partida) => ({ ...partida }));
        // Como el servidor real, la respuesta llega después, no dentro de la llamada.
        setTimeout(() => {
          avisar(callbacks.alTerminarSesion, partidas);
          activa = false;
        }, 0);
      };

      const avanzar = () => {
        if (!activa || sesion.terminada) {
          return;
        }
        const accion = modo === MODOS.EXPERTO ? simulacion.accionExperto(aleatorio)
          : (pulsada ?? simulacion.accionSinTecla);
        if (accion === null) {
          if (!esperando) {
            esperando = true;
            enviarEstado();
          }
          return;
        }
        esperando = false;
        const { recompensa, terminado, truncado } = simulacion.paso(accion);
        paso += 1;
        retorno += recompensa;
        enviarImagen();
        enviarEstado();
        if (terminado || truncado) {
          const partida = { indice: sesion.partidas.length, pasos: paso, retorno,
            semilla: semillaPartida() };
          sesion.partidas.push(partida);
          paresTerminados += paso;
          avisar(callbacks.alTerminarPartida, { ...partida });
          if (modo === MODOS.EXPERTO && sesion.partidas.length >= objetivo) {
            terminar();
            return;
          }
          simulacion.reiniciar();
          paso = 0;
          retorno = 0;
          enviarImagen();
          enviarEstado();
        }
      };

      const reloj = setInterval(avanzar, 1000 / CONFIGURACION_SIMULADA[entorno].fps);
      setTimeout(() => {
        enviarImagen();
        enviarEstado();
      }, 0);

      return {
        enviarAccion(accion) {
          if (!activa || sesion.terminada) {
            return false;
          }
          pulsada = accion;
          return true;
        },
        finalizar() {
          if (!activa) {
            return false;
          }
          // Si el experto acaba de terminar, sesion_terminada ya está de camino.
          if (!sesion.terminada) {
            terminar();
          }
          return true;
        },
        cerrar() {
          activa = false;
          clearInterval(reloj);
        },
      };
    },

    guardar(id, { nombre, partidas } = {}) {
      const sesion = sesiones.get(id);
      if (!sesion) {
        return rechazar(404, `Sesión desconocida: ${id}`);
      }
      if (!sesion.terminada) {
        return rechazar(422, 'La sesión todavía no ha terminado');
      }
      if (typeof nombre !== 'string' || !nombre.trim()) {
        return rechazar(422, 'El conjunto necesita un nombre');
      }
      if (!Array.isArray(partidas) || partidas.length === 0) {
        return rechazar(422, 'Elige al menos una partida');
      }
      const vistas = new Set();
      let pares = 0;
      for (const { indice, tramo } of partidas) {
        const partida = sesion.partidas[indice];
        const [inicio, fin] = Array.isArray(tramo) ? tramo : [];
        if (!partida || vistas.has(indice) || !Number.isInteger(inicio) || !Number.isInteger(fin)
            || inicio < 0 || inicio >= fin || fin > partida.pasos) {
          return rechazar(422, `Partida o tramo no válido: ${JSON.stringify({ indice, tramo })}`);
        }
        vistas.add(indice);
        pares += fin - inicio;
      }
      sesiones.delete(id);
      return Promise.resolve({ nombre: nombre.trim(), partidas: partidas.length, pares });
    },

    descartar(id) {
      if (!sesiones.delete(id)) {
        return rechazar(404, `Sesión desconocida: ${id}`);
      }
      return Promise.resolve();
    },
  };
}
