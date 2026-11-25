/**
 * Servidor de trabajos simulado: imita en el navegador el protocolo único de la API
 * (src/app/trabajos.py y rutas_trabajos.py) para las funcionalidades que aún no tienen
 * backend. Mantiene estado, historial de progreso y eventos, y la última imagen, y al
 * seguir un trabajo manda lo mismo que el WebSocket real: estado, historial, imagen y
 * después lo que ocurra en directo, con cierre al terminar.
 *
 * No hay cola ni límite de procesos: cada trabajo empieza en cuanto se crea.
 */

import { ESTADOS_CONEXION } from '../api/websocket.js';
import { ESTADOS_FINALES, ESTADOS_TRABAJO } from '../api/trabajos.js';

/**
 * @typedef {object} CanalSimulado Lo que recibe la tarea para hablar con el «servidor».
 * @property {(medidas: object) => void} progreso
 * @property {(imagen: Blob) => void} imagen
 * @property {(resultado: object) => void} terminar
 * @property {(mensaje: string) => void} fallar
 */

/**
 * @typedef {object} TareaSimulada
 * @property {() => void} paso Avanza la tarea; se llama cada intervalo mientras corre.
 * @property {(orden: object) => void} [orden] Órdenes propias del tipo (vista, finalizar…).
 */

/**
 * Crea el servidor simulado.
 *
 * @param {object} [opciones]
 * @param {number} [opciones.intervalo] Milisegundos entre pasos de cada tarea.
 * @param {() => number} [opciones.reloj] Segundos actuales (por defecto, Date.now() / 1000).
 * @returns {{crear: (tipo: string, argumentos: object, crearTarea: (canal: CanalSimulado)
 *   => TareaSimulada) => object, listar: Function, obtener: Function, seguir: Function}}
 */
export function crearServidorTrabajosSimulado({ intervalo = 200, reloj = () => Date.now() / 1000 } = {}) {
  const trabajos = new Map();
  let contador = 0;

  const foto = (t) => JSON.parse(JSON.stringify(t.datos));
  const avisar = (t, nombre, ...argumentos) => {
    for (const s of [...t.suscriptores]) {
      s[nombre]?.(...argumentos);
    }
  };
  const cerrarSuscriptores = (t) => {
    for (const s of [...t.suscriptores]) {
      s.alCambiarEstado?.(ESTADOS_CONEXION.CERRADA);
    }
    t.suscriptores.clear();
  };
  const cambiarEstado = (t, estado, extra = {}) => {
    Object.assign(t.datos, { estado }, extra);
    avisar(t, 'alEstado', foto(t));
    if (ESTADOS_FINALES.has(estado)) {
      clearInterval(t.reloj);
      t.reloj = null;
      cerrarSuscriptores(t);
    }
  };
  const arrancar = (t) => {
    t.reloj = setInterval(() => t.tarea.paso(), intervalo);
  };

  return {
    crear(tipo, argumentos, crearTarea) {
      contador += 1;
      const t = {
        datos: { id: `simulado-${contador}`, tipo, estado: ESTADOS_TRABAJO.EN_MARCHA,
          argumentos: JSON.parse(JSON.stringify(argumentos)), interactivo: false, creado: reloj(),
          iniciado: reloj(), terminado: null, progresos: 0, resultado: null, error: null },
        progreso: [], imagen: null, suscriptores: new Set(), reloj: null,
      };
      t.tarea = crearTarea({
        progreso(medidas) {
          t.progreso.push({ ...medidas });
          t.datos.progresos += 1;
          avisar(t, 'alProgreso', { ...medidas });
        },
        imagen(imagen) {
          t.imagen = imagen;
          avisar(t, 'alImagen', imagen);
        },
        terminar: (resultado) => cambiarEstado(t, ESTADOS_TRABAJO.TERMINADO,
          { resultado, terminado: reloj() }),
        fallar: (mensaje) => cambiarEstado(t, ESTADOS_TRABAJO.FALLIDO,
          { error: mensaje, terminado: reloj() }),
      });
      trabajos.set(t.datos.id, t);
      arrancar(t);
      return foto(t);
    },

    listar: ({ tipo } = {}) => [...trabajos.values()]
      .filter((t) => !tipo || t.datos.tipo === tipo).map(foto),

    obtener: (id) => (trabajos.has(id) ? foto(trabajos.get(id)) : null),

    seguir(id, callbacks = {}) {
      const t = trabajos.get(id);
      let activa = true;
      const propios = { ...callbacks };
      // Como el WebSocket real: todo llega después de abrir, nunca dentro de la llamada.
      setTimeout(() => {
        if (!activa) {
          return;
        }
        callbacks.alCambiarEstado?.(ESTADOS_CONEXION.ABIERTA);
        if (!t) {
          callbacks.alError?.(new Error(`No existe el trabajo ${id}`));
          callbacks.alCambiarEstado?.(ESTADOS_CONEXION.CERRADA);
          activa = false;
          return;
        }
        callbacks.alEstado?.(foto(t));
        callbacks.alHistorial?.({ progreso: t.progreso.map((m) => ({ ...m })), eventos: [] });
        if (t.imagen) {
          callbacks.alImagen?.(t.imagen);
        }
        if (ESTADOS_FINALES.has(t.datos.estado)) {
          callbacks.alCambiarEstado?.(ESTADOS_CONEXION.CERRADA);
          activa = false;
          return;
        }
        t.suscriptores.add(propios);
      }, 0);
      callbacks.alCambiarEstado?.(ESTADOS_CONEXION.CONECTANDO);

      const enviar = (accion) => {
        if (!activa || !t || ESTADOS_FINALES.has(t.datos.estado)) {
          return false;
        }
        accion();
        return true;
      };
      return {
        pausar: () => enviar(() => {
          if (t.datos.estado === ESTADOS_TRABAJO.EN_MARCHA) {
            clearInterval(t.reloj);
            t.reloj = null;
            cambiarEstado(t, ESTADOS_TRABAJO.PAUSADO);
          }
        }),
        reanudar: () => enviar(() => {
          if (t.datos.estado === ESTADOS_TRABAJO.PAUSADO) {
            arrancar(t);
            cambiarEstado(t, ESTADOS_TRABAJO.EN_MARCHA);
          }
        }),
        cancelar: () => enviar(() => cambiarEstado(t, ESTADOS_TRABAJO.CANCELADO,
          { terminado: reloj() })),
        ordenar: (orden) => enviar(() => {
          // Como la API real: finalizar en pausa reanuda para poder terminar.
          if (orden?.tipo === 'finalizar' && t.datos.estado === ESTADOS_TRABAJO.PAUSADO) {
            arrancar(t);
            cambiarEstado(t, ESTADOS_TRABAJO.EN_MARCHA);
          }
          t.tarea.orden?.(orden);
        }),
        cerrar() {
          activa = false;
          t?.suscriptores.delete(propios);
        },
      };
    },
  };
}
