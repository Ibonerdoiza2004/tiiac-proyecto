/**
 * Contrato con el backend de «Jugar tú» y «Ver agente» (laboratorio-jugar-api).
 *
 * Es una propuesta pendiente de acordar con el backend. En el frontend, todas las rutas y
 * los mensajes están en este archivo, así que cambiarlos solo toca aquí; en el backend, el
 * contrato está en el código de src/app (la documentación HTTP sale en /docs).
 *
 * HTTP (prefijo /api):
 * - GET  /laboratorio/{entorno}            → ConfiguracionJuego
 * - POST /sesiones  {entorno, modo, partidas?} → Sesion
 * - POST /sesiones/{id}/guardar {nombre, partidas: [{indice, tramo: [inicio, fin]}]}
 *                                          → ResultadoGuardado
 * - DELETE /sesiones/{id}                  → 204, descarta la sesión sin guardar
 *
 * WebSocket /api/sesiones/{id}:
 * - Del navegador: {tipo: 'accion', accion: entero | null} con la acción que se mantiene
 *   pulsada (null si no hay ninguna) y {tipo: 'finalizar'}.
 * - Del servidor: cada imagen del entorno como JPEG binario, y los mensajes JSON
 *   {tipo: 'estado', ...EstadoJuego}, {tipo: 'partida_terminada', partida: Partida},
 *   {tipo: 'sesion_terminada', partidas: Partida[]} y {tipo: 'error', mensaje}.
 *
 * Los tramos de pares van desde 0 y sin incluir el final, como en src/data: [0, 20]
 * son los pares que la interfaz muestra como 1–20.
 */

import { crearClienteHttp } from './http.js';
import { conectarWebSocket } from './websocket.js';

/** Modos de una sesión. */
export const MODOS = Object.freeze({ MANUAL: 'manual', EXPERTO: 'experto' });

/** Tipos de los mensajes JSON que envía el servidor por el WebSocket. */
export const MENSAJES_SERVIDOR = Object.freeze({
  ESTADO: 'estado',
  PARTIDA_TERMINADA: 'partida_terminada',
  SESION_TERMINADA: 'sesion_terminada',
  ERROR: 'error',
});

/**
 * @typedef {object} ConfiguracionJuego
 * @property {number} fps Imágenes por segundo (render_fps del entorno: velocidad ×1).
 * @property {number} partidas_experto Partidas por defecto al grabar con el experto (las
 *   del perfil: partidas_train + partidas_validacion).
 */

/**
 * @typedef {object} Sesion
 * @property {string} id Identificador de la sesión.
 * @property {string} entorno Clave del entorno.
 * @property {'manual' | 'experto'} modo
 * @property {number} fps Imágenes por segundo.
 * @property {number | null} partidas Partidas pedidas (null al jugar a mano: hasta que pare).
 */

/**
 * @typedef {object} EstadoJuego
 * @property {number} partida Número de la partida en curso, desde 1.
 * @property {number} paso Pasos jugados en la partida en curso.
 * @property {number} retorno Recompensa acumulada en la partida en curso.
 * @property {number} pares Pares de las partidas terminadas más los de la partida en curso.
 * @property {boolean} esperando Si la partida está parada esperando una tecla.
 */

/**
 * @typedef {object} Partida
 * @property {number} indice Posición en la sesión, desde 0.
 * @property {number} pasos Pasos (pares) de la partida.
 * @property {number} retorno Recompensa total.
 * @property {number} semilla Semilla con la que empezó.
 */

/**
 * @typedef {object} ResultadoGuardado
 * @property {string} nombre Nombre del conjunto guardado en la biblioteca.
 * @property {number} partidas Partidas guardadas.
 * @property {number} pares Pares guardados.
 */

/**
 * @typedef {object} CallbacksSesion
 * @property {(estado: EstadoJuego) => void} [alEstado]
 * @property {(imagen: Blob) => void} [alImagen]
 * @property {(partida: Partida) => void} [alTerminarPartida]
 * @property {(partidas: Partida[]) => void} [alTerminarSesion]
 * @property {(error: Error) => void} [alError]
 * @property {(estado: string, cierre?: import('./websocket.js').Cierre) => void} [alCambiarEstado]
 *   Recibe los cambios de la conexión; al cerrarse, también cómo terminó
 *   (ESTADOS_CONEXION de ./websocket.js).
 */

/**
 * @typedef {object} ConexionJuego
 * @property {(accion: number | null) => boolean} enviarAccion Acción mantenida, o null.
 * @property {() => boolean} finalizar Pide terminar la sesión.
 * @property {() => void} cerrar Cierra la conexión sin avisar al servidor.
 */

/**
 * @typedef {object} ApiLaboratorio
 * @property {(entorno: string) => Promise<ConfiguracionJuego>} configuracion
 * @property {(datos: {entorno: string, modo: string, partidas?: number}) => Promise<Sesion>}
 *   crearSesion
 * @property {(id: string, callbacks: CallbacksSesion) => ConexionJuego} conectar
 * @property {(id: string, datos: {nombre: string,
 *   partidas: {indice: number, tramo: [number, number]}[]}) => Promise<ResultadoGuardado>}
 *   guardar
 * @property {(id: string) => Promise<void>} descartar
 */

/**
 * Reparte un mensaje JSON del servidor entre los callbacks de la sesión.
 *
 * @param {unknown} mensaje Mensaje ya decodificado.
 * @param {CallbacksSesion} callbacks
 */
export function repartirMensaje(mensaje, callbacks) {
  const { alEstado, alTerminarPartida, alTerminarSesion, alError } = callbacks;
  const tipo = mensaje && typeof mensaje === 'object' ? mensaje.tipo : undefined;
  if (tipo === MENSAJES_SERVIDOR.ESTADO) {
    const { tipo: _, ...estado } = mensaje;
    alEstado?.(estado);
  } else if (tipo === MENSAJES_SERVIDOR.PARTIDA_TERMINADA) {
    alTerminarPartida?.(mensaje.partida);
  } else if (tipo === MENSAJES_SERVIDOR.SESION_TERMINADA) {
    alTerminarSesion?.(mensaje.partidas);
  } else if (tipo === MENSAJES_SERVIDOR.ERROR) {
    alError?.(new Error(String(mensaje.mensaje ?? 'Error del servidor')));
  } else {
    alError?.(new Error(`Mensaje desconocido del servidor: ${JSON.stringify(mensaje)}`));
  }
}

/**
 * Crea la API del laboratorio sobre el backend real.
 *
 * @param {object} [opciones]
 * @param {import('./http.js').ClienteHttp} [opciones.http] Cliente HTTP.
 * @param {typeof conectarWebSocket} [opciones.conectar] Abre un WebSocket (los tests
 *   pasan uno simulado).
 * @returns {ApiLaboratorio}
 */
export function crearApiLaboratorio({ http = crearClienteHttp(), conectar = conectarWebSocket } = {}) {
  const ruta = (id) => `/sesiones/${encodeURIComponent(id)}`;
  return {
    configuracion: (entorno) => http.obtener(`/laboratorio/${encodeURIComponent(entorno)}`),
    crearSesion: (datos) => http.enviar('/sesiones', datos),
    conectar(id, callbacks = {}) {
      const conexion = conectar(ruta(id), {
        alMensaje: (mensaje) => repartirMensaje(mensaje, callbacks),
        alImagen: (imagen) => callbacks.alImagen?.(imagen),
        alError: (error) => callbacks.alError?.(error),
        alCambiarEstado: (estado, ...cierre) => callbacks.alCambiarEstado?.(estado, ...cierre),
      });
      return {
        enviarAccion: (accion) => conexion.enviar({ tipo: 'accion', accion }),
        finalizar: () => conexion.enviar({ tipo: 'finalizar' }),
        cerrar: () => conexion.cerrar(),
        conectando: () => conexion.conectando?.() ?? false,
      };
    },
    guardar: (id, datos) => http.enviar(`${ruta(id)}/guardar`, datos),
    descartar: async (id) => {
      await http.borrar(ruta(id));
    },
  };
}
