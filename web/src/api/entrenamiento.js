/**
 * Contrato con el backend de «Entrenar experto» y «Entrenar e imitar»
 * (laboratorio-entrenar-api) y la lista de demostraciones de la biblioteca.
 *
 * Es una propuesta pendiente de acordar con el backend. En el frontend, todas las rutas y
 * los mensajes están en este archivo, así que cambiarlos solo toca aquí; en el backend, el
 * contrato está en el código de src/app (la documentación HTTP sale en /docs).
 *
 * HTTP (prefijo /api):
 * - GET  /laboratorio/{entorno}/ajustes/{metodo}   → EsquemaAjustes (valores del perfil)
 * - GET  /biblioteca/demostraciones?entorno={entorno} → Demostracion[]
 * - POST /entrenamientos {entorno, metodo, ajustes, datos?, vista, wandb} → Entrenamiento
 *
 * WebSocket /api/entrenamientos/{id}:
 * - Del navegador: {tipo: 'pausar'}, {tipo: 'reanudar'}, {tipo: 'vista', activa} y
 *   {tipo: 'finalizar'}.
 * - Del servidor: imágenes JPEG binarias (solo con la vista encendida) y los mensajes
 *   JSON {tipo: 'progreso', paso, medidas}, {tipo: 'estado', estado}, {tipo: 'terminado',
 *   modelo, resumen} y {tipo: 'error', mensaje}.
 *
 * metodo es 'experto', 'bc' o 'gail'. ajustes solo lleva lo que cambia el usuario, anidado
 * como en configs/ (por ejemplo {gail: {discriminador: {learning_rate: 0.02}}}): el backend
 * lo aplica como una capa más sobre el perfil.
 */

import { crearClienteHttp } from './http.js';
import { conectarWebSocket } from './websocket.js';

/** Métodos que se pueden entrenar. */
export const METODOS = Object.freeze({ EXPERTO: 'experto', BC: 'bc', GAIL: 'gail' });

/** Métodos que juegan partidas mientras entrenan y por eso admiten la vista. */
export const METODOS_CON_VISTA = Object.freeze(new Set([METODOS.EXPERTO, METODOS.GAIL]));

/** Métodos que imitan demostraciones y por eso necesitan datos. */
export const METODOS_CON_DATOS = Object.freeze(new Set([METODOS.BC, METODOS.GAIL]));

/** Tipos de los mensajes JSON que envía el servidor por el WebSocket. */
export const MENSAJES_ENTRENAMIENTO = Object.freeze({
  PROGRESO: 'progreso',
  ESTADO: 'estado',
  TERMINADO: 'terminado',
  ERROR: 'error',
});

/** Estados del entrenamiento que envía el servidor. */
export const ESTADOS_ENTRENAMIENTO = Object.freeze({ EN_MARCHA: 'en_marcha', PAUSADO: 'pausado' });

/**
 * @typedef {object} CampoAjuste
 * @property {string} clave Nombre de la clave en configs/, como learning_rate.
 * @property {'entero' | 'real' | 'booleano' | 'opcion' | 'lista_enteros'} tipo
 * @property {number | boolean | string | number[]} valor Valor del perfil.
 * @property {number} [min] Mínimo admitido (enteros, reales y listas); falta si no hay.
 * @property {number} [max] Máximo admitido; falta si no hay.
 * @property {string[]} [opciones] Valores admitidos (tipo opcion).
 * @property {string} [ayuda] De dónde sale el valor (el comentario de configs/).
 */

/**
 * @typedef {object} SeccionAjustes
 * @property {string} clave Ruta de la sección en configs/, como gail.discriminador.
 * @property {CampoAjuste[]} campos
 */

/**
 * @typedef {object} EsquemaAjustes
 * @property {string} metodo
 * @property {{unidad: string, total: number, campo?: string, bloque?: string}} duracion
 *   Cuánto dura con los valores del perfil y de qué campos sale (DuracionApi en
 *   src/app/contrato.py): el valor de campo o, con bloque, ese valor redondeado hacia
 *   arriba a un múltiplo del de bloque.
 * @property {SeccionAjustes[]} secciones
 */

/**
 * @typedef {object} Demostracion
 * @property {string} id Identificador en la biblioteca.
 * @property {string} nombre
 * @property {'conjunto' | 'subconjunto'} tipo
 * @property {number} partidas
 * @property {number | null} pares
 * @property {string} fuente 'experto', 'humano' o 'combinado'.
 */

/**
 * @typedef {object} Entrenamiento
 * @property {string} id
 * @property {string} entorno
 * @property {string} metodo
 * @property {{unidad: string, total: number}} duracion Con los ajustes pedidos.
 */

/**
 * @typedef {object} CallbacksEntrenamiento
 * @property {(progreso: {paso: number, medidas: Record<string, number>}) => void} [alProgreso]
 * @property {(imagen: Blob) => void} [alImagen]
 * @property {(estado: string) => void} [alEstado] ESTADOS_ENTRENAMIENTO.
 * @property {(fin: {modelo: {nombre: string, incompleto: boolean}, resumen: object}) => void}
 *   [alTerminar]
 * @property {(error: Error) => void} [alError]
 * @property {(estado: string, cierre?: import('./websocket.js').Cierre) => void} [alCambiarEstado]
 *   Cambios de la conexión y, al cerrarse, cómo terminó
 *   (ESTADOS_CONEXION de ./websocket.js).
 */

/**
 * @typedef {object} ConexionEntrenamiento
 * @property {() => boolean} pausar
 * @property {() => boolean} reanudar
 * @property {(activa: boolean) => boolean} cambiarVista
 * @property {() => boolean} finalizar
 * @property {() => void} cerrar Cierra la conexión sin avisar al servidor.
 */

/**
 * @typedef {object} ApiEntrenamiento
 * @property {(entorno: string, metodo: string, opciones?: {senal?: AbortSignal})
 *   => Promise<EsquemaAjustes>} ajustes
 * @property {(entorno: string, opciones?: {senal?: AbortSignal}) => Promise<Demostracion[]>}
 *   demostraciones
 * @property {(datos: object, opciones?: {senal?: AbortSignal}) => Promise<Entrenamiento>} crear
 * @property {(id: string, callbacks: CallbacksEntrenamiento) => ConexionEntrenamiento} conectar
 */

/**
 * Reparte un mensaje JSON del servidor entre los callbacks del entrenamiento.
 *
 * @param {unknown} mensaje Mensaje ya decodificado.
 * @param {CallbacksEntrenamiento} callbacks
 */
export function repartirMensajeEntrenamiento(mensaje, callbacks) {
  const tipo = mensaje && typeof mensaje === 'object' ? mensaje.tipo : undefined;
  if (tipo === MENSAJES_ENTRENAMIENTO.PROGRESO) {
    callbacks.alProgreso?.({ paso: mensaje.paso, medidas: mensaje.medidas ?? {} });
  } else if (tipo === MENSAJES_ENTRENAMIENTO.ESTADO) {
    callbacks.alEstado?.(mensaje.estado);
  } else if (tipo === MENSAJES_ENTRENAMIENTO.TERMINADO) {
    callbacks.alTerminar?.({ modelo: mensaje.modelo, resumen: mensaje.resumen ?? {} });
  } else if (tipo === MENSAJES_ENTRENAMIENTO.ERROR) {
    callbacks.alError?.(new Error(String(mensaje.mensaje ?? 'Error del servidor')));
  } else {
    callbacks.alError?.(new Error(`Mensaje desconocido del servidor: ${JSON.stringify(mensaje)}`));
  }
}

/**
 * Crea la API de entrenamiento sobre el backend real.
 *
 * @param {object} [opciones]
 * @param {import('./http.js').ClienteHttp} [opciones.http] Cliente HTTP.
 * @param {typeof conectarWebSocket} [opciones.conectar] Abre un WebSocket (los tests pasan
 *   uno simulado).
 * @returns {ApiEntrenamiento}
 */
export function crearApiEntrenamiento({ http = crearClienteHttp(), conectar = conectarWebSocket } = {}) {
  const codificar = encodeURIComponent;
  return {
    ajustes: (entorno, metodo, { senal } = {}) =>
      http.obtener(`/laboratorio/${codificar(entorno)}/ajustes/${codificar(metodo)}`, { senal }),
    demostraciones: (entorno, { senal } = {}) =>
      http.obtener('/biblioteca/demostraciones', { parametros: { entorno }, senal }),
    crear: (datos, { senal } = {}) => http.enviar('/entrenamientos', datos, { senal }),
    conectar(id, callbacks = {}) {
      const conexion = conectar(`/entrenamientos/${codificar(id)}`, {
        alMensaje: (mensaje) => repartirMensajeEntrenamiento(mensaje, callbacks),
        alImagen: (imagen) => callbacks.alImagen?.(imagen),
        alError: (error) => callbacks.alError?.(error),
        alCambiarEstado: (estado, ...cierre) => callbacks.alCambiarEstado?.(estado, ...cierre),
      });
      return {
        pausar: () => conexion.enviar({ tipo: 'pausar' }),
        reanudar: () => conexion.enviar({ tipo: 'reanudar' }),
        cambiarVista: (activa) => conexion.enviar({ tipo: 'vista', activa: Boolean(activa) }),
        finalizar: () => conexion.enviar({ tipo: 'finalizar' }),
        cerrar: () => conexion.cerrar(),
        conectando: () => conexion.conectando?.() ?? false,
      };
    },
  };
}
