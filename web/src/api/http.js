/**
 * Cliente HTTP de la API del backend.
 *
 * Las rutas son relativas a `base` (por defecto `/api`, el mismo origen que la interfaz:
 * en desarrollo lo reenvía el proxy de Vite y en producción lo sirve FastAPI). Las
 * respuestas JSON se devuelven ya decodificadas y cualquier fallo, de red o de la API,
 * llega como `ErrorApi`.
 */

export const BASE_API = '/api';

/** Error de una llamada a la API, con el código HTTP (0 si no hubo respuesta). */
export class ErrorApi extends Error {
  /**
   * @param {string} mensaje Descripción para mostrar.
   * @param {object} datos
   * @param {number} datos.estado Código HTTP, o 0 si la petición no llegó a responderse.
   * @param {unknown} [datos.detalle] Cuerpo de error devuelto por la API, si lo hay.
   * @param {unknown} [datos.causa] Error original (por ejemplo, el de red).
   */
  constructor(mensaje, { estado = 0, detalle = null, causa } = {}) {
    super(mensaje, causa === undefined ? undefined : { cause: causa });
    this.name = 'ErrorApi';
    this.estado = estado;
    this.detalle = detalle;
  }
}

/**
 * Comprueba que una ruta de la API es una cadena que empieza por `/`.
 *
 * @param {unknown} ruta Ruta que se va a pedir, como `/entornos`.
 * @returns {string} La misma ruta.
 * @throws {TypeError} Si no es una cadena o no empieza por `/`.
 */
export function comprobarRuta(ruta) {
  if (typeof ruta !== 'string' || !ruta.startsWith('/') || ruta.startsWith('//')) {
    throw new TypeError(`La ruta de la API debe empezar por una sola «/»: ${String(ruta)}`);
  }
  return ruta;
}

/**
 * Une la base y la ruta y añade los parámetros de consulta.
 *
 * @param {string} base Prefijo de la API, con o sin «/» final.
 * @param {string} ruta Ruta que empieza por «/».
 * @param {Record<string, string | number | boolean | null | undefined>} [parametros]
 *   Parámetros de consulta; los `null` y `undefined` se omiten.
 * @returns {string} URL relativa o absoluta, según sea `base`.
 */
export function construirUrl(base, ruta, parametros = {}) {
  const url = `${base.replace(/\/+$/, '')}${comprobarRuta(ruta)}`;
  const consulta = new URLSearchParams();
  for (const [clave, valor] of Object.entries(parametros)) {
    if (valor !== null && valor !== undefined) {
      consulta.append(clave, String(valor));
    }
  }
  const texto = consulta.toString();
  if (!texto) {
    return url;
  }
  return `${url}${url.includes('?') ? '&' : '?'}${texto}`;
}

/**
 * Extrae un mensaje legible del cuerpo de error de FastAPI (`{"detail": ...}`).
 *
 * @param {unknown} cuerpo Cuerpo decodificado de la respuesta.
 * @param {number} estado Código HTTP.
 * @returns {string} Mensaje para el usuario.
 */
export function mensajeDeError(cuerpo, estado) {
  const detalle = cuerpo && typeof cuerpo === 'object' ? cuerpo.detail : cuerpo;
  if (typeof detalle === 'string' && detalle.trim()) {
    return detalle;
  }
  if (Array.isArray(detalle)) {
    const mensajes = detalle
      .map((error) => (error && typeof error.msg === 'string' ? error.msg : null))
      .filter(Boolean);
    if (mensajes.length) {
      return mensajes.join('; ');
    }
  }
  return `La API respondió con el código ${estado}`;
}

// Un cuerpo vacío (como el de un 204) da null.
async function leerCuerpo(respuesta) {
  const texto = await respuesta.text();
  if (!texto) {
    return null;
  }
  const tipo = respuesta.headers.get('content-type') ?? '';
  if (tipo.includes('json')) {
    try {
      return JSON.parse(texto);
    } catch (error) {
      if (respuesta.ok) {
        throw new ErrorApi('La API devolvió un JSON no válido', {
          estado: respuesta.status,
          detalle: texto,
          causa: error,
        });
      }
      return texto;
    }
  }
  return texto;
}

/**
 * @typedef {object} OpcionesPeticion
 * @property {Record<string, string | number | boolean | null | undefined>} [parametros]
 *   Parámetros de consulta.
 * @property {AbortSignal} [senal] Señal para cancelar la petición.
 */

/**
 * @typedef {object} ClienteHttp
 * @property {(ruta: string, opciones?: OpcionesPeticion) => Promise<unknown>} obtener GET.
 * @property {(ruta: string, cuerpo?: unknown, opciones?: OpcionesPeticion) => Promise<unknown>}
 *   enviar POST con cuerpo JSON.
 * @property {(ruta: string, cuerpo?: unknown, opciones?: OpcionesPeticion) => Promise<unknown>}
 *   actualizar PATCH con cuerpo JSON.
 * @property {(ruta: string, opciones?: OpcionesPeticion) => Promise<unknown>} borrar DELETE.
 */

/**
 * Crea un cliente HTTP para la API.
 *
 * @param {object} [opciones]
 * @param {string} [opciones.base] Prefijo de las rutas (por defecto `/api`).
 * @param {typeof globalThis.fetch} [opciones.fetch] Implementación de `fetch` (los tests
 *   pasan una simulada).
 * @returns {ClienteHttp} Cliente con un método por verbo HTTP.
 */
export function crearClienteHttp({ base = BASE_API, fetch: hacerFetch } = {}) {
  const pedir = async (metodo, ruta, cuerpo, { parametros, senal } = {}) => {
    const url = construirUrl(base, ruta, parametros);
    const funcion = hacerFetch ?? globalThis.fetch;
    const cabeceras = { Accept: 'application/json' };
    const init = { method: metodo, headers: cabeceras, signal: senal };
    if (cuerpo !== undefined) {
      cabeceras['Content-Type'] = 'application/json';
      init.body = JSON.stringify(cuerpo);
    }
    let respuesta;
    try {
      respuesta = await funcion(url, init);
    } catch (error) {
      if (error?.name === 'AbortError') {
        throw error;
      }
      throw new ErrorApi('No se pudo conectar con la API', { estado: 0, causa: error });
    }
    const datos = await leerCuerpo(respuesta);
    if (!respuesta.ok) {
      throw new ErrorApi(mensajeDeError(datos, respuesta.status), {
        estado: respuesta.status,
        detalle: datos,
      });
    }
    return datos;
  };

  return {
    obtener: (ruta, opciones) => pedir('GET', ruta, undefined, opciones),
    enviar: (ruta, cuerpo, opciones) => pedir('POST', ruta, cuerpo, opciones),
    actualizar: (ruta, cuerpo, opciones) => pedir('PATCH', ruta, cuerpo, opciones),
    borrar: (ruta, opciones) => pedir('DELETE', ruta, undefined, opciones),
  };
}
