/**
 * Formulario de hiperparámetros: convierte el esquema del backend en campos de texto,
 * comprueba lo que escribe el usuario y construye la capa de ajustes que se envía, solo
 * con lo que cambia respecto al perfil y anidada como en configs/.
 */

const ENTERO = /^[+-]?\d+$/;
const REAL = /^[+-]?(\d+(\.\d*)?|\.\d+)(e[+-]?\d+)?$/i;

/**
 * Identificador de un campo en el formulario: ruta de su sección y su clave.
 *
 * @param {{clave: string}} seccion
 * @param {{clave: string}} campo
 * @returns {string} Por ejemplo gail.discriminador.learning_rate.
 */
export const idCampo = (seccion, campo) => `${seccion.clave}.${campo.clave}`;

/**
 * Valor del formulario para un valor del perfil: texto, salvo en los booleanos.
 *
 * @param {import('../api/entrenamiento.js').CampoAjuste} campo
 * @param {unknown} valor
 * @returns {string | boolean}
 */
export function aFormulario(campo, valor) {
  if (campo.tipo === 'booleano') {
    return Boolean(valor);
  }
  return Array.isArray(valor) ? valor.join(', ') : String(valor);
}

/**
 * Valores iniciales del formulario: los del perfil.
 *
 * @param {import('../api/entrenamiento.js').EsquemaAjustes} esquema
 * @returns {Record<string, string | boolean>}
 */
export function valoresIniciales(esquema) {
  return Object.fromEntries(esquema.secciones.flatMap((seccion) => seccion.campos
    .map((campo) => [idCampo(seccion, campo), aFormulario(campo, campo.valor)])));
}

/**
 * Comprueba que los números estén entre el mínimo y el máximo del campo, si los tiene
 * (el esquema los omite cuando no hay límite).
 *
 * @param {import('../api/entrenamiento.js').CampoAjuste} campo
 * @param {number[]} numeros
 * @param {string} sujeto «Tiene» para un número, «Cada número tiene» para una lista.
 * @returns {string | null} El error, o null si están dentro.
 */
function fueraDeRango(campo, numeros, sujeto) {
  if (campo.min != null && numeros.some((n) => n < campo.min)) {
    return `${sujeto} que ser como mínimo ${campo.min}.`;
  }
  if (campo.max != null && numeros.some((n) => n > campo.max)) {
    return `${sujeto} que ser como mucho ${campo.max}.`;
  }
  return null;
}

const enRango = (campo, numero) => {
  const error = fueraDeRango(campo, [numero], 'Tiene');
  return error ? { error } : { valor: numero };
};

/**
 * Lee lo escrito en un campo según su tipo.
 *
 * @param {import('../api/entrenamiento.js').CampoAjuste} campo
 * @param {string | boolean} entrada Texto del campo, o booleano en las casillas.
 * @returns {{valor: unknown} | {error: string}}
 */
export function leerCampo(campo, entrada) {
  if (campo.tipo === 'booleano') {
    return typeof entrada === 'boolean' ? { valor: entrada } : { error: 'Tiene que ser sí o no.' };
  }
  const texto = String(entrada ?? '').trim();
  if (campo.tipo === 'entero') {
    const numero = Number(texto);
    if (!ENTERO.test(texto) || !Number.isSafeInteger(numero)) {
      return { error: 'Escribe un número entero.' };
    }
    return enRango(campo, numero);
  }
  if (campo.tipo === 'real') {
    // Se admite la coma decimal: «0,01» es 0.01.
    const normalizado = texto.replace(',', '.');
    const numero = Number(normalizado);
    if (!REAL.test(normalizado) || !Number.isFinite(numero)) {
      return { error: 'Escribe un número, como 0.01 o 1e-3.' };
    }
    return enRango(campo, numero);
  }
  if (campo.tipo === 'lista_enteros') {
    const partes = texto.split(/[\s,;]+/).filter(Boolean);
    if (!partes.length || !partes.every((parte) => ENTERO.test(parte))) {
      return { error: 'Escribe números enteros separados por comas, como 100, 100.' };
    }
    const numeros = partes.map(Number);
    const error = fueraDeRango(campo, numeros, 'Cada número tiene');
    return error ? { error } : { valor: numeros };
  }
  if (campo.tipo === 'opcion') {
    return (campo.opciones ?? []).includes(texto) ? { valor: texto }
      : { error: `Elige una de: ${(campo.opciones ?? []).join(', ')}.` };
  }
  return { error: `Tipo de campo desconocido: ${campo.tipo}` };
}

const iguales = (a, b) => JSON.stringify(a) === JSON.stringify(b);

/**
 * Comprueba todo el formulario y construye los ajustes que cambian respecto al perfil.
 *
 * @param {import('../api/entrenamiento.js').EsquemaAjustes} esquema
 * @param {Record<string, string | boolean>} valores Valores del formulario por idCampo.
 * @returns {{valido: boolean, errores: Record<string, string>, ajustes: object,
 *   cambiados: number}}
 */
export function validarAjustes(esquema, valores) {
  const errores = {};
  const ajustes = {};
  let cambiados = 0;
  for (const seccion of esquema.secciones) {
    for (const campo of seccion.campos) {
      const id = idCampo(seccion, campo);
      const leido = leerCampo(campo, valores[id] ?? aFormulario(campo, campo.valor));
      if ('error' in leido) {
        errores[id] = leido.error;
      } else if (!iguales(leido.valor, campo.valor)) {
        cambiados += 1;
        const ruta = seccion.clave.split('.');
        let destino = ajustes;
        for (const parte of ruta) {
          destino[parte] ??= {};
          destino = destino[parte];
        }
        destino[campo.clave] = leido.valor;
      }
    }
  }
  return { valido: Object.keys(errores).length === 0, errores, ajustes, cambiados };
}

/**
 * Campos de un esquema por su ruta completa en configs/, para comprobar ajustes ya
 * construidos (lo usa el backend simulado, como hará el real).
 *
 * @param {import('../api/entrenamiento.js').EsquemaAjustes} esquema
 * @returns {Map<string, import('../api/entrenamiento.js').CampoAjuste>}
 */
export function camposPorRuta(esquema) {
  return new Map(esquema.secciones.flatMap((seccion) => seccion.campos
    .map((campo) => [idCampo(seccion, campo), campo])));
}

/**
 * Recorre unos ajustes anidados y devuelve sus hojas como [ruta, valor].
 *
 * @param {object} ajustes
 * @param {string} [prefijo]
 * @returns {[string, unknown][]}
 */
export function hojasDeAjustes(ajustes, prefijo = '') {
  if (!ajustes || typeof ajustes !== 'object' || Array.isArray(ajustes)) {
    throw new TypeError('Los ajustes deben ser un objeto anidado como en configs/');
  }
  return Object.entries(ajustes).flatMap(([clave, valor]) => {
    const ruta = prefijo ? `${prefijo}.${clave}` : clave;
    return valor && typeof valor === 'object' && !Array.isArray(valor)
      ? hojasDeAjustes(valor, ruta) : [[ruta, valor]];
  });
}
