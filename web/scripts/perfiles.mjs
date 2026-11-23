/**
 * Genera src/simulado/perfiles.json: los hiperparámetros de cada perfil de configs/, con su
 * tipo y de dónde sale cada valor, para que el backend simulado del Laboratorio los ofrezca
 * igual que lo hará el real (GET /api/laboratorio/{entorno}/ajustes/{metodo}).
 *
 * Así el frontend no escribe ningún hiperparámetro a mano: si cambia un perfil, se vuelve a
 * generar con `npm run perfiles` (un test falla mientras el JSON no esté al día).
 *
 * Resuelve como src/utils/configuracion.py: default.yaml y encima el perfil del entorno;
 * para el experto, además, experto.trpo encima de trpo.
 */

import { readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { isMap, isScalar, isSeq, parseDocument } from 'yaml';

const RAIZ = join(dirname(fileURLToPath(import.meta.url)), '..', '..', 'configs');
const SALIDA = join(dirname(fileURLToPath(import.meta.url)), '..', 'src', 'simulado', 'perfiles.json');

/**
 * Valores admitidos de los campos de texto, por clave o por sección y clave (que manda).
 * Copian ACTIVACIONES (src/models/factory.py, también para trpo), OPTIMIZADORES
 * (src/training/bc.py), el único optimizador del discriminador (src/training/gail.py) y
 * MODOS_RECOMPENSA (src/training/gail.py); un test comprueba que siguen coincidiendo.
 */
export const OPCIONES = Object.freeze({
  activacion: ['relu', 'tanh'],
  'bc.optimizador': ['adam', 'rmsprop', 'sgd'],
  'gail.discriminador.optimizador': ['adam'],
  recompensa: ['log_d', 'menos_log_uno_menos_d'],
});

// retorno_objetivo es entero en unos perfiles y real en otros (Reacher: −3,5).
const SIEMPRE_REALES = new Set(['retorno_objetivo']);
// Enteros que pueden no ser positivos.
const ENTEROS_SIN_MINIMO = new Set(['retorno_objetivo']);
// Claves que no son hiperparámetros de ese método o que se ajustan aparte.
const EXCLUIDAS = { experto: new Set(['semilla', 'trpo']), gail: new Set(['discriminador']) };

const limpiar = (texto) => (texto ?? '').replace(/\s+/g, ' ').trim() || undefined;

/** Convierte un nodo YAML en {valor, ayuda} recursivo, con el comentario de cada clave. */
function leerNodo(nodo) {
  if (isMap(nodo)) {
    return Object.fromEntries(nodo.items.map((par, indice) => {
      const hijo = leerNodo(par.value);
      if (!isMap(par.value)) {
        // El comentario de encima de la primera clave lo guarda la propia sección.
        hijo.ayuda = limpiar(par.value?.comment) ?? limpiar(par.key?.commentBefore)
          ?? (indice === 0 ? limpiar(nodo.commentBefore) : undefined);
      }
      return [String(par.key.value), hijo];
    }));
  }
  if (isSeq(nodo)) {
    return { valor: nodo.items.map((item) => (isScalar(item) ? item.value : item.toJSON())) };
  }
  // En YAML 0.0 y 10.0 son reales, pero en JavaScript son el mismo número que 0 y 10:
  // se apunta si estaban escritos como reales.
  const escritoReal = isScalar(nodo) && typeof nodo.value === 'number'
    && /[.eE]/.test(String(nodo.source ?? ''));
  return { valor: isScalar(nodo) ? nodo.value : nodo, ...(escritoReal ? { real: true } : {}) };
}

const esHoja = (valor) => valor && Object.hasOwn(valor, 'valor') && !Object.hasOwn(valor, 'campos');

/** Combina capa sobre base: las hojas de la capa sustituyen a las de la base. */
function combinar(base, capa) {
  const resultado = { ...base };
  for (const [clave, valor] of Object.entries(capa ?? {})) {
    resultado[clave] = esHoja(valor) || !base?.[clave] || esHoja(base[clave])
      ? valor : combinar(base[clave], valor);
  }
  return resultado;
}

function tipoDe(seccionClave, clave, valor, real) {
  if (typeof valor === 'boolean') {
    return { tipo: 'booleano' };
  }
  if (typeof valor === 'number') {
    if (real || SIEMPRE_REALES.has(clave) || !Number.isInteger(valor)) {
      return { tipo: 'real' };
    }
    return ENTEROS_SIN_MINIMO.has(clave) ? { tipo: 'entero' } : { tipo: 'entero', min: 1 };
  }
  if (Array.isArray(valor) && valor.every(Number.isInteger)) {
    return { tipo: 'lista_enteros', min: 1 };
  }
  const opciones = OPCIONES[`${seccionClave}.${clave}`] ?? OPCIONES[clave];
  if (typeof valor === 'string' && opciones) {
    return { tipo: 'opcion', opciones: [...opciones] };
  }
  throw new Error(`No sé qué tipo de campo es ${clave}: ${JSON.stringify(valor)}`);
}

function seccion(clave, nodos, excluidas = new Set()) {
  const campos = Object.entries(nodos)
    .filter(([nombre, nodo]) => esHoja(nodo) && !excluidas.has(nombre))
    .map(([nombre, { valor, ayuda, real }]) => {
      if (valor === null || valor === undefined) {
        throw new Error(`${clave}.${nombre} no tiene valor en el perfil`);
      }
      return { clave: nombre, ...tipoDe(clave, nombre, valor, real), valor, ...(ayuda ? { ayuda } : {}) };
    });
  return { clave, campos };
}

/**
 * Hiperparámetros de cada entorno y método, resueltos desde configs/.
 *
 * @param {string} [raiz] Carpeta configs/.
 * @returns {Record<string, Record<string, {metodo: string, duracion: {unidad: string,
 *   total: number}, secciones: object[]}>>}
 */
export function generarPerfiles(raiz = RAIZ) {
  const leer = (ruta) => leerNodo(parseDocument(readFileSync(ruta, 'utf-8')).contents);
  const base = leer(join(raiz, 'default.yaml'));
  const entornos = readdirSync(join(raiz, 'entornos')).filter((f) => f.endsWith('.yaml')).sort();
  const perfiles = {};
  for (const archivo of entornos) {
    const c = combinar(base, leer(join(raiz, 'entornos', archivo)));
    const trpoExperto = combinar(c.trpo, c.experto.trpo);
    const frecuencia = seccion('demostraciones', { frecuencia_pares: c.demostraciones.frecuencia_pares });
    perfiles[archivo.replace(/\.yaml$/, '')] = {
      experto: {
        metodo: 'experto',
        duracion: { unidad: 'pasos', total: c.experto.pasos_totales.valor },
        secciones: [seccion('experto', c.experto, EXCLUIDAS.experto), seccion('experto.trpo', trpoExperto)],
      },
      bc: {
        metodo: 'bc',
        duracion: { unidad: 'epocas', total: c.bc.epocas_maximas.valor },
        secciones: [seccion('bc', c.bc), frecuencia],
      },
      gail: {
        metodo: 'gail',
        duracion: { unidad: 'iteraciones', total: c.gail.iteraciones.valor },
        secciones: [seccion('gail', c.gail, EXCLUIDAS.gail),
          seccion('gail.discriminador', c.gail.discriminador), seccion('trpo', c.trpo), frecuencia],
      },
    };
  }
  return perfiles;
}

/** Texto del JSON tal como se guarda. */
export const textoPerfiles = (perfiles) => `${JSON.stringify(perfiles, null, 2)}\n`;

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  writeFileSync(SALIDA, textoPerfiles(generarPerfiles()));
  console.log(`Escrito ${SALIDA}`);
}
