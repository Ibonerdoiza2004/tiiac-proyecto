/**
 * Textos y formatos de la Biblioteca: cómo se llama y cómo se muestra cada cosa que manda
 * la API (tipos, orígenes, fuentes y los detalles de cada elemento).
 */

import { LARGO_NOMBRE } from '../api/biblioteca.js';
import { formatearNumero } from '../entrenamiento/curvas.js';

export const TIPOS = Object.freeze({ experto: 'Experto (TRPO)', gail: 'GAIL', bc: 'BC',
  demostraciones: 'Demostraciones', conjunto: 'Conjunto', subconjunto: 'Subconjunto' });
export const ORIGENES = Object.freeze({ congelado: 'Congelado (E1)', entrega: 'Entrega de la E1',
  dashboard: 'Del dashboard' });
export const FUENTES = Object.freeze({ experto: 'Experto', humano: 'Jugadas a mano', combinado: 'Combinadas' });
export const SPLITS = Object.freeze({ train: 'Train', validacion: 'Validación' });

/** Cómo se llama cada clave de los detalles; las que no están salen con su clave. */
export const ETIQUETAS_DETALLES = Object.freeze({
  conjunto: 'Sale del conjunto',
  partidas: 'Partidas',
  pares: 'Pares estado-acción',
  splits: 'Reparto en train y validación',
  cantidad: 'Partidas de entrenamiento',
  subconjunto: 'Partidas de entrenamiento',
  repeticion: 'Repetición',
  semilla: 'Semilla',
  pasos: 'Pasos de entrenamiento',
  iteraciones: 'Iteraciones',
  epocas: 'Épocas',
  mejor_retorno: 'Mejor retorno evaluado',
  objetivo: 'Retorno objetivo',
  objetivo_alcanzado: 'Objetivo alcanzado',
  incompleto: 'Finalizado antes de tiempo',
  segundos: 'Duración del entrenamiento',
  huella_dataset: 'Huella de los datos',
});

// Salen arriba: el tipo (conjunto o subconjunto) y la fuente («Quién jugó»).
const OMITIDOS = new Set(['tipo', 'fuente']);
const CARACTERES_HUELLA = 12;

/** Una duración en segundos, en s, min o h. */
export function formatearDuracion(segundos) {
  if (typeof segundos !== 'number' || !Number.isFinite(segundos) || segundos < 0) {
    return '—';
  }
  if (segundos < 120) {
    return `${Math.round(segundos)} s`;
  }
  if (segundos < 7200) {
    return `${Math.round(segundos / 60)} min`;
  }
  return `${formatearNumero(segundos / 3600)} h`;
}

/** Una fecha ISO como día, mes, año y hora; si no se entiende, tal cual. */
export function formatearFecha(iso) {
  if (typeof iso !== 'string' || !iso) {
    return '—';
  }
  const fecha = new Date(iso);
  return Number.isNaN(fecha.getTime()) ? iso
    : fecha.toLocaleString('es-ES', { day: '2-digit', month: '2-digit', year: 'numeric', hour: '2-digit',
      minute: '2-digit' });
}

/** Un número: los enteros con separador de miles, los demás con 3 cifras significativas. */
export function formatearCifra(valor) {
  if (typeof valor !== 'number' || !Number.isFinite(valor)) {
    return '—';
  }
  return Number.isInteger(valor) ? valor.toLocaleString('es-ES') : formatearNumero(valor);
}

/** El valor de un detalle como texto, según su clave. */
export function valorDetalle(clave, valor) {
  if (valor === null || valor === undefined) {
    return '—';
  }
  if (typeof valor === 'boolean') {
    return valor ? 'Sí' : 'No';
  }
  if (clave === 'segundos') {
    return formatearDuracion(valor);
  }
  if (clave === 'huella_dataset' && typeof valor === 'string' && valor.length > CARACTERES_HUELLA) {
    return `${valor.slice(0, CARACTERES_HUELLA)}…`;
  }
  if (typeof valor === 'number') {
    return formatearCifra(valor);
  }
  return typeof valor === 'string' ? valor : JSON.stringify(valor);
}

/**
 * Filas de los detalles de un elemento, en el orden en que los manda la API.
 *
 * @param {Record<string, unknown> | null | undefined} detalles
 * @returns {{clave: string, etiqueta: string, texto: string, completo: string | null}[]}
 *   completo es el valor entero cuando el texto lo acorta (la huella).
 */
export function filasDetalles(detalles) {
  return Object.entries(detalles ?? {}).filter(([clave]) => !OMITIDOS.has(clave)).map(([clave, valor]) => {
    const texto = valorDetalle(clave, valor);
    return { clave, etiqueta: ETIQUETAS_DETALLES[clave] ?? clave, texto,
      completo: typeof valor === 'string' && texto !== valor ? valor : null };
  });
}

/** Tipo de un elemento: en unas demostraciones, si son conjunto o subconjunto. */
export function textoTipo(elemento) {
  const tipo = elemento.tipo === 'demostraciones' ? elemento.detalles?.tipo ?? elemento.tipo : elemento.tipo;
  return TIPOS[tipo] ?? String(tipo);
}

/**
 * Error del nombre de un subconjunto (las reglas de NombreUsuario), o null si vale.
 *
 * @param {string} nombre
 * @returns {string | null}
 */
export function errorNombre(nombre) {
  if (!/\S/.test(nombre)) {
    return 'Escribe un nombre.';
  }
  return nombre.trim().length > LARGO_NOMBRE ? `Como mucho ${LARGO_NOMBRE} caracteres.` : null;
}

/**
 * Partidas y pares de una elección.
 *
 * @param {import('../api/biblioteca.js').Partida[]} partidas
 * @param {Set<number>} elegidas id_episodio elegidos.
 * @returns {{partidas: number, pares: number}}
 */
export function resumenEleccion(partidas, elegidas) {
  const dentro = partidas.filter((p) => elegidas.has(p.id_episodio));
  return { partidas: dentro.length, pares: dentro.reduce((suma, p) => suma + p.pasos, 0) };
}

/** Órdenes de la tabla de partidas. */
export const ORDENES = Object.freeze({
  id: { etiqueta: 'Por número', comparar: (a, b) => a.id_episodio - b.id_episodio },
  retorno: { etiqueta: 'Por retorno, de mayor a menor', comparar: (a, b) => b.retorno - a.retorno
    || a.id_episodio - b.id_episodio },
  train: { etiqueta: 'Orden de train (las N primeras entrenan con N)', comparar: (a, b) =>
    (a.orden_train ?? Infinity) - (b.orden_train ?? Infinity) || a.id_episodio - b.id_episodio },
});

/** Partidas ordenadas con una de ORDENES, sin modificar la lista. */
export function ordenarPartidas(partidas, orden) {
  return [...partidas].sort((ORDENES[orden] ?? ORDENES.id).comparar);
}
