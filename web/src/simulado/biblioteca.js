/**
 * Biblioteca simulada: la misma ApiBiblioteca que src/api/biblioteca.js, en el navegador,
 * sobre el almacén que comparten las APIs simuladas (./almacen.js).
 *
 * Modelos: por entorno, el experto congelado y unos BC y GAIL de entrega inventados, más
 * los que se entrenen en el Laboratorio simulado. Demostraciones: el conjunto congelado de
 * cada entorno (con el reparto de configs/), lo que se guarda al jugar y los subconjuntos
 * que se crean aquí, que aparecen también para entrenar. Responde con los mismos errores
 * que FastAPI (404 y 422).
 */

import { LARGO_NOMBRE } from '../api/biblioteca.js';
import { ErrorApi } from '../api/http.js';
import { ENTORNOS, buscarEntorno } from '../entornos.js';
import { crearAlmacen } from './almacen.js';
import PERFILES from './perfiles.json';

const FECHA_CONGELADO = '2026-11-01T00:00:00Z';
const copia = (valor) => JSON.parse(JSON.stringify(valor));

function rechazar(estado, mensaje) {
  return Promise.reject(new ErrorApi(mensaje, { estado, detalle: { detail: mensaje } }));
}

const campoExperto = (entorno, clave) => PERFILES[entorno].experto.secciones
  .find((s) => s.clave === 'experto').campos.find((c) => c.clave === clave).valor;

/** Duración del perfil de un método, con la clave que usan los detalles de la biblioteca. */
const duracion = (entorno, metodo) => {
  const { unidad, total } = PERFILES[entorno][metodo].duracion;
  return { [unidad]: total };
};

/** Modelos de partida de la biblioteca simulada de un entorno, con la forma de la API. */
export function modelosIniciales(entorno) {
  const { nombre } = buscarEntorno(entorno);
  const base = { entorno, creado: FECHA_CONGELADO, descripcion: null, fuente: null, artefacto: null,
    valido: true, motivo: null };
  return [
    { ...base, id: `experto-${entorno}`, tipo: 'experto', nombre: `Experto congelado · ${nombre}`,
      origen: 'congelado', detalles: { ...duracion(entorno, 'experto'),
        objetivo: campoExperto(entorno, 'retorno_objetivo') } },
    ...[1, 10].flatMap((cantidad) => [
      { ...base, id: `gail-${entorno}-${cantidad}`, tipo: 'gail', nombre: `GAIL · ${cantidad} partidas`,
        origen: 'entrega', detalles: { cantidad, repeticion: 0, ...duracion(entorno, 'gail') } },
      { ...base, id: `bc-${entorno}-${cantidad}`, tipo: 'bc', nombre: `BC · ${cantidad} partidas`,
        origen: 'entrega', detalles: { subconjunto: cantidad, repeticion: 0, ...duracion(entorno, 'bc') } },
    ]),
  ];
}

/**
 * Modelos de la biblioteca simulada de un entorno: los iniciales y los entrenados en el
 * Laboratorio simulado, con la forma de ElementoBibliotecaApi.
 *
 * @param {import('./almacen.js').Almacen} almacen
 * @param {string} entorno
 * @returns {import('../api/biblioteca.js').ElementoBiblioteca[]}
 */
export function modelosSimulados(almacen, entorno) {
  return [
    ...modelosIniciales(entorno),
    ...almacen.modelos.filter((m) => m.entorno === entorno).map((m) => ({
      id: m.id, entorno, tipo: m.metodo, nombre: m.nombre, origen: 'dashboard', creado: m.creado ?? null,
      descripcion: null, fuente: null, artefacto: null, valido: true, motivo: null,
      detalles: { [m.resumen.unidad]: m.resumen.paso, incompleto: m.incompleto },
    })),
  ];
}

/** Una demostración del almacén como ElementoBibliotecaApi. */
function comoElemento(d) {
  return {
    id: d.id, entorno: d.entorno, tipo: 'demostraciones', nombre: d.nombre, origen: d.origen ?? 'dashboard',
    creado: d.creado ?? null, descripcion: null, fuente: d.fuente, artefacto: null, valido: true, motivo: null,
    detalles: { tipo: d.tipo, partidas: d.partidas, pares: d.pares, fuente: d.fuente,
      splits: Boolean(d.detallePartidas?.some((p) => p.split)),
      ...(d.tipo === 'subconjunto' ? { conjunto: d.conjunto } : {}) },
  };
}

/** Una demostración del almacén como DemostracionApi. */
function comoDemostracion(d) {
  return { id: d.id, entorno: d.entorno, nombre: d.nombre, tipo: d.tipo, partidas: d.partidas, pares: d.pares,
    fuente: d.fuente, origen: d.origen ?? 'dashboard', valido: true, motivo: null };
}

const entornoValido = (entorno) => ENTORNOS.some((e) => e.id === entorno);

/**
 * Crea la API de la Biblioteca simulada.
 *
 * @param {object} [opciones]
 * @param {import('./almacen.js').Almacen} [opciones.almacen] El almacén compartido con el
 *   resto de APIs simuladas.
 * @returns {import('../api/biblioteca.js').ApiBiblioteca}
 */
export function crearApiBibliotecaSimulada({ almacen = crearAlmacen() } = {}) {
  const demostracion = (id) => almacen.demostraciones.find((d) => d.id === id);
  const conjuntoDe = (d) => (d.tipo === 'subconjunto' ? demostracion(d.conjunto) : d);

  return {
    modelos(entorno) {
      return entornoValido(entorno) ? Promise.resolve(copia(modelosSimulados(almacen, entorno)))
        : rechazar(422, `Entorno desconocido: ${entorno}`);
    },

    demostraciones(entorno) {
      return entornoValido(entorno)
        ? Promise.resolve(copia(almacen.demostraciones.filter((d) => d.entorno === entorno).map(comoDemostracion)))
        : rechazar(422, `Entorno desconocido: ${entorno}`);
    },

    elemento(id) {
      const d = demostracion(id);
      if (d) {
        return Promise.resolve(copia(comoElemento(d)));
      }
      const modelo = ENTORNOS.flatMap((e) => modelosSimulados(almacen, e.id)).find((m) => m.id === id);
      return modelo ? Promise.resolve(copia(modelo)) : rechazar(404, `No existe el elemento ${id}`);
    },

    partidas(id) {
      const d = demostracion(id);
      if (!d) {
        return ENTORNOS.some((e) => modelosSimulados(almacen, e.id).some((m) => m.id === id))
          ? rechazar(422, `${id} no son demostraciones`) : rechazar(404, `No existe el elemento ${id}`);
      }
      if (d.tipo === 'conjunto') {
        return Promise.resolve(copia(d.detallePartidas));
      }
      // Las de su conjunto; en un subconjunto no hay splits (se reparten al entrenar).
      const elegidas = new Set(d.ids);
      return Promise.resolve(copia(conjuntoDe(d).detallePartidas.filter((p) => elegidas.has(p.id_episodio))
        .map((p) => ({ ...p, split: null, orden_train: null }))));
    },

    crearSubconjunto({ conjunto, partidas, nombre } = {}) {
      const origen = demostracion(conjunto);
      if (!origen) {
        return rechazar(422, `No existe el conjunto ${conjunto}`);
      }
      if (origen.tipo !== 'conjunto') {
        return rechazar(422, `${conjunto} no es un conjunto de demostraciones`);
      }
      if (typeof nombre !== 'string' || nombre.length < 1 || nombre.length > LARGO_NOMBRE || !/\S/.test(nombre)) {
        return rechazar(422, `El nombre debe tener entre 1 y ${LARGO_NOMBRE} caracteres y alguno visible`);
      }
      if (!Array.isArray(partidas) || partidas.length === 0) {
        return rechazar(422, 'Un subconjunto toma al menos una partida');
      }
      if (new Set(partidas).size !== partidas.length) {
        return rechazar(422, 'Hay partidas repetidas en el subconjunto');
      }
      const pasos = new Map(origen.detallePartidas.map((p) => [p.id_episodio, p.pasos]));
      const ajenas = partidas.filter((i) => !Number.isInteger(i) || !pasos.has(i));
      if (ajenas.length) {
        return rechazar(422, `Partidas que no son del conjunto: ${ajenas.join(', ')}`);
      }
      const nuevo = {
        id: almacen.nuevoId('subconjunto'), entorno: origen.entorno, nombre, tipo: 'subconjunto',
        partidas: partidas.length, pares: partidas.reduce((suma, i) => suma + pasos.get(i), 0),
        fuente: origen.fuente, origen: 'dashboard', creado: new Date().toISOString(),
        conjunto, ids: [...partidas].sort((a, b) => a - b),
      };
      almacen.demostraciones.push(nuevo);
      return Promise.resolve(copia(comoElemento(nuevo)));
    },
  };
}
