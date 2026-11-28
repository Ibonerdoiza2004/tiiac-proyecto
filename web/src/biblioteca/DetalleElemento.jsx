import { useState } from 'react';

import useCarga from '../componentes/useCarga.js';
import CrearSubconjunto from './CrearSubconjunto.jsx';
import TablaPartidas from './TablaPartidas.jsx';
import { FUENTES, ORIGENES, filasDetalles, formatearFecha, textoTipo } from './textos.js';

/**
 * Detalle de un elemento de la biblioteca: sus datos, los detalles de su entrenamiento o
 * de sus partidas y, en unas demostraciones válidas, sus partidas; en un conjunto válido,
 * además, crear un subconjunto.
 *
 * @param {object} props
 * @param {import('../api/biblioteca.js').ApiBiblioteca} props.api
 * @param {string} props.id Elemento elegido.
 * @param {(id: string) => void} props.alElegir Abre otro elemento (el conjunto de un subconjunto).
 * @param {(id: string) => string | undefined} [props.nombreDe] Nombre de otro elemento por su id.
 * @param {(elemento: import('../api/biblioteca.js').ElementoBiblioteca) => void} props.alCrear
 *   Tras crear un subconjunto.
 * @returns {JSX.Element}
 */
export default function DetalleElemento({ api, id, alElegir, alCrear, nombreDe = () => undefined }) {
  const elemento = useCarga(() => api.elemento(id), id);
  // El panel lo monta de nuevo con cada id (key), así que no llegan datos de otro elemento.
  const e = elemento.datos;
  const demostraciones = e?.tipo === 'demostraciones' && e.valido;
  const partidas = useCarga(demostraciones ? () => api.partidas(id) : null, `${id}:${demostraciones}`);
  const [creando, setCreando] = useState(false);
  const esConjunto = demostraciones && e.detalles?.tipo === 'conjunto';

  if (elemento.error) {
    return (
      <div className="detalle">
        <p className="error" role="alert">No se pudo cargar el elemento ({elemento.error}).</p>
        <button type="button" onClick={elemento.recargar}>Reintentar</button>
      </div>
    );
  }
  if (!e) {
    return <div className="detalle"><p className="nota">Cargando…</p></div>;
  }

  const filas = [
    ['Tipo', textoTipo(e)],
    ['Origen', ORIGENES[e.origen] ?? e.origen ?? '—'],
    ['Creado', formatearFecha(e.creado)],
    ...(e.tipo === 'demostraciones' ? [['Quién jugó', FUENTES[e.fuente] ?? e.fuente ?? '—']] : []),
    ...(e.artefacto ? [['Artefacto de W&B', e.artefacto]] : []),
  ];

  return (
    <div className="detalle">
      <h3>{e.nombre}</h3>
      <p className="apagado">Id <code>{e.id}</code></p>
      {e.valido ? null : <p className="error">No válido: {e.motivo ?? 'sin motivo'}</p>}
      {e.descripcion ? <p>{e.descripcion}</p> : null}
      <dl className="ficha">
        {filas.map(([etiqueta, valor]) => (
          <div key={etiqueta}><dt>{etiqueta}</dt><dd>{valor}</dd></div>
        ))}
        {filasDetalles(e.detalles).map(({ clave, etiqueta, texto, completo }) => (
          <div key={clave}>
            <dt>{etiqueta}</dt>
            <dd title={completo ?? undefined}>
              {clave === 'conjunto' && typeof e.detalles.conjunto === 'string' ? (
                <button type="button" className="enlace" onClick={() => alElegir(e.detalles.conjunto)}>
                  {nombreDe(e.detalles.conjunto) ?? texto}
                </button>
              ) : texto}
            </dd>
          </div>
        ))}
      </dl>

      {demostraciones ? (
        <div className="partidas-detalle">
          {partidas.error ? (
            <p className="error" role="alert">
              No se pudieron cargar las partidas ({partidas.error}).{' '}
              <button type="button" className="enlace" onClick={partidas.recargar}>Reintentar</button>
            </p>
          ) : null}
          {!partidas.datos && !partidas.error ? <p className="nota">Cargando las partidas…</p> : null}
          {partidas.datos && creando ? (
            <CrearSubconjunto api={api} conjunto={e} partidas={partidas.datos} alCancelar={() => setCreando(false)}
              alCrear={(nuevo) => {
                setCreando(false);
                alCrear(nuevo);
              }} />
          ) : null}
          {partidas.datos && !creando ? (
            <>
              <div className="partidas-cabecera">
                <h4>Partidas</h4>
                {esConjunto ? (
                  <button type="button" onClick={() => setCreando(true)}>Crear un subconjunto</button>
                ) : null}
              </div>
              <TablaPartidas partidas={partidas.datos} />
            </>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
