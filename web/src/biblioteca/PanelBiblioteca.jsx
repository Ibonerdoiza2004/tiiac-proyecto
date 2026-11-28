import { useEffect, useId, useState } from 'react';

import { TIPOS_MODELO } from '../api/biblioteca.js';
import useCarga from '../componentes/useCarga.js';
import { buscarEntorno } from '../entornos.js';
import DetalleElemento from './DetalleElemento.jsx';
import { FUENTES, ORIGENES, TIPOS, formatearCifra, formatearFecha, textoTipo } from './textos.js';

const SECCIONES = Object.freeze({ modelos: 'Modelos', demostraciones: 'Demostraciones' });

/** Estado de un elemento en las listas. */
const Estado = ({ elemento }) => (elemento.valido ? 'Válido'
  : <span className="no-valido" title={elemento.motivo ?? undefined}>No válido</span>);

/**
 * Pestaña Biblioteca: modelos y demostraciones del entorno, el detalle del que se elija y
 * crear subconjuntos de un conjunto de demostraciones.
 *
 * @param {object} props
 * @param {string} props.entorno Clave del entorno.
 * @param {import('../api/biblioteca.js').ApiBiblioteca} props.api
 * @returns {JSX.Element}
 */
export default function PanelBiblioteca({ entorno, api }) {
  const id = useId();
  const { nombre } = buscarEntorno(entorno);
  const [seccion, setSeccion] = useState('modelos');
  const [tipo, setTipo] = useState('');
  const [elegido, setElegido] = useState(null);
  const [aviso, setAviso] = useState(null);
  const modelos = useCarga(() => api.modelos(entorno), `modelos:${entorno}`);
  const demostraciones = useCarga(() => api.demostraciones(entorno), `demostraciones:${entorno}`);
  const lista = seccion === 'modelos' ? modelos : demostraciones;

  useEffect(() => {
    setElegido(null);
    setAviso(null);
  }, [entorno]);

  const cambiarSeccion = (nueva) => {
    setSeccion(nueva);
    setElegido(null);
    setAviso(null);
  };

  // El conjunto de un subconjunto es otra demostración: se abre en su sección.
  const abrir = (otro) => {
    setSeccion('demostraciones');
    setElegido(otro);
    setAviso(null);
  };

  const filas = seccion === 'modelos'
    ? (modelos.datos ?? []).filter((m) => (tipo ? m.tipo === tipo : TIPOS_MODELO.includes(m.tipo)))
    : demostraciones.datos ?? [];
  const contar = (carga) => (carga.datos ? ` (${carga.datos.length})` : '');

  return (
    <div className="biblioteca">
      <div className="pestanas" role="group" aria-label="Qué ver">
        {Object.entries(SECCIONES).map(([clave, etiqueta]) => (
          <button key={clave} type="button" aria-pressed={seccion === clave} onClick={() => cambiarSeccion(clave)}>
            {etiqueta}{contar(clave === 'modelos' ? modelos : demostraciones)}
          </button>
        ))}
      </div>

      <div className="rejilla-dos">
        <div className="biblioteca-lista">
          <div className="biblioteca-filtros">
            {seccion === 'modelos' ? (
              <label className="campo campo-en-linea" htmlFor={`${id}-tipo`}>
                <span>Tipo</span>
                <select id={`${id}-tipo`} value={tipo} onChange={(e) => setTipo(e.target.value)}>
                  <option value="">Todos</option>
                  {TIPOS_MODELO.map((t) => <option key={t} value={t}>{TIPOS[t]}</option>)}
                </select>
              </label>
            ) : null}
            <button type="button" className="enlace" onClick={lista.recargar}>Actualizar</button>
          </div>
          {lista.error ? <p className="error" role="alert">No se pudo cargar la lista ({lista.error}).</p> : null}
          {!lista.datos && !lista.error ? <p className="nota">Cargando…</p> : null}
          {lista.datos && filas.length === 0 ? (
            <p className="nota">
              {seccion === 'modelos'
                ? `No hay modelos${tipo ? ` de tipo ${TIPOS[tipo]}` : ''} de ${nombre}. Entrénalos en el Laboratorio.`
                : `No hay demostraciones de ${nombre}. Grábalas en el Laboratorio.`}
            </p>
          ) : null}
          {filas.length ? (
            <div className="tabla-envoltorio">
              <table className="tabla-elegible">
                <caption className="solo-lectores">{SECCIONES[seccion]} de {nombre}</caption>
                <thead>
                  {seccion === 'modelos' ? (
                    <tr><th scope="col">Nombre</th><th scope="col">Tipo</th><th scope="col">Origen</th>
                      <th scope="col">Creado</th><th scope="col">Estado</th></tr>
                  ) : (
                    <tr><th scope="col">Nombre</th><th scope="col">Tipo</th><th scope="col">Partidas</th>
                      <th scope="col">Pares</th><th scope="col">Quién jugó</th><th scope="col">Estado</th></tr>
                  )}
                </thead>
                <tbody>
                  {filas.map((e) => (
                    <tr key={e.id} className={e.id === elegido ? 'elegida' : undefined}>
                      <th scope="row">
                        <button type="button" className="enlace" aria-pressed={e.id === elegido}
                          onClick={() => {
                            setElegido(e.id);
                            setAviso(null);
                          }}>{e.nombre}</button>
                      </th>
                      <td>{seccion === 'modelos' ? textoTipo(e) : TIPOS[e.tipo] ?? e.tipo ?? '—'}</td>
                      {seccion === 'modelos' ? (
                        <>
                          <td>{ORIGENES[e.origen] ?? e.origen ?? '—'}</td>
                          <td>{formatearFecha(e.creado)}</td>
                        </>
                      ) : (
                        <>
                          <td>{formatearCifra(e.partidas)}</td>
                          <td>{formatearCifra(e.pares)}</td>
                          <td>{FUENTES[e.fuente] ?? e.fuente ?? '—'}</td>
                        </>
                      )}
                      <td><Estado elemento={e} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : null}
        </div>

        <div className="biblioteca-detalle">
          {aviso ? <p className="aviso-exito" role="status">{aviso}</p> : null}
          {elegido ? (
            <DetalleElemento key={elegido} api={api} id={elegido} alElegir={abrir}
              nombreDe={(otro) => demostraciones.datos?.find((d) => d.id === otro)?.nombre}
              alCrear={(nuevo) => {
                demostraciones.recargar();
                setElegido(nuevo.id);
                setAviso(`Subconjunto «${nuevo.nombre}» guardado en la biblioteca.`);
              }} />
          ) : <p className="nota">Elige un elemento de la lista para ver su detalle.</p>}
        </div>
      </div>
    </div>
  );
}
