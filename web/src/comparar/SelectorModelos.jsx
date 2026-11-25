import { TIPOS_MODELO } from '../api/comparar.js';

const TITULOS = { experto: 'Expertos', gail: 'GAIL', bc: 'BC' };
const ORIGENES = { congelado: 'congelado', entrega: 'entrega E1', dashboard: 'del dashboard' };

/**
 * Modelos de la biblioteca del entorno, agrupados por tipo, más el agente aleatorio como
 * referencia. Los que no son válidos se muestran desactivados con el motivo.
 *
 * @param {object} props
 * @param {import('../api/comparar.js').ElementoBiblioteca[] | null} props.modelos null mientras carga.
 * @param {string | null} props.error Error al cargar la lista.
 * @param {Set<string>} props.elegidos Ids elegidos.
 * @param {boolean} props.aleatorio Si se añade el agente aleatorio.
 * @param {(id: string, elegido: boolean) => void} props.alCambiar
 * @param {(elegido: boolean) => void} props.alCambiarAleatorio
 * @param {() => void} props.alActualizar
 * @param {boolean} [props.bloqueado] Con una comparación en marcha no se cambia.
 * @returns {JSX.Element}
 */
export default function SelectorModelos({ modelos, error, elegidos, aleatorio, alCambiar,
  alCambiarAleatorio, alActualizar, bloqueado = false }) {
  return (
    <fieldset className="modelos" disabled={bloqueado}>
      <legend>Modelos</legend>
      {error ? <p className="error" role="alert">No se pudo cargar la biblioteca ({error}).</p> : null}
      {modelos === null && !error ? <p className="nota">Cargando la biblioteca…</p> : null}
      {modelos?.length === 0 ? (
        <p className="nota">No hay modelos de este entorno en la biblioteca. Entrénalos en el Laboratorio.</p>
      ) : null}
      <div className="modelos-grupos">
        {TIPOS_MODELO.map((tipo) => {
          const delTipo = (modelos ?? []).filter((m) => m.tipo === tipo);
          return delTipo.length ? (
            <div key={tipo} className="modelos-grupo">
              <h3>{TITULOS[tipo]}</h3>
              {delTipo.map((m) => (
                <label key={m.id} className="eleccion" title={m.valido ? undefined : m.motivo ?? undefined}>
                  <input type="checkbox" checked={elegidos.has(m.id)} disabled={!m.valido}
                    onChange={(e) => alCambiar(m.id, e.target.checked)} />
                  <span>
                    <strong>{m.nombre}</strong>
                    <span className="apagado">
                      {m.valido ? (ORIGENES[m.origen] ?? m.origen ?? '') : `no válido: ${m.motivo ?? 'sin motivo'}`}
                    </span>
                  </span>
                </label>
              ))}
            </div>
          ) : null;
        })}
        <div className="modelos-grupo">
          <h3>Referencia</h3>
          <label className="eleccion">
            <input type="checkbox" checked={aleatorio} onChange={(e) => alCambiarAleatorio(e.target.checked)} />
            <span><strong>Aleatorio</strong><span className="apagado">acciones al azar</span></span>
          </label>
        </div>
      </div>
      <button type="button" className="enlace" onClick={alActualizar}>Actualizar la biblioteca</button>
    </fieldset>
  );
}
