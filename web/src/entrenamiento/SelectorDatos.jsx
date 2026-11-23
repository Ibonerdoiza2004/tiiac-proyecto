import { useId } from 'react';

import { contar } from '../laboratorio/revision.js';

const FUENTES = { experto: 'experto', humano: 'jugadas a mano', combinado: 'combinadas' };

/**
 * Lista de conjuntos y subconjuntos de demostraciones del entorno para elegir con qué
 * datos entrenar BC o GAIL.
 *
 * @param {object} props
 * @param {import('../api/entrenamiento.js').Demostracion[] | null} props.lista null mientras carga.
 * @param {string | null} props.error Error al cargar la lista.
 * @param {Set<string>} props.elegidos IDs elegidos.
 * @param {(id: string, elegido: boolean) => void} props.alCambiar
 * @param {() => void} props.alActualizar Vuelve a pedir la lista.
 * @param {string} [props.aviso] Error de validación, como no haber elegido nada.
 * @returns {JSX.Element}
 */
export default function SelectorDatos({ lista, error, elegidos, alCambiar, alActualizar, aviso }) {
  const idAviso = `${useId()}-aviso`;
  return (
    <fieldset className="datos" aria-describedby={aviso ? idAviso : undefined}>
      <legend>Demostraciones</legend>
      {error ? (
        <p className="error" role="alert">No se pudo cargar la lista ({error}).</p>
      ) : null}
      {lista === null && !error ? <p className="nota">Cargando…</p> : null}
      {lista?.length === 0 ? (
        <p className="nota">No hay demostraciones de este entorno. Grábalas en «Explorar y generar».</p>
      ) : null}
      {lista?.length ? (
        <ul className="datos-lista">
          {lista.map((d) => (
            <li key={d.id}>
              <label>
                <input type="checkbox" checked={elegidos.has(d.id)}
                  onChange={(e) => alCambiar(d.id, e.target.checked)} />
                <span>{d.nombre}</span>
                <span className="apagado">
                  {d.tipo} · {contar(d.partidas, 'partida', 'partidas')}
                  {d.pares === null || d.pares === undefined ? '' : ` · ${contar(d.pares, 'par', 'pares')}`}
                  {' · '}{FUENTES[d.fuente] ?? d.fuente}
                </span>
              </label>
            </li>
          ))}
        </ul>
      ) : null}
      <div className="datos-pie">
        <span className="apagado">
          {contar(elegidos.size, 'elegido', 'elegidos')}. Los pares repetidos se cuentan una vez.
        </span>
        <button type="button" onClick={alActualizar}>Actualizar lista</button>
      </div>
      {aviso ? <p className="error" id={idAviso}>{aviso}</p> : null}
    </fieldset>
  );
}
