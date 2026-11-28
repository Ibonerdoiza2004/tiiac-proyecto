import { useId, useState } from 'react';

import { ORDENES, SPLITS, formatearCifra, ordenarPartidas } from './textos.js';

/**
 * Partidas de unas demostraciones: pasos, retorno y en qué parte del reparto están. Con
 * elegidas y alCambiar, cada fila tiene una casilla para elegirla.
 *
 * @param {object} props
 * @param {import('../api/biblioteca.js').Partida[]} props.partidas
 * @param {Set<number>} [props.elegidas] id_episodio elegidos (sin él, solo se muestran).
 * @param {(id: number, elegida: boolean) => void} [props.alCambiar]
 * @param {boolean} [props.bloqueada] Desactiva las casillas.
 * @returns {JSX.Element}
 */
export default function TablaPartidas({ partidas, elegidas, alCambiar, bloqueada = false }) {
  const id = useId();
  const conTrain = partidas.some((p) => p.orden_train !== null && p.orden_train !== undefined);
  const [orden, setOrden] = useState('id');
  const ordenes = Object.entries(ORDENES).filter(([clave]) => clave !== 'train' || conTrain);
  const elegir = Boolean(elegidas && alCambiar);

  return (
    <div className="tabla-partidas">
      <label className="campo campo-en-linea" htmlFor={`${id}-orden`}>
        <span>Ordenar</span>
        <select id={`${id}-orden`} value={orden} onChange={(e) => setOrden(e.target.value)}>
          {ordenes.map(([clave, { etiqueta }]) => <option key={clave} value={clave}>{etiqueta}</option>)}
        </select>
      </label>
      <div className="tabla-envoltorio">
        <table>
          <thead>
            <tr>
              {elegir ? <th scope="col"><span className="solo-lectores">Elegir</span></th> : null}
              <th scope="col">Partida</th>
              <th scope="col">Pasos</th>
              <th scope="col">Retorno</th>
              <th scope="col">Reparto</th>
            </tr>
          </thead>
          <tbody>
            {ordenarPartidas(partidas, orden).map((p) => (
              <tr key={p.id_episodio}>
                {elegir ? (
                  <td>
                    <input type="checkbox" aria-label={`Partida ${p.id_episodio}`} disabled={bloqueada}
                      checked={elegidas.has(p.id_episodio)} onChange={(e) => alCambiar(p.id_episodio, e.target.checked)} />
                  </td>
                ) : null}
                <th scope="row">{p.id_episodio}</th>
                <td>{formatearCifra(p.pasos)}</td>
                <td>{formatearCifra(p.retorno)}</td>
                <td>
                  {SPLITS[p.split] ?? '—'}
                  {p.split === 'train' && typeof p.orden_train === 'number' ? ` · ${p.orden_train + 1}.ª` : ''}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
