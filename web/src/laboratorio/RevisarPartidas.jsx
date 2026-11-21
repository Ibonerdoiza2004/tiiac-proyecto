import { useMemo, useState } from 'react';

import { contar, eleccionInicial, validarRevision } from './revision.js';

const decimales = (valor) => valor.toLocaleString('es-ES', { maximumFractionDigits: 1 });

/**
 * Pantalla para revisar las partidas de una sesión y elegir qué partidas y qué tramo de
 * pares de cada una se guardan en la biblioteca.
 *
 * @param {object} props
 * @param {import('../api/laboratorio.js').Partida[]} props.partidas Partidas completas.
 * @param {(nombre: string, seleccion: {indice: number, tramo: [number, number]}[]) => void}
 *   props.alGuardar
 * @param {() => void} props.alDescartar
 * @param {boolean} [props.guardando] Mientras se guarda, los botones no repiten la orden.
 * @returns {JSX.Element}
 */
export default function RevisarPartidas({ partidas, alGuardar, alDescartar, guardando = false }) {
  const [elecciones, setElecciones] = useState(() => eleccionInicial(partidas));
  const [nombre, setNombre] = useState('');
  const [intentado, setIntentado] = useState(false);
  const revision = useMemo(() => validarRevision(partidas, elecciones, nombre),
    [partidas, elecciones, nombre]);
  const mostrar = (error) => (intentado ? error : undefined);

  const cambiar = (indice, campo, valor) => setElecciones((anteriores) => ({
    ...anteriores, [indice]: { ...anteriores[indice], [campo]: valor } }));

  const enviar = (evento) => {
    evento.preventDefault();
    setIntentado(true);
    if (revision.valida && !guardando) {
      alGuardar(nombre, revision.seleccion);
    }
  };

  return (
    <form className="revision" onSubmit={enviar} aria-label="Revisar partidas" noValidate>
      <h3>Revisa las partidas</h3>
      <p className="nota">Elige qué partidas guardar y, de cada una, qué pares (del primero al último).</p>
      <ul className="revision-lista">
        {partidas.map((partida) => {
          const eleccion = elecciones[partida.indice];
          const error = mostrar(revision.errores.partidas[partida.indice]);
          const idError = `revision-error-${partida.indice}`;
          return (
            <li key={partida.indice}>
              <label className="revision-partida">
                <input type="checkbox" checked={eleccion.incluir}
                  onChange={(e) => cambiar(partida.indice, 'incluir', e.target.checked)} />
                <span>Partida {partida.indice + 1}</span>
                <span className="apagado">{contar(partida.pasos, 'par', 'pares')} · puntuación {decimales(partida.retorno)} · semilla {partida.semilla}</span>
              </label>
              <div className="revision-tramo">
                <label>Desde
                  <input type="number" inputMode="numeric" min="1" max={partida.pasos} value={eleccion.desde}
                    disabled={!eleccion.incluir} aria-invalid={Boolean(error)} aria-describedby={error ? idError : undefined}
                    onChange={(e) => cambiar(partida.indice, 'desde', e.target.value)} />
                </label>
                <label>Hasta
                  <input type="number" inputMode="numeric" min="1" max={partida.pasos} value={eleccion.hasta}
                    disabled={!eleccion.incluir} aria-invalid={Boolean(error)} aria-describedby={error ? idError : undefined}
                    onChange={(e) => cambiar(partida.indice, 'hasta', e.target.value)} />
                </label>
              </div>
              {error ? <p className="error" id={idError}>{error}</p> : null}
            </li>
          );
        })}
      </ul>
      <label className="campo">Nombre del conjunto
        <input type="text" value={nombre} placeholder="cartpole_humano_1" onChange={(e) => setNombre(e.target.value)}
          aria-invalid={Boolean(mostrar(revision.errores.nombre))} />
      </label>
      {mostrar(revision.errores.nombre) ? <p className="error">{revision.errores.nombre}</p> : null}
      {mostrar(revision.errores.general) ? <p className="error" role="alert">{revision.errores.general}</p> : null}
      <p className="revision-total" aria-live="polite">
        Se guardarán {contar(revision.totales.partidas, 'partida', 'partidas')} y {contar(revision.totales.pares, 'par', 'pares')}.
      </p>
      <div className="controles">
        <button type="submit" className="primario" disabled={guardando}>
          {guardando ? 'Guardando…' : 'Guardar en la biblioteca'}
        </button>
        <button type="button" onClick={alDescartar} disabled={guardando}>Descartar sesión</button>
      </div>
    </form>
  );
}
