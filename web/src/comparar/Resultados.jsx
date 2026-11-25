import { formatearNumero } from '../entrenamiento/curvas.js';
import { clasificar } from './estadisticas.js';

const DURACIONES = [['pasos', 'pasos'], ['iteraciones', 'iteraciones'], ['epocas', 'épocas']];

/** Texto del historial de entrenamiento de un modelo, a partir de sus detalles. */
export function textoHistorial(modelo) {
  const d = modelo?.detalles ?? {};
  const partes = DURACIONES.filter(([clave]) => typeof d[clave] === 'number')
    .map(([clave, nombre]) => `${formatearNumero(d[clave])} ${nombre}`);
  if (typeof d.segundos === 'number') {
    partes.push(d.segundos < 120 ? `${Math.round(d.segundos)} s` : `${Math.round(d.segundos / 60)} min`);
  }
  if (d.incompleto) {
    partes.push('incompleto');
  }
  return partes.join(' · ') || '—';
}

/** Con qué datos se entrenó un modelo (cantidad de partidas de las demostraciones). */
export function textoDatos(modelo) {
  const d = modelo?.detalles ?? {};
  const partidas = d.cantidad ?? d.subconjunto;
  if (typeof partidas === 'number') {
    return `${partidas} ${partidas === 1 ? 'partida' : 'partidas'}${typeof d.repeticion === 'number' ? ` · repetición ${d.repeticion}` : ''}`;
  }
  return modelo?.tipo === 'experto' ? 'refuerzo (TRPO)' : '—';
}

/**
 * Resultados de la comparación: barras de la media, tabla con media ± desviación y la
 * última partida, puntuación de cada partida e historial de entrenamiento de cada modelo.
 *
 * @param {object} props
 * @param {import('./estadisticas.js').ResumenModelo[]} props.resumenes En el orden elegido.
 * @param {(id: string) => string} props.nombre Nombre legible de un modelo.
 * @param {(id: string) => import('../api/comparar.js').ElementoBiblioteca | undefined} props.modelo
 * @returns {JSX.Element}
 */
export default function Resultados({ resumenes, nombre, modelo }) {
  const clasificados = clasificar(resumenes);
  const medias = resumenes.map((r) => r.media).filter((m) => m !== null);
  const minimo = Math.min(0, ...medias);
  const maximo = Math.max(0, ...medias);
  const ancho = (m) => (maximo === minimo ? 0 : ((m - minimo) / (maximo - minimo)) * 100);
  const filas = Math.max(0, ...resumenes.map((r) => r.partidas));

  return (
    <div className="resultados">
      <div className="resultados-bloque">
        <h3>Recompensa media por partida</h3>
        {medias.length ? (
          <div className="barras">
            {clasificados.map((r) => (
              <div key={r.modelo} className="barra-fila">
                <span>{nombre(r.modelo)}</span>
                <span className="barra-pista" aria-hidden="true">
                  <span style={{ width: `${r.media === null ? 0 : ancho(r.media)}%` }} />
                </span>
                <span className="cifra">{r.media === null ? '—' : formatearNumero(r.media)}</span>
              </div>
            ))}
          </div>
        ) : <p className="nota">Los resultados aparecerán con la primera partida.</p>}
        <div className="tabla-envoltorio">
          <table>
            <caption className="solo-lectores">Resultados por modelo, de mejor a peor media</caption>
            <thead><tr><th scope="col">Modelo</th><th scope="col">Partidas</th><th scope="col">Media ± DE</th><th scope="col">Última</th></tr></thead>
            <tbody>
              {clasificados.map((r) => (
                <tr key={r.modelo}>
                  <th scope="row">{nombre(r.modelo)}</th>
                  <td>{r.partidas}</td>
                  <td>{r.media === null ? '—' : `${formatearNumero(r.media)}${r.desviacion === null ? '' : ` ± ${formatearNumero(r.desviacion)}`}`}</td>
                  <td>{r.ultima === null ? '—' : formatearNumero(r.ultima)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="nota">DE: desviación típica entre partidas; con una sola partida aún no se calcula.</p>
        {filas ? (
          <details className="detalle-partidas">
            <summary>Puntuación de cada partida</summary>
            <div className="tabla-envoltorio">
              <table>
                <thead>
                  <tr><th scope="col">Partida</th>{resumenes.map((r) => <th key={r.modelo} scope="col">{nombre(r.modelo)}</th>)}</tr>
                </thead>
                <tbody>
                  {Array.from({ length: filas }, (_, i) => (
                    <tr key={i}>
                      <th scope="row">{i + 1}</th>
                      {resumenes.map((r) => (
                        <td key={r.modelo}>{r.retornos[i] === undefined ? '—' : formatearNumero(r.retornos[i])}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </details>
        ) : null}
      </div>

      <div className="resultados-bloque">
        <h3>Historial de entrenamiento</h3>
        <p className="nota">Datos guardados al entrenar cada modelo; no salen de esta comparación.</p>
        <div className="tabla-envoltorio">
          <table>
            <thead><tr><th scope="col">Modelo</th><th scope="col">Entrenamiento</th><th scope="col">Datos</th></tr></thead>
            <tbody>
              {resumenes.map((r) => (
                <tr key={r.modelo}>
                  <th scope="row">{nombre(r.modelo)}</th>
                  <td>{modelo(r.modelo) ? textoHistorial(modelo(r.modelo)) : 'No aplica'}</td>
                  <td>{modelo(r.modelo) ? textoDatos(modelo(r.modelo)) : 'No aplica'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
