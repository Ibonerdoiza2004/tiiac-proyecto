import ExplorarGenerar from '../laboratorio/ExplorarGenerar.jsx';

/**
 * Pestaña Laboratorio: a la izquierda, jugar, ver al experto y grabar partidas; a la
 * derecha, entrenar (todavía sin hacer).
 *
 * @param {object} props
 * @param {string} props.entorno Clave del entorno elegido.
 * @param {import('../api/laboratorio.js').ApiLaboratorio} props.api Backend del Laboratorio.
 * @param {(ocupado: boolean) => void} [props.alCambiarOcupado] Avisa si hay una sesión abierta.
 * @returns {JSX.Element} Contenido de la pestaña.
 */
export default function Laboratorio({ entorno, api, alCambiarOcupado }) {
  return (
    <section aria-label="Laboratorio" className="rejilla-dos">
      <ExplorarGenerar entorno={entorno} api={api} alCambiarOcupado={alCambiarOcupado} />
      <article className="panel" aria-labelledby="panel-entrenar">
        <h2 id="panel-entrenar">Entrenar e imitar</h2>
        <p className="nota">Entrena GAIL o BC con las demostraciones que elijas.</p>
        <p className="pendiente">Todavía no disponible.</p>
      </article>
    </section>
  );
}
