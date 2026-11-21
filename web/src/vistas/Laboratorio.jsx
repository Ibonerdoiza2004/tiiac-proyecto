import { buscarEntorno } from '../entornos.js';

/**
 * Pestaña Laboratorio: jugar, ver al experto y entrenar. De momento solo el esqueleto
 * de sus dos paneles.
 *
 * @param {object} props
 * @param {string} props.entorno Clave del entorno elegido.
 * @returns {JSX.Element} Contenido de la pestaña.
 */
export default function Laboratorio({ entorno }) {
  const { nombre, jugarAMano } = buscarEntorno(entorno);
  return (
    <section aria-label="Laboratorio" className="rejilla-dos">
      <article className="panel" aria-labelledby="panel-explorar">
        <h2 id="panel-explorar">Explorar y generar</h2>
        <p className="nota">
          {jugarAMano
            ? `Juega a ${nombre}, mira al experto y graba demostraciones.`
            : `En ${nombre}, las demostraciones se graban con el experto.`}
        </p>
        <p className="pendiente">Todavía no disponible.</p>
      </article>
      <article className="panel" aria-labelledby="panel-entrenar">
        <h2 id="panel-entrenar">Entrenar e imitar</h2>
        <p className="nota">Entrena GAIL o BC con las demostraciones que elijas.</p>
        <p className="pendiente">Todavía no disponible.</p>
      </article>
    </section>
  );
}
