import { buscarEntorno } from '../entornos.js';

/**
 * Pestaña Comparar: evaluar varios modelos con las mismas partidas. De momento solo el
 * esqueleto.
 *
 * @param {object} props
 * @param {string} props.entorno Clave del entorno elegido.
 * @returns {JSX.Element} Contenido de la pestaña.
 */
export default function Comparar({ entorno }) {
  const { nombre } = buscarEntorno(entorno);
  return (
    <section aria-label="Comparación de modelos" className="seccion">
      <h2>Una misma prueba. Todos tus modelos.</h2>
      <p className="nota">Mismas partidas y semillas para todos los modelos de {nombre}.</p>
      <p className="pendiente">Todavía no disponible.</p>
    </section>
  );
}
