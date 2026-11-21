import { buscarEntorno } from '../entornos.js';

/**
 * Pestaña Biblioteca: modelos y demostraciones guardados. De momento solo el esqueleto.
 *
 * @param {object} props
 * @param {string} props.entorno Clave del entorno elegido.
 * @returns {JSX.Element} Contenido de la pestaña.
 */
export default function Biblioteca({ entorno }) {
  const { nombre } = buscarEntorno(entorno);
  return (
    <section aria-label="Biblioteca" className="seccion">
      <h2>Modelos y demostraciones</h2>
      <p className="nota">Lo que hay guardado para {nombre}, con sus metadatos.</p>
      <p className="pendiente">Todavía no disponible.</p>
    </section>
  );
}
