import PanelBiblioteca from '../biblioteca/PanelBiblioteca.jsx';
import { buscarEntorno } from '../entornos.js';

/**
 * Pestaña Biblioteca: modelos y demostraciones guardados, con sus metadatos, y
 * subconjuntos de demostraciones.
 *
 * @param {object} props
 * @param {string} props.entorno Clave del entorno elegido.
 * @param {import('../api/biblioteca.js').ApiBiblioteca} props.apiBiblioteca
 * @returns {JSX.Element} Contenido de la pestaña.
 */
export default function Biblioteca({ entorno, apiBiblioteca }) {
  const { nombre } = buscarEntorno(entorno);
  return (
    <section aria-label="Biblioteca" className="seccion">
      <h2>Modelos y demostraciones</h2>
      <p className="nota">Lo que hay guardado para {nombre}, con sus metadatos.</p>
      <PanelBiblioteca entorno={entorno} api={apiBiblioteca} />
    </section>
  );
}
