import PanelComparar from '../comparar/PanelComparar.jsx';
import { buscarEntorno } from '../entornos.js';

/**
 * Pestaña Comparar: evaluar varios modelos de la biblioteca con las mismas partidas.
 *
 * @param {object} props
 * @param {string} props.entorno Clave del entorno elegido.
 * @param {import('../api/comparar.js').ApiComparar} props.apiComparar
 * @param {(ocupado: boolean) => void} [props.alCambiarOcupado] Avisa si hay una en marcha.
 * @returns {JSX.Element} Contenido de la pestaña.
 */
export default function Comparar({ entorno, apiComparar, alCambiarOcupado }) {
  const { nombre } = buscarEntorno(entorno);
  return (
    <section aria-label="Comparación de modelos" className="seccion">
      <h2>Una misma prueba. Todos tus modelos.</h2>
      <p className="nota">Mismas partidas y semillas para todos los modelos de {nombre}.</p>
      <PanelComparar entorno={entorno} api={apiComparar} alCambiarOcupado={alCambiarOcupado} />
    </section>
  );
}
