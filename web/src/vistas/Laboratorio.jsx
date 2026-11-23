import { useCallback, useEffect, useState } from 'react';

import { METODOS } from '../api/entrenamiento.js';
import PanelEntrenamiento from '../entrenamiento/PanelEntrenamiento.jsx';
import ExplorarGenerar from '../laboratorio/ExplorarGenerar.jsx';

const IMITAR = [METODOS.GAIL, METODOS.BC];

/**
 * Pestaña Laboratorio: a la izquierda, jugar, entrenar un experto, verlo jugar y grabar
 * partidas; a la derecha, entrenar GAIL o BC con las demostraciones elegidas. Los dos
 * paneles pueden trabajar a la vez.
 *
 * @param {object} props
 * @param {string} props.entorno Clave del entorno elegido.
 * @param {import('../api/laboratorio.js').ApiLaboratorio} props.api Backend del Laboratorio.
 * @param {import('../api/entrenamiento.js').ApiEntrenamiento} props.apiEntrenamiento Backend de
 *   los entrenamientos.
 * @param {(ocupado: boolean) => void} [props.alCambiarOcupado] Avisa si alguno de los dos
 *   paneles tiene algo en marcha.
 * @returns {JSX.Element} Contenido de la pestaña.
 */
export default function Laboratorio({ entorno, api, apiEntrenamiento, alCambiarOcupado }) {
  const [ocupados, setOcupados] = useState({ explorar: false, imitar: false });
  const avisarExplorar = useCallback((ocupado) => setOcupados((o) => ({ ...o, explorar: ocupado })), []);
  const avisarImitar = useCallback((ocupado) => setOcupados((o) => ({ ...o, imitar: ocupado })), []);
  const ocupado = ocupados.explorar || ocupados.imitar;

  useEffect(() => {
    alCambiarOcupado?.(ocupado);
  }, [alCambiarOcupado, ocupado]);

  return (
    <section aria-label="Laboratorio" className="rejilla-dos">
      {/* Con key, cada entorno empieza sin la imagen, los contadores ni los avisos del anterior. */}
      <ExplorarGenerar key={entorno} entorno={entorno} api={api} apiEntrenamiento={apiEntrenamiento}
        alCambiarOcupado={avisarExplorar} />
      <article className="panel" aria-labelledby="panel-entrenar">
        <h2 id="panel-entrenar">Entrenar e imitar</h2>
        <p className="nota">Entrena GAIL o BC con las demostraciones que elijas.</p>
        <PanelEntrenamiento key={entorno} entorno={entorno} api={apiEntrenamiento} metodos={IMITAR}
          alCambiarOcupado={avisarImitar} />
      </article>
    </section>
  );
}
