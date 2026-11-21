import { ENTORNOS, buscarEntorno } from '../entornos.js';

/**
 * Franja con la etapa, el objetivo y el tipo de control del entorno elegido.
 *
 * @param {object} props
 * @param {string} props.entorno Clave del entorno.
 * @returns {JSX.Element} Descripción del entorno.
 */
export default function ContextoEntorno({ entorno }) {
  const datos = buscarEntorno(entorno);
  return (
    <div className="contexto-entorno" id="contexto-entorno" aria-live="polite">
      <strong>
        Etapa {datos.etapa} de {ENTORNOS.length}
      </strong>
      <p>{datos.objetivo}</p>
      <span>
        {datos.gymnasium} · acciones {datos.acciones}
        {datos.jugarAMano ? ' · se puede jugar a mano' : ''}
      </span>
    </div>
  );
}
