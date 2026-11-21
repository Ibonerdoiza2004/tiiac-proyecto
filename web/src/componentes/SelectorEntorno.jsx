import { ENTORNOS } from '../entornos.js';

/**
 * Desplegable con los cinco entornos.
 *
 * @param {object} props
 * @param {string} props.entorno Clave del entorno elegido.
 * @param {(entorno: string) => void} props.alCambiarEntorno Recibe la clave elegida.
 * @param {boolean} [props.bloqueado] Impide cambiarlo (por ejemplo, con un trabajo en
 *   marcha en ese entorno).
 * @returns {JSX.Element} Campo de selección del entorno.
 */
export default function SelectorEntorno({ entorno, alCambiarEntorno, bloqueado = false }) {
  return (
    <label className="campo selector-entorno">
      <span>Entorno</span>
      <select
        value={entorno}
        disabled={bloqueado}
        aria-describedby="contexto-entorno"
        title={bloqueado ? 'Termina las ejecuciones en marcha para cambiarlo' : undefined}
        onChange={(evento) => alCambiarEntorno(evento.target.value)}
      >
        {ENTORNOS.map((opcion) => (
          <option key={opcion.id} value={opcion.id}>
            {opcion.etapa} · {opcion.nombre}
          </option>
        ))}
      </select>
    </label>
  );
}
