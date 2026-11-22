import { VISTAS } from '../vistas.js';

/**
 * Cabecera con la marca y las pestañas principales.
 *
 * @param {object} props
 * @param {string} props.vista Clave de la pestaña activa.
 * @param {(vista: string) => void} props.alCambiarVista Recibe la pestaña elegida.
 * @returns {JSX.Element} Cabecera del dashboard.
 */
export default function Cabecera({ vista, alCambiarVista }) {
  return (
    <header className="cabecera">
      <div className="marca">
        GAIL<span className="apagado">lab</span>
      </div>
      <nav className="navegacion" aria-label="Navegación principal">
        {VISTAS.map((opcion) => (
          <button
            key={opcion.id}
            type="button"
            aria-pressed={opcion.id === vista}
            onClick={() => alCambiarVista(opcion.id)}
          >
            {opcion.etiqueta}
          </button>
        ))}
      </nav>
    </header>
  );
}
