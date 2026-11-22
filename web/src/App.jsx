import { useState } from 'react';

import { crearApiPorDefecto } from './api/elegir.js';
import Cabecera from './componentes/Cabecera.jsx';
import ContextoEntorno from './componentes/ContextoEntorno.jsx';
import SelectorEntorno from './componentes/SelectorEntorno.jsx';
import { ENTORNO_INICIAL, buscarEntorno } from './entornos.js';
import { VISTAS, VISTA_INICIAL, buscarVista } from './vistas.js';
import Biblioteca from './vistas/Biblioteca.jsx';
import Comparar from './vistas/Comparar.jsx';
import Laboratorio from './vistas/Laboratorio.jsx';

const CONTENIDO = { laboratorio: Laboratorio, comparar: Comparar, biblioteca: Biblioteca };

/**
 * Raíz del dashboard: pestaña activa y entorno elegido, comunes a todas las pestañas.
 *
 * Las tres pestañas siguen montadas aunque no se vean, para que una sesión del
 * Laboratorio no se pierda al cambiar de pestaña. Con una sesión abierta no se puede
 * cambiar de entorno.
 *
 * @param {object} [props]
 * @param {string} [props.vistaInicial] Pestaña con la que arranca.
 * @param {string} [props.entornoInicial] Entorno con el que arranca.
 * @param {import('./api/laboratorio.js').ApiLaboratorio} [props.apiLaboratorio] Backend del
 *   Laboratorio (por defecto, el que indique VITE_API).
 * @returns {JSX.Element} La aplicación.
 */
export default function App({ vistaInicial = VISTA_INICIAL, entornoInicial = ENTORNO_INICIAL,
  apiLaboratorio }) {
  const [vista, setVista] = useState(() => buscarVista(vistaInicial).id);
  const [entorno, setEntorno] = useState(() => buscarEntorno(entornoInicial).id);
  const [api] = useState(() => apiLaboratorio ?? crearApiPorDefecto());
  const [ocupado, setOcupado] = useState(false);
  const { titulo, subtitulo } = buscarVista(vista);

  return (
    <div className="dashboard">
      <Cabecera vista={vista} alCambiarVista={setVista} />
      <main className="principal">
        <div className="introduccion">
          <div>
            <h1>{titulo}</h1>
            <p className="subtitulo">{subtitulo}</p>
          </div>
          <SelectorEntorno entorno={entorno} alCambiarEntorno={setEntorno} bloqueado={ocupado} />
        </div>
        <ContextoEntorno entorno={entorno} />
        {VISTAS.map(({ id }) => {
          const Contenido = CONTENIDO[id];
          return (
            <div key={id} hidden={id !== vista}>
              <Contenido entorno={entorno} api={api} alCambiarOcupado={setOcupado} />
            </div>
          );
        })}
      </main>
    </div>
  );
}
