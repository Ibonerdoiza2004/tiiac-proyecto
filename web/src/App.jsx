import { useState } from 'react';

import Cabecera from './componentes/Cabecera.jsx';
import ContextoEntorno from './componentes/ContextoEntorno.jsx';
import SelectorEntorno from './componentes/SelectorEntorno.jsx';
import { ENTORNO_INICIAL, buscarEntorno } from './entornos.js';
import { VISTA_INICIAL, buscarVista } from './vistas.js';
import Biblioteca from './vistas/Biblioteca.jsx';
import Comparar from './vistas/Comparar.jsx';
import Laboratorio from './vistas/Laboratorio.jsx';

const CONTENIDO = { laboratorio: Laboratorio, comparar: Comparar, biblioteca: Biblioteca };

/**
 * Raíz del dashboard: pestaña activa y entorno elegido, comunes a todas las pestañas.
 *
 * @param {object} [props]
 * @param {string} [props.vistaInicial] Pestaña con la que arranca.
 * @param {string} [props.entornoInicial] Entorno con el que arranca.
 * @returns {JSX.Element} La aplicación.
 */
export default function App({ vistaInicial = VISTA_INICIAL, entornoInicial = ENTORNO_INICIAL }) {
  const [vista, setVista] = useState(() => buscarVista(vistaInicial).id);
  const [entorno, setEntorno] = useState(() => buscarEntorno(entornoInicial).id);
  const { titulo, subtitulo } = buscarVista(vista);
  const Contenido = CONTENIDO[vista];

  return (
    <div className="dashboard">
      <Cabecera vista={vista} alCambiarVista={setVista} />
      <main className="principal">
        <div className="introduccion">
          <div>
            <h1>{titulo}</h1>
            <p className="subtitulo">{subtitulo}</p>
          </div>
          <SelectorEntorno entorno={entorno} alCambiarEntorno={setEntorno} />
        </div>
        <ContextoEntorno entorno={entorno} />
        <Contenido entorno={entorno} />
      </main>
    </div>
  );
}
