import { useState } from 'react';

import { crearApisPorDefecto } from './api/elegir.js';
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
 * Laboratorio no se pierda al cambiar de pestaña. Con algo en marcha en cualquier
 * pestaña (una sesión, un entrenamiento o una comparación) no se puede cambiar de entorno.
 *
 * @param {object} [props]
 * @param {string} [props.vistaInicial] Pestaña con la que arranca.
 * @param {string} [props.entornoInicial] Entorno con el que arranca.
 * @param {import('./api/laboratorio.js').ApiLaboratorio} [props.apiLaboratorio] Backend de
 *   jugar y ver al agente (por defecto, el que indique VITE_API).
 * @param {import('./api/entrenamiento.js').ApiEntrenamiento} [props.apiEntrenamiento] Backend
 *   de los entrenamientos (por defecto, el que indique VITE_API).
 * @param {import('./api/comparar.js').ApiComparar} [props.apiComparar] Backend de Comparar
 *   (por defecto, el que indique VITE_API).
 * @param {import('./api/biblioteca.js').ApiBiblioteca} [props.apiBiblioteca] Backend de la
 *   Biblioteca (por defecto, el que indique VITE_API).
 * @returns {JSX.Element} La aplicación.
 */
export default function App({ vistaInicial = VISTA_INICIAL, entornoInicial = ENTORNO_INICIAL,
  apiLaboratorio, apiEntrenamiento, apiComparar, apiBiblioteca }) {
  const [vista, setVista] = useState(() => buscarVista(vistaInicial).id);
  const [entorno, setEntorno] = useState(() => buscarEntorno(entornoInicial).id);
  const [apis] = useState(() => {
    const porDefecto = apiLaboratorio && apiEntrenamiento && apiComparar && apiBiblioteca ? null
      : crearApisPorDefecto();
    return { laboratorio: apiLaboratorio ?? porDefecto.laboratorio,
      entrenamiento: apiEntrenamiento ?? porDefecto.entrenamiento,
      comparar: apiComparar ?? porDefecto.comparar,
      biblioteca: apiBiblioteca ?? porDefecto.biblioteca };
  });
  // Cada pestaña avisa por separado: si una se libera, las demás siguen bloqueando.
  const [ocupadas, setOcupadas] = useState({});
  const ocupado = Object.values(ocupadas).some(Boolean);
  const [avisos] = useState(() => Object.fromEntries(VISTAS.map(({ id }) => [id,
    (valor) => setOcupadas((antes) => (antes[id] === valor ? antes : { ...antes, [id]: valor }))])));
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
              <Contenido entorno={entorno} api={apis.laboratorio} apiEntrenamiento={apis.entrenamiento}
                apiComparar={apis.comparar} apiBiblioteca={apis.biblioteca} alCambiarOcupado={avisos[id]} />
            </div>
          );
        })}
      </main>
    </div>
  );
}
