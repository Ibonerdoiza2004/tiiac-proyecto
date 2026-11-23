import { useId, useState } from 'react';

import { formatearNumero, nombreMedida, trazo } from './curvas.js';

const TAMANO = { ancho: 600, alto: 160 };

/**
 * Curva en directo de una medida del entrenamiento, con un desplegable para elegir cuál.
 *
 * @param {object} props
 * @param {Record<string, [number, number][]>} props.series Puntos [paso, valor] por medida.
 * @param {string} props.unidad Unidad del eje horizontal (pasos, épocas, iteraciones).
 * @returns {JSX.Element}
 */
export default function Curvas({ series, unidad }) {
  const id = useId();
  const medidas = Object.keys(series);
  const [elegida, setElegida] = useState(null);
  const medida = medidas.includes(elegida) ? elegida : medidas[0];

  if (!medida) {
    return <p className="curvas-vacia">La curva aparecerá con el primer progreso.</p>;
  }
  const puntos = series[medida];
  const linea = trazo(puntos, TAMANO);
  const ultimo = puntos.at(-1)[1];
  const nombre = nombreMedida(medida);

  return (
    <div className="curvas">
      <div className="curvas-cabecera">
        {medidas.length > 1 ? (
          <label className="campo" htmlFor={`${id}-medida`}>Medida
            <select id={`${id}-medida`} value={medida} onChange={(e) => setElegida(e.target.value)}>
              {medidas.map((m) => <option key={m} value={m}>{nombreMedida(m)}</option>)}
            </select>
          </label>
        ) : <span>{nombre}</span>}
        <span className="curvas-ultimo">Último: <strong>{formatearNumero(ultimo)}</strong></span>
      </div>
      <svg viewBox={`0 0 ${TAMANO.ancho} ${TAMANO.alto}`} preserveAspectRatio="none" role="img"
        aria-label={`${nombre}: ${puntos.length} puntos, último valor ${formatearNumero(ultimo)}`}>
        <path d={linea.d} fill="none" stroke="currentColor" strokeWidth="2" vectorEffect="non-scaling-stroke" />
      </svg>
      <div className="curvas-ejes">
        <span>{formatearNumero(linea.minimo)} – {formatearNumero(linea.maximo)}</span>
        <span>{formatearNumero(linea.desde)} – {formatearNumero(linea.hasta)} {unidad}</span>
      </div>
    </div>
  );
}
