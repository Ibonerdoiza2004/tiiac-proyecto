import { useEffect, useId, useState } from 'react';

import { aFormulario, idCampo, leerCampo } from './ajustes.js';

const cambiado = (campo, entrada) => {
  const leido = leerCampo(campo, entrada);
  return 'error' in leido || JSON.stringify(leido.valor) !== JSON.stringify(campo.valor);
};

/**
 * Hiperparámetros generados a partir del esquema que manda el backend: una sección
 * plegable por sección de configs/, con el valor del perfil y de dónde sale. La primera
 * empieza abierta, las demás se abren a mano y una sección con errores se abre sola.
 *
 * @param {object} props
 * @param {import('../api/entrenamiento.js').EsquemaAjustes} props.esquema
 * @param {Record<string, string | boolean>} props.valores Valores por idCampo.
 * @param {Record<string, string>} props.errores Errores por idCampo (solo se muestran).
 * @param {(id: string, valor: string | boolean) => void} props.alCambiar
 * @returns {JSX.Element}
 */
export default function FormularioAjustes({ esquema, valores, errores, alCambiar }) {
  const prefijo = useId();
  const [abiertas, setAbiertas] = useState(() => new Set(esquema.secciones.slice(0, 1).map((s) => s.clave)));
  const conErrores = esquema.secciones
    .filter((s) => s.campos.some((campo) => errores[idCampo(s, campo)]))
    .map((s) => s.clave).join('|');
  useEffect(() => {
    if (conErrores) {
      setAbiertas((anteriores) => new Set([...anteriores, ...conErrores.split('|')]));
    }
  }, [conErrores]);
  const alternar = (clave, abierta) => setAbiertas((anteriores) => {
    if (anteriores.has(clave) === abierta) {
      return anteriores;
    }
    const nuevas = new Set(anteriores);
    if (abierta) {
      nuevas.add(clave);
    } else {
      nuevas.delete(clave);
    }
    return nuevas;
  });
  return (
    <div className="ajustes">
      {esquema.secciones.map((seccion) => {
        const cambios = seccion.campos.filter((campo) => cambiado(campo, valores[idCampo(seccion, campo)])).length;
        return (
          <details key={seccion.clave} className="ajustes-seccion" open={abiertas.has(seccion.clave)}
            onToggle={(e) => alternar(seccion.clave, e.currentTarget.open)}>
            <summary>
              <code>{seccion.clave}</code>
              <span className="apagado">{seccion.campos.length} valores{cambios ? ` · ${cambios} cambiados` : ''}</span>
            </summary>
            {seccion.campos.map((campo) => {
              const id = idCampo(seccion, campo);
              const dom = `${prefijo}-${id}`;
              // Hasta que el panel copia los valores del perfil, se muestra el del perfil.
              const valor = valores[id] ?? aFormulario(campo, campo.valor);
              const error = errores[id];
              const distinto = cambiado(campo, valor);
              const describe = [campo.ayuda ? `${dom}-ayuda` : null, error ? `${dom}-error` : null]
                .filter(Boolean).join(' ') || undefined;
              const comun = { id: dom, 'aria-invalid': Boolean(error), 'aria-describedby': describe };
              return (
                <div key={id} className={`ajuste${distinto ? ' ajuste-cambiado' : ''}`}>
                  {campo.tipo === 'booleano' ? (
                    <label className="ajuste-casilla" htmlFor={dom}>
                      <input type="checkbox" {...comun} checked={Boolean(valor)}
                        onChange={(e) => alCambiar(id, e.target.checked)} />
                      <code>{campo.clave}</code>
                    </label>
                  ) : (
                    <>
                      <label htmlFor={dom}><code>{campo.clave}</code></label>
                      {campo.tipo === 'opcion' ? (
                        <select {...comun} value={valor} onChange={(e) => alCambiar(id, e.target.value)}>
                          {campo.opciones.map((opcion) => <option key={opcion} value={opcion}>{opcion}</option>)}
                        </select>
                      ) : (
                        <input type="text" {...comun} value={valor} spellCheck={false}
                          inputMode={campo.tipo === 'lista_enteros' ? 'text' : 'decimal'}
                          onChange={(e) => alCambiar(id, e.target.value)} />
                      )}
                    </>
                  )}
                  {distinto ? (
                    <span className="ajuste-perfil">Perfil: {String(aFormulario(campo, campo.valor))}</span>
                  ) : null}
                  {campo.ayuda ? <p className="nota" id={`${dom}-ayuda`}>{campo.ayuda}</p> : null}
                  {error ? <p className="error" id={`${dom}-error`}>{error}</p> : null}
                </div>
              );
            })}
          </details>
        );
      })}
    </div>
  );
}
