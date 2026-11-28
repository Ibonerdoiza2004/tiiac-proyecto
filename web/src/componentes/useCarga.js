import { useCallback, useEffect, useState } from 'react';

const mensajeDe = (error) => (error instanceof Error ? error.message : String(error));

/**
 * Pide algo al backend cuando cambia la clave y descarta la respuesta si llega tarde (de
 * una clave anterior o después de desmontar).
 *
 * Al volver a pedir con la misma clave (recargar) se conservan los datos que había mientras
 * llega la respuesta, para que la lista no parpadee; con otra clave, se vacían.
 *
 * @param {(() => Promise<unknown>) | null} cargar Hace la petición; null, no pide nada.
 * @param {string} clave Lo que determina la petición: si cambia, se vuelve a pedir.
 * @returns {{datos: unknown, error: string | null, recargar: () => void}}
 */
export default function useCarga(cargar, clave) {
  const [estado, setEstado] = useState({ datos: null, error: null, clave: null });
  const [intento, setIntento] = useState(0);
  useEffect(() => {
    if (!cargar) {
      return undefined;
    }
    let vigente = true;
    setEstado((anterior) => ({ datos: anterior.clave === clave ? anterior.datos : null, error: null, clave }));
    cargar().then((datos) => vigente && setEstado({ datos, error: null, clave }))
      .catch((error) => vigente && setEstado({ datos: null, error: mensajeDe(error), clave }));
    return () => {
      vigente = false;
    };
    // cargar es una función nueva en cada render: lo que cambia la petición es clave, e
    // intento la repite.
  }, [clave, intento]);
  return { datos: estado.datos, error: estado.error, recargar: useCallback(() => setIntento((n) => n + 1), []) };
}
