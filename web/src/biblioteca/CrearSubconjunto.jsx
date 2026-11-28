import { useEffect, useId, useRef, useState } from 'react';

import { LARGO_NOMBRE } from '../api/biblioteca.js';
import TablaPartidas from './TablaPartidas.jsx';
import { errorNombre, formatearCifra, resumenEleccion } from './textos.js';

/** Milisegundos que se espera la respuesta al crear el subconjunto. */
export const ESPERA_CREAR_MS = 15_000;
const SIN_RESPUESTA = 'El servidor no respondió al crear el subconjunto.';
const mensajeDe = (error) => (error instanceof Error ? error.message : String(error));

/**
 * Formulario para crear un subconjunto de un conjunto eligiendo partidas enteras.
 *
 * @param {object} props
 * @param {import('../api/biblioteca.js').ApiBiblioteca} props.api
 * @param {import('../api/biblioteca.js').ElementoBiblioteca} props.conjunto
 * @param {import('../api/biblioteca.js').Partida[]} props.partidas Las del conjunto.
 * @param {(elemento: import('../api/biblioteca.js').ElementoBiblioteca) => void} props.alCrear
 * @param {() => void} props.alCancelar
 * @returns {JSX.Element}
 */
export default function CrearSubconjunto({ api, conjunto, partidas, alCrear, alCancelar }) {
  const id = useId();
  const [elegidas, setElegidas] = useState(new Set());
  const [nombre, setNombre] = useState(() => `Subconjunto de ${conjunto.nombre}`.slice(0, LARGO_NOMBRE));
  const [intentado, setIntentado] = useState(false);
  const [creando, setCreando] = useState(false);
  const [error, setError] = useState(null);
  const montado = useRef(true);

  // En el cuerpo y no solo en la limpieza: en desarrollo React monta los efectos dos veces.
  useEffect(() => {
    montado.current = true;
    return () => {
      montado.current = false;
    };
  }, []);

  const resumen = resumenEleccion(partidas, elegidas);
  const avisoNombre = errorNombre(nombre);
  const avisoPartidas = resumen.partidas === 0 ? 'Elige al menos una partida.' : null;
  const deTrain = partidas.filter((p) => p.split === 'train');
  const elegirSolo = (lista) => setElegidas(new Set(lista.map((p) => p.id_episodio)));

  const crear = async () => {
    setIntentado(true);
    setError(null);
    if (avisoNombre || avisoPartidas) {
      return;
    }
    setCreando(true);
    const control = new AbortController();
    const espera = setTimeout(() => control.abort(), ESPERA_CREAR_MS);
    try {
      const nuevo = await api.crearSubconjunto({ conjunto: conjunto.id,
        partidas: [...elegidas].sort((a, b) => a - b), nombre: nombre.trim() }, { senal: control.signal });
      if (montado.current) {
        alCrear(nuevo);
      }
    } catch (e) {
      if (montado.current) {
        setError(control.signal.aborted ? SIN_RESPUESTA : mensajeDe(e));
      }
    } finally {
      clearTimeout(espera);
      if (montado.current) {
        setCreando(false);
      }
    }
  };

  return (
    <form className="crear-subconjunto" aria-label="Crear subconjunto" noValidate onSubmit={(e) => {
      e.preventDefault();
      crear();
    }}>
      <h4>Nuevo subconjunto</h4>
      <p className="nota">
        Elige partidas enteras de este conjunto. El subconjunto guarda solo cuáles son: si el
        conjunto cambia o se borra, deja de ser válido. Al entrenar con él, las partidas se
        reparten otra vez en train y validación.
      </p>
      <div className="controles controles-compactos">
        <button type="button" disabled={creando} onClick={() => elegirSolo(partidas)}>Todas</button>
        <button type="button" disabled={creando} onClick={() => setElegidas(new Set())}>Ninguna</button>
        {deTrain.length ? (
          <button type="button" disabled={creando} onClick={() => elegirSolo(deTrain)}>Solo las de train</button>
        ) : null}
      </div>
      <TablaPartidas partidas={partidas} elegidas={elegidas} bloqueada={creando}
        alCambiar={(partida, elegida) => setElegidas((antes) => {
          const nuevas = new Set(antes);
          if (elegida) {
            nuevas.add(partida);
          } else {
            nuevas.delete(partida);
          }
          return nuevas;
        })} />
      <p className="nota" role="status">
        {resumen.partidas} {resumen.partidas === 1 ? 'partida elegida' : 'partidas elegidas'} ·{' '}
        {formatearCifra(resumen.pares)} pares
      </p>
      {intentado && avisoPartidas ? <p className="error" role="alert">{avisoPartidas}</p> : null}

      <label className="campo" htmlFor={`${id}-nombre`}>
        <span>Nombre</span>
        <input id={`${id}-nombre`} type="text" value={nombre} maxLength={LARGO_NOMBRE} disabled={creando}
          aria-invalid={intentado && Boolean(avisoNombre)} onChange={(e) => setNombre(e.target.value)} />
      </label>
      {intentado && avisoNombre ? <p className="error" role="alert">{avisoNombre}</p> : null}
      {error ? <p className="error" role="alert">{error}</p> : null}
      <div className="controles">
        <button type="button" disabled={creando} onClick={alCancelar}>Cancelar</button>
        <button type="submit" className="primario" disabled={creando}>
          {creando ? 'Guardando…' : 'Guardar subconjunto'}
        </button>
      </div>
    </form>
  );
}
