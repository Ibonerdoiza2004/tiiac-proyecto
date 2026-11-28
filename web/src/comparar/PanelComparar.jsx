import { useEffect, useId, useState } from 'react';

import { ALEATORIO, MAXIMO_PARTIDAS, TIPOS_MODELO } from '../api/comparar.js';
import { ESTADOS_TRABAJO } from '../api/trabajos.js';
import useCarga from '../componentes/useCarga.js';
import { buscarEntorno } from '../entornos.js';
import VistaEntorno from '../laboratorio/VistaEntorno.jsx';
import { partidasCompletas, resumir } from './estadisticas.js';
import Resultados from './Resultados.jsx';
import SelectorModelos from './SelectorModelos.jsx';
import useComparacion, { FASES_COMPARACION } from './useComparacion.js';

/** Fases en las que hay una comparación en marcha y no se puede cambiar de entorno. */
export const FASES_COMPARACION_OCUPADAS = new Set([FASES_COMPARACION.PREPARANDO,
  FASES_COMPARACION.SIGUIENDO, FASES_COMPARACION.TERMINANDO]);

const ETIQUETAS_ESTADO = {
  [ESTADOS_TRABAJO.EN_COLA]: 'En cola',
  [ESTADOS_TRABAJO.EN_MARCHA]: 'Comparando',
  [ESTADOS_TRABAJO.PAUSADO]: 'Pausada',
  [ESTADOS_TRABAJO.TERMINADO]: 'Terminada',
  [ESTADOS_TRABAJO.FALLIDO]: 'Falló',
  [ESTADOS_TRABAJO.CANCELADO]: 'Cancelada',
};

/** Partidas escritas: undefined si se deja vacío (las de la evaluación), null si no valen. */
export function leerPartidas(texto) {
  const limpio = String(texto).trim();
  if (!limpio) {
    return undefined;
  }
  const valor = Number(limpio);
  return /^\d+$/.test(limpio) && valor >= 1 && valor <= MAXIMO_PARTIDAS ? valor : null;
}

/** Fecha corta de un trabajo (segundos desde 1970). */
function fecha(segundos) {
  if (typeof segundos !== 'number') {
    return '';
  }
  return new Date(segundos * 1000).toLocaleString('es-ES', { day: '2-digit', month: '2-digit',
    hour: '2-digit', minute: '2-digit' });
}

/**
 * Pestaña Comparar: elegir modelos de la biblioteca y cuántas partidas, seguir la
 * comparación (con vista opcional), ver media ± desviación y la puntuación de cada
 * partida, y abrir comparaciones anteriores.
 *
 * @param {object} props
 * @param {string} props.entorno Clave del entorno.
 * @param {import('../api/comparar.js').ApiComparar} props.api
 * @param {(ocupado: boolean) => void} [props.alCambiarOcupado] Avisa si hay una en marcha.
 * @returns {JSX.Element}
 */
export default function PanelComparar({ entorno, api, alCambiarOcupado }) {
  const id = useId();
  const { nombre } = buscarEntorno(entorno);
  const comp = useComparacion(api);
  const { fase, trabajo } = comp;
  const ocupado = FASES_COMPARACION_OCUPADAS.has(fase);

  const modelos = useCarga(() => api.modelos(entorno), entorno);
  // La lista se vuelve a pedir al empezar o terminar una comparación, y si la que se
  // intenta ver ya no existe en el servidor.
  const anteriores = useCarga(() => api.comparaciones(entorno),
    `${entorno}:${trabajo?.id ?? ''}:${fase === FASES_COMPARACION.TERMINADA}:${fase === FASES_COMPARACION.INEXISTENTE}`);
  const [elegidos, setElegidos] = useState(new Set());
  const [aleatorio, setAleatorio] = useState(true);
  const [partidas, setPartidas] = useState('');
  const [vista, setVista] = useState(false);
  const [intentado, setIntentado] = useState(false);
  const { nueva } = comp;

  useEffect(() => {
    nueva();
    setElegidos(new Set());
    setIntentado(false);
  }, [entorno, nueva]);

  useEffect(() => {
    alCambiarOcupado?.(ocupado);
  }, [alCambiarOcupado, ocupado]);

  const porId = new Map((modelos.datos ?? []).map((m) => [m.id, m]));
  const comparables = (modelos.datos ?? []).filter((m) => TIPOS_MODELO.includes(m.tipo));
  const nombreDe = (clave) => (clave === ALEATORIO ? 'Aleatorio' : porId.get(clave)?.nombre ?? clave);

  const valorPartidas = leerPartidas(partidas);
  const errorPartidas = valorPartidas === null
    ? `Escribe un número entero de partidas entre 1 y ${MAXIMO_PARTIDAS}, o déjalo vacío.` : null;
  // Ids en el orden de la biblioteca, para que la tabla salga igual cada vez; solo los que
  // siguen en ella y son válidos (la lista se puede actualizar después de elegir).
  const ids = comparables.filter((m) => m.valido && elegidos.has(m.id)).map((m) => m.id);
  const avisoModelos = ids.length + Number(aleatorio) < 2
    ? 'Elige al menos dos participantes (puede ser uno y el aleatorio).' : null;

  const iniciar = () => {
    setIntentado(true);
    if (errorPartidas || avisoModelos) {
      return;
    }
    // Vacío da undefined, que no viaja en el JSON: el servidor usa las de la evaluación.
    comp.iniciar({ entorno, modelos: ids, aleatorio, partidas: valorPartidas, vista });
  };

  const argumentos = trabajo?.argumentos;
  const participantes = argumentos
    ? [...(argumentos.modelos ?? []), ...(argumentos.aleatorio ? [ALEATORIO] : [])]
    : [...ids, ...(aleatorio ? [ALEATORIO] : [])];
  const total = argumentos?.partidas;
  const completas = partidasCompletas(comp.acumulado, participantes);
  const porcentaje = total ? Math.min(100, Math.round((completas / total) * 100)) : 0;
  const estadoTrabajo = trabajo?.estado;
  const enMarcha = estadoTrabajo === ESTADOS_TRABAJO.EN_MARCHA;
  const etiqueta = { [FASES_COMPARACION.PREPARANDO]: 'Preparando…', [FASES_COMPARACION.TERMINANDO]: 'Terminando…',
    [FASES_COMPARACION.DESCONECTADA]: 'Sin conexión', [FASES_COMPARACION.INEXISTENTE]: 'No existe' }[fase]
    ?? ETIQUETAS_ESTADO[estadoTrabajo] ?? 'Conectando…';
  const siguiendo = fase === FASES_COMPARACION.SIGUIENDO;

  return (
    <div className="comparar">
      {fase === FASES_COMPARACION.INACTIVA ? (
        <form aria-label="Nueva comparación" noValidate onSubmit={(e) => {
          e.preventDefault();
          iniciar();
        }}>
          <SelectorModelos modelos={modelos.datos ? comparables : null} error={modelos.error}
            elegidos={elegidos} aleatorio={aleatorio} alActualizar={modelos.recargar}
            alCambiarAleatorio={setAleatorio}
            alCambiar={(clave, elegido) => setElegidos((antes) => {
              const nuevos = new Set(antes);
              if (elegido) {
                nuevos.add(clave);
              } else {
                nuevos.delete(clave);
              }
              return nuevos;
            })} />
          {intentado && avisoModelos ? <p className="error" role="alert">{avisoModelos}</p> : null}

          <label className="campo campo-corto" htmlFor={`${id}-partidas`}>
            <span>Partidas por modelo</span>
            <input id={`${id}-partidas`} type="number" inputMode="numeric" min="1" max={MAXIMO_PARTIDAS}
              placeholder="Las de la evaluación de la E1" value={partidas}
              aria-invalid={intentado && Boolean(errorPartidas)} aria-describedby={`${id}-ayuda-partidas`}
              onChange={(e) => setPartidas(e.target.value)} />
          </label>
          <p className="nota" id={`${id}-ayuda-partidas`}>
            Vacío: las mismas partidas y semilla que la evaluación de la E1. Todos los modelos
            juegan cada partida con la misma semilla.
          </p>
          {intentado && errorPartidas ? <p className="error" role="alert">{errorPartidas}</p> : null}

          <label className="casilla" htmlFor={`${id}-vista`}>
            <input id={`${id}-vista`} type="checkbox" checked={vista} onChange={(e) => setVista(e.target.checked)} />
            Ver las partidas mientras compara (más lento; se puede cambiar en marcha)
          </label>
          <div className="controles">
            <button type="submit" className="primario" disabled={!modelos.datos}>Iniciar comparación</button>
          </div>
        </form>
      ) : (
        <>
          <div className="progreso-cabecera">
            <span>
              {participantes.length} participantes{total ? ` · ${total} partidas` : ''}
              {typeof argumentos?.semilla === 'number' ? ` · semilla ${argumentos.semilla}` : ''}
            </span>
            <span role="status">{etiqueta}</span>
          </div>
          <div className="barra" role="progressbar" aria-label="Progreso de la comparación"
            aria-valuemin={0} aria-valuemax={100} aria-valuenow={porcentaje}>
            <div style={{ width: `${porcentaje}%` }} />
          </div>
          <p className="nota">
            {completas}{total ? ` de ${total}` : ''} partidas jugadas por todos los modelos
            {fase === FASES_COMPARACION.TERMINADA && trabajo?.resultado?.incompleta
              ? ' · finalizada antes de tiempo: los resultados son de las partidas jugadas' : ''}
          </p>

          {siguiendo && !trabajo?.resultado ? (
            <label className="casilla" htmlFor={`${id}-vista-viva`}>
              <input id={`${id}-vista-viva`} type="checkbox" checked={comp.vista} disabled={!enMarcha}
                onChange={(e) => comp.cambiarVista(e.target.checked)} />
              Ver las partidas
            </label>
          ) : null}
          {comp.vista && siguiendo ? (
            <VistaEntorno imagen={comp.imagen} nombre={nombre} vacia="Esperando la primera imagen…" />
          ) : null}

          <Resultados resumenes={resumir(comp.acumulado, participantes)} nombre={nombreDe}
            modelo={(clave) => porId.get(clave)} />

          <div className="controles">
            {siguiendo ? (
              <>
                {estadoTrabajo === ESTADOS_TRABAJO.PAUSADO ? (
                  <button type="button" onClick={comp.reanudar}>Reanudar</button>
                ) : (
                  <button type="button" onClick={comp.pausar} disabled={!enMarcha}>Pausar</button>
                )}
                <button type="button" onClick={comp.cancelar}>Cancelar</button>
                <button type="button" className="primario" onClick={comp.finalizar}
                  disabled={!enMarcha && estadoTrabajo !== ESTADOS_TRABAJO.PAUSADO}>Finalizar</button>
              </>
            ) : null}
            {fase === FASES_COMPARACION.DESCONECTADA ? (
              <button type="button" onClick={comp.reconectar}>Reconectar</button>
            ) : null}
            {[FASES_COMPARACION.TERMINADA, FASES_COMPARACION.DESCONECTADA, FASES_COMPARACION.INEXISTENTE].includes(fase) ? (
              <button type="button" className="primario" onClick={comp.nueva}>Nueva comparación</button>
            ) : null}
          </div>
          {siguiendo ? (
            <p className="nota">
              Finalizar se queda con las partidas jugadas; cancelar descarta la comparación.
            </p>
          ) : null}
        </>
      )}
      {comp.error ? <p className="error" role="alert">{comp.error}</p> : null}

      <div className="anteriores">
        <div className="anteriores-cabecera">
          <h3 id={`${id}-anteriores`}>Comparaciones anteriores de {nombre}</h3>
          <button type="button" className="enlace" onClick={anteriores.recargar}>Actualizar</button>
        </div>
        {anteriores.error ? (
          <p className="error" role="alert">No se pudo cargar la lista ({anteriores.error}).</p>
        ) : null}
        {anteriores.datos?.length === 0 ? <p className="nota">Todavía no hay ninguna.</p> : null}
        {anteriores.datos?.length ? (
          <ul className="anteriores-lista">
            {anteriores.datos.map((t) => (
              <li key={t.id}>
                <span>
                  {fecha(t.creado)} · {(t.argumentos?.modelos?.length ?? 0) + Number(Boolean(t.argumentos?.aleatorio))}{' '}
                  participantes · {t.argumentos?.partidas ?? '?'} partidas
                </span>
                <span className="apagado">{ETIQUETAS_ESTADO[t.estado] ?? t.estado}</span>
                <button type="button" disabled={ocupado || t.id === trabajo?.id}
                  onClick={() => comp.ver(t)}>Ver</button>
              </li>
            ))}
          </ul>
        ) : null}
      </div>
    </div>
  );
}
