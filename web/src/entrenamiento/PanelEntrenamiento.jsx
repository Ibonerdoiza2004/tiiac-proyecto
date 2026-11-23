import { useCallback, useEffect, useId, useState } from 'react';

import { METODOS, METODOS_CON_DATOS, METODOS_CON_VISTA } from '../api/entrenamiento.js';
import { buscarEntorno } from '../entornos.js';
import VistaEntorno from '../laboratorio/VistaEntorno.jsx';
import { valoresIniciales, validarAjustes } from './ajustes.js';
import Curvas from './Curvas.jsx';
import { formatearNumero } from './curvas.js';
import FormularioAjustes from './FormularioAjustes.jsx';
import SelectorDatos from './SelectorDatos.jsx';
import useEntrenamiento, { FASES_ENTRENO, FASES_ENTRENO_OCUPADAS } from './useEntrenamiento.js';

const NOMBRES = { [METODOS.EXPERTO]: 'Experto (TRPO)', [METODOS.GAIL]: 'GAIL', [METODOS.BC]: 'BC' };
const UNIDADES = { pasos: 'pasos', epocas: 'épocas', iteraciones: 'iteraciones' };
const mensajeDe = (error) => (error instanceof Error ? error.message : String(error));

/** Pide algo al backend cuando cambia la clave y lo descarta si llega tarde. */
function useCarga(cargar, clave) {
  const [estado, setEstado] = useState({ datos: null, error: null });
  const [intento, setIntento] = useState(0);
  useEffect(() => {
    if (!cargar) {
      return undefined;
    }
    let vigente = true;
    setEstado({ datos: null, error: null });
    cargar().then((datos) => vigente && setEstado({ datos, error: null }))
      .catch((error) => vigente && setEstado({ datos: null, error: mensajeDe(error) }));
    return () => {
      vigente = false;
    };
    // cargar es una función nueva en cada render: lo que cambia lo que pide es clave, e
    // intento fuerza otra petición.
  }, [clave, intento]);
  return { ...estado, recargar: useCallback(() => setIntento((n) => n + 1), []) };
}

/**
 * Formulario, progreso y controles de un entrenamiento. Se usa con ['experto'] en
 * «Explorar y generar» y con ['gail', 'bc'] en «Entrenar e imitar».
 *
 * @param {object} props
 * @param {string} props.entorno Clave del entorno.
 * @param {import('../api/entrenamiento.js').ApiEntrenamiento} props.api
 * @param {string[]} props.metodos Métodos que se pueden elegir (el primero, por defecto).
 * @param {(ocupado: boolean) => void} [props.alCambiarOcupado] Avisa si hay uno en marcha.
 * @returns {JSX.Element}
 */
export default function PanelEntrenamiento({ entorno, api, metodos, alCambiarOcupado }) {
  const id = useId();
  const { nombre } = buscarEntorno(entorno);
  const entreno = useEntrenamiento(api);
  const { fase } = entreno;
  const ocupado = FASES_ENTRENO_OCUPADAS.has(fase);
  const [metodo, setMetodo] = useState(metodos[0]);
  const conDatos = METODOS_CON_DATOS.has(metodo);
  const conVista = METODOS_CON_VISTA.has(metodo);

  const esquema = useCarga(() => api.ajustes(entorno, metodo), `${entorno}:${metodo}`);
  const lista = useCarga(conDatos ? () => api.demostraciones(entorno) : null, `${entorno}:${conDatos}`);
  const [valores, setValores] = useState({});
  const [elegidos, setElegidos] = useState(new Set());
  const [vistaInicial, setVistaInicial] = useState(false);
  const [wandb, setWandb] = useState(false);
  const [intentado, setIntentado] = useState(false);

  useEffect(() => {
    setValores(esquema.datos ? valoresIniciales(esquema.datos) : {});
    setIntentado(false);
  }, [esquema.datos]);

  useEffect(() => {
    alCambiarOcupado?.(ocupado);
  }, [alCambiarOcupado, ocupado]);

  const revision = esquema.datos ? validarAjustes(esquema.datos, valores) : null;
  const avisoDatos = conDatos && elegidos.size === 0 ? 'Elige al menos un conjunto de demostraciones.' : null;

  const iniciar = () => {
    setIntentado(true);
    if (!revision?.valido || avisoDatos) {
      return;
    }
    entreno.iniciar({
      entorno, metodo, ajustes: revision.ajustes,
      ...(conDatos ? { datos: [...elegidos] } : {}),
      vista: conVista && vistaInicial, wandb,
    });
  };

  const total = entreno.entreno?.duracion.total ?? esquema.datos?.duracion.total;
  const unidad = UNIDADES[entreno.entreno?.duracion.unidad ?? esquema.datos?.duracion.unidad] ?? '';
  const porcentaje = total ? Math.min(100, Math.round((entreno.paso / total) * 100)) : 0;
  const etiqueta = { [FASES_ENTRENO.PREPARANDO]: 'Preparando…', [FASES_ENTRENO.TERMINANDO]: 'Terminando…',
    [FASES_ENTRENO.TERMINADO]: 'Terminado' }[fase]
    ?? (entreno.pendiente === 'pausar' ? 'Pausando…' : entreno.pendiente === 'reanudar' ? 'Reanudando…'
      : entreno.pausado ? 'Pausado' : 'Entrenando');

  return (
    <div className="entrenamiento">
      {metodos.length > 1 ? (
        <div className="pestanas" role="group" aria-label="Método de entrenamiento">
          {metodos.map((m) => (
            <button key={m} type="button" aria-pressed={m === metodo}
              disabled={fase !== FASES_ENTRENO.INACTIVO} onClick={() => setMetodo(m)}>{NOMBRES[m]}</button>
          ))}
        </div>
      ) : null}

      {fase === FASES_ENTRENO.INACTIVO ? (
        <form aria-label={`Entrenar ${NOMBRES[metodo]}`} noValidate onSubmit={(e) => {
          e.preventDefault();
          iniciar();
        }}>
          {conDatos ? (
            <SelectorDatos lista={lista.datos} error={lista.error} elegidos={elegidos}
              aviso={intentado ? avisoDatos : null} alActualizar={lista.recargar}
              alCambiar={(clave, elegido) => setElegidos((anteriores) => {
                const nuevos = new Set(anteriores);
                if (elegido) {
                  nuevos.add(clave);
                } else {
                  nuevos.delete(clave);
                }
                return nuevos;
              })} />
          ) : null}

          <h3 className="ajustes-titulo">Hiperparámetros de {nombre}</h3>
          {esquema.error ? (
            <div className="error" role="alert">
              No se pudieron cargar los valores del perfil ({esquema.error}).{' '}
              <button type="button" onClick={esquema.recargar}>Reintentar</button>
            </div>
          ) : null}
          {!esquema.datos && !esquema.error ? <p className="nota">Cargando los valores del perfil…</p> : null}
          {esquema.datos ? (
            <>
              <p className="nota">
                Valores del perfil de {nombre}; cambia los que quieras. Dura {formatearNumero(total)} {unidad}.
              </p>
              <FormularioAjustes key={esquema.datos.metodo} esquema={esquema.datos} valores={valores}
                errores={intentado ? revision.errores : {}}
                alCambiar={(clave, valor) => setValores((anteriores) => ({ ...anteriores, [clave]: valor }))} />
              <div className="ajustes-pie">
                <span className="apagado">
                  {revision.cambiados ? `${revision.cambiados} cambiados respecto al perfil` : 'Sin cambios respecto al perfil'}
                </span>
                <button type="button" disabled={!revision.cambiados && revision.valido}
                  onClick={() => setValores(valoresIniciales(esquema.datos))}>Restablecer valores del perfil</button>
              </div>
              {intentado && !revision.valido ? (
                <p className="error" role="alert">Hay valores que no son válidos: revisa los marcados.</p>
              ) : null}
            </>
          ) : null}

          {conVista ? (
            <label className="casilla" htmlFor={`${id}-vista`}>
              <input id={`${id}-vista`} type="checkbox" checked={vistaInicial}
                onChange={(e) => setVistaInicial(e.target.checked)} />
              Ver las partidas mientras entrena (más lento; se puede cambiar en marcha)
            </label>
          ) : null}
          <label className="casilla" htmlFor={`${id}-wandb`}>
            <input id={`${id}-wandb`} type="checkbox" checked={wandb} onChange={(e) => setWandb(e.target.checked)} />
            Registrar en W&amp;B
          </label>
          <div className="controles">
            <button type="submit" className="primario" disabled={!esquema.datos}>
              Entrenar {NOMBRES[metodo]}
            </button>
          </div>
        </form>
      ) : (
        <>
          <div className="progreso-cabecera">
            <span>{NOMBRES[entreno.entreno?.metodo ?? metodo]}</span>
            <span role="status">{etiqueta}</span>
          </div>
          <div className="barra" role="progressbar" aria-label="Progreso del entrenamiento"
            aria-valuemin={0} aria-valuemax={100} aria-valuenow={porcentaje}>
            <div style={{ width: `${porcentaje}%` }} />
          </div>
          <p className="nota">{formatearNumero(entreno.paso)} / {formatearNumero(total)} {unidad} · {porcentaje} %</p>

          {conVista && fase !== FASES_ENTRENO.TERMINADO && fase !== FASES_ENTRENO.PREPARANDO ? (
            <label className="casilla" htmlFor={`${id}-vista-viva`}>
              <input id={`${id}-vista-viva`} type="checkbox" checked={entreno.vista}
                disabled={fase !== FASES_ENTRENO.ENTRENANDO} onChange={(e) => entreno.cambiarVista(e.target.checked)} />
              Ver las partidas del entrenamiento
            </label>
          ) : null}
          {entreno.vista && fase === FASES_ENTRENO.ENTRENANDO ? (
            <VistaEntorno imagen={entreno.imagen} nombre={nombre} vacia="Esperando la primera imagen…" />
          ) : null}

          <Curvas series={entreno.series} unidad={unidad} />

          {fase === FASES_ENTRENO.TERMINADO ? (
            <>
              {entreno.resultado ? (
                <p className="aviso-exito" role="status">
                  Modelo guardado en la biblioteca: «{entreno.resultado.modelo.nombre}»
                  {entreno.resultado.modelo.incompleto ? ' (incompleto: se finalizó antes de tiempo)' : ''}.
                </p>
              ) : null}
              <div className="controles">
                <button type="button" className="primario" onClick={entreno.nuevo}>Nuevo entrenamiento</button>
              </div>
            </>
          ) : (
            <div className="controles">
              {entreno.pausado ? (
                <button type="button" onClick={entreno.reanudar}
                  disabled={fase !== FASES_ENTRENO.ENTRENANDO || Boolean(entreno.pendiente)}>Reanudar</button>
              ) : (
                <button type="button" onClick={entreno.pausar}
                  disabled={fase !== FASES_ENTRENO.ENTRENANDO || Boolean(entreno.pendiente)}>Pausar</button>
              )}
              <button type="button" className="primario" onClick={entreno.finalizar}
                disabled={fase !== FASES_ENTRENO.ENTRENANDO}>Finalizar</button>
            </div>
          )}
        </>
      )}
      {entreno.error ? <p className="error" role="alert">{entreno.error}</p> : null}
    </div>
  );
}
