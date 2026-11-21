import { useEffect, useState } from 'react';

import { MODOS } from '../api/laboratorio.js';
import { buscarEntorno } from '../entornos.js';
import { contar } from './revision.js';
import RevisarPartidas from './RevisarPartidas.jsx';
import { accionDeTeclas, accionesBotones, ayudaTeclado, esTeclaDeJuego,
  normalizarTecla } from './teclado.js';
import useSesionJuego, { FASES, FASES_OCUPADAS } from './useSesionJuego.js';
import VistaEntorno from './VistaEntorno.jsx';

const MAXIMO_PARTIDAS = 1000;
const ETIQUETAS = {
  [FASES.INACTIVA]: 'Preparado', [FASES.PREPARANDO]: 'Preparando…', [FASES.TERMINANDO]: 'Terminando…',
  [FASES.REVISION]: 'Terminada', [FASES.GUARDANDO]: 'Guardando…', [FASES.GUARDADA]: 'Guardada',
};
const EDITABLES = new Set(['INPUT', 'SELECT', 'TEXTAREA']);
const decimales = (valor) => valor.toLocaleString('es-ES', { maximumFractionDigits: 1 });

function partidasValidas(texto) {
  const valor = /^\d+$/.test(String(texto).trim()) ? Number(texto) : NaN;
  return Number.isInteger(valor) && valor >= 1 && valor <= MAXIMO_PARTIDAS ? valor : null;
}

/**
 * Panel «Explorar y generar» del Laboratorio: jugar a mano (CartPole y MountainCar) o ver
 * al experto, contar partidas y pares, y revisar y guardar las partidas en la biblioteca.
 *
 * @param {object} props
 * @param {string} props.entorno Clave del entorno elegido.
 * @param {import('../api/laboratorio.js').ApiLaboratorio} props.api Backend real o simulado.
 * @param {(ocupado: boolean) => void} [props.alCambiarOcupado] Avisa de si hay una sesión
 *   abierta, para no cambiar de entorno a mitad.
 * @returns {JSX.Element}
 */
export default function ExplorarGenerar({ entorno, api, alCambiarOcupado }) {
  const { nombre, jugarAMano } = buscarEntorno(entorno);
  const juego = useSesionJuego(api);
  const { fase, estado, sesion } = juego;
  const [modo, setModo] = useState(jugarAMano ? MODOS.MANUAL : MODOS.EXPERTO);
  const [partidas, setPartidas] = useState('');
  const [errorPartidas, setErrorPartidas] = useState(null);
  const ocupado = FASES_OCUPADAS.has(fase);
  const manual = modo === MODOS.MANUAL;
  const jugando = fase === FASES.JUGANDO;

  useEffect(() => {
    setModo(jugarAMano ? MODOS.MANUAL : MODOS.EXPERTO);
    setErrorPartidas(null);
    let vigente = true;
    api.configuracion(entorno).then((configuracion) => {
      if (vigente) {
        setPartidas(String(configuracion.partidas_experto));
      }
    }).catch(() => {});
    return () => {
      vigente = false;
    };
  }, [api, entorno, jugarAMano]);

  useEffect(() => {
    alCambiarOcupado?.(ocupado);
  }, [alCambiarOcupado, ocupado]);

  const { enviarAccion } = juego;
  useEffect(() => {
    if (!jugando || sesion?.modo !== MODOS.MANUAL) {
      return undefined;
    }
    const pulsadas = [];
    const actualizar = () => enviarAccion(accionDeTeclas(entorno, pulsadas));
    const abajo = (evento) => {
      if (EDITABLES.has(evento.target?.tagName) || !esTeclaDeJuego(entorno, evento.key)) {
        return;
      }
      evento.preventDefault();
      const tecla = normalizarTecla(evento.key);
      if (!pulsadas.includes(tecla)) {
        pulsadas.push(tecla);
        actualizar();
      }
    };
    const arriba = (evento) => {
      const posicion = pulsadas.indexOf(normalizarTecla(evento.key));
      if (posicion >= 0) {
        pulsadas.splice(posicion, 1);
        actualizar();
      }
    };
    const soltarTodo = () => {
      if (pulsadas.length) {
        pulsadas.length = 0;
        actualizar();
      }
    };
    window.addEventListener('keydown', abajo);
    window.addEventListener('keyup', arriba);
    window.addEventListener('blur', soltarTodo);
    return () => {
      window.removeEventListener('keydown', abajo);
      window.removeEventListener('keyup', arriba);
      window.removeEventListener('blur', soltarTodo);
    };
  }, [entorno, enviarAccion, jugando, sesion]);

  const iniciar = () => {
    if (manual) {
      juego.iniciar({ entorno, modo });
      return;
    }
    const cantidad = partidasValidas(partidas);
    if (cantidad === null) {
      setErrorPartidas(`Escribe un número entero de partidas entre 1 y ${MAXIMO_PARTIDAS}.`);
      return;
    }
    setErrorPartidas(null);
    juego.iniciar({ entorno, modo, partidas: cantidad });
  };

  const botones = accionesBotones(entorno);
  const etiqueta = jugando ? (estado.esperando ? 'Esperando tecla' : 'En marcha') : ETIQUETAS[fase];
  const aviso = jugando && sesion?.modo === MODOS.MANUAL ? ayudaTeclado(entorno) : undefined;
  const objetivo = sesion?.partidas;

  return (
    <article className="panel" aria-labelledby="panel-explorar">
      <h2 id="panel-explorar">Explorar y generar</h2>
      <div className="pestanas" role="group" aria-label="Modo de interacción">
        <button type="button" aria-pressed={manual} disabled={!jugarAMano || ocupado}
          onClick={() => setModo(MODOS.MANUAL)}>Jugar tú</button>
        <button type="button" aria-pressed={!manual} disabled={ocupado}
          onClick={() => setModo(MODOS.EXPERTO)}>Ver agente</button>
      </div>
      {jugarAMano ? null : (
        <p className="nota">En {nombre} solo se puede ver al experto y grabar partidas con él.</p>
      )}
      {manual ? (
        <p className="nota">Juega hasta que quieras parar. La partida en curso no se guarda si no ha terminado.</p>
      ) : (
        <>
          <label className="campo campo-corto">Partidas del experto
            <input type="number" inputMode="numeric" min="1" max={MAXIMO_PARTIDAS} value={partidas}
              disabled={ocupado} aria-invalid={Boolean(errorPartidas)}
              aria-describedby={errorPartidas ? 'error-partidas' : undefined}
              onChange={(e) => {
                setPartidas(e.target.value);
                setErrorPartidas(null);
              }} />
          </label>
          {errorPartidas ? <p className="error" id="error-partidas">{errorPartidas}</p> : null}
          <p className="nota">El experto juega con su política estocástica, una semilla por partida.</p>
        </>
      )}

      {fase === FASES.REVISION || fase === FASES.GUARDANDO ? (
        <RevisarPartidas partidas={juego.partidas} guardando={fase === FASES.GUARDANDO}
          alGuardar={juego.guardar} alDescartar={juego.descartar} />
      ) : (
        <>
          <VistaEntorno imagen={juego.imagen} nombre={nombre} estado={etiqueta} aviso={aviso} />
          <dl className="contadores">
            <div><dt>Partida</dt><dd>{estado.partida}{objetivo ? ` / ${objetivo}` : ''}</dd></div>
            <div><dt>Puntuación</dt><dd>{decimales(estado.retorno)}</dd></div>
            <div><dt>Pares recogidos</dt><dd>{estado.pares}</dd></div>
            <div><dt>Partidas terminadas</dt><dd>{juego.terminadas.length}</dd></div>
          </dl>
          <div className="controles">
            {ocupado ? (
              <button type="button" className="primario" onClick={juego.finalizar} disabled={!jugando}>
                Finalizar
              </button>
            ) : (
              <button type="button" className="primario" onClick={iniciar}>
                {manual ? 'Iniciar sesión' : 'Ver al experto'}
              </button>
            )}
            {jugando && sesion?.modo === MODOS.MANUAL && botones ? (
              <>
                {[['izquierda', '←', 'Izquierda'], ['derecha', '→', 'Derecha']].map(([lado, flecha, texto]) => (
                  <button key={lado} type="button" aria-label={texto}
                    onPointerDown={() => enviarAccion(botones[lado])}
                    onPointerUp={() => enviarAccion(null)} onPointerLeave={() => enviarAccion(null)}
                    onPointerCancel={() => enviarAccion(null)}>{flecha}</button>
                ))}
              </>
            ) : null}
          </div>
        </>
      )}

      {fase === FASES.GUARDADA && juego.guardado ? (
        <p className="aviso-exito" role="status">
          Guardado «{juego.guardado.nombre}»: {contar(juego.guardado.partidas, 'partida', 'partidas')} y {contar(juego.guardado.pares, 'par', 'pares')}.
        </p>
      ) : null}
      {juego.error ? <p className="error" role="alert">{juego.error}</p> : null}
    </article>
  );
}
