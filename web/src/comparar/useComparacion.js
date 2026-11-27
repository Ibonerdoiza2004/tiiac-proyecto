import { useCallback, useEffect, useRef, useState } from 'react';

import { CIERRE_NO_EXISTE, ESTADOS_FINALES, ESTADOS_TRABAJO } from '../api/trabajos.js';
import { ESTADOS_CONEXION } from '../api/websocket.js';
import { acumular, acumularTodo, desdeResultado } from './estadisticas.js';

/** Fases de la comparación en la interfaz. */
export const FASES_COMPARACION = Object.freeze({
  INACTIVA: 'inactiva',
  PREPARANDO: 'preparando',
  SIGUIENDO: 'siguiendo',
  TERMINANDO: 'terminando',
  TERMINADA: 'terminada',
  DESCONECTADA: 'desconectada',
  INEXISTENTE: 'inexistente',
});

/** Milisegundos que se espera la respuesta al crear la comparación. */
export const ESPERA_CREAR_MS = 15_000;
/** Milisegundos que se espera el estado final tras pedir finalizar. */
export const ESPERA_FIN_MS = 30_000;

const CONEXION_PERDIDA = 'Se perdió la conexión con el servidor. La comparación puede seguir allí: reconecta para verla.';
const FIN_SIN_RESPUESTA = 'El servidor no confirmó el final de la comparación. Reconecta para ver cómo va.';
const CREAR_SIN_RESPUESTA = 'El servidor no respondió al crear la comparación.';
const NO_EXISTE = 'Esta comparación ya no existe en el servidor: no las conserva al reiniciarse.';
const AUN_CONECTANDO = 'Todavía se está conectando con el servidor: vuelve a intentarlo en un momento.';
const mensajeDe = (error) => (error instanceof Error ? error.message : String(error));

/**
 * Una comparación: la crea o sigue una ya creada (también una terminada, para ver su
 * resultado), acumula sus retornos y la pausa, reanuda, finaliza o cambia la vista.
 *
 * El protocolo de trabajos permite reconectarse: el historial que llega al conectar trae
 * todo lo anterior y se rehace la vista desde él. Si la conexión se cierra antes del
 * estado final, o este no llega en ESPERA_FIN_MS tras finalizar, la fase pasa a
 * DESCONECTADA con un error (el del servidor, si avisó antes de cerrar) y se puede
 * reconectar. Si el servidor ya no tiene el trabajo (cierre 4404, por ejemplo tras
 * reiniciarse), pasa a INEXISTENTE, sin reconectar.
 *
 * @param {import('../api/comparar.js').ApiComparar} api
 * @returns {object} Estado (fase, trabajo, acumulado, imagen, vista, error) y acciones
 *   (iniciar, ver, reconectar, pausar, reanudar, cancelar, finalizar, cambiarVista,
 *   nueva).
 */
export default function useComparacion(api) {
  const [fase, setFase] = useState(FASES_COMPARACION.INACTIVA);
  const [trabajo, setTrabajo] = useState(null);
  const [acumulado, setAcumulado] = useState({});
  const [imagen, setImagen] = useState(null);
  const [vista, setVista] = useState(false);
  const [error, setError] = useState(null);
  const conexion = useRef(null);
  const ultimo = useRef(null);
  const esperaFin = useRef(null);
  const vistaActiva = useRef(false);
  const montado = useRef(true);

  const cerrarConexion = useCallback(() => {
    clearTimeout(esperaFin.current);
    conexion.current?.cerrar();
    conexion.current = null;
  }, []);

  const desconectar = useCallback((motivo, fase = FASES_COMPARACION.DESCONECTADA) => {
    cerrarConexion();
    setFase(fase);
    setError(motivo);
  }, [cerrarConexion]);

  useEffect(() => {
    montado.current = true;
    return () => {
      montado.current = false;
      cerrarConexion();
    };
  }, [cerrarConexion]);

  const seguir = useCallback((id) => {
    cerrarConexion();
    setError(null);
    setImagen(null);
    setFase(FASES_COMPARACION.SIGUIENDO);
    ultimo.current = null;
    // El cliente avisa de «conectando» dentro de esta llamada, cuando propia aún es null:
    // un cierre solo cuenta si es de la conexión actual.
    let propia = null;
    propia = api.seguir(id, {
      alEstado: (nuevo) => {
        ultimo.current = nuevo;
        setTrabajo(nuevo);
        if (ESTADOS_FINALES.has(nuevo.estado)) {
          clearTimeout(esperaFin.current);
          setFase(FASES_COMPARACION.TERMINADA);
          // El resultado guardado manda: trae todos los retornos aunque falte progreso.
          const final = desdeResultado(nuevo.resultado);
          if (final) {
            setAcumulado(final);
          }
          if (nuevo.estado === ESTADOS_TRABAJO.CANCELADO) {
            setError('Comparación cancelada.');
          } else if (nuevo.estado !== ESTADOS_TRABAJO.TERMINADO) {
            setError(nuevo.error ?? 'La comparación falló.');
          }
        } else {
          setError(null);  // un aviso anterior del servidor ya no viene al caso
        }
      },
      // Al conectar llega el estado antes que el historial: si ya era final, su resultado
      // manda sobre el progreso.
      alHistorial: ({ progreso }) => setAcumulado(
        (ESTADOS_FINALES.has(ultimo.current?.estado) && desdeResultado(ultimo.current.resultado))
          || acumularTodo(progreso)),
      alProgreso: (medidas) => setAcumulado((anterior) => acumular(anterior, medidas)),
      alImagen: (nueva) => {
        if (vistaActiva.current) {
          setImagen(nueva);
        }
      },
      alError: (e) => setError(mensajeDe(e)),
      // El servidor cierra tras el estado final; cerrar antes es perder la conexión.
      alCambiarEstado: (estado, cierre) => {
        if (estado === ESTADOS_CONEXION.CERRADA && conexion.current === propia
            && !ESTADOS_FINALES.has(ultimo.current?.estado)) {
          if (cierre?.codigo === CIERRE_NO_EXISTE) {
            desconectar(NO_EXISTE, FASES_COMPARACION.INEXISTENTE);
          } else {
            desconectar(cierre?.error ?? CONEXION_PERDIDA);
          }
        } else if (estado === ESTADOS_CONEXION.CERRADA && conexion.current === propia) {
          conexion.current = null;
        }
      },
    });
    conexion.current = propia;
  }, [api, cerrarConexion, desconectar]);

  const iniciar = useCallback(async (peticion) => {
    cerrarConexion();
    setFase(FASES_COMPARACION.PREPARANDO);
    setError(null);
    setTrabajo(null);
    setAcumulado({});
    setImagen(null);
    vistaActiva.current = Boolean(peticion.vista);
    setVista(vistaActiva.current);
    const control = new AbortController();
    const espera = setTimeout(() => control.abort(), ESPERA_CREAR_MS);
    let nuevo;
    try {
      nuevo = await api.crear(peticion, { senal: control.signal });
    } catch (e) {
      if (montado.current) {
        setFase(FASES_COMPARACION.INACTIVA);
        setError(control.signal.aborted ? CREAR_SIN_RESPUESTA : mensajeDe(e));
      }
      return;
    } finally {
      clearTimeout(espera);
    }
    if (!montado.current) {
      return;
    }
    setTrabajo(nuevo);
    seguir(nuevo.id);
  }, [api, cerrarConexion, seguir]);

  const ver = useCallback((anterior) => {
    // Lo último que se sabe de la vista es lo que se pidió al crearla.
    vistaActiva.current = Boolean(anterior.argumentos?.vista);
    setVista(vistaActiva.current);
    setAcumulado({});
    setTrabajo(anterior);
    seguir(anterior.id);
  }, [seguir]);

  const ordenar = useCallback((enviar, alEnviar) => {
    if (!conexion.current) {
      return;
    }
    // El aviso anterior se quita antes: la respuesta puede llegar dentro del envío.
    setError(null);
    if (!enviar(conexion.current)) {
      // Aún conectando se puede reintentar; si no, la conexión se ha perdido.
      if (conexion.current.conectando?.()) {
        setError(AUN_CONECTANDO);
      } else {
        desconectar(CONEXION_PERDIDA);
      }
      return;
    }
    alEnviar?.();
  }, [desconectar]);

  const pausar = useCallback(() => ordenar((c) => c.pausar()), [ordenar]);
  const reanudar = useCallback(() => ordenar((c) => c.reanudar()), [ordenar]);
  const cancelar = useCallback(() => ordenar((c) => c.cancelar()), [ordenar]);
  const cambiarVista = useCallback((activa) => ordenar((c) => c.ordenar({ tipo: 'vista', activa }), () => {
    vistaActiva.current = activa;
    setVista(activa);
    if (!activa) {
      setImagen(null);
    }
  }), [ordenar]);
  const finalizar = useCallback(() => ordenar((c) => c.ordenar({ tipo: 'finalizar' }), () => {
    // Si el estado final llegó ya durante el envío, no hay nada que esperar.
    if (ESTADOS_FINALES.has(ultimo.current?.estado)) {
      return;
    }
    setFase(FASES_COMPARACION.TERMINANDO);
    clearTimeout(esperaFin.current);
    esperaFin.current = setTimeout(() => desconectar(FIN_SIN_RESPUESTA), ESPERA_FIN_MS);
  }), [desconectar, ordenar]);

  const reconectar = useCallback(() => {
    if (trabajo) {
      seguir(trabajo.id);
    }
  }, [seguir, trabajo]);

  const nueva = useCallback(() => {
    cerrarConexion();
    setFase(FASES_COMPARACION.INACTIVA);
    setTrabajo(null);
    setAcumulado({});
    setImagen(null);
    setError(null);
  }, [cerrarConexion]);

  return { fase, trabajo, acumulado, imagen, vista, error,
    iniciar, ver, reconectar, pausar, reanudar, cancelar, finalizar, cambiarVista, nueva };
}
