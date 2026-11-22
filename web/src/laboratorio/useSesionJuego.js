import { useCallback, useEffect, useRef, useState } from 'react';

import { ESTADOS_CONEXION } from '../api/websocket.js';

/** Fases de una sesión de juego. */
export const FASES = Object.freeze({
  INACTIVA: 'inactiva',
  PREPARANDO: 'preparando',
  JUGANDO: 'jugando',
  TERMINANDO: 'terminando',
  REVISION: 'revision',
  GUARDANDO: 'guardando',
  GUARDADA: 'guardada',
});

/** Fases en las que hay una sesión abierta y no se debe cambiar de entorno. */
export const FASES_OCUPADAS = Object.freeze(new Set([FASES.PREPARANDO, FASES.JUGANDO,
  FASES.TERMINANDO, FASES.REVISION, FASES.GUARDANDO]));

/**
 * Milisegundos que se espera sesion_terminada tras pedir finalizar. Hace falta además de
 * vigilar el cierre: un servidor colgado o una red cortada no siempre cierran el WebSocket.
 */
export const ESPERA_FIN_MS = 10_000;

const CONEXION_PERDIDA = 'Se perdió la conexión con el servidor. Las partidas de esta sesión no se han guardado.';
const FIN_SIN_RESPUESTA = 'El servidor no respondió al finalizar. Las partidas de esta sesión no se han guardado.';

const ESTADO_INICIAL = { partida: 1, paso: 0, retorno: 0, pares: 0, esperando: false };

const mensajeDe = (error) => (error instanceof Error ? error.message : String(error));

/**
 * Sesión de «Jugar tú» o «Ver agente»: la crea, se conecta, recibe imágenes y progreso,
 * la termina y guarda o descarta sus partidas.
 *
 * Si la conexión se cierra antes de sesion_terminada, o esta no llega en ESPERA_FIN_MS
 * tras finalizar, la sesión vuelve a INACTIVA con un error.
 *
 * @param {import('../api/laboratorio.js').ApiLaboratorio} api
 * @returns {object} Estado de la sesión (fase, sesion, estado, imagen, partidas,
 *   terminadas, error, guardado) y acciones (iniciar, enviarAccion, finalizar, guardar
 *   y descartar).
 */
export default function useSesionJuego(api) {
  const [fase, setFase] = useState(FASES.INACTIVA);
  const [sesion, setSesion] = useState(null);
  const [estado, setEstado] = useState(ESTADO_INICIAL);
  const [imagen, setImagen] = useState(null);
  const [terminadas, setTerminadas] = useState([]);
  const [partidas, setPartidas] = useState([]);
  const [error, setError] = useState(null);
  const [guardado, setGuardado] = useState(null);
  const conexion = useRef(null);
  const esperaFin = useRef(null);

  const cerrarConexion = useCallback(() => {
    clearTimeout(esperaFin.current);
    conexion.current?.cerrar();
    conexion.current = null;
  }, []);

  const abandonar = useCallback((motivo) => {
    cerrarConexion();
    setFase(FASES.INACTIVA);
    setError(motivo);
  }, [cerrarConexion]);

  useEffect(() => cerrarConexion, [cerrarConexion]);

  const iniciar = useCallback(async ({ entorno, modo, partidas: pedidas }) => {
    cerrarConexion();
    setFase(FASES.PREPARANDO);
    setError(null);
    setGuardado(null);
    setImagen(null);
    setEstado(ESTADO_INICIAL);
    setTerminadas([]);
    setPartidas([]);
    try {
      const nueva = await api.crearSesion({ entorno, modo, ...(pedidas ? { partidas: pedidas } : {}) });
      setSesion(nueva);
      conexion.current = api.conectar(nueva.id, {
        alEstado: setEstado,
        alImagen: setImagen,
        alTerminarPartida: (partida) => setTerminadas((anteriores) => [...anteriores, partida]),
        alTerminarSesion: (todas) => {
          cerrarConexion();
          setPartidas(todas);
          setFase(todas.length ? FASES.REVISION : FASES.INACTIVA);
          if (!todas.length) {
            setError('La sesión terminó sin ninguna partida completa: no hay nada que guardar.');
          }
        },
        alError: (e) => setError(mensajeDe(e)),
        // Solo hay conexión en JUGANDO y TERMINANDO: al recibir sesion_terminada se cierra
        // desde aquí, y cerrar() ya no avisa.
        alCambiarEstado: (nuevoEstado) => {
          if (nuevoEstado === ESTADOS_CONEXION.CERRADA) {
            abandonar(CONEXION_PERDIDA);
          }
        },
      });
      setFase(FASES.JUGANDO);
    } catch (e) {
      setFase(FASES.INACTIVA);
      setError(mensajeDe(e));
    }
  }, [abandonar, api, cerrarConexion]);

  const enviarAccion = useCallback((accion) => conexion.current?.enviarAccion(accion) ?? false, []);

  const finalizar = useCallback(() => {
    if (!conexion.current) {
      return;
    }
    if (!conexion.current.finalizar()) {
      abandonar(CONEXION_PERDIDA);
      return;
    }
    setFase(FASES.TERMINANDO);
    clearTimeout(esperaFin.current);
    esperaFin.current = setTimeout(() => abandonar(FIN_SIN_RESPUESTA), ESPERA_FIN_MS);
  }, [abandonar]);

  const guardar = useCallback(async (nombre, seleccion) => {
    setFase(FASES.GUARDANDO);
    setError(null);
    try {
      setGuardado(await api.guardar(sesion.id, { nombre: nombre.trim(), partidas: seleccion }));
      setFase(FASES.GUARDADA);
    } catch (e) {
      setFase(FASES.REVISION);
      setError(mensajeDe(e));
    }
  }, [api, sesion]);

  const descartar = useCallback(async () => {
    setError(null);
    try {
      await api.descartar(sesion.id);
      setFase(FASES.INACTIVA);
      setPartidas([]);
    } catch (e) {
      setError(mensajeDe(e));
    }
  }, [api, sesion]);

  return { fase, sesion, estado, imagen, terminadas, partidas, error, guardado,
    iniciar, enviarAccion, finalizar, guardar, descartar };
}
