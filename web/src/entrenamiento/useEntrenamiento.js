import { useCallback, useEffect, useRef, useState } from 'react';

import { ESTADOS_ENTRENAMIENTO } from '../api/entrenamiento.js';
import { ESTADOS_CONEXION } from '../api/websocket.js';
import { agregarProgreso } from './curvas.js';

/** Fases de un entrenamiento en la interfaz. */
export const FASES_ENTRENO = Object.freeze({
  INACTIVO: 'inactivo',
  PREPARANDO: 'preparando',
  ENTRENANDO: 'entrenando',
  TERMINANDO: 'terminando',
  TERMINADO: 'terminado',
});

/** Fases con un entrenamiento abierto, en las que no se debe cambiar de entorno. */
export const FASES_ENTRENO_OCUPADAS = Object.freeze(new Set([FASES_ENTRENO.PREPARANDO,
  FASES_ENTRENO.ENTRENANDO, FASES_ENTRENO.TERMINANDO]));

/** Milisegundos que se espera la respuesta al crear el entrenamiento. */
export const ESPERA_CREAR_MS = 15_000;

/**
 * Milisegundos que se espera el mensaje terminado tras finalizar: el servidor tiene que
 * parar y guardar el modelo, así que se le da más margen que a una sesión de juego.
 */
export const ESPERA_FIN_ENTRENO_MS = 60_000;

const CONEXION_PERDIDA = 'Se perdió la conexión con el servidor. El entrenamiento puede seguir allí, pero esta página ya no lo sigue.';
const FIN_SIN_RESPUESTA = 'El servidor no confirmó el final del entrenamiento.';
const CREAR_SIN_RESPUESTA = 'El servidor no respondió al crear el entrenamiento.';

const mensajeDe = (error) => (error instanceof Error ? error.message : String(error));

/**
 * Un entrenamiento de experto, BC o GAIL: lo crea, recibe el progreso, las imágenes y el
 * estado, y lo pausa, reanuda, finaliza o cambia la vista.
 *
 * Si la conexión se cierra antes del mensaje terminado, si no llega en
 * ESPERA_FIN_ENTRENO_MS tras finalizar o si crear no responde en ESPERA_CREAR_MS, la fase
 * deja de estar ocupada y queda un error: la interfaz nunca se queda bloqueada.
 *
 * @param {import('../api/entrenamiento.js').ApiEntrenamiento} api
 * @returns {object} Estado (fase, entreno, estado, pendiente, paso, series, imagen, vista,
 *   resultado, error) y acciones (iniciar, pausar, reanudar, cambiarVista, finalizar, nuevo).
 */
export default function useEntrenamiento(api) {
  const [fase, setFase] = useState(FASES_ENTRENO.INACTIVO);
  const [entreno, setEntreno] = useState(null);
  const [estado, setEstado] = useState(null);
  const [pendiente, setPendiente] = useState(null);
  const [paso, setPaso] = useState(0);
  const [series, setSeries] = useState({});
  const [imagen, setImagen] = useState(null);
  const [vista, setVista] = useState(false);
  const [resultado, setResultado] = useState(null);
  const [error, setError] = useState(null);
  const conexion = useRef(null);
  const esperaFin = useRef(null);
  const vistaActiva = useRef(false);
  const montado = useRef(true);

  const cerrarConexion = useCallback(() => {
    clearTimeout(esperaFin.current);
    conexion.current?.cerrar();
    conexion.current = null;
  }, []);

  const abandonar = useCallback((motivo) => {
    cerrarConexion();
    setPendiente(null);
    setFase(FASES_ENTRENO.TERMINADO);
    setError(motivo);
  }, [cerrarConexion]);

  useEffect(() => {
    montado.current = true;
    return () => {
      montado.current = false;
      cerrarConexion();
    };
  }, [cerrarConexion]);

  const iniciar = useCallback(async (peticion) => {
    cerrarConexion();
    setFase(FASES_ENTRENO.PREPARANDO);
    setError(null);
    setResultado(null);
    setSeries({});
    setPaso(0);
    setImagen(null);
    setEstado(null);
    setPendiente(null);
    const control = new AbortController();
    const espera = setTimeout(() => control.abort(), ESPERA_CREAR_MS);
    let nuevo;
    try {
      nuevo = await api.crear(peticion, { senal: control.signal });
    } catch (e) {
      if (montado.current) {
        setFase(FASES_ENTRENO.INACTIVO);
        setError(control.signal.aborted ? CREAR_SIN_RESPUESTA : mensajeDe(e));
      }
      return;
    } finally {
      clearTimeout(espera);
    }
    if (!montado.current) {
      return;
    }
    vistaActiva.current = Boolean(peticion.vista);
    setVista(vistaActiva.current);
    setEntreno(nuevo);
    setFase(FASES_ENTRENO.ENTRENANDO);
    conexion.current = api.conectar(nuevo.id, {
      alProgreso: (progreso) => {
        setPaso(progreso.paso);
        setSeries((anteriores) => agregarProgreso(anteriores, progreso));
      },
      alImagen: (nueva) => {
        if (vistaActiva.current) {
          setImagen(nueva);
        }
      },
      alEstado: (nuevoEstado) => {
        setEstado(nuevoEstado);
        setPendiente(null);
      },
      alTerminar: (fin) => {
        cerrarConexion();
        setPendiente(null);
        setResultado(fin);
        setFase(FASES_ENTRENO.TERMINADO);
      },
      alError: (e) => setError(mensajeDe(e)),
      // Solo hay conexión hasta el mensaje terminado: después se cierra desde aquí, y
      // cerrar() ya no avisa.
      alCambiarEstado: (estadoConexion) => {
        if (estadoConexion === ESTADOS_CONEXION.CERRADA) {
          abandonar(CONEXION_PERDIDA);
        }
      },
    });
  }, [abandonar, api, cerrarConexion]);

  const ordenar = useCallback((enviar, alEnviar) => {
    if (!conexion.current) {
      return;
    }
    if (!enviar(conexion.current)) {
      abandonar(CONEXION_PERDIDA);
      return;
    }
    alEnviar?.();
  }, [abandonar]);

  const pausar = useCallback(() => ordenar((c) => c.pausar(), () => setPendiente('pausar')), [ordenar]);
  const reanudar = useCallback(() => ordenar((c) => c.reanudar(), () => setPendiente('reanudar')),
    [ordenar]);

  const cambiarVista = useCallback((activa) => ordenar((c) => c.cambiarVista(activa), () => {
    vistaActiva.current = activa;
    setVista(activa);
    if (!activa) {
      setImagen(null);
    }
  }), [ordenar]);

  const finalizar = useCallback(() => ordenar((c) => c.finalizar(), () => {
    setFase(FASES_ENTRENO.TERMINANDO);
    clearTimeout(esperaFin.current);
    esperaFin.current = setTimeout(() => abandonar(FIN_SIN_RESPUESTA), ESPERA_FIN_ENTRENO_MS);
  }), [abandonar, ordenar]);

  const nuevo = useCallback(() => {
    setFase(FASES_ENTRENO.INACTIVO);
    setError(null);
    setResultado(null);
    setSeries({});
    setImagen(null);
    setPaso(0);
  }, []);

  return { fase, entreno, estado, pausado: estado === ESTADOS_ENTRENAMIENTO.PAUSADO, pendiente,
    paso, series, imagen, vista, resultado, error,
    iniciar, pausar, reanudar, cambiarVista, finalizar, nuevo };
}
