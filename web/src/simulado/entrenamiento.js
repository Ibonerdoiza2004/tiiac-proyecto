/**
 * Backend simulado de entrenamiento: implementa la misma ApiEntrenamiento que
 * src/api/entrenamiento.js, en el navegador, mientras no existe laboratorio-entrenar-api.
 *
 * No entrena nada: las curvas son inventadas (se acercan al objetivo con ruido, a unas
 * 40 actualizaciones por entrenamiento) para poder probar la interfaz. Los hiperparámetros
 * sí son los de configs/ (perfiles.json) y se validan como lo hará el backend real.
 */

import { ErrorApi } from '../api/http.js';
import { ESTADOS_ENTRENAMIENTO, METODOS, METODOS_CON_DATOS, METODOS_CON_VISTA }
  from '../api/entrenamiento.js';
import { camposPorRuta, hojasDeAjustes, leerCampo } from '../entrenamiento/ajustes.js';
import { crearAlmacen } from './almacen.js';
import { crearAleatorio, crearSimulacion } from './entornos.js';
import PERFILES from './perfiles.json';

/** Milisegundos entre dos actualizaciones simuladas. */
export const INTERVALO_SIMULADO_MS = 200;
/** Actualizaciones que dura, más o menos, un entrenamiento simulado completo. */
export const ACTUALIZACIONES = 40;

function rechazar(estado, mensaje) {
  return Promise.reject(new ErrorApi(mensaje, { estado, detalle: { detail: mensaje } }));
}

const copia = (valor) => JSON.parse(JSON.stringify(valor));

/** Comprueba los ajustes contra el esquema y devuelve los valores efectivos por ruta. */
function efectivos(esquema, ajustes) {
  const campos = camposPorRuta(esquema);
  const valores = new Map([...campos].map(([ruta, campo]) => [ruta, campo.valor]));
  for (const [ruta, valor] of hojasDeAjustes(ajustes)) {
    const campo = campos.get(ruta);
    if (!campo) {
      throw new Error(`Ajuste desconocido para este método: ${ruta}`);
    }
    // El valor llega ya convertido: se comprueba con el mismo criterio que el formulario.
    const entrada = campo.tipo === 'booleano' ? valor : Array.isArray(valor) ? valor.join(',') : valor;
    const leido = typeof valor === 'string' && campo.tipo !== 'opcion'
      ? { error: 'debe ser un número, no un texto' } : leerCampo(campo, entrada);
    if ('error' in leido) {
      throw new Error(`Ajuste ${ruta} no válido: ${leido.error}`);
    }
    valores.set(ruta, leido.valor);
  }
  return valores;
}

/** Curvas inventadas: se acercan a un valor final con ruido, según la fracción hecha. */
function medidasSimuladas(metodo, fraccion, valores, aleatorio, paso, anterior) {
  const ruido = (escala) => (aleatorio() - 0.5) * escala;
  const acercar = (inicio, fin, rapidez) => inicio + (fin - inicio) * (1 - Math.exp(-rapidez * fraccion));
  if (metodo === METODOS.EXPERTO) {
    const objetivo = valores.get('experto.retorno_objetivo');
    const inicio = objetivo > 0 ? objetivo * 0.04 : objetivo * 3;
    // Más allá del objetivo, también si es negativo (MountainCar, Acrobot, Reacher), para que
    // lo alcance hacia el 60 % del presupuesto, como el experto congelado de CartPole.
    const final = objetivo + Math.abs(objetivo) * 0.1;
    return { retorno_medio: acercar(inicio, final, 4) + ruido(Math.abs(objetivo) * 0.05) };
  }
  if (metodo === METODOS.BC) {
    return {
      perdida_train: acercar(0.7, 0.05, 5) + ruido(0.02),
      perdida_validacion: acercar(0.75, 0.12, 4) + ruido(0.03),
    };
  }
  const medidas = {
    recompensa_gail_media: acercar(0.2, 0.69, 3) + ruido(0.05),
    discriminador_aciertos: acercar(0.95, 0.55, 3) + ruido(0.04),
  };
  // Se valida cada evaluar_cada iteraciones y en la última, aunque una actualización salte varias.
  const cada = valores.get('gail.evaluar_cada');
  if (Math.floor(paso / cada) > Math.floor(anterior / cada) || fraccion >= 1) {
    medidas.retorno_validacion = acercar(20, 480, 3) + ruido(30);
  }
  return medidas;
}

/**
 * Crea la API de entrenamiento simulada.
 *
 * @param {object} [opciones]
 * @param {import('./almacen.js').Almacen} [opciones.almacen] Biblioteca simulada, la misma
 *   que la de crearApiLaboratorioSimulada para ver lo que se guarda al jugar.
 * @param {(entorno: string, foto: object) => Promise<Blob>} [opciones.dibujar] Dibuja la
 *   vista (por defecto, dibujarFotograma de ./dibujo.js, que necesita un navegador).
 * @param {number} [opciones.semilla] Semilla de las curvas inventadas.
 * @returns {import('../api/entrenamiento.js').ApiEntrenamiento}
 */
export function crearApiEntrenamientoSimulada({ almacen = crearAlmacen(), dibujar, semilla = 5000 } = {}) {
  const entrenamientos = new Map();
  const pintar = dibujar
    ?? ((entorno, foto) => import('./dibujo.js').then((m) => m.dibujarFotograma(entorno, foto)));

  const esquema = (entorno, metodo) => PERFILES[entorno]?.[metodo];

  return {
    ajustes(entorno, metodo) {
      if (!PERFILES[entorno]) {
        return rechazar(404, `Entorno desconocido: ${entorno}`);
      }
      return esquema(entorno, metodo) ? Promise.resolve(copia(esquema(entorno, metodo)))
        : rechazar(422, `Método desconocido: ${metodo}`);
    },

    demostraciones(entorno) {
      if (!PERFILES[entorno]) {
        return rechazar(404, `Entorno desconocido: ${entorno}`);
      }
      // Solo los campos de DemostracionApi, como el backend real.
      return Promise.resolve(copia(almacen.demostraciones
        .filter((d) => d.entorno === entorno)
        .map(({ id, nombre, tipo, partidas, pares, fuente }) => ({ id, nombre, tipo, partidas, pares, fuente }))));
    },

    crear({ entorno, metodo, ajustes = {}, datos = [], vista = false, wandb = false } = {}) {
      if (!PERFILES[entorno]) {
        return rechazar(422, `Entorno desconocido: ${entorno}`);
      }
      if (!esquema(entorno, metodo)) {
        return rechazar(422, `Método desconocido: ${metodo}`);
      }
      if (typeof vista !== 'boolean' || typeof wandb !== 'boolean') {
        return rechazar(422, 'vista y wandb deben ser booleanos');
      }
      if (vista && !METODOS_CON_VISTA.has(metodo)) {
        return rechazar(422, 'BC no juega partidas mientras entrena: no tiene vista');
      }
      if (!Array.isArray(datos) || new Set(datos).size !== datos.length) {
        return rechazar(422, 'datos debe ser una lista de identificadores sin repetir');
      }
      if (METODOS_CON_DATOS.has(metodo) && datos.length === 0) {
        return rechazar(422, 'Elige al menos un conjunto de demostraciones');
      }
      if (!METODOS_CON_DATOS.has(metodo) && datos.length > 0) {
        return rechazar(422, 'El experto no usa demostraciones');
      }
      const ajenos = datos.filter((id) => !almacen.demostraciones
        .some((d) => d.id === id && d.entorno === entorno));
      if (ajenos.length) {
        return rechazar(422, `Demostraciones que no son de ${entorno}: ${ajenos.join(', ')}`);
      }
      let valores;
      try {
        valores = efectivos(esquema(entorno, metodo), ajustes);
      } catch (error) {
        return rechazar(422, error.message);
      }
      // Como el real: el valor del campo de la duración, redondeado al bloque en el experto.
      const { unidad, campo, bloque } = esquema(entorno, metodo).duracion;
      const total = bloque ? Math.ceil(valores.get(campo) / valores.get(bloque)) * valores.get(bloque)
        : valores.get(campo);
      const id = almacen.nuevoId('entrenamiento');
      entrenamientos.set(id, { entorno, metodo, ajustes: copia(ajustes), datos: [...datos], vista,
        wandb, valores, duracion: { unidad, total }, semilla: semilla + entrenamientos.size });
      return Promise.resolve({ id, entorno, metodo, duracion: { unidad, total } });
    },

    conectar(id, callbacks = {}) {
      const entreno = entrenamientos.get(id);
      if (!entreno) {
        throw new ErrorApi(`Entrenamiento desconocido: ${id}`, { estado: 404 });
      }
      entrenamientos.delete(id);
      const { entorno, metodo, valores, duracion } = entreno;
      const aleatorio = crearAleatorio(entreno.semilla);
      const simulacion = METODOS_CON_VISTA.has(metodo)
        ? crearSimulacion(entorno, crearAleatorio(entreno.semilla + 1)) : null;
      const salto = metodo === METODOS.EXPERTO
        ? Math.max(1, Math.ceil(duracion.total / ACTUALIZACIONES / valores.get('experto.trpo.pasos_por_iteracion')))
          * valores.get('experto.trpo.pasos_por_iteracion')
        : Math.max(1, Math.ceil(duracion.total / ACTUALIZACIONES));
      let activa = true;
      let terminado = false;
      let vista = entreno.vista;
      let paso = 0;
      let reloj = null;
      let mejor = { valor: Infinity, paso: 0 };
      let ultimas = {};

      const avisar = (callback, ...argumentos) => {
        if (activa) {
          callback?.(...argumentos);
        }
      };
      const parar = () => {
        clearInterval(reloj);
        reloj = null;
      };
      const terminar = (incompleto, motivo) => {
        terminado = true;
        parar();
        const nombre = `${metodo}_${entorno}_${almacen.modelos.filter((m) => m.entorno === entorno
          && m.metodo === metodo).length + 1}`;
        const resumen = { unidad: duracion.unidad, paso, total: duracion.total, motivo,
          medidas: ultimas, ajustes: entreno.ajustes, datos: entreno.datos, wandb: entreno.wandb };
        almacen.modelos.push({ id: almacen.nuevoId('modelo'), nombre, entorno, metodo, incompleto,
          resumen, creado: new Date().toISOString() });
        // Como el servidor real, la respuesta llega después, no dentro de la llamada.
        setTimeout(() => {
          avisar(callbacks.alTerminar, { modelo: { nombre, incompleto }, resumen: copia(resumen) });
          activa = false;
        }, 0);
      };
      const dibujarVista = () => {
        if (!vista || !simulacion) {
          return;
        }
        for (let i = 0; i < 5; i += 1) {
          const { terminado: fin, truncado } = simulacion.paso(simulacion.accionExperto(aleatorio));
          if (fin || truncado) {
            simulacion.reiniciar();
          }
        }
        pintar(entorno, simulacion.foto())
          .then((imagen) => avisar(callbacks.alImagen, imagen))
          .catch((error) => avisar(callbacks.alError, error));
      };
      const avanzar = () => {
        if (!activa || terminado) {
          return;
        }
        const anterior = paso;
        paso = Math.min(duracion.total, paso + salto);
        const fraccion = paso / duracion.total;
        ultimas = medidasSimuladas(metodo, fraccion, valores, aleatorio, paso, anterior);
        avisar(callbacks.alProgreso, { paso, medidas: { ...ultimas } });
        dibujarVista();
        if (metodo === METODOS.EXPERTO && ultimas.retorno_medio >= valores.get('experto.retorno_objetivo')) {
          terminar(false, 'objetivo alcanzado');
        } else if (metodo === METODOS.BC && ultimas.perdida_validacion < mejor.valor) {
          mejor = { valor: ultimas.perdida_validacion, paso };
        } else if (metodo === METODOS.BC && paso - mejor.paso >= valores.get('bc.paciencia')) {
          terminar(false, 'parada temprana');
        }
        if (!terminado && paso >= duracion.total) {
          terminar(false, 'presupuesto completo');
        }
      };
      const arrancar = () => {
        reloj = setInterval(avanzar, INTERVALO_SIMULADO_MS);
      };

      setTimeout(() => avisar(callbacks.alEstado, ESTADOS_ENTRENAMIENTO.EN_MARCHA), 0);
      arrancar();

      const orden = (accion) => {
        if (!activa) {
          return false;
        }
        if (!terminado) {
          accion();
        }
        return true;
      };
      return {
        pausar: () => orden(() => {
          if (reloj) {
            parar();
            setTimeout(() => avisar(callbacks.alEstado, ESTADOS_ENTRENAMIENTO.PAUSADO), 0);
          }
        }),
        reanudar: () => orden(() => {
          if (!reloj) {
            arrancar();
            setTimeout(() => avisar(callbacks.alEstado, ESTADOS_ENTRENAMIENTO.EN_MARCHA), 0);
          }
        }),
        cambiarVista: (activar) => orden(() => {
          vista = Boolean(activar) && simulacion !== null;
        }),
        // Si ya ha terminado, su mensaje está de camino: finalizar no lo pierde.
        finalizar: () => orden(() => terminar(true, 'finalizado por el usuario')),
        cerrar() {
          activa = false;
          parar();
        },
      };
    },
  };
}
