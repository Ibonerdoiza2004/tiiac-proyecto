/**
 * Entornos del dashboard, en el orden de las etapas del proyecto.
 *
 * `id` es el valor de `--entorno` y el nombre del perfil en `configs/entornos/`
 * (`TipoEntorno` en `src/utils/configuracion.py`); `gymnasium` es el `entorno.id` de ese
 * perfil. Un test comprueba que coinciden con los perfiles.
 *
 * @typedef {object} Entorno
 * @property {string} id Clave del perfil (`cartpole`, `mountaincar`…).
 * @property {string} nombre Nombre que se muestra.
 * @property {string} gymnasium Identificador de Gymnasium.
 * @property {number} etapa Posición en el orden del proyecto (1 a 5).
 * @property {'discretas' | 'continuas'} acciones Tipo de espacio de acciones.
 * @property {boolean} jugarAMano Si se puede jugar y grabar demostraciones con el teclado.
 * @property {string} objetivo Para qué sirve el entorno en el proyecto.
 */

/** @type {readonly Entorno[]} */
export const ENTORNOS = Object.freeze(
  [
    {
      id: 'cartpole',
      nombre: 'CartPole',
      gymnasium: 'CartPole-v1',
      etapa: 1,
      acciones: 'discretas',
      jugarAMano: true,
      objetivo: 'Validar la implementación y la comparación inicial.',
    },
    {
      id: 'mountaincar',
      nombre: 'MountainCar',
      gymnasium: 'MountainCar-v0',
      etapa: 2,
      acciones: 'discretas',
      jugarAMano: true,
      objetivo: 'Evaluar secuencias de acciones que permiten acumular impulso.',
    },
    {
      id: 'acrobot',
      nombre: 'Acrobot',
      gymnasium: 'Acrobot-v1',
      etapa: 3,
      acciones: 'discretas',
      jugarAMano: false,
      objetivo: 'Comprobar el método con una dinámica articulada diferente.',
    },
    {
      id: 'reacher',
      nombre: 'Reacher',
      gymnasium: 'Reacher-v5',
      etapa: 4,
      acciones: 'continuas',
      jugarAMano: false,
      objetivo: 'Pasar a acciones continuas y estudiar cuándo BC resulta competitivo.',
    },
    {
      id: 'halfcheetah',
      nombre: 'HalfCheetah',
      gymnasium: 'HalfCheetah-v5',
      etapa: 5,
      acciones: 'continuas',
      jugarAMano: false,
      objetivo: 'Evaluar locomoción con múltiples controles continuos.',
    },
  ].map((entorno) => Object.freeze(entorno)),
);

/** Entorno con el que arranca el dashboard. */
export const ENTORNO_INICIAL = 'cartpole';

/**
 * Busca un entorno por su clave.
 *
 * @param {string} id Clave del perfil, como `cartpole`.
 * @returns {Entorno} El entorno con esa clave.
 * @throws {RangeError} Si no hay ningún entorno con esa clave.
 */
export function buscarEntorno(id) {
  const entorno = ENTORNOS.find((candidato) => candidato.id === id);
  if (!entorno) {
    throw new RangeError(`Entorno desconocido: ${String(id)}`);
  }
  return entorno;
}
