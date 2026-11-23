# Contrato de la API: «Jugar tú» y «Ver agente»

Propuesta para acordar entre `laboratorio-jugar-web` (frontend) y `laboratorio-jugar-api`
(backend). El frontend ya la usa contra un backend simulado (`web/src/simulado/`); todo lo
que hay aquí está también en `web/src/api/laboratorio.js`, que es lo único que habría que
tocar si cambia. Las rutas cuelgan de `/api`.

## HTTP

| Método y ruta | Cuerpo | Respuesta |
|---|---|---|
| `GET /laboratorio/{entorno}` | | `{fps, partidas_experto}` |
| `POST /sesiones` | `{entorno, modo, partidas?}` | `{id, entorno, modo, fps, partidas}` |
| `POST /sesiones/{id}/guardar` | `{nombre, partidas: [{indice, tramo: [inicio, fin]}]}` | `{nombre, partidas, pares}` |
| `DELETE /sesiones/{id}` | | 204 |

- `entorno` es la clave del perfil (`cartpole`, `mountaincar`, `acrobot`, `reacher`,
  `halfcheetah`) y `modo`, `manual` o `experto`.
- `fps` es el `render_fps` del entorno (velocidad ×1) y `partidas_experto`, las del perfil
  (`partidas_train + partidas_validacion`).
- `partidas` solo se usa en modo `experto`; si falta, las del perfil. En modo `manual`
  la respuesta trae `partidas: null`: se juega hasta que el jugador pare.
- El modo `manual` solo se admite en CartPole y MountainCar.
- Los tramos de pares van desde 0 y sin incluir el final, como en `src/data`: `[0, 20]`
  son los pares que la interfaz muestra como 1–20.
- Errores como en FastAPI: código HTTP y `{"detail": "mensaje"}` (o la lista de errores
  de validación). La interfaz muestra el mensaje tal cual: 404 si no existe la sesión o el
  entorno, 422 si los datos no son válidos (por ejemplo, un tramo fuera de la partida).

## WebSocket `/sesiones/{id}`

Del navegador al servidor (JSON):

| Mensaje | Cuándo |
|---|---|
| `{"tipo": "accion", "accion": 1}` | Al pulsar o soltar teclas: la acción que se mantiene, o `null` si no hay ninguna |
| `{"tipo": "finalizar"}` | El jugador termina la sesión |

Del servidor al navegador:

| Mensaje | Contenido |
|---|---|
| Binario | Cada imagen del entorno en JPEG (`rgb_array`) |
| `{"tipo": "estado", "partida", "paso", "retorno", "pares", "esperando"}` | Progreso de la partida en curso; `pares` cuenta las terminadas y la actual |
| `{"tipo": "partida_terminada", "partida": {indice, pasos, retorno, semilla}}` | Al acabar cada partida |
| `{"tipo": "sesion_terminada", "partidas": [...]}` | Al terminar (por `finalizar` o porque el experto completó las pedidas): solo partidas completas |
| `{"tipo": "error", "mensaje"}` | Cualquier fallo |

Comportamiento acordado en el plan (sección 2 y rama 8):

- CartPole no tiene acción de «no hacer nada»: con `accion: null` la partida se para
  (`esperando: true`) hasta que llegue otra acción. En MountainCar, `null` es la acción 1,
  no acelerar (pendiente de confirmar).
- Acciones: CartPole 0 = izquierda, 1 = derecha; MountainCar 0 = izquierda, 1 = nada,
  2 = derecha. Las teclas las traduce el frontend.
- Velocidad ×1: un paso por imagen, a `fps` imágenes por segundo.
- Al jugar a mano, cuando acaba una partida empieza otra con una semilla nueva.
- La partida en curso al finalizar no se guarda: el formato de las demostraciones exige
  que la última transición termine o se trunque.

# Contrato de la API: entrenamiento

Propuesta para acordar entre `laboratorio-entrenar-web` y `laboratorio-entrenar-api`, igual
que la de arriba: el frontend la usa contra el backend simulado y todo está también en
`web/src/api/entrenamiento.js`. Las rutas cuelgan de `/api`.

## HTTP

| Método y ruta | Cuerpo | Respuesta |
|---|---|---|
| `GET /laboratorio/{entorno}/ajustes/{metodo}` | | `{metodo, duracion: {unidad, total}, secciones}` |
| `GET /biblioteca/demostraciones?entorno={entorno}` | | `[{id, nombre, tipo, partidas, pares, fuente}]` |
| `POST /entrenamientos` | `{entorno, metodo, ajustes, datos?, vista, wandb}` | `{id, entorno, metodo, duracion: {unidad, total}}` |

- `metodo` es `experto`, `bc` o `gail`.
- `secciones` son los hiperparámetros del perfil resuelto: `[{clave, campos: [{clave, tipo,
  valor, min?, opciones?, ayuda?}]}]`. La clave de la sección es su ruta en `configs/`
  (`bc`, `trpo`, `gail`, `gail.discriminador`, `demostraciones`, `experto`, y
  `experto.trpo` con los valores de `trpo` que usa el experto). `tipo` es `entero`, `real`,
  `booleano`, `opcion` o `lista_enteros`, y `ayuda`, de dónde sale el valor.
- `ajustes` lleva solo lo que cambia el usuario, anidado como en `configs/`, por ejemplo
  `{"gail": {"discriminador": {"learning_rate": 0.02}}}`: es una capa más sobre el perfil.
- `datos` son los `id` de los conjuntos y subconjuntos elegidos (obligatorio en BC y
  GAIL); los pares repetidos se cuentan una vez.
- `vista` solo se admite en experto y GAIL; `wandb` es `false` por defecto.
- `duracion` es lo que mide el progreso: `pasos` en el experto, `epocas` en BC e
  `iteraciones` en GAIL.

## WebSocket `/entrenamientos/{id}`

Del navegador al servidor (JSON): `{"tipo": "pausar"}`, `{"tipo": "reanudar"}`,
`{"tipo": "vista", "activa": true}` y `{"tipo": "finalizar"}`.

Del servidor al navegador:

| Mensaje | Contenido |
|---|---|
| Binario | Imágenes en JPEG de las partidas del entrenamiento, solo con la vista encendida |
| `{"tipo": "progreso", "paso", "medidas": {nombre: valor}}` | Tras cada iteración, época o evaluación; `paso` en la unidad de `duracion` |
| `{"tipo": "estado", "estado"}` | `en_marcha` o `pausado`, al cambiar |
| `{"tipo": "terminado", "modelo": {nombre, incompleto}, "resumen"}` | Al acabar o tras `finalizar` (`incompleto: true`); el modelo ya está en la biblioteca |
| `{"tipo": "error", "mensaje"}` | Cualquier fallo |

Comportamiento previsto en el plan (sección 3 y rama 10): un modelo por ejecución; pausar y
reanudar no cambian el resultado; al finalizar antes de tiempo se guarda el modelo de ese
momento, marcado como incompleto.
