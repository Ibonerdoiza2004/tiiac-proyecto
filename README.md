# TIIAC: GAIL y Behavioral Cloning

Replicación educativa de **Generative Adversarial Imitation Learning** (Ho y
Ermon, 2016): se compara GAIL con Behavioral Cloning (BC), tomando como
referencia un experto TRPO y una política aleatoria. CartPole es el entorno
principal y el que cumple la regla 5/20. MountainCar, Acrobot, Reacher y
HalfCheetah tienen también su perfil (ver [Datos](#datos)).

- Paper: [Generative Adversarial Imitation Learning](https://arxiv.org/abs/1606.03476)
- Seguimiento de experimentos: [proyecto W&B `tiiac-proyecto`](https://wandb.ai/a-corral-universidad-de-deusto/tiiac-proyecto)
- Autores: Asier Corral e Ibon Erdoiza.
- Probado en Linux con Python 3.12.

## Resultados

### Introducción
En este apartado explicaremos los resultados que hemos obtenido. Nuestro objetivo es comprobar si podemos llegar a las mismas conclusiones a las que se llegaron en el paper original, partiendo de nuestros resultados.

Se comparan cuatro agentes: Una política aleatoria, un experto, un agente entrenado mediante BC y un agente entrenado mediante GAIL. Para la evaluación, cada agente juega 50 episodios deterministas con las mismas semillas. Los resultados que comparamos en este apartado por cada entorno son la media de las recompensas obtenidas en la evaluación y la recompensa normalizada, donde 0 es la recompensa obtenida por la política aleatoria y 1 la obtenida por el experto.

### Experto y política aleatoria
Nuestros expertos igualan o superan a los del paper en todos los entornos, por lo que la exigencia para BC y GAIL es igual o mayor que en el paper de referencia. La política aleatoria también da valores parecidos, excepto en CartPole y Acrobot, los cuales han cambiado la duración máxima de los episodios. CartPole-v1 permite 500 pasos, mientras que la v0 solo permitía 200. Este incremento de los pasos hace que la política aleatoria consiga un porcentaje más pequeño de la recompensa alcanzable, modificando la normalización de la recompensa. En Acrobot-v1 la política aleatoria obtiene −500 en lugar de −200, lo que infla la recompensa normalizada. Por eso, estos entornos no son del todo comparables con los del paper.

### Resultados por entorno
#### CartPole

| Partidas | Aleatoria | Aleatoria (paper) | BC | BC (paper) | GAIL | GAIL (paper) | Experto | Experto (paper) |
|---|---|---|---|---|---|---|---|---|
| 1 | 21,3 | 18,6 | 406,1 (0,80) | 72,0 (0,29) | 500,0 (1,00) | 200,0 (1,00) | 500,0 | 200,0 |
| 4 | 21,3 | 18,6 | 500,0 (1,00) | 169,2 (0,83) | 499,3 (1,00) | 200,0 (1,00) | 500,0 | 200,0 |
| 7 | 21,3 | 18,6 | 500,0 (1,00) | 188,6 (0,94) | 500,0 (1,00) | 200,0 (1,00) | 500,0 | 200,0 |
| 10 | 21,3 | 18,6 | 500,0 (1,00) | 177,2 (0,87) | 500,0 (1,00) | 200,0 (1,00) | 500,0 | 200,0 |

El agente entrenado con GAIL, al igual que en el paper, alcanza la recompensa máxima o se queda muy cerca independientemente de la cantidad de partidas con las que se entrene. La diferencia está en BC, que en nuestra replicación consigue resultados mucho mejores.

La mejoría de BC ocurre probablemente a dos razones, y las dos tienen que ver con la forma de validar. La primera es que en el paper los pares de validación se sacan de las mismas partidas que los de entrenamiento, por lo que se parecen mucho a ellos y no permiten saber si el modelo generaliza a partidas nuevas. Nosotros, en cambio, validamos con 5 partidas distintas (las mismas en todos los entrenamientos) y nos quedamos con la época que mejor funciona en ellas. La segunda es que, al usar para validación el 30% de los pares del subconjunto, el paper entrena solo con el 70% restante, algo que pesa más cuantas menos partidas hay.

GAIL consigue la recompensa máxima entrenando solo durante 20 iteraciones, mientras que en el paper entrena durante 300. En la validación del entrenamiento, la media ya está entre 381 y 478 en la iteración 10, y entre 499 y 500 en la 20, por lo que la reducción de iteraciones no ha afectado al resultado. Por eso es bastante probable que el GAIL del paper también alcanzara la recompensa máxima mucho antes de las 300 iteraciones, pero que, como GAIL no tiene early stopping, siguiera entrenando.

#### MountainCar

| Partidas | Aleatoria | Aleatoria (paper) | BC | BC (paper) | GAIL | GAIL (paper) | Experto | Experto (paper) |
|---|---|---|---|---|---|---|---|---|
| 1 | −200,0 | −200,0 | −143,5 (0,55) | −136,8 (0,62) | −101,1 (0,97) | −101,6 (0,97) | −97,8 | −98,8 |
| 4 | −200,0 | −200,0 | −118,7 (0,80) | −133,3 (0,66) | −99,2 (0,99) | −101,4 (0,97) | −97,8 | −98,8 |
| 7 | −200,0 | −200,0 | −113,4 (0,85) | −127,3 (0,72) | −98,6 (0,99) | −99,9 (0,99) | −97,8 | −98,8 |
| 10 | −200,0 | −200,0 | −108,6 (0,89) | −123,1 (0,76) | −99,7 (0,98) | −100,8 (0,98) | −97,8 | −98,8 |

Los resultados de GAIL son parecidos a los del paper. Con una única partida ya consigue una puntuación muy similar a la del experto y GAIL supera a BC en 19 de las 20 comparaciones con la misma cantidad y repetición. Cuando se han usado 4 o más partidas, al final del entrenamiento el discriminador solamente acierta alrededor de la mitad de sus predicciones, lo que sugiere que el agente toma acciones muy parecidas a las del experto.

BC también muestra una tendencia parecida a la del paper. BC mejora a medida que tiene más partidas, pero no llega al experto. Con una partida obtiene un resultado algo peor que el del paper, y a partir de 4 partidas lo supera. Sus resultados son menos regulares que los de GAIL (todos los episodios llegan a la meta, pero algunos tardan mucho más que el experto).

#### Acrobot

| Partidas | Aleatoria | Aleatoria (paper) | BC | BC (paper) | GAIL | GAIL (paper) | Experto | Experto (paper) |
|---|---|---|---|---|---|---|---|---|
| 1 | −500,0 | −200,0 | −93,8 (0,95) | −130,6 (0,56) | −71,0 (1,00) | −77,3 (0,98) | −72,1 | −75,3 |
| 4 | −500,0 | −200,0 | −84,2 (0,97) | −93,2 (0,86) | −68,5 (1,01) | −83,1 (0,94) | −72,1 | −75,3 |
| 7 | −500,0 | −200,0 | −77,9 (0,99) | −96,9 (0,83) | −68,0 (1,01) | −82,6 (0,94) | −72,1 | −75,3 |
| 10 | −500,0 | −200,0 | −79,1 (0,98) | −95,1 (0,84) | −67,4 (1,01) | −78,9 (0,97) | −72,1 | −75,3 |

GAIL obtiene unos resultados muy buenos en este entorno. Consigue una recompensa mayor que la del paper y que la de BC con todas las cantidades de partidas e incluso queda algo por encima del experto. Al final del entrenamiento el discriminador sigue distinguiendo bien al agente del experto, acertando entre el 84% y el 100%.

BC también consigue mayor recompensa que en el paper con todas las cantidades de partidas. Al igual que en MountainCar, sus resultados son menos regulares que los de GAIL.

#### HalfCheetah

| Partidas | Aleatoria | Aleatoria (paper) | BC | BC (paper) | GAIL | GAIL (paper) | Experto | Experto (paper) |
|---|---|---|---|---|---|---|---|---|
| 4 | −320,7 | −282,4 | −342,4 (0,00) | −493,6 (−0,04) | 4775,3 (0,95) | 4515,7 (1,01) | 5040,9 | 4463,5 |
| 11 | −320,7 | −282,4 | 1061,5 (0,26) | 637,6 (0,19) | 4970,2 (0,99) | 4280,7 (0,96) | 5040,9 | 4463,5 |
| 18 | −320,7 | −282,4 | 2293,3 (0,49) | 2705,0 (0,63) | 4931,8 (0,98) | 4749,4 (1,06) | 5040,9 | 4463,5 |
| 25 | −320,7 | −282,4 | 4782,3 (0,95) | 3718,6 (0,84) | 4728,1 (0,94) | 4840,1 (1,08) | 5040,9 | 4463,5 |

GAIL consigue entre el 94 % y el 99 % de la recompensa del experto con todas las cantidades de partidas. En retorno, supera al paper en 3 de las 4 cantidades, pero en recompensa normalizada queda por debajo en 3 de ellas.

BC muestra la misma tendencia que en el paper. Con 4 partidas no mejora a la política aleatoria, mejora mucho a medida que tiene más partidas, y con 25 llega a un resultado muy parecido al de GAIL. Con 11 y 18 partidas sus resultados son muy irregulares, igual que en el paper. Con 18 partidas los episodios se reparten casi a partes iguales entre los dos extremos de puntuación.

#### Reacher

Para este entorno, el paper tiene una tabla con ejecuciones con distintos parámetros, nosotros tomamos el GAIL sin regularización de entropía (λ = 0), que es el equivalente al nuestro. El paper no prueba con 25 partidas.

| Partidas | Aleatoria | Aleatoria (paper) | BC | BC (paper) | GAIL | GAIL (paper) | Experto | Experto (paper) |
|---|---|---|---|---|---|---|---|---|
| 4 | −43,8 | −43,2 | −8,3 (0,88) | −11,0 (0,82) | −27,3 (0,41) | −67,2 (−0,61) | −3,4 | −4,1 |
| 11 | −43,8 | −43,2 | −9,3 (0,85) | −6,2 (0,95) | −20,7 (0,57) | −6,1 (0,95) | −3,4 | −4,1 |
| 18 | −43,8 | −43,2 | −10,4 (0,83) | −4,8 (0,98) | −94,6 (−1,26) | −8,3 (0,89) | −3,4 | −4,1 |
| 25 | −43,8 | −43,2 | −10,9 (0,81) | — | −12,1 (0,78) | — | −3,4 | — |

Es el único entorno en el que GAIL obtiene peores resultados que BC con todas las cantidades de partidas. En el paper, GAIL con 4 partidas queda por debajo de la política aleatoria, pero con 11 y 18 partidas se recupera y el nuestro no. La recompensa de Reacher tiene en cuenta la distancia a la diana y el coste de control. En la distancia, GAIL consigue valores parecidos a los de BC (aunque están lejos del experto), pero la diferencia está en el coste de control, donde el de GAIL va de −4,0 a −86,7 y el de BC de −0,5 a −1,4. Con 18 partidas, donde actúa peor que la política aleatoria, el 92 % de su recompensa negativa es coste de control. Al final del entrenamiento, el discriminador acierta prácticamente el 100% de sus predicciones, tanto con el experto como con el agente.

BC obtiene mejores resultados que GAIL, aunque su rendimiento real es más limitado de lo que indica la recompensa normalizada. Una política que deja el brazo quieto obtiene un 0,79 normalizado porque no paga ningún coste de control. BC solo mejora a esa política entre 0,8 y 3,4 puntos. Si se mira únicamente la distancia a la diana, BC acerca el brazo a la diana mucho menos que el experto. Además, a diferencia del paper, donde BC se queda cada vez más cerca de su experto, nuestro BC empeora a medida que tiene más partidas.

### Comparación entre BC y GAIL

La principal conclusión del paper es que GAIL necesita muchas menos demostraciones que BC para acercarse al experto y nuestros resultados la confirman en todos los entornos menos en Reacher. Con la menor cantidad de partidas, GAIL ya se acerca a la recompensa del experto, mientras que BC necesita más datos para acercarse.

GAIL también es más regular que BC en MountainCar y Acrobot, con todas las cantidades de partidas. La diferencia es especialmente grande en HalfCheetah con 11 y 18 partidas, donde BC alterna episodios en los que corre casi como el experto con otros en los que apenas avanza.

Aun así, GAIL tiene una desventaja respecto a BC. BC se entrena en pocos segundos y no necesita jugar en el entorno, mientras que GAIL necesita jugar cientos de miles o millones de pasos y tarda desde medio minuto en CartPole hasta casi tres horas en HalfCheetah y Reacher.

### Limitaciones

- **Pocas repeticiones en los entornos MuJoCo.** En Reacher y HalfCheetah solo hay un agente por cantidad de partidas, así que no se puede medir la variación entre semillas. Por ejemplo, no se puede saber si el −94,6 de GAIL con 18 partidas en Reacher es algo puntual.
- **Versiones distintas de los entornos.** Salvo en MountainCar, usamos versiones más recientes que las del paper (CartPole-v1, Acrobot-v1, Reacher-v5 y HalfCheetah-v5), en parte porque algunas de las originales ya no existen en Gymnasium. Por eso, los retornos no se pueden comparar directamente con los del paper, y en Acrobot la recompensa normalizada sale inflada.
- **Expertos distintos.** Nuestros expertos son iguales o mejores que los del paper, así que la recompensa normalizada no mide exactamente lo mismo en los dos casos.
- **Solo se mide el retorno.** La evaluación compara la recompensa obtenida, no si el agente se comporta igual que el experto.

### Trabajo futuro

- **Más repeticiones en Reacher y HalfCheetah.** Al menos 3 por cantidad de partidas, para poder medir la variación entre semillas. Cada entrenamiento de GAIL tarda unas 2 h 45 min, así que tendría un coste importante.
- **Comprobar el efecto de la recompensa del discriminador.** Entrenar GAIL en CartPole, MountainCar y Acrobot con una recompensa constante, sin discriminador, y comparar con nuestros resultados. Así se sabría cuánto del éxito de GAIL en esos entornos se debe a imitar al experto y cuánto al signo de la recompensa.
- **Más épocas para BC en MountainCar.** Con 4 partidas, las 5 repeticiones llegaron al máximo de 500 épocas sin que la parada temprana se activara, así que podría mejorar entrenando más.

## Ejecutar desde cero

### 1. Descargar el proyecto y crear el entorno

```bash
git clone https://github.com/Ibonerdoiza2004/tiiac-proyecto.git
cd tiiac-proyecto
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Si `python3.12 -m venv` falla por `ensurepip`, falta el paquete del sistema:
`sudo apt install python3.12-venv`, y se repite el comando.

### 2. Comprobar que todo funciona

No necesita W&B ni datos:

```bash
python -m pytest tests/ -q
python tools/tiiac_check.py .
python run.py --smoke
```

El checker ya ejecuta el smoke; el último comando sirve para verlo por separado
(ver [El smoke](#el-smoke)).

### 3. Iniciar sesión en W&B

```bash
wandb login --relogin --host https://api.wandb.ai
```

La clave se obtiene en [wandb.ai/authorize](https://wandb.ai/authorize) y tiene
que ser de una cuenta con acceso a la organización
`a-corral-universidad-de-deusto` (ver [Seguimiento W&B](#seguimiento-wb)).

### 4. Ejecutar el experimento

```bash
python run.py --entorno cartpole --descargar --grupo <nombre>
```

Descarga de W&B el experto y las demostraciones congelados (si no están ya en
local) y comprueba sus huellas. Después entrena 20 BC y 20 GAIL (subconjuntos de
1, 4, 7 y 10 partidas, 5 repeticiones), evalúa los 42 agentes (experto,
aleatorio, BC y GAIL) con 50 episodios cada uno e imprime una línea de métricas
por agente.

Los otros entornos se ejecutan igual, cambiando `--entorno`. Usan los tamaños del
código original, así que tardan más: 37 y 50 min MountainCar y Acrobot, y unas
2 h 45 min Reacher y HalfCheetah (ver [Tiempos medidos](#tiempos-medidos)):

| Entorno | BC + GAIL | Agentes evaluados | Iteraciones de GAIL |
|---|---|---|---|
| `cartpole` | 20 + 20 (4 cantidades × 5 repeticiones) | 42 | 20 de 5000 pasos |
| `mountaincar`, `acrobot` | 20 + 20 (4 × 5) | 42 | 300 de 5000 pasos |
| `reacher`, `halfcheetah` | 4 + 4 (4 × 1) | 10 | 500 de 50 000 pasos |

Conviene lanzarlos de uno en uno: cada experimento ya reparte sus
entrenamientos en `experimento.procesos` procesos.

## Argumentos de `run.py`

| Argumento | Para qué sirve | Por defecto |
|---|---|---|
| `--entorno <nombre>` | Perfil de entorno (`configs/entornos/<nombre>.yaml`): `cartpole`, `mountaincar`, `acrobot`, `reacher` o `halfcheetah`. | `cartpole` |
| `--config <yaml>` | Configuración base sobre la que se aplica el perfil. | `configs/default.yaml` |
| `--smoke` | Ejecuta el flujo completo en miniatura (ver [El smoke](#el-smoke)). | — |
| `--descargar` | Baja de W&B el experto y las demostraciones congelados que falten. Solo descarga en carpetas ausentes o vacías y nunca sobrescribe. | — |
| `--datos <carpeta>` | Usa otra carpeta de demostraciones en lugar de `datos.demostraciones`. Se comprueba que su huella sea la de las congeladas. | la del perfil |
| `--experto <carpeta>` | Usa otra carpeta del experto en lugar de `datos.experto`, con la misma comprobación. | la del perfil |
| `--salida <carpeta>` | Carpeta nueva del experimento. Con `--solo-evaluar`, la carpeta de la ejecución que se vuelve a evaluar. | `checkpoints/run/<entorno>/<fecha>` |
| `--solo-evaluar` | No entrena: vuelve a evaluar los BC y GAIL de `--salida` y guarda un `evaluacion-<fecha>.json` nuevo. Necesita `--salida`. | — |
| `--sin-wandb` | No registra nada en W&B. `--descargar` sigue necesitando la sesión. | — |
| `--grupo <nombre>` | Grupo de W&B que reúne los runs de la ejecución. | uno aleatorio |

`--smoke` ignora `--datos`, `--experto`, `--descargar` y `--salida`, porque
crea todo en una carpeta temporal.

**`--grupo`** acepta cualquier texto. En W&B un grupo no se crea de antemano:
si nadie lo ha usado, aparece uno nuevo; si ya existe, los runs se suman a él
sin sobrescribir nada. Sin `--grupo`, `run.py` crea un identificador aleatorio
de 12 caracteres hexadecimales y lo imprime al empezar (`grupo W&B: …`).
Conviene un nombre distinto para cada ejecución que no se quiera mezclar con
otras.

Códigos de salida:

| Código | Significado |
|---|---|
| 0 | Todo correcto. |
| 1 | Falló algún entrenamiento; los demás terminan, pero no se evalúa. |
| 2 | La configuración, los datos, la salida o la sesión de W&B no son válidos (se comprueban antes de entrenar), o falló la evaluación. |
| 130 | Interrumpido con Ctrl+C. |

## El smoke

```bash
python run.py --smoke
```

Ejecuta el flujo real en miniatura para comprobar que el pipeline llega al final.
Aplica la capa `configs/smoke.yaml`, desactiva W&B (no pide sesión) y no usa los
datos congelados. Todo se crea en una carpeta temporal que se borra al terminar:

1. Entrena un experto de 1000 pasos.
2. Graba 3 partidas: 2 de train y 1 de validación.
3. Entrena 2 BC y 2 GAIL (subconjuntos de 1 y 2 partidas, 1 repetición,
   2 iteraciones de GAIL).
4. Evalúa los 6 agentes con 3 episodios cada uno.

Ha ido bien si termina con `=== métricas ===`, una línea por agente y código 0
(`echo $?`). Los números no significan nada. En CartPole tarda unos 5 s (ver
[Tiempos medidos](#tiempos-medidos)); para medirlo, `time python run.py --smoke`.
Con `--entorno` se prueba el perfil de otro entorno.

## Qué produce un experimento

En disco, en `checkpoints/run/<entorno>/<fecha>/` (o en `--salida`), que no se
versiona:

```text
bc/cantidad_<c>/repeticion_<r>.pt          política de BC con sus metadatos
gail/cantidad_<c>/repeticion_<r>/
    modelo.zip                             agente TRPO de GAIL
    discriminador.pt                       pesos y normalización del discriminador
    gail.json                              configuración, resumen, procedencia y hardware
evaluacion.json                            protocolo y retornos de todos los agentes
```

En W&B, cada ejecución forma un grupo con un run por entrenamiento,
`tiiac-bc-<entorno>-c<c>-r<r>` y `tiiac-gail-<entorno>-c<c>-r<r>` (41 runs en
total en CartPole), y `tiiac-comparacion-<entorno>`, que aparece al terminar la
evaluación con las tablas `comparacion` y `episodios`. Las métricas se agrupan en
secciones por prefijo: `bc/`, `trpo/`, `discriminador/`, `reajuste/` y
`validacion/`.

En *Runs*, filtra por `Group` para ver una ejecución y agrupa por `barrido.metodo` o
`barrido.cantidad` para comparar curvas.

## Datos

El experto y las demostraciones de cada entorno no se versionan: están congelados en
W&B (grupo `congelado`) y `run.py --descargar` los baja a
`checkpoints/expertos/<entorno>/congelado/` y `data/demostraciones/<entorno>/congelado/`.
Antes de entrenar, `run.py` comprueba sus huellas SHA-256, que fija la sección `datos`
de cada perfil (`configs/entornos/<entorno>.yaml`), y rechaza los que no coincidan.

| Entorno | Experto | Demostraciones | Partidas (train + validación) |
|---|---|---|---|
| CartPole-v1 | `experto-CartPole-v1-92c44313863e` | `demostraciones-5c5168e7f5f8` | 15 (10 + 5) |
| MountainCar-v0 | `experto-MountainCar-v0-d492bda5bd8f` | `demostraciones-86c31ceb8ace` | 15 (10 + 5) |
| Acrobot-v1 | `experto-Acrobot-v1-fad2206a39f6` | `demostraciones-45531794f3b9` | 15 (10 + 5) |
| Reacher-v5 | `experto-Reacher-v5-a69d425ddb0f` | `demostraciones-1b16407e4fd8` | 30 (25 + 5) |
| HalfCheetah-v5 | `experto-HalfCheetah-v5-331994138466` | `demostraciones-9a81e93f08fa` | 30 (25 + 5) |

Se generan con `scripts/entrenar_experto.py` y `scripts/get_data.py` (ver
[Etapas por separado](#etapas-por-separado)). Son datos propios, generados en los
entornos de Gymnasium (licencia MIT); Reacher y HalfCheetah usan además MuJoCo
(licencia Apache 2.0).

## Etapas por separado

`run.py` encadena todo, pero cada etapa tiene su script. El primero imprime un
grupo de W&B que se pasa a los siguientes para reunir los runs:

```bash
python scripts/entrenar_experto.py --entorno cartpole
python scripts/get_data.py --entorno cartpole --experto checkpoints/expertos/cartpole/<fecha> --grupo <grupo>
python scripts/entrenar_bc.py --entorno cartpole --datos data/demostraciones/cartpole/<fecha> --grupo <grupo>
python scripts/entrenar_gail.py --entorno cartpole --datos data/demostraciones/cartpole/<fecha> --grupo <grupo>
python scripts/evaluar.py --entorno cartpole --experto checkpoints/expertos/cartpole/<fecha> --bc checkpoints/bc/cartpole/<fecha> --gail checkpoints/gail/cartpole/<fecha> --grupo <grupo>
```

Los barridos (`entrenar_bc.py`, `entrenar_gail.py` y `evaluar.py`) aceptan
`--cantidades` y `--repeticiones` para repartirlos entre sesiones. Todos
aceptan `--sin-wandb` y `--salida`.

## Añadir otro entorno

1. Añadir su nombre a `TipoEntorno` en `src/utils/configuracion.py`; de ahí salen
   las opciones de `--entorno`.
2. Crear `configs/entornos/<nombre>.yaml` con todos los valores que
   `configs/default.yaml` deja a `null`. Las seis claves de `datos` van con el
   valor provisional `pendiente`, porque los datos todavía no existen.
3. Comprobar el pipeline con `python run.py --smoke --entorno <nombre>`.
4. Generar y congelar el experto y las demostraciones con `scripts/entrenar_experto.py`
   y `scripts/get_data.py` (ver [Etapas por separado](#etapas-por-separado)) y
   sustituir `pendiente` por sus rutas, huellas y artefactos.

Mientras quede algún `pendiente`, se pueden ejecutar el smoke y los scripts, pero
`run.py --entorno <nombre>` sale con código 2 y dice qué valor falta.

## Seguimiento W&B

El proyecto es privado y pertenece a la organización
`a-corral-universidad-de-deusto`. Para ver los runs o descargar los artefactos
hace falta una cuenta con acceso: pedidlo a los autores. Con una clave de otra
organización, W&B responde con un error 403.

La credencial se guarda fuera del repositorio (`wandb login`). Para no registrar un
experimento, usa `--sin-wandb`; los tests y el smoke no registran nada.

| Grupo | Qué contiene |
|---|---|
| `congelado` | Los expertos y las demostraciones congelados |
| `entrega-e1` | El experimento completo de CartPole medido para la entrega |
| `entrega-e1-cartpole` | El experimento completo de CartPole repetido el 2026-11-12 (run de comparación `ft49odps`), con los mismos retornos que `entrega-e1`; es el que usa [Resultados](#resultados) |
| `entrega-e1-mountaincar`, `entrega-e1-acrobot` | El experimento completo de MountainCar y de Acrobot |
| `entrega-e1-reacher`, `entrega-e1-halfcheetah` | El experimento completo de Reacher y de HalfCheetah (runs de comparación `u0imo43m` y `k7v7i4fo`). El de Reacher contiene también runs de intentos interrumpidos |
| `ajuste-experto-halfcheetah`, `ajuste-experto-reacher` | La comparación de hiperparámetros del experto (ver [Ajuste del experto](#ajuste-del-experto)) |

## Tiempos medidos

`experimento.procesos` (en `configs/default.yaml`) fija cuántos entrenamientos
se ejecutan a la vez. Está en 5, elegido para un PC de 6 núcleos físicos. En
otra máquina conviene ponerlo como mucho en su número de núcleos físicos,
contando unos 830 MB de RAM por proceso. No cambia los resultados, solo el
tiempo.

Con los datos congelados y la configuración actual:

| Medición | Tiempo | Hardware | Fecha |
|---|---|---|---|
| Smoke de CartPole (`run.py --smoke`, dentro del checker) | 5 s | Linux (WSL2), Intel Core i7-11800H, 16 procesadores lógicos, sin GPU | 2026-10-30 |
| Experimento completo de CartPole (`time python run.py --entorno cartpole --grupo entrega-e1 --descargar`, online): 20 BC, 20 GAIL de 20 iteraciones y evaluación | 7 min 13 s, de ellos 67 s de evaluación (resumen de W&B) | Linux (WSL2), Intel Core i7-11800H, 16 procesadores lógicos, 5 procesos, sin GPU | 2026-10-30 |
| Experto congelado de CartPole (60 000 pasos, parada al llegar a 500) | 14 s | Linux (WSL2), misma máquina | 2026-10-30 |
| Demostraciones congeladas de CartPole (15 partidas, 7500 pasos) | 1,7 s | Linux (WSL2), misma máquina | 2026-10-30 |
| Ajuste del experto de HalfCheetah (cinco variantes a la vez, un hilo cada una, hasta unos 5 M pasos o el objetivo) | 1803–3390 s por variante | Linux (WSL2), Intel Core i7-11800H | 2026-10-31 |
| Ajuste del experto de Reacher (ídem) | 1641–6028 s por variante | Linux (WSL2), Intel Core i7-11800H | 2026-10-31 |
| Expertos congelados de MountainCar (1 000 000 pasos), Acrobot (980 000), Reacher (25 001 984) y HalfCheetah (7 168 000) | 428 s, 506 s, 23 532 s (6 h 32 min) y 3390 s (57 min) | Linux (WSL2), Intel Core i7-11800H, los cuatro a la vez, un hilo cada uno | 2026-11-01 |
| Demostraciones congeladas de MountainCar, Acrobot, Reacher y HalfCheetah | 0,7 s, 0,6 s, 0,6 s y 9,0 s | Linux (WSL2), misma máquina | 2026-11-01 |
| Experimento completo de MountainCar (`time python run.py --entorno mountaincar --descargar --grupo entrega-e1-mountaincar`, online, commit `83323cb`): 20 BC, 20 GAIL de 300 iteraciones y evaluación de 42 agentes | 37 min 6 s, de ellos 41 s de evaluación (`evaluacion.json`) | Linux (WSL2), Intel Core i7-11800H, 5 procesos, sin GPU | 2026-11-02 |
| Experimento completo de Acrobot (ídem, grupo `entrega-e1-acrobot`) | 49 min 34 s, de ellos 38 s de evaluación | Linux (WSL2), misma máquina | 2026-11-02 |
| Experimento completo de Reacher (`python run.py --entorno reacher --descargar --grupo entrega-e1-reacher`, online, commit `a1fa25d`): 4 BC, 4 GAIL de 500 iteraciones a la vez y evaluación de 10 agentes | Unas 2 h 44 min, de ellos 12 s de evaluación (sin `time`: desde la creación de la carpeta de salida hasta `evaluacion.json`) | Linux (WSL2), misma máquina | 2026-11-14 |
| Experimento completo de HalfCheetah (ídem, grupo `entrega-e1-halfcheetah`) | Unas 2 h 46 min, de ellos 42 s de evaluación (medido igual) | Linux (WSL2), misma máquina | 2026-11-14 |

La ejecución de 7 min 13 s es el grupo `entrega-e1` de W&B (run de comparación
`r6oajx1r`). El *Runtime* que muestra W&B en una fila agrupada resume los runs
del grupo y no equivale al tiempo de la ejecución.

## Decisiones donde el paper no especifica

| Ambigüedad o diferencia | Qué decidimos | Por qué |
|---|---|---|
| Iteraciones de GAIL (paper: 300 en los entornos clásicos) | 20 en CartPole | Presupuesto de la regla 5/20 |
| Repeticiones (código original: 7 en los clásicos, 1 en MuJoCo) | 5 en CartPole, MountainCar y Acrobot; 1 en Reacher y HalfCheetah, como el original | Repartir el barrido en 5 procesos sin compartir núcleo |
| Entornos por iteración de TRPO | 10 entornos en paralelo con los mismos pasos por iteración (5000 en CartPole), en GAIL y en el experto; el experto de Reacher y HalfCheetah usa 2, como RL Zoo | Unas 2,7 veces más rápido en CartPole, medido |
| Normalización de observaciones de TRPO | Desactivada en CartPole; activada en los demás, como el original | En CartPole no hace falta (elección propia); en MuJoCo sí |
| Objetivos de la red de valor tras actualizar D | Retornos GAE, no Monte Carlo | Coherente con el resto de TRPO en sb3 |
| Ajuste del crítico | Adam por minibatches, no región de confianza | Implementación de sb3-contrib |
| Elección del experto | El mejor por evaluación, parando al alcanzar el retorno objetivo | El original usa un snapshot fijo |
| Retorno objetivo del experto | 500 en CartPole (el máximo; el umbral de Gymnasium es 475), −90 en MountainCar, −70 en Acrobot, −3,5 en Reacher y 5000 en HalfCheetah, por encima del experto del paper (tabla 1) y del umbral de Gymnasium. MountainCar y Acrobot se evalúan con 50 partidas | El código original no lo fija. Con el umbral de Gymnasium, MountainCar y Acrobot paraban justo al cruzarlo, por debajo del experto del paper |
| Pasos máximos del experto | 100 000 en CartPole, 1 000 000 en MountainCar y Acrobot y 25 000 000 en Reacher y HalfCheetah; si no es múltiplo de los pasos por iteración, para en la primera que lo supera (Reacher y HalfCheetah: 25 001 984) | En MuJoCo, el presupuesto del snapshot `iter0000500` del original (500 × 50 000); en el resto, tope propio |
| Hiperparámetros del experto | Los de `trpo` (paper y código original), salvo en Reacher y HalfCheetah, que usan los de RL Zoo sin normalizar la recompensa (`experto.trpo`; ver [Ajuste del experto](#ajuste-del-experto)) | El paper no dice cómo se entrenaron sus expertos más allá de TRPO; GAIL conserva los suyos |
| Normalización de la entrada del discriminador | Fija con los pares expertos de train | El original la actualiza también con pares del agente |
| Validación de partidas (paper: 70/30 de pares) | 5 partidas de validación, separadas por trayectoria, además de las de train (10, o 25 en Reacher y HalfCheetah) | Evitar fugas entre episodios |
| Parada de BC | Paciencia de 20 épocas sobre la pérdida de validación | El paper solo dice «hasta que deja de bajar» |
| Versiones de Acrobot y HalfCheetah (original: v0 y v1) | `Acrobot-v1` y `HalfCheetah-v5` | Las del original ya no existen en Gymnasium |
| Tamaños de Reacher (paper: 200 iteraciones, subconjuntos de 4, 11 y 18 partidas y 7 políticas) | Los de los entornos MuJoCo del código original: 500 iteraciones, subconjuntos de 4, 11, 18 y 25 y 1 repetición | Reacher no aparece en los pipelines del código original |

### Ajuste del experto

`experto.trpo` cambia claves de la sección `trpo` solo al entrenar el experto; GAIL
sigue con los valores del paper. En Reacher y HalfCheetah se compararon variantes de
los hiperparámetros de TRPO de RL Zoo, de unos 5 M pasos cada una, en los grupos de
W&B `ajuste-experto-reacher` y `ajuste-experto-halfcheetah`. El equipo eligió en los
dos la de RL Zoo sin normalizar la recompensa, con su red (256 × 256 ReLU en Reacher y
64 × 64 tanh en HalfCheetah); sus valores están en `experto.trpo` de cada perfil.

## Dashboard (en desarrollo)

El frontend del aplicativo de la E2 está en `web/`: React con JavaScript y Vite, con
Vitest y Testing Library para los componentes y Playwright para las pruebas de punta a
punta. Hace falta Node.js 24 o posterior (probado con 26.13.0, la LTS); las versiones
exactas de las dependencias están en `web/package-lock.json`. Desde `web/`:

| Para qué | Comando |
|---|---|
| Instalar las dependencias | `npm ci` |
| Servidor de desarrollo (http://127.0.0.1:5173) | `npm run dev` |
| Tests de componentes | `npm test` |
| Navegador para Playwright, una vez | `npx playwright install chromium` |
| Tests de punta a punta | `npm run e2e` |
| Compilar para producción (`web/dist/`) | `npm run build` |

En desarrollo, Vite reenvía `/api` (HTTP y WebSocket) a `http://127.0.0.1:8000`, o a la
dirección de `TIIAC_API_URL`. En producción, FastAPI sirve `web/dist/`.

Hecho hasta ahora: las pestañas, el selector de entorno y, en el Laboratorio, «Jugar tú»
(CartPole y MountainCar, con las flechas o A y D) y «Ver agente» (los cinco entornos), con
contadores de partidas y pares y la revisión de qué partidas y pares guardar; y
«Entrenar experto» y «Entrenar e imitar» (GAIL y BC): formulario con todos los
hiperparámetros del perfil, elección de demostraciones, curvas en directo, vista de las
partidas (experto y GAIL) y pausar, reanudar y finalizar. Mientras no existe el backend, la
interfaz usa uno simulado en el navegador (`web/src/simulado/`): CartPole y MountainCar
con las ecuaciones de Gymnasium, animaciones de ejemplo en los demás y curvas de
entrenamiento inventadas; no entrena ni guarda nada de verdad. Los hiperparámetros del
simulado salen de `configs/` con `npm run perfiles` (un test avisa si no están al día). Con
`VITE_API=real npm run dev` usa el backend real. Las rutas y los mensajes propuestos están
en [`docs/contrato-api.md`](docs/contrato-api.md).

## Estructura del repositorio

```text
run.py                  punto de entrada: smoke, experimento y reevaluación
configs/                toda la configuración: default.yaml, perfil de cada entorno y smoke.yaml
src/data/               entornos, demostraciones, splits por trayectoria y loaders
src/models/factory.py   política de BC y discriminador
src/training/           TRPO y experto, BC, GAIL, barridos, evaluación y flujo de run.py
src/utils/              configuración, semillas, seguimiento W&B, archivos y validación
src/app/                esqueleto del aplicativo (E2)
web/                    frontend del aplicativo (React con Vite) y sus tests
scripts/                cada etapa por separado
tests/                  tests (pytest)
tools/tiiac_check.py    comprobador de la asignatura
docs/                   propuesta y presentación
```

## Uso de IA

Las conversaciones del proyecto se registran en [`AI_USAGE.md`](AI_USAGE.md).
