# TIIAC: GAIL y Behavioral Cloning

Replicación educativa de **Generative Adversarial Imitation Learning** (Ho y
Ermon, 2016): se compara GAIL con Behavioral Cloning (BC), tomando como
referencia un experto TRPO y una política aleatoria. CartPole es el entorno
principal y el que cumple la regla 5/20. MountainCar, Acrobot, Reacher y
HalfCheetah tienen también su perfil (ver [Otros entornos](#otros-entornos)).

- Paper: [Generative Adversarial Imitation Learning](https://arxiv.org/abs/1606.03476)
- Seguimiento de experimentos: [proyecto W&B `tiiac-proyecto`](https://wandb.ai/a-corral-universidad-de-deusto/tiiac-proyecto)
- Autores: Asier Corral e Ibon Erdoiza.
- Probado en Linux con Python 3.12.

## Resultado

Pendiente de completar por el equipo.

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
código original, así que tardan horas en lugar de minutos (ver [Tiempos medidos](#tiempos-medidos)):

| Entorno | BC + GAIL | Agentes evaluados | Iteraciones de GAIL |
|---|---|---|---|
| `cartpole` | 20 + 20 (4 cantidades × 5 repeticiones) | 42 | 20 de 5000 pasos |
| `mountaincar`, `acrobot` | 20 + 20 (4 × 5) | 42 | 300 de 5000 pasos |
| `reacher`, `halfcheetah` | 4 + 4 (4 × 1) | 10 | 500 de 50 000 pasos |

Conviene lanzarlos de uno en uno: cada experimento ya reparte sus
entrenamientos en `experimento.procesos` procesos.

### Comandos estándar

| Para qué | Comando |
|---|---|
| Smoke | `python run.py --smoke` |
| Experimento completo | `python run.py --entorno cartpole --descargar --grupo <nombre>` |
| Otro entorno | `python run.py --entorno <entorno> --descargar --grupo <nombre>` |

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

Para ver una ejecución: en *Runs*, **Filter** → `Group` = `<nombre>`. Para
comparar curvas, **Group** por `barrido.metodo` o `barrido.cantidad`, que cada
run de BC y GAIL lleva en su configuración.

## Datos

Las demostraciones y el experto no se versionan: se generan con los scripts de
`scripts/`, se congelan y se publican en W&B, en el grupo `congelado`. La sección
`datos` del perfil de cada entorno fija sus carpetas, la huella de las partidas,
la de `modelo.zip` y los artefactos; `run.py` rechaza los datos cuyas huellas no
coincidan.

### CartPole

Congelado el 2026-10-30 (commit `15416db8bdec494f55bba3e2ff527907bdc6fafc`) en
`data/demostraciones/cartpole/congelado/` y
`checkpoints/expertos/cartpole/congelado/`. Se generó en Linux (WSL2) con
Python 3.12.3 y las versiones de `requirements.txt`. Sus huellas SHA-256 son:

- `demostraciones.npz`: `5c5168e7f5f8ad8dc415ff8e13ff0acc3d9fe8e0445e4f9d0298f0817198aff9`
- `splits.json`: `804bb82f4c5b16ce4a49e97bde07911d1af1687a83a33cafabe7e9442f52a9d3`
- contenido de las partidas: `b31d19eda39c1167b67330995e26659d0782b43ae83cfde7b773e8b1df221f58`
- `modelo.zip` del experto: `92c44313863ebe30c616be30664b2742650ee80cd833f2b799550fe742d517ac`

Está en W&B (runs `tiiac-experto-cartpole` y `tiiac-datos-cartpole`) como los
artefactos `experto-CartPole-v1-92c44313863e` y `demostraciones-5c5168e7f5f8`.
`run.py --descargar` los baja y los comprueba. Para descargarlos sin ejecutar
nada más (sin comprobar las huellas, que comprobará `run.py` después):

```bash
wandb artifact get a-corral-universidad-de-deusto/tiiac-proyecto/demostraciones-5c5168e7f5f8:latest --root data/demostraciones/cartpole/congelado
wandb artifact get a-corral-universidad-de-deusto/tiiac-proyecto/experto-CartPole-v1-92c44313863e:latest --root checkpoints/expertos/cartpole/congelado
```

### Otros entornos

MountainCar, Acrobot, Reacher y HalfCheetah están congelados desde el 2026-11-01
(commit `a6b8bdb2bbfce379554813c477bfaf33ef82e412`, sin cambios sin commit) en
`data/demostraciones/<entorno>/congelado/` y `checkpoints/expertos/<entorno>/congelado/`.
Se generaron en Linux (WSL2) con Python 3.12.3, `OMP_NUM_THREADS=1`, los cuatro a la
vez y `--grupo congelado`. Sustituyen a los congelados el 2026-10-31 desde el commit
`6ba7aae`, que se retiraron al cambiar la configuración de su experto.

| Entorno | Objetivo del experto | Hiperparámetros del experto | Experto congelado | Partidas |
|---|---|---|---|---|
| MountainCar-v0 | −90, evaluado con 50 partidas | Sección `trpo` (paper y código original) | 1 000 000 pasos, sin alcanzar el objetivo | 15 (10 + 5), 1471 pasos |
| Acrobot-v1 | −70, evaluado con 50 partidas | Sección `trpo` (paper y código original) | 980 000 pasos, objetivo alcanzado | 15 (10 + 5), 1088 pasos |
| Reacher-v5 | −3,5 | RL Zoo para ReacherBulletEnv-v0, sin normalizar la recompensa (`experto.trpo`) | 25 001 984 pasos, sin alcanzar el objetivo | 30 (25 + 5), 1500 pasos |
| HalfCheetah-v5 | 5000 | RL Zoo para HalfCheetah-v4, sin normalizar la recompensa (`experto.trpo`) | 7 168 000 pasos, objetivo alcanzado | 30 (25 + 5), 30 000 pasos |

Los objetivos son elección propia, por encima del experto del paper (tabla 1) y
del umbral de Gymnasium (ver
[Decisiones](#decisiones-donde-el-paper-no-especifica)). Si no se alcanza, se
guarda el experto de la mejor evaluación.

Artefactos de W&B y huellas SHA-256 (`modelo.zip`, `demostraciones.npz`,
`splits.json` y contenido de las partidas):

| Entorno | Artefactos | Huellas |
|---|---|---|
| MountainCar-v0 | `experto-MountainCar-v0-d492bda5bd8f`, `demostraciones-86c31ceb8ace` | modelo `d492bda5bd8fa7ca9b6eb949153cbf395ef835c6c1853077b430f5e2d42e0dd4`; NPZ `86c31ceb8aceeaf7b300eaad20eba47927b49abfdb3039d4c4358c89814da364`; splits `dbda14e1a433358e71903f4fd071e80f6bd8184c18066c17619efcfd25be2828`; partidas `746797832e766391fcf0baec91211c82eb0817852cdfe0d074148febf700c1cc` |
| Acrobot-v1 | `experto-Acrobot-v1-fad2206a39f6`, `demostraciones-45531794f3b9` | modelo `fad2206a39f6ef55f9822d5d4bd8738b3e04dacd1d425002d74804fdc7ccdcf4`; NPZ `45531794f3b9781b9c25a117ab9c136b1b9a6606507572e97bcc3497c973594a`; splits `b81107d4f3450c6ba8710229ef1ab7ca2b9cde3126ea1d2de3a3c84a4e077dbb`; partidas `efad26895751d70b6104815b7ff2f1b418abf26a0a5fe9353725622494a27c80` |
| Reacher-v5 | `experto-Reacher-v5-a69d425ddb0f`, `demostraciones-1b16407e4fd8` | modelo `a69d425ddb0f4dad6c0d8fd42e96a9295f982dc43e9cad8674cf3ff73f5c22b2`; NPZ `1b16407e4fd8f3ce1da578ce75ca626962129b26201ed93afef8407869e1b1e0`; splits `d886173eea50dc03614944ce06bb355157f7998abe0bac7edf42da6403126f31`; partidas `b2f098f94c7962c2c758eaf57647349dd84fb8b85a497426238986c5a98f384f` |
| HalfCheetah-v5 | `experto-HalfCheetah-v5-331994138466`, `demostraciones-9a81e93f08fa` | modelo `3319941384669b19fb854a56bb70a65e3eba24a1c5740488b979e0c691759762`; NPZ `9a81e93f08fad6ccb6d2082b2ae365f32ba6e41a51bb8068ec534107769424f9`; splits `2e943144b0c5647d6794cc2c3c1b409a2162ea8499b025688fe9d2bd3c220c61`; partidas `8d9ba527575966f027a93cde2e6f287ebfd2f07f7832cb0315dfc165b5f1d123` |

Se descargan como los de CartPole, cambiando los nombres de los artefactos y las
carpetas, o con `run.py --entorno <entorno> --descargar`.

### Regenerar o congelar los datos

```bash
python scripts/entrenar_experto.py --entorno cartpole
python scripts/get_data.py --entorno cartpole --experto checkpoints/expertos/cartpole/<fecha> \
  --grupo <grupo> --huella-esperada b31d19eda39c1167b67330995e26659d0782b43ae83cfde7b773e8b1df221f58
```

`get_data.py` graba las partidas con el experto estocástico (en CartPole, 10 de
train y 5 de validación, separadas por trayectoria), guarda `demostraciones.npz`
y `splits.json` con los subconjuntos de train e imprime las huellas. Con
`--huella-esperada` comprueba si las partidas coinciden con las congeladas. Para
congelar un entorno se usan los mismos dos scripts desde un commit limpio, con
`--salida` en las carpetas `congelado` y `--grupo congelado`.

Comprobado el 2026-10-30: en la misma máquina, un experto reentrenado desde cero
graba exactamente las mismas partidas (el NPZ y los splits cambian de bytes porque
sus metadatos citan el experto, pero no la huella de las partidas). Las semillas
salen de `configs/`, pero en otra plataforma el redondeo puede cambiar las
acciones del experto y, con ellas, las partidas: los mismos datos generados en
Windows con Python 3.13 tenían otra huella. Por eso el experimento parte de los
datos congelados.

### Procedencia y licencias

Son datos propios, generados en los entornos de Gymnasium (licencia MIT). Reacher
y HalfCheetah usan además MuJoCo (licencia Apache 2.0); `gymnasium[mujoco]` y
`mujoco` están en `requirements.txt`.

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
4. Generar y congelar el experto y las demostraciones (ver
   [Regenerar o congelar los datos](#regenerar-o-congelar-los-datos)) y sustituir
   `pendiente` por sus rutas, huellas y artefactos.

Mientras quede algún `pendiente`, se pueden ejecutar el smoke y los scripts, pero
`run.py --entorno <nombre>` sale con código 2 y dice qué valor falta.

## Seguimiento W&B

El proyecto es privado y pertenece a la organización
`a-corral-universidad-de-deusto`. Para ver los runs o descargar los artefactos
hace falta una cuenta con acceso: pedidlo a los autores. Con una clave de otra
organización, W&B responde con un error 403.

La credencial se guarda fuera del repositorio (`wandb login`). Para trabajar
sin conexión, cambia `wandb.modo` a `offline` en la configuración y sincroniza
luego con `wandb sync`. Para no registrar un experimento, usa `--sin-wandb`.
Los tests y el smoke usan `disabled`.

| Grupo | Qué contiene |
|---|---|
| `congelado` | Los expertos y las demostraciones congelados |
| `entrega-e1` | El experimento completo de CartPole medido para la entrega |
| `ajuste-experto-halfcheetah`, `ajuste-experto-reacher` | La comparación de hiperparámetros del experto (ver [Ajuste del experto](#ajuste-del-experto)) |

## Reproducibilidad y regla 5/20

Las semillas están en `configs/` y solo se inicializan en `src/utils/seed.py`;
las dependencias tienen versiones exactas en `requirements.txt`. Cada
entrenamiento de BC y GAIL usa un solo hilo de PyTorch, así que en la misma
máquina da el mismo resultado en serie y en paralelo.

- **Smoke:** 5 s en CPU, muy por debajo de los 5 minutos.
- **Experimento completo de CartPole:** 7 min 13 s en CPU (ver
  [Tiempos medidos](#tiempos-medidos)), por debajo de los 20 minutos. TRPO, las
  políticas y el discriminador se ejecutan en CPU (`device="cpu"`), así que una
  GPU no acelera estos entrenamientos.
- **Otros entornos:** la regla se aplica al experimento de CartPole. MountainCar,
  Acrobot, Reacher y HalfCheetah usan los tamaños del código original y no la
  cumplen: el único barrido medido, el de GAIL en Reacher con datos de prueba,
  tardó 2 h 37 min.
- **Pendiente:** medirlo en una T4 de Colab gratuito o dejar aquí la
  justificación de por qué no se mide.

### Tiempos medidos

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

La ejecución de 7 min 13 s es el grupo `entrega-e1` de W&B (run de comparación
`r6oajx1r`). El *Runtime* que muestra W&B en una fila agrupada resume los runs
del grupo y no equivale al tiempo de la ejecución.

Mediciones anteriores, con otros datos congelados o con otra configuración:

| Medición | Tiempo | Hardware | Fecha |
|---|---|---|---|
| Expertos retirados de MountainCar (150 000 pasos, parada en el umbral de Gymnasium), Acrobot (100 000, ídem), Reacher (25 000 000, sin llegar al objetivo) y HalfCheetah (ídem) | 40 s, 34 s, 4503 s (75 min) y 5312 s (89 min) | Linux (WSL2), Intel Core i7-11800H, un experto a la vez | 2026-10-30/31 |
| Barrido GAIL de Reacher con datos de prueba (4 entrenamientos a la vez, 500 iteraciones de 50 000 pasos, sin W&B) | 9393 s (2 h 37 min), unos 18,7 s por iteración | Linux (WSL2), Intel Core i7-11800H, 4 procesos | 2026-10-30 |
| Experimento completo de CartPole con los datos congelados en Windows, ya sustituidos (grupo `4a2683ef47f9`) | 17 min 32 s (1051,8 s), de ellos 5 min 53 s de evaluación | Windows 11, AMD64 Family 23 Model 160, 8 procesadores lógicos, 5 procesos, sin GPU | 2026-10-28 |
| Barrido de CartPole con 80 iteraciones de GAIL y evaluación | unos 42 min | Windows 11, 8 procesadores lógicos | 2026-10-28 |
| Experto de CartPole (50 000 pasos, 1 entorno) | 53 s | Linux (WSL2), 12 núcleos | 2026-10-16 |
| Demostraciones de CartPole (15 partidas, 7401 pasos, 244 KB) | 6 s | Linux (WSL2), 12 núcleos | 2026-10-19 |
| Iteración de GAIL en CartPole con 10 entornos, un entrenamiento a la vez | 2,1–2,6 s | Linux (WSL2) | 2026-10-21/22 |
| Iteración de GAIL en CartPole con 10 entornos, 4 entrenamientos a la vez | 4,2 s | Linux (WSL2), 6 núcleos físicos, 7,9 GB | 2026-10-22 |

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

`experto.trpo` cambia claves de la sección `trpo` solo al entrenar el experto;
GAIL sigue con los valores del paper. En Reacher y HalfCheetah se compararon
variantes de esos hiperparámetros, de unos 5 M pasos cada una, en un grupo de W&B
propio por entorno. La elegida está en `experto.trpo` del perfil, con su fuente.

**HalfCheetah** (2026-10-31, grupo `ajuste-experto-halfcheetah`, con el objetivo
de entonces, 4800). Se compararon los hiperparámetros de TRPO de RL Zoo para
HalfCheetah-v4 (red 64 × 64, 2048 pasos por iteración, `target_kl` 0,04,
recompensa normalizada), tres variaciones suyas (red del paper, `target_kl` 0,01
y sin normalizar la recompensa) y la sección `trpo` del perfil con la recompensa
normalizada. El equipo eligió la de mejor retorno, RL Zoo sin normalizar la
recompensa, que paró al alcanzar 4800 a los 2,87 M pasos. Después se entrenó esa
configuración con su red de 64 × 64 y con la del paper, 100 × 100 (runs del mismo
grupo, objetivo 5000, unos 5 M pasos cada uno), y el equipo se quedó con 64 × 64.

**Reacher** (2026-10-31, grupo `ajuste-experto-reacher`, con el objetivo de
entonces, −3,75). RL Zoo solo tiene TRPO para `ReacherBulletEnv-v0` (PyBullet, no
MuJoCo), así que sus valores eran un punto de partida, no un ajuste para este
entorno. Se compararon:

| Variante | Qué cambia |
|---|---|
| `rl_zoo` | Hiperparámetros de TRPO de RL Zoo para ReacherBulletEnv-v0 (red 256 × 256 ReLU, `log_std` inicial −1, sin inicialización ortogonal, 2048 pasos por iteración, recompensa normalizada…) |
| `rl_zoo_red_100` | `rl_zoo` con la red del paper (100 × 100 tanh) |
| `rl_zoo_mujoco` | Los valores comunes de RL Zoo sin los propios de ReacherBulletEnv-v0 (red 64 × 64 tanh) |
| `rl_zoo_sin_norm_recompensa` | `rl_zoo` sin normalizar la recompensa |
| `actual_norm_recompensa` | La sección `trpo` del perfil con la recompensa normalizada |

El equipo eligió `rl_zoo_sin_norm_recompensa`, la única que alcanzó −3,75, en su
última iteración (5 M pasos).

## Estructura del repositorio

```text
run.py                  punto de entrada: smoke, experimento y reevaluación
configs/                toda la configuración: default.yaml, perfil de cada entorno y smoke.yaml
src/data/               entornos, demostraciones, splits por trayectoria y loaders
src/models/factory.py   política de BC y discriminador
src/training/           TRPO y experto, BC, GAIL, barridos, evaluación y flujo de run.py
src/utils/              configuración, semillas, seguimiento W&B, archivos y validación
src/app/                esqueleto del aplicativo (E2)
scripts/                cada etapa por separado
tests/                  tests (pytest)
tools/tiiac_check.py    comprobador de la asignatura
docs/                   propuesta y presentación
```

## Uso de IA

Las conversaciones del proyecto se registran en [`AI_USAGE.md`](AI_USAGE.md).
