# TIIAC: GAIL y Behavioral Cloning

Replicación educativa de **Generative Adversarial Imitation Learning** (Ho y
Ermon, 2016), comparando GAIL y Behavioral Cloning con un experto TRPO y una
política aleatoria. El primer entorno implementado es CartPole.

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

El checker ya ejecuta el smoke; el último comando sirve para verlo por separado.

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
local), comprueba sus huellas, entrena 20 BC y 20 GAIL (subconjuntos de 1, 4, 7 y 10 partidas,
5 repeticiones), evalúa los 42 agentes (experto, aleatorio, BC y GAIL) con
50 episodios cada uno e imprime una línea de métricas por agente.

### Comandos estándar

| Para qué | Comando |
|---|---|
| Smoke | `python run.py --smoke` |
| Experimento completo | `python run.py --entorno cartpole --descargar --grupo <nombre>` |

## Argumentos de `run.py`

| Argumento | Para qué sirve | Por defecto |
|---|---|---|
| `--entorno cartpole` | Perfil de entorno (`configs/entornos/<entorno>.yaml`). De momento solo existe `cartpole`. | `cartpole` |
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
(`echo $?`). Los números no significan nada. Tarda unos 5 s (ver
[Tiempos medidos](#tiempos-medidos)); para medirlo, `time python run.py --smoke`.

## Qué produce un experimento

En disco, en `checkpoints/run/cartpole/<fecha>/` (o en `--salida`), que no se
versiona:

```text
bc/cantidad_<c>/repeticion_<r>.pt          política de BC con sus metadatos
gail/cantidad_<c>/repeticion_<r>/
    modelo.zip                             agente TRPO de GAIL
    discriminador.pt                       pesos y normalización del discriminador
    gail.json                              configuración, resumen, procedencia y hardware
evaluacion.json                            protocolo y retornos de los 42 agentes
```

En W&B, cada ejecución forma un grupo de 41 runs: 20
`tiiac-bc-cartpole-c<c>-r<r>`, 20 `tiiac-gail-cartpole-c<c>-r<r>` y
`tiiac-comparacion-cartpole`, que aparece al terminar la evaluación con las
tablas `comparacion` y `episodios`. Las métricas se agrupan en secciones por
prefijo: `bc/`, `trpo/`, `discriminador/`, `reajuste/` y `validacion/`.

Para ver una ejecución: en *Runs*, **Filter** → `Group` = `<nombre>`. Para
comparar curvas, **Group** por `barrido.metodo` o `barrido.cantidad`, que cada
run de BC y GAIL lleva en su configuración.

## Datos

Las demostraciones y el experto no se versionan. Están congelados desde el
2026-10-30 (commit `15416db8bdec494f55bba3e2ff527907bdc6fafc`) en
`data/demostraciones/cartpole/congelado/` y
`checkpoints/expertos/cartpole/congelado/`. Se generaron en Linux (WSL2) con
Python 3.12.3 y las versiones de `requirements.txt`. Sus huellas SHA-256 son:

- `demostraciones.npz`: `5c5168e7f5f8ad8dc415ff8e13ff0acc3d9fe8e0445e4f9d0298f0817198aff9`
- `splits.json`: `804bb82f4c5b16ce4a49e97bde07911d1af1687a83a33cafabe7e9442f52a9d3`
- contenido de las partidas: `b31d19eda39c1167b67330995e26659d0782b43ae83cfde7b773e8b1df221f58`
- `modelo.zip` del experto: `92c44313863ebe30c616be30664b2742650ee80cd833f2b799550fe742d517ac`

Se publicaron en W&B (grupo `congelado`, runs `tiiac-experto-cartpole` y
`tiiac-datos-cartpole`) como los artefactos `experto-CartPole-v1-92c44313863e`
y `demostraciones-5c5168e7f5f8`. `run.py --descargar` los baja y comprueba.
Para descargarlos sin ejecutar nada más (sin comprobar las huellas, que
comprobará `run.py` después):

```bash
wandb artifact get a-corral-universidad-de-deusto/tiiac-proyecto/demostraciones-5c5168e7f5f8:latest --root data/demostraciones/cartpole/congelado
wandb artifact get a-corral-universidad-de-deusto/tiiac-proyecto/experto-CartPole-v1-92c44313863e:latest --root checkpoints/expertos/cartpole/congelado
```

Para regenerarlos, el procedimiento reproducible es:

```bash
python scripts/entrenar_experto.py --entorno cartpole
python scripts/get_data.py --entorno cartpole --experto checkpoints/expertos/cartpole/<fecha> \
  --grupo <grupo> --huella-esperada b31d19eda39c1167b67330995e26659d0782b43ae83cfde7b773e8b1df221f58
```

`get_data.py` graba 15 partidas con el experto estocástico (10 de train y 5 de
validación, separadas por trayectoria), guarda `demostraciones.npz` y
`splits.json` con subconjuntos de train de 1, 4, 7 y 10 partidas e imprime las
huellas. Con `--huella-esperada` comprueba si las partidas coinciden con las
congeladas. Comprobado el 2026-10-30: en la misma máquina, un experto
reentrenado desde cero graba exactamente las mismas partidas (el NPZ y los
splits cambian de bytes porque sus metadatos citan el experto, pero no la huella
de las partidas). Las semillas salen de `configs/`, pero en otra plataforma el
redondeo puede cambiar las acciones del experto y, con ellas, las partidas: los
mismos datos generados en Windows con Python 3.13 tenían otra huella. Si las
huellas no coinciden, `run.py` rechaza esos datos; por eso el experimento parte
de los congelados.

Procedencia: datos propios generados en `CartPole-v1` de Gymnasium (licencia
MIT). `configs/entornos/cartpole.yaml` (sección `datos`) fija las carpetas, las
huellas y los artefactos congelados.

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

## Seguimiento W&B

El proyecto es privado y pertenece a la organización
`a-corral-universidad-de-deusto`. Para ver los runs o descargar los artefactos
hace falta una cuenta con acceso: pedidlo a los autores. Con una clave de otra
organización, W&B responde con un error 403.

La credencial se guarda fuera del repositorio (`wandb login`). Para trabajar
sin conexión, cambia `wandb.modo` a `offline` en la configuración y sincroniza
luego con `wandb sync`. Para no registrar un experimento, usa `--sin-wandb`.
Los tests y el smoke usan `disabled`.

## Tiempos medidos

`experimento.procesos` (en `configs/default.yaml`) fija cuántos entrenamientos
se ejecutan a la vez. Está en 5, elegido para un PC de 6 núcleos físicos. En
otra máquina conviene ponerlo como mucho en su número de núcleos físicos,
contando unos 830 MB de RAM por proceso. No cambia los resultados, solo el
tiempo.

Con los datos congelados y la configuración actual:

| Medición | Tiempo | Hardware | Fecha |
|---|---|---|---|
| Smoke (`run.py --smoke`, dentro del checker) | 5 s | Linux (WSL2), Intel Core i7-11800H, 16 procesadores lógicos, sin GPU | 2026-10-30 |
| Experimento completo (`time python run.py --entorno cartpole --grupo entrega-e1 --descargar`, online): 20 BC, 20 GAIL de 20 iteraciones y evaluación | 7 min 13 s, de ellos 67 s de evaluación (resumen de W&B) | Linux (WSL2), Intel Core i7-11800H, 16 procesadores lógicos, 5 procesos, sin GPU | 2026-10-30 |
| Experto congelado (60 000 pasos, parada al llegar a 500) | 14 s | Linux (WSL2), Intel Core i7-11800H, 16 procesadores lógicos, sin GPU | 2026-10-30 |
| Demostraciones congeladas (15 partidas, 7500 pasos) | 1,7 s | Linux (WSL2), misma máquina | 2026-10-30 |

La ejecución de 7 min 13 s es el grupo `entrega-e1` de W&B (run de comparación
`r6oajx1r`). El *Runtime* que muestra W&B en una fila agrupada resume los runs
del grupo y no equivale al tiempo de la ejecución.

Mediciones anteriores, con otros datos congelados o con otra configuración:

| Medición | Tiempo | Hardware | Fecha |
|---|---|---|---|
| Experimento completo con los datos congelados en Windows, ya sustituidos (grupo `4a2683ef47f9`) | 17 min 32 s (1051,8 s), de ellos 5 min 53 s de evaluación | Windows 11, AMD64 Family 23 Model 160, 8 procesadores lógicos, 5 procesos, sin GPU | 2026-10-28 |
| Barrido con 80 iteraciones de GAIL y evaluación | unos 42 min | Windows 11, 8 procesadores lógicos | 2026-10-28 |
| Experto (50 000 pasos, 1 entorno) | 53 s | Linux (WSL2), 12 núcleos | 2026-10-16 |
| Demostraciones (15 partidas, 7401 pasos, 244 KB) | 6 s | Linux (WSL2), 12 núcleos | 2026-10-19 |
| Iteración de GAIL con 10 entornos, un entrenamiento a la vez | 2,1–2,6 s | Linux (WSL2) | 2026-10-21/22 |
| Iteración de GAIL con 10 entornos, 4 entrenamientos a la vez | 4,2 s | Linux (WSL2), 6 núcleos físicos, 7,9 GB | 2026-10-22 |

## Reproducibilidad y regla 5/20

Las semillas están en `configs/` y solo se inicializan en `src/utils/seed.py`;
las dependencias tienen versiones exactas en `requirements.txt`. Cada
entrenamiento usa un solo hilo de PyTorch, así que en la misma máquina da el
mismo resultado en serie y en paralelo.

- **Smoke:** 5 s en CPU, muy por debajo de los 5 minutos.
- **Experimento completo:** 7 min 13 s en CPU con los datos actuales (tabla
  anterior), por debajo de los 20 minutos. TRPO,
  las políticas y el discriminador se ejecutan en CPU (`device="cpu"`), así que
  una GPU no acelera estos entrenamientos.
- **Pendiente:** medirlo en una T4 de Colab gratuito o dejar aquí la
  justificación de por qué no se mide.

## Decisiones donde el paper no especifica

| Ambigüedad o diferencia | Qué decidimos | Por qué |
|---|---|---|
| Iteraciones de GAIL (paper: 300) | 20 en CartPole | Presupuesto de la regla 5/20 |
| Repeticiones (código original: 7) | 5 | Repartir el barrido en 5 procesos sin compartir núcleo |
| Entornos por iteración de TRPO | 10 entornos de 500 pasos (mismos 5000 pasos) | Unas 2,7 veces más rápido, medido |
| Objetivos de la red de valor tras actualizar D | Retornos GAE, no Monte Carlo | Coherente con el resto de TRPO en sb3 |
| Ajuste del crítico | Adam por minibatches, no región de confianza | Implementación de sb3-contrib |
| Elección del experto | El mejor por evaluación, parando al llegar a 500 | El original usa un snapshot fijo |
| Normalización de la entrada del discriminador | Fija con los pares expertos de train | El original la actualiza también con pares del agente |
| Validación de partidas (paper: 70/30 de pares) | 10 partidas de train y 5 de validación, separadas por trayectoria | Evitar fugas entre episodios |
| Parada de BC | Paciencia de 20 épocas sobre la pérdida de validación | El paper solo dice «hasta que deja de bajar» |

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
