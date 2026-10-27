# TIIAC: GAIL y Behavioral Cloning

Replicación educativa de **Generative Adversarial Imitation Learning** (Ho y
Ermon, 2016), comparando GAIL y Behavioral Cloning con un experto TRPO y una
política aleatoria. El primer entorno implementado es CartPole.

- Paper: [Generative Adversarial Imitation Learning](https://arxiv.org/abs/1606.03476)
- Seguimiento de experimentos: [proyecto W&B `tiiac-proyecto`](https://wandb.ai/a-corral-universidad-de-deusto/tiiac-proyecto)
- Autores: Asier Corral e Ibon Erdoiza.

## Resultado

Pendiente de completar por el equipo.

## Instalación

Se usa Python 3.12 y las versiones fijadas en `requirements.txt`:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Datos

Las demostraciones y el experto no se versionan. En CartPole, el destino acordado
para congelarlos es `checkpoints/expertos/cartpole/congelado/` y
`data/demostraciones/cartpole/congelado/`. Con los nombres exactos de los
artefactos se pueden recuperar en otra máquina con:

```bash
wandb artifact get a-corral-universidad-de-deusto/tiiac-proyecto/demostraciones-<12-caracteres>:latest --root data/demostraciones/cartpole/congelado
wandb artifact get a-corral-universidad-de-deusto/tiiac-proyecto/experto-CartPole-v1-<12-caracteres>:latest --root checkpoints/expertos/cartpole/congelado
```

Los nombres concretos de los artefactos y las huellas SHA-256 del NPZ, los
splits y las partidas se anotarán aquí al comprobar la publicación online. En
este checkout no están disponibles esos archivos ni sus huellas; no se deben
tratar las rutas como congeladas hasta completar esa comprobación. Para
regenerarlos o contrastarlos con una sesión de W&B activa:

```bash
python scripts/entrenar_experto.py --entorno cartpole
python scripts/get_data.py --entorno cartpole \
  --experto checkpoints/expertos/cartpole/<fecha> --grupo <grupo>
python scripts/get_data.py --entorno cartpole \
  --experto checkpoints/expertos/cartpole/<fecha> --grupo <grupo> \
  --huella-esperada <huella-de-las-partidas>
```

`get_data.py` graba 15 partidas con el experto estocástico: 10 para train y 5
para validación. Guarda `demostraciones.npz` y `splits.json` en la carpeta
nueva, con subconjuntos train de 1, 4, 7 y 10 partidas, e imprime las huellas
SHA-256 del NPZ, de los splits y del contenido de las partidas. La semilla se
lee de `configs/`. Con `--huella-esperada <huella-de-las-partidas>` comprueba
que la regeneración produce las mismas partidas y muestra las huellas obtenidas
aunque no coincidan.

Procedencia: datos propios generados en `CartPole-v1` de Gymnasium (licencia
MIT). `configs/entornos/cartpole.yaml` fija las carpetas de congelación y la
huella de las partidas cuando se hayan generado y verificado.

## Recorrido experimental

El experto, las demostraciones y los modelos se guardan localmente y no se
versionan. El experto imprime un identificador de grupo W&B; pásalo a los pasos
siguientes para reunir los runs:

```bash
python scripts/entrenar_experto.py --entorno cartpole
python scripts/get_data.py --entorno cartpole --experto checkpoints/expertos/cartpole/<fecha> --grupo <grupo>
python scripts/entrenar_bc.py --entorno cartpole --datos data/demostraciones/cartpole/<fecha> --grupo <grupo>
python scripts/entrenar_gail.py --entorno cartpole --datos data/demostraciones/cartpole/<fecha> --grupo <grupo>
python scripts/evaluar.py --entorno cartpole --experto checkpoints/expertos/cartpole/<fecha> --bc checkpoints/bc/cartpole/<fecha> --gail checkpoints/gail/cartpole/<fecha> --grupo <grupo>
```

El punto de entrada completo es `python run.py --entorno cartpole`; requiere
las carpetas congeladas anteriores. `python run.py --smoke` ejecuta el flujo
real en miniatura con un experto y datos temporales, sin escribir checkpoints
ni datos en el repositorio. El smoke terminó en 10 s dentro del checker local
de Python 3.13 en Windows, usando CPU; el modelo de CPU no estuvo disponible
para identificarlo en esta sesión. Los scripts individuales siguen disponibles
para ejecutar etapas por separado.

Los barridos aceptan `--cantidades` y `--repeticiones` para repartirlos entre
sesiones. `experimento.procesos` limita cuántos entrenamientos se ejecutan a la
vez. El seguimiento guarda runs, configuración, procedencia, resúmenes y
artefactos; la evaluación registra las tablas de comparación y de episodios.

## Seguimiento W&B

El proyecto se ha elegido privado y usará la entidad compartida
`a-corral-universidad-de-deusto`. Inicia sesión en W&B Cloud con una clave de
[wandb.ai/authorize](https://wandb.ai/authorize):

```bash
wandb login --relogin --host https://api.wandb.ai
```

La clave tiene que ser de una cuenta con acceso a la organización
`a-corral-universidad-de-deusto`; con una clave de otra organización, W&B
responde con un error 403 al abrir el run.

La credencial se guarda fuera del repositorio. Para trabajar sin conexión,
cambia `wandb.modo` a `offline` en la configuración y sincroniza luego con
`wandb sync`. Para no registrar un experimento, usa `--sin-wandb`. Tests y smoke
usan `disabled`.

## Tiempos medidos

- Experto CartPole: 53 s en CPU, WSL2 con 12 núcleos, para 50 000 pasos; medido
  el 16 de octubre.
- Datos CartPole: 6 s en CPU, WSL2 con 12 núcleos, para 15 partidas, 7401 pasos
  y 244 KB; medido el 19 de octubre.
- GAIL CartPole con 10 entornos: 2,1–2,6 s por iteración con un entrenamiento
  a la vez; con 4 entrenamientos simultáneos, 4,2 s por iteración. Medido en
  CPU; la segunda medición fue en el PC de 6 núcleos físicos y 7,9 GB de RAM
  disponible en WSL2.
- `experimento.procesos` está en 5 para ese PC de 6 núcleos físicos.
- El tiempo del barrido completo sigue pendiente de medir con las carpetas
  congeladas y una sesión de W&B Cloud disponible.

## Reproducibilidad (presupuesto)

Las semillas están en `configs/` y se inicializan en `src/utils/seed.py`; las
dependencias tienen versiones exactas en `requirements.txt`. El smoke cumple el
límite de 5 minutos en CPU. El barrido completo está pendiente de medir; TRPO
usa CPU, así que una T4 no acelera estos entrenamientos.

## Decisiones donde el paper no especifica

Pendiente de documentar por el equipo.

## Comprobaciones

```bash
python run.py --smoke --entorno cartpole
python -m pytest tests/ -q
python tools/tiiac_check.py .
```

El smoke usa W&B desactivado. El entrenamiento completo no forma parte de las
comprobaciones rutinarias.

## Uso de IA

Las conversaciones del proyecto se registran en [`AI_USAGE.md`](AI_USAGE.md).
