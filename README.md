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

Las demostraciones y el experto no se versionan. Están congelados en
`checkpoints/expertos/cartpole/congelado/` y
`data/demostraciones/cartpole/congelado/`, el 2026-10-28 con el commit
`2a9be76305d0ac83009261f3abf8de3a9718a189`. Las huellas SHA-256 son:

- `demostraciones.npz`: `ce06866eff929b74d4a377fb9262d7d977a387636542777d14cb6addd9fb204b`
- `splits.json`: `f6ae1c547e7d08e69d4622cacc069027349b41ec435c7529706e28b46933f910`
- contenido de las partidas: `84dd2815e125ac677e764fef55e8cfb2ed1bddbf68b8c5e773b75bbb0f22fd73`
- `modelo.zip` del experto: `4c5f7ad00efdbf97bba1b79d934b572df40ddee48d0f732b23a9d78ad4582195`

Los runs W&B del grupo `f24995f3d703` son `tiiac-experto-cartpole` y
`tiiac-datos-cartpole`; los artefactos son `experto-CartPole-v1-4c5f7ad00efd`
y `demostraciones-ce06866eff92`. En otra máquina se pueden descargar con:

```bash
python run.py --entorno cartpole --descargar
# También se pueden descargar individualmente:
wandb artifact get a-corral-universidad-de-deusto/tiiac-proyecto/demostraciones-ce06866eff92:latest --root data/demostraciones/cartpole/congelado
wandb artifact get a-corral-universidad-de-deusto/tiiac-proyecto/experto-CartPole-v1-4c5f7ad00efd:latest --root checkpoints/expertos/cartpole/congelado
```

`--descargar` baja solo las carpetas ausentes o vacías, usando la sesión W&B de
la organización configurada, y comprueba las huellas antes de entrenar. Nunca
sobrescribe una carpeta que ya contiene archivos.

Para regenerarlos o contrastarlos con una sesión de W&B activa:

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
MIT). `configs/entornos/cartpole.yaml` fija las carpetas congeladas y la huella
de las partidas.

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

- Experto congelado: 41 s para 60 000 pasos, `CartPole-v1`, el 28 de octubre
  de 2026, CPU en Windows 11 (AMD64 Family 23 Model 160, 8 procesadores
  lógicos; GPU no usada). El retorno medio de parada fue 500 en 10 episodios.
- Datos congelados: 4,84 s para 15 partidas y 7500 pasos, en la misma máquina.
- Flujo completo online con 20 iteraciones de GAIL: 1051,8 s (17 min 32 s),
  medidos con `run.py --entorno cartpole` el 28 de octubre de 2026. BC/GAIL y
  carga de los agentes ocuparon 11 min 39 s hasta iniciar la evaluación; la
  fase final paralela, con 50 episodios por agente, tomó 5 min 53 s de pared.
  El resumen W&B registra 244,8 s dentro de la evaluación. CPU Windows 11,
  AMD64 Family 23 Model 160, 8 procesadores lógicos, 5 procesos, sin GPU.
  W&B sincronizó los 20 BC, 20 GAIL y la comparación; grupo `4a2683ef47f9`,
  run de comparación `pevcn0t1`.
- El barrido previo de 80 iteraciones también se completó online: unos 42 min
  incluyendo la comparación de 50 episodios. Sirvió para medir el presupuesto;
  el perfil CartPole queda en 20 iteraciones.

- Experto CartPole: 53 s en CPU, WSL2 con 12 núcleos, para 50 000 pasos; medido
  el 16 de octubre.
- Datos CartPole: 6 s en CPU, WSL2 con 12 núcleos, para 15 partidas, 7401 pasos
  y 244 KB; medido el 19 de octubre.
- GAIL CartPole con 10 entornos: 2,1–2,6 s por iteración con un entrenamiento
  a la vez; con 4 entrenamientos simultáneos, 4,2 s por iteración. Medido en
  CPU; la segunda medición fue en el PC de 6 núcleos físicos y 7,9 GB de RAM
  disponible en WSL2.
- `experimento.procesos` está en 5 para ese PC de 6 núcleos físicos.

## Reproducibilidad (presupuesto)

Las semillas están en `configs/` y se inicializan en `src/utils/seed.py`; las
dependencias tienen versiones exactas en `requirements.txt`. El smoke cumple el
límite de 5 minutos en CPU. El flujo completo de CartPole, incluida la evaluación
paralela de 50 episodios, tardó 17 min 32 s en esta CPU, dentro de la regla 5/20.
TRPO usa CPU, así que una T4 no acelera estos entrenamientos.

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
