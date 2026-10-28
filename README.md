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

Los runs W&B del grupo `f24995f3d703` son `tiiac-experto-cartpole` y
`tiiac-datos-cartpole`; los artefactos son `experto-CartPole-v1-4c5f7ad00efd`
y `demostraciones-ce06866eff92`. En otra máquina se pueden descargar con:

```bash
wandb artifact get a-corral-universidad-de-deusto/tiiac-proyecto/demostraciones-ce06866eff92:latest --root data/demostraciones/cartpole/congelado
wandb artifact get a-corral-universidad-de-deusto/tiiac-proyecto/experto-CartPole-v1-4c5f7ad00efd:latest --root checkpoints/expertos/cartpole/congelado
```

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
- Barrido completo online con 20 iteraciones de GAIL: 14 min 08 s desde el
  inicio del flujo hasta acabar los entrenamientos BC/GAIL; 20 min 56 s de
  pared incluyendo la evaluación final común de 50 episodios. Su resumen W&B
  registra 293 s de evaluación. Medido el 28 de octubre en la misma CPU, con
  5 procesos, sin GPU. W&B sincronizó los 20 BC, 20 GAIL y la comparación.
  Grupo W&B `085dfc3fb185`; comparación `9m0ruxv4`.
  La evaluación se informa aparte porque no forma parte del entrenamiento de
  los modelos. El entrenamiento quedó bajo 20 minutos; el flujo con evaluación
  superó el límite por 56 s.
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
límite de 5 minutos en CPU. El entrenamiento BC/GAIL medido cumple el límite de
20 minutos en CPU; la evaluación final de 50 episodios registró 293 s en W&B. TRPO usa
CPU, así que una T4 no acelera estos entrenamientos.

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
