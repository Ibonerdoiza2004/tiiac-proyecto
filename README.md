# TIIAC: GAIL y Behavioral Cloning

Replicación educativa de **Generative Adversarial Imitation Learning** (Ho y
Ermon, 2016), comparando GAIL y Behavioral Cloning con un experto TRPO y una
política aleatoria. El primer entorno implementado es CartPole.

- Paper: [Generative Adversarial Imitation Learning](https://arxiv.org/abs/1606.03476)
- Seguimiento de experimentos: proyecto W&B `tiiac-2026` (enlace público pendiente de que
  el equipo comparta la URL).
- Autores: Asier Corral e Ibon Erdoiza.

## Instalación

Se usa Python 3.12 y las versiones fijadas en `requirements.txt`:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Recorrido experimental

El experto, las demostraciones y los modelos se guardan localmente y no se versionan.
Cada paso imprime un identificador de grupo W&B. Pásalo al siguiente comando con
`--grupo` para reunir la ejecución completa en un grupo:

```bash
python scripts/entrenar_experto.py --entorno cartpole
python scripts/get_data.py --entorno cartpole --experto checkpoints/expertos/cartpole/<fecha> --grupo <grupo>
python scripts/entrenar_bc.py --entorno cartpole --datos data/demostraciones/cartpole/<fecha> --grupo <grupo>
python scripts/entrenar_gail.py --entorno cartpole --datos data/demostraciones/cartpole/<fecha> --grupo <grupo>
python scripts/evaluar.py --entorno cartpole --experto checkpoints/expertos/cartpole/<fecha> --bc checkpoints/bc/cartpole/<fecha> --gail checkpoints/gail/cartpole/<fecha> --grupo <grupo>
```

Los barridos aceptan `--cantidades` y `--repeticiones` para repartirlos entre sesiones.
`experimento.procesos` en la configuración limita los entrenamientos simultáneos.
La configuración por defecto registra runs en W&B online. En el primer uso, inicia
sesión en una terminal con `wandb login`; la credencial se guarda fuera del repositorio.

Para trabajar sin conexión, cambia `wandb.modo` a `offline` en la configuración; W&B
guardará los runs localmente y se pueden sincronizar después con `wandb sync`. Para una
ejecución sin seguimiento, añade `--sin-wandb` al comando correspondiente. Los tests y
el smoke usan `disabled`.

El sistema registra runs por tarea (`experto`, `datos`, `bc`, `gail` y `comparacion`),
su configuración, procedencia, resumen, etiquetas y artefactos versionados. La tabla
de comparación y los retornos por episodio también quedan en el JSON local de evaluación.
Los nombres de métricas siguen siendo los actuales hasta acordarlos con el equipo.

## Comprobaciones

```bash
python run.py --smoke --entorno cartpole
python -m pytest tests/ -q
python tools/tiiac_check.py .
```

El smoke se ejecuta con W&B desactivado. El entrenamiento completo no forma parte de
las comprobaciones rutinarias.

## Uso de IA

Las conversaciones del proyecto se registran en [`AI_USAGE.md`](AI_USAGE.md).
