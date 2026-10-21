<!--
PLANTILLA. Sustituye TODO lo que va entre <angulos>. Borra este comentario.

Un README se lee en treinta segundos o no se lee. Quien lo abre quiere saber
tres cosas: qué es esto, cómo lo ejecuto, y funcionó o no. En ese orden.

Recuerda que en noviembre otra pareja va a clonar este repositorio y va a
intentar reproducir una métrica SIN PREGUNTAROS NADA. Este fichero es lo único
que van a tener.
-->

# <Título del proyecto>

Replicación de **<Autores, año · Título del paper>** sobre **<dataset>**,
con <aplicativo: buscador semántico / detector en vídeo / lo que sea> construido encima.

- **Paper:** <enlace a arXiv o DOI>
- **Implementación de referencia:** <enlace, o «ninguna: el paper se implementa desde cero»>
- **Seguimiento de experimentos:** <enlace al proyecto de W&B — obligatorio>
- **Autoría:** <Nombre 1>, <Nombre 2>

## Resultado

| Métrica | Paper | Nuestra réplica | Escala |
|---|---|---|---|
| <accuracy> | <0.72> | <0.68> | <CIFAR-10, subset de 5.000> |

<Una frase honesta sobre la diferencia. Una desviación bien explicada puntúa
igual que un acierto; una desviación sin explicar, no.>

## Instalación

```bash
git clone <url>
cd <repo>
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Datos

Las demostraciones no se descargan: se graban con nuestro experto TRPO, y ni
el experto ni los datos se versionan. Se obtienen con:

```bash
python scripts/entrenar_experto.py --entorno cartpole
python scripts/get_data.py --entorno cartpole --experto checkpoints/expertos/cartpole/<fecha>
```

`get_data.py` graba 15 partidas con el experto estocástico, como el código
original, sin filtrarlas. Escribe `demostraciones.npz` y `splits.json` (10
partidas de train con subconjuntos 1, 4, 7 y 10, y 5 de validación) en
`data/demostraciones/cartpole/<fecha>` e imprime las huellas SHA-256. Las
semillas salen de `demostraciones.semilla` en `configs/default.yaml`. Con
`--huella-esperada <huella de las partidas>` comprueba que una regeneración da
las mismas partidas. Medido el 19 de octubre en CPU (WSL2, 12 núcleos):
CartPole, 7401 pasos y 244 KB en 6 s.

Procedencia: datos propios, generados en `CartPole-v1` de Gymnasium (licencia
MIT). Datos congelados: pendiente de acordar en el equipo; cuando se congelen,
aquí irán su ubicación y sus huellas.

## Ejecución

```bash
# Experto TRPO (no se versiona): se regenera con la misma semilla de la config.
# Medido: 53 s en CPU para CartPole. --sin-wandb si no hay cuenta de W&B.
python scripts/entrenar_experto.py --entorno cartpole

# GAIL (no se versiona): un entrenamiento por tamaño de subconjunto y repetición,
# guardados en checkpoints/gail/<entorno>/<fecha>. --cantidades y --repeticiones
# eligen una parte del barrido para repartirlo en varias sesiones.
# Medido: unos 6 s por iteración en CPU para CartPole; presupuesto pendiente.
python scripts/entrenar_gail.py --entorno cartpole --datos data/demostraciones/cartpole/<fecha>

# Smoke test: el pipeline entero en miniatura. Menos de 5 minutos en CPU.
python run.py --smoke --entorno cartpole

# Entrenamiento completo. Menos de 20 minutos en una T4 de Colab gratuito.
python run.py --config configs/default.yaml --entorno cartpole

# El aplicativo, sobre el checkpoint entrenado.
python -m src.app.serve --checkpoint checkpoints/mejor.pt
```

## Estructura

```
configs/      Todos los hiperparámetros. Ninguno escrito a fuego en el código
src/data/     Carga, splits y transformaciones
src/models/   La arquitectura replicada
src/training/ Bucle de entrenamiento y evaluación
src/app/      La capa de servicio: carga un checkpoint y responde
scripts/      Descarga de datos y utilidades
tests/        Tests de forma, invariancia y reproducibilidad
run.py        Punto de entrada único
```

## Reproducibilidad

- Semilla fijada en `src/utils/seed.py`, valor en `configs/`.
- Dependencias con versión exacta en `requirements.txt`.
- Presupuesto de cómputo declarado: **<tiempo real> en <hardware>**.
- Qué se recortó respecto al paper y qué se pierde con ello: <explicación>.

## Decisiones donde el paper no especifica

<La sección que distingue replicar de copiar. Lista lo que tuvisteis que
decidir vosotros y con qué criterio. Sale de la ficha de replicación.>

| Ambigüedad del paper | Qué decidimos | Por qué |
|---|---|---|
| <no dice el schedule de lr> | <cosine con warmup de 5 épocas> | <es lo habitual en la familia de trabajos citada> |

## Tests

```bash
pytest tests/ -q
```

## Uso de IA

Documentado en [`AI_USAGE.md`](AI_USAGE.md), según la guía docente.
