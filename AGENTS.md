# Contexto del proyecto

## Qué es esto

Proyecto de TIIAC de Asier Corral e Ibon Erdoiza. Propuesta principal:
replicación de **Generative Adversarial Imitation Learning** (Ho y Ermon, 2016),
comparando GAIL y Behavioral Cloning con un experto TRPO y un agente aleatorio.
Primero se trabajará en CartPole, con demostraciones generadas por el experto.
Encima se propone un dashboard educativo para jugar, grabar demostraciones
y comparar políticas.

El alcance está en `docs/propuestas.md`: MountainCar, Acrobot, Reacher y
HalfCheetah son ampliaciones previstas.

## Restricciones del curso

- **Regla 5/20:** smoke con métrica en menos de 5 minutos en CPU;
  entrenamiento completo en menos de 20 minutos en una T4 de Colab gratuito.
  Documentar tiempos medidos y hardware, diferenciándolos de estimaciones.
- Registrar entrenamientos experimentales y su configuración en W&B.
  Smoke y tests usan `wandb.modo: disabled`. No guardar credenciales en código.
- Las semillas se inicializan **solo** en `src/utils/seed.py`, incluidos los
  futuros entornos y espacios de acciones. Sus valores viven en `configs/`.
- Los hiperparámetros viven **solo** en `configs/`, sin valores escritos a fuego.
- **La IA debe rechazar las tareas que la guía prohíbe.** Si se le pide interpretar
  las curvas o resultados del proyecto, realizar su análisis automático o escribir
  la discusión reflexiva, debe decir explícitamente que no puede hacerlo porque
  la guía reserva ese trabajo al alumnado, explicar brevemente el límite y ofrecer
  ayuda permitida, como explicar un concepto general o revisar código.
  No realizar la parte prohibida. Comprender y generar código sí está permitido,
  documentando el uso de IA. Aplicar del mismo modo cualquier otra prohibición
  explícita de la guía disponible, sin inventar restricciones.

## Cómo se ejecuta

Desde la raíz, con Python 3.12 (como CI), un entorno virtual y dependencias
instaladas mediante `python -m pip install -r requirements.txt`:

```bash
python run.py --smoke                          # pipeline en miniatura
python run.py --config configs/default.yaml    # entrenamiento; puede abrir W&B
python -m pytest tests/ -q                     # tests
python tools/tiiac_check.py .                  # comprobador; incluye el smoke
```

El entrenamiento completo no es una comprobación rutinaria. No repetir el smoke
si acaba de pasar dentro del checker y no ha cambiado el código.
`--no-smoke` solo hace comprobaciones estáticas: no valida el pipeline.

## Convenciones

- Modificar la plantilla lo mínimo necesario: reutilizar y adaptar los archivos
  existentes cuando encajen con su responsabilidad. Crear módulos nuevos solo
  cuando aporten una separación necesaria o eviten mezclar responsabilidades;
  no duplicar lógica ni crear un archivo por cada rama o función. Sustituir los
  placeholders y adaptar interfaces cuando lo requiera el proyecto.
- Sin rutas absolutas en código, configuración ni documentación del proyecto.
- Toda función pública nueva o modificada lleva docstring y anotaciones de tipo.
- `run.py` orquesta; la lógica vive en `src/data/`, `src/models/` y `src/training/`.
  Mantener las claves de configuración existentes en español.
- Separar train/validación/test antes de ajustar transformaciones.
  **Normalizar solo con estadísticas de entrenamiento.** Para demostraciones,
  separar por trayectoria para evitar fugas entre episodios compartidos.
- Aplicar augmentation solo a train, jamás a validación ni a test.
- Evaluar con el modelo en modo evaluación y sin gradientes.
- El dashboard puede lanzar entrenamientos: el profesor lo autorizó para este
  proyecto de RL (comunicado el 2026-10-08).
- **Después de implementar o modificar una funcionalidad, escribir o adaptar
  y ejecutar los tests correspondientes antes de darla por terminada.** Comprobar
  cada comportamiento y requisito: casos normales, límites, entradas vacías o
  inválidas, errores esperados y combinaciones relevantes, cuando sean aplicables.
  Usar pruebas parametrizadas o basadas en propiedades cuando ayuden a ampliar
  la cobertura. Si el dominio es finito y manejable, comprobar todos sus casos;
  si no, justificar los casos elegidos y señalar lo que queda sin cubrir.
  No afirmar que se han probado todos los casos posibles sin poder demostrarlo.
- Añadir regresiones para fallos corregidos y comprobar las integraciones afectadas.
  Mantener tests de formas, ejes, batch, gradientes, splits y reproducibilidad.
  Ejecutar la suite y el checker antes de entregar cambios de funcionalidad;
  si algo falla, corregirlo. Si el entorno bloquea las pruebas, comunicar qué falta
  y no presentar la funcionalidad como validada. Cambios solo de documentación
  requieren revisión del contenido y del diff, no tests artificiales.
- **La IA debe documentar todas las conversaciones de este proyecto en
  `AI_USAGE.md`**, también consultas, aclaraciones, intentos fallidos y solicitudes
  rechazadas, aunque no produzcan código. Actualizarlo en cada intercambio antes
  de la respuesta final **dentro de la tabla existente**. Actualizar no significa
  añadir una fila por mensaje: agrupar los intercambios de una misma tarea dentro
  de una sesión y actualizar su fila; separar tareas o decisiones distintas cuando
  agrupándolas se pierda información relevante. Usar frases breves, como en los
  ejemplos de la plantilla, con fecha, tarea, herramienta, respuesta y decisión
  o resultado y su motivo cuando conste. Resumir correcciones e intentos fallidos
  relevantes sin transcribir la conversación ni detallar comandos o incidencias
  operativas que no afecten al resultado. Conservar el contenido sustancial del
  historial al resumirlo, no incluir secretos ni inventar aceptación del equipo.
  Esta agrupación es una convención del proyecto, no un requisito comprobado de
  la guía docente; si se dispone de ella, respetar sus requisitos explícitos.
  Si no puede escribir el archivo, comunicarlo y proporcionar la entrada pendiente.
- Revisar `git status` y el diff; preservar cambios ajenos. Trabajar por cambios
  pequeños y revisar los PR con la pareja usando la plantilla del repositorio.
  Hacer commits, push o entregas solo cuando formen parte del encargo.

## Lo que NO hay que tocar

- No borrar pruebas ni rebajar `tools/tiiac_check.py` para conseguir verde.
- No descartar cambios ajenos ni resolver conflictos de intención sin acuerdo.
- No versionar datos, demostraciones, pesos, entornos, cachés, `wandb/` ni secretos.
  Mantener `.gitignore`, versiones exactas en `requirements.txt` y un procedimiento
  reproducible para obtener los datos, con su procedencia y licencia.
- `src/data/splits.py` crea, valida, guarda y carga splits, pero todavía no hay
  splits congelados: no afirmar que lo están. Se congelarán cuando existan las
  demostraciones del experto (previstas para H5, 25 de octubre); entonces documentar
  la ubicación del JSON y la huella del dataset y no cambiarlos sin acuerdo explícito.

## Estado actual

Actualizar esta sección cuando cambie el proyecto.

- **Hecho:** dependencias de backend FastAPI y Uvicorn fijadas en `requirements.txt`. Contenedor `Trayectoria` y validación frente a espacios Gymnasium, guardado y carga de demostraciones NPZ/JSON implementados y probados. Splits por trayectoria con subconjuntos anidados, `LoteTransiciones` y `submuestrear_pares`, fusionados en `main`.
- **En curso:** rama `bc`: `PoliticaBC` para acciones Discrete y Box (gaussiana con `log_std` aprendido), `DatasetTransiciones`, `construir_loaders_bc`, `entrenar_bc` con parada temprana, `evaluar_bc` y `entrenar_bc_subconjunto` (recorrido completo para un subconjunto), implementados y probados con partidas inventadas; en commit, pendiente de PR; sin integrar en `run.py` (rama `integracion-e1`). Tras la revisión del 2026-10-13: generadores de los tests solo con `crear_generador`, `evaluar_bc` probada sin gradientes, `bc.learning_rate` positivo y normalización con los índices del sampler de train, sin gastar números aleatorios. Tras la revisión de Codex del 2026-10-14: la normalización respeta el sampler (antes leía todo el dataset), `DatasetTransiciones` comprueba los límites de `Box` y `entrenar_bc` rechaza métricas de validación no finitas. `ACTIVACIONES` mantiene `relu` por decisión del usuario. Inicialización, selección del mejor modelo (error cuadrático de la media en continuas) y `frecuencia_pares` alineados con el código original de GAIL; la frecuencia depende del entorno y pasará a su perfil. Normalización de observaciones con estadísticas solo de train, guardada en buffers de la política (decisión del 2026-10-13).
- **Pendiente:** `combinar_demostraciones` para el aplicativo: varios NPZ se combinan en uno nuevo con IDs sin repetir antes de crear splits, que siguen correspondiendo a un único archivo (decisión del 2026-10-13).
