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
- **Hecho:** BC fusionada en `main` mediante PR #3: `PoliticaBC` para acciones `Discrete` y `Box`, `DatasetTransiciones`, loaders y entrenamiento/evaluación BC. La integración de BC en `run.py` sigue en `integracion-e1`. La normalización usa estadísticas de train, las acciones respetan los límites del espacio y las métricas no finitas se rechazan. El código usa `crear_generador` para las semillas independientes.
- **Hecho:** configuración por entornos fusionada en `main` mediante PR #4: `TipoEntorno` (CartPole activo), resolución recursiva base → perfil → ajustes, validación de claves desconocidas y valores pendientes, metadatos de procedencia y capa smoke; `run.py` acepta `--entorno`. Perfil `CartPole-v1` con la fuente de cada valor; presupuesto del experto de 100 000 pasos y retorno objetivo 500 (máximo de `CartPole-v1`; el umbral de Gymnasium es 475). Qué hace `retorno_objetivo` quedó decidido en `trpo-experto` (ver abajo). Validado con Python 3.12: 735 tests y checker 95/100. Falta enlazar el proyecto W&B en README.
- **Hecho:** entornos fusionados en `main` mediante PR #6: `crear_entorno` y `crear_entorno_vectorizado` en `src/data/entornos.py`, con `Monitor` dentro de cada instancia para conservar el retorno real bajo el envoltorio de GAIL, y siembra en `src/utils/seed.py`; `trpo.n_entornos` en la config. En la rama `trpo-experto` (15 de octubre), tras la auditoría y la revisión del PR: una sola validación de semillas (`validar_semilla`, que admite enteros de NumPy en todas las funciones), semillas por instancia con `semillas_independientes` (`SeedSequence.spawn`; con `semilla + i` las bases cercanas compartían instancias), espacios sembrados antes del `reset` y tests de reproducibilidad entre vectorizados, de integración con `EntornoRecompensaGAIL`, de `render_mode` y de errores. Avisos para `trpo-experto`: no pasar `seed=` a TRPO (resiembra los entornos fuera de `seed.py`), comprobar que `pasos_por_iteracion` sea divisible por `n_entornos` y que la evaluación siembre cada episodio desde `seed.py`; `sembrar_entorno` gasta un `reset`, así que la primera partida usa el segundo estado inicial de la semilla.
- **Hecho:** recompensa de GAIL fusionada en `main` mediante PR #5: `Discriminador` en `factory.py` (como el original: en discretas solo el estado y una salida por acción; en `Box`, estado y acción concatenados), con una base común `_PerceptronNormalizado` que comparte con `PoliticaBC` (BC queda idéntica: mismas claves y pesos con la misma semilla). `tensores_de_pares` en `loaders.py`, reutilizada por `DatasetTransiciones`. En `src/training/gail.py`: `preparar_discriminador` (normalización fija con los pares expertos de train, elección propia frente al original), `calcular_recompensas_gail` (−log(1−D) o log D según `gail.recompensa` del perfil), `muestrear_expertos` (con reemplazo), `actualizar_discriminador` (media por grupo de BCE − 0.001·entropía, Adam 0.01, 1 paso) y `EntornoRecompensaGAIL` (`VecEnvWrapper` que devuelve un `LoteTransiciones` por bloque y deja la recompensa original en `infos`). Sección `gail` en `default.yaml`. 866 tests, 27 de 28 mutaciones detectadas (la otra era una comprobación duplicada, eliminada) y checker 95/100. Tras la revisión del 15 de octubre: los tests ya no crean generadores con semilla fuera de `seed.py`, la validación de dimensiones es común a `PoliticaBC` y `Discriminador`, la entropía registrada pesa igual cada grupo y hay tests de partidas truncadas y de las copias que guarda el envoltorio. Tras la revisión del PR: el envoltorio rechaza `Discrete` con `start` distinto de 0, que sb3 no admite (sus políticas generan índices 0..n−1); `calcular_recompensas_gail` y BC siguen traduciendo el `start` de las acciones del entorno. En la rama `trpo-experto` (15 de octubre), los dos fallos aplazados de la revisión del PR: el envoltorio pasa al entorno la misma copia de las acciones que puntúa y guarda (antes podían desincronizarse entre `step_async` y `step_wait`), y `tensores_de_pares` comprueba la forma antes de `len`, así que un estado escalar da `ValueError`.
- **Hecho:** TRPO y experto fusionados en `main` mediante PR #7 (rama `trpo-experto`, creada como `gail`), con los arreglos pendientes de `rewards-gail` y `entornos` (arriba). Tamaños del experimento en la config para que cada entorno tenga los suyos: `experimento.repeticiones` (7), `demostraciones.partidas_train` (10), `partidas_validacion` (5), `subconjuntos` ([1, 4, 7, 10]) y `gail.iteraciones` (300), en `default.yaml` y repetidos en el perfil de CartPole para que no cambien si cambia la base; el smoke usa 1 repetición. En el original, HalfCheetah usa subconjuntos 4, 11, 18, 25, 1 repetición, frecuencia 20 y 50 000 pares por iteración; Reacher no aparece en los pipelines revisados. TRPO y experto en `src/training/trpo.py`: `crear_trpo` (hiperparámetros de `trpo` en la config, incluidos los del crítico y la búsqueda lineal; red 2 × 100 tanh; CPU; sin `seed=`; comprueba divisibilidades y espacios), `iteracion_trpo` (un bloque sin reiniciar el entorno), `evaluar_retorno` (evaluación mínima con una semilla por episodio vía `reiniciar_con_semilla`, hasta la rama de Asier), `entrenar_experto` (semillas derivadas de `experto.semilla`, evalúa la política estocástica cada `experto.evaluar_cada` iteraciones, guarda la mejor y para al alcanzar `retorno_objetivo`; elección propia frente al snapshot fijo del original) y `guardar_experto`/`cargar_experto` (`modelo.zip` y `experto.json` con config, resumen, versiones y hardware; sin sobrescribir). Normalización de observaciones opcional por perfil con `VecNormalize` (desactivada en CartPole; sus estadísticas van en el resumen). `scripts/entrenar_experto.py` es el procedimiento reproducible (`--sin-wandb` para no registrar). La evaluación usa un generador de PyTorch aislado por episodio (`generador_torch_aislado` en `seed.py`): la misma semilla da el mismo resultado y evaluar no altera el entrenamiento. El script comprueba la carpeta de salida antes de entrenar, y la config exige al menos 2 pasos y 2 de batch si se estandarizan las ventajas. `iteracion_trpo` mide solo las partidas que terminan en su iteración, `experto.pasos_totales` debe ser múltiplo de `trpo.pasos_por_iteracion` `experto.semilla` pasa por `validar_semilla` (admite enteros de NumPy, que el JSON guarda como enteros) y `experto.json` es JSON estricto, con el commit de git y si había cambios sin commit (`estado_git`); el script avisa si los había o si no pudo leer git. Medido el 16 de octubre en CPU (WSL2, 12 núcleos; el modelo no usa la GPU): CartPole alcanza 500 de retorno medio en 10 episodios estocásticos tras 50 000 pasos y 53 s; dos ejecuciones con la misma config dan el mismo experto. Falta medir en una T4. Tests con acciones discretas (CartPole) y continuas (Pendulum); 25 de 26 mutaciones detectadas y la restante cubierta después. Diferencias con el original que hay que documentar: la red de valor se ajusta con Adam por minibatches, no con un paso de región de confianza, y el experto se elige por evaluación. El bucle GAIL con TRPO y el reajuste de la red de valor tras actualizar D (decidido el 14 de octubre, con retornos GAE) siguen pendientes.
- **En curso:** rama `generar-demostraciones` (19 de octubre), en commit salvo el arreglo de la revisión del PR. En `demostraciones.py`: `jugar_partida` (una partida con una política como función de observación a acción, con `reiniciar_con_semilla` y `generador_torch_aislado`; guarda una copia de cada estado y pasa otra a la política, para que modificar su entrada no cambie la partida aunque el entorno reutilice su buffer; revisión del PR), `grabar_demostraciones` (semillas de `semillas_independientes`, IDs 0..N−1, valida frente a los espacios), `retorno_partida` (suma paso a paso: `sum` de Python 3.12 compensa el redondeo y cambiaba el último decimal), `huella_partidas` (SHA-256 solo del contenido, sin metadatos), `config_demostraciones` y `generar_dataset` (graba, guarda `demostraciones.npz`, crea `splits.json` y borra el NPZ si fallan los splits). Config: `demostraciones.semilla` (de ella salen la de las partidas y la de los splits) y `demostraciones.determinista: false`, como el original; se guardan todas las partidas, sin filtrar. En `trpo.py`: `politica_trpo`, `evaluar_retorno` sobre `jugar_partida` (mismos resultados exactos que antes) y `cargar_politica_experto` (comprueba el entorno; metadatos sin fecha para que regenerar dé el mismo archivo). `comprobar_cantidades` de `splits.py` pasa a ser pública. `scripts/get_data.py` adaptado como procedimiento reproducible de los datos (`--huella-esperada` compara la huella de las partidas, que no cambia con el commit). Sin W&B: queda para otra rama. 1082 tests; 33 mutaciones detectadas tras reforzar los tests (una destapó que el estado se copiaba después de llamar a la política). Medido el 19 de octubre en CPU: experto en 22 s y 50 000 pasos; 15 partidas, 7401 pasos y 244 KB en 6 s; regenerar da los mismos bytes; BC con 1, 4, 7 y 10 partidas entrena en menos de 2 s cada una. Falta: acordar con Asier la congelación (generar desde un commit limpio y documentar ubicación y huellas).
- **Pendiente:** `combinar_demostraciones` para el aplicativo: varios NPZ se combinan en uno nuevo con IDs sin repetir antes de crear splits, que siguen correspondiendo a un único archivo (decisión del 2026-10-13).
