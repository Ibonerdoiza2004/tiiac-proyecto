# Ficha de propuestas de proyecto · Hito H1

**Entrega: domingo 20 de septiembre, 23:59, en la Tarea «H1 · Propuestas de proyecto».**
**Decisión: domingo 27 de septiembre (hito H2).** Sin visto bueno no se sigue adelante.

Hasta **tres propuestas, en orden de preferencia**. La primera tiene que estar completa; la segunda y
la tercera son vuestras alternativas si la primera no pasa la puerta. Las tres tienen las mismas cuatro
comprobaciones.

---

## Cómo se entrega

1. Copiad esta página entera a un fichero `docs/propuestas.md` de vuestro repositorio y rellenadla ahí.
   Es Markdown: las tablas se editan en cualquier editor, y así queda en la historia de _commits_.
2. Marcad las casillas cambiando `[ ]` por `[x]` **solo cuando sea verdad**. Una casilla marcada sin
   evidencia en la tabla de debajo cuenta como no marcada.
3. Haced _commit_ y _push_, y en la Tarea de ALUD pegad **la URL del fichero en GitHub**. Entrega uno,
   cuenta por los dos.
4. El 27 de septiembre tendréis la decisión de cada propuesta en la retroalimentación de la Tarea. La
   aprobada se convierte en vuestro `docs/viabilidad.md`, que es el que se corrige con E1.

---

## Las dos vías

**Vía A · lista curada.** Uno de los doce papers ya comprobados, que están en el capítulo _Lista de
papers_ de este Libro: ResNet, U-Net, YOLO (v1), Vision Transformer, CycleGAN, Grad-CAM, SimCLR,
ejemplos adversarios (FGSM), Word2Vec, Transformer, BERT (solo preentrenamiento) y Neural
Collaborative Filtering. Sabemos que caben en la regla 5/20 y que tienen un aplicativo natural encima.
La ficha se rellena igual, pero es rápido. **Máximo dos parejas por paper**, por orden de entrega.

**Vía B · propuesta propia.** Cualquier paper o caso industrial que os interese, si la ficha pasa las
cuatro comprobaciones. Da más trabajo al principio, y ese trabajo de acotar es justo lo que venís a
aprender aquí.

---

## Pareja

|                                     |                                                                       |
| ----------------------------------- | --------------------------------------------------------------------- |
| **Nombres**                         | Asier Corral e Ibon Erdoiza                                           |
| **Repositorio**                     | https://github.com/Ibonerdoiza2004/tiiac-proyecto.git                 |
| **Rol profesional al que apuntáis** | Nos vemos ambos como Ingenieros de IA especializados en RL o NLP      |

---

## Propuesta 1

### Identificación

|                            |                                                                     |
| -------------------------- | ------------------------------------------------------------------- |
| **Paper o referencia**     | Generative Adversarial Imitation Learning                           |
| **Autores y año**          | Jonathan Ho y Stefano Ermon, 2016                                   |
| **Enlace**                 | https://arxiv.org/pdf/1606.03476                                    |
| **Vía**                    | B (propuesta propia)                             |
| **Por qué este y no otro** | El proyecto de RL es uno de los que más hemos disfrutado y nos quedamos con la espina de hacer algo sobre Imitation Learning, así que nos ha parecido buena idea hacer algo relacionado.     |
### Comprobación 1 · Datos

- [ ] Son **públicos y descargables hoy**. Enlace que funciona, no una promesa.
- [ ] La licencia permite el uso académico.

|                                       |                                                                |
| ------------------------------------- | -------------------------------------------------------------- |
| **Dataset**                           | Generado por nosotros mediante TRPO en los entornos evaluados        |
| **Enlace de descarga**                | Entornos de Gymnasium: https://gymnasium.farama.org/environments/classic_control/cart_pole/                                                                        |   
| **Tamaño**                            |                                                                |
| **Licencia**                          |                                                                |
| **¿Hace falta registro o solicitud?** | No                                                             |

### Comprobación 2 · Especificación

- [x] Hay implementación de referencia, **o bien** el paper especifica la arquitectura lo bastante como
      para implementarla sin adivinar.

|                                                          |                                                                                                               |
| -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| **¿Hay código de referencia?**                           |  Sí: https://github.com/openai/imitation             |
| **Si no lo hay, ¿el paper da la arquitectura completa?** | El paper define redes de 2 capas ocultas de 100 neuronas con tanh y Adam para el discriminador, TRPO para experto y GAIL y Adam para BC                           |
| **Qué NO especifica el paper**                           | No especifica el learning rate de Adam, todos los hiperparámetros de TRPO, el método exacto de inicialización de pesos, cómo se seleccionan los pares por trayectoria ni el criterio de parada de BC. |

### Comprobación 3 · Cómputo · la regla 5/20

- [x] Cabe en el presupuesto, **con el plan de recorte escrito**.

|                                                |                                                                                                                   |
| ---------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| **Escala del paper original**                  | Se usa un total de 9 entornos. Para CartPole, usan conjuntos de 1, 4, 7 y 10 trayectorias, con unos 50 pares por trayectoria. GAIL realiza 300 iteraciones de 5.000 interacciones, es decir, 1,5 millones por entrenamiento. Se evalúan 7 inicializaciones y 50 episodios por política. No se especifican hardware ni tiempo de ejecución.                                                         |
| **Escala que vais a hacer vosotros**           | La evaluación progresará por etapas: (1) CartPole; (2) MountainCar; (3) Acrobot; (4) Reacher y (5) HalfCheetah, como objetivo ideal. La elección de los entornos busca aumentar la dificultad poco a poco. CartPole nos permitirá comprobar que la implementación funciona en un problema sencillo y con acciones discretas. MountainCar añade la necesidad de encadenar acciones para ganar impulso, mientras que Acrobot introduce una dinámica articulada distinta. Después, Reacher servirá para estudiar el paso a acciones continuas y ver si BC sigue siendo competitivo. HalfCheetah sería el reto final, porque combina locomoción y varios controles continuos. En cada entorno compararemos BC y GAIL con un experto y un agente aleatorio, manteniendo los tamaños de demostraciones del paper cuando sea aplicable.  |
| **Tiempo estimado del entrenamiento completo** | Mediremos el tiempo de cada etapa y mantendremos como criterio la regla 5/20 (en el entorno Cartpole): el smoke test deberá completarse en menos de 5 minutos en CPU y el entrenamiento completo en menos de 20 minutos en una T4. La primera estimación para CartPole es inferior a 10 minutos. |
| **Qué se pierde al recortar**                  | Si no podemos completar todas las etapas, priorizaremos CartPole, MountainCar y Acrobot, y dejaremos a un lado Reacher y HalfCheetah. Perdiendo la posibilidad de buscar correlación entre la complejidad del entorno y el resultado de los distintos modelos. |
| **Hardware que vais a usar**                   | Portatil con RTX 3060 Laptop / Colab T4                                                                                |




### Comprobación 4 · Aplicativo

- [x] Hay una capa de servicio natural encima de la replicación.

|                                |                                                                                   |
| ------------------------------ | --------------------------------------------------------------------------------- |
| **Qué construís encima**       | Un dashboard interactivo para jugar a CartPole, grabar demostraciones, entrenar experto/BC/GAIL y visualizar y comparar sus resultados.                           |
| **Quién lo usaría y para qué** | Estudiantes y profesores para experimentar con aprendizaje por imitación de manera sencilla y poder ver de manera visual las diferencias entre métodos.  |
| **Qué necesita del modelo**    | Como entrada, las 4 observaciones de CartPole (posición y velocidad del carro, ángulo y velocidad angular del palo). Como salida, acción izquierda o derecha.                                              |

### Riesgo principal

|                                                    |                                                                                              |
| -------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| **Qué es lo que más probablemente va a salir mal** | Que el desarrollo del aplicativo nos lleve más tiempo del previsto y nos deje poco tiempo para replicar la arquitectura del paper.                                                                      |
| **Qué haríais si pasa**                            | Si vemos que el desarrollo del dashboard nos va a llevar más tiempo del que tenemos disponible, reduciremos algunas funciones del aplicativo que no sean estrictamente necesarias (Poder activar/desactivar la visualización durante el entrenamiento, elegir hiperparámetros desde el dashboard,...). |

### Visto bueno del profesor (H2) · no rellenar

|                 |                                            |
| --------------- | ------------------------------------------ |
| **Decisión**    | Aprobado / Aprobado con recorte / Devuelto |
| **Condiciones** |                                            |

---

## Propuesta 2 (alternativa)

### Identificación

|                            |                                                                     |
| -------------------------- | ------------------------------------------------------------------- |
| **Paper o referencia**     | ParaDetox: Text Detoxification with Parallel Data                                                   |
| **Autores y año**          | Varvara Logacheva, Daryna Dementieva, Sergey Ustyantsev, Daniil Moskovskiy, David Dalei, Irina Krotova, Nikita Semenov nd Alexander Panchenko                                                                    |
| **Enlace**                 | _https://aclanthology.org/2022.acl-long.469.pdfDowdp                                   
| **Vía**                    | B (propuesta propia)                             |
| **Por qué este y no otro** | _Dos líneas. Idealmente conectado con el rol profesional de arriba_ |

### Comprobación 1 · Datos

- [ x ] Son **públicos y descargables hoy**. Enlace que funciona, no una promesa.
- [ x ] La licencia permite el uso académico.

|                                       |                                                                |
| ------------------------------------- | -------------------------------------------------------------- |
| **Dataset**                           | ParaDetox Corpus                                                               |
| **Enlace de descarga**                | https://github.com/s-nlp/paradetox/blob/main/paradetox/paradetox.tsv
| **Tamaño**                            | 1.55MB, 11936 lineas de texto (he had steel balls too !	he was brave too!)                         |   
| **Licencia**                          |CC0 1.0 Universal (Dominio Público)                                                                |
| **¿Hace falta registro o solicitud?** | No  |

### Comprobación 2 · Especificación

- [x] Hay implementación de referencia, **o bien** el paper especifica la arquitectura lo bastante como
      para implementarla sin adivinar.

|                                                          |                                                                                                               |
| -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| **¿Hay código de referencia?**                           | https://github.com/s-nlp/paradetox                                                                                            |
| **Si no lo hay, ¿el paper da la arquitectura completa?** | El paper especifica fine-tuning de modelos Seq2Seq (BART y T5) usando pérdida Cross-Entropy de generación y optimizador AdamW ($lr=3\cdot 10^{-5}$).                                                        |
| **Qué NO especifica el paper**                           |No detalla los parámetros de *decoding* en producción (ej. *beam search vs top-p sampling*) ni umbrales dinámicos de filtrado para equilibrar toxicidad vs preservación de significado. |

### Comprobación 3 · Cómputo · la regla 5/20

- [x] Cabe en el presupuesto, **con el plan de recorte escrito**.

|                                                |                                                                                                                   |
| ---------------------------------------------- | -----------------------------------------------------------------------------------------------------------------|
| **Escala del paper original**                  | Dataset completo (~20k pares), entrenamiento con BART-large / BART-base durante 10 epochs en GPUs de alta gama (NVIDIA V100/A100)                                                        |
| **Escala que vais a hacer vosotros**           | Subconjunto de 5,000 pares tóxico-neutro, fine-tuning de `t5-small` (60M parámetros) o `facebook/bart-base` durante 3-4 epochs                                                                       |
| **Tiempo estimado del entrenamiento completo** | 8 a 12 minutos en una GPU T4 (Google Colab)                                         |
| **Qué se pierde al recortar**                  | La metrica bajará levemente (~3-5%) al usar un modelo más pequeño y menos datos. |
| **Hardware que vais a usar**                   | Colab                                                                                |

### Comprobación 4 · Aplicativo

- [x] Hay una capa de servicio natural encima de la replicación.

|                                |                                                                                   |
| ------------------------------ | --------------------------------------------------------------------------------- |
| **Qué construís encima**       | Una API REST (FastAPI)                           |
| **Quién lo usaría y para qué** | Moderadores de comunidades o usuarios de foros para reescribir de forma educada un comentario agresivo antes de publicarlo en la red |
| **Qué necesita del modelo**    | Entrada: frase tóxica (string). Salida: frase desintoxicada (string). Latencia aceptable: < 500ms por frase.                                             |

### Riesgo principal

|                                                    |                                                                                              |
| -------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| **Qué es lo que más probablemente va a salir mal** |Que el modelo borre demasiado texto para evitar la toxicidad, alterando o destruyendo el significado original de la frase                                                                                              |
| **Qué haríais si pasa**                            | Implementar un post-filtro de similitud  que descarte reescrituras cuya similitud sea menor a 0.7|

### Visto bueno del profesor (H2) · no rellenar

|                 |                                            |
| --------------- | ------------------------------------------ |
| **Decisión**    | Aprobado / Aprobado con recorte / Devuelto |
| **Condiciones** |                                            |

---

## Propuesta 3 (alternativa)

### Identificación

|                            |                                                                     |
| -------------------------- | ------------------------------------------------------------------- |
| **Paper o referencia**     | _Título completo_                                                   |
| **Autores y año**          |                                                                     |
| **Enlace**                 | _arXiv, DOI o web del paper_                                        |
| **Vía**                    | A (lista curada) / B (propuesta propia)                             |
| **Por qué este y no otro** | _Dos líneas. Idealmente conectado con el rol profesional de arriba_ |

### Comprobación 1 · Datos

- [ ] Son **públicos y descargables hoy**. Enlace que funciona, no una promesa.
- [ ] La licencia permite el uso académico.

|                                       |                                                                |
| ------------------------------------- | -------------------------------------------------------------- |
| **Dataset**                           |                                                                |
| **Enlace de descarga**                |                                                                |
| **Tamaño**                            | _En MB/GB y en número de ejemplos_                             |
| **Licencia**                          |                                                                |
| **¿Hace falta registro o solicitud?** | _Si la respuesta es «hay que pedir acceso y tardan», es un no_ |

### Comprobación 2 · Especificación

- [ ] Hay implementación de referencia, **o bien** el paper especifica la arquitectura lo bastante como
      para implementarla sin adivinar.

|                                                          |                                                                                                               |
| -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| **¿Hay código de referencia?**                           | _Sí (enlace) / No_                                                                                            |
| **Si no lo hay, ¿el paper da la arquitectura completa?** | _Capas, dimensiones, función de pérdida, optimizador_                                                         |
| **Qué NO especifica el paper**                           | _Lo importante. Si la lista está vacía, no habéis leído el paper con suficiente atención: siempre falta algo_ |

### Comprobación 3 · Cómputo · la regla 5/20

- [ ] Cabe en el presupuesto, **con el plan de recorte escrito**.

|                                                |                                                                                                                   |
| ---------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| **Escala del paper original**                  | _Dataset completo, N épocas, qué hardware, cuánto tardó_                                                          |
| **Escala que vais a hacer vosotros**           | _Subset de N, M épocas, modelo reducido a…_                                                                       |
| **Tiempo estimado del entrenamiento completo** | _En minutos, en la máquina que vayáis a usar, y cómo lo habéis medido_                                            |
| **Qué se pierde al recortar**                  | _La respuesta honesta. «La métrica bajará de 0.72 a algo en torno a 0.6» es una buena respuesta; «nada» no lo es_ |
| **Hardware que vais a usar**                   | _Portátil / Colab gratuito / otro_                                                                                |

### Comprobación 4 · Aplicativo

- [ ] Hay una capa de servicio natural encima de la replicación.

|                                |                                                                                   |
| ------------------------------ | --------------------------------------------------------------------------------- |
| **Qué construís encima**       | _Un buscador, un detector en vídeo, una API, un panel…_                           |
| **Quién lo usaría y para qué** | _Una frase. Si no se os ocurre, el proyecto no cumple el objetivo del aplicativo_ |
| **Qué necesita del modelo**    | _Entrada, salida, latencia aceptable_                                             |

### Riesgo principal

|                                                    |                                                                                              |
| -------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| **Qué es lo que más probablemente va a salir mal** |                                                                                              |
| **Qué haríais si pasa**                            | _Útil: «si no converge, reducimos a MNIST y lo declaramos». Inútil: «nada, está controlado»_ |

### Visto bueno del profesor (H2) · no rellenar

|                 |                                            |
| --------------- | ------------------------------------------ |
| **Decisión**    | Aprobado / Aprobado con recorte / Devuelto |
| **Condiciones** |                                            |
