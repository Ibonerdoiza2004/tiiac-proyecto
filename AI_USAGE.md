# Bitácora de uso de IA

Obligatoria según la guía docente. Vale **4 puntos de E2**.

**No se puntúa usar poco.** Se puntúa el criterio: una entrada que dice
«se lo pedimos, nos dio esto, lo rechazamos porque…» vale más nota que no
haber usado ninguna herramienta.

**La línea roja:** la IA se puede usar para comprender y generar código,
citándola. **No** para el análisis automático de los resultados ni para la
discusión reflexiva sobre el trabajo. Eso es lo que evalúa la asignatura, y se
comprueba en la defensa.

---

| Fecha | Tarea | Herramienta | Qué devolvió | Qué hicimos y por qué |
|---|---|---|---|---|
| 2026-09-23 | Diseñar AGENTS.md siguiendo la guía de trabajo e inspeccionando el repositorio | Codex | Propuesta de instrucciones adaptada al proyecto GAIL/CartPole | Se utilizó como base y se pidió recuperar la estructura original del archivo para mantener su organización. |
| 2026-09-23 | Recuperar la estructura original y concretar las reglas de uso de IA y de pruebas | Codex | AGENTS.md con las seis secciones originales, rechazo explícito de tareas prohibidas, registro de conversaciones y pruebas tras cada funcionalidad | Se ajustaron las instrucciones solicitadas y se corrigió el estado actual: propuestas.md está terminado; los demás campos se dejaron vacíos porque su información no era correcta. Las pruebas deben cubrir casos normales, límites, errores y combinaciones relevantes, con exhaustividad cuando sea viable. |

---

<!--
Ejemplos del nivel que se espera (bórralos cuando tengas los tuyos):

| 12-nov | Escribir el Dataset con el subset del paper | Claude Code | Clase completa con __len__ y __getitem__ | **Aceptado con cambios.** Normalizaba con estadísticas del conjunto entero; lo cambiamos a las de train para no meter fuga |
| 15-nov | «¿Por qué mi loss no baja?» | Claude | Cinco hipótesis | **Rechazado como análisis.** Lo usamos como lista de comprobación; el diagnóstico (lr 100× alto) lo encontramos con el LR range test |
-->
