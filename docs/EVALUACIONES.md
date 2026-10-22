# Evaluaciones

Cómo se tomaron las decisiones técnicas que dependían de datos, con el método, los resultados y sus límites. Todas las preguntas y las respuestas esperadas se **fijaron antes de medir** y no se ajustaron a los resultados.

> **Advertencia general.** Las muestras son pequeñas (8 a 14 preguntas) y las escribió el autor. Son evidencia de orientación para decidir, no pruebas estadísticas.

Las preguntas de recuperación están en `src/analyst/rag/evaluation.py` y la evaluación de respuestas en `scripts/evaluate_answers.py`.

## 1. Modelo de embeddings

**Pregunta:** ¿qué modelo recupera mejor los fragmentos de documentos en español?

**Método.** 14 preguntas, cada una con los identificadores de evidencia que deberían aparecer. Solo búsqueda semántica (para aislar el efecto del modelo), 73 fragmentos, top 6. Se miden: **acierto** (aparece al menos un fragmento esperado), **recall** (qué fracción de los esperados aparece) y **MRR** (qué tan arriba queda el primer acierto).

| Métrica | Chroma por defecto (MiniLM, 384 dim) | `nomic-embed-text` (768 dim) | **`bge-m3` (1024 dim)** |
|---|---|---|---|
| Acierto@6 | 9 / 14 | 13 / 14 | **14 / 14** |
| Recall@6 | 0,44 | 0,68 | **0,80** |
| MRR | 0,46 | 0,78 | **0,89** |
| Indexar 73 fragmentos (modelo ya cargado) | 2 s | 14 s | 14–16 s |

**Decisión:** `bge-m3`. Resuelve además el caso que fallaban los otros dos: "desabastecimiento en la región sur", un sinónimo de "faltantes" (primer acierto en la posición 2; con los otros, en la 7 y la 8).

**Límites:** solo tres modelos comparados; los otros dos son modelos orientados principalmente al inglés.

## 2. Recuperación: léxica, semántica e híbrida

**Pregunta:** ¿ayuda combinar la búsqueda por palabras clave (BM25) con la semántica?

**Método.** Con `bge-m3`: las mismas 14 preguntas, más **12 preguntas nuevas de validación** escritas aparte. La fusión usa Reciprocal Rank Fusion (k = 60). Se añaden precisión, nDCG (calidad del orden) y la tasa de que el primer fragmento sea relevante.

**Preguntas de ajuste (14):**

| Modo | Acierto | Recall | Precisión | nDCG | MRR | 1.º relevante |
|---|---|---|---|---|---|---|
| Léxica (BM25) | 13/14 | 0,77 | 0,27 | 0,67 | 0,70 | 0,57 |
| Semántica | 14/14 | 0,80 | 0,27 | 0,76 | **0,89** | **0,79** |
| Híbrida | 14/14 | **0,89** | **0,31** | 0,73 | 0,73 | 0,50 |

**Preguntas de validación (12):**

| Modo | Acierto | Recall | Precisión | nDCG | MRR | 1.º relevante |
|---|---|---|---|---|---|---|
| Léxica (BM25) | 11/12 | 0,78 | 0,22 | 0,74 | 0,81 | 0,75 |
| Semántica | 12/12 | 0,86 | 0,25 | 0,77 | 0,89 | 0,83 |
| Híbrida | 12/12 | **0,89** | **0,26** | **0,83** | 0,89 | 0,83 |

**Lectura:** las dos muestras se contradicen en el orden. En las 14 originales la semántica ordena mejor; en las 12 nuevas gana la híbrida. La híbrida tiene siempre el mejor recall (cubre más evidencia). Con muestras tan pequeñas, buena parte de la diferencia es ruido. Un caso que ilustra qué aporta cada una: en "¿quién sugirió que Centro podría ser la región más afectada?" la semántica pone la evidencia correcta en la posición 5 y la híbrida en la 1, porque BM25 coincide con "sugirió" y "supervisora".

## 3. Ajuste de pesos en la fusión: no se adoptó

**Pregunta:** ¿mejora la híbrida si la búsqueda semántica cuenta más que la léxica?

**Método.** Pesos ×1, ×1,5, ×2 y ×3 para la semántica. **Regla fijada antes de ver resultados:** (1) sobre las 14 preguntas de ajuste, elegir el peso con mayor nDCG entre los que mantienen 14/14 de acierto; (2) adoptarlo solo si, sobre las 12 preguntas **nuevas**, su nDCG y su acierto no son peores que los del híbrido sin pesos.

| Peso de la semántica | Acierto | Recall | nDCG | MRR | 1.º relevante |
|---|---|---|---|---|---|
| ×1 (sin pesos) | 14/14 | 0,89 | 0,73 | 0,73 | 0,50 |
| ×1,5 | 14/14 | 0,83 | 0,72 | 0,77 | 0,57 |
| ×2 | 14/14 | 0,83 | 0,76 | 0,84 | 0,71 |
| ×3 | 14/14 | 0,83 | 0,77 | 0,88 | 0,79 |

La regla 1 eligió ×3. En la validación, **×3 empeoró** frente al híbrido sin pesos (nDCG 0,77 frente a 0,83), así que **la regla 2 dice no adoptarlo**. Era un ajuste a la muestra de 14 preguntas que no se sostuvo con preguntas nuevas, que es justo para lo que sirve la validación. El soporte de pesos queda en el código, probado y desactivado.

## 4. Modelo de generación (observación cualitativa)

**Método.** Un prototipo del flujo completo con 6 preguntas (las 4 de referencia y 2 más), leyendo las respuestas. **No es una medición formal.**

| | `mistral` | `llama3` |
|---|---|---|
| Planificador | Duplicaba análisis (prompts de hasta 3.152 tokens, cerca del límite) | Un análisis, más rápido |
| ¿Es consistente el reporte con los datos? | Respondió **"Sí"**, que es incorrecto | "No son consistentes" |
| ¿Las promociones causaron la caída? | Atribuyó las promociones a Sur, donde no se observaron | "No se puede afirmar…", con cautelas |

**Decisión:** `llama3`. La lección más importante: **ningún modelo es fiable por sí solo**, y por eso el contraste con el reporte, las cifras y las limitaciones de causalidad los calcula el código.

## 5. Evaluación de las respuestas completas

**Pregunta:** ¿qué modo de búsqueda en documentos deja mejores respuestas?

**Método.** `llama3`, 8 preguntas (las 4 de referencia y 4 nuevas), `python scripts/evaluate_answers.py`. **Métricas y regla fijadas antes de medir:**
1. *Principal:* qué fracción de las afirmaciones comprobables del modelo (hipótesis, hechos con documento y contradicciones) **sobrevive al validador**.
2. *Cobertura:* en las preguntas con evidencia esperada, si la respuesta final cita al menos una de las fuentes esperadas.
3. *Desempate:* tiempo medio.
4. *Regla:* gana el modo con más afirmaciones válidas, siempre que su cobertura no sea peor.

| Corrida | Modo | Afirmaciones válidas | Cobertura | Tiempo medio |
|---|---|---|---|---|
| 1 | Híbrido | 42 % (5/12) | 1/5 | 8,3 s |
| 1 | Semántico | 29 % (4/14) | 2/5 | 8,2 s |
| 2, tras corregir el planificador | Híbrido | **53 % (8/15)** | 2/5 | 8,5 s |
| 2, tras corregir el planificador | Semántico | 39 % (7/18) | **3/5** | 9,6 s |

**Defecto descubierto en la corrida 1:** el planificador rechazaba como "fuera de alcance" la pregunta "¿Qué dicen los comerciantes sobre faltantes de producto en el Sur?" (no nombra la marca), en ambos modos. Se corrigió con una red de seguridad en código y se repitió la evaluación con las mismas métricas y la misma regla.

**Resultado:** por la regla, gana el **modo semántico** (el híbrido tiene más afirmaciones válidas pero peor cobertura), y por eso es el valor por defecto de `RETRIEVAL_MODE`.

**Lectura honesta:** es un **empate técnico**. La diferencia de cobertura es una sola pregunta; con 8 preguntas y un modelo no determinista queda dentro del ruido. La cobertura tiene una debilidad conocida: solo cuenta las fuentes citadas en hechos, hipótesis y contradicciones, no las de información insuficiente. No se cambió la métrica después de ver los resultados. La fracción de afirmaciones válidas es baja en ambos modos (39–53 %) porque el modelo escribe hipótesis aun cuando no hay documentos; el validador las descarta y las muestra. Es el comportamiento esperado de los guardarraíles.

## 6. Defectos encontrados al probar y cómo se corrigieron

Probar con los modelos reales destapó problemas que las pruebas con modelos simulados no veían:

| Defecto | Corrección |
|---|---|
| Con un esquema JSON sin límites en una lista, `mistral` entró en bucle hasta cortar la salida | Todo el esquema está acotado (valores cerrados, máximos de elementos) |
| `localhost` en Windows añadía unos 2,4 s por petición (resuelve primero a IPv6) | Se usa `127.0.0.1` y una conexión reutilizada: de 2.400 ms a unos 55 ms por consulta |
| El modelo dijo que el reporte era consistente con los datos | El contraste se calcula en código y el modelo solo lo narra |
| Una hipótesis citaba un solo cálculo para hablar de "competencia" | El validador exige que una hipótesis cite un documento |
| Una afirmación sobre Sur citaba un fragmento de Centro | El validador exige que el fragmento mencione la región (y el canal) de la afirmación |
| "La caída se atribuye a la competencia, pero no se puede demostrar causalidad" | En causalidad el resumen lo escribe el código |
| Caídas casi idénticas se presentaban como "la mayor" | Diferencias menores a 1 punto porcentual se describen como "caída similar" |
| El planificador rechazaba preguntas válidas que no nombraban la marca | Red de seguridad: si la pregunta trata temas del análisis, se trata como en alcance |
| El indicador de la página decía "Ollama apagado" con Ollama funcionando | Corregido, con una prueba que exige que la página lea solo claves que la API devuelve |

## 7. Qué dicen los datos

**Variación de ventas netas de Brío, T2 frente a T1** (datos limpios):

| Región | T1 (COP) | T2 (COP) | Variación | Aporte a la caída total |
|---|---|---|---|---|
| **Sur** | 363.431.600 | 253.404.600 | **−30,27 %** | 73,1 % |
| Centro | 314.314.900 | 285.924.400 | −9,03 % | 18,9 % |
| Occidente | 253.170.900 | 239.821.500 | −5,27 % | 8,9 % |
| Norte | 272.236.000 | 262.773.400 | −3,48 % | 6,3 % |
| Oriente | 230.096.600 | 240.789.800 | +4,65 % | −7,1 % |
| **Total** | 1.433.250.000 | 1.282.713.700 | **−10,50 %** | 100 % |

**La mayor caída es Sur,** que explica casi las tres cuartas partes de la caída total. Es pareja en los dos canales y en las dos presentaciones, y ocurre como un escalón entre marzo y abril.

**Reporte comercial frente a los datos:**

| Región | Reporte | Datos | Diferencia | ¿Coincide (±1 pp)? |
|---|---|---|---|---|
| Centro | −23 % | −9,03 % | −13,97 pp | **No** |
| Sur | −7 % | −30,27 % | +23,27 pp | **No** |
| Norte | −4 % | −3,48 % | −0,52 pp | Sí |
| Oriente | +4 % | +4,65 % | −0,65 pp | Sí |
| Occidente | −6 % | −5,27 % | −0,73 pp | Sí |

El reporte discrepa de forma material solo en Centro y Sur, y por eso ubica la mayor caída en Centro cuando está en Sur. **Ningún otro corte de los datos reproduce las cifras del reporte** (unidades en vez de pesos, un solo canal o presentación, otras ventanas de meses, T1 y T2 invertidos): el mejor error medio fue de 7,6 puntos porcentuales.

Sobre la causalidad: la base de ventas **no tiene variables de promoción ni de precio**, y los estudios cualitativos no incluyen grupo de control ni mediciones antes y después. Los datos no permiten medir el efecto de las promociones de la competencia, y el 70 % que le atribuye el reporte no es verificable.
