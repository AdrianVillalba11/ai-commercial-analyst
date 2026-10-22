# Respuestas de referencia del sistema

- Fecha: 2026-10-22 10:38
- Python: 3.13.14 · Sistema: Windows 11
- Modelo de generación: llama3 · Modelo de embeddings: bge-m3 · Búsqueda por defecto: semantic

> **Qué es este archivo:** la salida de una ejecución real del sistema en la fecha indicada. **No son respuestas fijas.** Las preguntas 5 y 6 no están en el enunciado: se agregan para mostrar que el sistema acepta consultas nuevas. Un modelo de lenguaje local puede redactar de forma distinta en cada ejecución. Lo que no cambia son las cifras, el veredicto del contraste con el reporte y las limitaciones de causalidad, porque salen de código y no del modelo. Para regenerarlo: `python scripts/generate_evidence.py --answers`.

## Pregunta 1 · Cuantitativa

> ¿Cuál fue la región con mayor caída porcentual en ventas netas de Brío entre T1 y T2 de 2026? Muestra los valores y el cálculo.

**Configuración:** modelo `llama3` · búsqueda `semantic` · plan `cuantitativa` (modelo) · 8.2 s

**Resumen:** La región con mayor caída porcentual en ventas netas de Brío entre T1 y T2 de 2026 fue el Sur, con una variación de -30,27%.

**Hechos**

- Mayor caída de ventas netas (COP) por región: Sur, -30,27 % (T1 363.431.600 → T2 253.404.600). Explica el 73,1 % de la caída total de Brío. _[CALC-1; calculo]_
- Crecieron: Oriente +4,65 %. _[CALC-1; calculo]_
- Total Brío (ventas netas (COP)): -10,50 % (T1 1.433.250.000 → T2 1.282.713.700). _[CALC-1; calculo]_

**Descartado por el validador** (afirmaciones del modelo que no pasaron las verificaciones)

- «La caída en ventas netas en el Sur puede estar relacionada con la competencia en el mercado local, según fuentes de la región.» → es una hipótesis y no cita ningún documento que la sugiera

**Cálculos**

```
[CALC-1] Brío — venta neta (COP) — por región (sin filtros), T2 frente a T1
- Sur: T1 363.431.600 | T2 253.404.600 | variación -110.027.000 (-30,27 %) | aporta 73,1 % del cambio total
- Centro: T1 314.314.900 | T2 285.924.400 | variación -28.390.500 (-9,03 %) | aporta 18,9 % del cambio total
- Occidente: T1 253.170.900 | T2 239.821.500 | variación -13.349.400 (-5,27 %) | aporta 8,9 % del cambio total
- Norte: T1 272.236.000 | T2 262.773.400 | variación -9.462.600 (-3,48 %) | aporta 6,3 % del cambio total
- Oriente: T1 230.096.600 | T2 240.789.800 | variación 10.693.200 (4,65 %) | aporta -7,1 % del cambio total
- Total Brío: T1 1.433.250.000 | T2 1.282.713.700 | variación -150.536.300 (-10,50 %)
Mayor caída: Sur (-30,27 %)
```

## Pregunta 2 · Integrada (cifras y documentos)

> ¿Qué hallazgos de consumidor y observaciones comerciales podrían ayudar a explicar el deterioro de Brío?

**Configuración:** modelo `llama3` · búsqueda `semantic` · plan `integrada` (modelo) · 14.4 s

**Resumen:** El deterioro de Brío se asocia con la disminución de ventas en Sur y la mayor caída en Sur/Tradicional (-30,28%). La disponibilidad percibida y la competitividad promocional pueden haber influido.

**Hechos**

- Mayor caída de ventas netas (COP) por región y canal: Sur / Tradicional, -30,28 % (T1 202.100.800 → T2 140.900.300). Explica el 40,7 % de la caída total de Brío. _[CALC-1; calculo]_
- Crecieron: Oriente / Tradicional +4,62 %, Oriente / Moderno +4,69 %. _[CALC-1; calculo]_
- Total Brío (ventas netas (COP)): -10,50 % (T1 1.433.250.000 → T2 1.282.713.700). _[CALC-1, CALC-2; calculo]_
- Caída similar en todos los grupos por presentación: entre -10,44 % y -10,84 % (diferencias menores a 1 punto porcentual). _[CALC-2; calculo]_

**Hipótesis (no demostradas)**

- La mayor caída en Sur/Tradicional puede estar relacionada con la disponibilidad percibida y la competitividad promocional. _[OC-SUR-01; modelo]_

**Información insuficiente**

- No se puede determinar cuánto de la caída de ventas se explica por disponibilidad o promoción. _[modelo]_

**Descartado por el validador** (afirmaciones del modelo que no pasaron las verificaciones)

- «Dinámica competitiva. Se estima que las promociones de competidores explican aproximadamente el 70 % de la contracción de ventas de Brío.» → es una afirmación del reporte comercial, no un hecho verificado
- «La variación en Sur/Tradicional (-30,28%) es mayor que la variación en Sur/Moderno (-30,26%).» → una contradicción es entre fuentes: debe citar al menos dos

**Evidencia consultada:** REP-02, REP-S1-1, ENT-CEN-04, INV-02, INV-S0-2, OC-SUR-01

**Avisos:**
- Se eliminó la fuente INV-02 (no menciona ese canal): la afirmación habla del canal Tradicional.

**Cálculos**

```
[CALC-1] Brío — venta neta (COP) — por región y canal (sin filtros), T2 frente a T1
- Sur / Tradicional: T1 202.100.800 | T2 140.900.300 | variación -61.200.500 (-30,28 %) | aporta 40,7 % del cambio total
- Sur / Moderno: T1 161.330.800 | T2 112.504.300 | variación -48.826.500 (-30,26 %) | aporta 32,4 % del cambio total
- Centro / Moderno: T1 144.625.100 | T2 131.503.100 | variación -13.122.000 (-9,07 %) | aporta 8,7 % del cambio total
- Centro / Tradicional: T1 169.689.800 | T2 154.421.300 | variación -15.268.500 (-9,00 %) | aporta 10,1 % del cambio total
- Occidente / Moderno: T1 107.163.900 | T2 101.399.600 | variación -5.764.300 (-5,38 %) | aporta 3,8 % del cambio total
- Occidente / Tradicional: T1 146.007.000 | T2 138.421.900 | variación -7.585.100 (-5,20 %) | aporta 5,0 % del cambio total
- Norte / Tradicional: T1 147.416.800 | T2 142.287.700 | variación -5.129.100 (-3,48 %) | aporta 3,4 % del cambio total
- Norte / Moderno: T1 124.819.200 | T2 120.485.700 | variación -4.333.500 (-3,47 %) | aporta 2,9 % del cambio total
- Oriente / Tradicional: T1 131.497.600 | T2 137.567.400 | variación 6.069.800 (4,62 %) | aporta -4,0 % del cambio total
- Oriente / Moderno: T1 98.599.000 | T2 103.222.400 | variación 4.623.400 (4,69 %) | aporta -3,1 % del cambio total
- Total Brío: T1 1.433.250.000 | T2 1.282.713.700 | variación -150.536.300 (-10,50 %)
Mayor caída: Sur / Tradicional (-30,28 %)
```

```
[CALC-2] Brío — venta neta (COP) — por presentación (sin filtros), T2 frente a T1
- Familiar: T1 219.235.200 | T2 195.468.600 | variación -23.766.600 (-10,84 %) | aporta 15,8 % del cambio total
- Individual: T1 1.214.014.800 | T2 1.087.245.100 | variación -126.769.700 (-10,44 %) | aporta 84,2 % del cambio total
- Total Brío: T1 1.433.250.000 | T2 1.282.713.700 | variación -150.536.300 (-10,50 %)
Caída similar en todos los grupos (entre -10,44 % y -10,84 %): las diferencias entre grupos son menores a 1 punto porcentual.
```

## Pregunta 3 · Contraste con el reporte comercial

> ¿Son consistentes las principales conclusiones del reporte comercial con los datos de ventas?

**Configuración:** modelo `llama3` · búsqueda `semantic` · plan `contraste_reporte` (modelo) · 16.6 s

**Resumen:** El reporte NO es consistente con los datos: 2 de 5 regiones difieren más de 1 pp (Centro, Sur) y señala a Centro como la región de mayor caída, cuando los datos indican Sur (-30,27 %). _(redactado por el sistema a partir de los cálculos)_

**Hechos**

- Mayor caída de ventas netas (COP) por región: Sur, -30,27 % (T1 363.431.600 → T2 253.404.600). Explica el 73,1 % de la caída total de Brío. _[CALC-1; calculo]_
- Crecieron: Oriente +4,65 %. _[CALC-1; calculo]_
- Total Brío (ventas netas (COP)): -10,50 % (T1 1.433.250.000 → T2 1.282.713.700). _[CALC-1; calculo]_
- Coinciden con los datos (±1 pp): Norte, Oriente, Occidente. _[CONTRASTE-1; contraste]_

**Reporte comercial frente a los datos** (variación T2 vs T1 por región)

| Región | Reporte | Datos | Diferencia | Resultado |
|---|---:|---:|---:|---|
| Norte | -4 % | -3,48 % | -0,52 pp | Coincide |
| Centro | -23 % | -9,03 % | -13,97 pp | No coincide |
| Sur | -7 % | -30,27 % | +23,27 pp | No coincide |
| Oriente | +4 % | 4,65 % | -0,65 pp | Coincide |
| Occidente | -6 % | -5,27 % | -0,73 pp | Coincide |

**Contradicciones entre fuentes**

- Según el reporte, Centro varió -23 %, pero los datos dan -9,03 % (diferencia de -13,97 pp). _[CONTRASTE-1, REP-REGION-CENTRO; contraste]_
- Según el reporte, Sur varió -7 %, pero los datos dan -30,27 % (diferencia de +23,27 pp). _[CONTRASTE-1, REP-REGION-SUR; contraste]_
- REP-01 afirma algo que los datos contradicen: Centro es la región con mayor caída. Los datos indican Sur (-30,27 %) como la región de mayor caída. _[CONTRASTE-1, REP-01; contraste]_

**Información insuficiente**

- REP-02 atribuye un porcentaje de la caída a una causa; no es verificable: la base de ventas no contiene variables de promoción ni de precio (columnas: registro_id, mes, marca, region, canal, punto_venta_id, presentacion, unidades, venta_neta_cop). _[CONTRASTE-1, REP-02; contraste]_
- No se puede afirmar que la caída de ventas en el Sur esté causada por la disponibilidad y la competitividad promocional de la marca, ya que no se dispone de evidencia causal demostrada. _[modelo]_

**Descartado por el validador** (afirmaciones del modelo que no pasaron las verificaciones)

- «La caída de ventas en el Sur puede estar relacionada con la disponibilidad y la competitividad promocional de la marca, según el equipo comercial.» → menciona Sur, pero la fuente citada no lo respalda (REP-S1-1 no menciona ninguna región)

**Evidencia consultada:** REP-S5-1, REP-S1-1, CTX-CONVENCIONES-DE-ANALISIS, REP-S7-1, REP-S0-1, INV-05

**Avisos:**
- Se eliminó la fuente REP-S1-1 (no menciona ninguna región): la afirmación habla de Sur.

**Cálculos**

```
[CALC-1] Brío — venta neta (COP) — por región (sin filtros), T2 frente a T1
- Sur: T1 363.431.600 | T2 253.404.600 | variación -110.027.000 (-30,27 %) | aporta 73,1 % del cambio total
- Centro: T1 314.314.900 | T2 285.924.400 | variación -28.390.500 (-9,03 %) | aporta 18,9 % del cambio total
- Occidente: T1 253.170.900 | T2 239.821.500 | variación -13.349.400 (-5,27 %) | aporta 8,9 % del cambio total
- Norte: T1 272.236.000 | T2 262.773.400 | variación -9.462.600 (-3,48 %) | aporta 6,3 % del cambio total
- Oriente: T1 230.096.600 | T2 240.789.800 | variación 10.693.200 (4,65 %) | aporta -7,1 % del cambio total
- Total Brío: T1 1.433.250.000 | T2 1.282.713.700 | variación -150.536.300 (-10,50 %)
Mayor caída: Sur (-30,27 %)
```

```
[CONTRASTE-1] Reporte comercial frente a datos de ventas (variación T2 vs T1 por región). Ya verificado con código: úsalo tal cual y no lo contradigas.
- Norte: reporte -4 % | datos -3,48 % | diferencia -0,52 pp | coincide (tolerancia ±1 pp)
- Centro: reporte -23 % | datos -9,03 % | diferencia -13,97 pp | NO coincide (tolerancia ±1 pp)
- Sur: reporte -7 % | datos -30,27 % | diferencia +23,27 pp | NO coincide (tolerancia ±1 pp)
- Oriente: reporte +4 % | datos 4,65 % | diferencia -0,65 pp | coincide (tolerancia ±1 pp)
- Occidente: reporte -6 % | datos -5,27 % | diferencia -0,73 pp | coincide (tolerancia ±1 pp)
Región con mayor caída: reporte Centro | datos Sur (-30,27 %) | NO coincide
Afirmación REP-01: CONTRADICHA
Afirmación REP-02: NO VERIFICABLE (la base de ventas no contiene variables de promoción ni de precio (columnas: registro_id, mes, marca, region, canal, punto_venta_id, presentacion, unidades, venta_neta_cop))
Veredicto: El reporte NO es consistente con los datos: 2 de 5 regiones difieren más de 1 pp (Centro, Sur) y señala a Centro como la región de mayor caída, cuando los datos indican Sur (-30,27 %).
```

## Pregunta 4 · Causalidad

> ¿Podemos afirmar que las promociones de la competencia causaron la caída de ventas? ¿Qué evidencia lo sustenta o limita?

**Configuración:** modelo `llama3` · búsqueda `semantic` · plan `causalidad` (modelo) · 15.5 s

**Resumen:** No se puede afirmar causalidad. La base de ventas no contiene variables de promoción ni de precio: los datos no permiten medir el efecto de las promociones de la competencia. _(redactado por el sistema a partir de los cálculos)_

**Hechos**

- Mayor caída de ventas netas (COP) por región: Sur, -30,27 % (T1 363.431.600 → T2 253.404.600). Explica el 73,1 % de la caída total de Brío. _[CALC-1; calculo]_
- Crecieron: Oriente +4,65 %. _[CALC-1; calculo]_
- Total Brío (ventas netas (COP)): -10,50 % (T1 1.433.250.000 → T2 1.282.713.700). _[CALC-1; calculo]_

**Contradicciones entre fuentes**

- REP-02 estima que las promociones de competidores explican aproximadamente el 70 % de la contracción de ventas de Brío, mientras que INV-04 no encontró evidencia concluyente sobre promociones. _[REP-02, INV-04; modelo]_

**Hipótesis (no demostradas)**

- La competencia podría explicar parte de la contracción de ventas de Brío, según REP-02 y OC-CEN-02. _[REP-02, OC-CEN-02; modelo]_

**Información insuficiente**

- La base de ventas no contiene variables de promoción ni de precio: los datos no permiten medir el efecto de las promociones de la competencia. _[calculo]_
- La evidencia cualitativa (entrevistas y visitas) no mide impacto: no hubo grupo de control ni mediciones antes y después, así que no establece causalidad. _[REP-02, INV-04, INV-LIM-02; calculo]_
- No se puede afirmar que las promociones de la competencia causaron la caída de ventas de Brío, ya que no se midieron elasticidades ni se verificó un cambio causal asociado con una promoción o una ruptura de inventario, según INV-LIM-02. _[modelo]_

**Evidencia consultada:** REP-02, INV-04, OC-CEN-02, INV-REGION-CENTRO, REP-S1-1, REP-S7-1, INV-LIM-02

**Cálculos**

```
[CALC-1] Brío — venta neta (COP) — por región (sin filtros), T2 frente a T1
- Sur: T1 363.431.600 | T2 253.404.600 | variación -110.027.000 (-30,27 %) | aporta 73,1 % del cambio total
- Centro: T1 314.314.900 | T2 285.924.400 | variación -28.390.500 (-9,03 %) | aporta 18,9 % del cambio total
- Occidente: T1 253.170.900 | T2 239.821.500 | variación -13.349.400 (-5,27 %) | aporta 8,9 % del cambio total
- Norte: T1 272.236.000 | T2 262.773.400 | variación -9.462.600 (-3,48 %) | aporta 6,3 % del cambio total
- Oriente: T1 230.096.600 | T2 240.789.800 | variación 10.693.200 (4,65 %) | aporta -7,1 % del cambio total
- Total Brío: T1 1.433.250.000 | T2 1.282.713.700 | variación -150.536.300 (-10,50 %)
Mayor caída: Sur (-30,27 %)
```

## Pregunta 5 · Consulta nueva (cuantitativa con filtro)

> ¿Cómo variaron las unidades vendidas de Brío en la presentación Familiar entre T1 y T2, y en qué canal fue mayor la caída?

**Configuración:** modelo `llama3` · búsqueda `semantic` · plan `cuantitativa` (modelo) · 12.9 s

**Resumen:** Caída similar en todos los grupos por presentación y canal (presentacion = Familiar): entre -10,53 % y -11,23 % (diferencias menores a 1 punto porcentual). _(redactado por el sistema a partir de los cálculos)_

**Hechos**

- Caída similar en todos los grupos por presentación y canal (presentacion = Familiar): entre -10,53 % y -11,23 % (diferencias menores a 1 punto porcentual). _[CALC-1; calculo]_
- Total Brío con filtro presentacion = Familiar (unidades): -10,86 % (T1 43.964 → T2 39.191). _[CALC-1; calculo]_

**Descartado por el validador** (afirmaciones del modelo que no pasaron las verificaciones)

- «La caída en las unidades vendidas de Brío en la presentación Familiar entre T1 y T2 puede estar relacionada con la competencia en el mercado.» → es una hipótesis y no cita ningún documento que la sugiera
- «La variación de unidades vendidas de Brío en la presentación Familiar entre T1 y T2 fue negativa en todos los grupos, con caídas similares entre -10,53% y -11,23%. La mayor caída se dio en el canal Familiar / Moderno.» → señala una mayor caída, pero los cálculos muestran caídas similares en todos los grupos (diferencias menores a 1 punto porcentual)

**Cálculos**

```
[CALC-1] Brío — unidades — por presentación y canal (presentacion = Familiar), T2 frente a T1
- Familiar / Moderno: T1 20.278 | T2 18.000 | variación -2.278 (-11,23 %) | aporta 47,7 % del cambio total
- Familiar / Tradicional: T1 23.686 | T2 21.191 | variación -2.495 (-10,53 %) | aporta 52,3 % del cambio total
- Total Brío: T1 43.964 | T2 39.191 | variación -4.773 (-10,86 %)
Caída similar en todos los grupos (entre -10,53 % y -11,23 %): las diferencias entre grupos son menores a 1 punto porcentual.
```

## Pregunta 6 · Consulta nueva (integrada en una región)

> ¿Qué dicen los comerciantes sobre los faltantes de producto en el Sur?

**Configuración:** modelo `llama3` · búsqueda `semantic` · plan `integrada` (plan por defecto) · 8.1 s

**Resumen:** Los comerciantes en el Sur mencionan la falta de producto como una posible razón para la caída de ventas de Brío.

**Hechos**

- Mayor caída de ventas netas (COP) por región (region = Sur): Sur, -30,27 % (T1 363.431.600 → T2 253.404.600) _[CALC-1; calculo]_
- Total Brío con filtro region = Sur (ventas netas (COP)): -30,27 % (T1 363.431.600 → T2 253.404.600). _[CALC-1; calculo]_

**Hipótesis (no demostradas)**

- La falta de producto puede haber contribuido a la caída de ventas de Brío en el Sur. _[OC-SUR-01, REP-03, INV-02; modelo]_

**Información insuficiente**

- No se puede determinar cuánto de la caída de ventas se explica por disponibilidad o promoción. _[modelo]_

**Evidencia consultada:** INV-01, OC-SUR-01, REP-03, INV-02, OC-REGION-SUR, INV-REGION-SUR

**Avisos:**
- Planificador: El modelo marcó la pregunta como fuera de alcance, pero trata temas del análisis comercial; se trató como una pregunta en alcance.

**Cálculos**

```
[CALC-1] Brío — venta neta (COP) — por región (region = Sur), T2 frente a T1
- Sur: T1 363.431.600 | T2 253.404.600 | variación -110.027.000 (-30,27 %) | aporta 100,0 % del cambio total
- Total Brío: T1 363.431.600 | T2 253.404.600 | variación -110.027.000 (-30,27 %)
Mayor caída: Sur (-30,27 %)
```

