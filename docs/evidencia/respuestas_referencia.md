# Respuestas de referencia del sistema

- Fecha: 2026-10-21 20:25
- Python: 3.13.14 · Sistema: Windows 11
- Modelo de generación: llama3 · Modelo de embeddings: bge-m3 · Búsqueda por defecto: semantic

> **Qué es este archivo:** la salida de una ejecución real del sistema en la fecha indicada. **No son respuestas fijas.** Un modelo de lenguaje local puede redactar de forma distinta en cada ejecución. Lo que no cambia son las cifras, el veredicto del contraste con el reporte y las limitaciones de causalidad, porque salen de código y no del modelo. Para regenerarlo: `python scripts/generate_evidence.py --answers`.

## Pregunta 1 · Cuantitativa

> ¿Cuál fue la región con mayor caída porcentual en ventas netas de Brío entre T1 y T2 de 2026? Muestra los valores y el cálculo.

**Configuración:** modelo `llama3` · búsqueda `semantic` · plan `cuantitativa` (modelo) · 6.1 s

**Resumen:** La región con mayor caída porcentual en ventas netas de Brío entre T1 y T2 de 2026 fue el Sur, con una variación de -30,27%.

**Hechos**

- Mayor caída de ventas netas (COP) por región: Sur, -30,27 % (T1 363.431.600 → T2 253.404.600). Explica el 73,1 % de la caída total de Brío. _[CALC-1; calculo]_
- Crecieron: Oriente +4,65 %. _[CALC-1; calculo]_
- Total Brío (ventas netas (COP)): -10,50 % (T1 1.433.250.000 → T2 1.282.713.700). _[CALC-1; calculo]_

**Descartado por el validador** (afirmaciones del modelo que no pasaron las verificaciones)

- «La caída en ventas netas en el Sur puede estar relacionada con la competencia en el mercado regional.» → es una hipótesis y no cita ningún documento que la sugiera

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

**Configuración:** modelo `llama3` · búsqueda `semantic` · plan `integrada` (modelo) · 9.7 s

**Resumen:** El deterioro de Brío se asocia con la caída de ventas en el Sur, que aporta el 73,1% del cambio total. La disponibilidad y la competitividad promocional de la marca pueden ser factores clave.

**Hechos**

- Mayor caída de ventas netas (COP) por región: Sur, -30,27 % (T1 363.431.600 → T2 253.404.600). Explica el 73,1 % de la caída total de Brío. _[CALC-1; calculo]_
- Crecieron: Oriente +4,65 %. _[CALC-1; calculo]_
- Total Brío (ventas netas (COP)): -10,50 % (T1 1.433.250.000 → T2 1.282.713.700). _[CALC-1; calculo]_

**Hipótesis (no demostradas)**

- La disponibilidad y la competitividad promocional de la marca pueden ser factores clave en el deterioro de Brío. _[REP-S1-1, INV-02, OC-SUR-01; modelo]_

**Información insuficiente**

- No se puede afirmar con la información disponible qué tan importante es la disponibilidad y la competitividad promocional en el deterioro de Brío. _[modelo]_

**Descartado por el validador** (afirmaciones del modelo que no pasaron las verificaciones)

- «Dinámica competitiva. Se estima que las promociones de competidores explican aproximadamente el 70 % de la contracción de ventas de Brío.» → es una afirmación del reporte comercial, no un hecho verificado
- «No se encontraron contradicciones entre fuentes.» → dice que no hay contradicciones: no es una contradicción

**Evidencia consultada:** REP-02, REP-S1-1, ENT-CEN-04, INV-02, INV-S0-2, OC-SUR-01

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

## Pregunta 3 · Contraste con el reporte comercial

> ¿Son consistentes las principales conclusiones del reporte comercial con los datos de ventas?

**Configuración:** modelo `llama3` · búsqueda `semantic` · plan `contraste_reporte` (modelo) · 12.0 s

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

**Descartado por el validador** (afirmaciones del modelo que no pasaron las verificaciones)

- «La caída de ventas en el Sur puede estar relacionada con la disponibilidad y la competitividad promocional de la marca, según el equipo comercial.» → menciona Sur, pero la fuente citada no lo respalda (REP-S5-1 no menciona ninguna región; REP-S1-1 no menciona ninguna región)
- «No se puede afirmar que la caída de ventas en el Sur esté relacionada con la disponibilidad y la competitividad promocional de la marca, ya que no se dispone de información sobre la base de ventas que contenga variables de promoción ni de precio.» → dice que no hay información de la base de ventas, pero los cálculos se hicieron con ella

**Evidencia consultada:** REP-S5-1, REP-S1-1, CTX-CONVENCIONES-DE-ANALISIS, REP-S7-1, REP-S0-1, INV-05

**Avisos:**
- Se eliminó la fuente REP-S5-1 (no menciona ninguna región): la afirmación habla de Sur.
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

**Configuración:** modelo `llama3` · búsqueda `semantic` · plan `causalidad` (modelo) · 11.8 s

**Resumen:** No se puede afirmar causalidad. La base de ventas no contiene variables de promoción ni de precio: los datos no permiten medir el efecto de las promociones de la competencia. _(redactado por el sistema a partir de los cálculos)_

**Hechos**

- Mayor caída de ventas netas (COP) por región: Sur, -30,27 % (T1 363.431.600 → T2 253.404.600). Explica el 73,1 % de la caída total de Brío. _[CALC-1; calculo]_
- Crecieron: Oriente +4,65 %. _[CALC-1; calculo]_
- Total Brío (ventas netas (COP)): -10,50 % (T1 1.433.250.000 → T2 1.282.713.700). _[CALC-1; calculo]_

**Hipótesis (no demostradas)**

- La temporalidad de los relatos puede limitar la capacidad de inferir causas _[INV-LIM-02; modelo]_

**Información insuficiente**

- La base de ventas no contiene variables de promoción ni de precio: los datos no permiten medir el efecto de las promociones de la competencia. _[calculo]_
- La evidencia cualitativa (entrevistas y visitas) no mide impacto: no hubo grupo de control ni mediciones antes y después, así que no establece causalidad. _[REP-02, INV-04, INV-LIM-02; calculo]_
- No se puede afirmar que las promociones de la competencia causaron la caída de ventas de Brío _[modelo]_

**Descartado por el validador** (afirmaciones del modelo que no pasaron las verificaciones)

- «Se estima que las promociones de competidores explican aproximadamente el 70 % de la contracción de ventas de Brío» → es una afirmación del reporte comercial, no un hecho verificado
- «La mayor caída en ventas en el Sur podría estar relacionada con la mayor presencia de promociones de la competencia en esta región» → menciona Sur, pero la fuente citada no lo respalda (OC-CEN-02 dice de Centro)
- «La evidencia no respalda inferir la mayor caída en ventas en el Sur» → no cita ninguna fuente válida de esta consulta

**Evidencia consultada:** REP-02, INV-04, OC-CEN-02, INV-REGION-CENTRO, REP-S1-1, REP-S7-1, INV-LIM-02

**Avisos:**
- Se eliminó la fuente OC-CEN-02 (dice de Centro): la afirmación habla de Sur.
- Se eliminó la fuente INV-REGION-CENTRO (dice de Centro): la afirmación habla de Sur.

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

