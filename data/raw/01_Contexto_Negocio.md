# Almena Foods | Contexto de negocio

## Situación
Almena Foods comercializa cinco marcas de consumo masivo a través de canales tradicionales y modernos. El equipo de mercadeo estudia el desempeño de **Brío** durante el primer semestre de 2026 y necesita contrastar datos comerciales con investigaciones de consumidor y observaciones en puntos de venta.

Los analistas utilizan actualmente una base consolidada de ventas, un estudio cualitativo, notas de campo y un reporte comercial. Estas fuentes pueden contener diferencias de calidad, alcance o interpretación. La gerencia necesita que las conclusiones cuantitativas sean verificables y que las hipótesis comerciales se distingan de los hechos.

## Información disponible
- `02_Ventas_2026.csv`: ventas mensuales de cinco marcas entre enero y junio de 2026, en cinco regiones y dos canales.
- `03_Investigacion_Consumidor.pdf`: investigación cualitativa sobre preferencias y hábitos de compra.
- `04_Observaciones_Canales.docx`: observaciones registradas durante visitas a puntos de venta.
- `05_Reporte_Comercial.pdf`: diagnóstico y recomendaciones de un analista comercial.

## Diccionario de la base de ventas

| Campo | Significado |
| --- | --- |
| `registro_id` | Identificador de un registro comercial. Debe representar una observación única. |
| `mes` | Mes de registro, con formato previsto `AAAA-MM`. |
| `marca` | Marca comercial. En los archivos de datos, `Brio` representa a Brío. |
| `region` | Región comercial: Norte, Centro, Sur, Oriente u Occidente. |
| `canal` | Tradicional o Moderno. |
| `punto_venta_id` | Identificador operativo del punto de venta. |
| `presentacion` | Presentación del producto: Individual o Familiar. |
| `unidades` | Unidades registradas en la observación. |
| `venta_neta_cop` | Valor neto registrado en pesos colombianos (COP), sin separadores de miles. |

Cada fila representa una observación comercial mensual para una combinación de marca, región, canal, punto de venta y presentación. Pueden existir varias observaciones para una misma combinación.

## Convenciones de análisis
- **Primer trimestre (T1):** enero, febrero y marzo de 2026.
- **Segundo trimestre (T2):** abril, mayo y junio de 2026.
- **Variación absoluta de ventas:** venta neta T2 menos venta neta T1.
- **Variación porcentual:** (venta neta T2 / venta neta T1 - 1) × 100.
- La comparación se refiere a **venta neta en COP**, salvo que la consulta solicite explícitamente unidades.

No se ha entregado una política para imputar datos faltantes o interpretar movimientos negativos. Cualquier decisión de calidad de datos que afecte el análisis debe explicitarse y poder verificarse.
