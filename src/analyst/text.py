"""
UTILIDADES DE TEXTO COMPARTIDAS

Qué hace: concentra en un solo lugar la normalización de texto (quitar tildes y mayúsculas)
y las expresiones regulares que reconocen los nombres de las regiones.
Con qué se conecta: lo usan las métricas, el troceado de documentos, el buscador léxico,
el planificador y las comprobaciones de guardrails.
Idea clave: si un módulo normaliza distinto que otro, dejan de coincidir las búsquedas y las
validaciones. Por eso hay una sola definición y todos la importan.
"""

import re
import unicodedata

from analyst.data.cleaning import REGIONS


# =====================================================================
# BLOQUE: Normalización
# =====================================================================
def fold(text: str) -> str:
    """Pasa a minúsculas y quita tildes: 'Brío', 'BRIO' y 'brio' quedan iguales."""
    decomposed = unicodedata.normalize("NFKD", str(text).casefold())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


# =====================================================================
# BLOQUE: Reconocer regiones en un texto
# Hay dos variantes a propósito:
#  - REGION_EXACT: exige la mayúscula inicial. Se usa con el texto de los documentos para no
#    confundir la región "Centro" con "el centro de la ciudad".
#  - REGION_FOLDED: se usa sobre texto ya normalizado con fold() (minúsculas, sin tildes),
#    que es como se revisa lo que escribe el modelo.
# =====================================================================
REGION_EXACT = re.compile(r"\b(" + "|".join(REGIONS) + r")\b")
REGION_FOLDED = re.compile(r"\b(" + "|".join(fold(r) for r in REGIONS) + r")\b")
