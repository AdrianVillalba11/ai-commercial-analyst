"""
REPORTE DE CALIDAD DE DATOS

Qué hace: guarda, fila por fila, cada corrección o exclusión que hizo la limpieza.
Con qué se conecta: lo llena data/cleaning.py; lo muestra la API (panel "Calidad de datos") y lo
consulta analytics/metrics.py para avisar cuando una cifra depende de filas excluidas.
Idea clave: nada se descarta sin dejar constancia (qué regla, qué valor, qué se hizo).
"""
from collections import Counter
from dataclasses import asdict, dataclass, field

CORRECTED = "corregido"
EXCLUDED = "excluido"


@dataclass(frozen=True)
class Issue:
    """Un hallazgo de calidad: una fila corregida o excluida y por qué."""

    registro_id: str
    rule: str
    field: str
    original_value: str
    action: str  # CORRECTED o EXCLUDED
    detail: str
    marca: str = ""


@dataclass
class QualityReport:
    """Contadores de filas y lista de hallazgos de toda la limpieza."""

    rows_in: int = 0
    rows_out: int = 0
    issues: list[Issue] = field(default_factory=list)

    def add(self, issue: Issue) -> None:
        """Registra un hallazgo."""
        self.issues.append(issue)

    def for_brand(self, marca: str) -> list[Issue]:
        """Hallazgos de una marca (sin distinguir mayúsculas)."""
        return [i for i in self.issues if i.marca.casefold() == marca.casefold()]

    def summary(self) -> dict:
        """Totales y conteo por regla/acción, para el resumen del panel."""
        by_rule = Counter((i.rule, i.action) for i in self.issues)
        return {
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "rows_excluded": self.rows_in - self.rows_out,
            "issues_total": len(self.issues),
            "by_rule": [
                {"rule": rule, "action": action, "count": n}
                for (rule, action), n in sorted(by_rule.items())
            ],
        }

    def to_dict(self) -> dict:
        """Versión lista para convertir a JSON."""
        return {"summary": self.summary(), "issues": [asdict(i) for i in self.issues]}
