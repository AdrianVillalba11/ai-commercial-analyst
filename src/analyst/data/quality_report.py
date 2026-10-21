"""Data quality report: a traceable list of every correction or exclusion made."""
from collections import Counter
from dataclasses import asdict, dataclass, field

CORRECTED = "corregido"
EXCLUDED = "excluido"


@dataclass(frozen=True)
class Issue:
    registro_id: str
    rule: str
    field: str
    original_value: str
    action: str  # CORRECTED or EXCLUDED
    detail: str
    marca: str = ""


@dataclass
class QualityReport:
    rows_in: int = 0
    rows_out: int = 0
    issues: list[Issue] = field(default_factory=list)

    def add(self, issue: Issue) -> None:
        self.issues.append(issue)

    def for_brand(self, marca: str) -> list[Issue]:
        return [i for i in self.issues if i.marca.casefold() == marca.casefold()]

    def summary(self) -> dict:
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
        return {"summary": self.summary(), "issues": [asdict(i) for i in self.issues]}
