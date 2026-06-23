"""T4.5 — Rule-based structured failure diagnostics.

Produces a structured ``DiagnosticReport`` for any strategy based on
backtest metrics + gene shape, *without* invoking an LLM.  Used both:

* Standalone (cheap, deterministic) when running with LLM disabled.
* As pre-processing before an LLM call so the prompt carries hard
  facts rather than re-asking the model to read metrics.

Complementary to ``llm/diagnostics.py`` (which returns a single
failure-mode string for LLM prompts).  This module instead returns
**all** findings with severity tags so callers can filter, sort, or
render them.  No dependency on the LLM stack.

Categories of findings
----------------------
``trading``      — trade count / coverage problems
``risk``         — drawdown, tail-risk, exposure
``efficiency``   — win-rate, sharpe, profit-per-trade
``robustness``   — walk-forward gap, holdout degradation, cross-pair var
``structure``    — gene complexity / anti-patterns / suspicious mix

Severity scale (low/medium/high/critical) maps roughly to:
  critical = strategy must be discarded
  high     = strategy is unlikely to survive HoF
  medium   = strategy is plausible but degraded
  low      = informational / nuisance flag
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------


@dataclass
class Finding:
    code: str          # short stable id, e.g. "no_trades", "high_dd"
    severity: str      # "low" | "medium" | "high" | "critical"
    category: str      # see module docstring
    message: str       # human-readable
    detail: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DiagnosticReport:
    findings: List[Finding] = field(default_factory=list)
    score: float = 1.0  # 1.0 healthy, 0.0 catastrophic
    summary: str = ""

    def by_severity(self, *severities: str) -> List[Finding]:
        s = set(severities)
        return [f for f in self.findings if f.severity in s]

    def by_category(self, *categories: str) -> List[Finding]:
        c = set(categories)
        return [f for f in self.findings if f.category in c]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "summary": self.summary,
            "findings": [
                {
                    "code": f.code,
                    "severity": f.severity,
                    "category": f.category,
                    "message": f.message,
                    "detail": dict(f.detail),
                }
                for f in self.findings
            ],
        }


# ---------------------------------------------------------------------------
# Thresholds (overridable via config)
# ---------------------------------------------------------------------------


DEFAULT_THRESHOLDS: Dict[str, float] = {
    "min_trades": 5,
    "low_trades": 15,
    "max_drawdown": 0.20,
    "critical_drawdown": 0.40,
    "min_win_rate": 30.0,
    "min_sharpe": 0.5,
    "negative_sharpe": 0.0,
    "max_complexity": 8,
    "wf_gap_threshold": 0.15,
    "holdout_degradation_threshold": 0.30,
    "high_cross_pair_cov": 1.0,
    "tiny_avg_profit": 0.001,
}


_SEVERITY_WEIGHT = {"low": 0.05, "medium": 0.15, "high": 0.35, "critical": 0.65}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get(metrics: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in metrics and metrics[k] is not None:
            return metrics[k]
    return default


def _compute_score(findings: Sequence[Finding]) -> float:
    if not findings:
        return 1.0
    penalty = sum(_SEVERITY_WEIGHT.get(f.severity, 0.1) for f in findings)
    return max(0.0, 1.0 - penalty)


def _summary(findings: Sequence[Finding]) -> str:
    if not findings:
        return "healthy"
    crits = [f for f in findings if f.severity == "critical"]
    if crits:
        return f"critical: {crits[0].code}"
    highs = [f for f in findings if f.severity == "high"]
    if highs:
        return f"degraded: {highs[0].code}"
    return f"{len(findings)} minor findings"


# ---------------------------------------------------------------------------
# Diagnosis
# ---------------------------------------------------------------------------


def diagnose(
    metrics: Dict[str, Any],
    *,
    thresholds: Optional[Dict[str, float]] = None,
    anti_pattern_report: Optional[Dict[str, Any]] = None,
) -> DiagnosticReport:
    """Build a structured ``DiagnosticReport`` for a strategy.

    ``metrics`` keys we look at (all optional, missing → no finding):
      num_trades, total_trades, max_drawdown, win_rate, sharpe_ratio,
      profit, avg_profit, indicator_count, condition_count, wf_gap,
      holdout_degradation, cross_pair_cov, fitness, val_fitness.

    ``anti_pattern_report`` is the dict returned by
    ``anti_pattern.summarise_anti_patterns`` — when present and
    non-empty, a finding is added describing the bad pairs.
    """
    th = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        th.update(thresholds)

    findings: List[Finding] = []

    num_trades = float(_get(metrics, "num_trades", "total_trades", default=0))
    dd = float(_get(metrics, "max_drawdown", "drawdown", default=0))
    win_rate = _get(metrics, "win_rate")
    sharpe = _get(metrics, "sharpe_ratio", "sharpe")
    profit = _get(metrics, "profit", "total_profit")
    avg_profit = _get(metrics, "avg_profit")
    ind_count = int(_get(metrics, "indicator_count", default=0))
    cond_count = int(_get(metrics, "condition_count", default=0))
    wf_gap = _get(metrics, "wf_gap")
    holdout_deg = _get(metrics, "holdout_degradation")
    cov = _get(metrics, "cross_pair_cov", "cross_pair_cov_profit")

    # ---- Trading (must trade enough to be evaluable) ----
    if num_trades == 0:
        findings.append(
            Finding(
                code="no_trades",
                severity="critical",
                category="trading",
                message="Strategy produced 0 trades — conditions never matched.",
                detail={"num_trades": 0},
            )
        )
    elif num_trades < th["min_trades"]:
        findings.append(
            Finding(
                code="too_few_trades",
                severity="high",
                category="trading",
                message=f"Only {int(num_trades)} trades (<{int(th['min_trades'])}) — sample too small.",
                detail={"num_trades": num_trades, "threshold": th["min_trades"]},
            )
        )
    elif num_trades < th["low_trades"]:
        findings.append(
            Finding(
                code="low_trade_count",
                severity="low",
                category="trading",
                message=f"Only {int(num_trades)} trades — statistical confidence is weak.",
                detail={"num_trades": num_trades},
            )
        )

    # ---- Risk ----
    if dd >= th["critical_drawdown"]:
        findings.append(
            Finding(
                code="critical_drawdown",
                severity="critical",
                category="risk",
                message=f"Drawdown {dd:.1%} exceeds critical threshold {th['critical_drawdown']:.0%}.",
                detail={"max_drawdown": dd, "threshold": th["critical_drawdown"]},
            )
        )
    elif dd >= th["max_drawdown"]:
        findings.append(
            Finding(
                code="high_drawdown",
                severity="high",
                category="risk",
                message=f"Drawdown {dd:.1%} exceeds tolerance {th['max_drawdown']:.0%}.",
                detail={"max_drawdown": dd, "threshold": th["max_drawdown"]},
            )
        )

    # ---- Efficiency ----
    if win_rate is not None and num_trades >= th["min_trades"]:
        wr = float(win_rate)
        if wr < th["min_win_rate"]:
            findings.append(
                Finding(
                    code="low_win_rate",
                    severity="medium",
                    category="efficiency",
                    message=f"Win-rate {wr:.1f}% < {th['min_win_rate']:.0f}%.",
                    detail={"win_rate": wr, "threshold": th["min_win_rate"]},
                )
            )

    if sharpe is not None and num_trades >= th["min_trades"]:
        sh = float(sharpe)
        if sh < th["negative_sharpe"]:
            findings.append(
                Finding(
                    code="negative_sharpe",
                    severity="high",
                    category="efficiency",
                    message=f"Sharpe {sh:.2f} is negative — losing risk-adjusted.",
                    detail={"sharpe_ratio": sh},
                )
            )
        elif sh < th["min_sharpe"]:
            findings.append(
                Finding(
                    code="weak_sharpe",
                    severity="medium",
                    category="efficiency",
                    message=f"Sharpe {sh:.2f} < {th['min_sharpe']:.2f}.",
                    detail={"sharpe_ratio": sh, "threshold": th["min_sharpe"]},
                )
            )

    if (
        avg_profit is not None
        and num_trades >= th["min_trades"]
        and abs(float(avg_profit)) < th["tiny_avg_profit"]
    ):
        findings.append(
            Finding(
                code="tiny_edge",
                severity="medium",
                category="efficiency",
                message=f"Average profit per trade {float(avg_profit):.4f} is below fee/slippage noise.",
                detail={"avg_profit": float(avg_profit)},
            )
        )

    # ---- Robustness ----
    if wf_gap is not None and float(wf_gap) > th["wf_gap_threshold"]:
        findings.append(
            Finding(
                code="walkforward_gap",
                severity="high",
                category="robustness",
                message=f"Walk-forward fitness gap {float(wf_gap):.2f} suggests overfitting.",
                detail={"wf_gap": float(wf_gap), "threshold": th["wf_gap_threshold"]},
            )
        )
    if (
        holdout_deg is not None
        and float(holdout_deg) > th["holdout_degradation_threshold"]
    ):
        findings.append(
            Finding(
                code="holdout_degradation",
                severity="high",
                category="robustness",
                message=f"Holdout degradation {float(holdout_deg):.2f} — strategy unstable out-of-sample.",
                detail={
                    "holdout_degradation": float(holdout_deg),
                    "threshold": th["holdout_degradation_threshold"],
                },
            )
        )
    if cov is not None and float(cov) > th["high_cross_pair_cov"]:
        findings.append(
            Finding(
                code="cross_pair_instability",
                severity="medium",
                category="robustness",
                message=f"Cross-pair profit CoV {float(cov):.2f} > {th['high_cross_pair_cov']:.2f} — relies on a few pairs.",
                detail={"cross_pair_cov": float(cov)},
            )
        )

    # ---- Structure / anti-patterns ----
    if (ind_count + cond_count) > th["max_complexity"]:
        findings.append(
            Finding(
                code="excessive_complexity",
                severity="medium",
                category="structure",
                message=f"{ind_count} indicators + {cond_count} conditions exceeds budget {int(th['max_complexity'])}.",
                detail={
                    "indicator_count": ind_count,
                    "condition_count": cond_count,
                    "budget": th["max_complexity"],
                },
            )
        )
    if anti_pattern_report and anti_pattern_report.get("n_pairs", 0) > 0:
        pairs = anti_pattern_report.get("pairs", [])
        first = pairs[0] if pairs else {}
        findings.append(
            Finding(
                code="anti_pattern_pair",
                severity="medium",
                category="structure",
                message=(
                    f"Indicator combination contains {anti_pattern_report['n_pairs']} "
                    f"known under-performing pair(s), e.g. {first.get('a')}×{first.get('b')}."
                ),
                detail={
                    "pairs": pairs,
                    "multiplier": anti_pattern_report.get("multiplier", 1.0),
                },
            )
        )

    # Order by severity for stable consumption
    sev_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    findings.sort(key=lambda f: (sev_order.get(f.severity, 9), f.code))

    return DiagnosticReport(
        findings=findings,
        score=_compute_score(findings),
        summary=_summary(findings),
    )


def diagnose_to_text(report: DiagnosticReport, *, max_lines: int = 8) -> str:
    """Render a compact human-readable summary suitable for logs or
    LLM prompt context.  Most severe first."""
    if not report.findings:
        return "✓ healthy"
    lines = [f"[{report.summary}]"]
    for f in report.findings[:max_lines]:
        lines.append(f"  - ({f.severity}) {f.code}: {f.message}")
    if len(report.findings) > max_lines:
        lines.append(f"  … {len(report.findings) - max_lines} more")
    return "\n".join(lines)


__all__ = [
    "Finding",
    "DiagnosticReport",
    "DEFAULT_THRESHOLDS",
    "diagnose",
    "diagnose_to_text",
]
