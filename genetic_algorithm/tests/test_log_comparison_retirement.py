"""Regression tests for retiring log-derived GA decisions."""

from __future__ import annotations

from genetic_algorithm.scripts.aggregate_comparison import main as aggregate_main
from genetic_algorithm.scripts.benchmark_report import generate_report, try_generate_charts


def test_aggregate_comparison_fails_closed(capsys) -> None:
    assert aggregate_main([]) == 2
    captured = capsys.readouterr()
    assert "retired" in captured.err
    assert "candidate-bound" in captured.err


def test_benchmark_report_labels_logs_and_disables_ranking(tmp_path) -> None:
    (tmp_path / "run1_baseline.status").write_text("COMPLETED")
    (tmp_path / "run1_baseline.log").write_text(
        "Best fitness: 999999\nAverage fitness: 888888\nEVOLUTION COMPLETE profit=777777\n"
    )

    report = generate_report(tmp_path)

    assert "LEGACY DIAGNOSTIC ONLY — NOT DECISION-ELIGIBLE" in report
    assert "AUTOMATED RANKING DISABLED" in report
    assert "RANKINGS" not in report
    assert "KEY COMPARISONS (automated)" not in report
    assert "outperforms" not in report


def test_benchmark_comparison_charts_are_disabled(tmp_path, capsys) -> None:
    try_generate_charts(tmp_path)

    assert "charts disabled" in capsys.readouterr().out
    assert not (tmp_path / "benchmark_comparison.png").exists()
