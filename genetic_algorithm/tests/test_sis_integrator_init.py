"""Regression tests for SIS initialization helpers."""

from genetic_algorithm.intelligence.sis_integrator import AdaptiveWeightTracker


def test_adaptive_weight_tracker_configure_for_short_run_does_not_require_sis_config():
    tracker = AdaptiveWeightTracker({"RSI": 1.0}, {"<": 1.0}, evidence_trust_max=0.6)

    tracker.configure_for_run_length(6)

    assert tracker._max_generations == 6
    assert tracker._evidence_trust_initial == 0.3


def test_immigrant_disabled_still_allows_sis_event_logging(tmp_path):
    import logging
    from types import SimpleNamespace

    from genetic_algorithm.intelligence.sis_integrator import SISIntegrator

    log_path = tmp_path / "sis_events.jsonl"
    sis = SISIntegrator(
        {
            "genetic_algorithm": {"generations": 6},
            "sis": {
                "enabled": True,
                "models_dir": "genetic_algorithm/ml/models",
                "corpus_path": "genetic_algorithm/data/strategy_corpus_clustered.parquet",
                "log_file": str(log_path),
                "hooks": {
                    "seed_filtering": False,
                    "immigrants": False,
                    "indicator_weights": True,
                    "operator_weights": True,
                    "synergy_weights": True,
                },
            },
            "output": {"dir": str(tmp_path)},
        },
        logging.getLogger("test-sis-event-logging"),
    )
    gene = SimpleNamespace(
        indicators=[SimpleNamespace(type="RSI")],
        entry_conditions=[SimpleNamespace(operator="<")],
    )
    individual = SimpleNamespace(strategy_gene=gene, fitness=0.5)
    ga = SimpleNamespace(population=SimpleNamespace(individuals=[individual]))

    assert sis.immigrant_provider(ga, 1) == []

    assert log_path.exists()
    assert '"n_sis_immigrants": 0' in log_path.read_text()
