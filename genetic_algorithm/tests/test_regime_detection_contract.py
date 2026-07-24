"""Reachability regression for the active regime auto-detection path."""

from __future__ import annotations

import pandas as pd

from genetic_algorithm.market import regime_aware


def test_auto_detect_constructs_detector_with_canonical_method(monkeypatch):
    observed: dict[str, object] = {}

    class FakeDetector:
        def __init__(self, *, method):
            observed["method"] = method

        def classify_periods(self, **kwargs):
            observed["classify"] = kwargs
            return ["segment"]

        def get_balanced_segments(self, segments, *, segments_per_regime):
            observed["balanced"] = (segments, segments_per_regime)
            return segments

        def split_segments_by_role(self, segments, **kwargs):
            observed["split"] = (segments, kwargs)
            return {
                "optimization": list(segments),
                "model_selection": [],
                "holdout": [],
            }

    monkeypatch.setattr(
        regime_aware,
        "load_ohlcv_data",
        lambda **_kwargs: pd.DataFrame({"close": [100.0]}),
    )
    monkeypatch.setattr(regime_aware, "RegimeDetector", FakeDetector)

    result = regime_aware._auto_detect_segments(
        {
            "backtesting": {"pairs": ["BTC/USDT"], "timerange": "20250101-20250201"},
            "regime_aware": {
                "method": "sma_adx",
                "period_days": 30,
                "min_period_days": 10,
                "embargo_days": 2,
                "segments_per_regime": 1,
                "holdout_ratio": 0.2,
            },
        }
    )

    assert observed["method"] == "sma_adx"
    assert observed["balanced"] == (["segment"], 1)
    assert result == {
        "optimization": ["segment"],
        "model_selection": [],
        "holdout": [],
    }
