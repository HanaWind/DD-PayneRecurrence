from __future__ import annotations

import pytest

from ddpayne.training.train import _exponential_learning_rate_factor


def test_exponential_schedule_reaches_configured_end_at_final_step():
    start = 1e-3
    end = 3e-4
    steps = 3

    assert _exponential_learning_rate_factor(0, start, end, steps) == pytest.approx(1.0)
    assert start * _exponential_learning_rate_factor(steps, start, end, steps) == pytest.approx(end)


@pytest.mark.parametrize(
    ("start", "end", "steps"),
    [(0.0, 1e-4, 10), (1e-3, 0.0, 10), (1e-3, 1e-4, 0)],
)
def test_exponential_schedule_rejects_invalid_configuration(start, end, steps):
    with pytest.raises(ValueError):
        _exponential_learning_rate_factor(0, start, end, steps)
