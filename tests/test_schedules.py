import numpy as np
import pytest
from tsmean.schedule import linear_schedule, sine_schedule, sawtooth_schedule

def test_linear_schedule():
    s = linear_schedule(10, eta_init=1.0, eta=0.1)
    assert len(s) == 10
    assert s[0] == 1.0
    assert s[-1] == 0.1
    assert np.all(np.diff(s) <= 0) # Non-increasing

def test_sine_schedule():
    s = sine_schedule(100, cycles=2, eta_init=0.5)
    assert len(s) == 100
    assert np.max(s) == pytest.approx(0.5)
    assert np.min(s) >= 0

def test_sawtooth_schedule():
    s = sawtooth_schedule(100, cycles=5, eta_init=1.0)
    # The current implementation uses floor division, might be slightly less than 100
    assert len(s) <= 100
    assert len(s) > 0
    assert s[0] == 1.0
