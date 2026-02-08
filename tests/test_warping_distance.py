import numpy as np
import pytest
from tsmean.warping_distance import dtw, dtwr, cost, as_time_series

def test_as_time_series():
    # Univariate
    x = np.array([1.0, 2.0, 3.0])
    x_ts = as_time_series(x)
    assert x_ts.shape == (3, 1)
    
    # Multivariate
    x_mv = np.random.randn(10, 3)
    x_ts_mv = as_time_series(x_mv)
    assert x_ts_mv.shape == (10, 3)
    
    # Type error
    with pytest.raises(TypeError):
        as_time_series([1, 2, 3])
        
    # Dimension error
    with pytest.raises(ValueError):
        as_time_series(np.random.randn(2, 2, 2))

def test_dtw_basic():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 2.2, 3.0])
    
    dist, path = dtw(x, y)
    
    assert dist >= 0
    assert isinstance(path, list)
    assert path[0] == (0, 0)
    assert path[-1] == (len(x)-1, len(y)-1)

def test_dtw_identical():
    x = np.random.randn(10)
    dist, path = dtw(x, x)
    assert pytest.approx(dist) == 0.0
    assert len(path) == len(x)
    for i in range(len(x)):
        assert path[i] == (i, i)

def test_dtw_warping_window():
    x = np.random.randn(20)
    y = np.random.randn(20)
    
    dist_no_w, _ = dtw(x, y, w=None)
    dist_w, _ = dtw(x, y, w=1)
    
    # With a small window, the distance should be >= distance without window
    assert dist_w >= dist_no_w

def test_dtwr():
    x = np.random.randn(10)
    y = np.random.randn(10)
    
    dist, path = dtwr(x, y)
    assert dist >= 0
    assert len(path) > 0

def test_cost():
    x = np.array([1.0, 2.0])
    y = np.array([1.0, 3.0])
    path = [(0, 0), (1, 1)]
    # (1-1)^2 + (2-3)^2 = 0 + 1 = 1
    c = cost(as_time_series(x), as_time_series(y), path)
    assert pytest.approx(c) == 1.0
