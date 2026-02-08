import numpy as np
import pytest
from tsmean.optimizer import Adam, AdaDelta, RMSProp, SGD, HSGD

@pytest.mark.parametrize("optimizer_class", [Adam, AdaDelta, RMSProp, SGD, HSGD])
def test_optimizer_step(optimizer_class):
    # Initialize optimizer
    if optimizer_class == HSGD:
        optimizer = optimizer_class(eta=0.1, newton=True)
    elif optimizer_class == AdaDelta:
        optimizer = optimizer_class()
    else:
        optimizer = optimizer_class(eta=0.1)
    
    # Test 1D parameters
    params = np.array([1.0, 2.0, 3.0])
    grads = np.array([0.1, 0.2, 0.3])
    H_inv = np.array([1.0, 1.0, 1.0])
    
    updated = optimizer.step(params.copy(), grads, H_inv=H_inv)
    
    assert updated.shape == params.shape
    assert not np.allclose(updated, params)

@pytest.mark.parametrize("optimizer_class", [Adam, RMSProp, SGD])
def test_optimizer_descent(optimizer_class):
    """Test if the optimizer descends on a simple quadratic target f(x) = x^2."""
    optimizer = optimizer_class(eta=0.1)
    x = np.array([10.0])
    
    for _ in range(5):
        g = 2 * x # gradient of x^2
        x_new = optimizer.step(x, g)
        assert np.abs(x_new) < np.abs(x)
        x = x_new

def test_adam_bias_correction():
    opt = Adam(eta=0.1)
    x = np.array([1.0])
    g = np.array([0.5])
    
    # First step should include bias correction
    x1 = opt.step(x.copy(), g)
    assert not np.allclose(x1, x)

def test_hsgd_newton_vs_isotropic():
    x = np.array([1.0, 1.0])
    g = np.array([1.0, 2.0])
    H_inv = np.array([0.5, 0.1]) # Diagonal inverse Hessian
    
    # Newton step: x - eta * H_inv * g
    opt_newton = HSGD(eta=1.0, newton=True)
    x_n = opt_newton.step(x.copy(), g, H_inv=H_inv)
    # Expected: [1.0 - 1.0*0.5*1.0, 1.0 - 1.0*0.1*2.0] = [0.5, 0.8]
    assert np.allclose(x_n, [0.5, 0.8])
    
    # Isotropic step: x - eta * 2 * min(H_inv) * g
    opt_iso = HSGD(eta=1.0, newton=False)
    x_i = opt_iso.step(x.copy(), g, H_inv=H_inv)
    # min(H_inv) = 0.1. Step: x - 1.0 * 0.2 * g = [1.0 - 0.2*1.0, 1.0 - 0.2*2.0] = [0.8, 0.6]
    assert np.allclose(x_i, [0.8, 0.6])
