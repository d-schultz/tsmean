import numpy as np
import pytest
from tsmean.mean_algorithm import SubgradientMeanAlgorithm, dba, ssg
from tsmean.optimizer import SGD

def test_subgradient_computation():
    optimizer = SGD(eta=0.1)
    algo = SubgradientMeanAlgorithm(optimizer=optimizer)
    
    # Simple setup: mean should be between two identical series
    x = np.array([[1.0], [2.0]])
    X_batch = [x, x]
    
    g, f, H_inv = algo.subgradient(X_batch, x, return_hessian=True)
    
    assert pytest.approx(f) == 0.0
    assert np.all(g == 0.0)
    assert H_inv is not None
    assert H_inv.shape == x.shape

def test_dba_convergence():
    # Generate 10 identical sine waves
    t = np.linspace(0, 2*np.pi, 20)
    base = np.sin(t)
    X = [base.copy() for _ in range(10)]
    
    # DBA should converge to the base sine wave immediately or very quickly
    result = dba(X, n_epochs=5, f_tol=1e-6)
    
    assert result.success
    assert result.frechet_var < 1e-5
    assert np.allclose(result.mean.flatten(), base, atol=1e-3)

def test_ssg_execution():
    t = np.linspace(0, 2*np.pi, 20)
    X = [np.sin(t) + np.random.normal(0, 0.1, 20) for _ in range(5)]
    
    # Just check if it runs without error and returns a result
    result = ssg(X, n_epochs=2, batch_size=2)
    
    assert result.mean.shape == (20,)
    assert len(result.F_history) >= 0

def test_mean_algorithm_result_print(capsys):
    res = dba([np.array([1, 2, 3])], n_epochs=1)
    res.print_stats()
    captured = capsys.readouterr()
    assert "terminated at epoch" in captured.out.lower()
    assert "Fréchet variation" in captured.out
