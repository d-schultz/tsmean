from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import ArrayLike

class BaseOptimizer(ABC):
    """Abstract base class for all Optimizers."""
    
    @property
    @abstractmethod
    def HESSIAN_REQUIRED(self) -> bool:
        """Indicate whether the optimizer requires Hessian information."""
        pass
    
    @abstractmethod
    def step(self, x: np.ndarray, g: np.ndarray, H_inv: np.ndarray | None = None) -> np.ndarray:
        """Perform an optimization step.

        Parameters
        ----------
        x : np.ndarray
            Current parameter values.
        g : np.ndarray
            Current gradient values.
        H_inv : np.ndarray, optional
            Inverse Hessian information, if required by the optimizer.

        Returns
        -------
        np.ndarray
            Updated parameter values.
        """
        pass


class Adam(BaseOptimizer): 
    """Adam optimizer.
    
    Parameters
    ----------
    eta : float or ArrayLike, default=0.001
        Step size or schedule of step sizes.
    beta1 : float, default=0.9
        Exponential decay rate for the first moment estimates.
    beta2 : float, default=0.999
        Exponential decay rate for the second moment estimates.
    eps : float, default=1e-8
        Small constant for numerical stability.
    """
    @property
    def HESSIAN_REQUIRED(self) -> bool:
        return False

    def __init__(self, eta: float | ArrayLike = 0.001, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8): 
        if beta1 >= 1 or beta1 < 0 or beta2 >= 1  or beta2 < 0:
            raise ValueError(f"Invalid parameters: beta1={beta1} and beta2={beta2} must be in the interval [0, 1).")
                
        if isinstance(eta, (int, float)):
            self.eta = np.array([eta])
        else:
            self.eta = np.asarray(eta)

        self.beta1 = beta1 
        self.beta2 = beta2 
        self.eps = eps 
        self.t = 0 
        self.m = 0
        self.v = 0
        
    def step(self, x: np.ndarray, g: np.ndarray, H_inv: np.ndarray | None = None) -> np.ndarray: 
        
        if self.t == 0:
            self.m = np.zeros_like(x)
            self.v = np.zeros_like(x)
        
        # Select learning rate (before incrementing t)
        if self.t < len(self.eta):
            eta = self.eta[self.t]
        else:
            eta = self.eta[-1]
        
        # Update moments
        self.m = self.beta1 * self.m + (1 - self.beta1) * g 
        self.v = self.beta2 * self.v + (1 - self.beta2) * g * g 
        
        # Increment iteration counter (Adam uses t=1, 2, 3, ... for bias correction)
        self.t += 1
        
        # Bias-corrected moment estimates
        m_hat = self.m / (1 - self.beta1 ** self.t) 
        v_hat = self.v / (1 - self.beta2 ** self.t) 
        
        # Update parameters
        x = x - eta * m_hat / (v_hat ** 0.5 + self.eps)

        return x



class AdaDelta(BaseOptimizer): 
    """AdaDelta optimizer.
    
    AdaDelta is an adaptive learning rate method that doesn't require manual tuning 
    of a learning rate. It adapts the learning rate based on a moving window of 
    gradient updates.
    
    Parameters
    ----------
    rho : float, default=0.95
        Exponential decay rate for the moving average of squared gradients and updates.
        Must be in the interval [0, 1).
    eps : float, default=1e-6
        Small constant for numerical stability.
    """
    @property
    def HESSIAN_REQUIRED(self) -> bool:
        return False

    def __init__(self, rho=0.95, eps=1e-6): 
        if rho >= 1 or rho < 0:
            raise ValueError(f"Invalid parameter: rho={rho} must be in the interval [0, 1).")
                
        self.rho = rho
        self.eps = eps 
        self.is_init = False
        self.g = 0
        self.dx = 0
        
    
    def step(self, x: np.ndarray, g: np.ndarray, H_inv: np.ndarray | None = None) -> np.ndarray: 
        
        # Init on first run
        if not self.is_init:
            self.g = np.zeros_like(g)
            self.dx = np.zeros_like(x)
            self.is_init = True 

        # accumulate gradient
        self.g = self.rho * self.g + (1 - self.rho) * g ** 2
        
        # compute update
        dx = - np.sqrt((self.dx + self.eps) / (self.g + self.eps)) * g

        # accumulate updates
        self.dx = self.rho * self.dx + (1 - self.rho) * dx ** 2

        # apply update
        x = x + dx

        return x
    


class RMSProp(BaseOptimizer):
    """RMSProp optimizer.
        
    Parameters
    ----------
    eta : float or ArrayLike, default=0.001
        Step size or schedule of step sizes.
    beta : float, default=0.9
        Exponential decay rate for the moving average of squared gradients.
    eps : float, default=1e-8
        Small constant for numerical stability.
    """

    @property
    def HESSIAN_REQUIRED(self) -> bool:
        return False

    def __init__(self, eta : float | ArrayLike=0.001, beta : float = 0.9, eps : float = 1e-8):
        if isinstance(eta, (int, float)):
            self.eta = np.array([eta])
        else:
            self.eta = np.asarray(eta)

        self.beta = beta
        self.eps = eps 
        self.G = None # moving average of squared gradients
        self.t = 0 # iteration counter
    
    def step(self, x: np.ndarray, g: np.ndarray, H_inv: np.ndarray | None = None) -> np.ndarray: 

        if self.G is None:
            self.G = np.zeros_like(x)

        if self.t < len(self.eta):
            eta = self.eta[self.t]
        else:
            eta = self.eta[-1]

        self.G = self.beta * self.G + (1 - self.beta) * g * g 
        x = x - eta / (self.G ** 0.5 + self.eps) * g

        self.t += 1

        return x


class SGD(BaseOptimizer):
    """Stochastic Gradient Descent optimizer.
    
    Parameters
    ----------
    eta : float or ArrayLike, default=0.01
        Step size or schedule of step sizes.
    """    

    @property
    def HESSIAN_REQUIRED(self) -> bool:
        return False

    def __init__(self, eta : float | ArrayLike = 0.01):
        if isinstance(eta, (int, float)):
            self.eta = np.array([eta])
        else:
            self.eta = np.asarray(eta)

        self.t = 0 # iteration counter
        
    def step(self, x: np.ndarray, g: np.ndarray, H_inv: np.ndarray | None = None) -> np.ndarray: 
        """Performs update step for SGD optimizer.
        
        Parameters
        ----------
        x : np.ndarray
            Current parameters.
        g : np.ndarray
            Current gradient.
    
        Returns
        -------
        x : np.ndarray
            Updated parameters.

        """
        if self.t < len(self.eta):
            eta = self.eta[self.t]
        else:
            eta = self.eta[-1]

        x = x - eta*g

        self.t += 1

        return x

class HSGD(BaseOptimizer):
    """Hessian Stochastic Gradient Descent (HSGD) optimizer.

    A second-order optimization method that incorporates Hessian information
    for adaptive scaling of gradient updates. Supports both the full Stochastic
    Newton method and an isotropic variant.

    Parameters
    ----------
    eta : float or ArrayLike, default=1.0
        Step size or schedule of step sizes.
    newton : bool, default=True
        If True, performs Stochastic Newton update.
        If False, performs isotropic inverse Hessian-scaled update.

    Notes
    -----
    The Stochastic Newton method performs updates of the form::

        x ← x - eta * H_inv * g
    
    The isotropic variant uses::

        x ← x - eta * 2 * min(H_inv) * g
    
    where H_inv is the diagonal of the inverse Hessian matrix.
    """
    @property
    def HESSIAN_REQUIRED(self) -> bool:
        return True


    def __init__(self, eta : float | ArrayLike = 1, newton=True):
        if isinstance(eta, (int, float)):
            self.eta = np.array([eta])
        else:
            self.eta = np.asarray(eta)  
        self.t = 0 # iteration counter
        self.newton = newton
        
    def step(self, x: np.ndarray, g: np.ndarray, H_inv: np.ndarray | None = None) -> np.ndarray: 
        """Performs update step for Adaptive SGD optimizer, requires diagonal of inverse hessian
        
        Parameters
        ----------
        x : np.ndarray
            Current parameters.
        g : np.ndarray
            Current gradient.
        H_inv : np.ndarray
            Diagonal of the inverse Hessian matrix.
    
        Returns
        -------
        x : np.ndarray
            Updated parameters.

        """

        assert H_inv is not None, "H_inv must be provided for ASGD optimizer."
        
        if self.t < len(self.eta):
            eta = self.eta[self.t]
        else:
            eta = self.eta[-1]


        if self.newton: # stochastic newton update
            x = x - eta * H_inv * g 
        else: # isotropic inverse Hessian-scaled update
            x = x - eta * 2*np.min(H_inv) * g       # H_inv = N/2  V^-1 # eta < N/max(V)

        self.t += 1

        return x
    
    
    
