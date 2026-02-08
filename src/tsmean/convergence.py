from dataclasses import dataclass
from typing import List, Any
from abc import ABC, abstractmethod
import numpy as np


@dataclass
class ConvergenceStatistics:
    """Dataclass for storing convergence statistics.

    Attributes
    ----------
    values : List[List[float]]
        List of lists of values. Each inner list values[i]is a history 
        of values for a specific metric.
    names : List[str]
        List of names of the metrics. names[i] corresponds to values[i].
    n_updates : int
        Number of updates when convergence was detected. If convergence was not detected, 
        this value is infinity.
    message : str
        Message indicating the reason for convergence.
    """
    values: List[List[float]]
    names: List[str]
    n_updates: int
    message: str




class AbstractConvergenceChecker(ABC):
    """Abstract base class for convergence checkers.
       For checking convergence based on function values, parameters or gradients.

       Attributes
       ----------
       **kwargs : dict
           Keyword arguments for the convergence checker.
    """
    
    @abstractmethod
    def __init__(self, **kwargs):
        """Initialize the convergence checker.

        Parameters
        ----------
        **kwargs : dict
            Configuration parameters for the specific convergence checker.
        """
        pass

    @abstractmethod
    def has_converged(self, f_new : float | None, x_new : np.ndarray, 
                      g_new : np.ndarray, **kwargs) -> bool:
        """Checks if the optimization has converged.
        
        Parameters
        ----------
        f_new : float | None
            New function value.
        x_new : np.ndarray
            New parameter values.
        g_new : np.ndarray
            New gradient values.
        **kwargs : dict
            Keyword arguments for the convergence checker.
        
        Returns
        -------
        bool
            True if converged, False otherwise.
        """
        pass

    @abstractmethod
    def get_statistics(self) -> ConvergenceStatistics:
        """Returns ConvergenceStatistics object.
        
        Returns
        -------
        ConvergenceStatistics
            Statistics of the convergence checker.
        """
        pass



class FunctionValueConvergenceChecker(AbstractConvergenceChecker):
    """Checks for convergence based on function value history.
    
    Parameters
    ----------
    patience : int, default=10
        Number of consecutive updates with function value within tolerance.
    f_tol : float, default=1e-3
        Tolerance for the function value. If the function value 
        remains within this tolerance for `patience` updates, 
        convergence is assumed.

    Attributes
    ----------
    f_history : list
        History of function values.
    update_counter : int
        Number of updates.
    n_updates_converged : int
        Number of updates when convergence was detected.
    message : str
        Message indicating the reason for convergence.
    """
    def __init__(self, **kwargs):
        # keyword arguments
        self.patience = kwargs.get("patience", 10)
        self.f_tol = kwargs.get("f_tol", 1e-3)

        # internal state
        self.f_history = []
        self.update_counter = 0
        self.n_updates_converged = np.inf
        self.message = ""

    def has_converged(self, f_new : float | None, x_new : np.ndarray, 
                      g_new : np.ndarray, **kwargs)  -> bool:
        """Checks if the optimization has converged.
        
        Parameters
        ----------
        f_new : float | None
            New function value. If None, update is ignored and convergence 
            is not checked.
        x_new : np.ndarray
            New parameter value. Not used.
        g_new : np.ndarray
            New gradient value. Not used.
        **kwargs : dict
            Keyword arguments for the convergence checker.
        
        Returns
        -------
        bool
            True if converged, False otherwise.
        """
        if f_new is None:
            return False
        
        self.update_counter += 1    

        # buffer function values        
        self.f_history.append(f_new)
        f_window = self.f_history[-self.patience:]
        #if len(self.f_history) > self.patience:
        #    self.f_history.pop(0)

        # check for convergence #

        # if not enough updates, return False
        if len(self.f_history) < self.patience: 
            return False

        # if function value has not changed within tolerance, return True
        if np.allclose(f_window, f_window[-1], rtol=self.f_tol, atol=1e-8):
            self.message = f"Converged because function value remained stable " \
                           f"(rel_tol={self.f_tol}) for {self.patience} updates"
            self.n_updates_converged = self.update_counter
            return True
        
        return False

    def get_statistics(self) -> ConvergenceStatistics:
        """Get the convergence statistics.

        Returns
        -------
        ConvergenceStatistics
            Object containing histories, update count, and convergence reason.
        """
        return ConvergenceStatistics(
            values=[self.f_history],
            names=["Function value"],
            n_updates=self.n_updates_converged,
            message=self.message
        )



class FunctionSlopeConvergenceChecker(AbstractConvergenceChecker):
    """Check for convergence based on the slope of function value history.

    Calculates the trend (slope) over a sliding window of function values and 
    signals convergence if the slope is near zero for a while or fluctuates 
    around zero.

    Parameters
    ----------
    min_updates : int
        Minimum number of updates before checking for convergence.
    window_size : int
        Size of the window used to calculate the slope.
    slope_tol : float, < 0
        Tolerance for the slope. Should be a small negative number, 
        because the slope should be negative for minimization. An 
        increasing slope indicates that the function is no longer decreasing.
    n_zero_crossings : int
        Number of zero crossings required for convergence. If the slope 
        crosses zero often, it indicates that the function is no longer 
        decreasing in a stable manner.
    patience : int
        Number of consecutive near-zero slopes (> slope_tol) required for convergence.

    Attributes
    ----------
    zero_crossings_counter : int
        Number of zero crossings.
    near_zero_plus_counter : int
        Number of near-zero slopes.
    update_counter : int
        Number of updates.
    n_updates_converged : int
        Number of updates when convergence was detected.
    f_history : list
        History of function values.
    slope_history : list
        History of slopes.
    message : str
        Message indicating the reason for convergence.  
    """
    def __init__(self, **kwargs):
        # keyword arguments
        self.min_updates = kwargs.get("min_updates", 500)
        self.window_size = kwargs.get("window_size", 10)
        self.slope_tol = kwargs.get("slope_tol", -0.001)
        self.n_zero_crossings = kwargs.get("n_zero_crossings", 2)
        self.patience = kwargs.get("patience", 2)

        # internal state
        self.zero_crossings_counter = 0
        self.near_zero_plus_counter = 0
        self.update_counter = 0
        self.n_updates_converged = np.inf

        self.f_history = []
        self.slope_history = []
        self.message = ""

    def has_converged(self, f_new : float | None, x_new : np.ndarray, g_new : np.ndarray, **kwargs)  -> bool:
        """Check for convergence based on the slope of the function value history.

        Parameters
        ----------
        f_new : float or None
            New function value. If None, the update is ignored.
        x_new : np.ndarray
            New parameter values. Not used by this checker.
        g_new : np.ndarray
            New gradient values. Not used by this checker.
        **kwargs : dict
            Additional arguments.

        Returns
        -------
        bool
            True if convergence criteria are met, False otherwise.
        """
        if f_new is None:
            return False
            
        self.update_counter += 1
        self.f_history.append(f_new)
        if len(self.f_history) > self.window_size:
            self.f_history.pop(0)
        
        slope = _get_slope(self.f_history)
        self.slope_history.append(slope)

        # if not enough updates, indicate no convergence
        if self.update_counter < self.min_updates:
            return False


        # check for convergence:
        # check if new slope is at least small negative
        if slope > self.slope_tol:
            self.near_zero_plus_counter += 1
        else:
            self.near_zero_plus_counter = 0

        # check if slope crossed zero
        if len(self.slope_history) > 1 and self.slope_history[-2] is not None:
            if slope  * self.slope_history[-2] < 0:
                self.zero_crossings_counter += 1
            
        if self.zero_crossings_counter >= self.n_zero_crossings:
            self.message = "Converged because slope fluctuated around zero for " \
                         + str(self.n_zero_crossings) + " updates"
            self.n_updates_converged = self.update_counter
            return True

        if self.near_zero_plus_counter >= self.patience:
            self.message = "Converged because slope remained > " + str(self.slope_tol) \
                         + " for " + str(self.patience) + " updates"
            self.n_updates_converged = self.update_counter
            return True
        
        return False

    def get_statistics(self) -> ConvergenceStatistics:
        """Returns the statistics of the convergence checker.
        
        Returns
        -------
        ConvergenceStatistics
            Statistics of the convergence checker.
        """
        return ConvergenceStatistics(
            values=[self.slope_history],
            names=["Slope of function value"],
            n_updates=self.n_updates_converged,
            message=self.message
        )


def _get_slope(history : List[float], normalize : bool = True) -> float:
    """Calculate the slope of history using linear regression.
    
    Parameters
    ----------
    history : List[float]  
        List of values.
    normalize : bool
        Whether to normalize the slope by the mean of the history.
    
    Returns
    -------
    float
        Slope of history.
    """
    # Extract the relevant window
    y = np.array(history)

    # Create the time axis (e.g. 0, 1, 2, ... , len(y) - 1)
    t = np.arange(len(y))
    
    # if n==1, slope is undefined, return -inf
    n = len(y)
    if n < 2:
        return -np.inf
    
    # Linear regression (Least Squares):
    sum_t = np.sum(t)
    sum_y = np.sum(y)
    sum_ty = np.sum(t * y)
    sum_t2 = np.sum(t**2)
    
    # Formula for the slope m
    slope = (n * sum_ty - sum_t * sum_y) / (n * sum_t2 - sum_t**2) 
    
    # normalize slope by mean of y
    if normalize:
        mu = np.mean(y)
        return slope / (mu+0.0001) # avoid division by zero
    else:
        return slope


