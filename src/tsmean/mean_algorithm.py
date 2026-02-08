from tsmean.warping_distance import dtw, dtwr
from tsmean.optimizer import BaseOptimizer, Adam, SGD, RMSProp, HSGD, AdaDelta
from tsmean.schedule import sawtooth_schedule, sine_schedule, linear_schedule
from tsmean.convergence import AbstractConvergenceChecker, ConvergenceStatistics, FunctionSlopeConvergenceChecker, FunctionValueConvergenceChecker

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Union, Sequence, Callable
from numpy.typing import ArrayLike
from dataclasses import dataclass

# Define type for collection of time series
ArrayCollection = Union[np.ndarray, Sequence[np.ndarray]]



@dataclass
class MeanAlgorithmResult:
    """Result of the mean algorithm.
    
    Attributes
    ----------
    mean : np.ndarray
        The estimated mean time series.
    success : bool
        Whether the algorithm converged before reaching the maximum number of epochs.
    message : str
        A message describing the termination reason.
    termination_epoch : int
        The epoch at which the algorithm terminated.
    frechet_var : float
        The value F(x) of the Fréchet function at the estimated mean x.
    F_history : list[float]
        History of the Fréchet function values.
    example_counter : int
        The number of examples processed.
    algo_name : str
        Name of the algorithm.
    convergence_statistics : ConvergenceStatistics
        Statistics collected by the convergence checker.
    """

    mean : np.ndarray
    success : bool
    message : str
    termination_epoch : int
    frechet_var : float
    F_history : list[float]
    example_counter : int
    Feval_interval : int
    algo_name : str
    convergence_statistics : ConvergenceStatistics | None = None
    

    def plot(self, figsize : Tuple[int,int] = (6.4, 4.8)):
        if not (self.F_history is None or len(self.F_history) == 0):
            if self.convergence_statistics is not None:
                n_stats = len(self.convergence_statistics.values)
            else:
                n_stats = 0
            
            # plot Frechet variation
            x_steps = np.arange(len(self.F_history)) * self.Feval_interval
            plt.figure(figsize=figsize)
            plt.subplot(n_stats+1,1,1)
            plt.title(f"{self.algo_name} - Fréchet variation")
            plt.plot(x_steps, self.F_history)
            plt.xlabel(f"Examples processed")
            
            # plot stats
            for i in range(n_stats):
                plt.subplot(n_stats+1,1,2+i)
                plt.title(self.convergence_statistics.names[i])
                plt.plot(self.convergence_statistics.values[i])
                plt.xlabel(f"Evaluations")
            
            plt.tight_layout()
            plt.show()
    
    def print_stats(self):
        print(f"{self.algo_name} terminated at epoch {self.termination_epoch}")
        print(f"- Message: {self.message}")
        print(f"- Success: {self.success}")
        print(f"- Fréchet variation: {self.frechet_var}")
        print(f"- Examples processed: {self.example_counter}")




class SubgradientMeanAlgorithm(object):
    """(Stochastic) Subgradient Mean Algorithm for time series averaging.

    Class for finding a Fréchet mean of time series under dynamic time 
    warping (DTW) using subgradient methods.
    
    In the standard case, the subgradient methods are applied to optimize:
                
        min F(x) = 1/N * sum_{y in X} dtw(x, y)^2

    where dtw is the squared DTW distance with Euclidean local cost.
    
    Parameters
    ----------
    optimizer : BaseOptimizer
        Optimizer for updating the current mean estimate. Must implement a `step()` method.
    warping_distance : callable, default=dtw
        A function returning a distance and a warping path between two time series.
    """

    def __init__(self, optimizer: BaseOptimizer, warping_distance: Callable = dtw,
                 algo_name : str = "Subgradient Mean Algorithm"):
        self.warping_distance = warping_distance
        self.optimizer = optimizer
        self.algo_name = algo_name


    
    def subgradient(self, X_batch : ArrayCollection, x : np.ndarray, return_hessian : bool = False) \
                    -> Tuple[np.ndarray, float, np.ndarray | None]:
        """Compute the subgradient and optionally the Hessian.

        Computes a subgradient of the Fréchet function for a batch of time series, 
        and optionally the diagonal of the Hessian matrix. 
        Assumes squared Euclidean DTW distances.

        Parameters
        ----------
        X_batch : ArrayCollection
            The current batch of time series.
        x : np.ndarray
            Current estimate of the mean where the gradient is evaluated.
        return_hessian : bool, default=False
            Whether to compute and return the diagonal of the inverse Hessian.

        Returns
        -------
        g : np.ndarray
            The computed subgradient.
        f : float
            Fréchet variation of the batch at `x`.
        H_inv : np.ndarray or None
            Diagonal of the inverse Hessian (as a vector) if `return_hessian` is True, 
            otherwise None.
        """

        N = len(X_batch)
        g = np.zeros_like(x, dtype=np.float64)
        f = 0.0
        V = np.zeros_like(x,dtype=int) # vector representing diagonal of valence matrix

        for y in X_batch:
            dist,path = self.warping_distance(x,y)
            f += dist

            # compute subgradient contribution of this example
            for (i,j) in path:
                g[i] += x[i] - y[j]
            
            # accumulate diagonal of valence matrix 
            if return_hessian:
                for (i,j) in path:
                    V[i] += 1
        g *= 2.0/N
        f /= N
        H_inv = (2.0/N * V + 1e-10)**(-1) if return_hessian else None
        
        return g, f, H_inv


    def compute_mean(self, X : ArrayCollection, 
                     x_init : np.ndarray | None = None, 
                     batch_size : int = 1, 
                     n_epochs : int = 50, 
                     convergence_checker : AbstractConvergenceChecker | None = None,
                     computeF : bool = False, 
                     Feval_interval : int = 1, 
                     return_best_solution : bool = False) -> MeanAlgorithmResult:        
        """Compute approximate Fréchet mean using stochastic subgradient methods.

        Parameters  
        ----------
        X : ArrayCollection
            Collection of time series to average.
        x_init : np.ndarray, optional
            Initial estimate for the mean. If None, a random element from `X` is used.
        batch_size : int, default=1
            Number of samples per update (minibatch size).
        n_epochs : int, default=50
            Maximum number of passes through the entire dataset.
        convergence_checker : AbstractConvergenceChecker, optional
            Object to check for early convergence. If None, runs for exactly `n_epochs`.
        computeF : bool, default=False
            Whether to periodically evaluate the full Fréchet variation.
        Feval_interval : int, default=1
            Frequency of Fréchet variation evaluation (in terms of number of samples).
        return_best_solution : bool, default=False
            If True, returns the solution with the lowest observed Fréchet variation.

        Returns
        -------
        MeanAlgorithmResult
            Object containing the estimated mean, history, and status information.
        """
        N = len(X)
        if x_init is None:
            x_init = X[np.random.randint(N)]
        
        n_epochs = int(n_epochs)
        batch_size = int(batch_size)

        if not computeF and return_best_solution:
            raise ValueError('Cannot track best solution if computeF==False. ' 
                             'Either set return_best_solution=False, or computeF=True.')
        

        assert x_init is not None, "x_init must not be None here."
        
        x = x_init
        x_best = x
        
        # Frechet variation history
        F_history = [] 
        
        # best Frechet variation seen so far
        F_best = np.inf 
        
        # number of processed examples
        example_counter = 0 
        
        # init success and message
        success = False
        message = "Optimization terminated after reaching the maximum number of epochs."

        stop = False
        epoch = 0
        
        for epoch in range(n_epochs):
            if stop:
                break

            # randomly reshuffle X at the start of each epoch
            perm = np.random.permutation(N)

            # process minibatches
            for i in range(0,N,batch_size):

                # select next minibatch
                idx_batch = perm[i:i+batch_size]
                X_batch = [ X[k] for k in idx_batch ]

                # compute subgradient g, Fréchet variation f_batch w.r.t. minibatch X_batch,
                # and inverse Hessian if demanded. H_inv is None if not demanded.
                g, f_batch, H_inv = self.subgradient(X_batch,x, return_hessian=self.optimizer.HESSIAN_REQUIRED)
                
                n_Feval = 0
                # compute F if demanded
                if computeF:
                    # compute number of copies of F(x)=frechet(x, X) to be appended to F in order to serve the Feval_interval
                    n_Feval = ((example_counter+batch_size-1)//Feval_interval) - ((example_counter-1)//Feval_interval) 
                    if n_Feval > 0:
                        f = frechet(x, X, f_batch=f_batch, idx_batch=idx_batch, warping_distance=self.warping_distance)

                        F_history.extend([f]*n_Feval)
                        
                        # update best solution
                        if f < F_best:
                            x_best = x
                            F_best = f

                

                # update x
                if self.optimizer.HESSIAN_REQUIRED:
                    x = self.optimizer.step(x,g,H_inv)
                else:
                    x = self.optimizer.step(x,g)

                example_counter += len(X_batch)


                # Check convergence
                if convergence_checker is not None:
                    # if n_Feval==0, then F_history[-1] is not updated, 
                    # so we pass None as new_f, the convergence checker will handle this
                    if n_Feval > 0:
                        new_f = F_history[-1]
                    else:
                        new_f = None
                    if convergence_checker.has_converged(f_new=new_f, x_new=x, g_new=g):
                        stop = True
                        success = True
                        stats = convergence_checker.get_statistics()
                        message = stats.message
        # end of epochs loop

        # compute final Frechet variation
        if computeF:
            # append Frechet variation F(x) of final soultion x
            f = frechet(x, X, warping_distance=self.warping_distance)
            F_history.append(f)
            if return_best_solution:
                x = x_best
                Fx = F_best
            else:
                Fx = F_history[-1]
        else:
            Fx = None
        
        stats = convergence_checker.get_statistics() if convergence_checker else None

        result = MeanAlgorithmResult(mean=x, success=success, message=message, 
            termination_epoch=epoch, frechet_var=Fx, 
            F_history=F_history, Feval_interval=Feval_interval, 
            example_counter=example_counter, algo_name=self.algo_name,
            convergence_statistics=stats)
        
        return result


def frechet(x : np.ndarray, X : ArrayCollection, f_batch : float | None = None, 
            idx_batch : list[int] | None = None, warping_distance : Callable = dtw) -> float:
    """Compute the Fréchet variation of a collection of time series at a point x.

    Parameters
    ----------
    x : np.ndarray
        The point at which to evaluate the Fréchet variation.
    X : ArrayCollection
        The collection of time series.
    f_batch : float, optional
        Pre-computed Fréchet variation of a minibatch.
    idx_batch : list of int, optional
        Indices of the time series in the minibatch.
    warping_distance : callable, default=dtw
        Warping distance function.

    Returns
    -------
    float
        The computed Fréchet variation.
    """
        
    N = len(X)
    N_batch = len(idx_batch) if idx_batch is not None else 0
    idx_batch_set = set(idx_batch) if idx_batch is not None else set()
    
    f = f_batch * N_batch if f_batch is not None else 0.0

    for i in range(N):
        if idx_batch is None or i not in idx_batch_set:
            dist,_ = warping_distance(x, X[i])
            f = f + dist
    f /= N
    return f




#################### Default Value Generators ####################


def default_init(X):
    """Get the default initial mean estimate.

    The default initial mean estimate is a randomly selected time series from the dataset.

    Parameters
    ----------
    X : ArrayCollection
        The dataset.

    Returns
    -------
    np.ndarray
        The default initial mean estimate.
    """
    return X[np.random.randint(len(X))]

def default_n_epochs(X: ArrayCollection) -> int:
    """Generate a reasonable default for the number of epochs.

    Parameters
    ----------
    X : ArrayCollection
        The dataset.

    Returns
    -------
    int
        Default number of epochs.
    """
    N = len(X)
    return int(max(50, np.ceil(5000/N) ))

def default_step_size_schedule(eta0 : float, eta : float, n_eta : int) -> list[float]:
    """Get the default step size schedule"""
    if isinstance(eta, (np.ndarray, list)):
        etas = eta
    elif isinstance(eta, (float, int)):
        etas = linear_schedule(size=n_eta, eta_init=eta0, eta=eta)
    else:
        raise ValueError("eta must be a float, int, list or np.ndarray")
    return etas 





#################### Convenience Functions ####################

########## Subgradient Mean Algorithm Template ##########

def subgradient_mean_algorithm_template(
                X: ArrayCollection,
                x0 : np.ndarray,
                n_epochs : int,
                batch_size : int,
                computeF : bool,
                Feval_interval : int,
                return_best_solution : bool,
                optimizer : BaseOptimizer,
                convergence_checker : AbstractConvergenceChecker | None = None,
                warping_distance : Callable = dtw,
                algo_name : str = "Subgradient Mean Algorithm") -> MeanAlgorithmResult:
    """Run stochastic subgradient method (SSG) with generic optimizer and convergence checker.

    Parameters
    ----------
    X : ArrayCollection
        Collection of time series to average.
    x0 : np.ndarray
        Initial mean estimate.
    n_epochs : int
        Maximum number of epochs.
    batch_size : int
        Number of samples per update.
    computeF : bool
        Whether to evaluate the Fréchet variation.
    Feval_interval : int
        Interval (in samples) for evaluating the Fréchet variation.
    return_best_solution : bool
        Whether to return the best solution encountered.
    optimizer : BaseOptimizer
        Optimizer instance.
    convergence_checker : AbstractConvergenceChecker, optional
        Object for early termination check.
    warping_distance : callable, default=dtw
        Warping distance function.
    algo_name : str, optional
        Name of the algorithm.

    Returns
    -------
    MeanAlgorithmResult
        Result object containing final mean and statistics.
    """
    if n_epochs is None:
        n_epochs = default_n_epochs(X)
    
    ssg_mean_algo = SubgradientMeanAlgorithm(optimizer=optimizer,
                                             warping_distance=warping_distance,
                                             algo_name=algo_name)
    
    result = ssg_mean_algo.compute_mean(
                X=X,
                x_init=x0,
                batch_size=batch_size,
                n_epochs=n_epochs, 
                convergence_checker=convergence_checker, 
                computeF=computeF,  
                Feval_interval=Feval_interval, 
                return_best_solution=return_best_solution)
    return result

    
########## DBA Variants##########


def dba(X : ArrayCollection, 
        x0 : np.ndarray | None = None, 
        n_epochs : int | None = None, 
        computeF : bool = True, 
        Feval_interval : int | None = None, 
        f_tol : float = 0.01, 
        patience : int = 3,
        warping_distance : Callable[[np.ndarray, np.ndarray], float] = dtw,
        ) -> MeanAlgorithmResult:
    """Runs DTW Barycenter Averaging (DBA) with termination criterion based on 
       function values

    The algorithm is a Majorize-Minimize algorithm which is implemented as a 2nd order 
    subgradient method with a Newton step in the direction of the inverse Hessian.
    It is precisely equivalent to DBA.
    
    The algorithm terminates if the function value does not improve enough for
    `patience` epochs, measured by the relative improvement `f_tol`.
    
    If `f_tol = 0`, the algorithm terminates if the function value does not improve
    for `patience` epochs. If additionally the optimal path backtracking of dtw 
    is deterministically implemented, the solution can not be improved further. 
    In this case, the algorithm can stop immediately so `patience` can safely 
    be set to 1.
    
    Parameters
    ----------
    X : ArrayCollection (List of np.ndarrays or np.ndarray) such that X[i] is a time series
        Collection of time series to be averaged
    x0 : np.ndarray | None
        Initial mean estimate (initial iterate). If None, a random time series
        of X is used.
    n_epochs : int | None
        Maximum number of epochs. If None, n_epochs is set to 5000/N, but at least 50.
    computeF : bool
        Whether to compute the function values, requires no substantial 
        additional cost for full-batch methods such as DBA
    Feval_interval : int | None, optional, default None
        Interval at which to evaluate the Fréchet function. If None, 
        Feval_interval is set to N.
        Only scales the number of function evaluations, no additional cost here.
        A function value is provided for each Feval_interval processed examples.
        If Feval_interval==N, the function value is provided for each epoch.
        If Feval_interval==1, the function value is provided for each example.
    f_tol : float, optional, default 0.01
        Relative tolerance for the function value convergence checker. See 
        tsmean.convergence_checker.FunctionValueConvergenceChecker for more information.
    patience : int, optional, default 3
        Number of epochs to wait for improvement before stopping. See 
        tsmean.convergence_checker.FunctionValueConvergenceChecker for more information.
    warping_distance : Callable[[np.ndarray, np.ndarray], float]
        Warping distance to be used. Defaults to dtw.
        Other squared Euclidean dtw variants can be used as well, e.g. dtwr with 
        random path backtracking or dtw with warping window (mask), or alternative 
        step conditions.
    Returns
    -------
    result : MeanAlgorithmResult
        Result of the algorithm
    """

    N = len(X)

    if x0 is None:
        x0 = default_init(X)

    if Feval_interval is None:
        Feval_interval = N

    
    # choose optimizer that performs Newton step in the direction of the inverse Hessian
    optimizer = HSGD(eta=1.0, newton=True) 

    # choose convergence checker that checks for function value convergence
    convergence_checker = FunctionValueConvergenceChecker(f_tol=f_tol, 
                                                          patience=patience)


    # return_best_solution=False because DBA is a Majorize-Minimize algorithm, 
    # so the solution is always improving, hence the best solution is the 
    # last solution -> no need to store the best solution as this may block 
    # the possibility to set computeF=False.
    return subgradient_mean_algorithm_template(
        X=X,
        x0=x0,
        batch_size=N,
        n_epochs=n_epochs,
        computeF=computeF,
        Feval_interval=Feval_interval,
        convergence_checker=convergence_checker,
        optimizer=optimizer,
        return_best_solution=False,
        algo_name="DBA"
    )







def dba_random_path(X : ArrayCollection, 
                    x0 : np.ndarray | None = None, 
                    n_epochs : int | None = None, 
                    computeF : bool = True, 
                    Feval_interval : int | None = None, 
                    f_tol : float = 0.01, 
                    window_size : int = 3) -> MeanAlgorithmResult:
    """Runs DTW Barycenter Averaging (DBA) with random path selection.
    
    DBA with random path can potentially improve the solution quality of DBA 
    in situations where DBA stucks in a critical point which is not a local 
    minimum, i.e. there is another active component function which can only 
    be chosen by finding a different configuration of warping paths.

    Parameters
    ----------
    X : ArrayCollection
        The time series collection.
    x0 : np.ndarray | None, optional
        The initial solution. See :func:`dba` for details.
    n_epochs : int | None, optional
        The number of epochs. See :func:`dba` for details.
    computeF : bool, optional
        Whether to compute the function values. See :func:`dba` for details.
    Feval_interval : int | None, optional
        The interval at which to evaluate the function. See :func:`dba` for details.
    f_tol : float, optional
        The tolerance for the function value. See :func:`dba` for details.
    window_size : int, optional
        The window size for the function value. See :func:`dba` for details.
    
    Returns
    -------
    MeanAlgorithmResult
        The result of the algorithm.
    """
    
    return dba(X=X,x0=x0,n_epochs=n_epochs,warping_distance=dtwr,
                   computeF=computeF,Feval_interval=Feval_interval,f_tol=f_tol,
                   window_size=window_size, algo_name="DBA-r")





########## SSG Variants without Convergence Checker ##########

def sg_autoschedule(X : ArrayCollection,
        x0 : np.ndarray | None = None,
        n_epochs : int | None = None,
        eta : float = 0.5,
        computeF : bool = False,
        Feval_interval : int = 1,
        return_best_solution : bool = False,
        warping_distance : Callable[[np.ndarray, np.ndarray], float] = dtw,
        algo_name : str = "SG-auto") -> MeanAlgorithmResult:
    """Runs full-batch subgradient method (SG) without convergence checker.
    
    The algorithm stops after n_epochs.
    
    Parameters
    ----------
    X : ArrayCollection
        The time series collection.
    x0 : np.ndarray | None, optional
        The 
    n_epochs : int | None, optional
        The number of epochs. See :func:`ssg` for details.
    eta : float, default=0.5
        Step size scale. Use 0.5 for maximum GD progress on component 
        functions, or 1.0 for the maximum step size that ensures descent.
    computeF : bool, optional
        Whether to compute the function values. See :func:`ssg` for details.
    Feval_interval : int | None, optional
        The interval at which to evaluate the function. See :func:`ssg` for details.
    return_best_solution : bool, optional
        Whether to return the best solution found during the optimization. 
        See :func:`ssg` for details.
    warping_distance : Callable[[np.ndarray, np.ndarray], float], optional
        The warping distance to be used. Defaults to dtw.
    algo_name : str, optional
        Name of the algorithm.
    
    Returns
    -------
    MeanAlgorithmResult
        The result of the algorithm.
    """
    
    optimizer = HSGD(eta=eta, newton=False)

    return subgradient_mean_algorithm_template(
        X=X, 
        x0=x0, 
        n_epochs=n_epochs, 
        batch_size=len(X),
        optimizer=optimizer,
        computeF=computeF, 
        Feval_interval=Feval_interval, 
        return_best_solution=return_best_solution, 
        convergence_checker=None,
        warping_distance=warping_distance)



def ssg(X : ArrayCollection,
        x0 : np.ndarray | None = None,
        n_epochs : int | None = None,
        batch_size : int = 1,
        eta0 : float = 0.1,
        eta : float | ArrayLike = 0.01,
        n_eta : int = 250,
        computeF : bool = False,
        Feval_interval : int = 1,
        return_best_solution : bool = False,
        warping_distance : Callable[[np.ndarray, np.ndarray], float] = dtw,
        algo_name : str = "SSG") -> MeanAlgorithmResult:
    """Runs stochastic subgradient method (SSG) without convergence checker.
    
    The algorithm stops after n_epochs.
    
    Parameters
    ----------
    X : ArrayCollection
        Collection of time series to be averaged
    x0 : np.ndarray | None
        Initial mean estimate (initial iterate). If None, a random time series
        of X is used.
    n_epochs : int | None
        Maximum number of epochs. If None, n_epochs is set to 5000/N, but at least 50.
    batch_size : int
        Batch size
    eta0 : float
        Initial step size
    eta : float | ArrayLike
        Final step size or list of step sizes (i.e. step size schedule)
        If eta is a list (schedule), eta0 and n_eta are ignored.
        If eta is a float, a linear schedule of length n_eta is generated using 
        the starting step size eta0 and the final step size eta.
        A constant step size can be realized by setting eta0=eta and n_eta=1.
    n_eta : int
        Number of step sizes used to generate a linear schedule if eta is a float
    computeF : bool
        Whether to compute the function values. If computeF==True, a decreasing
        batch_size leads to an increase in computational costs.
    Feval_interval : int
        Interval at which to evaluate the Fréchet function value. The higher 
        Feval_interval, the lower the computational cost, but the less often 
        the function value is evaluated.
    return_best_solution : bool
        Whether to return the best solution found during the optimization. 
        If True, computeF must be True, otherwise an error is raised.
    warping_distance : Callable[[np.ndarray, np.ndarray], float]
        Warping distance to be used. Defaults to dtw.
    algo_name : str, optional
        Name of the algorithm.
    
    Returns
    -------
    result : MeanAlgorithmResult
        Result of the algorithm
    """

    etas = default_step_size_schedule(eta0, eta, n_eta)

    optimizer = SGD(eta=etas)

    return subgradient_mean_algorithm_template(
        X=X, 
        x0=x0, 
        n_epochs=n_epochs, 
        batch_size=batch_size, 
        optimizer=optimizer,
        computeF=computeF, 
        Feval_interval=Feval_interval, 
        return_best_solution=return_best_solution, 
        convergence_checker=None,
        warping_distance=warping_distance,
        algo_name=algo_name)



def adam(X : ArrayCollection,
        x0 : np.ndarray | None = None,
        n_epochs : int | None = None,
        batch_size : int = 1,
        eta0 : float = 0.1,
        eta : float | ArrayLike = 0.01,
        n_eta : int = 250,
        beta1 : float = 0.9,
        beta2 : float = 0.999,
        eps : float = 1e-8,
        computeF : bool = False,
        Feval_interval : int = 1,
        return_best_solution : bool = False,
        warping_distance : Callable[[np.ndarray, np.ndarray], float] = dtw,
        algo_name : str = "Adam") -> MeanAlgorithmResult:
    """Runs stochastic subgradient method (SSG) with Adam optimizer 
    and without convergence checker.
    
    The algorithm stops after n_epochs.
    
    Parameters
    ----------
    X : ArrayCollection
        Collection of time series to be averaged.
    x0 : np.ndarray | None
        Initial mean estimate (initial iterate). If None, a random time series
        of X is used.
    n_epochs : int | None
        Maximum number of epochs. See :func:`ssg`.
    batch_size : int
        Batch size. See :func:`ssg`.
    eta0 : float
        Initial step size. See :func:`ssg`.
    eta : float | ArrayLike
        Final step size or list of step sizes (i.e. step size schedule). 
        See :func:`ssg`.
    n_eta : int
        Number of step sizes used to generate a linear schedule if eta is a float.
        See :func:`ssg`.
    beta1 : float
        The exponential decay rate for the first moment estimates.
    beta2 : float
        The exponential decay rate for the second moment estimates.
    eps : float
        A small constant for numerical stability.
    computeF : bool
        Whether to compute the function values. See :func:`ssg`.
    Feval_interval : int
        The interval at which to evaluate the function. See :func:`ssg`.
    return_best_solution : bool
        Whether to return the best solution found during the optimization. 
        See :func:`ssg`.
    warping_distance : Callable[[np.ndarray, np.ndarray], float]
        Warping distance to be used. Defaults to dtw.
    algo_name : str, optional
        Name of the algorithm.
    
    Returns
    -------
    result : MeanAlgorithmResult
        Result of the algorithm
    """
    if isinstance(eta, (np.ndarray, list)):
        etas = eta
    elif isinstance(eta, (float, int)):
        etas = linear_schedule(size=n_eta, eta_init=eta0, eta=eta)
    else:
        raise ValueError("eta must be a float, int, list or np.ndarray")
    
    optimizer = Adam(eta=etas, beta1=beta1, beta2=beta2, eps=eps)

    return subgradient_mean_algorithm_template(
        X=X, 
        x0=x0, 
        n_epochs=n_epochs, 
        batch_size=batch_size, 
        optimizer=optimizer,
        computeF=computeF, 
        Feval_interval=Feval_interval, 
        return_best_solution=return_best_solution, 
        convergence_checker=None,
        warping_distance=warping_distance,
        algo_name=algo_name)




def adadelta(X : ArrayCollection,
        x0 : np.ndarray | None = None,
        n_epochs : int | None = None,
        batch_size : int = 1,
        rho : float = 0.9,
        eps : float = 1e-8,
        computeF : bool = False,
        Feval_interval : int = 1,
        return_best_solution : bool = False,
        warping_distance : Callable[[np.ndarray, np.ndarray], float] = dtw,
        algo_name : str = "AdaDelta") -> MeanAlgorithmResult:
    """Runs stochastic subgradient method (SSG) with AdaDelta optimizer 
    and without convergence checker.
    
    The algorithm stops after n_epochs.
    
    Parameters
    ----------
    X : ArrayCollection
        Collection of time series to be averaged.
    x0 : np.ndarray | None
        Initial mean estimate (initial iterate). If None, a random time series
        of X is used.
    n_epochs : int | None
        Maximum number of epochs. See :func:`ssg`.
    batch_size : int
        Batch size. See :func:`ssg`.
    rho : float
        The exponential decay rate for the moving average of squared gradients.
    eps : float
        A small constant for numerical stability.
    computeF : bool
        Whether to compute the function values. See :func:`ssg`.
    Feval_interval : int
        The interval at which to evaluate the function. See :func:`ssg`.
    return_best_solution : bool
        Whether to return the best solution found during the optimization. 
        See :func:`ssg`.
    warping_distance : Callable[[np.ndarray, np.ndarray], float]
        Warping distance to be used. Defaults to dtw.
    algo_name : str, optional
        Name of the algorithm.
    
    Returns
    -------
    result : MeanAlgorithmResult
        Result of the algorithm
    """
    
    optimizer = AdaDelta(rho=rho, eps=eps)

    return subgradient_mean_algorithm_template(
        X=X, 
        x0=x0, 
        n_epochs=n_epochs, 
        batch_size=batch_size, 
        optimizer=optimizer,
        computeF=computeF, 
        Feval_interval=Feval_interval, 
        return_best_solution=return_best_solution, 
        convergence_checker=None,
        warping_distance=warping_distance,
        algo_name=algo_name)




def rmsprop(X : ArrayCollection,
        x0 : np.ndarray | None = None,
        n_epochs : int | None = None,
        batch_size : int = 1,
        eta0 : float = 0.1,
        eta : float | ArrayLike = 0.01,
        n_eta : int = 250,
        beta : float = 0.9,
        eps : float = 1e-8,
        computeF : bool = False,
        Feval_interval : int = 1,
        return_best_solution : bool = False,
        warping_distance : Callable[[np.ndarray, np.ndarray], float] = dtw,
        algo_name : str = "RMSprop") -> MeanAlgorithmResult:
    """Runs stochastic subgradient method (SSG) with RMSprop optimizer 
    and without convergence checker.
    
    The algorithm stops after n_epochs.
    
    Parameters
    ----------
    X : ArrayCollection
        Collection of time series to be averaged.
    x0 : np.ndarray | None
        Initial mean estimate (initial iterate). If None, a random time series
        of X is used.
    n_epochs : int | None
        Maximum number of epochs. See :func:`ssg`.
    batch_size : int
        Batch size. See :func:`ssg`.
    eta0 : float
        Initial step size. See :func:`ssg`.
    eta : float | ArrayLike
        Final step size or list of step sizes (i.e. step size schedule). 
        See :func:`ssg`.
    n_eta : int
        Number of step sizes used to generate a linear schedule if eta is a float.
        See :func:`ssg`.
    beta : float
        The exponential decay rate for the moving average of squared gradients.
    eps : float
        A small constant for numerical stability.
    computeF : bool
        Whether to compute the function values. See :func:`ssg`.
    Feval_interval : int
        The interval at which to evaluate the function. See :func:`ssg`.
    return_best_solution : bool
        Whether to return the best solution found during the optimization. 
        See :func:`ssg`.
    warping_distance : Callable[[np.ndarray, np.ndarray], float]
        Warping distance to be used. Defaults to dtw.
    algo_name : str, optional
        Name of the algorithm.
    
    Returns
    -------
    result : MeanAlgorithmResult
        Result of the algorithm
    """
    if isinstance(eta, (np.ndarray, list)):
        etas = eta
    elif isinstance(eta, (float, int)):
        etas = linear_schedule(size=n_eta, eta_init=eta0, eta=eta)
    else:
        raise ValueError("eta must be a float, int, list or np.ndarray")
    
    optimizer = RMSProp(eta=etas, beta=beta, eps=eps)

    return subgradient_mean_algorithm_template(
        X=X, 
        x0=x0, 
        n_epochs=n_epochs, 
        batch_size=batch_size, 
        optimizer=optimizer,
        computeF=computeF, 
        Feval_interval=Feval_interval, 
        return_best_solution=return_best_solution, 
        convergence_checker=None,
        warping_distance=warping_distance,
        algo_name=algo_name)







########## SSG Variants with Slope Convergence Checker ##########




def ssg_slope(X : ArrayCollection,
        x0 : np.ndarray | None = None,
        n_epochs : int | None = None,
        batch_size : int = 1,
        eta0 : float = 0.1,
        eta : float | ArrayLike = 0.01,
        n_eta : int = 250,
        Feval_interval : int = 1,
        return_best_solution : bool = False,
        slope_window_size : int = 30,
        slope_tol : float = -0.001,
        n_zero_crossings : int = 10,
        patience : int = 5,
        min_updates : int = 500,
        warping_distance : Callable[[np.ndarray, np.ndarray], float] = dtw,
        algo_name : str = "SSG-slope") -> MeanAlgorithmResult:
    """Run Stochastic Subgradient (SSG) method with a slope-based convergence checker.

    Parameters
    ----------
    X, x0, n_epochs, batch_size, eta0, eta, n_eta, computeF, Feval_interval, return_best_solution
        See :func:`ssg`.
    slope_window_size : int, default=30
        Window size for slope calculation.
    slope_tol : float, default=-0.001
        Tolerance for the slope check.
    n_zero_crossings : int, default=10
        Number of zero crossings required for convergence.
    patience : int, default=5
        Number of updates with near-zero slope required.
    min_updates : int, default=500
        Minimum number of updates before checking convergence.
    warping_distance : Callable[[np.ndarray, np.ndarray], float]
        Warping distance to be used. Defaults to dtw.
    algo_name : str, optional
        Name of the algorithm.
    
    Returns 
    -------
    MeanAlgorithmResult
        Result object containing final mean and statistics.
    """
    

    etas = default_step_size_schedule(eta0, eta, n_eta)
    
    optimizer = SGD(eta=etas)

    convergence_checker = FunctionSlopeConvergenceChecker(
                            min_updates=min_updates,
                            window_size=slope_window_size, 
                            slope_tol=slope_tol, 
                            n_zero_crossings=n_zero_crossings, 
                            patience=patience)

    return subgradient_mean_algorithm_template(
        X=X, 
        x0=x0, 
        n_epochs=n_epochs, 
        batch_size=batch_size, 
        optimizer=optimizer,
        computeF=True, 
        Feval_interval=Feval_interval, 
        return_best_solution=return_best_solution, 
        convergence_checker=convergence_checker,
        warping_distance=warping_distance,
        algo_name=algo_name)



def adam_slope(X : ArrayCollection,
        x0 : np.ndarray | None = None,
        n_epochs : int | None = None,
        batch_size : int = 1,
        eta0 : float = 0.1,
        eta : float | ArrayLike = 0.01,
        n_eta : int = 250,
        beta1 : float = 0.9,
        beta2 : float = 0.999,
        eps : float = 1e-8,
        Feval_interval : int = 1,
        return_best_solution : bool = True,
        slope_window_size : int = 30,
        slope_tol : float = -0.001,
        n_zero_crossings : int = 10,
        patience : int = 5,
        min_updates : int = 10,
        warping_distance : Callable[[np.ndarray, np.ndarray], float] = dtw,
        algo_name : str = "Adam-slope") -> MeanAlgorithmResult:
    

    etas = default_step_size_schedule(eta0, eta, n_eta)
    
    optimizer = Adam(eta=etas, beta1=beta1, beta2=beta2, eps=eps)

    convergence_checker = FunctionSlopeConvergenceChecker(
                            min_updates=min_updates,
                            window_size=slope_window_size, 
                            slope_tol=slope_tol, 
                            n_zero_crossings=n_zero_crossings, 
                            patience=patience)

    return subgradient_mean_algorithm_template(
        X=X, 
        x0=x0, 
        n_epochs=n_epochs, 
        batch_size=batch_size, 
        optimizer=optimizer,
        computeF=True, 
        Feval_interval=Feval_interval, 
        return_best_solution=return_best_solution, 
        convergence_checker=convergence_checker,
        warping_distance=warping_distance,
        algo_name=algo_name)


