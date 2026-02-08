import numpy as np
from numba import njit, float64, int64
from typing import Tuple, List, Union, Dict, Any
import time

from typing import Sequence, Union
from numpy.typing import ArrayLike



def dtw(x: np.ndarray, y: np.ndarray, w: int | None = None, random_path: bool = False) -> Tuple[float, List[Tuple[int, int]]]:
    """Compute squared DTW distance between two sequences.

    This function uses dynamic programming to find the optimal alignment
    between two time series.

    Features:
    - Handles time series of unequal lengths.
    - Uses Sakoe-Chiba warping window to limit the search space (optional).
    - Uses Euclidean distance as the local cost measure.
    - Returns squared distance to avoid unnecessary square roots.
    - Uses Numba for JIT compilation for maximum performance.
    
    Parameters
    ----------
    x : np.ndarray
        First time series of shape (n, d) or (n,).
    y : np.ndarray
        Second time series of shape (m, d) or (m,).
    w : int, optional
        Sakoe Chiba Warping Window
    random_path : bool, optional
        If True, uses random choice when backtracking the optimal path in case of ties.


    Returns
    -------
    distance : float
        DTW distance.
    path : List[Tuple[int, int]]
        Optimal warping path as list of index pairs (i,j).
    
        
    Example (univariate)
    --------------------
        >>> x = np.random.randn(500)
        >>> y = np.random.randn(600)
        >>> dist, path = dtw(x, y) 
    
        
    Example (multivariate (d=3) with warping window of 50)
    ------------------------------------------------------
        >>> x = np.random.randn(500, 3)
        >>> y = np.random.randn(600, 3)
        >>> dist, path = dtw(x, y, w=50)
    """

    x = as_time_series(x)
    y = as_time_series(y)

    n, m = x.shape[0], y.shape[0]

    if w is None:
        w = max(n, m)  # No restriction
    else:
        w = max(w, abs(n - m))  # Ensure window is valid
    
    assert w is not None  # For Pylance type checkers

    dtw_matrix = build_matrix(x, y, w)
    distance = float(dtw_matrix[n, m])

    if random_path:
        path_i, path_j = backtrack_path_random_choice(dtw_matrix)
    else:
        path_i, path_j = backtrack_path(dtw_matrix)
    
    path = list(zip(path_i.tolist(), path_j.tolist()))
    
    return distance, path

def dtwr(x: np.ndarray, y: np.ndarray, w: int | None = None) -> Tuple[float, List[Tuple[int, int]]]:
    """Compute squared DTW distance between two sequences with random path selection.
        
        Parameters
        ----------
        x : np.ndarray
            First time series of shape (n, d) or (n,).
        y : np.ndarray
            Second time series of shape (m, d) or (m,).
        w : int, optional
            Sakoe Chiba Warping Window

        Returns
        -------
        distance : float
            DTW distance.
        path : List[Tuple[int, int]]
            Optimal warping path as list of index pairs (i,j).
        
            
        Example (univariate)
        --------------------
            >>> x = np.random.randn(500)
            >>> y = np.random.randn(600)
            >>> dist, path = dtwr(x, y) 
        
            
        Example (multivariate (d=3) with warping window of 50)
        ------------------------------------------------------
            >>> x = np.random.randn(500, 3)
            >>> y = np.random.randn(600, 3)
            >>> dist, path = dtwr(x, y, w=50)
    """
    return dtw(x, y, w=w, random_path=True)


def cost(x : np.ndarray, y : np.ndarray, path : List[Tuple[int, int]]) -> float:
    """Compute cost of aligning two time series wrt. a specific warping path"""
    return float(np.sum([np.linalg.norm(x[i] - y[j])**2 for (i, j) in path]))



def as_time_series(x : np.ndarray) -> np.ndarray:
    """
    Normalize time series to shape (n, d).
    Accepts shape (n,) or (n, d).
    """
    if not isinstance(x, np.ndarray):
        raise TypeError("Time series must be a numpy array.")
    if x.ndim == 1:
        return x[:, None]
    if x.ndim == 2:
        return x
    raise ValueError("Time series must have shape (n,) or (n, d)")



@njit()
def build_matrix(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    """Build DTW (accumulated) cost matrix with Sakoe-Chiba band."""
    n, m = x.shape[0], y.shape[0]
    dtw_matrix = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    dtw_matrix[0, 0] = 0.0
    for i in range(1, n + 1):
        band_start = max(1, i - window)
        band_end = min(m + 1, i + window + 1)
        for j in range(band_start, band_end):
            cost = 0.0
            for d in range(x.shape[1]):
                diff = x[i-1,d] - y[j-1,d]
                cost = cost + diff * diff

            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],
                dtw_matrix[i, j-1],
                dtw_matrix[i-1, j-1]
            )
    return dtw_matrix

@njit()
def backtrack_path(dtw_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Find optimal warping path in DTW matrix."""
    n, m = dtw_matrix.shape[0] - 1, dtw_matrix.shape[1] - 1
    
    max_len = n + m + 1
    path_i = np.full(max_len, -1, dtype=np.int64)
    path_j = np.full(max_len, -1, dtype=np.int64)

    
    i, j = n, m
    len_path = 0
    while i > 0 or j > 0:
        path_i[len_path] = i - 1
        path_j[len_path] = j - 1
        len_path += 1
        
        if i == 0:
            j -= 1
            continue
        if j == 0:
            i -= 1
            continue
        
        next = min(
            dtw_matrix[i-1, j],
            dtw_matrix[i, j-1],
            dtw_matrix[i-1, j-1]
        ) 
        if dtw_matrix[i-1, j-1] == next:
            i -= 1
            j -= 1
        elif dtw_matrix[i-1, j] == next:
            i -= 1
        else:
            j -= 1
    
    # Reverse and cut to length
    return np.flip(path_i[:len_path]), np.flip(path_j[:len_path])


@njit()
def backtrack_path_random_choice(dtw_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Find optimal warping path in DTW matrix."""
    n, m = dtw_matrix.shape[0] - 1, dtw_matrix.shape[1] - 1
    
    max_len = n + m + 1
    path_i = np.full(max_len, -1, dtype=np.int64)
    path_j = np.full(max_len, -1, dtype=np.int64)

    
    i, j = n, m
    len_path = 0
    while i > 0 or j > 0:
        path_i[len_path] = i - 1
        path_j[len_path] = j - 1
        len_path += 1
        
        if i == 0:
            j -= 1
            continue
        if j == 0:
            i -= 1
            continue
        
        v_diag = dtw_matrix[i-1, j-1]
        v_up = dtw_matrix[i-1, j]
        v_left = dtw_matrix[i, j-1]

        min_val = min(v_diag, v_up, v_left)

        candidates = []
        if np.isclose(v_diag, min_val):
            candidates.append(0)
        if np.isclose(v_up, min_val):
            candidates.append(1)
        if np.isclose(v_left, min_val):
            candidates.append(2)

        choice = candidates[np.random.randint(len(candidates))]
        if choice == 0:
            i -= 1
            j -= 1
        elif choice == 1:
            i -= 1
        else:
            j -= 1
    
    # Reverse and cut to length
    return np.flip(path_i[:len_path]), np.flip(path_j[:len_path])


# Example usage
if __name__ == "__main__":
    x = np.random.randn(1000)
    y = np.random.randn(1000)

    # warmup ! do not measure the first run, because of numba jit compilation
    dist, path = dtw(x[:5], y[:5])
    
    # runtime measurement
    start = time.perf_counter()
    dist, path = dtw(x, y, w=100)
    elapsed = time.perf_counter() - start
    
    print(f"Elapsed time: {elapsed:.6f}s")
    print(f"DTW distance: {dist:.4f}")
    print(f"Length of warping path: {len(path)}")
    print(f"Cost of warping path: {cost(x, y, path):.4f}")
