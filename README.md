# tsmean: Time Series Mean Under Dynamic Time Warping

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Numba](https://img.shields.io/badge/accelerated-Numba-orange.svg)](https://numba.pydata.org/)

**tsmean** is a high-performance Python library for calculating the Fréchet mean (averaging) of time series data under Dynamic Time Warping (DTW). It implements state-of-the-art stochastic subgradient methods and classic algorithms like DBA, all optimized with Numba for maximum speed.

Time series averaging under DTW is formulated as an optimization problem. A (Fréchet) mean of a time series dataset $X$ is any time series $x$ that minimizes the Fréchet function

$$F(x) = \sum_{y \in X} \text{dtw}(x,y)^2.$$

tsmean computes the Fréchet mean of a time series dataset using various optimization algorithms. A Fréchet mean under DTW averages the time series while accounting for optimal temporal alignments. A simple example of a Fréchet mean under DTW looks like this:


<p align="center">
  <img src="resources/tsmean.png" alt="tsmean Visualization">
</p>



---

## Key Features

- **Blazing Fast**: DTW distance and path calculations are JIT-compiled with [Numba](https://numba.pydata.org/), ensuring C-like performance.
- **Modern Optimizers**: Includes `Adam`, `RMSProp`, `SGD`, and `HSGD` (Hessian-scaled SGD) for robust convergence.
- **Stochastic Subgradient Methods**: Efficiently average large collections of time series using minibatch updates.
- **Classic DBA**: Optimized implementation of DTW Barycenter Averaging.
- **UCR Archive Integration**: Built-in support for loading and validating datasets from the [UCR Time Series Archive](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/).
- **Visualization Suite**: Tools to plot Fréchet variation history and time series alignments.

---

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/your-username/tsmean.git
cd tsmean
pip install .
```

For development:
```bash
pip install -e ".[dev]"
```

---

## Quick Start

Finding the mean of multiple time series is straightforward:

```python
import numpy as np
import matplotlib.pyplot as plt
from tsmean.mean_algorithm import dba, adam_slope


# Generate some sample time series (sine waves with noise)
t = np.linspace(0, 2*np.pi, 100)
X = [np.sin(t + np.random.uniform(-0.5, 0.5)) + np.random.normal(0, 0.1, 100) for _ in range(50)]

# Approximate a mean using DBA
result_dba = dba(X)
print(f"DBA Fréchet Variation: {result_dba.frechet_var:.4f}")

# Approximate a mean using Stochastic Subgradient method with Adam optimizer
result_ssg = adam_slope(X, batch_size=1, eta0=0.1, n_epochs=50, 
                        Feval_interval=50, patience=3, slope_window_size=10)
print(f"Adam Fréchet Variation: {result_ssg.frechet_var:.4f}")

# Print optimization stats
result_dba.print_stats()
result_ssg.print_stats()

# Plot optimization history
result_dba.plot()
result_ssg.plot()

# Plot the means
plt.plot(result_dba.mean, label="DBA")
plt.plot(result_ssg.mean, label="SSG")
plt.legend()
plt.title("DBA and SSG Means")
plt.show()
```

---

## Algorithms Overview

### DTW Barycenter Averaging (DBA)
A Majorize-Minimize algorithm that is equivalent to a 2nd-order subgradient method with a Newton step. It is highly effective for smaller datasets but operates on the full batch.

### Stochastic Subgradient (SSG)
Ideal for large-scale data. By using minibatches and modern optimizers like **Adam**, SSG can find high-quality averages much faster than full-batch methods.

### Convergence Monitoring
tsmean allows you to monitor convergence using:
- **Function Value Tracking**: Stop when the Fréchet variation stabilizes.
- **Slope Detection**: Stop when the improvement rate falls below a threshold.

---

## UCR Dataset Support

Easily work with the UCR Archive:

```python
import tsmean

# Set your UCR path
tsmean.set_ucr_path('/path/to/UCRArchive_2018/')

# Load a dataset
X_train, y_train = tsmean.load_ucr_dataset('CBF', include_labels=True)

# Plot it
tsmean.plot_dataset('CBF')
```

---

## License

Distributed under the MIT License. See `LICENSE` for more information.

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---
*Created by [David Schultz](mailto:dasch85@gmail.com)*
