# tsmean
The python package for Time Series Averaging under Dynamic Time Warping.

Time series averaging under DTW is formulated as an optimization problem. A (Fréchet) mean of a time series dataset $X$ is any time series $x$ that minimizes the Fréchet function

$$F(x) = \sum_{y \in X} \text{dtw}(x,y)^2.$$

# Features
The package provides
- modular subgradient methods for minimizing the Fréchet function
- a collection of optimizers, such as SGD, Adam, AdaDelta, RMSProp and a Newton method
- different step size schedules
- different termination criteria
- a DTW Barycenter Averaging (DBA) implementation
- functions for loading UCR Datasets
- visualization of alignments between time series, including mean alignment plots
- fast DTW implementation in numba

# Installation
