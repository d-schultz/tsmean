__version__ = "0.1.0"
__author__ = "David Schultz"
__email__ = "dasch85@gmail.com"

# Essential Algorithms and Distance
from .warping_distance import dtw
from .mean_algorithm import (
    dba,
    ssg,
    ssg_slope,
    adam,
    adam_slope,
    adadelta,
    rmsprop
)

# Key Dataset Utilities
from .dataset import (
    set_ucr_path,
    get_ucr_path,
    load_ucr_dataset,
    validate_ucr_archive
)

# Primary Visualization
from .visualization import (
    alignment_plot,
    mean_alignment_plot
)

__all__ = [
    "dtw",
    "dba",
    "ssg",
    "ssg_slope",
    "adam",
    "adam_slope",
    "adadelta",
    "rmsprop",
    "set_ucr_path",
    "get_ucr_path",
    "load_ucr_dataset",
    "validate_ucr_archive",
    "alignment_plot",
    "mean_alignment_plot",
]
