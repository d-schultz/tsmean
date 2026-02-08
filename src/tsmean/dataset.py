import numpy as np
import os 
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from pathlib import Path


# UCR Archive download information
UCR_ARCHIVE_URL = "https://www.cs.ucr.edu/~eamonn/time_series_data_2018/"
UCR_ARCHIVE_INFO = """
The UCR Time Series Archive is required but not found.

To use the UCR datasets:
1. Download the archive from: {url}
2. Extract the archive to a directory
3. Configure the path using ONE of these methods:

   a) Programmatically (recommended for notebooks):
      >>> import tsmean
      >>> tsmean.set_ucr_path('/path/to/UCRArchive_2018/')
   
   b) Environment variable (recommended for scripts):
      $ export TSMEAN_UCR_PATH=/path/to/UCRArchive_2018/
   
   c) Config file (recommended for permanent setup):
      $ mkdir -p ~/.tsmean
      $ echo "UCR_PATH=/path/to/UCRArchive_2018/" > ~/.tsmean/config

Note: The UCR archive may be password-protected.
Please respect the terms of use set by the archive maintainers.
"""


def _get_ucr_path():
    """Get the UCR archive path from multiple sources (in order of priority):
    
    1. Programmatically set path via set_ucr_path()
    2. Environment variable TSMEAN_UCR_PATH
    3. User config file ~/.tsmean/config
    4. Raise error with instructions (no silent fallback)
    
    Returns
    -------
    str
        Path to the UCR archive directory
        
    Raises
    ------
    FileNotFoundError
        If no UCR path is configured
    """
    # 1. Check if path was set programmatically
    if hasattr(_get_ucr_path, '_custom_path') and _get_ucr_path._custom_path is not None:
        return _get_ucr_path._custom_path
    
    # 2. Check environment variable
    env_path = os.environ.get('TSMEAN_UCR_PATH')
    if env_path:
        return env_path
    
    # 3. Check user config file
    config_path = Path.home() / '.tsmean' / 'config'
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                for line in f:
                    if line.startswith('UCR_PATH='):
                        return line.split('=', 1)[1].strip()
        except Exception:
            pass
    
    # 4. No path configured - raise helpful error
    raise FileNotFoundError(UCR_ARCHIVE_INFO.format(url=UCR_ARCHIVE_URL))


def set_ucr_path(path):
    """Set the UCR archive path programmatically.
    
    This setting takes precedence over environment variables and config files.
    The path is validated to ensure it exists and contains the expected structure.
    If DataSummary.csv is missing, it will be automatically downloaded.
    
    Parameters
    ----------
    path : str or Path
        Path to the UCR archive directory
        
    Raises
    ------
    FileNotFoundError
        If the path does not exist
    ValueError
        If the path exists but doesn't appear to be a valid UCR archive
        
    Examples
    --------
    >>> import tsmean
    >>> tsmean.set_ucr_path('/data/UCRArchive_2018/')
    >>> # Now all dataset loading functions will use this path
    """
    path = str(path)
    
    # Validate path exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path does not exist: {path}")
    
    # Check if it looks like UCR archive (check for DataSummary.csv)
    summary_path = os.path.join(path, 'DataSummary.csv')
    if not os.path.exists(summary_path):
        # Try to download DataSummary.csv automatically
        try:
            _download_data_summary(path)
        except Exception as e:
            raise ValueError(
                f"Path exists but doesn't appear to be a valid UCR archive.\n"
                f"Expected to find 'DataSummary.csv' in: {path}\n"
                f"Attempted to download DataSummary.csv but failed: {e}\n"
                f"Please ensure you've extracted the full UCR archive."
            )
    
    _get_ucr_path._custom_path = path


def _download_data_summary(ucr_path):
    """Download DataSummary.csv to the UCR archive directory.
    
    DataSummary.csv is a publicly available metadata file that provides
    information about all datasets in the UCR archive. It's not included
    in the main archive download but is useful for filtering and searching.
    
    Parameters
    ----------
    ucr_path : str
        Path to the UCR archive directory
        
    Raises
    ------
    Exception
        If download fails
    """
    import urllib.request
    import urllib.error
    import ssl
    
    url = "https://www.cs.ucr.edu/~eamonn/time_series_data_2018/DataSummary.csv"
    dest_path = os.path.join(ucr_path, 'DataSummary.csv')
    
    print(f"DataSummary.csv not found. Downloading from {url}...")
    
    # Try with default SSL context first
    try:
        urllib.request.urlretrieve(url, dest_path)
        print(f"✓ Successfully downloaded DataSummary.csv to {dest_path}")
        return
    except urllib.error.URLError as e:
        # Check if it's an SSL error
        if 'CERTIFICATE_VERIFY_FAILED' in str(e) or isinstance(e.reason, ssl.SSLError):
            # SSL error - try with unverified context as fallback
            print(f"⚠ SSL verification failed, trying with unverified context...")
            try:
                # Create unverified SSL context (only for this specific download)
                ssl_context = ssl._create_unverified_context()
                with urllib.request.urlopen(url, context=ssl_context) as response:
                    with open(dest_path, 'wb') as out_file:
                        out_file.write(response.read())
                print(f"✓ Successfully downloaded DataSummary.csv to {dest_path}")
                return
            except Exception as e2:
                raise Exception(
                    f"Failed to download DataSummary.csv even with unverified SSL context.\n"
                    f"Original error: {e}\n"
                    f"Fallback error: {e2}\n"
                    f"Please manually download from: {url}"
                )
        else:
            # Non-SSL error
            raise Exception(
                f"Failed to download DataSummary.csv: {e}\n"
                f"Please manually download from: {url}"
            )
    except Exception as e:
        raise Exception(
            f"Failed to download DataSummary.csv: {e}\n"
            f"Please manually download from: {url}"
        )


def get_ucr_path():
    """Get the currently configured UCR archive path.
    
    Returns
    -------
    str
        Path to the UCR archive directory
        
    Raises
    ------
    FileNotFoundError
        If no UCR path is configured
        
    Examples
    --------
    >>> import tsmean
    >>> print(tsmean.get_ucr_path())
    '/data/UCRArchive_2018/'
    """
    return _get_ucr_path()


def validate_ucr_archive(path=None, verbose=True, download_summary=True):
    """Validate that the UCR archive is properly configured and accessible.
    
    Parameters
    ----------
    path : str or Path, optional
        Path to validate. If None, uses the currently configured path.
    verbose : bool, default=True
        If True, print validation results
    download_summary : bool, default=True
        If True and DataSummary.csv is missing, attempt to download it automatically
        
    Returns
    -------
    bool
        True if archive is valid, False otherwise
        
    Examples
    --------
    >>> import tsmean
    >>> tsmean.validate_ucr_archive('/data/UCRArchive_2018/')
    ✓ Path exists
    ✓ DataSummary.csv found
    ✓ Found 128 dataset directories
    Archive is valid!
    True
    """
    if path is None:
        try:
            path = _get_ucr_path()
        except FileNotFoundError as e:
            if verbose:
                print("✗ No UCR path configured")
                print(str(e))
            return False
    
    path = str(path)
    
    # Check 1: Path exists
    if not os.path.exists(path):
        if verbose:
            print(f"✗ Path does not exist: {path}")
        return False
    if verbose:
        print(f"✓ Path exists: {path}")
    
    # Check 2: DataSummary.csv exists
    summary_path = os.path.join(path, 'DataSummary.csv')
    if not os.path.exists(summary_path):
        if download_summary:
            if verbose:
                print("⚠ DataSummary.csv not found, attempting to download...")
            try:
                _download_data_summary(path)
                if verbose:
                    print("✓ DataSummary.csv downloaded successfully")
            except Exception as e:
                if verbose:
                    print(f"✗ Failed to download DataSummary.csv: {e}")
                return False
        else:
            if verbose:
                print("✗ DataSummary.csv not found")
            return False
    else:
        if verbose:
            print("✓ DataSummary.csv found")
    
    # Check 3: Count dataset directories
    try:
        dataset_dirs = [d for d in os.listdir(path) 
                       if os.path.isdir(os.path.join(path, d)) 
                       and d != 'Missing_value_and_variable_length_datasets_adjusted']
        if verbose:
            print(f"✓ Found {len(dataset_dirs)} dataset directories")
        
        if len(dataset_dirs) < 100:  # UCR archive has 128 datasets
            if verbose:
                print(f"⚠ Warning: Expected ~128 datasets, found {len(dataset_dirs)}")
    except Exception as e:
        if verbose:
            print(f"✗ Error reading directory: {e}")
        return False
    
    if verbose:
        print("\n✓ Archive is valid!")
    return True


# Initialize custom path storage
_get_ucr_path._custom_path = None

def _get_ucr_summary_path():
    """Get the path to the UCR summary CSV file.
    
    If DataSummary.csv doesn't exist, attempts to download it automatically.
    
    Returns
    -------
    str
        Path to DataSummary.csv
        
    Raises
    ------
    FileNotFoundError
        If DataSummary.csv cannot be found or downloaded
    """
    ucr_path = _get_ucr_path()
    summary_path = os.path.join(ucr_path, 'DataSummary.csv')
    
    # If DataSummary.csv doesn't exist, try to download it
    if not os.path.exists(summary_path):
        try:
            _download_data_summary(ucr_path)
        except Exception as e:
            raise FileNotFoundError(
                f"DataSummary.csv not found at {summary_path} and automatic download failed.\n"
                f"Error: {e}\n"
                f"You can manually download it from: "
                f"https://www.cs.ucr.edu/~eamonn/time_series_data_2018/DataSummary.csv"
            )
    
    return summary_path



# List of all datasets in the UCR archive
DATASETS = ['ACSF1', 'Adiac', 'AllGestureWiimoteX', 'AllGestureWiimoteY', 'AllGestureWiimoteZ', 'ArrowHead', 'BME', 'Beef', 'BeetleFly', 'BirdChicken', 'CBF', 'Car', 'Chinatown', 'ChlorineConcentration', 'CinCECGTorso', 'Coffee', 'Computers', 'CricketX', 'CricketY', 'CricketZ', 'Crop', 'DiatomSizeReduction', 'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect', 'DistalPhalanxTW', 'DodgerLoopDay', 'DodgerLoopGame', 'DodgerLoopWeekend', 'ECG200', 'ECG5000', 'ECGFiveDays', 'EOGHorizontalSignal', 'EOGVerticalSignal', 'Earthquakes', 'ElectricDevices', 'EthanolLevel', 'FaceAll', 'FaceFour', 'FacesUCR', 'FiftyWords', 'Fish', 'FordA', 'FordB', 'FreezerRegularTrain', 'FreezerSmallTrain', 'Fungi', 'GestureMidAirD1', 'GestureMidAirD2', 'GestureMidAirD3', 'GesturePebbleZ1', 'GesturePebbleZ2', 'GunPoint', 'GunPointAgeSpan', 'GunPointMaleVersusFemale', 'GunPointOldVersusYoung', 'Ham', 'HandOutlines', 'Haptics', 'Herring', 'HouseTwenty', 'InlineSkate', 'InsectEPGRegularTrain', 'InsectEPGSmallTrain', 'InsectWingbeatSound', 'ItalyPowerDemand', 'LargeKitchenAppliances', 'Lightning2', 'Lightning7', 'Mallat', 'Meat', 'MedicalImages', 'MelbournePedestrian', 'MiddlePhalanxOutlineAgeGroup', 'MiddlePhalanxOutlineCorrect', 'MiddlePhalanxTW', 'MixedShapesRegularTrain', 'MixedShapesSmallTrain', 'MoteStrain', 'NonInvasiveFetalECGThorax1', 'NonInvasiveFetalECGThorax2', 'OSULeaf', 'OliveOil', 'PLAID', 'PhalangesOutlinesCorrect', 'Phoneme', 'PickupGestureWiimoteZ', 'PigAirwayPressure', 'PigArtPressure', 'PigCVP', 'Plane', 'PowerCons', 'ProximalPhalanxOutlineAgeGroup', 'ProximalPhalanxOutlineCorrect', 'ProximalPhalanxTW', 'RefrigerationDevices', 'Rock', 'ScreenType', 'SemgHandGenderCh2', 'SemgHandMovementCh2', 'SemgHandSubjectCh2', 'ShakeGestureWiimoteZ', 'ShapeletSim', 'ShapesAll', 'SmallKitchenAppliances', 'SmoothSubspace', 'SonyAIBORobotSurface1', 'SonyAIBORobotSurface2', 'StarLightCurves', 'Strawberry', 'SwedishLeaf', 'Symbols', 'SyntheticControl', 'ToeSegmentation1', 'ToeSegmentation2', 'Trace', 'TwoLeadECG', 'TwoPatterns', 'UMD', 'UWaveGestureLibraryAll', 'UWaveGestureLibraryX', 'UWaveGestureLibraryY', 'UWaveGestureLibraryZ', 'Wafer', 'Wine', 'WordSynonyms', 'Worms', 'WormsTwoClass', 'Yoga']

def load_ucr_dataset(name, include_train=True, include_test=True, include_labels=False, max_sample_size=np.inf, random_sample=False, remove_nan=True, classes=None):
    """Load datasets from the UCR Time Series Classification Archive.

    Parameters
    ----------
    name : str
        The name of the dataset to load (must match the folder name in the archive).
    include_train : bool, default=True
        Whether to include training data.
    include_test : bool, default=True
        Whether to include testing data.
    include_labels : bool, default=False
        Whether to return labels along with the data.
    max_sample_size : float or int, default=np.inf
        The maximum number of samples to load.
    random_sample : bool, default=False
        Whether to randomly sample data if available samples exceed `max_sample_size`.
    remove_nan : bool, default=True
        Whether to remove NaN values from the time series data.
    classes : list or range, optional
        A collection of class labels to filter the data. If None, all classes are included.

    Returns
    -------
    X : list of np.ndarray
        The time series data. Returned if `include_labels` is False.
    (X, y) : tuple
        Tuple of (data, labels) if `include_labels` is True.

    Raises
    ------
    FileNotFoundError
        If the dataset files cannot be found at the configured path.
    ValueError
        If filtering by specified classes results in an empty dataset.

    Notes
    -----
    - The function assumes files are in `<name>_TRAIN.tsv` and `<name>_TEST.tsv` format.
    - If `max_sample_size` is None, it is treated as `np.inf`.

    Examples
    --------
    >>> X, y = load_ucr_dataset('ArrowHead', include_labels=True)
    >>> X_train = load_ucr_dataset('ArrowHead', include_test=False)
    """
    if max_sample_size is None:
        max_sample_size=np.inf

    if classes is not None and max_sample_size < np.inf and random_sample is False:
        warnings.warn("random_sampe is set to True, because max_sample_size and classes are specified")
        random_sample = True
    
    X = []
    y = []
    current_sample_size = 0
    if include_train:
        path = os.path.join(_get_ucr_path(), name, name + '_TRAIN.tsv')
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                if current_sample_size < max_sample_size or random_sample:
                    data = line.split()
                    label = int(data[0])
                    x = np.array(data[1:], dtype=float)
                    X.append(x)
                    y.append(label)
                    current_sample_size += 1

    if include_test:
        path = os.path.join(_get_ucr_path(), name, name + '_TEST.tsv')
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                if current_sample_size < max_sample_size or random_sample:
                    data = line.split()
                    label = int(data[0])
                    x = np.array(data[1:], dtype=float)
                    X.append(x)
                    y.append(label)
                    current_sample_size += 1
    

    # filter for specified classes
    if classes is not None:
        ind = [y_i in classes for y_i in y]

        X = [X[i] for i,b in enumerate(ind) if b is True]
        y = [y[i] for i,b in enumerate(ind) if b is True]

        if y is None or y == [] or np.size(y) == 0:
            raise Exception('Empty dataset produced, because specified classes do not match labels in dataset')
        
        current_sample_size = len(y)

    if random_sample and current_sample_size > max_sample_size:
        idx = np.random.choice(range(current_sample_size),max_sample_size,replace=False)
        X = [X[i] for i in idx]
        y = [y[i] for i in idx]

    if remove_nan:
        for i,x in enumerate(X):
            X[i] = x = x[~np.isnan(x)]

    if include_labels:
        return X,y
    else:
        return X


def get_all_ucr_dataset_names():
    """Get sorted list of all dataset names available in the UCR archive.

    Returns
    -------
    list of str
        Sorted list of dataset names.
    """
    ucr_path = _get_ucr_path()
    filenames= os.listdir(ucr_path) 
    # print(filenames)
    result = []
    for filename in filenames:
        if not filename.startswith('.') and os.path.isdir(os.path.join(ucr_path, filename)): # check whether the current object is a folder or not
            result.append(filename)
    if result.count('Missing_value_and_variable_length_datasets_adjusted'):
        result.remove('Missing_value_and_variable_length_datasets_adjusted')
    result.sort()
    return result

def get_random_ucr_datasets(num=1):
    """Get a list of random dataset names from the UCR archive.

    Parameters
    ----------
    num : int, default=1
        Number of random dataset names to return.

    Returns
    -------
    list of str
        List of randomly selected dataset names.
    """
    N = len(DATASETS)
    if num >= N:
        return DATASETS
    else:
        return np.random.choice(DATASETS,num, replace=False).tolist()

def __get_ucr_summary_df(expand=True):
    """Load and process the UCR DataSummary.csv metadata and return it as a DataFrame.

    Internal helper function.
    """
    df = pd.read_csv(_get_ucr_summary_path(), sep=",", header=0)
    if expand:
        df["Total"] = df["Train "] + df["Test "]
        df["Cost Train"] = [row["Train "] * ( int(row["Length"])**2 if row["Length"] != 'Vary' else 1) for i,row in df.iterrows()]
        df["Cost Test"] = [row["Test "] * ( int(row["Length"])**2 if row["Length"] != 'Vary' else 1) for i,row in df.iterrows()]
        df["Cost Total"] = [row["Total"] * ( int(row["Length"])**2 if row["Length"] != 'Vary' else 1) for i,row in df.iterrows()]
    return df

def get_ucr_datasets(min_size=1, max_size=np.inf, size_criterion="Train ", min_length=1, max_length=np.inf, sortby="Cost Train", include_vary=False):
    """Get list of dataset names filtered by size and length criteria.

    Parameters
    ----------
    min_size : int, default=1
        Minimum number of samples.
    max_size : float or int, default=np.inf
        Maximum number of samples.
    size_criterion : {"Train ", "Test ", "Total"}, default="Train "
        Which set to use for size filtering.
    min_length : int, default=1
        Minimum time series length.
    max_length : float or int, default=np.inf
        Maximum time series length.
    sortby : {"Cost Train", "Cost Test", "Cost Total"}, default="Cost Train"
        Metric to sort results by.
    include_vary : bool, default=False
        Whether to include datasets with varying sequence lengths.

    Returns
    -------
    list of str
        Filtered and sorted list of dataset names.
    """
    assert size_criterion in ["Train ", "Test ", "Total"]
    assert sortby in ["Cost Train", "Cost Test", "Cost Total"]

    df = __get_ucr_summary_df()
    
    # filter by size
    df = df [ (df[size_criterion] >= min_size) &  (df[size_criterion] <= max_size) ]

    # filter by length
    idx_vary = df['Length'].apply(lambda x: str(x).strip() == 'Vary')
    def safe_int_filter(x):
        try:
            val = int(x)
            return min_length <= val <= max_length
        except (ValueError, TypeError):
            return False
            
    idx_size = df['Length'].apply(safe_int_filter)
    if include_vary:
        df = df[idx_vary | idx_size]
    else:
        df = df[ [not elem for elem in idx_vary] & idx_size]

    df.sort_values(by=sortby,ascending=True,inplace=True)
    datasets = list(df["Name"])
    return datasets

def get_large_ucr_datasets(min_size=5000):
    """Get list of datasets with more than the specified total samples.

    Parameters
    ----------
    min_size : int, default=5000
        Minimum total samples (train + test).

    Returns
    -------
    list of str
        List of large dataset names.
    """
    return get_ucr_datasets(min_size=min_size, size_criterion="Total", sortby="Cost Total", include_vary=True)


def plot_dataset(dataset, include_train=True, include_test=True, max_sample_size=np.inf, random_sample=True):
    """Plot time series from a UCR dataset, colored by class label.

    Parameters
    ----------
    dataset : str
        Name of the dataset to plot.
    include_train : bool, default=True
        Whether to include training sequences.
    include_test : bool, default=True
        Whether to include test sequences.
    max_sample_size : float or int, default=np.inf
        Maximum number of samples to plot.
    random_sample : bool, default=True
        Whether to randomly sample sequences.
    """
    X,y = load_ucr_dataset(dataset, include_train=include_train, include_test=include_test, include_labels=1, 
                         max_sample_size=max_sample_size, random_sample=random_sample)
    plt.figure()
    
    # Track which labels we've already added to legend
    labels_added = set()
    
    for x, yy in zip(X, y):
        # Only add label for first occurrence of each class
        if yy not in labels_added:
            plt.plot(x, color=plt.cm.tab20(yy), label=f'Class {yy}')
            labels_added.add(yy)
        else:
            plt.plot(x, color=plt.cm.tab20(yy))
        
    plt.legend()
    plt.show()
    return
    
