# UCR Archive Configuration

The `tsmean` package provides flexible configuration for the UCR Time Series Archive path.

## Quick Start

```python
import tsmean

# Set the path (with validation)
tsmean.set_ucr_path('/path/to/UCRArchive_2018/')

# Validate the archive
tsmean.validate_ucr_archive()

# Load datasets
X, y = tsmean.load_ucr_dataset('GunPoint', include_labels=True)
```

## Downloading the UCR Archive

**The UCR Time Series Archive is NOT included with tsmean and must be downloaded separately.**

1. **Visit**: https://www.cs.ucr.edu/~eamonn/time_series_data_2018/
2. **Register** and download the archive (may require password)
3. **Extract** the archive to a directory of your choice
4. **Configure** the path using one of the methods below

**Note**: 
- The main archive may be password-protected
- `DataSummary.csv` (metadata file) will be **automatically downloaded** if missing
- Please respect the terms of use set by the UCR archive maintainers

## Configuration Methods (in order of priority)

### 1. Programmatic Configuration (Recommended for Notebooks)

```python
import tsmean

# Set and validate the path
tsmean.set_ucr_path('/path/to/UCRArchive_2018/')

# The path is automatically validated:
# - Checks if path exists
# - Checks for DataSummary.csv
# - Raises clear errors if invalid

# Now load datasets
X, y = tsmean.load_ucr_dataset('GunPoint', include_labels=True)
```

### 2. Environment Variable (Recommended for Scripts/Production)

```bash
# Set environment variable before running Python
export TSMEAN_UCR_PATH=/path/to/UCRArchive_2018/

# Or in your .bashrc/.zshrc for permanent setup
echo 'export TSMEAN_UCR_PATH=/path/to/UCRArchive_2018/' >> ~/.bashrc
```

Then in Python:
```python
import tsmean

# No need to set path - it's read from environment
X = tsmean.load_ucr_dataset('Coffee')
```

### 3. User Config File (Recommended for Permanent Setup)

```bash
# Create config directory and file
mkdir -p ~/.tsmean
echo "UCR_PATH=/path/to/UCRArchive_2018/" > ~/.tsmean/config
```

Then in Python:
```python
import tsmean

# Path is automatically loaded from config file
X = tsmean.load_ucr_dataset('Beef')
```

## Validation

### Validate Archive

```python
import tsmean

# Validate specific path
tsmean.validate_ucr_archive('/path/to/UCRArchive_2018/')

# Or validate currently configured path
tsmean.set_ucr_path('/path/to/UCRArchive_2018/')
tsmean.validate_ucr_archive()
```

Output:
```
✓ Path exists: /path/to/UCRArchive_2018/
✓ DataSummary.csv found
✓ Found 128 dataset directories

✓ Archive is valid!
```

### Check Current Configuration

```python
import tsmean

# Get currently configured path
print(tsmean.get_ucr_path())
```

## Error Handling

If no path is configured, you'll get a helpful error message:

```python
import tsmean

# Without configuration
X = tsmean.load_ucr_dataset('GunPoint')
```

```
FileNotFoundError: 
The UCR Time Series Archive is required but not found.

To use the UCR datasets:
1. Download the archive from: https://www.cs.ucr.edu/~eamonn/time_series_data_2018/
2. Extract the archive to a directory
3. Configure the path using ONE of these methods:
   ...
```

## Examples

### Temporary Path Override

```python
import tsmean

# Set path for this session
tsmean.set_ucr_path('/tmp/UCRArchive_2018/')
X1 = tsmean.load_ucr_dataset('Beef')

# Change path mid-session
tsmean.set_ucr_path('/data/UCRArchive_2018/')
X2 = tsmean.load_ucr_dataset('Coffee')
```

### Docker/Container Setup

```dockerfile
FROM python:3.10

# Install tsmean
RUN pip install tsmean

# Set UCR path via environment variable
ENV TSMEAN_UCR_PATH=/data/UCRArchive_2018

# Copy your UCR archive
COPY UCRArchive_2018/ /data/UCRArchive_2018/
```

## FAQ

**Q: Why doesn't tsmean auto-download the UCR archive?**

A: The UCR archive:
- Requires registration and acceptance of terms
- Is password-protected
- Is ~2GB in size
- Should be explicitly downloaded by users who understand its terms of use

**Q: What about DataSummary.csv?**

A: `DataSummary.csv` is a small (~50KB) metadata file that is:
- **Publicly accessible** (no password required)
- **Automatically downloaded** when you configure the UCR path
- **Optional** but useful for filtering and searching datasets
- Downloaded from: https://www.cs.ucr.edu/~eamonn/time_series_data_2018/DataSummary.csv

**Q: Can I disable automatic DataSummary.csv download?**

A: Yes, use `download_summary=False` in `validate_ucr_archive()`:
```python
tsmean.validate_ucr_archive(download_summary=False)
```

**Q: Can I use a subset of the archive?**

A: Yes, but ensure the directory structure is maintained. `DataSummary.csv` will be downloaded automatically if missing.

**Q: What if I get "Path exists but doesn't appear to be a valid UCR archive"?**

A: This means the path exists but doesn't look like a UCR archive directory. Ensure you've:
- Extracted the archive (not just downloaded the .zip)
- Pointed to the root directory (e.g., `UCRArchive_2018/`, not a subdirectory)
- Have write permissions (for automatic DataSummary.csv download)

