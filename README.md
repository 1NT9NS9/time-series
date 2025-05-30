# Time Series Analysis with Python

This project provides a comprehensive time series analysis toolkit using Python. It demonstrates key time series operations including data loading, resampling, statistical calculations, stationarity testing, and visualization.

## Features

- **Data Loading**: Fetch financial time series data using yfinance
- **DatetimeIndex Management**: Proper handling of datetime indices
- **Data Resampling**: Convert daily data to monthly frequency
- **Moving Statistics**: Calculate moving averages and standard deviations
- **Stationarity Testing**: Augmented Dickey-Fuller (ADF) test implementation
- **Comprehensive Visualization**: Multiple plots for time series analysis

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

Run the complete analysis with default settings (Bitcoin, 5 years):

```python
from time_series import TimeSeriesAnalyzer

# Initialize analyzer
analyzer = TimeSeriesAnalyzer(symbol='BTC-USD', period='5y')

# Run complete analysis
results = analyzer.run_complete_analysis()
```

### Custom Analysis

```python
# Analyze different cryptocurrency with custom parameters
analyzer = TimeSeriesAnalyzer(symbol='ETH-USD', period='3y')

# Run analysis with custom moving average windows
results = analyzer.run_complete_analysis(windows=[6, 12, 18, 24])
```

### Step-by-Step Analysis

```python
# Initialize analyzer
analyzer = TimeSeriesAnalyzer(symbol='BTC-USD', period='2y')

# Step 1: Load data
data = analyzer.load_data()

# Step 2: Resample to monthly
monthly_data = analyzer.resample_to_monthly()

# Step 3: Calculate moving statistics
stats = analyzer.calculate_moving_statistics(windows=[6, 12])

# Step 4: Test for stationarity
adf_results = analyzer.adf_test()

# Step 5: Create visualizations
analyzer.plot_time_series()
```

## Class Methods

### `TimeSeriesAnalyzer`

#### `__init__(symbol='BTC-USD', period='5y')`
Initialize the analyzer with a symbol and time period.

#### `load_data()`
Download and load time series data using yfinance. Sets proper DatetimeIndex.

#### `resample_to_monthly(column='Close')`
Resample daily data to monthly frequency using the last value of each month.

#### `calculate_moving_statistics(data=None, windows=[12, 24])`
Calculate moving averages and standard deviations for specified window sizes.

#### `adf_test(data=None, significance_level=0.05)`
Perform Augmented Dickey-Fuller test to check for stationarity.

#### `plot_time_series(figsize=(15, 12))`
Create comprehensive visualization with 4 subplots:
- Daily closing prices
- Monthly resampled data
- Moving averages
- Moving standard deviations

#### `run_complete_analysis(symbol=None, windows=[12, 24])`
Execute the complete analysis pipeline and return all results.

## Output

The analysis provides:

1. **Data Information**: Date ranges, data shapes, and basic statistics
2. **Stationarity Test Results**: ADF test statistics, p-values, and interpretation
3. **Visualizations**: Four-panel plot showing different aspects of the time series
4. **Summary Statistics**: Descriptive statistics for the monthly data

## Example Output

```
============================================================
COMPLETE TIME SERIES ANALYSIS FOR BTC-USD
============================================================
Loading data for BTC-USD...
Data loaded successfully!
Date range: 2019-01-02 00:00:00 to 2024-01-01 00:00:00
Shape: (1258, 7)
Resampling Close data from daily to monthly...
Monthly data shape: (60,)
Monthly date range: 2019-01-31 00:00:00 to 2023-12-31 00:00:00
Calculating moving statistics...
Moving statistics calculated for windows: [12, 24]

==================================================
AUGMENTED DICKEY-FULLER TEST FOR STATIONARITY
==================================================
ADF Statistic: -1.234567
p-value: 0.123456
Critical Values:
	1%: -3.123456
	5%: -2.123456
	10%: -1.123456

Interpretation:
✗ Fail to reject null hypothesis (p-value 0.123456 > 0.05)
✗ The time series is NON-STATIONARY
```

## Dependencies

- `yfinance`: For downloading financial data
- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computing
- `matplotlib`: Plotting and visualization
- `seaborn`: Statistical data visualization
- `statsmodels`: Statistical modeling (ADF test)

## Notes

- The script uses the closing price by default for analysis
- Monthly resampling takes the last value of each month
- The ADF test helps determine if differencing is needed for time series modeling
- Non-stationary series may require differencing or transformation for certain analyses

## Customization

You can easily modify the script to:
- Use different data sources
- Apply different resampling frequencies
- Add more statistical tests
- Include additional visualization types
- Analyze multiple time series simultaneously 