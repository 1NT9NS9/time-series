import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class TimeSeriesAnalyzer:
    def __init__(self, symbol='BTC-USD', period='5y'):
        """
        Initialize the TimeSeriesAnalyzer
        
        Parameters:
        symbol (str): Symbol to analyze (default: 'BTC-USD' for Bitcoin)
        period (str): Period for data download (default: '5y')
        """
        self.symbol = symbol
        self.period = period
        self.data = None
        self.monthly_data = None
        
    def load_data(self):
        """Load time series data using yfinance"""
        print(f"Loading data for {self.symbol}...")
        
        # Download data
        ticker = yf.Ticker(self.symbol)
        self.data = ticker.history(period=self.period)
        
        # Ensure DatetimeIndex
        if not isinstance(self.data.index, pd.DatetimeIndex):
            self.data.index = pd.to_datetime(self.data.index)
        
        print(f"Data loaded successfully!")
        print(f"Date range: {self.data.index.min()} to {self.data.index.max()}")
        print(f"Shape: {self.data.shape}")
        
        return self.data
    
    def resample_to_monthly(self, column='Close'):
        """
        Resample daily data to monthly data
        
        Parameters:
        column (str): Column to resample (default: 'Close')
        """
        print(f"Resampling {column} data from daily to monthly...")
        
        # Resample to monthly using last value of each month
        self.monthly_data = self.data[column].resample('M').last()
        
        print(f"Monthly data shape: {self.monthly_data.shape}")
        print(f"Monthly date range: {self.monthly_data.index.min()} to {self.monthly_data.index.max()}")
        
        return self.monthly_data
    
    def calculate_moving_statistics(self, data=None, windows=[12, 24]):
        """
        Calculate moving average and standard deviation
        
        Parameters:
        data (pd.Series): Data to calculate statistics for (default: monthly_data)
        windows (list): List of window sizes for moving statistics
        """
        if data is None:
            data = self.monthly_data
        
        if data is None:
            raise ValueError("No data available. Please load and resample data first.")
        
        print("Calculating moving statistics...")
        
        # Create DataFrame to store results
        stats_df = pd.DataFrame(index=data.index)
        stats_df['Original'] = data
        
        # Calculate moving averages and standard deviations
        for window in windows:
            stats_df[f'MA_{window}'] = data.rolling(window=window).mean()
            stats_df[f'STD_{window}'] = data.rolling(window=window).std()
        
        self.stats_df = stats_df
        print(f"Moving statistics calculated for windows: {windows}")
        
        return stats_df
    
    def adf_test(self, data=None, significance_level=0.05):
        """
        Perform Augmented Dickey-Fuller test for stationarity
        
        Parameters:
        data (pd.Series): Data to test (default: monthly_data)
        significance_level (float): Significance level for the test
        """
        if data is None:
            data = self.monthly_data
        
        if data is None:
            raise ValueError("No data available. Please load and resample data first.")
        
        print("\n" + "="*50)
        print("AUGMENTED DICKEY-FULLER TEST FOR STATIONARITY")
        print("="*50)
        
        # Remove NaN values
        clean_data = data.dropna()
        
        # Perform ADF test
        adf_result = adfuller(clean_data, autolag='AIC')
        
        # Extract results
        adf_statistic = adf_result[0]
        p_value = adf_result[1]
        critical_values = adf_result[4]
        
        print(f"ADF Statistic: {adf_statistic:.6f}")
        print(f"p-value: {p_value:.6f}")
        print("Critical Values:")
        for key, value in critical_values.items():
            print(f"\t{key}: {value:.6f}")
        
        # Interpretation
        print("\nInterpretation:")
        if p_value <= significance_level:
            print(f"✓ Reject null hypothesis (p-value {p_value:.6f} <= {significance_level})")
            print("✓ The time series is STATIONARY")
        else:
            print(f"✗ Fail to reject null hypothesis (p-value {p_value:.6f} > {significance_level})")
            print("✗ The time series is NON-STATIONARY")
        
        return {
            'adf_statistic': adf_statistic,
            'p_value': p_value,
            'critical_values': critical_values,
            'is_stationary': p_value <= significance_level
        }
    
    def plot_time_series(self, figsize=(15, 12)):
        """
        Create comprehensive time series plots
        
        Parameters:
        figsize (tuple): Figure size for the plots
        """
        if self.data is None or self.monthly_data is None:
            raise ValueError("No data available. Please load and resample data first.")
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'Time Series Analysis for {self.symbol}', fontsize=16, fontweight='bold')
        
        # Plot 1: Original daily data
        axes[0, 0].plot(self.data.index, self.data['Close'], alpha=0.7, linewidth=0.8)
        axes[0, 0].set_title('Daily Closing Prices')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Price ($)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Monthly resampled data
        axes[0, 1].plot(self.monthly_data.index, self.monthly_data, 'o-', linewidth=2, markersize=4)
        axes[0, 1].set_title('Monthly Closing Prices (Resampled)')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Price ($)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Moving averages (if calculated)
        if hasattr(self, 'stats_df'):
            axes[1, 0].plot(self.stats_df.index, self.stats_df['Original'], 'o-', 
                           label='Original', linewidth=2, markersize=3)
            
            # Plot moving averages
            ma_columns = [col for col in self.stats_df.columns if col.startswith('MA_')]
            for col in ma_columns:
                axes[1, 0].plot(self.stats_df.index, self.stats_df[col], 
                               label=col, linewidth=2)
            
            axes[1, 0].set_title('Moving Averages')
            axes[1, 0].set_xlabel('Date')
            axes[1, 0].set_ylabel('Price ($)')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Moving standard deviation
            std_columns = [col for col in self.stats_df.columns if col.startswith('STD_')]
            for col in std_columns:
                axes[1, 1].plot(self.stats_df.index, self.stats_df[col], 
                               label=col, linewidth=2)
            
            axes[1, 1].set_title('Moving Standard Deviation')
            axes[1, 1].set_xlabel('Date')
            axes[1, 1].set_ylabel('Standard Deviation')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            # If no moving statistics calculated, show price distribution
            axes[1, 0].hist(self.monthly_data.dropna(), bins=30, alpha=0.7, edgecolor='black')
            axes[1, 0].set_title('Monthly Price Distribution')
            axes[1, 0].set_xlabel('Price ($)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Show returns
            returns = self.monthly_data.pct_change().dropna()
            axes[1, 1].plot(returns.index, returns, 'o-', linewidth=1, markersize=3)
            axes[1, 1].set_title('Monthly Returns')
            axes[1, 1].set_xlabel('Date')
            axes[1, 1].set_ylabel('Return (%)')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def run_complete_analysis(self, symbol=None, windows=[12, 24]):
        """
        Run complete time series analysis pipeline
        
        Parameters:
        symbol (str): Symbol to analyze (optional, uses default if not provided)
        windows (list): Window sizes for moving statistics
        """
        if symbol:
            self.symbol = symbol
        
        print("="*60)
        print(f"COMPLETE TIME SERIES ANALYSIS FOR {self.symbol}")
        print("="*60)
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Resample to monthly
        self.resample_to_monthly()
        
        # Step 3: Calculate moving statistics
        self.calculate_moving_statistics(windows=windows)
        
        # Step 4: Test for stationarity
        adf_results = self.adf_test()
        
        # Step 5: Create plots
        self.plot_time_series()
        
        # Summary statistics
        print("\n" + "="*50)
        print("SUMMARY STATISTICS")
        print("="*50)
        print(f"Symbol: {self.symbol}")
        print(f"Period: {self.period}")
        print(f"Total daily observations: {len(self.data)}")
        print(f"Total monthly observations: {len(self.monthly_data)}")
        print(f"Monthly price statistics:")
        print(self.monthly_data.describe())
        
        return {
            'data': self.data,
            'monthly_data': self.monthly_data,
            'stats_df': self.stats_df if hasattr(self, 'stats_df') else None,
            'adf_results': adf_results
        }

# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = TimeSeriesAnalyzer(symbol='BTC-USD', period='5y')
    
    # Run complete analysis
    results = analyzer.run_complete_analysis(windows=[6, 12, 24])
    
    # You can also run individual steps:
    # analyzer.load_data()
    # analyzer.resample_to_monthly()
    # analyzer.calculate_moving_statistics()
    # analyzer.adf_test()
    # analyzer.plot_time_series()
