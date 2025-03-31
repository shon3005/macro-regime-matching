import pandas as pd
import numpy as np
from fredapi import Fred
import yfinance as yf
from datetime import datetime
import os
from dotenv import load_dotenv

def load_api_key():
    """Load FRED API key from environment variables."""
    load_dotenv()
    api_key = os.getenv('FRED_API_KEY')
    if not api_key:
        raise ValueError("FRED API key not found in environment variables")
    return api_key

def get_sp500_data(start_date: str) -> pd.Series:
    """
    Fetch S&P 500 data from Yahoo Finance.
    """
    print("\nFetching S&P 500 data from Yahoo Finance...")
    try:
        # Get data up to today
        end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
        df = yf.download('^GSPC', start=start_date, end=end_date)
        
        # Convert daily to month-end using last price
        monthly = df['Close'].resample('ME').last()
        print(f"✓ S&P 500 | Available from: {monthly.index[0].strftime('%Y-%m-%d')} to {monthly.index[-1].strftime('%Y-%m-%d')}")
        return monthly
        
    except Exception as e:
        print(f"✗ Error fetching S&P 500 data: {str(e)}")
        return None

def resample_to_monthly(series: pd.Series, variable_type: str = 'stock', freq: str = 'M') -> pd.Series:
    """
    Resample data to month-end frequency.
    
    Parameters:
    -----------
    series : pd.Series
        Input time series
    variable_type : str
        Type of variable ('stock' or 'flow')
    freq : str
        Original frequency of the data ('M' or 'Q')
        
    Returns:
    --------
    pd.Series
        Resampled monthly series
    """
    if freq == 'Q':
        # First convert to quarterly end frequency
        series = series.asfreq('QE', method='ffill')
        # Then forward fill to month-end
        series = series.asfreq('ME', method='ffill')
    else:
        # For monthly data, just ensure it's month-end
        series = series.asfreq('ME')
        
    return series

def calculate_changes(data: pd.DataFrame, periods: int = 3) -> pd.DataFrame:
    """
    Calculate n-period raw value changes for each variable.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input DataFrame with variables
    periods : int
        Number of periods to calculate changes over (default: 3 months)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with change indicators for each variable
    """
    changes = pd.DataFrame(index=data.index)
    
    for col in data.columns:
        # Calculate raw value changes for all variables
        changes[f'{col}_chg_{periods}m'] = data[col].diff(periods)
    
    return changes

def calculate_momentum(data: pd.DataFrame, periods: int = 3) -> pd.DataFrame:
    """
    Calculate the change in n-period changes (change of change) for each variable.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input DataFrame with variables
    periods : int
        Number of periods for initial change calculation (default: 3 months)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with change of change indicators
    """
    # First calculate n-period changes
    changes = calculate_changes(data, periods)
    
    # Then calculate the absolute change in those changes
    coc = pd.DataFrame(index=data.index)  # coc = change of change
    for col in changes.columns:
        coc[f'{col}_coc'] = changes[col].diff().abs()  # Absolute change in the change
    
    return coc

def get_fred_data():
    """Fetch macro data from FRED API."""
    START_DATE = '2006-03-01'
    END_DATE = pd.Timestamp.now().strftime('%Y-%m-%d')  # Today's date
    fred = Fred(api_key=load_api_key())
    
    # Create full date range using month-end dates
    date_range = pd.date_range(start=START_DATE, end=END_DATE, freq='ME')
    all_data = pd.DataFrame(index=date_range)
    
    # Fetch data for each variable
    print("\nFetching data from FRED API...")
    print("=" * 50)
    
    # Define variable groups, their FRED series IDs, variable types, and frequencies
    variables = {
        'Output and Growth': {
            'GDP': ('GDP', 'flow', 'Q'),           # Gross Domestic Product (Quarterly)
            'GDPC1': ('GDPC1', 'flow', 'Q'),      # Real GDP (Quarterly)
            'GDPPOT': ('GDPPOT', 'flow', 'Q'),    # Real Potential GDP (Quarterly)
            'INDPRO': ('INDPRO', 'flow', 'M'),    # Industrial Production Index
            'TCU': ('TCU', 'stock', 'M'),         # Capacity Utilization
            'IPMAN': ('IPMAN', 'flow', 'M'),      # Industrial Production: Manufacturing
            'DGORDER': ('DGORDER', 'flow', 'M'),  # Durable Goods Orders
        },
        'Inflation': {
            'CPI': ('CPIAUCSL', 'stock', 'M'),     # Consumer Price Index
            'Core CPI': ('CPILFESL', 'stock', 'M'), # Core CPI
            'PPI': ('PPIACO', 'stock', 'M'),       # Producer Price Index
            'Core PCE': ('PCEPILFE', 'stock', 'M'), # Core PCE Price Index
            'Inflation Exp': ('MICH', 'stock', 'M'), # Michigan Inflation Expectations
        },
        'Employment': {
            'Unemployment': ('UNRATE', 'stock', 'M'),    # Unemployment Rate
            'Nonfarm Payrolls': ('PAYEMS', 'flow', 'M'), # Nonfarm Payrolls
            'Initial Claims': ('ICSA', 'flow', 'M'),    # Initial Jobless Claims
            'Labor Force Part': ('CIVPART', 'stock', 'M'), # Labor Force Participation Rate
            'Avg Hours': ('AWHNONAG', 'stock', 'M'),     # Average Weekly Hours
            'Avg Earnings': ('CES0500000003', 'stock', 'M'), # Average Hourly Earnings
            'Job Openings': ('JTSJOL', 'flow', 'M'),    # Job Openings
        },
        'Consumer': {
            'PCE': ('PCE', 'flow', 'M'),           # Personal Consumption Expenditures
            'Retail Sales': ('RSAFS', 'flow', 'M'), # Retail Sales
            'Consumer Sentiment': ('UMCSENT', 'stock', 'M'), # Consumer Sentiment
            'Savings Rate': ('PSAVERT', 'stock', 'M'), # Personal Savings Rate
            'Consumer Credit': ('TOTALSL', 'flow', 'M'), # Consumer Credit
        },
        'Housing': {
            'Housing Starts': ('HOUST', 'flow', 'M'),    # Housing Starts
            'Building Permits': ('PERMIT', 'flow', 'M'), # Building Permits
            'New Home Sales': ('HSN1F', 'flow', 'M'),   # New Home Sales
            'Home Prices': ('CSUSHPINSA', 'stock', 'M'), # Case-Shiller Home Price Index
            'Mortgage Rate': ('MORTGAGE30US', 'stock', 'M'), # 30-Year Mortgage Rate
        },
        'Financial Markets': {
            'Fed Funds': ('DFF', 'stock', 'M'),         # Federal Funds Rate
            '10Y Treasury': ('DGS10', 'stock', 'M'),    # 10-Year Treasury Rate
            'VIX': ('VIXCLS', 'stock', 'M'),           # VIX Volatility Index
            'HY Spread': ('BAMLH0A0HYM2', 'stock', 'M'), # High Yield Bond Spread
            'Dollar Index': ('DTWEXBGS', 'stock', 'M'),   # Trade Weighted Dollar Index
        },
        'International Trade': {
            'Trade Balance': ('BOPGSTB', 'flow', 'M'),  # Trade Balance
            'Net Investment': ('NETFI', 'flow', 'Q'),   # Net Foreign Investment (Quarterly)
            'Intl Reserves': ('TOTRESNS', 'flow', 'M'), # International Reserves
            'USD/CNY': ('DEXCHUS', 'stock', 'M'),       # USD/CNY Exchange Rate
            'USD/EUR': ('DEXUSEU', 'stock', 'M'),       # USD/EUR Exchange Rate
        },
        'Government': {
            'Federal Debt': ('GFDEBTN', 'flow', 'Q'),    # Federal Debt (Quarterly)
            'Budget Surplus': ('MTSDS133FMS', 'flow', 'M'), # Federal Surplus/Deficit
            'Debt to GDP': ('GFDEGDQ188S', 'stock', 'Q'),  # Federal Debt to GDP (Quarterly)
            'Govt Consumption': ('GCEC1', 'flow', 'Q'),    # Government Consumption (Quarterly)
        },
        'Business': {
            'Inventories': ('BUSINV', 'flow', 'M'),     # Business Inventories
            'Monetary Base': ('BOGMBASE', 'stock', 'M'), # Monetary Base
            'Business Loans': ('BUSLOANS', 'flow', 'M'), # Commercial and Industrial Loans
            'CFNAI': ('CFNAI', 'stock', 'M'),           # Chicago Fed National Activity Index
        },
        'Commodities': {
            'Oil Price': ('DCOILWTICO', 'stock', 'M'),   # WTI Crude Oil Price
            'Gold Price': ('IQ12260', 'stock', 'M'),     # Gold Fixing Price
            'Natural Gas': ('DHHNGSP', 'stock', 'M'),    # Natural Gas Price
            'Wheat Price': ('PWHEAMTUSDM', 'stock', 'M'), # Wheat Price
        }
    }
    
    # Fetch data for each variable
    print("\nFetching data from FRED API...")
    print("=" * 50)
    
    for category, series_dict in variables.items():
        print(f"\n{category}:")
        for name, (series_id, var_type, freq) in series_dict.items():
            try:
                # Fetch data from FRED with explicit start and end dates
                series = fred.get_series(series_id, observation_start=START_DATE, observation_end=END_DATE)
                # Resample based on variable type and frequency
                series = resample_to_monthly(series, var_type, freq)
                print(f"✓ {name:20} | Available from: {series.index[0].strftime('%Y-%m-%d')} to {series.index[-1].strftime('%Y-%m-%d')} | Freq: {freq}")
                
                # Add to DataFrame
                all_data[name] = series
                
            except Exception as e:
                print(f"✗ Error fetching {name} ({series_id}): {str(e)}")
    
    # Get S&P 500 data
    sp500 = get_sp500_data(START_DATE)
    if sp500 is not None:
        all_data['S&P 500'] = sp500
    
    # Calculate 3-month changes and change of change
    changes = calculate_changes(all_data, periods=3)
    coc = calculate_momentum(all_data, periods=3)
    
    # Combine all metrics
    combined_data = pd.concat([
        all_data,  # Original levels
        changes,   # 3-month changes
        coc       # Change of change
    ], axis=1)
    
    # Save to CSV
    output_file = 'macro_data.csv'
    combined_data.to_csv(output_file)
    print(f"\nData saved to {output_file}")
    print(f"Total variables: {len(combined_data.columns)}")
    print(f"  - Levels: {len(all_data.columns)}")
    print(f"  - 3m Changes: {len(changes.columns)}")
    print(f"  - Change of Change: {len(coc.columns)}")
    
    # Find the latest non-NaN date for each column and print
    latest_dates = []
    for col in all_data.columns:
        last_valid = all_data[col].last_valid_index()
        if last_valid is not None:
            latest_dates.append(f"{col}: {last_valid.strftime('%Y-%m-%d')}")
    
    print("\nLatest available dates:")
    for date in sorted(latest_dates, key=lambda x: x.split(': ')[1], reverse=True)[:5]:
        print(f"  {date}")
    
    return combined_data

if __name__ == "__main__":
    data = get_fred_data()
