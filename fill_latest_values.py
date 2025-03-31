import pandas as pd
import numpy as np
from datetime import datetime

def fill_latest_values(input_file: str, output_file: str = None) -> pd.DataFrame:
    """
    Fill missing values at the end of each column with its last valid entry.
    Special handling for quarterly change-of-change series to fill zeros with next value.
    
    Parameters:
    -----------
    input_file : str
        Path to input CSV file
    output_file : str, optional
        Path to save the filled data. If None, will append '_filled' to input filename
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with filled values
    """
    # Read the data
    print(f"Reading data from {input_file}")
    df = pd.read_csv(input_file, index_col=0, parse_dates=True)
    
    # List of quarterly change-of-change columns that need special handling
    quarterly_coc = [
        'GDP_chg_3m_coc', 'GDPC1_chg_3m_coc', 'GDPPOT_chg_3m_coc',
        'Net Investment_chg_3m_coc', 'Federal Debt_chg_3m_coc',
        'Debt to GDP_chg_3m_coc', 'Govt Consumption_chg_3m_coc'
    ]
    
    # Get the last valid index for each column
    last_valid = df.apply(lambda x: x.last_valid_index())
    print("\nLast valid dates before filling:")
    for col in sorted(last_valid.index):
        if last_valid[col] is not None:
            print(f"  {col:20}: {last_valid[col].strftime('%Y-%m-%d')}")
    
    # Forward fill each column from its last valid entry to the end
    latest_date = df.index.max()
    for col in df.columns:
        last_valid_idx = df[col].last_valid_index()
        if last_valid_idx is not None and last_valid_idx < latest_date:
            last_value = df.loc[last_valid_idx, col]
            df.loc[last_valid_idx:, col] = last_value
    
    # Special handling for quarterly change-of-change series
    print("\nFilling zero values in quarterly change-of-change series:")
    for col in quarterly_coc:
        if col in df.columns:
            # Find sequences of zeros
            zero_mask = (df[col] == 0)
            if zero_mask.any():
                # For each zero value, find the next non-zero value
                for idx in df[zero_mask].index:
                    next_values = df.loc[idx:, col]
                    next_nonzero = next_values[next_values != 0]
                    if len(next_nonzero) > 0:
                        df.loc[idx, col] = next_nonzero.iloc[0]
                print(f"  Filled zeros in {col}")
    
    # Save filled data
    if output_file is None:
        output_file = input_file.replace('.csv', '_filled.csv')
    df.to_csv(output_file)
    print(f"\nSaved filled data to {output_file}")
    
    # Print summary
    print("\nFilled values summary:")
    print(f"Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"Total columns: {len(df.columns)}")
    
    return df

if __name__ == "__main__":
    # Fill missing values in macro_data.csv
    fill_latest_values('macro_data.csv')
