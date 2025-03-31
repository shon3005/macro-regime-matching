# Macro Regime Analysis and Matching

This repository contains tools for analyzing and identifying similar macroeconomic regimes throughout history. It fetches data from various sources, processes it, and applies machine learning techniques to identify historical periods that most closely match the current economic environment.

## Overview

The system works by:
1. Fetching macroeconomic data from FRED API and Yahoo Finance
2. Processing and filling any missing values
3. Analyzing the data to identify similar historical regimes
4. Generating visualizations and outputs for analysis

## Requirements

- Python 3.6+
- FRED API key (obtain from https://fred.stlouisfed.org/docs/api/api_key.html)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/macro-regime-matching.git
   cd macro-regime-matching
   ```

2. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the root directory with your FRED API key:
   ```
   FRED_API_KEY=your_fred_api_key_here
   ```

## Data Collection

### 1. Fetch Macro Data

To collect macroeconomic data from FRED and other sources:

```
python fetch_macro_data.py
```

This will create a CSV file containing various macroeconomic indicators including:
- GDP, Industrial Production, and other growth metrics
- Inflation measures (CPI, PCE)
- Employment statistics
- Financial market indicators
- Housing data
- And more

### 2. Fill Latest Values

To handle missing values and ensure the dataset is up-to-date:

```
python fill_latest_values.py
```

This script will:
- Identify and fill missing values
- Ensure the most recent data points are included
- Prepare the data for analysis

## Running the Analysis

### Refined Analysis

For a more detailed analysis with carefully selected variables:

```
python refined_regime_analysis.py
```

This will:
1. Process the data using a refined set of variables
2. Calculate various metrics (percentage changes, absolute changes, etc.)
3. Perform similarity analysis to identify historical regimes matching the current environment
4. Generate visualizations and output files in the `results_refined` directory

## Output and Results

The analysis generates various outputs in dedicated directories:
- `results_refined`: Refined analysis with carefully selected variables

Key outputs include:
- CSV files with similarity scores
- Visualizations of similar regimes
- Time series comparisons
- Feature importance analysis
