#!/usr/bin/env python3
"""
Refined analysis of US macro regimes using a carefully selected set of variables.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create a dedicated results directory for refined analysis
results_dir = 'results_refined'
os.makedirs(results_dir, exist_ok=True)

# Load the data
data = pd.read_csv('macro_data_filled.csv', index_col=0, parse_dates=True)
print(f"Data ranges from {data.index.min()} to {data.index.max()}")

# Define the current regime (12 months is a standard business cycle window)
window = 12
current_regime = data.iloc[-window:].copy()
current_start = current_regime.index.min().strftime('%Y-%m')
current_end = current_regime.index.max().strftime('%Y-%m')
print(f"Current regime defined as: {current_start} to {current_end}")

# ----- REFINED VARIABLE SELECTION -----
# Based on user's specific requirements for variable selections

# Variables to calculate percentage change
pct_change_vars = [
    'Nonfarm Payrolls', 
    'Job Openings', 
    'Avg Earnings', 
    'Core CPI', 
    'PCE', 
    'Retail Sales', 
    'Housing Starts', 
    'Oil Price', 
    'GDP', 
    'CPI', 
    'S&P 500'
]

# Variables to calculate absolute change
abs_change_vars = [
    'INDPRO', 
    'Unemployment', 
    'CFNAI', 
    'Labor Force Part', 
    'Inflation Exp', 
    'Fed Funds', 
    '10Y Treasury', 
    'HY Spread', 
    'VIX', 
    'Consumer Sentiment'
]

# Variables to include level data
level_vars = [
    'CFNAI',
    'Inflation Exp'
]

# Check if variables exist in the dataset
missing_vars = []
for var in pct_change_vars + abs_change_vars:
    if var not in data.columns:
        missing_vars.append(var)

if missing_vars:
    print(f"Warning: The following variables are missing from the dataset: {', '.join(missing_vars)}")
    
# Total count of variables we'll use for analysis
total_vars = len(pct_change_vars) + len(abs_change_vars) - len(level_vars) + len(level_vars)
print(f"Using {total_vars} carefully selected variables for regime analysis")

# Print the variable categories
print("\nVariables by calculation type:")
print(f"Percentage change variables: {', '.join(pct_change_vars)}")
print(f"Absolute change variables: {', '.join([v for v in abs_change_vars if v not in level_vars])}")
print(f"Level variables: {', '.join(level_vars)}")

# Export the selected variables
with open(f'{results_dir}/selected_variables.csv', 'w') as f:
    f.write("Variable,Calculation_Type\n")
    for var in pct_change_vars:
        f.write(f"{var},Percentage Change\n")
    for var in abs_change_vars:
        if var not in level_vars:
            f.write(f"{var},Absolute Change\n")
    for var in level_vars:
        f.write(f"{var},Level\n")

# Process variables according to specified change types
selected_vars = pct_change_vars + [v for v in abs_change_vars if v not in level_vars] + level_vars

print(f"\nSelected variables:")
print(", ".join(selected_vars))

# Export the selected variables
with open(f'{results_dir}/selected_variables.csv', 'w') as f:
    f.write("Variable,Type\n")
    for var in pct_change_vars:
        f.write(f"{var},Percentage Change\n")
    for var in abs_change_vars:
        if var not in level_vars:
            f.write(f"{var},Absolute Change\n")
    for var in level_vars:
        f.write(f"{var},Level\n")

# ----- SIMILARITY ANALYSIS WITH STANDARDIZATION -----
# Prepare all feature sets for each time window based on specified types of changes
all_features = []
all_periods = []

# For the current regime
current_point = data.iloc[-1].copy()  # Most recent data point
year_ago_point = data.iloc[-13].copy()  # Data point from 12 months ago

# Create empty dictionary to store features
current_features = {}

# Calculate YoY percentage changes for percentage change variables
for var in pct_change_vars:
    base_var = var  # The base variable name
    if base_var in data.columns:
        # Calculate percentage change: (current - year_ago) / year_ago * 100
        if year_ago_point[base_var] != 0:  # Avoid division by zero
            pct_change = ((current_point[base_var] - year_ago_point[base_var]) / abs(year_ago_point[base_var])) * 100
        else:
            # Handle division by zero
            pct_change = 0 if current_point[base_var] == 0 else 100 if current_point[base_var] > 0 else -100
        current_features[f"{base_var}_pct"] = pct_change

# Calculate absolute changes for absolute change variables
for var in abs_change_vars:
    base_var = var  # The base variable name
    if base_var in data.columns and var not in level_vars:
        # Calculate absolute change: current - year_ago
        current_features[f"{base_var}_abs"] = current_point[base_var] - year_ago_point[base_var]

# Include level variables as-is
for var in level_vars:
    if var in data.columns:
        current_features[var] = current_point[var]

# Store the current features
current_end_date = data.index[-1].strftime('%Y-%m')
current_start_date = data.index[-13].strftime('%Y-%m')
all_features.append(current_features)
all_periods.append(f"{current_start_date} to {current_end_date}")

# For all historical windows
for i in range(12, len(data) - 1):
    hist_point = data.iloc[i].copy()
    year_ago = data.iloc[i-12].copy()
    
    # Create empty dictionary to store features
    hist_features = {}
    
    # Calculate YoY percentage changes for percentage change variables
    for var in pct_change_vars:
        base_var = var  # The base variable name
        if base_var in data.columns:
            # Calculate percentage change: (current - year_ago) / year_ago * 100
            if year_ago[base_var] != 0:  # Avoid division by zero
                pct_change = ((hist_point[base_var] - year_ago[base_var]) / abs(year_ago[base_var])) * 100
            else:
                # Handle division by zero
                pct_change = 0 if hist_point[base_var] == 0 else 100 if hist_point[base_var] > 0 else -100
            hist_features[f"{base_var}_pct"] = pct_change
    
    # Calculate absolute changes for absolute change variables
    for var in abs_change_vars:
        base_var = var  # The base variable name
        if base_var in data.columns and var not in level_vars:
            # Calculate absolute change: current - year_ago
            hist_features[f"{base_var}_abs"] = hist_point[base_var] - year_ago[base_var]
    
    # Include level variables as-is
    for var in level_vars:
        if var in data.columns:
            hist_features[var] = hist_point[var]
    
    # Store the historical features
    hist_date = data.index[i].strftime('%Y-%m')
    year_ago_date = data.index[i-12].strftime('%Y-%m')
    all_features.append(hist_features)
    all_periods.append((f"{year_ago_date} to {hist_date}", data.index[i].year))

# Convert to a dataframe for easier standardization
features_df = pd.DataFrame(all_features)

# Handle NaN and inf values that can result from calculations
features_df = features_df.replace([np.inf, -np.inf], np.nan)
features_df = features_df.fillna(features_df.mean())

# Print information about the features
print(f"\nUsing {features_df.shape[1]} calculated features for similarity analysis")
print("Feature list:")
for col in features_df.columns:
    print(f"- {col}")

# Standardize all features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features_df)

# Get the current (standardized) regime
current_scaled = scaled_features[0]

# Calculate cosine similarity between current regime and all historical regimes
cosine_scores = []
for i in range(len(scaled_features) - 1):
    similarity = cosine_similarity(current_scaled.reshape(1, -1), scaled_features[i+1].reshape(1, -1))[0][0]
    cosine_scores.append(similarity)

# Calculate Euclidean distance-based similarity
euclidean_scores = []
for i in range(len(scaled_features) - 1):
    # Distance calculation
    distance = euclidean_distances(current_scaled.reshape(1, -1), scaled_features[i+1].reshape(1, -1))[0][0]
    # Convert to similarity (higher = more similar)
    similarity = 1.0 / (1.0 + distance)
    euclidean_scores.append(similarity)

# Calculate k-nearest neighbors based similarity
k = 10  # Number of neighbors to consider
knn = NearestNeighbors(n_neighbors=min(k, len(scaled_features)-1), metric='euclidean')
knn.fit(scaled_features[1:])  # Fit on historical regimes

# Calculate distances from each historical point to each other historical point
# This will help us identify which periods have similar patterns to our current regime
all_distances = []
for i in range(len(scaled_features) - 1):
    # For each historical period, find its k nearest neighbors
    distances_i, _ = knn.kneighbors(scaled_features[i+1].reshape(1, -1))
    # Average distance to its k nearest neighbors
    avg_distance = np.mean(distances_i[0])
    all_distances.append(avg_distance)

# Calculate distances from current period to each historical period
current_distances = []
for i in range(len(scaled_features) - 1):
    # Distance from current period to this historical period
    distance = euclidean_distances(current_scaled.reshape(1, -1), scaled_features[i+1].reshape(1, -1))[0][0]
    current_distances.append(distance)

# Calculate KNN similarity scores
# A period is similar to the current regime if:
# 1. It's close to the current regime (low current_distances)
# 2. Its pattern of distances to other periods is similar to the current regime's pattern
knn_scores = []
for i in range(len(scaled_features) - 1):
    # Direct similarity based on distance to current regime
    direct_similarity = 1.0 / (1.0 + current_distances[i])
    
    # Pattern similarity - how similar are the distances from this point to others
    # compared to distances from current point to others
    # (simplification: just use the avg distance to k nearest neighbors)
    pattern_similarity = 1.0 / (1.0 + abs(all_distances[i] - np.mean(current_distances[:k])))
    
    # Combined similarity score with higher weight on direct similarity
    similarity = 0.7 * direct_similarity + 0.3 * pattern_similarity
    knn_scores.append(similarity)

# Get the indices of the k nearest neighbors for ranking purposes
_, indices = knn.kneighbors(current_scaled.reshape(1, -1))

# Create dataframes with the results from all metrics
cosine_df = pd.DataFrame({
    'Period': [p[0] for p in all_periods[1:]],
    'End_Year': [p[1] for p in all_periods[1:]],
    'Cosine_Similarity': cosine_scores
})

euclidean_df = pd.DataFrame({
    'Period': [p[0] for p in all_periods[1:]],
    'End_Year': [p[1] for p in all_periods[1:]],
    'Euclidean_Similarity': euclidean_scores
})

knn_df = pd.DataFrame({
    'Period': [p[0] for p in all_periods[1:]],
    'End_Year': [p[1] for p in all_periods[1:]],
    'KNN_Similarity': knn_scores
})

# Combine all results
results_df = cosine_df.merge(euclidean_df, on=['Period', 'End_Year'])
results_df = results_df.merge(knn_df, on=['Period', 'End_Year'])

# Export all similarity scores
results_df.to_csv(f'{results_dir}/all_similarity_metrics.csv', index=False)

# ----- DISPLAY COMPARISON RESULTS -----
# Rank by each metric
cosine_top = results_df.sort_values('Cosine_Similarity', ascending=False).head(10)
euclidean_top = results_df.sort_values('Euclidean_Similarity', ascending=False).head(10)
knn_top = results_df.sort_values('KNN_Similarity', ascending=False).head(10)

print("\nTop 10 most similar historical regimes by COSINE SIMILARITY:")
print(cosine_top[['Period', 'End_Year', 'Cosine_Similarity']].to_string(index=False))

print("\nTop 10 most similar historical regimes by EUCLIDEAN DISTANCE:")
print(euclidean_top[['Period', 'End_Year', 'Euclidean_Similarity']].to_string(index=False))

print("\nTop 10 most similar historical regimes by K-NEAREST NEIGHBORS:")
print(knn_top[['Period', 'End_Year', 'KNN_Similarity']].to_string(index=False))

# Compare the overlap between the different metrics
cosine_set = set(cosine_top['Period'])
euclidean_set = set(euclidean_top['Period'])
knn_set = set(knn_top['Period'])

print("\nOverlap analysis between different similarity metrics:")
print(f"Periods in both Cosine and Euclidean top 10: {len(cosine_set.intersection(euclidean_set))}")
print(f"Periods in both Cosine and KNN top 10: {len(cosine_set.intersection(knn_set))}")
print(f"Periods in both Euclidean and KNN top 10: {len(euclidean_set.intersection(knn_set))}")
print(f"Periods in all three metrics' top 10: {len(cosine_set.intersection(euclidean_set).intersection(knn_set))}")

# ----- CREATE VISUALIZATIONS TO COMPARE METRICS -----
plt.figure(figsize=(15, 10))

# Calculate correlations between the metrics
corr = results_df[['Cosine_Similarity', 'Euclidean_Similarity', 'KNN_Similarity']].corr()
print("\nCorrelation between similarity metrics:")
print(corr)

# Plot the correlation matrix
plt.subplot(2, 2, 1)
sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation between Similarity Metrics')

# Plot the top periods by each metric
plt.subplot(2, 2, 2)
plt.scatter(results_df['Cosine_Similarity'], results_df['Euclidean_Similarity'], alpha=0.7)
plt.xlabel('Cosine Similarity')
plt.ylabel('Euclidean Similarity')
plt.title('Cosine vs Euclidean')

# Highlight the points that are in the top 10 of both metrics
for idx, row in results_df.iterrows():
    if row['Period'] in cosine_set.intersection(euclidean_set):
        plt.scatter(row['Cosine_Similarity'], row['Euclidean_Similarity'], color='red', s=100)
        plt.annotate(row['Period'].split(' to ')[1], 
                    (row['Cosine_Similarity'], row['Euclidean_Similarity']),
                    xytext=(5, 5), textcoords='offset points')

plt.subplot(2, 2, 3)
plt.scatter(results_df['Cosine_Similarity'], results_df['KNN_Similarity'], alpha=0.7)
plt.xlabel('Cosine Similarity')
plt.ylabel('KNN Similarity')
plt.title('Cosine vs KNN')

# Highlight the points that are in the top 10 of both metrics
for idx, row in results_df.iterrows():
    if row['Period'] in cosine_set.intersection(knn_set):
        plt.scatter(row['Cosine_Similarity'], row['KNN_Similarity'], color='red', s=100)
        plt.annotate(row['Period'].split(' to ')[1], 
                    (row['Cosine_Similarity'], row['KNN_Similarity']),
                    xytext=(5, 5), textcoords='offset points')

plt.subplot(2, 2, 4)
plt.scatter(results_df['Euclidean_Similarity'], results_df['KNN_Similarity'], alpha=0.7)
plt.xlabel('Euclidean Similarity')
plt.ylabel('KNN Similarity')
plt.title('Euclidean vs KNN')

# Highlight the points that are in the top 10 of both metrics
for idx, row in results_df.iterrows():
    if row['Period'] in euclidean_set.intersection(knn_set):
        plt.scatter(row['Euclidean_Similarity'], row['KNN_Similarity'], color='red', s=100)
        plt.annotate(row['Period'].split(' to ')[1], 
                    (row['Euclidean_Similarity'], row['KNN_Similarity']),
                    xytext=(5, 5), textcoords='offset points')

plt.tight_layout()
plt.savefig(f'{results_dir}/similarity_metrics_comparison.png', dpi=300)
print(f"\nCreated visualization: {results_dir}/similarity_metrics_comparison.png")

# ----- PROCEED WITH THE ORIGINAL ANALYSIS USING COSINE SIMILARITY -----
# Now we'll continue with the original analysis using cosine similarity

# Calculate similarity with all historical periods
similarity_scores = cosine_scores
periods = all_periods[1:]  # Skip the current regime

# Create a dataframe with the results
cosine_results_df = pd.DataFrame({
    'Period': [p[0] for p in periods],
    'End_Year': [p[1] for p in periods],
    'Similarity': similarity_scores
})

# Sort by similarity (highest first)
cosine_results_df = cosine_results_df.sort_values('Similarity', ascending=False)

# Export all similarity scores
cosine_results_df.to_csv(f'{results_dir}/all_regime_similarities.csv', index=False)

# ----- RESULTS -----
# First, display all-time most similar periods
print("\nTop 10 most similar historical regimes (all time) by Cosine Similarity:")
print(cosine_results_df.head(10).to_string(index=False))

# Export top 10 results
cosine_results_df.head(10).to_csv(f'{results_dir}/top10_similar_regimes.csv', index=False)

# Then filter to pre-2020 periods for historical comparison
pre_2020_df = cosine_results_df[cosine_results_df['End_Year'] < 2020].copy()
print("\nTop 15 most similar historical regimes (pre-2020) by Cosine Similarity:")
print(pre_2020_df.head(15).to_string(index=False))

# Export pre-2020 similar regimes
pre_2020_df.head(15).to_csv(f'{results_dir}/pre2020_similar_regimes.csv', index=False)

# ----- IDENTIFY DISTINCT REGIME CLUSTERS -----
# Extract month and year to identify specific periods
pre_2020_df['Start_Period'] = pre_2020_df['Period'].apply(lambda x: x.split(' to ')[0])
pre_2020_df['Month_Year'] = pre_2020_df['Start_Period'].apply(lambda x: pd.to_datetime(x).strftime('%b %Y'))

# Group by month-year to identify specific regime periods
period_avg_similarity = pre_2020_df.groupby('Month_Year')['Similarity'].mean().sort_values(ascending=False)

print("\nHistorical regime periods by average similarity:")
for period, sim in period_avg_similarity.head(10).items():
    print(f"{period}: {sim:.4f}")

# Export period averages
period_avg_df = pd.DataFrame(period_avg_similarity).reset_index()
period_avg_df.columns = ['Month_Year', 'Avg_Similarity']
period_avg_df.to_csv(f'{results_dir}/period_avg_similarity.csv', index=False)

# ----- ANALYSIS OF CURRENT ECONOMIC VARIABLES -----
print("\n----- Current Regime Economic Indicators -----")

# Calculate current values for key indicators
current_values = current_regime[selected_vars].mean()

# Calculate YoY changes
previous_year = data.iloc[-window-12:-12][selected_vars].mean()
yoy_changes = ((current_values - previous_year) / previous_year) * 100

# Create and export economic indicators dataframe
indicators_df = pd.DataFrame({
    'Variable': selected_vars,
    'Current_Value': [current_values[var] if var in current_values else np.nan for var in selected_vars],
    'YoY_Change_Pct': [yoy_changes[var] if var in yoy_changes else np.nan for var in selected_vars]
})
indicators_df = indicators_df.sort_values('YoY_Change_Pct', key=abs, ascending=False)
indicators_df.to_csv(f'{results_dir}/current_economic_indicators.csv', index=False)

# Display current values and YoY changes for key indicators
print("\nKey Indicators (Current Value | YoY Change):")
formatted_data = []
for var in selected_vars:
    if var in current_values and var in previous_year:
        current = current_values[var]
        yoy = yoy_changes[var]
        if not np.isnan(current) and not np.isnan(yoy):
            formatted_data.append((var, current, yoy))

# Sort by absolute YoY change to highlight biggest movers
formatted_data.sort(key=lambda x: abs(x[2]), reverse=True)
for var, current, yoy in formatted_data:
    print(f"{var}: {current:.2f} | {yoy:.2f}%")

# ----- VISUALIZE REGIME TRANSITIONS -----
# Create a time series of similarity scores
plt.figure(figsize=(14, 8))

# Prepare data for plotting
plot_df = cosine_results_df.copy()
plot_df['Start_Date'] = plot_df['Period'].apply(lambda x: pd.to_datetime(x.split(' to ')[0]))
plot_df = plot_df.sort_values('Start_Date')

# Export plot data
plot_df.to_csv(f'{results_dir}/similarity_over_time.csv', index=False)

# Plot similarity over time
plt.plot(plot_df['Start_Date'], plot_df['Similarity'], marker='o', alpha=0.7, markersize=3)
plt.title(f'Similarity to Current Regime ({current_start} to {current_end})', fontsize=16)
plt.xlabel('Historical Period Start', fontsize=14)
plt.ylabel('Similarity Score (cosine)', fontsize=14)
plt.grid(True, alpha=0.3)

# Mark economic events
events = [
    ('2008-09', 'Financial Crisis'),
    ('2011-08', 'Debt Ceiling Crisis'),
    ('2013-05', 'Taper Tantrum'),
    ('2016-06', 'Brexit Vote'),
    ('2018-12', 'Fed Pivot'),
    ('2020-03', 'COVID-19'),
    ('2022-03', 'Fed Tightening')
]

# Export economic events
pd.DataFrame(events, columns=['Date', 'Event']).to_csv(f'{results_dir}/economic_events.csv', index=False)

for date, label in events:
    event_date = pd.to_datetime(date)
    plt.axvline(x=event_date, color='red', linestyle='--', alpha=0.5)
    plt.text(event_date, 0.4, label, rotation=90, verticalalignment='bottom')

# Highlight the top 5 most similar pre-2020 periods
top5 = pre_2020_df.head(5).copy()
top5['Start_Date'] = top5['Period'].apply(lambda x: pd.to_datetime(x.split(' to ')[0]))

plt.scatter(top5['Start_Date'], top5['Similarity'], color='red', s=100, zorder=5)

for _, row in top5.iterrows():
    plt.annotate(row['Period'], 
                (row['Start_Date'], row['Similarity']),
                xytext=(10, 5), textcoords='offset points',
                fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{results_dir}/refined_regime_similarity_standardized.png', dpi=300)
print(f"\nCreated visualization: {results_dir}/refined_regime_similarity_standardized.png")

print(f"\nExported all refined analysis results to CSV files in the '{results_dir}' directory")
print("\nRefined analysis with standardization complete!")
