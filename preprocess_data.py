import pandas as pd

# Read the crime dataset
df = pd.read_csv('crime_dataset_india.csv')

# Convert date columns to datetime
date_columns = ['Date Reported', 'Date of Occurrence', 'Date Case Closed']
for col in date_columns:
    if col in df.columns:
        # Handle the specific date format
        df[col] = pd.to_datetime(df[col], format='%d-%m-%Y %H:%M', errors='coerce')

# Group by city and calculate statistics
city_stats = df.groupby('City').agg({
    'Report Number': 'count',
    'Crime Description': lambda x: x.value_counts().index[0],  # Most common crime
    'Crime Domain': lambda x: x.value_counts().index[0],  # Most common domain
}).reset_index()

# Rename columns
city_stats.columns = ['city', 'total_crimes', 'most_common_crime', 'most_common_domain']

# Calculate crime severity score (example scoring)
severity_weights = {
    'Violent Crime': 1.0,
    'Other Crime': 0.6,
    'Fire Accident': 0.7,
    'Traffic Fatality': 0.8
}

# Add severity score
city_stats['severity_score'] = city_stats['most_common_domain'].map(severity_weights)
city_stats['crime_risk_score'] = (city_stats['total_crimes'] * city_stats['severity_score'])

# Normalize the risk score
city_stats['crime_risk_score'] = (city_stats['crime_risk_score'] - city_stats['crime_risk_score'].min()) / \
                                (city_stats['crime_risk_score'].max() - city_stats['crime_risk_score'].min()) * 100

# Save the processed data
city_stats.to_csv('crime_stats_by_city.csv', index=False)
print("Data preprocessing complete. Check crime_stats_by_city.csv for results.")