#changed dataset
import pandas as pd

# Load your dataset
data = pd.read_csv("crime_merged.csv")

# Keep columns only till 'H' (i.e., first 8 columns)
data = data.iloc[:, :8]

# Save or display the result
data.to_csv("crime_filtered.csv", index=False)
print(data.head())
