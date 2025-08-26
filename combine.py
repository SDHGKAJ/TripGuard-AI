import pandas as pd
import glob
import os
from functools import reduce

# 1. Merge Crime datasets
crime_files = glob.glob("datasets/crime/*.csv")  # path to your 19 crime csvs
crime_dfs = []

for file in crime_files:
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()
    
    # Identify state/UT column and standardize its name
    if "area_name" in df.columns:
        df = df.rename(columns={"area_name": "state_ut"})
    elif "state/ut" in df.columns:
        df = df.rename(columns={"state/ut": "state_ut"})
    else:
        df = df.rename(columns={df.columns[0]: "state_ut"})
    
    # Check for 'year' column
    if "year" not in df.columns:
        print(f"⚠️  Skipping {file}: 'year' column not found.")
        continue
    
    # Keep state, year and totals
    total_cols = [col for col in df.columns if "total" in col.lower()]
    if not total_cols:
        print(f"⚠️  Skipping {file}: No 'total' columns found.")
        continue

    subset = df[["state_ut", "year"] + total_cols]
    
    # Rename totals uniquely (add filename prefix)
    prefix = os.path.basename(file).split(".")[0]
    subset = subset.rename(columns={col: f"{prefix}_{col}" for col in total_cols})
    
    crime_dfs.append(subset)

if not crime_dfs:
    print("❌ No valid crime datasets found to merge.")
else:
    # Merge all crime datasets together on standardized keys
    crime_master = reduce(lambda left, right: pd.merge(left, right, on=["state_ut", "year"], how="outer"), crime_dfs)

    # Save merged dataset
    crime_master.to_csv("crime_merged.csv", index=False)
    print("✅ Crime datasets merged into: crime_merged.csv")