import logging
import streamlit as st
import pandas as pd
from pathlib import Path
import difflib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).parent

# Configure logger to print to stdout so messages appear in the terminal running Streamlit
logger = logging.getLogger("tripguard")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def list_csvs(root: Path):
    files = sorted([p.name for p in root.glob("*.csv")])
    logger.info(f"Found CSV files: {files}")
    return files


def find_file(root: Path, filename: str) -> Path | None:
    """Look for filename in project root and in a `Datasets/` subfolder. Return Path or None."""
    candidates = [root / filename, root / 'Datasets' / filename]
    for p in candidates:
        if p.exists():
            return p
    # also try case-insensitive search for filename within root and Datasets
    for folder in [root, root / 'Datasets']:
        if folder.exists():
            for p in folder.glob('*.csv'):
                if p.name.lower() == filename.lower():
                    return p
    return None


def load_csv(path: Path) -> pd.DataFrame:
    logger.info(f"Loading CSV: {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {path.name} shape={df.shape}")
    return df


def main():
    st.set_page_config(page_title="TripGuard-AI — Data UI", layout="wide")
    st.title("TripGuard-AI — Simple Data UI")

    st.sidebar.header("Data selector")
    csv_files = list_csvs(ROOT)
    if not csv_files:
        st.sidebar.warning("No CSV files found in project root.")
        st.info("Place your CSV files (e.g., crime_merged.csv) in the project root.")
        return

    selected = st.sidebar.selectbox("Choose a CSV file", csv_files)
    keep_cols = st.sidebar.number_input("Number of columns to keep (from left)", min_value=1, max_value=100, value=8)
    show_rows = st.sidebar.number_input("Rows to preview", min_value=1, max_value=1000, value=5)

    file_path = ROOT / selected
    try:
        df = load_csv(file_path)
    except Exception as e:
        st.error(f"Failed to load {selected}: {e}")
        logger.exception(f"Failed to load {selected}")
        return

    st.subheader(f"Preview — {selected}")
    st.write(f"Shape: {df.shape}")
    try:
        st.dataframe(df.head(show_rows))
    except Exception:
        logger.exception("st.dataframe failed for preview; falling back to string display")
        st.write(df.head(show_rows).astype(str))

    st.markdown("---")
    st.subheader("Filter columns (left-most)")
    st.write(f"Keeping first {keep_cols} columns")

    if st.button("Filter & Save as crime_filtered.csv"):
        try:
            filtered = df.iloc[:, :keep_cols]
            out_path = ROOT / "crime_filtered.csv"
            filtered.to_csv(out_path, index=False)
            st.success(f"Saved filtered CSV to {out_path.name}")
            logger.info(f"Saved filtered CSV to {out_path}")
            st.download_button("Download filtered CSV", data=out_path.read_bytes(), file_name=out_path.name)
        except Exception as e:
            st.error(f"Failed to filter/save: {e}")
            logger.exception("Failed to filter/save CSV")

    st.markdown("---")
    st.subheader("Quick stats")
    try:
        st.write(df.describe(include='all'))
    except Exception:
        logger.exception("st.write failed for describe(); falling back to string display")
        st.write(df.describe(include='all').astype(str))

    st.markdown("---")
    st.subheader("Lookup by place")
    place = st.text_input("Enter place/region name (e.g. 'KARNATAKA')")
    if st.button("Get place data"):
        try:
            with st.spinner("Building merged dataset and computing scores..."):
                master_df = build_master_df(ROOT)
                model, features = train_risk_model(master_df)

            if master_df is None or master_df.empty:
                st.warning("Master dataset is empty or couldn't be built.")
            else:
                norm = (place or "").upper().strip()
                matches = master_df[master_df['region'].str.upper().str.strip() == norm]
                if matches.empty:
                    # fuzzy suggestions
                    choices = master_df['region'].unique().tolist()
                    sugg = difflib.get_close_matches(place, choices, n=5, cutoff=0.5)
                    if sugg:
                        st.info(f"No exact match for '{place}'. Did you mean: {', '.join(sugg)} ?")
                    else:
                        st.error(f"No match found for '{place}'.")
                else:
                    row = matches.iloc[0]
                    try:
                        st.write(row.to_frame().T)
                    except Exception:
                        logger.exception("st.write failed for matched row; falling back to string display")
                        st.write(row.to_frame().T.astype(str))
                    # model prediction for this row
                    X_row = row[features].to_frame().T
                    pred = model.predict(X_row)[0]
                    st.metric(label="Predicted risk score", value=f"{pred:.2f}")
                    st.write("Model features used:")
                    st.write(X_row)

        except Exception as e:
            st.error(f"Failed to get place data: {e}")
            logger.exception("Failed to get place data")


def build_master_df(root: Path) -> pd.DataFrame:
    """Recreate the master_df using the same preprocessing as ML_Project.ipynb.
    Returns a dataframe with computed scores and risk_score.
    """
    try:
        # --- Air ---
        air_cols_to_read = ['region', 'air_quality_PM2.5', 'air_quality_PM10']
        air_path = find_file(root, 'Air_Quality_Index.csv')
        if air_path is None:
            logger.error('Air_Quality_Index.csv not found in project root or Datasets/.')
            raise FileNotFoundError('Air_Quality_Index.csv not found')
        df_air_raw = pd.read_csv(air_path, usecols=[c for c in air_cols_to_read if c in pd.read_csv(air_path, nrows=0).columns])
        df_air_raw['region'] = df_air_raw['region'].str.upper().str.strip()
        name_corrections = {
            'ANDAMAN AND NICOBAR ISLANDS': 'A & N ISLANDS',
            'DADRA AND NAGAR HAVELI': 'D & N HAVELI',
            'DAMAN AND DIU': 'DAMAN & DIU'
        }
        df_air_raw['region'] = df_air_raw['region'].replace(name_corrections)
        df_air = df_air_raw.groupby('region').mean().reset_index()
        df_air['region'] = df_air['region'].str.upper()

        # --- Road ---
        road_path = find_file(root, 'Road_Condition_Compressed.csv')
        if road_path is None:
            logger.error('Road_Condition_Compressed.csv not found in project root or Datasets/.')
            raise FileNotFoundError('Road_Condition_Compressed.csv not found')
        csv_road = pd.read_csv(road_path)
        # rename as in notebook
        if 'State/ UT' in csv_road.columns:
            csv_road.rename(columns={'State/ UT': 'region'}, inplace=True)
        # try to compute total_accidents using known columns (best-effort)
        acc_cols = [c for c in csv_road.columns if 'Accident' in c and '2014' in c]
        if acc_cols and len(acc_cols) >= 2:
            csv_road['total_accidents'] = csv_road[acc_cols].sum(axis=1)
        else:
            # fallback: numeric sum of numeric columns
            numcols = csv_road.select_dtypes(include='number').columns
            csv_road['total_accidents'] = csv_road[numcols].sum(axis=1)
        df_road = csv_road[['region', 'total_accidents']].copy()
        df_road['region'] = df_road['region'].str.upper().str.strip()

        # --- Crime ---
        crime_path = find_file(root, 'crime_filtered.csv')
        if crime_path is None:
            logger.error('crime_filtered.csv not found in project root or Datasets/.')
            raise FileNotFoundError('crime_filtered.csv not found')
        csv_crime = pd.read_csv(crime_path)
        csv_crime['state_ut'] = csv_crime['state_ut'].str.upper()
        name_corrections = {
            'UTTARAKHAND': 'UTTARAKHAND',
            'UTTARPRADESH': 'UTTAR PRADESH'
        }
        csv_crime['state_ut'] = csv_crime['state_ut'].replace(name_corrections)
        crime_cols = [
            '01_District_wise_crimes_committed_IPC_2001_2012_total_ipc_crimes',
            '01_District_wise_crimes_committed_IPC_2013_total_ipc_crimes'
        ]
        for c in crime_cols:
            if c not in csv_crime.columns:
                csv_crime[c] = 0
        csv_crime[crime_cols] = csv_crime[crime_cols].fillna(0)
        df_maxes = csv_crime.groupby('state_ut')[crime_cols].max().reset_index()
        df_maxes['total_crimes'] = df_maxes[crime_cols[0]] + df_maxes[crime_cols[1]]
        df_crime_agg = df_maxes[['state_ut', 'total_crimes']].copy()
        df_crime_agg.rename(columns={'state_ut': 'region'}, inplace=True)
        df_crime_agg['region'] = df_crime_agg['region'].str.upper().str.strip()

        # --- Merge ---
        master_df = pd.merge(df_air, df_road, on='region')
        master_df = pd.merge(master_df, df_crime_agg, on='region')

        # --- Optional: Safety dataset ---
        # detect any CSV with 'safety' in the filename (case-insensitive)
        safety_files = [p for p in root.glob('*.csv') if 'safety' in p.name.lower()]
        if safety_files:
            # pick the first match
            sf = safety_files[0]
            try:
                df_safety = pd.read_csv(sf)
                # try to find a region column and a numeric safety column
                col_candidates = [c for c in df_safety.columns if 'region' in c.lower()]
                if col_candidates:
                    region_col = col_candidates[0]
                    df_safety.rename(columns={region_col: 'region'}, inplace=True)
                    df_safety['region'] = df_safety['region'].str.upper().str.strip()
                    # pick first numeric column as safety index (excluding region)
                    numeric_cols = df_safety.select_dtypes(include='number').columns.tolist()
                    if numeric_cols:
                        safety_col = numeric_cols[0]
                        df_safety = df_safety[['region', safety_col]].rename(columns={safety_col: 'safety_index'})
                        master_df = pd.merge(master_df, df_safety, on='region', how='left')
                        # fill missing safety_index with mean
                        master_df['safety_index'] = master_df['safety_index'].fillna(master_df['safety_index'].mean())
                        logger.info(f"Merged safety dataset {sf.name} using column {safety_col}")
                    else:
                        logger.warning(f"Found safety file {sf.name} but no numeric columns to use as safety index")
                else:
                    logger.warning(f"Found safety file {sf.name} but no region-like column")
            except Exception:
                logger.exception(f"Failed to read/merge safety file {sf}")

        # --- Scores ---
        weights = {
            'air_quality': 0.4,
            'total_accidents': 0.3,
            'total_crimes': 0.3
        }
        master_df['N_PM2_5'] = (master_df['air_quality_PM2.5'] / 30).clip(upper=1)
        master_df['N_PM10'] = (master_df['air_quality_PM10'] / 50).clip(upper=1)
        master_df['PollutionIndex'] = 0.6 * master_df['N_PM2_5'] + 0.4 * master_df['N_PM10']
        master_df['air_safety_score'] = (1 - master_df['PollutionIndex']) * 100
        total_accidents_sum = master_df['total_accidents'].sum()
        master_df['road_score'] = (1 - (master_df['total_accidents'] / total_accidents_sum)) * 100
        total_crimes_sum = master_df['total_crimes'].sum()
        master_df['crime_score'] = (1 - (master_df['total_crimes'] / total_crimes_sum)) * 100
        master_df['risk_score'] = (
            (master_df['air_safety_score'] * weights['air_quality']) +
            (master_df['road_score'] * weights['total_accidents']) +
            (master_df['crime_score'] * weights['total_crimes'])
        )

        return master_df
    except Exception:
        logger.exception("Error building master_df")
        return pd.DataFrame()


def train_risk_model(master_df):
    """Train a simple LinearRegression on the merged master_df and return (model, features)"""
    features = ['air_quality_PM2.5', 'air_quality_PM10', 'total_accidents', 'total_crimes']
    target = 'risk_score'
    X = master_df[features]
    y = master_df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    # include safety_index if present
    if 'safety_index' in master_df.columns and 'safety_index' not in features:
        features = features + ['safety_index']
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, features


if __name__ == "__main__":
    main()

from pathlib import Path
import pandas as pd

root = Path(".")
files = {
    "Air": root / "Datasets" / "Air_Quality_Index.csv",
    "Road": root / "Datasets" / "Road_Condition_Compressed.csv",
    "Crime": root / "crime_filtered.csv",
}
for name, path in files.items():
    print(f"--- {name} ({path}) ---")
    if not path.exists():
        print("MISSING FILE")
    else:
        try:
            df = pd.read_csv(path, nrows=0)
            print(list(df.columns))
        except Exception as e:
            print("Failed to read:", e)
    print()

import pandas as pd
from pathlib import Path
p = Path('Datasets') / 'safety_index_by_region.csv'
df = pd.read_csv(p)
print('Regions:', df['region'].unique()[:50])   # print first 50 regions
print('Columns:', list(df.columns))
