import logging
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except Exception:
    STREAMLIT_AVAILABLE = False
    st = None
import pandas as pd
from pathlib import Path
import difflib
from weather import get_5day_forecast
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
    if not STREAMLIT_AVAILABLE:
        logger.info("Streamlit not available; launching CLI fallback")
        run_cli_fallback()
        return

    st.set_page_config(page_title="TripGuard-AI — Data UI", layout="wide")
    st.title("TripGuard-AI Main page")

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

                    # Add weather forecast section
                    st.markdown("---")
                    st.subheader("5-Day Weather Forecast")
                    forecasts = get_5day_forecast(place)
                    
                    if len(forecasts) == 1 and "error" in forecasts[0]:
                        st.warning(forecasts[0]["error"])
                    else:
                        cols = st.columns(5)
                        for idx, forecast in enumerate(forecasts):
                            with cols[idx]:
                                st.markdown(f"**{forecast['date']}**")
                                st.markdown(f"🌡️ {forecast['temp_min']}°C - {forecast['temp_max']}°C")
                                st.markdown(f"💧 {forecast['humidity']}% humidity")
                                st.markdown(f"🌤️ {forecast['description']}")
                                st.markdown(f"💨 {forecast['wind_speed']} km/h")

        except Exception as e:
            st.error(f"Failed to get place data: {e}")
            logger.exception("Failed to get place data")

def run_cli_fallback():
    """A minimal command-line fallback when Streamlit isn't installed.
    Provides basic preview, filtering and the 'place' lookup using the same functions.
    """
    print("Streamlit is not installed or couldn't be imported. Running CLI fallback.")
    csv_files = list_csvs(ROOT)
    if not csv_files:
        print("No CSV files found in project root. Place CSVs (e.g., crime_merged.csv) in the project root.")
        return

    for i, name in enumerate(csv_files, 1):
        print(f"{i}. {name}")
    sel = input(f"Select file by number or name (1-{len(csv_files)}): ").strip()
    try:
        idx = int(sel) - 1
        selected = csv_files[idx]
    except Exception:
        if sel in csv_files:
            selected = sel
        else:
            print("Invalid selection")
            return

    file_path = ROOT / selected
    try:
        df = load_csv(file_path)
    except Exception as e:
        print(f"Failed to load {selected}: {e}")
        return

    print(f"Preview of {selected} (first 10 rows):")
    print(df.head(10).to_string())

    while True:
        cmd = input("Options: [f]ilter & save, [s]tats, [p]lace lookup, [q]uit: ").strip().lower()
        if cmd in ('q', 'quit'):
            break
        if cmd.startswith('f'):
            keep = input("Number of left columns to keep (default 8): ").strip()
            keep_cols = int(keep) if keep else 8
            filtered = df.iloc[:, :keep_cols]
            out_path = ROOT / "crime_filtered.csv"
            filtered.to_csv(out_path, index=False)
            print(f"Saved filtered CSV to {out_path}")
        elif cmd.startswith('s'):
            print(df.describe(include='all'))
        elif cmd.startswith('p'):
            place = input("Place/region name: ").strip()
            master_df = build_master_df(ROOT)
            if master_df is None or master_df.empty:
                print("Master dataset is empty or couldn't be built.")
                continue
            norm = place.upper().strip()
            matches = master_df[master_df['region'].str.upper().str.strip() == norm]
            if matches.empty:
                choices = master_df['region'].unique().tolist()
                sugg = difflib.get_close_matches(place, choices, n=5, cutoff=0.5)
                if sugg:
                    print(f"No exact match for '{place}'. Did you mean: {', '.join(sugg)} ?")
                else:
                    print(f"No match found for '{place}'.")
            else:
                row = matches.iloc[0]
                print(row.to_frame().T.to_string())
                model, features = train_risk_model(master_df)
                X_row = row[features].to_frame().T
                pred = model.predict(X_row)[0]
                print(f"Predicted risk score: {pred:.2f}")
        else:
            print("Unknown option")


def build_master_df(root: Path) -> pd.DataFrame:
    """Recreate the master_df using the same preprocessing as ML_Project.ipynb.
    Returns a dataframe with computed scores and risk_score.
    """
    try:
        # --- Air ---
        air_path = find_file(root, 'Air_Quality_Index.csv')
        if air_path is None:
            logger.error('Air_Quality_Index.csv not found in project root or Datasets/.')
            raise FileNotFoundError('Air_Quality_Index.csv not found')
        
        # Read air quality data and calculate regional averages
        df_air = pd.read_csv(air_path)
        df_air['region'] = df_air['region'].str.upper().str.strip()
        df_air = df_air.groupby('region')[['air_quality_PM2.5', 'air_quality_PM10']].mean().reset_index()
        
        # Normalize air quality metrics (higher is better)
        df_air['air_quality_PM2.5'] = 1 - (df_air['air_quality_PM2.5'] / df_air['air_quality_PM2.5'].max())
        df_air['air_quality_PM10'] = 1 - (df_air['air_quality_PM10'] / df_air['air_quality_PM10'].max())

        # --- Road ---
        road_path = find_file(root, 'Road_Condition_Compressed.csv')
        if road_path is None:
            logger.error('Road_Condition_Compressed.csv not found in project root or Datasets/.')
            raise FileNotFoundError('Road_Condition_Compressed.csv not found')
            
        # Read road condition data
        df_road = pd.read_csv(road_path)
        df_road.rename(columns={'State/ UT': 'region'}, inplace=True)
        
        # Calculate total accidents from accident columns
        accident_cols = [col for col in df_road.columns if 'Accident' in col]
        df_road['total_accidents'] = df_road[accident_cols].sum(axis=1)
        
        # Normalize accident score (higher is better = fewer accidents)
        df_road['total_accidents'] = 1 - (df_road['total_accidents'] / df_road['total_accidents'].max())
        df_road = df_road[['region', 'total_accidents']].copy()
        df_road['region'] = df_road['region'].str.upper().str.strip()

        # --- Crime ---
        crime_path = find_file(root, 'crime_filtered.csv')
        if crime_path is None:
            logger.error('crime_filtered.csv not found in project root or Datasets/.')
            raise FileNotFoundError('crime_filtered.csv not found')
            
        # Read crime data and map cities to states
        df_crime = pd.read_csv(crime_path)
        city_to_state = {
            'CHENNAI': 'TAMIL NADU',
            'MUMBAI': 'MAHARASHTRA',
            'DELHI': 'DELHI',
            'BANGALORE': 'KARNATAKA',
            'KOLKATA': 'WEST BENGAL',
            'HYDERABAD': 'TELANGANA',
            'AHMEDABAD': 'GUJARAT',
            'PUNE': 'MAHARASHTRA',
            'SURAT': 'GUJARAT',
            'JAIPUR': 'RAJASTHAN',
            'LUCKNOW': 'UTTAR PRADESH',
            'KANPUR': 'UTTAR PRADESH',
            'NAGPUR': 'MAHARASHTRA',
            'INDORE': 'MADHYA PRADESH',
            'THANE': 'MAHARASHTRA',
            'BHOPAL': 'MADHYA PRADESH',
            'PATNA': 'BIHAR',
            'VADODARA': 'GUJARAT',
            'LUDHIANA': 'PUNJAB',
            'AGRA': 'UTTAR PRADESH'
        }
        
        df_crime['region'] = df_crime['City'].str.upper().map(city_to_state)
        df_crime = df_crime[df_crime['region'].notna()]  # Remove unmapped cities
        
        # Count crimes by region
        crime_counts = df_crime.groupby('region').size().reset_index(name='total_crimes')
        
        # Normalize crime score (higher is better = fewer crimes)
        crime_counts['total_crimes'] = 1 - (crime_counts['total_crimes'] / crime_counts['total_crimes'].max())
        df_crime_agg = crime_counts.copy()

        # --- Merge ---
        master_df = pd.merge(df_air, df_road, on='region', how='outer')
        master_df = pd.merge(master_df, df_crime_agg, on='region', how='outer')
        
        # Fill missing values with means
        for col in ['air_quality_PM2.5', 'air_quality_PM10', 'total_accidents', 'total_crimes']:
            if col in master_df.columns:
                master_df[col] = master_df[col].fillna(master_df[col].mean())
        
        # --- Compute Risk Score ---
        weights = {
            'air_quality': 0.4,
            'accidents': 0.3,
            'crimes': 0.3
        }
        
        # Air quality score (already normalized, higher is better)
        master_df['air_score'] = (
            master_df['air_quality_PM2.5'] * 0.6 + 
            master_df['air_quality_PM10'] * 0.4
        ) * 100
        
        # Road safety score (already normalized, higher is better)
        master_df['road_score'] = master_df['total_accidents'] * 100
        
        # Crime safety score (already normalized, higher is better)
        master_df['crime_score'] = master_df['total_crimes'] * 100
        
        # Final risk score (0-100, higher is better)
        master_df['risk_score'] = (
            master_df['air_score'] * weights['air_quality'] +
            master_df['road_score'] * weights['accidents'] +
            master_df['crime_score'] * weights['crimes']
        )

        return master_df
    except Exception:
        logger.exception("Error building master_df")
        return pd.DataFrame()


def train_risk_model(master_df):
    """Train a simple LinearRegression on the merged master_df and return (model, features)"""
    features = ['air_quality_PM2.5', 'air_quality_PM10', 'total_accidents', 'total_crimes']
    # include safety_index if present
    if 'safety_index' in master_df.columns:
        features = features + ['safety_index']
    
    target = 'risk_score'
    X = master_df[features]
    y = master_df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
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
