# =============================================================================
# step0_explore_data.py
#
# PURPOSE: Quick data exploration - look at what columns and values we have
#          in both CSV files before doing anything else.
#
# RUN: python step0_explore_data.py
# =============================================================================

import pandas as pd
import os

# ---------------------------------------------------------------------------
# Helper: print a clean divider
# ---------------------------------------------------------------------------
def section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# 1. FIRMS VIIRS data  (use the near-real-time file first; fall back to archive)
# ---------------------------------------------------------------------------
section("FIRMS VIIRS Data")

firms_nrt_path  = "data/fire_nrt_SV-C2_823956.csv"
firms_arch_path = "data/fire_archive_SV-C2_823956.csv"

# Pick whichever file exists; the archive is huge so we only sample it
if os.path.exists(firms_nrt_path):
    firms_path  = firms_nrt_path
    firms_label = "NRT (near-real-time)"
    firms_df    = pd.read_csv(firms_path)
elif os.path.exists(firms_arch_path):
    firms_path  = firms_arch_path
    firms_label = "Archive (sampling 50 000 rows)"
    firms_df    = pd.read_csv(firms_path, nrows=50_000)
else:
    raise FileNotFoundError(
        "No FIRMS CSV found. Expected one of:\n"
        f"  {firms_nrt_path}\n  {firms_arch_path}"
    )

print(f"\nLoaded: {firms_label}")
print(f"Shape : {firms_df.shape[0]:,} rows x {firms_df.shape[1]} columns")
print(f"\nColumns:\n  {list(firms_df.columns)}")

print("\nFirst 3 rows:")
print(firms_df.head(3).to_string())

print("\nData types:")
print(firms_df.dtypes.to_string())

print("\nBasic statistics for key numeric columns:")
key_cols = [c for c in ["bright_ti4", "bright_ti5", "frp", "scan", "track"]
            if c in firms_df.columns]
print(firms_df[key_cols].describe().round(2).to_string())

# Show unique values for categorical columns
for col in ["confidence", "daynight", "satellite", "type"]:
    if col in firms_df.columns:
        vals = firms_df[col].value_counts().head(10)
        print(f"\n'{col}' value counts:\n{vals.to_string()}")

# Date range
if "acq_date" in firms_df.columns:
    firms_df["acq_date"] = pd.to_datetime(firms_df["acq_date"])
    print(f"\nDate range: {firms_df['acq_date'].min()} → {firms_df['acq_date'].max()}")

# Geographic bounding box
print(f"\nGeographic extent:")
print(f"  Latitude : {firms_df['latitude'].min():.3f} → {firms_df['latitude'].max():.3f}")
print(f"  Longitude: {firms_df['longitude'].min():.3f} → {firms_df['longitude'].max():.3f}")


# ---------------------------------------------------------------------------
# 2. Global Power Plant Database
# ---------------------------------------------------------------------------
section("Global Power Plant Database")

pp_path = "data/global_power_plant_database.csv"

if not os.path.exists(pp_path):
    raise FileNotFoundError(f"Power plant CSV not found: {pp_path}")

pp_df = pd.read_csv(pp_path, low_memory=False)

print(f"\nShape : {pp_df.shape[0]:,} rows x {pp_df.shape[1]} columns")
print(f"\nColumns:\n  {list(pp_df.columns)}")

print("\nFirst 3 rows:")
print(pp_df.head(3).to_string())

# Fuel types
fuel_col = next((c for c in pp_df.columns if "fuel" in c.lower()), None)
if fuel_col:
    print(f"\nFuel types (top 15):")
    print(pp_df[fuel_col].value_counts().head(15).to_string())

# Countries
country_col = next((c for c in pp_df.columns
                    if c.lower() in ("country", "country_long", "country_code")), None)
if country_col:
    print(f"\nCountries (top 10):")
    print(pp_df[country_col].value_counts().head(10).to_string())

# Capacity
cap_col = next((c for c in pp_df.columns if "capacity" in c.lower()), None)
if cap_col:
    pp_df[cap_col] = pd.to_numeric(pp_df[cap_col], errors="coerce")
    print(f"\nCapacity (MW) statistics:")
    print(pp_df[cap_col].describe().round(1).to_string())

# Lat/lon coverage
lat_col = next((c for c in pp_df.columns if "latitude"  in c.lower()), None)
lon_col = next((c for c in pp_df.columns if "longitude" in c.lower()), None)
if lat_col and lon_col:
    print(f"\nGeographic extent:")
    print(f"  Latitude : {pp_df[lat_col].min():.3f} → {pp_df[lat_col].max():.3f}")
    print(f"  Longitude: {pp_df[lon_col].min():.3f} → {pp_df[lon_col].max():.3f}")
    print(f"  Missing lat/lon: {pp_df[[lat_col, lon_col]].isna().any(axis=1).sum():,}")


# ---------------------------------------------------------------------------
# 3. Quick overlap check
# ---------------------------------------------------------------------------
section("Overlap Check: FIRMS bbox vs Power Plants")

if lat_col and lon_col:
    firms_lat_min = firms_df["latitude"].min()
    firms_lat_max = firms_df["latitude"].max()
    firms_lon_min = firms_df["longitude"].min()
    firms_lon_max = firms_df["longitude"].max()

    # Power plants inside the FIRMS bounding box
    in_box = pp_df[
        (pp_df[lat_col] >= firms_lat_min) & (pp_df[lat_col] <= firms_lat_max) &
        (pp_df[lon_col] >= firms_lon_min) & (pp_df[lon_col] <= firms_lon_max)
    ]
    print(f"\nPower plants inside FIRMS bounding box: {len(in_box):,} / {len(pp_df):,}")


print("\n\nDone. Explore the data above before running step1.")