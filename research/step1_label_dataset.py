# =============================================================================
# step1_label_dataset.py
#
# PURPOSE: Assign a label to every FIRMS detection:
#   0 = wildfire  (far from any known industrial heat source)
#   1 = industrial (close to a power plant or other facility)
#
# APPROACH:
#   - Load FIRMS detections and the Global Power Plant Database.
#   - For each FIRMS point, compute the haversine distance to every power plant.
#   - If the closest plant is within PROXIMITY_KM → label 1 (industrial).
#   - Otherwise                                   → label 0 (wildfire).
#
# The threshold is intentionally conservative: real fires can happen near
# plants, and industrial flares can look fire-like. A larger threshold means
# more industrial labels but also more false positives.
#
# OUTPUT: data/firms_labelled.csv
#
# RUN: python step1_label_dataset.py
# =============================================================================

import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration - adjust these if needed
# ---------------------------------------------------------------------------
PROXIMITY_KM = 5.0          # km: FIRMS point closer than this → industrial
MAX_FIRMS_ROWS = 200_000    # cap to keep processing time reasonable
RANDOM_SEED = 42

# Input paths
FIRMS_NRT_PATH  = "data/fire_nrt_SV-C2_823956.csv"
FIRMS_ARCH_PATH = "data/fire_archive_SV-C2_823956.csv"
PLANTS_PATH     = "data/global_power_plant_database.csv"

# Output path
OUTPUT_PATH = "data/firms_labelled.csv"


# ---------------------------------------------------------------------------
# 1. Load FIRMS data
# ---------------------------------------------------------------------------
print("Loading FIRMS data …")

if os.path.exists(FIRMS_NRT_PATH):
    firms = pd.read_csv(FIRMS_NRT_PATH)
    print(f"  Loaded NRT file: {len(firms):,} rows")
elif os.path.exists(FIRMS_ARCH_PATH):
    firms = pd.read_csv(FIRMS_ARCH_PATH, nrows=MAX_FIRMS_ROWS)
    print(f"  Loaded archive file (first {MAX_FIRMS_ROWS:,} rows): {len(firms):,} rows")
else:
    raise FileNotFoundError(
        "FIRMS CSV not found. Place one of the following in the data/ folder:\n"
        f"  {FIRMS_NRT_PATH}\n  {FIRMS_ARCH_PATH}"
    )

# Labeling uses location only; FIRMS brightness column names vary by product.
firms = firms.dropna(subset=["latitude", "longitude"])
firms = firms.reset_index(drop=True)
print(f"  After dropping NaN rows: {len(firms):,}")


# ---------------------------------------------------------------------------
# 2. Load Power Plant data
# ---------------------------------------------------------------------------
print("\nLoading power plant data …")

if not os.path.exists(PLANTS_PATH):
    raise FileNotFoundError(f"Power plant CSV not found: {PLANTS_PATH}")

plants_raw = pd.read_csv(PLANTS_PATH, low_memory=False)
print(f"  Loaded: {len(plants_raw):,} plants")

# Find the latitude / longitude column names (they differ by version)
lat_col = next((c for c in plants_raw.columns if "latitude"  in c.lower()), None)
lon_col = next((c for c in plants_raw.columns if "longitude" in c.lower()), None)

if lat_col is None or lon_col is None:
    raise ValueError("Could not find lat/lon columns in power plant CSV. "
                     f"Columns available: {list(plants_raw.columns)}")

plants = plants_raw[[lat_col, lon_col]].copy()
plants.columns = ["lat", "lon"]
plants = plants.dropna()
plants["lat"] = pd.to_numeric(plants["lat"], errors="coerce")
plants["lon"] = pd.to_numeric(plants["lon"], errors="coerce")
plants = plants.dropna()
print(f"  After dropping NaN lat/lon: {len(plants):,} plants")


# ---------------------------------------------------------------------------
# 3. Haversine distance function
# ---------------------------------------------------------------------------
def haversine_min_dist_km(point_lat, point_lon, plant_lats, plant_lons):
    """
    Return the minimum haversine distance (km) from one point
    to ANY plant in the arrays plant_lats / plant_lons.

    Uses vectorised numpy so it is fast even for 30 000 plants.
    """
    R = 6371.0  # Earth radius in km

    # Convert degrees → radians
    lat1 = np.radians(point_lat)
    lat2 = np.radians(plant_lats)
    lon1 = np.radians(point_lon)
    lon2 = np.radians(plant_lons)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

    distances = R * c
    return distances.min()


# ---------------------------------------------------------------------------
# 4. Label each FIRMS detection
# ---------------------------------------------------------------------------
print(f"\nLabelling {len(firms):,} FIRMS detections "
      f"(proximity threshold = {PROXIMITY_KM} km) …")
print("  This may take a few minutes for large files …")

plant_lats = plants["lat"].values
plant_lons = plants["lon"].values

labels = np.zeros(len(firms), dtype=int)      # default: wildfire (0)
min_dists = np.zeros(len(firms), dtype=float)  # store distance for reference

for i, (_, row) in enumerate(firms.iterrows()):
    d = haversine_min_dist_km(row["latitude"], row["longitude"],
                               plant_lats, plant_lons)
    min_dists[i] = d
    if d <= PROXIMITY_KM:
        labels[i] = 1  # industrial

    if (i + 1) % 10_000 == 0:
        pct = 100 * (i + 1) / len(firms)
        ind_so_far = labels[:i+1].sum()
        print(f"  {i+1:>7,} / {len(firms):,}  ({pct:.0f}%)  "
              f"industrial so far: {ind_so_far:,}")

firms["label"] = labels
firms["min_dist_to_plant_km"] = np.round(min_dists, 3)


# ---------------------------------------------------------------------------
# 5. Summary
# ---------------------------------------------------------------------------
n_wild  = (firms["label"] == 0).sum()
n_ind   = (firms["label"] == 1).sum()
total   = len(firms)

print(f"\nLabelling complete:")
print(f"  Wildfire   (0): {n_wild:>7,}  ({100*n_wild/total:.1f}%)")
print(f"  Industrial (1): {n_ind:>7,}  ({100*n_ind/total:.1f}%)")
print(f"  Total         : {total:>7,}")

if n_ind == 0:
    print("\n  WARNING: 0 industrial detections found. "
          "Try increasing PROXIMITY_KM (currently "
          f"{PROXIMITY_KM} km) or check that power plant lat/lon match "
          "the FIRMS geographic region.")

if n_ind < 100:
    print(f"\n  NOTE: Only {n_ind} industrial detections. "
          "The dataset is highly imbalanced — models may need "
          "class_weight='balanced' or oversampling.")


# ---------------------------------------------------------------------------
# 6. Save
# ---------------------------------------------------------------------------
os.makedirs("data", exist_ok=True)
firms.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved labelled dataset to: {OUTPUT_PATH}")
print(f"Columns: {list(firms.columns)}")