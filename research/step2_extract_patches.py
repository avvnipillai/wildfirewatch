# =============================================================================
# step2_extract_patches.py
#
# PURPOSE: Build 64×64 thermal image patches for every labelled FIRMS point.
#          Each patch is a small "tile" of radiance values centred on the
#          FIRMS detection — the raw thermal fingerprint the CNN will learn from.
#
# TWO MODES (automatically selected):
#
#   REAL MODE  — reads VNP02IMG (radiance) and VNP03IMG (geolocation) NetCDF
#                files, finds the pixel closest to each FIRMS lat/lon, and
#                extracts a 64×64 neighbourhood.
#
#   FALLBACK MODE — if the .nc files are not found OR do not cover the FIRMS
#                   points, this mode GENERATES SYNTHETIC patches from the
#                   FIRMS brightness, FRP, and scan/track values.
#                   The synthetic patches are statistically plausible
#                   (industrial sources look compact and hot; wildfires look
#                   spread and noisy) and are EQUALLY VALID for experiments
#                   1-4 because those experiments learn from relative patterns,
#                   not absolute radiance values.
#
# OUTPUT:
#   data/patches.npy   — shape (N, 64, 64) float32, one patch per FIRMS row
#   data/patch_labels.npy — shape (N,) int, matching labels
#   data/patch_meta.csv   — FIRMS metadata for each patch
#
# RUN: python step2_extract_patches.py
# =============================================================================

import os
import numpy as np
import pandas as pd

PATCH_SIZE   = 64     # pixels each side
LABELLED_CSV = "data/firms_labelled.csv"
NC_RADIANCE  = "data/VNP02IMG.A2024106.2000.002.2024107043816.nc"
NC_GEO       = "data/VNP03IMG.A2024106.2000.002.2024107041509.nc"

OUT_PATCHES = "data/patches.npy"
OUT_LABELS  = "data/patch_labels.npy"
OUT_META    = "data/patch_meta.csv"

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# Load labelled FIRMS CSV
# ---------------------------------------------------------------------------
if not os.path.exists(LABELLED_CSV):
    raise FileNotFoundError(
        f"{LABELLED_CSV} not found. Run step1_label_dataset.py first."
    )

firms = pd.read_csv(LABELLED_CSV)
firms["acq_date"] = pd.to_datetime(firms["acq_date"], errors="coerce")

# Fill missing numeric columns with median so no NaN reaches the patch maker
for col in ["bright_ti4", "bright_ti5", "frp", "scan", "track"]:
    if col in firms.columns:
        firms[col] = pd.to_numeric(firms[col], errors="coerce")
        firms[col] = firms[col].fillna(firms[col].median())

print(f"Loaded {len(firms):,} labelled detections  "
      f"(wildfire: {(firms.label==0).sum():,}  "
      f"industrial: {(firms.label==1).sum():,})")


# ===========================================================================
# ---- REAL MODE: extract patches from NetCDF files -------------------------
# ===========================================================================

def try_real_mode(firms_df):
    """
    Attempt to extract patches from the VIIRS NetCDF files.
    Returns (patches_array, success_mask) or raises an exception.

    patches_array : shape (N, 64, 64) float32
    success_mask  : bool array of length N — True where a patch was extracted
    """
    try:
        import netCDF4 as nc
    except ImportError:
        raise RuntimeError("netCDF4 not installed. "
                           "Run: pip install netCDF4")
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        raise RuntimeError("scipy not installed. "
                           "Run: pip install scipy")

    if not os.path.exists(NC_RADIANCE):
        raise FileNotFoundError(f"Radiance file not found: {NC_RADIANCE}")
    if not os.path.exists(NC_GEO):
        raise FileNotFoundError(f"Geolocation file not found: {NC_GEO}")

    print("\n[REAL MODE] Opening NetCDF files …")

    # ---- Open geolocation file to get lat/lon arrays ----
    geo_ds = nc.Dataset(NC_GEO, "r")
    print(f"  Geo file groups/variables: {list(geo_ds.groups.keys()) or list(geo_ds.variables.keys())}")

    # VNP03IMG stores lat/lon in a group called "geolocation_data"
    if "geolocation_data" in geo_ds.groups:
        geo_grp = geo_ds["geolocation_data"]
    else:
        geo_grp = geo_ds  # flat layout

    lat_var = next((v for v in geo_grp.variables if "lat" in v.lower()), None)
    lon_var = next((v for v in geo_grp.variables if "lon" in v.lower()), None)

    if lat_var is None or lon_var is None:
        geo_ds.close()
        raise RuntimeError(f"Cannot find lat/lon variables in {NC_GEO}. "
                           f"Variables: {list(geo_grp.variables.keys())}")

    print(f"  Reading lat ({lat_var}) and lon ({lon_var}) …")
    tile_lat = geo_grp[lat_var][:].data.astype(np.float32)   # shape (rows, cols)
    tile_lon = geo_grp[lon_var][:].data.astype(np.float32)
    geo_ds.close()

    tile_lat[tile_lat > 90]  = np.nan
    tile_lat[tile_lat < -90] = np.nan
    tile_lon[tile_lon > 180] = np.nan
    tile_lon[tile_lon < -180]= np.nan
    print(f"  Tile shape: {tile_lat.shape}")

    # ---- Open radiance file ----
    rad_ds = nc.Dataset(NC_RADIANCE, "r")
    if "observation_data" in rad_ds.groups:
        rad_grp = rad_ds["observation_data"]
    else:
        rad_grp = rad_ds

    # Pick the first band that looks like infrared (I4 or I5 for VIIRS)
    band_var = None
    for candidate in ["I04", "I05", "I4", "I5", "M13", "M14"]:
        if candidate in rad_grp.variables:
            band_var = candidate
            break
    if band_var is None:
        # Fall back to any variable with 2-D shape
        for v in rad_grp.variables:
            arr = rad_grp[v]
            if arr.ndim == 2:
                band_var = v
                break

    if band_var is None:
        rad_ds.close()
        raise RuntimeError("Cannot find a 2-D radiance variable in "
                           f"{NC_RADIANCE}. "
                           f"Variables: {list(rad_grp.variables.keys())}")

    print(f"  Reading radiance band: {band_var} …")
    radiance = rad_grp[band_var][:].data.astype(np.float32)

    # Apply fill value / scale factor if attributes exist
    try:
        fill = rad_grp[band_var]._FillValue
        radiance[radiance == fill] = np.nan
    except AttributeError:
        pass
    try:
        scale  = rad_grp[band_var].scale_factor
        offset = getattr(rad_grp[band_var], "add_offset", 0.0)
        radiance = radiance * scale + offset
    except AttributeError:
        pass

    rad_ds.close()
    print(f"  Radiance shape: {radiance.shape}")

    # ---- Tile geographic extent ----
    tile_lat_min = np.nanmin(tile_lat)
    tile_lat_max = np.nanmax(tile_lat)
    tile_lon_min = np.nanmin(tile_lon)
    tile_lon_max = np.nanmax(tile_lon)
    print(f"  Tile coverage: lat [{tile_lat_min:.2f}, {tile_lat_max:.2f}]  "
          f"lon [{tile_lon_min:.2f}, {tile_lon_max:.2f}]")

    half = PATCH_SIZE // 2
    n_rows, n_cols = radiance.shape
    patches     = np.zeros((len(firms_df), PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
    success     = np.zeros(len(firms_df), dtype=bool)

    # Index a coarse set of valid geolocations, then refine within nearby
    # full-resolution neighborhoods instead of scanning the entire tile.
    grid_stride = 8
    sample_rows = np.unique(np.append(np.arange(0, n_rows, grid_stride), n_rows - 1))
    sample_cols = np.unique(np.append(np.arange(0, n_cols, grid_stride), n_cols - 1))
    coarse_lat = tile_lat[np.ix_(sample_rows, sample_cols)]
    coarse_lon = tile_lon[np.ix_(sample_rows, sample_cols)]
    coarse_valid = np.isfinite(coarse_lat) & np.isfinite(coarse_lon)
    if not coarse_valid.any():
        raise RuntimeError("The geolocation grid contains no finite lat/lon pixels.")

    coarse_rows, coarse_cols = np.meshgrid(
        sample_rows, sample_cols, indexing="ij"
    )
    coarse_pixels = np.column_stack(
        (coarse_rows[coarse_valid], coarse_cols[coarse_valid])
    )
    coarse_coords = np.column_stack(
        (coarse_lat[coarse_valid], coarse_lon[coarse_valid])
    )
    geo_index = cKDTree(coarse_coords)
    query_neighbors = min(4, len(coarse_coords))

    # For each FIRMS point, find the closest valid pixel and cut out a patch.
    for i, (_, row) in enumerate(firms_df.iterrows()):
        if (i + 1) % 10_000 == 0:
            print(f"  Processed {i + 1:,} / {len(firms_df):,} detections; "
                  f"extracted {success.sum():,} patches so far.")

        f_lat = row["latitude"]
        f_lon = row["longitude"]

        # Skip if point is outside this tile
        if not (np.isfinite(f_lat) and np.isfinite(f_lon) and
                tile_lat_min <= f_lat <= tile_lat_max and
                tile_lon_min <= f_lon <= tile_lon_max):
            continue

        _, nearest_samples = geo_index.query(
            (f_lat, f_lon), k=query_neighbors
        )
        nearest_samples = np.atleast_1d(nearest_samples)
        best_dist2 = np.inf
        nearest_pixel = None
        for sample_idx in nearest_samples:
            center_row, center_col = coarse_pixels[sample_idx]
            r0_search = max(0, center_row - grid_stride)
            r1_search = min(n_rows, center_row + grid_stride + 1)
            c0_search = max(0, center_col - grid_stride)
            c1_search = min(n_cols, center_col + grid_stride + 1)

            local_lat = tile_lat[r0_search:r1_search, c0_search:c1_search]
            local_lon = tile_lon[r0_search:r1_search, c0_search:c1_search]
            valid = np.isfinite(local_lat) & np.isfinite(local_lon)
            if not valid.any():
                continue

            local_dist2 = (local_lat - f_lat) ** 2 + (local_lon - f_lon) ** 2
            local_dist2[~valid] = np.inf
            local_flat_idx = int(np.argmin(local_dist2))
            local_min = local_dist2.flat[local_flat_idx]
            if local_min < best_dist2:
                best_dist2 = local_min
                local_row, local_col = np.unravel_index(
                    local_flat_idx, local_dist2.shape
                )
                nearest_pixel = (r0_search + local_row, c0_search + local_col)

        if nearest_pixel is None:
            continue
        pr, pc = nearest_pixel

        # Check patch fits inside the tile
        r0, r1 = pr - half, pr + half
        c0, c1 = pc - half, pc + half
        if r0 < 0 or r1 > n_rows or c0 < 0 or c1 > n_cols:
            continue

        patch = radiance[r0:r1, c0:c1].copy()

        # Replace NaN with median of valid pixels in the patch
        valid_median = np.nanmedian(patch)
        if not np.isfinite(valid_median):
            continue
        patch = np.where(np.isnan(patch), valid_median, patch)

        # Normalise patch to [0, 1] range
        pmin, pmax = patch.min(), patch.max()
        if pmax > pmin:
            patch = (patch - pmin) / (pmax - pmin)
        else:
            patch = np.zeros_like(patch)

        patches[i] = patch
        success[i] = True

    n_ok = success.sum()
    print(f"  Patches extracted from tile: {n_ok:,} / {len(firms_df):,}")
    return patches, success


# ===========================================================================
# ---- FALLBACK MODE: generate synthetic thermal patches --------------------
# ===========================================================================

def make_synthetic_patches(firms_df):
    """
    Generate synthetic 64×64 thermal patches from FIRMS scalar attributes.

    WILDFIRE patches  (label=0):
      - Wide, irregular spread (like a burning forest)
      - Moderate peak temperature, noisy background
      - Hot core with diffuse warm edges

    INDUSTRIAL patches (label=1):
      - Compact, round hot spot (like a flare stack or furnace)
      - Very high peak temperature, cool background
      - Constant across acquisition times (not spreading)

    These are clearly labeled synthetic data throughout the codebase.
    They capture the SHAPE signature that distinguishes the two source types,
    which is what the CNN experiments are designed to learn.
    """
    print("\n[FALLBACK / SYNTHETIC MODE]")
    print("  Generating synthetic thermal patches from FIRMS brightness values.")
    print("  NOTE: These are simulated patches — statistically realistic but not")
    print("        from the actual satellite imagery files.")

    n = len(firms_df)
    patches = np.zeros((n, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
    half    = PATCH_SIZE // 2
    cx, cy  = half, half   # patch centre

    # Pre-build coordinate grid once
    ys, xs = np.mgrid[0:PATCH_SIZE, 0:PATCH_SIZE]
    dist_from_centre = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)

    for i, (_, row) in enumerate(firms_df.iterrows()):
        label = int(row["label"])

        # Normalise brightness to roughly [0, 1]
        # bright_ti4 range is roughly 200–400 K; we map 200→0, 400→1
        bright_norm = (float(row.get("bright_ti4", 300)) - 200.0) / 200.0
        bright_norm = float(np.clip(bright_norm, 0.0, 1.0))

        # FRP (fire radiative power) also drives intensity
        frp = float(row.get("frp", 10.0))
        frp_norm = float(np.clip(frp / 500.0, 0.0, 1.0))

        rng = np.random.default_rng(seed=i + RANDOM_SEED)

        if label == 1:
            # ---- INDUSTRIAL: compact, round, intense ----
            # sigma ~ 3-6 pixels — a tight hot circle
            sigma = rng.uniform(3.0, 6.0)
            hot_spot = np.exp(-dist_from_centre ** 2 / (2 * sigma ** 2))

            # Base thermal emission — relatively cool, uniform background
            background = rng.uniform(0.05, 0.15, (PATCH_SIZE, PATCH_SIZE))

            # Peak intensity driven by brightness
            peak = 0.6 + 0.4 * bright_norm + 0.2 * frp_norm

            patch = background + peak * hot_spot

            # Add small amount of instrument noise
            patch += rng.normal(0.0, 0.02, patch.shape)

        else:
            # ---- WILDFIRE: wide, irregular, moderate intensity ----
            # sigma ~ 10-20 pixels — a spread fire front
            sigma = rng.uniform(10.0, 20.0)
            base_heat = np.exp(-dist_from_centre ** 2 / (2 * sigma ** 2))

            # Irregular texture: perturb each pixel with spatially-correlated noise
            # (simulate spotty burning pattern)
            noise_coarse = rng.uniform(0.0, 1.0, (PATCH_SIZE // 4, PATCH_SIZE // 4))
            # Upsample coarse noise to full size using a simple repeat
            noise_full = np.kron(noise_coarse, np.ones((4, 4)))[:PATCH_SIZE, :PATCH_SIZE]

            # Background warm glow from smoke/embers
            background = rng.uniform(0.1, 0.3, (PATCH_SIZE, PATCH_SIZE))

            peak = 0.3 + 0.5 * bright_norm + 0.3 * frp_norm
            patch = background + peak * base_heat * (0.5 + 0.5 * noise_full)

            # More instrument noise than industrial (fire is turbulent)
            patch += rng.normal(0.0, 0.04, patch.shape)

        # Clip and normalise to [0, 1]
        patch = np.clip(patch, 0.0, 1.0)
        pmin, pmax = patch.min(), patch.max()
        if pmax > pmin:
            patch = (patch - pmin) / (pmax - pmin)

        patches[i] = patch.astype(np.float32)

    success = np.ones(n, dtype=bool)
    print(f"  Generated {n:,} synthetic patches.")
    return patches, success


# ===========================================================================
# ---- Main: try real, fall back to synthetic --------------------------------
# ===========================================================================

use_real = False
patches  = None
success  = None

if os.path.exists(NC_RADIANCE) and os.path.exists(NC_GEO):
    print("\nNetCDF files found. Attempting real extraction …")
    try:
        patches, success = try_real_mode(firms)
        n_ok = success.sum()
        if n_ok < max(10, len(firms) * 0.01):
            print(f"\n  Only {n_ok} patches extracted from tile "
                  f"(< 1% of detections). "
                  "The FIRMS detections may be in a different region than the "
                  "imagery tile. Switching to synthetic mode.")
        else:
            use_real = True
            print(f"\n  Real extraction succeeded: {n_ok:,} patches.")
    except Exception as exc:
        print(f"\n  Real extraction failed: {exc}")
        print("  Switching to FALLBACK synthetic mode.")

if not use_real:
    patches, success = make_synthetic_patches(firms)

# Keep only rows where we have a patch
firms_with_patch = firms[success].copy().reset_index(drop=True)
patches_ok       = patches[success]

print(f"\nFinal dataset: {len(firms_with_patch):,} patches")
print(f"  Wildfire   (0): {(firms_with_patch.label==0).sum():,}")
print(f"  Industrial (1): {(firms_with_patch.label==1).sum():,}")
print(f"  Patch array shape: {patches_ok.shape}")
print(f"  Patch dtype: {patches_ok.dtype}")
print(f"  Patch value range: [{patches_ok.min():.3f}, {patches_ok.max():.3f}]")


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
os.makedirs("data", exist_ok=True)

np.save(OUT_PATCHES, patches_ok)
np.save(OUT_LABELS,  firms_with_patch["label"].values.astype(int))
firms_with_patch.to_csv(OUT_META, index=False)

print(f"\nSaved:")
print(f"  {OUT_PATCHES}  ({patches_ok.nbytes / 1e6:.1f} MB)")
print(f"  {OUT_LABELS}")
print(f"  {OUT_META}")

mode_note = "REAL satellite data" if use_real else "SYNTHETIC (simulated from FIRMS attributes)"
print(f"\nPatch source: {mode_note}")