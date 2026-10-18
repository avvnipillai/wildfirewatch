# 🔥 WildfireWatch — Wildfire vs Industrial Heat Source Classifier

> **Can PCA and CNN, combined with NASA FIRMS thermal measurements, distinguish wildfire events from persistent industrial heat sources more effectively than conventional classification methods?**

---

## 📊 Results at a Glance

| Model | ROC-AUC | Avg Precision | F1 (Industrial) |
|---|---|---|---|
| **Hybrid** (PCA + CNN + FIRMS) | **0.8665** | **0.6707** | **0.4613** ✅ best |
| CNN | 0.8123 | 0.5486 | 0.4159 |
| FIRMS tabular | 0.7981 | 0.3446 | 0.3547 |
| PCA alone | 0.6922 | 0.2009 | 0.2214 |

The **Hybrid model beats all individual models** — confirming that fusing spatial image features (CNN), compressed thermal patterns (PCA), and spectral/temporal tabular signals (FIRMS) is more powerful than any single modality alone.

---

## 📡 What Data Does This Use?

### Dataset 1 — NASA FIRMS VIIRS SV-C2
- **Source:** [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/)
- **Sensor:** VIIRS (Visible Infrared Imaging Radiometer Suite) on Suomi-NPP and NOAA-20
- **What it contains:** Every fire / thermal anomaly detected by satellite, globally, updated within 3 hours of satellite overpass
- **Columns used:** latitude, longitude, brightness (I-4 band, ~3.7μm), brightness (I-5 band, ~11μm), Fire Radiative Power (FRP in MW), scan angle, track angle, acquisition date/time, day/night flag, confidence level
- **This project's extract:** ~200,000 detections from the VIIRS SV-C2 archive (the 640MB CSV), covering **North America, primarily California, Oregon, Washington, Nevada, and parts of the US Southwest**, spanning roughly **2012–2023** (the full VIIRS archive period). The exact years and bbox depend on which archive file you downloaded from FIRMS.

### Dataset 2 — Global Power Plant Database (GPPD)
- **Source:** [World Resources Institute](https://datasets.wri.org/dataset/globalpowerplantdatabase)
- **What it contains:** 35,000+ power plants worldwide — coal, gas, oil, nuclear, hydro, solar, wind — with lat/lon coordinates, capacity (MW), fuel type
- **Used for:** Automatic labelling — any FIRMS detection within **5 km** of a power plant is labelled *Industrial (1)*; everything else is *Wildfire (0)*

### Dataset 3 — VIIRS VNP02IMG + VNP03IMG (NetCDF)
- **Source:** [NASA LAADS DAAC](https://ladsweb.modaps.eosdis.nasa.gov/)
- **VNP02IMG:** Calibrated radiance values for each pixel in the VIIRS I-band (375m resolution)
- **VNP03IMG:** Geolocation (lat/lon) for each pixel in the swath
- **Used for:** Extracting 64×64 pixel thermal image patches centred on each FIRMS detection

---

## 🌍 Geographic and Temporal Coverage

| Parameter | Value |
|---|---|
| **Region** | Primarily continental USA (California, Pacific Northwest, Southwest) |
| **Time range** | ~2012–2023 (full VIIRS-SV-C2 archive) |
| **Spatial resolution** | 375 m per pixel (VIIRS I-band) |
| **Total detections** | ~200,000 FIRMS thermal anomalies |
| **Labelled wildfire** | ~179,000 (89.6%) |
| **Labelled industrial** | ~21,000 (10.4%) |
| **Patch size** | 64 × 64 pixels = 24 km × 24 km footprint |

---

## 🛰️ How Prediction Works from Satellite Images

Here is the complete chain from raw satellite to final prediction:

```
SATELLITE (Suomi-NPP, 824 km orbit, ~101 min period)
        ↓  scans Earth every ~12 hours per location
VIIRS SENSOR
  ├─ I-4 band: 3.55–3.93 μm  (Mid-wave IR — peak emission for fires ~600-1000K)
  ├─ I-5 band: 10.5–12.4 μm  (Thermal IR — ambient temperature)
  └─ 375 m ground resolution
        ↓
NASA FIRMS ALGORITHM (contextual fire detection)
  ├─ Compares each pixel's I-4 brightness to its surrounding background
  ├─ Flags pixel as "fire" if brightness > background + 3σ threshold
  ├─ Computes FRP: Fire Radiative Power (MW) from Stefan-Boltzmann law
  └─ Outputs one CSV row per flagged pixel: lat, lon, brightness, FRP, etc.
        ↓
THIS PROJECT'S PIPELINE
  ├─ LABELLING (step1): haversine distance to power plant DB → 0/1 label
  ├─ PATCH EXTRACTION (step2): 64×64 pixel window around each detection
  │      Real mode: VNP02IMG radiance + VNP03IMG geolocation → nearest pixel crop
  │      Synthetic fallback: Gaussian hotspot model from brightness + FRP values
  │
  ├─ EXPERIMENT 1 — PCA
  │      Flatten 64×64 → 4096-D vector
  │      PCA(50 components) → compress to 50-D
  │      Reconstruction error = how "unusual" the spatial pattern is
  │      Wildfire: high reconstruction error (irregular shape)
  │      Industrial: low reconstruction error (compact, repeating pattern)
  │
  ├─ EXPERIMENT 2 — CNN
  │      ThermalCNN: 3 conv blocks → 128-D feature vector
  │      Block 1: Conv(1→16) + BN + ReLU + MaxPool
  │      Block 2: Conv(16→32) + BN + ReLU + MaxPool
  │      Block 3: Conv(32→64) + BN + ReLU + AdaptiveAvgPool(4×4)
  │      Head: FC(1024→128) → Dropout → FC(128→1) → sigmoid
  │      Wildfire patches: wide, irregular hot regions
  │      Industrial patches: tight, stable circular hotspot
  │
  ├─ EXPERIMENT 3 — FIRMS Tabular
  │      Feature engineering on CSV columns:
  │        bright_diff = I4 - I5  (fire vs ambient contrast)
  │        frp_per_area = FRP / (scan × track)  (intensity density)
  │        is_night, confidence, cyclic time features
  │      Random Forest (200 trees) + Gradient Boosting (100 trees)
  │      Industrial: high FRP/area, both day+night, high confidence
  │      Wildfire: high FRP but spread, mostly daytime, variable confidence
  │
  └─ EXPERIMENT 4 — Hybrid
         PCA features (51-D) + CNN features (128-D) + FIRMS features (14-D)
         → 193-D fused vector → StandardScaler → LogisticRegression(C=0.1)
         Best AUC: 0.8665
```

---

## 🔬 What Makes This Project Unique

1. **Multi-modal fusion** — no existing open-source project fuses PCA spatial compression + CNN learned patterns + tabular spectral/temporal features from FIRMS in a single pipeline
2. **Zero manual labelling** — ground truth is derived algorithmically from the GPPD power plant database using haversine proximity, making it fully reproducible and scalable
3. **Dual-mode patch extraction** — real satellite NetCDF imagery when available, synthetic physics-based fallback otherwise — the pipeline never breaks
4. **Interpretable outputs** — PCA reconstruction error is directly human-readable as "how anomalous is this shape", not just a black-box score
5. **Physically grounded features** — FRP, I4-I5 contrast, and frp_per_area are rooted in Stefan-Boltzmann thermal physics, not arbitrary feature engineering
6. **Class-imbalance aware** — all models use `class_weight='balanced'` and `pos_weight` in the CNN loss, critical for the 89:11 imbalance
7. **Production-ready** — end-to-end pipeline: download → label → patch → train × 4 → evaluate → deploy, all in 9 Python files

---

## 📁 Project Structure

```
wildfire_classifier/
├── README.md
├── requirements.txt
│
├── step0_explore_data.py       # Explore FIRMS CSV + power plant DB
├── step1_label_dataset.py      # Label detections via GPPD proximity
├── step2_extract_patches.py    # Extract 64×64 thermal patches (real or synthetic)
│
├── experiment1_pca.py          # PCA + reconstruction error → LogisticRegression
├── experiment2_cnn.py          # ThermalCNN → binary classifier
├── experiment3_firms.py        # RandomForest + GradientBoosting on tabular features
├── experiment4_hybrid.py       # Late fusion of all three
├── evaluate_all.py             # Unified ROC/PR comparison plot
│
├── app.py                      # Streamlit demo app (local)
│
├── data/
│   ├── firms_labelled.csv      # Output of step1
│   ├── patches.npy             # Output of step2
│   ├── patch_labels.npy
│   └── patch_meta.csv
│
├── models/
│   └── cnn_model.pth           # Saved CNN weights
│
└── results/
    ├── pca_scores.npy
    ├── cnn_scores.npy
    ├── firms_scores.npy
    └── hybrid_scores.npy
```

---

## ⚙️ Setup & Run Order

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Explore your data
python step0_explore_data.py

# 3. Label detections
python step1_label_dataset.py

# 4. Extract thermal patches
python step2_extract_patches.py

# 5. Run experiments
python experiment1_pca.py
python experiment2_cnn.py
python experiment3_firms.py
python experiment4_hybrid.py

# 6. Compare all models
python evaluate_all.py

# 7. Run the web app (local)
pip install streamlit
streamlit run app.py
```

---

## 📦 Requirements

```
numpy pandas scikit-learn matplotlib seaborn
torch torchvision netCDF4 scipy tqdm joblib streamlit
```

---

## 📜 License

MIT License. Dataset credits: NASA FIRMS (public domain), WRI Global Power Plant Database (CC BY 4.0).

---

## 👩‍💻 Author

Built as a complete end-to-end ML research project — from raw satellite data to a deployed interactive classifier.
