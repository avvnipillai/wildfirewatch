"""
ml_engine.py — Core ML prediction logic for WildfireWatch.

Loads trained models once at startup (cached in module scope).
Called by Django views; never imports Django itself.
"""
import os, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')

# ─── CNN Architecture (must match experiment2_cnn.py) ────────────────────────
class ThermalCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(nn.Conv2d(1,16,3,padding=1),nn.BatchNorm2d(16),nn.ReLU(True),nn.MaxPool2d(2))
        self.block2 = nn.Sequential(nn.Conv2d(16,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(True),nn.MaxPool2d(2))
        self.block3 = nn.Sequential(nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(True),nn.AdaptiveAvgPool2d(4))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*4*4,128),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(128,1),
        )

    def forward(self, x):
        x = self.block1(x); x = self.block2(x); x = self.block3(x)
        return self.head(x).squeeze(1)

    def extract_features(self, x):
        x = self.block1(x); x = self.block2(x); x = self.block3(x)
        return self.head[:3](x)


# ─── Module-level singletons ─────────────────────────────────────────────────
_cnn   = None
_device = torch.device('cpu')

def get_cnn():
    global _cnn
    if _cnn is None:
        _cnn = ThermalCNN().to(_device)
        weights = os.path.join(os.path.dirname(__file__), '..', 'models', 'cnn_model.pth')
        if os.path.exists(weights):
            _cnn.load_state_dict(torch.load(weights, map_location=_device))
        _cnn.eval()
    return _cnn


# ─── Synthetic patch generation ──────────────────────────────────────────────
def make_patch(bright_ti4, frp, seed=0):
    SZ = 64; half = SZ // 2
    ys, xs = np.mgrid[0:SZ, 0:SZ]
    dist   = np.sqrt((xs-half)**2 + (ys-half)**2).astype(np.float32)
    b = np.clip((float(bright_ti4)-200)/200, 0, 1)
    f = np.clip(float(frp)/500, 0, 1)
    rng = np.random.default_rng(seed=seed)
    if f > 0.3 and b > 0.6:                      # industrial-like
        sigma = rng.uniform(3, 6)
        bg    = rng.uniform(0.05, 0.15, (SZ,SZ))
        peak  = 0.6 + 0.4*b
        patch = bg + peak * np.exp(-dist**2/(2*sigma**2))
        patch += rng.normal(0, 0.02, patch.shape)
    else:                                          # wildfire-like
        sigma = rng.uniform(10, 20)
        noise = np.kron(rng.uniform(0,1,(SZ//4,SZ//4)), np.ones((4,4)))[:SZ,:SZ]
        bg    = rng.uniform(0.1, 0.3, (SZ,SZ))
        peak  = 0.3 + 0.5*b
        patch = bg + peak * np.exp(-dist**2/(2*sigma**2)) * (0.5+0.5*noise)
        patch += rng.normal(0, 0.04, patch.shape)
    patch = np.clip(patch, 0, 1)
    mn, mx = patch.min(), patch.max()
    return ((patch-mn)/(mx-mn+1e-8)).astype(np.float32)


# ─── FIRMS feature engineering ────────────────────────────────────────────────
FIRMS_COLS = ['bright_ti4','bright_ti5','frp','scan','track',
              'bright_diff','frp_per_area','is_night','conf_encoded',
              'hour_sin','hour_cos','month_sin','month_cos','day_of_year']

def engineer_features(df: pd.DataFrame) -> np.ndarray:
    d = df.copy()
    if 'bright_ti4' not in d and 'brightness' in d:
        d['bright_ti4'] = d['brightness']

    for c in ['bright_ti4','bright_ti5','frp','scan','track']:
        default = d['bright_ti4'] if c == 'bright_ti5' and 'bright_ti4' in d else 0
        values = d[c] if c in d else pd.Series(default, index=d.index)
        d[c] = pd.to_numeric(values, errors='coerce').fillna(0)
    acq_time = d['acq_time'] if 'acq_time' in d else pd.Series(1200, index=d.index)
    d['acq_time'] = pd.to_numeric(acq_time, errors='coerce').fillna(1200).astype(int)
    d['hour_of_day'] = d['acq_time'] // 100
    acq_date = d['acq_date'] if 'acq_date' in d else pd.Series('2023-01-01', index=d.index)
    d['acq_date'] = pd.to_datetime(acq_date, errors='coerce')
    d['month']       = d['acq_date'].dt.month.fillna(6).astype(int)
    d['day_of_year'] = d['acq_date'].dt.dayofyear.fillna(180).astype(int)
    d['bright_diff'] = d['bright_ti4'] - d['bright_ti5']
    px = (d['scan'] * d['track']).replace(0, np.nan)
    d['frp_per_area']= (d['frp'] / px).fillna(0)
    daynight = d['daynight'] if 'daynight' in d else pd.Series('D', index=d.index)
    d['is_night']    = (daynight.astype('string').str.strip().str.upper() == 'N').astype(int)
    conf_map         = {'low':0,'nominal':1,'high':2}
    confidence = d['confidence'] if 'confidence' in d else pd.Series('nominal', index=d.index)
    d['conf_encoded']= confidence.astype('string').str.strip().str.lower().map(conf_map).fillna(1).astype(int)
    d['hour_sin'] = np.sin(2*np.pi*d['hour_of_day']/24)
    d['hour_cos'] = np.cos(2*np.pi*d['hour_of_day']/24)
    d['month_sin'] = np.sin(2*np.pi*d['month']/12)
    d['month_cos'] = np.cos(2*np.pi*d['month']/12)
    cols = [c for c in FIRMS_COLS if c in d.columns]
    return d[cols].values.astype(np.float32), cols


# ─── Main prediction function ─────────────────────────────────────────────────
def classify_detections(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input:  DataFrame with FIRMS columns
    Output: DataFrame with all input columns + prediction columns added
    """
    df = df.copy()
    if 'bright_ti4' not in df and 'brightness' in df:
        df['bright_ti4'] = df['brightness']

    n   = len(df)
    cnn = get_cnn()

    # 1. Patches
    patches = np.array([
        make_patch(r.get('bright_ti4', 300), r.get('frp', 10), seed=i)
        for i, (_, r) in enumerate(df.iterrows())
    ])

    # 2. PCA
    X_flat  = patches.reshape(n, -1)
    n_comp  = min(50, n-1, X_flat.shape[1])
    pca     = PCA(n_components=n_comp)
    X_pca   = pca.fit_transform(X_flat)
    X_back  = pca.inverse_transform(X_pca)
    recon   = np.mean((X_flat-X_back)**2, axis=1, keepdims=True)
    X_pca_f = np.hstack([X_pca, recon])
    pca_prob = 1/(1+np.exp(-(recon.flatten()-recon.mean())*10))

    # 3. CNN
    t = torch.tensor(patches[:,np.newaxis], dtype=torch.float32)
    with torch.no_grad():
        logits  = cnn(t).numpy()
        cnn_f   = cnn.extract_features(t).numpy()
    cnn_prob = 1/(1+np.exp(-logits))

    # 4. FIRMS tabular RF
    X_tab, feat_names = engineer_features(df)
    sc = StandardScaler(); X_tab_s = sc.fit_transform(X_tab)
    pseudo = (cnn_prob > 0.5).astype(int)
    if pseudo.sum() == 0: pseudo[np.argmax(cnn_prob)] = 1
    if pseudo.sum() == n:  pseudo[np.argmin(cnn_prob)] = 0
    rf = RandomForestClassifier(100, random_state=42, class_weight='balanced')
    rf.fit(X_tab_s, pseudo)
    firms_prob = rf.predict_proba(X_tab_s)[:,1]
    importances = dict(zip(feat_names, rf.feature_importances_))

    # 5. Hybrid
    sc2 = StandardScaler(); sc3 = StandardScaler()
    X_fused = np.hstack([sc2.fit_transform(X_pca_f),
                         sc3.fit_transform(cnn_f),
                         X_tab_s])
    lr = LogisticRegression(max_iter=1000, C=0.1, class_weight='balanced', random_state=42)
    lr.fit(X_fused, pseudo)
    hybrid_prob = lr.predict_proba(X_fused)[:,1]

    # 6. Ensemble
    ensemble = 0.2*pca_prob + 0.3*cnn_prob + 0.2*firms_prob + 0.3*hybrid_prob

    out = df.copy()
    out['pca_prob']    = np.round(pca_prob,   4)
    out['cnn_prob']    = np.round(cnn_prob,   4)
    out['firms_prob']  = np.round(firms_prob, 4)
    out['hybrid_prob'] = np.round(hybrid_prob,4)
    out['ensemble']    = np.round(ensemble,   4)
    out['label']       = np.where(ensemble > 0.5, 'industrial', 'wildfire')
    out['confidence']  = np.where(
        ensemble > 0.5,
        np.round((ensemble-0.5)*200, 1),
        np.round((0.5-ensemble)*200, 1)
    ).clip(0, 100)
    out['patch_seed']  = np.arange(n)
    out['top_feature'] = [_top_feature(importances) for _ in range(n)]
    return out


def _top_feature(importances: dict) -> str:
    if not importances: return 'frp'
    return max(importances, key=importances.get)