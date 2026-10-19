
"""
views.py — Django views for WildfireWatch.

Endpoints:
    GET /              -> home page (map + filter form)
    GET /api/data/     -> FIRMS data filtered by bbox + date -> GeoJSON
    GET /about/        -> how it works
"""

import io
import json
import logging
import os
from functools import lru_cache
from urllib.parse import urlencode

import numpy as np
import pandas as pd
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_http_methods

from .ml_engine import classify_detections, make_patch
from django.contrib.auth.decorators import login_required

logger = logging.getLogger(__name__)

# Path to the labelled FIRMS data (shipped with the app)
FIRMS_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "firms_labelled.csv"
)

WORLD_OUTLINE_PATH = os.path.join(
    os.path.dirname(__file__), "map_data", "world_outline.geojson"
)

_FIRMS_CACHE = None

DATASET_START_DATE = pd.Timestamp("2012-10-10")
DATASET_END_DATE = pd.Timestamp("2013-03-05")


@lru_cache(maxsize=1)
def _load_world_outline():
    """Load and cache the local Natural Earth country outlines."""
    with open(WORLD_OUTLINE_PATH, encoding="utf-8") as outline_file:
        return json.load(outline_file)


def _load_firms():
    """Load the built-in FIRMS dataset once and cache it."""
    global _FIRMS_CACHE

    if _FIRMS_CACHE is None and os.path.exists(FIRMS_DATA_PATH):
        df = pd.read_csv(FIRMS_DATA_PATH, low_memory=False)
        df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce")
        _FIRMS_CACHE = df

    return _FIRMS_CACHE


# ---------------------------------------------------------------------------
# Page views
# ---------------------------------------------------------------------------
@login_required
def home(request):
    """Main map page."""
    df = _load_firms()

    context = {
        "has_data": df is not None,
        "total_detections": f"{len(df):,}" if df is not None else "N/A",
    }

    return render(request, "classifier/home.html", context)


def about(request):
    """Display the About page."""
    return render(request, "classifier/about.html")


# ---------------------------------------------------------------------------
# API views
# ---------------------------------------------------------------------------

@require_http_methods(["GET"])
def api_world_outline(request):
    """Return the local Natural Earth country outlines used by the map."""
    return JsonResponse(_load_world_outline())

import hashlib, json as _json
_CLASSIFY_CACHE: dict = {}
@require_http_methods(["GET"])
def api_filter(request):
    df = _load_firms()
    if df is None:
        return JsonResponse({"error": "No built-in dataset found."}, status=404)

    try:
        lat_min = float(request.GET.get("lat_min", 25))
        lat_max = float(request.GET.get("lat_max", 50))
        lon_min = float(request.GET.get("lon_min", -130))
        lon_max = float(request.GET.get("lon_max", -65))

        # Round to 1 decimal so nearby identical views hit the cache
        cache_key = f"{round(lat_min,1)},{round(lat_max,1)},{round(lon_min,1)},{round(lon_max,1)}"

        if cache_key in _CLASSIFY_CACHE:
            return JsonResponse(_CLASSIFY_CACHE[cache_key], safe=False)

        mask = (
            df["latitude"].between(lat_min, lat_max)
            & df["longitude"].between(lon_min, lon_max)
            & df["acq_date"].between(DATASET_START_DATE, DATASET_END_DATE)
        )
        subset = df[mask].head(300)

        if len(subset) == 0:
            return JsonResponse({
                "type": "FeatureCollection", "features": [],
                "meta": {"count": 0, "message": "No detections in selected region/date range."},
            })

        results = classify_detections(subset)
        geojson = _to_geojson(results)
        _CLASSIFY_CACHE[cache_key] = geojson   # ← cache it
        return JsonResponse(geojson, safe=False)

    except Exception as e:
        logger.exception("api_filter error")
        return JsonResponse({"error": str(e)}, status=500)


@require_http_methods(["GET"])
def api_patch(request):
    """Return one generated thermal patch for a map popup."""
    try:
        brightness = float(request.GET["brightness"])
        frp = float(request.GET["frp"])
        seed = int(request.GET["seed"])

    except (KeyError, TypeError, ValueError):
        return JsonResponse(
            {
                "error": (
                    "brightness, frp, and integer seed parameters "
                    "are required"
                )
            },
            status=400,
        )

    if (
        not np.isfinite(brightness)
        or not np.isfinite(frp)
        or seed < 0
    ):
        return JsonResponse({
            "type": "FeatureCollection",
            "features": [],
            "meta": {
                "count": 0,
                "message": "API test: classification temporarily bypassed."
    }
})

    from matplotlib import colormaps
    from PIL import Image

    patch = make_patch(brightness, frp, seed)
    pixels = (colormaps["inferno"](patch) * 255).astype(np.uint8)

    image = Image.fromarray(pixels, mode="RGBA")
    image_data = io.BytesIO()
    image.save(image_data, format="PNG", optimize=True)

    response = HttpResponse(
        image_data.getvalue(),
        content_type="image/png",
    )
    response["Cache-Control"] = "public, max-age=86400"

    return response


# ---------------------------------------------------------------------------
# GeoJSON helper
# ---------------------------------------------------------------------------

def _to_geojson(df: pd.DataFrame) -> dict:
    """Convert classified FIRMS detections into a GeoJSON FeatureCollection."""
    features = []

    for _, r in df.iterrows():
        is_ind = r["label"] == "industrial"

        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [
                    float(r["longitude"]),
                    float(r["latitude"]),
                ],
            },
            "properties": {
                "label": r["label"],
                "confidence": float(r["confidence"]),
                "ensemble": float(r["ensemble"]),
                "pca_prob": float(r["pca_prob"]),
                "cnn_prob": float(r["cnn_prob"]),
                "firms_prob": float(r["firms_prob"]),
                "hybrid_prob": float(r["hybrid_prob"]),
                "brightness": float(r.get("bright_ti4", 0)),
                "frp": float(r.get("frp", 0)),
                "acq_date": str(r.get("acq_date", "")),
                "daynight": str(r.get("daynight", "D")),
                "confidence_level": str(r.get("confidence", "nominal")),
                "top_feature": str(r.get("top_feature", "frp")),
                "patch_url": "/api/patch/?" + urlencode({
                    "brightness": r.get("bright_ti4", 0),
                    "frp": r.get("frp", 0),
                    "seed": int(r.get("patch_seed", 0)),
                }),
                "color": "#1565c0" if is_ind else "#c94a1a",
                "marker_size": max(
                    6,
                    min(20, float(r.get("frp", 10)) / 20),
                ),
            },
        })

    wild = sum(
        1 for feature in features
        if feature["properties"]["label"] == "wildfire"
    )
    ind = len(features) - wild

    return {
        "type": "FeatureCollection",
        "features": features,
        "meta": {
            "count": len(features),
            "wildfires": wild,
            "industrial": ind,
        },
    }
