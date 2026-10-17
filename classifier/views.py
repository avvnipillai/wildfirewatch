"""
views.py — Django views for WildfireWatch.

Endpoints:
  GET  /              → home page (map + filter form)
  POST /classify/     → upload CSV or filter built-in data → JSON results
  GET  /api/data/     → FIRMS data filtered by bbox + date → GeoJSON
  GET  /about/        → how it works
"""

import json
import io
import os
import logging
from urllib.parse import urlencode
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from .ml_engine import classify_detections, make_patch

logger = logging.getLogger(__name__)

# ── Path to the labelled FIRMS data (shipped with the app) ───────────────────
FIRMS_DATA_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'firms_labelled.csv'
)
_FIRMS_CACHE = None

def _load_firms():
    global _FIRMS_CACHE
    if _FIRMS_CACHE is None and os.path.exists(FIRMS_DATA_PATH):
        df = pd.read_csv(FIRMS_DATA_PATH, low_memory=False)
        df['acq_date'] = pd.to_datetime(df['acq_date'], errors='coerce')
        _FIRMS_CACHE = df
    return _FIRMS_CACHE


# ── Views ────────────────────────────────────────────────────────────────────

def home(request):
    """Main map page."""
    df = _load_firms()
    context = {
        'has_data': df is not None,
        'date_min': df['acq_date'].min().date().isoformat() if df is not None else '2020-01-01',
        'date_max': df['acq_date'].max().date().isoformat() if df is not None else '2023-12-31',
        'total_detections': f'{len(df):,}' if df is not None else 'N/A',
    }
    return render(request, 'classifier/home.html', context)


def about(request):
    return render(request, 'classifier/about.html')


@csrf_exempt
@require_http_methods(['POST'])
def classify_upload(request):
    """
    POST with multipart CSV file.
    Returns JSON array of classified detections for the map.
    """
    try:
        if 'file' not in request.FILES:
            return JsonResponse({'error': 'No file uploaded'}, status=400)

        f = request.FILES['file']
        if not f.name.endswith('.csv'):
            return JsonResponse({'error': 'Only CSV files accepted'}, status=400)

        df = pd.read_csv(io.StringIO(f.read().decode('utf-8', errors='replace')),
                         low_memory=False)

        if 'latitude' not in df.columns or 'longitude' not in df.columns:
            return JsonResponse({'error': 'CSV must have latitude and longitude columns'}, status=400)

        df = df.head(300)   # cap for web performance
        results = classify_detections(df)
        return JsonResponse(_to_geojson(results), safe=False)

    except Exception as e:
        logger.exception('classify_upload error')
        return JsonResponse({'error': str(e)}, status=500)


@require_http_methods(['GET'])
def api_filter(request):
    """
    GET /api/data/?lat_min=&lat_max=&lon_min=&lon_max=&date_from=&date_to=&limit=200
    Returns classified GeoJSON from the built-in labelled FIRMS dataset.
    """
    df = _load_firms()
    if df is None:
        return JsonResponse({'error': 'No built-in dataset found. Upload a CSV instead.'}, status=404)

    try:
        lat_min  = float(request.GET.get('lat_min',  25))
        lat_max  = float(request.GET.get('lat_max',  50))
        lon_min  = float(request.GET.get('lon_min', -130))
        lon_max  = float(request.GET.get('lon_max',  -65))
        date_from = pd.to_datetime(request.GET.get('date_from', '2020-01-01'))
        date_to   = pd.to_datetime(request.GET.get('date_to',   '2023-12-31'))
        limit     = min(int(request.GET.get('limit', 300)), 500)

        mask = (
            df['latitude'].between(lat_min, lat_max) &
            df['longitude'].between(lon_min, lon_max) &
            df['acq_date'].between(date_from, date_to)
        )
        subset = df[mask].head(limit)

        if len(subset) == 0:
            return JsonResponse({'type': 'FeatureCollection', 'features': [],
                                 'meta': {'count': 0, 'message': 'No detections in selected region/date range.'}})

        results = classify_detections(subset)
        return JsonResponse(_to_geojson(results), safe=False)

    except Exception as e:
        logger.exception('api_filter error')
        return JsonResponse({'error': str(e)}, status=500)


@require_http_methods(['GET'])
def api_patch(request):
    """Return one generated thermal patch, loaded on demand by a map popup."""
    try:
        brightness = float(request.GET['brightness'])
        frp = float(request.GET['frp'])
        seed = int(request.GET['seed'])
    except (KeyError, TypeError, ValueError):
        return JsonResponse(
            {'error': 'brightness, frp, and integer seed parameters are required'},
            status=400,
        )

    if not np.isfinite(brightness) or not np.isfinite(frp) or seed < 0:
        return JsonResponse({'error': 'Patch parameters are out of range'}, status=400)

    from matplotlib import colormaps
    from PIL import Image

    patch = make_patch(brightness, frp, seed)
    pixels = (colormaps['inferno'](patch) * 255).astype(np.uint8)
    image = Image.fromarray(pixels, mode='RGBA')
    image_data = io.BytesIO()
    image.save(image_data, format='PNG', optimize=True)

    response = HttpResponse(image_data.getvalue(), content_type='image/png')
    response['Cache-Control'] = 'public, max-age=86400'
    return response


# ── GeoJSON helper ────────────────────────────────────────────────────────────
def _to_geojson(df: pd.DataFrame) -> dict:
    features = []
    for _, r in df.iterrows():
        is_ind = r['label'] == 'industrial'
        features.append({
            'type': 'Feature',
            'geometry': {
                'type': 'Point',
                'coordinates': [float(r['longitude']), float(r['latitude'])]
            },
            'properties': {
                'label':        r['label'],
                'confidence':   float(r['confidence']),
                'ensemble':     float(r['ensemble']),
                'pca_prob':     float(r['pca_prob']),
                'cnn_prob':     float(r['cnn_prob']),
                'firms_prob':   float(r['firms_prob']),
                'hybrid_prob':  float(r['hybrid_prob']),
                'brightness':   float(r.get('bright_ti4', 0)),
                'frp':          float(r.get('frp', 0)),
                'acq_date':     str(r.get('acq_date', '')),
                'daynight':     str(r.get('daynight', 'D')),
                'confidence_level': str(r.get('confidence', 'nominal')),
                'top_feature':  str(r.get('top_feature', 'frp')),
                'patch_url':    '/api/patch/?' + urlencode({
                    'brightness': r.get('bright_ti4', 0),
                    'frp': r.get('frp', 0),
                    'seed': int(r.get('patch_seed', 0)),
                }),
                'color':        '#1565c0' if is_ind else '#c94a1a',
                'marker_size':  max(6, min(20, float(r.get('frp', 10)) / 20)),
            }
        })

    wild = sum(1 for f in features if f['properties']['label'] == 'wildfire')
    ind  = len(features) - wild

    return {
        'type': 'FeatureCollection',
        'features': features,
        'meta': {
            'count':      len(features),
            'wildfires':  wild,
            'industrial': ind,
        }
    }