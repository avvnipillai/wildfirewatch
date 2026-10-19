# ── WildfireWatch Dockerfile ──────────────────────────────────────────────────
# Multi-stage build: keeps final image lean.

# ── Stage 1: build dependencies ───────────────────────────────────────────────
FROM python:3.11-slim AS builder
WORKDIR /build

# System libs needed for numpy/scipy/matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# ── Stage 2: runtime image ─────────────────────────────────────────────────────
FROM python:3.11-slim
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# System libs for runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 && rm -rf /var/lib/apt/lists/*

# Copy app source
COPY . .

# Create needed dirs
RUN mkdir -p data models results staticfiles media

# Collect static files
ENV DJANGO_SETTINGS_MODULE=wildfirewatch.settings
ENV PYTHONUNBUFFERED=1
ENV DJANGO_SECRET_KEY=place_with_a_generated_django_secret_key
RUN python manage.py collectstatic --noinput

# Expose port
EXPOSE 8000

# Gunicorn: 2 workers, 4 threads each — good for CPU-bound ML inference

CMD ["sh", "-c", "python -u manage.py migrate --noinput && python create_admin.py && exec gunicorn wildfirewatch.wsgi:application --bind 0.0.0.0:${PORT:-10000} --workers 1 --threads 4 --timeout 120 --access-logfile - --error-logfile -"]
