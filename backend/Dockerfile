FROM python:3.11-slim-bullseye

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

ENV PIP_USE_PEP517=0
ENV PIP_NO_BUILD_ISOLATION=1
ENV PIP_DEFAULT_TIMEOUT=300
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1

# Bake models into image
RUN mkdir -p /app/models/embeddings /app/models/reranker

COPY requirements.txt .
# Note: uv was tried here but its strict resolver rejects the existing
# dep conflicts (google-genai wants websockets<15.1 but project pulls 16.0;
# gtts wants click<8.2 but project pulls 8.3.2). --resolution=lowest-direct
# fell into ancient package versions with Python 2 print syntax. Sticking
# with pip + legacy-resolver until requirements.txt is properly pinned.
RUN pip install --use-deprecated=legacy-resolver -r requirements.txt

COPY . .

# Pre-download models during build phase
RUN python scripts/download_models.py

# Pre-download Skyfield astronomy data (panchang service).
# Without this, hip_main.dat (10MB) + de421.bsp (16MB) download from the
# internet on EVERY container boot = 2-3 minute cold start delay.
# Downloads to: /app/ (skyfield.api.load uses CWD) AND
#               ~/.local/share/jyotishganit/ (jyotishganit's custom Loader)
RUN python -c "\
import os; \
os.makedirs(os.path.expanduser('~/.local/share/jyotishganit'), exist_ok=True); \
from skyfield.api import Loader; \
from skyfield.data import hipparcos; \
jl = Loader(os.path.expanduser('~/.local/share/jyotishganit')); \
jl('de421.bsp'); \
jl.timescale(); \
print('de421.bsp + timescale cached to jyotishganit dir'); \
al = Loader('/app'); \
al('de421.bsp'); \
al.open(hipparcos.URL); \
print('hip_main.dat + de421.bsp cached to /app'); \
assert os.path.exists(os.path.expanduser('~/.local/share/jyotishganit/de421.bsp')), 'FATAL: de421.bsp missing'; \
print('VERIFIED: all skyfield data present')"

# Force offline mode for runtime
ENV TRANSFORMERS_OFFLINE=1
ENV HF_HUB_OFFLINE=1

EXPOSE 8080

# Use shell form for CMD to allow environment variable expansion ($PORT)
# This is critical for Google Cloud Run compatibility
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1 --timeout-keep-alive 65
