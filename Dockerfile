FROM python:3.13-slim

LABEL maintainer="alexspringer@pm.me" \
      description="Search Suggest API for Google Merchant Categories"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && export PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Copy only what's needed for dependency installation first
COPY pyproject.toml ./

# Create a requirements.txt file from pyproject.toml (excluding dev dependencies)
RUN pip install tomli && \
    python -c "import tomli; import json; f=open('pyproject.toml', 'rb'); p=tomli.load(f); print('\n'.join(p['project']['dependencies']))" > requirements.txt

# Install dependencies (no dev dependencies)
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY search_suggest/ search_suggest/
COPY data/ data/
COPY fastapi_run ./

# Make the fastapi_run script executable
RUN chmod +x ./fastapi_run

# Expose port (Heroku will override this)
ENV PORT=8000
EXPOSE $PORT

# Environment variables that can be overridden at runtime
ENV QDRANT_API_KEY="" \
    QDRANT_URL=""

# Run the application
CMD uvicorn search_suggest.api:app --host=0.0.0.0 --port=$PORT
