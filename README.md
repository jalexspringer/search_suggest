# Search Suggest API

A FastAPI application that finds Google merchant categories most similar to a given query using vector embeddings and semantic search.

## Features

- Parse Google Merchant Category taxonomy data
- Embed categories using OpenAI embeddings
- Store and search vectors using Qdrant
- API endpoint to find similar categories based on a query

## Setup

### Prerequisites

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) for package management

### Environment Variables

Create a `.env` file with the following variables:

```
QDRANT_API_KEY="your_qdrant_api_key"
QDRANT_URL="your_qdrant_url"
OPENAI_API_KEY="your_openai_api_key"
```

### Installation

1. Clone the repository
2. Install dependencies:

```bash
uv add -r requirements.txt
uv sync
source .venv/bin/activate
```

## Usage

### Run the API Server

Start the FastAPI development server with hot reloading:

```bash
./fastapi_dev
```

For production use:

```bash
./fastapi_run
```

### API Endpoints

- `GET /`: Web interface for the API
- `GET /models`: List available embedding models
- `GET /collections`: List available collections
- `GET /search`: Search for similar categories
  - Query parameters:
    - `query`: Search query (required)
    - `model`: Embedding model to use (default: BAAI/bge-small-en-v1.5)
    - `limit`: Maximum number of results to return (default: 10)
- `POST /compare`: Compare search results from multiple models
  - Request body:
    - `query`: Search query
    - `models`: List of models to compare
    - `limit`: Maximum number of results to return (default: 10)

## Development

For development, use the following commands:

```bash
# Run tests
uv run pytest

# Format code
uv run black .

# Lint code
uv run ruff check .

# Type checking
uv run mypy .
```

## Containerized Deployment

This application can be run in a container using Podman, which is useful for consistent deployment across environments.

### Prerequisites

- [Podman](https://podman.io/getting-started/installation) installed on your system
- [Podman Compose](https://github.com/containers/podman-compose) for multi-container setups

### Building and Running with Podman

1. Build the container image:

```bash
podman build -t search-suggest .
```

2. Run the container with environment variables:

```bash
podman run -p 8000:8000 \
  --env-file .env \
  search-suggest
```

Alternatively, you can specify environment variables directly:

```bash
podman run -p 8000:8000 \
  -e QDRANT_URL=your_qdrant_url \
  -e QDRANT_API_KEY=your_qdrant_api_key \
  search-suggest
```

### Using Podman Compose

For easier management, you can use Podman Compose:

1. Make sure your `.env` file is set up with the required variables:
   - `QDRANT_URL`: URL of your Qdrant instance (only needed for cloud Qdrant)
   - `QDRANT_API_KEY`: API key for Qdrant (only needed for cloud Qdrant)
   - `USE_LOCAL_QDRANT`: Set to "true" to use a local Qdrant instance via container

2. Run the application with Podman Compose:

```bash
podman-compose up -d
```

3. To stop the application:

```bash
podman-compose down
```

### Using Local Qdrant vs Cloud Qdrant

The application supports two modes for connecting to Qdrant:

1. **Cloud Qdrant** (default): Connects to a remote Qdrant instance using the provided URL and API key.
   - Set `USE_LOCAL_QDRANT="false"` in your `.env` file
   - Provide `QDRANT_URL` and `QDRANT_API_KEY` in your `.env` file

2. **Local Qdrant**: Runs a Qdrant container alongside the application.
   - Set `USE_LOCAL_QDRANT="true"` in your `.env` file
   - No need to provide `QDRANT_URL` or `QDRANT_API_KEY`
   - Data is persisted in a Docker volume

To switch between modes:

```bash
# For cloud Qdrant
echo "USE_LOCAL_QDRANT=false" >> .env
echo "QDRANT_URL=your_cloud_url" >> .env
echo "QDRANT_API_KEY=your_api_key" >> .env

# For local Qdrant
echo "USE_LOCAL_QDRANT=true" >> .env
```

When using local Qdrant, you'll need to populate the collections after starting the containers.

### GitHub Packages Deployment

This project is configured to automatically build and push Docker images to GitHub Packages when changes are pushed to the main branch and tests pass.

#### Using the GitHub Packages Image

1. Authenticate with GitHub Packages:

```bash
echo $GITHUB_TOKEN | podman login ghcr.io -u USERNAME --password-stdin
```

2. Pull the latest image:

```bash
podman pull ghcr.io/impactinc/search_suggest:latest
```

3. Run the container:

```bash
podman run -p 8000:8000 \
  --env-file .env \
  ghcr.io/impactinc/search_suggest:latest
```

#### CI/CD Pipeline

The GitHub Actions workflow in `.github/workflows/requirements.yml` handles:

1. Running tests on all pull requests and pushes to main
2. Building multi-architecture Docker images (amd64 and arm64)
3. Pushing images to GitHub Packages with appropriate tags:
   - `latest` - always points to the most recent build from main
   - `main` - the latest build from the main branch
   - `sha-<commit>` - specific commit hash for precise version control

## Local Embedding Models

This application uses local sentence-transformers models for generating embeddings, which offers several advantages:
- No API costs or rate limits
- Complete privacy of your data
- Fast performance for both single queries and batch processing

The following models are supported:
- `all-MiniLM-L6-v2`: Fast general-purpose model
- `BAAI/bge-small-en-v1.5`: Small BGE model optimized for search
- `BAAI/bge-base-en-v1.5`: Base BGE model with superior performance
- `intfloat/e5-small-v2`: Small E5 model for diverse queries
- `sentence-transformers/all-mpnet-base-v2`: High quality general purpose model
- `sentence-transformers/multi-qa-MiniLM-L6-cos-v1`: Specialized for question-answering
- `sentence-transformers/msmarco-MiniLM-L6-cos-v5`: Optimized for search queries