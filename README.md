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

## Deployment to Heroku

This application is configured for deployment to Heroku using the GitHub integration.

### Prerequisites

1. A Heroku account
2. The application pushed to GitHub

### Deployment Steps

1. Create a new Heroku app
2. Connect the app to your GitHub repository
3. Enable automatic deploys from the main branch
4. Set the following config vars in Heroku:
   - `QDRANT_API_KEY`: Your Qdrant API key
   - `QDRANT_URL`: Your Qdrant URL

The application uses the following files for Heroku deployment:
- `Procfile`: Defines the command to run the web server
- `runtime.txt`: Specifies Python 3.13.2 as the runtime

### Manual Deployment

If you prefer to deploy manually:

```bash
# Login to Heroku
heroku login

# Add Heroku remote
heroku git:remote -a your-heroku-app-name

# Push to Heroku
git push heroku main
```

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