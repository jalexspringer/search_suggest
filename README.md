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
uv add --all
```

## Usage

### Populate the Database

Before using the API, you need to populate the Qdrant database with taxonomy embeddings:

```bash
uv run python -m search_suggest.cli populate
```

Options:
- `--taxonomy-file`: Path to the taxonomy file (default: data/taxonomy.txt)
- `--max-level`: Maximum level of categories to include (default: 3)
- `--collection`: Name of the Qdrant collection (default: merchant_categories)

### Run the API Server

Start the FastAPI server:

```bash
fastapi dev
```

Or with uvicorn directly:

```bash
uv run uvicorn app:app --reload
```

### API Endpoints

- `GET /`: Root endpoint
- `GET /api/search`: Search for similar categories
  - Query parameters:
    - `query`: Search query (required)
    - `limit`: Maximum number of results to return (default: 10)
    - `collection`: Qdrant collection to search (default: merchant_categories)

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