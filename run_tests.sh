#!/bin/bash
# Script to run tests using uv

# Run all tests
echo "Running all tests..."
uv run pytest

# To run specific test files, uncomment and modify the lines below
# echo "Running specific tests..."
# uv run pytest tests/test_embeddings.py -v
# uv run pytest tests/test_api.py -v

# To run tests with specific markers, uncomment and modify the lines below
# echo "Running tests without slow tests..."
# uv run pytest -m "not slow" -v
