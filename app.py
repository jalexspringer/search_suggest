"""
Main application entry point for the search suggestions API.
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI

from search_suggest.api import app as api_app

# Load environment variables
load_dotenv()

# Export the FastAPI app
app = api_app

if __name__ == "__main__":
    # This block is used when running the app directly with python
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)