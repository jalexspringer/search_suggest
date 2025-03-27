"""
Test script to verify connections to Qdrant and OpenAI.
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient

def test_connections():
    """Test connections to Qdrant and OpenAI."""
    # Load environment variables
    load_dotenv()
    
    # Check OpenAI connection
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("❌ OPENAI_API_KEY environment variable is not set")
        return False
    
    try:
        client = OpenAI(api_key=openai_api_key)
        response = client.embeddings.create(
            input="Test embedding",
            model="text-embedding-3-small"
        )
        print(f"✅ Successfully connected to OpenAI API")
        print(f"   Embedding dimension: {len(response.data[0].embedding)}")
    except Exception as e:
        print(f"❌ Failed to connect to OpenAI API: {str(e)}")
        return False
    
    # Check Qdrant connection
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    if not qdrant_url or not qdrant_api_key:
        print("❌ QDRANT_URL or QDRANT_API_KEY environment variable is not set")
        return False
    
    try:
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        collections = client.get_collections()
        print(f"✅ Successfully connected to Qdrant")
        print(f"   Available collections: {[c.name for c in collections.collections]}")
    except Exception as e:
        print(f"❌ Failed to connect to Qdrant: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    if test_connections():
        print("\n✅ All connections successful! You can now run the populate command.")
    else:
        print("\n❌ Some connections failed. Please check your environment variables.")
