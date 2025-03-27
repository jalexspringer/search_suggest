"""
Vector store functionality using Qdrant.
"""
from typing import Dict, List, Optional, Tuple, Any
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models


class VectorStore:
    """Vector store for managing embeddings in Qdrant."""

    def __init__(self, url: str, api_key: str):
        """Initialize the vector store.

        Args:
            url: Qdrant server URL
            api_key: Qdrant API key
        """
        self.client = QdrantClient(url=url, api_key=api_key)
        
    def create_collection(
        self, 
        collection_name: str, 
        vector_size: int = 1536,
        distance: str = "cosine"
    ) -> None:
        """Create a collection for storing vectors.

        Args:
            collection_name: Name of the collection
            vector_size: Size of the embedding vectors
            distance: Distance metric to use (cosine, euclid, dot)
        """
        # Check if collection already exists
        collections = self.client.get_collections().collections
        if any(collection.name == collection_name for collection in collections):
            return
            
        # Create the collection
        # Map the distance string to the correct enum value
        distance_map = {
            "cosine": models.Distance.COSINE,
            "euclid": models.Distance.EUCLID,
            "dot": models.Distance.DOT
        }
        
        distance_enum = distance_map.get(distance.lower(), models.Distance.COSINE)
        
        # Create the collection
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=distance_enum,
            )
        )
        
    def upsert_vectors(
        self, 
        collection_name: str, 
        ids: List[str], 
        vectors: List[List[float]], 
        payloads: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Insert or update vectors in the collection.

        Args:
            collection_name: Name of the collection
            ids: List of vector IDs (strings)
            vectors: List of embedding vectors
            payloads: Optional list of payloads for each vector
        """
        if payloads is None:
            payloads = [{} for _ in ids]
            
        # Ensure each payload has the original ID stored
        for i, (id_str, payload) in enumerate(zip(ids, payloads)):
            # Make a copy of the payload to avoid modifying the original
            payloads[i] = payload.copy()
            # Store the original ID in the payload
            payloads[i]["original_id"] = id_str
            
        # Convert string IDs to numeric IDs for Qdrant
        # We'll use a hash of the string ID to generate a numeric ID
        numeric_ids = [abs(hash(id_str)) % (2**63) for id_str in ids]
            
        points = [
            models.PointStruct(
                id=numeric_id,
                vector=vector,
                payload=payload
            )
            for numeric_id, vector, payload in zip(numeric_ids, vectors, payloads)
        ]
        
        self.client.upsert(
            collection_name=collection_name,
            points=points
        )
        
    def search(
        self, 
        collection_name: str, 
        query_vector: List[float], 
        limit: int = 10
    ) -> List[Dict]:
        """Search for similar vectors in the collection.

        Args:
            collection_name: Name of the collection
            query_vector: Query embedding vector
            limit: Maximum number of results to return

        Returns:
            List of search results
        """
        results = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            with_payload=True,
            with_vectors=False
        )
        
        return [
            {
                "id": result.payload.get("original_id", str(result.id)),
                "score": result.score,
                "payload": result.payload
            }
            for result in results
        ]
