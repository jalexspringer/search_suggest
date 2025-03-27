"""
Embedding functionality for generating vector representations of text.
"""
from typing import Dict, List, Optional, Any
import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Dictionary of recommended models with their dimensions and characteristics
RECOMMENDED_MODELS = {
    "all-MiniLM-L6-v2": {
        "dimension": 384,
        "description": "Fast general-purpose model with decent performance",
        "speed": "Very Fast",
        "quality": "Good"
    },
    "BAAI/bge-small-en-v1.5": {
        "dimension": 384,
        "description": "Small BGE model optimized for search with excellent performance",
        "speed": "Fast",
        "quality": "Very Good"
    },
    "BAAI/bge-base-en-v1.5": {
        "dimension": 768,
        "description": "Base BGE model with superior search performance",
        "speed": "Medium",
        "quality": "Excellent"
    },
    "intfloat/e5-small-v2": {
        "dimension": 384,
        "description": "Small E5 model with strong performance on diverse queries",
        "speed": "Fast",
        "quality": "Very Good"
    },
    "sentence-transformers/all-mpnet-base-v2": {
        "dimension": 768,
        "description": "High quality general purpose model",
        "speed": "Medium",
        "quality": "Excellent"
    },
    "sentence-transformers/multi-qa-MiniLM-L6-cos-v1": {
        "dimension": 384,
        "description": "Specialized for question-answering, good for search queries",
        "speed": "Fast",
        "quality": "Very Good for Q&A"
    },
    "sentence-transformers/msmarco-MiniLM-L6-cos-v5": {
        "dimension": 384,
        "description": "Optimized for search queries from Bing",
        "speed": "Fast",
        "quality": "Very Good for Search"
    }
}

class EmbeddingService:
    """Service for generating embeddings from text."""
    
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        """Initialize the embedding service.
        
        Args:
            model_name: Name of the sentence-transformers model to use
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embedding_dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Loaded local embedding model: {model_name}")
        logger.info(f"Embedding dimension: {self.embedding_dimension}")
        
    def create_embedding(self, text: str) -> List[float]:
        """Create an embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        # For BGE models, add a prefix to improve retrieval performance
        if "bge" in self.model_name.lower():
            text = f"Represent this sentence for searching relevant passages: {text}"
            
        embedding = self.model.encode(text)
        return embedding.tolist()
        
    def create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for a batch of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        # For BGE models, add a prefix to improve retrieval performance
        if "bge" in self.model_name.lower():
            texts = [f"Represent this sentence for searching relevant passages: {text}" for text in texts]
            
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
    
    @classmethod
    def list_recommended_models(cls) -> Dict[str, Dict[str, Any]]:
        """List recommended embedding models with their characteristics.
        
        Returns:
            Dictionary of model information
        """
        return RECOMMENDED_MODELS
