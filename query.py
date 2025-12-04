"""
Search and retrieval module for TVH product findability system.

This module provides:
- Text and image embedding using OpenAI
- Semantic search using vector similarity
- Hybrid search combining semantic and keyword matching
- Image-based search capabilities
"""
import numpy as np
import pickle
import openai
from PIL import Image
import base64
import os
import re
from typing import Tuple

# Load OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load pre-computed embeddings and catalog data
try:
    DATA = pickle.load(open("data/embeddings.pkl", "rb"))
    df = DATA["df"]
    text_embeds = np.array(DATA["text_embeddings"])
    image_embeds = np.array(DATA["image_embeddings"])
except FileNotFoundError:
    print("Warning: embeddings.pkl not found. Run build_embeddings.py first.")
    df = None
    text_embeds = None
    image_embeds = None


def embed_text(query: str) -> np.ndarray:
    """
    Generate text embedding using OpenAI's text-embedding-3-large model.
    
    Args:
        query: Text string to embed
    
    Returns:
        Numpy array of embedding vector
    """
    resp = openai.embeddings.create(
        model="text-embedding-3-large",
        input=query
    )
    return np.array(resp.data[0].embedding)


def embed_image(file) -> np.ndarray:
    """
    Generate image embedding using OpenAI's image-embedding-3-large model.
    
    Args:
        file: File-like object containing image data
    
    Returns:
        Numpy array of embedding vector
    """
    b64 = base64.b64encode(file.read()).decode()
    resp = openai.embeddings.create(
        model="image-embedding-3-large",
        input={"image": b64}
    )
    return np.array(resp.data[0].embedding)


def compute_keyword_score(query: str, text: str) -> float:
    """
    Compute keyword matching score between query and text.
    Higher score for more keyword matches.
    
    Args:
        query: Search query string
        text: Text to match against
    
    Returns:
        Keyword matching score (0.0 to 1.0)
    """
    if not query or not text:
        return 0.0
    
    query_lower = query.lower()
    text_lower = text.lower()
    
    # Extract meaningful words (length > 2)
    query_words = [w for w in re.findall(r'\b\w+\b', query_lower) if len(w) > 2]
    
    if not query_words:
        return 0.0
    
    # Count matches
    matches = sum(1 for word in query_words if word in text_lower)
    
    # Normalize by number of query words
    score = matches / len(query_words) if query_words else 0.0
    
    return min(score, 1.0)


def search_text(query: str, top_k: int = 5, hybrid: bool = True, alpha: float = 0.7) -> Tuple:
    """
    Search products using text query with optional hybrid (semantic + keyword) approach.
    
    Args:
        query: Text search query
        top_k: Number of results to return
        hybrid: If True, combine semantic and keyword matching
        alpha: Weight for semantic score (1-alpha for keyword score) when hybrid=True
    
    Returns:
        Tuple of (results DataFrame, scores array)
    """
    if df is None or text_embeds is None:
        raise ValueError("Embeddings not loaded. Run build_embeddings.py first.")
    
    # Semantic search: compute cosine similarity
    q_emb = embed_text(query)
    semantic_scores = text_embeds @ q_emb  # Dot product (cosine similarity for normalized vectors)
    
    if hybrid:
        # Hybrid search: combine semantic and keyword scores
        keyword_scores = np.array([
            compute_keyword_score(query, str(row["title"]) + " " + str(row["description"]))
            for _, row in df.iterrows()
        ])
        
        # Normalize both scores to [0, 1]
        if semantic_scores.max() > semantic_scores.min():
            semantic_norm = (semantic_scores - semantic_scores.min()) / (semantic_scores.max() - semantic_scores.min())
        else:
            semantic_norm = np.zeros_like(semantic_scores)
        
        if keyword_scores.max() > keyword_scores.min():
            keyword_norm = (keyword_scores - keyword_scores.min()) / (keyword_scores.max() - keyword_scores.min())
        else:
            keyword_norm = np.zeros_like(keyword_scores)
        
        # Combine scores
        combined_scores = alpha * semantic_norm + (1 - alpha) * keyword_norm
        scores = combined_scores
    else:
        # Pure semantic search
        scores = semantic_scores
    
    # Get top K results
    idx = scores.argsort()[::-1][:top_k]
    return df.iloc[idx], scores[idx]


def search_image(file, top_k: int = 5) -> Tuple:
    """
    Search products using image query.
    
    Args:
        file: File-like object containing image data
        top_k: Number of results to return
    
    Returns:
        Tuple of (results DataFrame, scores array)
    """
    if df is None or image_embeds is None:
        raise ValueError("Embeddings not loaded. Run build_embeddings.py first.")
    
    q = embed_image(file)
    scores = image_embeds @ q  # Cosine similarity
    idx = scores.argsort()[::-1][:top_k]
    return df.iloc[idx], scores[idx]
