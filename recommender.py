"""
Recommendation system based on frequently-bought-together (co-purchase) data.

This module provides functions to load co-purchase relationships and
generate product recommendations for a given product ID.
"""
import pandas as pd
import os
from pathlib import Path

# Configuration
CO_PURCHASE_PATH = "data/co_purchase.csv"


def load_pairs():
    """
    Load co-purchase relationship data from CSV file.
    
    Returns:
        DataFrame with columns: product_id, related_product_id, score
    """
    if not os.path.exists(CO_PURCHASE_PATH):
        print(f"Warning: Co-purchase data not found at {CO_PURCHASE_PATH}")
        print("Run generate_copurchase.py to create the data file.")
        return pd.DataFrame(columns=["product_id", "related_product_id", "score"])
    
    df = pd.read_csv(CO_PURCHASE_PATH)
    return df


def recommend(product_id, top_k=5):
    """
    Get top K recommended products for a given product ID based on co-purchase data.
    
    Args:
        product_id: The product ID to get recommendations for (string or number)
        top_k: Number of recommendations to return (default: 5)
    
    Returns:
        DataFrame with recommended products, sorted by score (descending)
    """
    df = load_pairs()
    
    if df.empty:
        return pd.DataFrame(columns=["product_id", "related_product_id", "score"])
    
    # Convert product_id to string for matching
    product_id_str = str(product_id).strip()
    
    # Filter for the given product_id (try exact match first)
    sub = df[df["product_id"].astype(str) == product_id_str]
    
    # If no exact match, try case-insensitive
    if sub.empty:
        sub = df[df["product_id"].astype(str).str.upper() == product_id_str.upper()]
    
    if sub.empty:
        return pd.DataFrame(columns=["product_id", "related_product_id", "score"])
    
    # Sort by score and return top K
    sub = sub.sort_values("score", ascending=False)
    return sub.head(top_k)


def get_recommended_product_ids(product_id, top_k=5):
    """
    Get list of recommended product IDs (simpler interface).
    
    Args:
        product_id: The product ID to get recommendations for
        top_k: Number of recommendations to return
    
    Returns:
        List of recommended product IDs
    """
    recs = recommend(product_id, top_k)
    if recs.empty:
        return []
    return recs["related_product_id"].tolist()
