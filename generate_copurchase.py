"""
Generate co-purchase (frequently-bought-together) data for recommendations.

This module creates synthetic co-purchase relationships based on:
- Products on the same page (likely related)
- Products with similar descriptions
- Random associations to simulate real purchase patterns

In production, this would be replaced with actual transaction data analysis.
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path

# Configuration
CATALOG_PATH = "data/catalog_clean.csv"
OUTPUT_PATH = "data/co_purchase.csv"
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def generate_copurchase_data(df: pd.DataFrame, avg_links: int = 3) -> pd.DataFrame:
    """
    Generate co-purchase relationships for products.
    
    Args:
        df: DataFrame with product catalog (must have 'product_id' and 'page' columns)
        avg_links: Average number of related products per item
    
    Returns:
        DataFrame with columns: product_id, related_product_id, score
    """
    # Filter out rows without product_id
    df_valid = df[df["product_id"].notna() & (df["product_id"].astype(str).str.strip() != "")].copy()
    
    if len(df_valid) == 0:
        print("Warning: No valid product IDs found in catalog")
        return pd.DataFrame(columns=["product_id", "related_product_id", "score"])
    
    product_ids = df_valid["product_id"].unique().tolist()
    n_products = len(product_ids)
    
    print(f"Generating co-purchase data for {n_products} products...")
    
    copurchase_pairs = []
    
    page_groups = df_valid.groupby("page")["product_id"].apply(list).to_dict()
    
    for i, pid in enumerate(product_ids):
        related = set()
        
        pid_row = df_valid[df_valid["product_id"] == pid]
        if not pid_row.empty:
            page = pid_row.iloc[0]["page"]
            same_page_products = page_groups.get(page, [])
            # Add products from same page (excluding self)
            for other_pid in same_page_products:
                if other_pid != pid and other_pid not in related:
                    related.add(other_pid)
                    copurchase_pairs.append({
                        "product_id": pid,
                        "related_product_id": other_pid,
                        "score": 0.9 
                    })
        
        if not pid_row.empty:
            page = pid_row.iloc[0]["page"]
            nearby_pages = [p for p in range(max(1, page - 2), min(df_valid["page"].max() + 1, page + 3)) if p != page]
            for nearby_page in nearby_pages[:3]:  
                nearby_products = page_groups.get(nearby_page, [])
                for other_pid in nearby_products[:2]:  # Max 2 per nearby page
                    if other_pid != pid and other_pid not in related:
                        related.add(other_pid)
                        copurchase_pairs.append({
                            "product_id": pid,
                            "related_product_id": other_pid,
                            "score": 0.6  # Moderate score for nearby pages
                        })

        while len(related) < avg_links:
            random_idx = np.random.randint(0, n_products)
            random_pid = product_ids[random_idx]
            if random_pid != pid and random_pid not in related:
                related.add(random_pid)
                copurchase_pairs.append({
                    "product_id": pid,
                    "related_product_id": random_pid,
                    "score": 0.3  # Lower score for random associations
                })
    
    result_df = pd.DataFrame(copurchase_pairs)
    
    result_df = result_df.sort_values("score", ascending=False)
    result_df = result_df.drop_duplicates(subset=["product_id", "related_product_id"], keep="first")
    
    print(f"Generated {len(result_df)} co-purchase relationships")
    return result_df


def main():
    """Main function to generate and save co-purchase data."""
    print("Loading catalog...")
    df = pd.read_csv(CATALOG_PATH)
    
    print(f"Catalog loaded: {len(df)} rows")
    
    # Generate co-purchase data
    copurchase_df = generate_copurchase_data(df, avg_links=4)
    
    # Save to CSV
    copurchase_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Co-purchase data saved to {OUTPUT_PATH}")
    
    # Print statistics
    print(f"\nStatistics:")
    print(f"  Total relationships: {len(copurchase_df)}")
    print(f"  Unique products with recommendations: {copurchase_df['product_id'].nunique()}")
    print(f"  Average recommendations per product: {len(copurchase_df) / copurchase_df['product_id'].nunique():.2f}")


if __name__ == "__main__":
    main()

