import openai
import pandas as pd
import pickle
from PIL import Image
import base64
import io
import os
from dotenv import load_dotenv

load_dotenv()

CATALOG = "data/catalog_clean.csv"
CATALOG_ENHANCED = "data/catalog_with_descriptions.csv"
OUT = "data/embeddings.pkl"

openai.api_key = os.getenv("OPENAI_API_KEY")


def encode_text(text: str):
    """Embed text using OpenAI"""
    resp = openai.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return resp.data[0].embedding


def encode_image(image_path: str):
    """Embed image using OpenAI"""
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    resp = openai.embeddings.create(
        model="image-embedding-3-large",
        input={"image": b64}
    )
    return resp.data[0].embedding


def build_embeddings(use_enhanced: bool = True):
    """
    Build embeddings for all catalog items.
    Creates both text and image embeddings using OpenAI's embedding models.
    Saves embeddings to a pickle file for fast retrieval.
    
    Args:
        use_enhanced: If True, use catalog_with_descriptions.csv if it exists,
                     otherwise fall back to catalog_clean.csv
    """
    # Try to use enhanced catalog if available
    catalog_path = CATALOG
    if use_enhanced and os.path.exists(CATALOG_ENHANCED):
        catalog_path = CATALOG_ENHANCED
        print(f"Using enhanced catalog with LLM-generated descriptions: {CATALOG_ENHANCED}")
    else:
        print(f"Using standard catalog: {CATALOG}")
        if use_enhanced:
            print("  (To use LLM-generated descriptions, run: python3 generate_descriptions.py)")
    
    df = pd.read_csv(catalog_path)
    df = df[df["title"].notna() & (df["title"].astype(str).str.strip() != "")]
    df = df[df["description"].notna() & (df["description"].astype(str).str.strip() != "")]
    
    # Create combined text field for embedding
    df["text"] = df["title"].astype(str) + " " + df["description"].astype(str)
    df["image_path"] = df["page"].apply(lambda x: f"data/page_images/page_{x}.png")

    text_emb = []
    img_emb = []

    print(f"Encoding text embeddings for {len(df)} items...")
    for idx, t in enumerate(df["text"]):
        if (idx + 1) % 10 == 0:
            print(f"  Progress: {idx + 1}/{len(df)}")
        text_emb.append(encode_text(t))

    print(f"Encoding image embeddings for {len(df)} items...")
    for idx, p in enumerate(df["image_path"]):
        if (idx + 1) % 10 == 0:
            print(f"  Progress: {idx + 1}/{len(df)}")
        try:
            img_emb.append(encode_image(p))
        except Exception as e:
            print(f"  Warning: Could not encode image {p}: {e}")
            # Use zero vector as fallback (3072 dimensions for image-embedding-3-large)
            img_emb.append([0.0] * 3072)

    pickle.dump(
        {"df": df, "text_embeddings": text_emb, "image_embeddings": img_emb},
        open(OUT, "wb")
    )

    print("Saved embeddings to", OUT)


if __name__ == "__main__":
    build_embeddings()
