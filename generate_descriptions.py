"""
Generate enhanced product descriptions using LLM (OpenAI).

This module takes the extracted catalog data and generates rich,
detailed descriptions (200-500 words) for each product using OpenAI's
GPT models. The descriptions are based on product titles and any
available metadata.
"""
import pandas as pd
import openai
import os
import time
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

# Configuration
CATALOG_INPUT = "data/catalog_clean.csv"
CATALOG_OUTPUT = "data/catalog_with_descriptions.csv"
OPENAI_MODEL = "gpt-4o-mini"  # Using gpt-4o-mini for cost efficiency, can use gpt-4o for better quality
MAX_RETRIES = 3
DELAY_BETWEEN_REQUESTS = 0.5  # Seconds to wait between API calls

openai.api_key = os.getenv("OPENAI_API_KEY")


def generate_product_description(title: str, existing_desc: Optional[str] = None, 
                                  product_id: Optional[str] = None, 
                                  page: Optional[int] = None) -> str:
    """
    Generate a detailed product description using OpenAI LLM.
    
    Args:
        title: Product title
        existing_desc: Existing description (if any) to use as context
        product_id: Product ID (optional context)
        page: Page number (optional context)
    
    Returns:
        Generated description (200-500 words)
    """
    # Build context for the LLM
    context_parts = []
    
    if title:
        context_parts.append(f"Product Title: {title}")
    
    if existing_desc and len(str(existing_desc).strip()) > 10:
        # Use existing description as context but ask for enhancement
        context_parts.append(f"Existing Information: {str(existing_desc)[:500]}")
    
    if product_id:
        context_parts.append(f"Product ID: {product_id}")
    
    context = "\n".join(context_parts)
    
    # Create the prompt
    prompt = f"""You are a product description writer for TVH, a global supplier of parts for forklifts, industrial vehicles, construction and agricultural machinery.

Based on the following product information, write a comprehensive, professional product description for a label or decal product:

{context}

Requirements:
- Write 200-500 words
- Be professional and informative
- Include relevant details about:
  * Product purpose and use cases
  * Typical applications (forklifts, industrial vehicles, etc.)
  * Material properties (if applicable)
  * Installation/mounting information (if applicable)
  * Safety or compliance information (if applicable)
  * Color, size, or other distinguishing features
- Write in clear, technical but accessible language
- Focus on helping customers understand when and where to use this product
- If the product is a label or decal, mention typical placement locations

Write only the product description, no additional text or formatting."""

    # Call OpenAI API with retries
    for attempt in range(MAX_RETRIES):
        try:
            # Use OpenAI client (works with both old and new API styles)
            try:
                # Try new API style first
                client = openai.OpenAI(api_key=openai.api_key)
                response = client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a professional product description writer specializing in industrial parts and equipment."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=800  # Enough for 200-500 words
                )
            except AttributeError:
                # Fall back to old API style if needed
                response = openai.ChatCompletion.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a professional product description writer specializing in industrial parts and equipment."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=800
                )
            
            description = response.choices[0].message.content.strip()
            
            # Validate word count
            word_count = len(description.split())
            if word_count < 150:
                # If too short, ask for more detail
                return generate_extended_description(title, existing_desc, product_id, description)
            elif word_count > 600:
                # If too long, truncate intelligently
                words = description.split()
                description = " ".join(words[:500])
            
            return description
            
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"  Error generating description (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                print(f"  Failed to generate description after {MAX_RETRIES} attempts: {e}")
                # Return a fallback description
                return create_fallback_description(title, existing_desc)
    
    return create_fallback_description(title, existing_desc)


def generate_extended_description(title: str, existing_desc: Optional[str], 
                                   product_id: Optional[str], 
                                   initial_desc: str) -> str:
    """Extend a description that's too short."""
    prompt = f"""The following is a product description that needs to be expanded to 200-500 words:

Title: {title}
Current Description: {initial_desc}

Please expand this description to be more detailed (200-500 words), adding:
- More specific use cases and applications
- Technical details about materials and construction
- Installation and mounting guidance
- Safety considerations
- Compatibility information

Write only the expanded description."""
    
    try:
        # Try new API style first
        try:
            client = openai.OpenAI(api_key=openai.api_key)
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=800
            )
        except AttributeError:
            # Fall back to old API style
            response = openai.ChatCompletion.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=800
            )
        return response.choices[0].message.content.strip()
    except:
        return initial_desc


def create_fallback_description(title: str, existing_desc: Optional[str]) -> str:
    """
    Create a fallback description if LLM generation fails.
    
    Args:
        title: Product title
        existing_desc: Existing description
    
    Returns:
        Fallback description
    """
    if existing_desc and len(str(existing_desc).strip()) > 50:
        return str(existing_desc)
    
    # Create a basic description from title
    base_desc = f"""This {title} is a professional-grade label or decal designed for industrial vehicles and equipment. 
    
Suitable for use on forklifts, construction machinery, and agricultural equipment, this product provides clear identification and safety information. The label is designed to withstand typical industrial environments and provides durable, long-lasting performance.

Installation is straightforward, and the product is compatible with standard mounting methods used in industrial vehicle applications. This product is part of TVH's comprehensive range of labels and decals for industrial equipment."""
    
    return base_desc


def enhance_catalog_with_descriptions(input_csv: str = CATALOG_INPUT, 
                                      output_csv: str = CATALOG_OUTPUT,
                                      max_items: Optional[int] = None,
                                      start_from: int = 0):
    """
    Enhance catalog with LLM-generated descriptions.
    
    Args:
        input_csv: Path to input catalog CSV
        output_csv: Path to output catalog CSV with descriptions
        max_items: Maximum number of items to process (None for all)
        start_from: Index to start from (for resuming)
    """
    print("Loading catalog...")
    df = pd.read_csv(input_csv)
    
    # Filter out rows without titles
    df = df[df["title"].notna() & (df["title"].astype(str).str.strip() != "")]
    
    total_items = len(df)
    if max_items:
        df = df[:max_items]
    
    print(f"Processing {len(df)} products (starting from index {start_from})...")
    print(f"Using OpenAI model: {OPENAI_MODEL}")
    print("=" * 60)
    
    # Check if output file exists (for resuming)
    if os.path.exists(output_csv) and start_from == 0:
        response = input(f"Output file {output_csv} exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    # Process each product
    descriptions = []
    for idx, row in df.iterrows():
        if idx < start_from:
            descriptions.append(row.get("description", ""))
            continue
        
        title = str(row.get("title", ""))
        existing_desc = row.get("description", "")
        product_id = row.get("product_id", "")
        page = row.get("page", "")
        
        print(f"[{idx + 1}/{len(df)}] Generating description for: {title[:50]}...")
        
        # Generate description
        new_description = generate_product_description(
            title=title,
            existing_desc=existing_desc if pd.notna(existing_desc) else None,
            product_id=str(product_id) if pd.notna(product_id) else None,
            page=int(page) if pd.notna(page) else None
        )
        
        descriptions.append(new_description)
        
        # Save progress periodically
        if (idx + 1) % 10 == 0:
            df_temp = df.copy()
            df_temp["description"] = descriptions + [row.get("description", "") for _ in range(len(df) - len(descriptions))]
            df_temp.to_csv(output_csv, index=False)
            print(f"  Progress saved (processed {idx + 1} items)")
        
        # Rate limiting
        time.sleep(DELAY_BETWEEN_REQUESTS)
    
    df["description"] = descriptions
    
    # Save final output
    df.to_csv(output_csv, index=False)
    print("=" * 60)
    print(f"Enhanced catalog saved to: {output_csv}")
    print(f"   Total products processed: {len(df)}")
    
    desc_lengths = [len(str(d).split()) for d in descriptions]
    avg_words = sum(desc_lengths) / len(desc_lengths) if desc_lengths else 0
    print(f"   Average description length: {avg_words:.1f} words")
    print(f"   Min length: {min(desc_lengths) if desc_lengths else 0} words")
    print(f"   Max length: {max(desc_lengths) if desc_lengths else 0} words")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate LLM-based product descriptions")
    parser.add_argument("--max-items", type=int, help="Maximum number of items to process")
    parser.add_argument("--start-from", type=int, default=0, help="Start from this index (for resuming)")
    parser.add_argument("--input", type=str, default=CATALOG_INPUT, help="Input CSV file")
    parser.add_argument("--output", type=str, default=CATALOG_OUTPUT, help="Output CSV file")
    
    args = parser.parse_args()
    
    if not openai.api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        print("   Set it with: export OPENAI_API_KEY='your-key-here'")
        return
    
    enhance_catalog_with_descriptions(
        input_csv=args.input,
        output_csv=args.output,
        max_items=args.max_items,
        start_from=args.start_from
    )


if __name__ == "__main__":
    main()

