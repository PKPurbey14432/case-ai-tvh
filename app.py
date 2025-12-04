"""
TVH Product Finder - Main Streamlit Application

This is the main application for the TVH Findability demo. It provides:
- Text-based semantic search with hybrid (semantic + keyword) matching
- Image-based search
- Product recommendations based on frequently-bought-together data
- Interactive UI for exploring products and recommendations

Run with: streamlit run app.py
"""
import streamlit as st
from query import search_text, search_image
from recommender import recommend, get_recommended_product_ids
import pandas as pd
from PIL import Image
import os
import pickle

# Page configuration
st.set_page_config(
    page_title="TVH Product Finder",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .product-card {
        padding: 1rem;
        border: 1px solid #ddd;
        border-radius: 8px;
        margin: 1rem 0;
        background-color: #f9f9f9;
    }
    .recommendation-item {
        padding: 0.5rem;
        margin: 0.5rem 0;
        background-color: #e8f4f8;
        border-left: 3px solid #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)


def load_catalog_data():
    """Load catalog data from embeddings file."""
    try:
        with open("data/embeddings.pkl", "rb") as f:
            data = pickle.load(f)
        return data["df"]
    except FileNotFoundError:
        st.error("Embeddings file not found. Please run build_embeddings.py first.")
        st.stop()


def get_product_by_id(product_id, catalog_df):
    """
    Get product details by product ID.
    
    Args:
        product_id: Product ID to search for (can be string or number)
        catalog_df: Catalog DataFrame
    
    Returns:
        Product row if found, None otherwise
    """
    if product_id is None or (isinstance(product_id, str) and product_id.strip() == ""):
        return None
    
    # Try exact match first (as string)
    matches = catalog_df[catalog_df["product_id"].astype(str) == str(product_id)]
    if not matches.empty:
        return matches.iloc[0]
    
    # Try case-insensitive match
    matches = catalog_df[catalog_df["product_id"].astype(str).str.upper() == str(product_id).upper()]
    if not matches.empty:
        return matches.iloc[0]
    
    return None


def display_product_card(row, score=None, show_image=True):
    """
    Display a product card with details.
    
    Args:
        row: DataFrame row with product information
        score: Optional relevance score to display
        show_image: Whether to show product image
    """
    with st.container():
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if show_image and "image_path" in row and os.path.exists(row["image_path"]):
                try:
                    img = Image.open(row["image_path"])
                    st.image(img, width=300, use_container_width=True)
                except Exception as e:
                    st.write("Page Image")
            else:
                st.write("Page Image")
        
        with col2:
            # Product title
            title = str(row.get("title", "N/A"))[:100]  # Truncate long titles
            st.subheader(title)
            
            # Product ID and page
            col_id, col_page = st.columns(2)
            with col_id:
                if pd.notna(row.get("product_id")):
                    st.write(f"**Product ID:** `{row['product_id']}`")
            with col_page:
                st.write(f"**Page:** {row.get('page', 'N/A')}")
            
            # Score if provided
            if score is not None:
                st.metric("Relevance Score", f"{score:.4f}")
            
            # Description
            description = str(row.get("description", ""))[:500]  # Truncate long descriptions
            if description and description != "nan":
                with st.expander("View Description"):
                    st.write(description)
            
            st.divider()


def display_recommendations(product_id, catalog_df, top_n=5):
    """
    Display recommended products for a given product ID.
    
    Args:
        product_id: Product ID to get recommendations for
        catalog_df: Catalog DataFrame
        top_n: Number of recommendations to show
    """
    st.markdown("### Recommended Products (Frequently Bought Together)")
    
    if product_id is None or (isinstance(product_id, str) and product_id.strip() == ""):
        st.info("Product ID is missing. Cannot generate recommendations.")
        return
    
    # Convert to string for matching
    product_id_str = str(product_id).strip()
    
    # Get recommendations
    try:
        recs = recommend(product_id_str, top_k=top_n)
    except Exception as e:
        st.error(f"Error getting recommendations: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return
    
    if recs.empty:
        st.info(f"No recommendations available for product ID: `{product_id_str}`")
        st.caption("This product may not have co-purchase relationships in the dataset.")
        
        try:
            import pandas as pd
            copurchase_df = pd.read_csv("data/co_purchase.csv")
            available_ids = copurchase_df["product_id"].unique()[:10]
            with st.expander("🔍 Debug: Products with recommendations available"):
                st.write(f"Sample product IDs with recommendations: {', '.join(map(str, available_ids))}")
                st.caption("Try searching for one of these product IDs to see recommendations.")
        except:
            pass
        return
    
    # Display recommendations
    for idx, rec_row in recs.iterrows():
        related_id = rec_row["related_product_id"]
        score = rec_row["score"]
        
        # Get product details
        product = get_product_by_id(related_id, catalog_df)
        
        if product is not None:
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    title = str(product.get("title", "N/A"))[:80]
                    st.write(f"**{title}**")
                    st.write(f"Product ID: `{related_id}` | Page: {product.get('page', 'N/A')}")
                with col2:
                    st.metric("Match Score", f"{score:.2f}")
                st.divider()
        else:
            st.write(f"Product ID: `{related_id}` (details not found)")


# Main application
def main():
    """Main application function."""
    st.title("TVH Product Finder")
    st.markdown("**Multimodal Search & Recommendations for Labels & Signs Catalog**")
    st.markdown("---")
    
    catalog_df = load_catalog_data()
    
    with st.sidebar:
        st.header("Configuration")
        
        search_mode = st.radio(
            "Search Mode",
            ["Text Search", "Image Search"],
            help="Choose between text-based or image-based search"
        )
        
        if search_mode == "Text Search":
            use_hybrid = st.checkbox(
                "Enable Hybrid Search",
                value=True,
                help="Combine semantic and keyword matching for better results"
            )
            if use_hybrid:
                alpha = st.slider(
                    "Semantic vs Keyword Weight",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.7,
                    step=0.1,
                    help="Higher values favor semantic similarity, lower values favor keyword matching"
                )
            else:
                alpha = 1.0
        
        top_k = st.slider(
            "Number of Results",
            min_value=1,
            max_value=20,
            value=5,
            help="Number of search results to display"
        )
        
        st.markdown("---")
        st.markdown("### Statistics")
        st.write(f"**Total Products:** {len(catalog_df)}")
        st.write(f"**Total Pages:** {catalog_df['page'].nunique() if 'page' in catalog_df.columns else 'N/A'}")
        
        valid_ids = catalog_df[catalog_df['product_id'].notna() & 
                              (catalog_df['product_id'].astype(str).str.strip() != '')]
        st.write(f"**Products with IDs (for recommendations):** {len(valid_ids)}")
        
        if len(valid_ids) < len(catalog_df) * 0.5:
            st.warning("Many products don't have product IDs. Recommendations will only work for products with valid IDs.")
    
    # Main search interface
    if search_mode == "Text Search":
        st.markdown("### Text Search")
        st.markdown("Describe what you're looking for in natural language.")
        st.markdown("*Example: 'yellow caution label for hydraulic systems'*")
        
        query = st.text_input(
            "Enter your search query:",
            placeholder="e.g., warning label, safety decal, hydraulic label..."
        )
        
        if st.button("Search", type="primary", use_container_width=True):
            if query and query.strip():
                with st.spinner("Searching products..."):
                    try:
                        results, scores = search_text(
                            query,
                            top_k=top_k,
                            hybrid=use_hybrid if 'use_hybrid' in locals() else True,
                            alpha=alpha if 'alpha' in locals() else 0.7
                        )
                        
                        st.success(f"Found {len(results)} results")
                        st.markdown("---")
                        
                        for (idx, row), score in zip(results.iterrows(), scores):
                            display_product_card(row, score=score)
                            
                            product_id = row.get("product_id")
                            if (pd.notna(product_id) and 
                                str(product_id).strip() != "" and 
                                str(product_id).lower() != "nan"):
                                with st.expander(f"See recommendations for {product_id}"):
                                    display_recommendations(product_id, catalog_df, top_n=3)
                            
                            st.markdown("---")
                    
                    except Exception as e:
                        st.error(f"Error during search: {str(e)}")
                        st.info("Make sure you have set the OPENAI_API_KEY environment variable.")
            else:
                st.warning("Please enter a search query.")
    
    elif search_mode == "Image Search":
        st.markdown("### Image Search")
        st.markdown("Upload an image of a product to find similar items.")
        
        uploaded_file = st.file_uploader(
            "Upload an image:",
            type=["png", "jpg", "jpeg"],
            help="Upload an image of a label, decal, or sign"
        )
        
        if uploaded_file and st.button("Search", type="primary", use_container_width=True):
            with st.spinner("Analyzing image and searching products..."):
                try:
                    results, scores = search_image(uploaded_file, top_k=top_k)
                    
                    st.success(f"Found {len(results)} similar products")
                    st.markdown("---")
                    
                    # Display results
                    for (idx, row), score in zip(results.iterrows(), scores):
                        display_product_card(row, score=score)
                        
                        product_id = row.get("product_id")
                        if (pd.notna(product_id) and 
                            str(product_id).strip() != "" and 
                            str(product_id).lower() != "nan"):
                            with st.expander(f"See recommendations for {product_id}"):
                                display_recommendations(product_id, catalog_df, top_n=3)
                        
                        st.markdown("---")
                
                except Exception as e:
                    st.error(f"Error during image search: {str(e)}")
                    st.info("Make sure you have set the OPENAI_API_KEY environment variable.")
    
    # Footer with information
    st.markdown("---")
    with st.expander("ℹAbout This Demo"):
        st.markdown("""
        ### TVH Product Findability Demo
        
        This application demonstrates a multimodal search and recommendation system for TVH's 
        labels and signs catalog. It uses:
        
        - **Semantic Search**: OpenAI embeddings for understanding product descriptions
        - **Hybrid Search**: Combines semantic similarity with keyword matching
        - **Image Search**: Visual similarity search using image embeddings
        - **Recommendations**: Frequently-bought-together product suggestions
        
        #### How It Works:
        1. **Text Search**: Enter a natural language description of what you need
        2. **Image Search**: Upload an image to find visually similar products
        3. **Recommendations**: Click on any product to see related items
        
        #### Production Considerations:
        - Vector database (FAISS, Pinecone, Weaviate) for scalable search
        - Real transaction data for recommendation training
        - Caching and optimization for low-latency responses
        - Multilingual support for global markets
        - Integration with TVH's existing systems
        """)


if __name__ == "__main__":
    main()
