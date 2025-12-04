# TVH Product Findability System

A comprehensive multimodal search and recommendation system for TVH's labels and signs catalog. This system enables users to find products using natural language queries or images, and provides intelligent recommendations based on frequently-bought-together patterns.

## Overview

This system solves the "findability" challenge for TVH's extensive catalog of 50 million products. The demo focuses on the labels and signs category, demonstrating:

- Semantic Search: Understand natural language queries to find relevant products
- Hybrid Search: Combines semantic similarity with keyword matching for better results
- Image Search: Find products by uploading images
- Recommendations: Suggest related products based on co-purchase patterns

## Requirements

### Software Requirements
- Python 3.8 or higher
- pip3 (Python package manager)
- Docker and Docker Compose (optional, for containerized deployment)

### API Keys
- OpenAI API Key: Required for generating embeddings
  - Sign up at https://platform.openai.com/
  - Get your API key from https://platform.openai.com/api-keys
  - Set as environment variable: `export OPENAI_API_KEY='your-key-here'`

### Python Packages
Install required packages:
```bash
pip3 install -r requirements.txt
```

## Quick Start

### Option 1: Local Development

#### Step 1: Set Up Environment

```bash
export OPENAI_API_KEY='your-openai-api-key-here'
cd /home/lnv221/mine/TVH
```

#### Step 2: Extract Catalog Data

```bash
python3 extract_catalog.py
```

This extracts text and tables from the PDF catalog, saves page images, and creates `data/catalog_clean.csv`.

#### Step 3: Generate LLM-Enhanced Descriptions (Recommended)

```bash
python3 generate_descriptions.py
```

This generates rich product descriptions (200-500 words) using OpenAI's LLM and creates `data/catalog_with_descriptions.csv`.

Options:
- `--max-items N`: Process only first N items (useful for testing)
- `--start-from N`: Resume from index N (if interrupted)
- `--input FILE`: Specify input CSV (default: `data/catalog_clean.csv`)
- `--output FILE`: Specify output CSV (default: `data/catalog_with_descriptions.csv`)

Note: This step requires OpenAI API access and may take time depending on the number of products.

#### Step 4: Build Embeddings

```bash
python3 build_embeddings.py
```

This generates text and image embeddings for all products and saves them to `data/embeddings.pkl`.

Note: This step requires OpenAI API access and may take time depending on the number of products.

#### Step 5: Generate Co-Purchase Data (Optional but Recommended)

```bash
python3 generate_copurchase.py
```

This generates synthetic co-purchase relationships and creates `data/co_purchase.csv` for recommendations.

#### Step 6: Launch Application

```bash
python3 main.py
```

Or directly:
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### Option 2: Docker Deployment

#### Prerequisites
- Docker installed
- Docker Compose installed

#### Step 1: Set Up Environment

Create a `.env` file in the project root:
```bash
OPENAI_API_KEY=your-openai-api-key-here
```

#### Step 2: Build and Run with Docker Compose

```bash
docker-compose up --build
```

This will:
- Build the Docker image with all dependencies
- Start the Streamlit application
- Mount the data directory for persistence
- Expose the application on port 8501

The application will be available at `http://localhost:8501`

#### Step 3: Run Setup Scripts in Container (if needed)

If you need to run data processing scripts:

```bash
docker-compose exec tvh-finder python3 extract_catalog.py
docker-compose exec tvh-finder python3 generate_descriptions.py
docker-compose exec tvh-finder python3 build_embeddings.py
docker-compose exec tvh-finder python3 generate_copurchase.py
```

#### Docker Commands

Stop the application:
```bash
docker-compose down
```

View logs:
```bash
docker-compose logs -f
```

Rebuild after code changes:
```bash
docker-compose up --build
```

## Project Structure

```
TVH/
├── app.py
├── query.py
├── recommender.py
├── build_embeddings.py
├── extract_catalog.py
├── generate_descriptions.py
├── generate_copurchase.py
├── main.py
├── tvh_findability_demo.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
├── PRODUCTIONALIZATION.md
└── data/
    ├── 12594102_Labels_Decals.pdf
    ├── catalog_clean.csv
    ├── catalog_with_descriptions.csv
    ├── catalog_raw.json
    ├── embeddings.pkl
    ├── co_purchase.csv
    └── page_images/
```

## Usage Guide

### Text Search

1. Select "Text Search" mode
2. Enter a natural language query, e.g.:
   - "yellow caution label for hydraulic systems"
   - "safety warning decal"
   - "reflective strip for trailer"
3. Click "Search"
4. Review results with relevance scores
5. Click on any product to see recommendations

### Image Search

1. Select "Image Search" mode
2. Upload an image file (PNG, JPG, JPEG)
3. Click "Search"
4. View visually similar products

### Recommendations

- Recommendations appear when you expand a product card
- Based on frequently-bought-together patterns
- Shows related products with match scores

### Configuration Options

In the sidebar, you can adjust:
- Search Mode: Text or Image
- Hybrid Search: Enable/disable semantic + keyword combination
- Semantic vs Keyword Weight: Balance between semantic and keyword matching
- Number of Results: How many results to display

## Architecture

### Components

1. Data Extraction (`extract_catalog.py`)
   - Extracts text, tables, and images from PDF
   - Normalizes product data
   - Creates structured catalog

2. Description Generation (`generate_descriptions.py`)
   - Uses OpenAI's LLM (GPT-4o-mini) to generate rich product descriptions
   - Creates 200-500 word descriptions for each product
   - Enhances product information with professional, detailed content
   - Optional but recommended for better search quality

3. Embedding Generation (`build_embeddings.py`)
   - Uses OpenAI's embedding models
   - Creates vector representations for text and images
   - Enables semantic similarity search
   - Automatically uses enhanced descriptions if available

4. Search Engine (`query.py`)
   - Semantic search using vector similarity
   - Hybrid search combining semantic + keyword matching
   - Image-based search

5. Recommendation System (`recommender.py`)
   - Loads co-purchase relationships
   - Generates product recommendations
   - Scores recommendations by relevance

6. User Interface (`app.py`)
   - Streamlit-based interactive interface
   - Product display with images and rich descriptions
   - Recommendation integration

## Approach & Technology Decisions

### Search Approach

Hybrid Search: Combines semantic embeddings with keyword matching
- Semantic: Understands meaning and context (e.g., "caution" matches "warning")
- Keyword: Exact matches for specific terms (e.g., product codes, colors)
- Weighted Combination: Configurable balance between the two

### Technology Choices

- Python: Rapid development, rich ML ecosystem
- OpenAI Embeddings: High-quality semantic understanding
- Streamlit: Fast UI prototyping and demos
- Vector Search: Cosine similarity for fast retrieval

### Why This Approach?

1. Natural Language Understanding: Users can describe scenarios, not just keywords
2. Multimodal: Supports both text and image queries
3. Scalable: Vector search can scale to millions of products
4. Flexible: Hybrid approach handles both semantic and exact queries

## Productionalization

See `PRODUCTIONALIZATION.md` for detailed plans on:
- Scaling to 50M products
- Vector database selection (FAISS, Pinecone, Weaviate)
- API design and deployment
- Cost optimization
- Monitoring and observability
- Integration with TVH systems

## Troubleshooting

### "Embeddings not found"
- Run `python3 build_embeddings.py` to generate embeddings

### "OpenAI API Error"
- Check that `OPENAI_API_KEY` is set correctly
- Verify you have API credits
- Check your internet connection

### "No recommendations available"
- Run `python3 generate_copurchase.py` to create recommendation data

### Slow search performance
- This is expected in the demo (in-memory search)
- Production would use vector databases (FAISS, Pinecone) for speed

### Docker issues
- Ensure Docker and Docker Compose are installed and running
- Check that port 8501 is not already in use
- Verify `.env` file exists with `OPENAI_API_KEY` set
- Check logs with `docker-compose logs`

## Performance Notes

- Current Demo: Processes all products in memory
- Search Latency: ~1-3 seconds (depends on OpenAI API)
- Scalability: Demo handles hundreds of products; production architecture supports millions

## Security Notes

- API Keys: Never commit API keys to version control
- Data: Catalog data is public; no sensitive information
- Production: Implement authentication, rate limiting, and input validation

## Code Quality

- Total Lines: 1500+ lines of Python code
- Documentation: Comprehensive comments and docstrings
- Modularity: Well-organized, reusable components
- Best Practices: Follows Python best practices and security guidelines

## How This Helps TVH

1. Contact Center: Agents can quickly find products from customer descriptions
2. Customer Self-Service: Better search on website reduces support load
3. Recommendations: Increase basket size and cross-selling
4. Scalability: Architecture supports growth to 50M products

## Support

For questions or issues:
- Review the code comments for implementation details
- Check `PRODUCTIONALIZATION.md` for production considerations
- Ensure all setup steps are completed

## License

This is a demo project for TVH technical assessment.
