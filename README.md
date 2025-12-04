# TVH Product Findability System

A comprehensive multimodal search and recommendation system for TVH's labels and signs catalog. This system enables users to find products using natural language queries or images, and provides intelligent recommendations based on frequently-bought-together patterns.

## Overview

This system solves the "findability" challenge for TVH's extensive catalog of 50 million products. The demo focuses on the labels and signs category, demonstrating:

- Semantic Search: Understand natural language queries to find relevant products
- Hybrid Search: Combines semantic similarity with keyword matching for better results
- Image Search: Find products by uploading images
- Recommendations: Suggest related products based on co-purchase patterns

## Requirements

- Python 3.8 or higher
- Docker and Docker Compose
- OpenAI API Key

## Quick Start with Docker Compose

### Step 1: Set Up Environment

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your-openai-api-key-here
```

### Step 2: Run with Docker Compose

```bash
docker-compose up --build
```

This will:
- Build the Docker image with all dependencies
- Start the Streamlit application
- Mount the data directory for persistence
- Expose the application on port 8501

The application will be available at `http://localhost:8501`

### Step 3: Run Pipeline Steps (First Time Setup)

If this is the first time running, you need to process the catalog data. Run these commands in separate terminals or after the container is running:

```bash
docker-compose exec tvh-finder python3 extract_catalog.py
docker-compose exec tvh-finder python3 generate_descriptions.py
docker-compose exec tvh-finder python3 build_embeddings.py
docker-compose exec tvh-finder python3 generate_copurchase.py
```

Or use the automated pipeline runner:

```bash
docker-compose exec tvh-finder python3 main.py --run-pipelines
```

### Docker Commands

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

Run in detached mode:
```bash
docker-compose up -d
```

## Local Development

### Step 1: Install Dependencies

```bash
pip3 install -r requirements.txt
```

### Step 2: Set Up Environment

```bash
export OPENAI_API_KEY='your-openai-api-key-here'
```

Or create a `.env` file:
```bash
OPENAI_API_KEY=your-openai-api-key-here
```

### Step 3: Run Pipeline

Run all pipeline steps automatically:

```bash
python3 main.py --run-pipelines
```

Or run steps individually:

```bash
python3 extract_catalog.py
python3 generate_descriptions.py
python3 build_embeddings.py
python3 generate_copurchase.py
```

### Step 4: Launch Application

```bash
python3 main.py
```

Or directly:
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## Pipeline Steps

1. **Extract Catalog** (`extract_catalog.py`)
   - Extracts text, tables, and images from PDF catalog
   - Creates `data/catalog_clean.csv`

2. **Generate Descriptions** (`generate_descriptions.py`) - Optional
   - Generates rich product descriptions (200-500 words) using OpenAI LLM
   - Creates `data/catalog_with_descriptions.csv`
   - Options: `--max-items N`, `--start-from N`

3. **Build Embeddings** (`build_embeddings.py`)
   - Generates text and image embeddings using OpenAI
   - Creates `data/embeddings.pkl`
   - Automatically uses enhanced descriptions if available

4. **Generate Co-Purchase Data** (`generate_copurchase.py`) - Optional
   - Generates synthetic co-purchase relationships
   - Creates `data/co_purchase.csv` for recommendations

## Usage Guide

### Text Search

1. Select "Text Search" mode
2. Enter a natural language query
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
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── data/
    ├── 12594102_Labels_Decals.pdf
    ├── catalog_clean.csv
    ├── catalog_with_descriptions.csv
    ├── embeddings.pkl
    ├── co_purchase.csv
    └── page_images/
```

## Architecture

1. **Data Extraction**: Extracts product data from PDF catalogs
2. **Description Generation**: Uses OpenAI LLM to generate rich product descriptions
3. **Embedding Generation**: Creates semantic embeddings for text and images
4. **Search Engine**: Hybrid search combining semantic and keyword matching
5. **Recommendation System**: Frequently-bought-together recommendations
6. **User Interface**: Streamlit-based interactive interface

## Troubleshooting

### Embeddings not found
- Run `python3 build_embeddings.py` or use `python3 main.py --run-pipelines`

### OpenAI API Error
- Check that `OPENAI_API_KEY` is set correctly in `.env` file
- Verify you have API credits
- Check your internet connection

### No recommendations available
- Run `python3 generate_copurchase.py` to create recommendation data

### Docker issues
- Ensure Docker and Docker Compose are installed
- Check that port 8501 is not already in use
- Verify `.env` file exists with `OPENAI_API_KEY` set
- Check logs with `docker-compose logs`

### Port already in use
- Change port in `docker-compose.yml`: `"8502:8501"`
- Update main.py: `python3 main.py --port 8502`

## Performance Notes

- Current Demo: Processes all products in memory
- Search Latency: ~1-3 seconds (depends on OpenAI API)
- Scalability: Demo handles hundreds of products; production architecture supports millions

## Security Notes

- API Keys: Never commit API keys to version control
- Use `.env` file for local development
- Data: Catalog data is public; no sensitive information

