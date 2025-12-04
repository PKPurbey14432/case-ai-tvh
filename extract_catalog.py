import pdfplumber
import pandas as pd
import json
import re
from pathlib import Path

PDF_PATH = "data/12594102_Labels_Decals.pdf"
RAW_JSON_OUT = "data/catalog_raw.json"
CSV_OUT = "data/catalog_clean.csv"
IMG_DIR = Path("data/page_images")
IMG_DIR.mkdir(parents=True, exist_ok=True)



def clean_text(text):
    if not text:
        return ""
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_product_id(text):
    """Detect TVH-like ID or label codes"""
    if not text:
        return None

    patterns = [
        r"[A-Z0-9]{4,15}",              # Generic IDs
        r"\b\d{2}-\d{2}-\d{2}-\d{2}\b", # Group codes
    ]
    for p in patterns:
        m = re.search(p, text)
        if m:
            return m.group()
    return None


def parse_table(df, page_no):
    """
    Process a table DataFrame into structured product entries.
    """
    parsed_rows = []

    for idx, row in df.iterrows():
        row_dict = {}
        for col in df.columns:
            val = str(row[col]).strip()
            if val.lower() == "nan":
                val = ""
            row_dict[col] = clean_text(val)

        product_id = detect_product_id(" ".join(row_dict.values()))

        parsed_rows.append({
            "page": page_no,
            "type": "table",      # FIXED
            "product_id": product_id,
            "raw": row_dict
        })

    return parsed_rows


def extract_pdf():
    print("Extracting from PDF:", PDF_PATH)
    pdf = pdfplumber.open(PDF_PATH)

    all_items = []

    for page_no, page in enumerate(pdf.pages, start=1):
        print(f"Processing page {page_no}/{len(pdf.pages)}")

        img_path = IMG_DIR / f"page_{page_no}.png"
        page.to_image(resolution=150).save(img_path)

        all_items.append({
            "page": page_no,
            "type": "page_image",
            "image_path": str(img_path)
        })

        text = clean_text(page.extract_text())

        if text:
            all_items.append({
                "page": page_no,
                "type": "text",
                "product_id": detect_product_id(text),
                "content": text
            })

        try:
            tables = page.extract_tables()

            for table in tables:
                df = pd.DataFrame(table[1:], columns=table[0])
                parsed = parse_table(df, page_no)
                all_items.extend(parsed)

        except Exception as e:
            print("Table extraction error:", e)

    pdf.close()

    # SAVE RAW
    with open(RAW_JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(all_items, f, indent=2)

    return all_items


def normalize(raw_items):
    cleaned = []

    for item in raw_items:

        # Skip image entries
        if item["type"] == "page_image":
            continue

        # ----- TEXT -----
        if item["type"] == "text":
            cleaned.append({
                "page": item["page"],
                "product_id": item["product_id"],
                "title": item["content"][:60],
                "description": item["content"]
            })

        # ----- TABLE -----
        if item["type"] == "table":
            desc = []
            title = None

            for k, v in item["raw"].items():
                if not title and len(v) > 3:
                    title = v
                desc.append(f"{k}: {v}")

            cleaned.append({
                "page": item["page"],
                "product_id": item["product_id"],
                "title": title,
                "description": " | ".join(desc)
            })

    df = pd.DataFrame(cleaned)
    df.to_csv(CSV_OUT, index=False)
    print("Saved:", CSV_OUT)
    return df

if __name__ == "__main__":
    raw_items = extract_pdf()
    normalize(raw_items)
