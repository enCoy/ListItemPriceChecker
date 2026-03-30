import os
import base64
import json
from io import BytesIO
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from pdf2image import convert_from_path
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
import os
import requests

GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY")


class PriceHistory(BaseModel):
    latest: float
    all_prices: List[float]

def unpack_price_entry(entry):
    """Return (latest_price, all_prices_list) from PriceHistory or legacy dict/float."""
    if isinstance(entry, PriceHistory):
        return entry.latest, entry.all_prices
    elif isinstance(entry, dict):
        return entry.get("latest"), entry.get("all_prices", [])
    else:  # legacy float
        return entry, [entry]

# List models (v1beta endpoint)
url = "https://generativelanguage.googleapis.com/v1beta/models"
resp = requests.get(url, params={"key": GEMINI_API_KEY}, timeout=30)

print("HTTP", resp.status_code)
try:
    data = resp.json()
except ValueError:
    print("Non-JSON response:", resp.text)
else:
    print(json.dumps(data, indent=2))

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
# ==========================================
# 1. Define Structured Output (Pydantic)
# ==========================================
# We define exactly what we want the VLM to find.
# This ensures the LLM's output can be reliably parsed into our state.

class AmazonItem(BaseModel):
    title: str = Field(description="The exact title of the book or item")
    price: Optional[float] = Field(description="The current numeric price, without currency symbols. e.g., 29.99")
    quantity: Optional[int] = Field(description="The quantity listed in the cart")


class AmazonListExtraction(BaseModel):
    """A list of extracted items from an Amazon PDF list."""
    items: List[AmazonItem]


# We bind the structured output to the LLM
structured_vlm = llm.with_structured_output(AmazonListExtraction)

# ==========================================
# 2. Update LangGraph State
# ==========================================

class PdfTrackerState(BaseModel):
    pdf_path: str
    historical_prices: Dict[str, PriceHistory] = Field(default_factory=dict)
    extracted_items: List[AmazonItem] = Field(default_factory=list)
    discount_threshold: float = 0.15
    discount_alerts: List[str] = Field(default_factory=list)
    historical_file: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True

# ==========================================
# 3. Node 1: Multimodal Extraction
# ==========================================

def get_image_base64(image):
    """Helper to convert PIL image to base64 for LLM input."""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def normalize_entry(entry):
    """Ensure entry is always {'latest': float, 'all_prices': List[float]}"""

    if isinstance(entry, PriceHistory):
        latest = entry.latest
        all_prices = entry.all_prices

    elif isinstance(entry, dict):
        latest = entry.get("latest")

        raw_prices = entry.get("all_prices", [])

        # 🔴 CRITICAL FIX: flatten nested structures
        all_prices = []
        for p in raw_prices:
            if isinstance(p, dict):
                # If nested PriceHistory-like dict → extract latest
                if "latest" in p:
                    all_prices.append(p["latest"])
            else:
                all_prices.append(p)

    else:  # legacy float
        latest = entry
        all_prices = [entry]

    return {
        "latest": latest,
        "all_prices": all_prices
    }

def extract_prices_from_pdf_vision(state: PdfTrackerState) -> PdfTrackerState:
    pdf_path = getattr(state, "pdf_path", None)
    print(f"--- PARSING PDF WITH VISION: {pdf_path} ---")

    if not pdf_path or not os.path.exists(pdf_path):
        print(f"Error: File not found at {pdf_path}")
        return {"extracted_items": []}

    # A. Convert PDF pages to PIL Images
    print("Converting PDF to images...")
    try:
        pages = convert_from_path(pdf_path)
    except Exception as e:
        print(f"Poppler Error (ensure it's installed): {e}")
        return {"extracted_items": []}

    # B. Prepare messages for the VLM
    # We create a message containing text instructions AND the images.
    content = [
        {"type": "text",
         "text": "I am providing images of my printed Amazon cart/list. Extract every item, including its title, current price, and quantity. Ignore items with no price (out of stock). Return structured JSON."}
    ]

    # Add each page image to the prompt
    for i, page in enumerate(pages):
        print(f"Adding page {i + 1} to prompt...")
        base64_image = get_image_base64(page)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
        })

    # C. Invoke the VLM and get structured data
    print("Invoking Vision LLM (this might take a moment)...")
    message = HumanMessage(content=content)
    raw_result = structured_vlm.invoke([message])

    # LangChain structured output returns the Pydantic object directly.
    # Return the partial state expected by the graph runtime.
    return {"extracted_items": raw_result.items}


# ==========================================
# 4. Node 2: The Analyst (Title Matching)
# ==========================================

def analyze_discounts(state: PdfTrackerState) -> PdfTrackerState:
    print("--- ANALYZING FOR DISCOUNTS ---")
    historical = getattr(state, "historical_prices", {}) or {}
    current_items = getattr(state, "extracted_items", []) or []
    alerts = []

    for item in current_items:
        title = item.title

        if item.price is None:
            continue

        new_price = item.price

        if title in historical:
            entry = historical[title]
            old_price, all_prices = unpack_price_entry(entry)

            if old_price is None or not all_prices:
                continue

            min_price = min(all_prices)
            max_price = max(all_prices)
            is_new_price = new_price not in all_prices

            if not all_prices:
                continue

            min_price = min(all_prices)
            max_price = max(all_prices)
            is_new_price = new_price not in all_prices

            # --- Core logic (no thresholds) ---

            # Case 1: New lowest ever observed
            if is_new_price and new_price < min_price:
                alerts.append(
                    f"DEAL ALERT: '{title}' has a NEW LOWEST price ever: ${new_price:.2f} (prev min: ${min_price:.2f})"
                )

            # Case 2: Matches historical minimum
            elif (not is_new_price) and new_price == min_price:
                alerts.append(
                    f"At previously observed minimum: '{title}' is at ${new_price:.2f}"
                )

            # Optional: new highest ever (symmetric signal)
            elif is_new_price and new_price > max_price:
                alerts.append(
                    f"New highest price observed: '{title}' is now ${new_price:.2f} (prev max: ${max_price:.2f})"
                )

            # Optional: simple direction signal (no threshold)
            elif old_price is not None:
                if new_price < old_price:
                    alerts.append(
                        f"Price drop: '{title}' (${old_price:.2f} → ${new_price:.2f})"
                    )
                elif new_price > old_price:
                    alerts.append(
                        f"Price increase: '{title}' (${old_price:.2f} → ${new_price:.2f})"
                    )
    return {"discount_alerts": alerts}


# ==========================================
# 5. Node 3: The Reporter
# ==========================================

def report_findings(state: PdfTrackerState):
    print("--- FINAL REPORT ---")

    alerts = getattr(state, "discount_alerts", []) or []

    if not alerts:
        print("No alerts.")
        return state

    print(f"{len(alerts)} alert(s) found:\n")
    for alert in alerts:
        print(alert)

    return state

def make_serializable(obj):
    """Recursively convert Pydantic models (PriceHistory) into dicts for JSON."""
    if isinstance(obj, BaseModel):
        return obj.dict()
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    else:
        return obj

# New: Node to persist historical prices after each run
def save_historical_prices(state: PdfTrackerState):
    """Update per-PDF historical file with the latest observed prices from extracted_items."""
    print("--- SAVING HISTORICAL PRICES ---")

    # Determine historical file
    historical_file = getattr(state, "historical_file", None)
    if not historical_file:
        try:
            base = os.path.splitext(os.path.basename(getattr(state, "pdf_path")))[0]
            historical_file = os.path.join(os.path.dirname(getattr(state, "pdf_path")), f"{base}_historical_prices.json")
        except Exception:
            historical_file = "historical_prices.json"

    historical = getattr(state, "historical_prices", {}) or {}
    items = getattr(state, "extracted_items", []) or []

    updated = False
    for item in items:
        title = getattr(item, "title", None)
        price = getattr(item, "price", None)

        if title and (price is not None):
            if title not in historical:
                historical[title] = {
                    "latest": price,
                    "all_prices": [price]
                }
                updated = True
            else:
                entry = normalize_entry(historical[title])

                # Add new price if unique
                if price not in entry["all_prices"]:
                    entry["all_prices"].append(price)
                    updated = True

                # Always update latest
                if entry["latest"] != price:
                    entry["latest"] = price
                    updated = True

                historical[title] = entry

    # --- ADD THIS HELPER FUNCTION INSIDE THE SAVE FUNCTION ---
    def make_serializable(obj):
        """Recursively convert Pydantic models to dicts for JSON."""
        if isinstance(obj, BaseModel):
            return obj.dict()
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        else:
            return obj

    # Convert historical to fully JSON-serializable form
    serializable_historical = make_serializable(historical)

    try:
        with open(historical_file, "w", encoding="utf-8") as f:
            json.dump(serializable_historical, f, indent=2)

        if updated:
            print(f"Wrote {len([i for i in items if getattr(i, 'price', None) is not None])} prices to {historical_file}")
        else:
            print(f"No price changes; {historical_file} left with current values.")
    except Exception as e:
        print(f"Error saving historical prices to {historical_file}: {e}")

    # Return the updated portion of the state
    return {"historical_prices": historical}


# ==========================================
# 6. Build and Run the Graph
# ==========================================

workflow = StateGraph(PdfTrackerState)

# Add Nodes
workflow.add_node("parser", extract_prices_from_pdf_vision)
workflow.add_node("analyst", analyze_discounts)
workflow.add_node("reporter", report_findings)
workflow.add_node("saver", save_historical_prices)

# Add Edges
workflow.set_entry_point("parser")
workflow.add_edge("parser", "analyst")
workflow.add_edge("analyst", "reporter")
workflow.add_edge("reporter", "saver")
workflow.add_edge("saver", END)

app = workflow.compile()

import json

if __name__ == "__main__":
    # 1. Load your ACTUAL historical prices from wherever you store them
    # (e.g., reading from a local JSON file)
    try:
        with open("historical_prices.json", "r") as f:
            historical_data = json.load(f)
    except FileNotFoundError:
        historical_data = {} # Start fresh if no history exists

    # 2. Process every PDF in the directory, creating a per-PDF historical file
    pdf_dir = os.path.abspath(os.path.dirname(__file__))
    pdf_files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]

    if not pdf_files:
        print("No PDF files found in", pdf_dir)
    else:
        print(f"Found {len(pdf_files)} PDF(s). Processing each separately...")

    for pdf_path in pdf_files:
        base = os.path.splitext(os.path.basename(pdf_path))[0]
        historical_file = os.path.join(os.path.dirname(pdf_path), f"{base}_historical_prices.json")

        # Load historical prices for this PDF if present
        try:
            with open(historical_file, "r", encoding="utf-8") as f:
                historical_data = json.load(f)
        except FileNotFoundError:
            historical_data = {}

        initial_state = {
            "pdf_path": pdf_path,
            "historical_prices": historical_data,
            "historical_file": historical_file,
            "extracted_items": [],
            "discount_alerts": []
        }

        print("\n=======================================" )
        print(f"Processing: {os.path.basename(pdf_path)}")
        print("Historical file:", historical_file)

        try:
            app.invoke(PdfTrackerState(**initial_state))
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
















































