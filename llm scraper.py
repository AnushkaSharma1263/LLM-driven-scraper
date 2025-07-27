import requests
from bs4 import BeautifulSoup
import time
import os
import json
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- Step 1: Scrape the Earth911 search results page ---
SEARCH_URL = "https://search.earth911.com/"
SEARCH_PARAMS = {
    "what": "Electronics",
    "where": "10001",
    "list_filter": "all",
    "max_distance": "100"
}

headers = {
    "User-Agent": "Mozilla/5.0"
}

response = requests.get(SEARCH_URL, params=SEARCH_PARAMS, headers=headers)
soup = BeautifulSoup(response.text, "html.parser")

# --- Step 2: Parse the first 3 facility entries ---
facilities = []
for facility in soup.select(".result-item")[:3]:
    name = facility.select_one(".result-title")
    address = facility.select_one(".result-address")
    updated = facility.select_one(".result-updated")
    materials = facility.select_one(".result-materials")
    facilities.append({
        "business_name": name.text.strip() if name else "",
        "last_update_date": updated.text.strip().replace("Last updated: ", "") if updated else "",
        "street_address": address.text.strip() if address else "",
        "raw_materials": materials.text.strip() if materials else ""
    })

if not facilities:
    print("No facilities found. The website may require JavaScript rendering or has changed its structure.")
    exit(1)

# --- Step 3: Local embedding-based mapping of materials ---
master_items = [
    "Computers, Laptops, Tablets",
    "Monitors, TVs (CRT & Flat Screen)",
    "Cell Phones, Smartphones",
    "Printers, Copiers, Fax Machines",
    "Audio/Video Equipment",
    "Gaming Consoles",
    "Small Appliances (Microwaves, Toasters, etc.)",
    "Computer Peripherals (Keyboards, Mice, Cables, etc.)",
    "Household Batteries (AA, AAA, 9V, etc.)",
    "Rechargeable Batteries",
    "Lithium-ion Batteries",
    "Button/Watch Batteries",
    "Power Tool Batteries",
    "E-bike/Scooter Batteries",
    "Car/Automotive Batteries",
    "Latex/Water-based Paint",
    "Oil-based Paint and Stains",
    "Spray Paint",
    "Paint Thinners and Solvents",
    "Household Cleaners",
    "Pool Chemicals",
    "Pesticides and Herbicides",
    "Automotive Fluids (Oil, Antifreeze)",
    "Needles and Syringes",
    "Lancets",
    "Auto-injectors (EpiPens)",
    "Insulin Pens",
    "Home Dialysis Equipment",
    "Clothing and Shoes",
    "Household Textiles (Towels, Bedding)",
    "Fabric Scraps",
    "Accessories (Belts, Bags, etc.)",
    "Fluorescent Bulbs and CFLs",
    "Mercury Thermometers",
    "Smoke Detectors",
    "Fire Extinguishers",
    "Propane Tanks",
    "Mattresses and Box Springs",
    "Large Appliances (Fridges, Washers, etc.)",
    "Construction Debris (Residential Quantities)"
]

item_to_category = {}
for item in master_items:
    if item in [
        "Computers, Laptops, Tablets",
        "Monitors, TVs (CRT & Flat Screen)",
        "Cell Phones, Smartphones",
        "Printers, Copiers, Fax Machines",
        "Audio/Video Equipment",
        "Gaming Consoles",
        "Small Appliances (Microwaves, Toasters, etc.)",
        "Computer Peripherals (Keyboards, Mice, Cables, etc.)"
    ]:
        item_to_category[item] = "Electronics"
    elif item in [
        "Household Batteries (AA, AAA, 9V, etc.)",
        "Rechargeable Batteries",
        "Lithium-ion Batteries",
        "Button/Watch Batteries",
        "Power Tool Batteries",
        "E-bike/Scooter Batteries",
        "Car/Automotive Batteries"
    ]:
        item_to_category[item] = "Batteries"
    elif item in [
        "Latex/Water-based Paint",
        "Oil-based Paint and Stains",
        "Spray Paint",
        "Paint Thinners and Solvents",
        "Household Cleaners",
        "Pool Chemicals",
        "Pesticides and Herbicides",
        "Automotive Fluids (Oil, Antifreeze)"
    ]:
        item_to_category[item] = "Paint & Chemicals"
    elif item in [
        "Needles and Syringes",
        "Lancets",
        "Auto-injectors (EpiPens)",
        "Insulin Pens",
        "Home Dialysis Equipment"
    ]:
        item_to_category[item] = "Medical Sharps"
    elif item in [
        "Clothing and Shoes",
        "Household Textiles (Towels, Bedding)",
        "Fabric Scraps",
        "Accessories (Belts, Bags, etc.)"
    ]:
        item_to_category[item] = "Textiles & Clothing"
    else:
        item_to_category[item] = "Other Important Materials"

# Load local embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')
master_embeddings = embedder.encode(master_items)

def map_materials_with_local_model(raw_materials, top_k=3, threshold=0.6):
    if not raw_materials:
        return {"materials_category": [], "materials_accepted": []}
    lines = [line.strip() for line in raw_materials.split('\n') if line.strip()]
    accepted_items = set()
    categories = set()
    for line in lines:
        query_emb = embedder.encode([line])
        sims = cosine_similarity(query_emb, master_embeddings)[0]
        for idx in np.argsort(sims)[::-1][:top_k]:
            if sims[idx] >= threshold:
                accepted_items.add(master_items[idx])
                categories.add(item_to_category[master_items[idx]])
    return {
        "materials_category": list(categories),
        "materials_accepted": list(accepted_items)
    }

# --- Step 4: Build the final JSON array ---
output = []
for fac in facilities:
    llm_result = map_materials_with_local_model(fac["raw_materials"])
    output.append({
        "business_name": fac["business_name"],
        "last_update_date": fac["last_update_date"],
        "street_address": fac["street_address"],
        "materials_category": llm_result["materials_category"],
        "materials_accepted": llm_result["materials_accepted"]
    })
    time.sleep(0.5)  # To avoid hammering CPU

print(json.dumps(output, indent=2)) 