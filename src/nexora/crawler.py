"""
src/nexora/crawler.py

Fetches a product page and extracts structured product data.

Returns a ProductData dict. On failure, returns an error dict.
"""
from future import annotations

import re
from typing import Optional
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

def extract_price(soup: BeautifulSoup, text: str) -> Optional[float]:
"""Try multiple strategies to extract a price."""
# JSON-LD
for script in soup.find_all("script", type="application/ld+json"):
try:
import json
data = json.loads(script.string or "")
if isinstance(data, dict):
offers = data.get("offers", {})
if isinstance(offers, dict):
p = offers.get("price") or offers.get("lowPrice")
if p:
return float(str(p).replace(",", ""))
except Exception:
pass

# Meta tags  
for meta in soup.find_all("meta", property=re.compile(r"product:price")):  
    try:  
        return float(meta.get("content", "").replace(",", ""))  
    except Exception:  
        pass  

# Regex on page text  
matches = re.findall(r"\$\s*([\d,]+(?:\.\d{2})?)", text)  
if matches:  
    try:  
        return float(matches[0].replace(",", ""))  
    except Exception:  
        pass  

return None

def detect_category(title: str, description: str, url: str) -> str:
"""Simple keyword-based category detection."""
text = (title + " " + description + " " + url).lower()
mapping = {
"electronics": ["phone", "laptop", "tablet", "camera", "headphone", "speaker", "tv", "monitor"],
"fashion": ["shirt", "dress", "shoe", "jacket", "pants", "clothing", "wear", "fashion"],
"beauty": ["skin", "makeup", "cosmetic", "cream", "serum", "lipstick", "foundation"],
"home": ["sofa", "chair", "table", "bed", "kitchen", "home", "furniture", "decor"],
"sports": ["gym", "sport", "fitness", "workout", "yoga", "running", "cycling"],
"food": ["food", "snack", "drink", "coffee", "tea", "protein", "supplement"],
"toys": ["toy", "game", "lego", "doll", "puzzle", "kids", "children"],
"health": ["health", "vitamin", "supplement", "medical", "dental", "pharmacy"],
}
for category, keywords in mapping.items():
if any(kw in text for kw in keywords):
return category
return "other"

async def fetch_product(url: str, timeout: float = 10.0) -> dict:
"""
Fetch a product URL and extract structured data.

Returns  
-------  
dict with keys: title, description, price, category, url, error (if any)  
"""  
try:  
    parsed = urlparse(url)  
    if not parsed.scheme or not parsed.netloc:  
        return {"error": "Invalid URL: missing scheme or domain", "url": url}  
except Exception:  
    return {"error": "Malformed URL", "url": url}  

headers = {  
    "User-Agent": (  
        "Mozilla/5.0 (compatible; NexoraBot/1.0; "  
        "+https://nexora.ai/bot)"  
    ),  
    "Accept": "text/html,application/xhtml+xml",  
    "Accept-Language": "en-US,en;q=0.9",  
}  

try:  
    async with httpx.AsyncClient(  
        follow_redirects=True,  
        timeout=timeout,  
        headers=headers,  
    ) as client:  
        resp = await client.get(url)  
        resp.raise_for_status()  
        html = resp.text  
except httpx.TimeoutException:  
    return {"error": "Request timed out. The product page took too long to respond.", "url": url}  
except httpx.HTTPStatusError as e:  
    return {"error": f"HTTP {e.response.status_code} from product page.", "url": url}  
except Exception as e:  
    return {"error": f"Could not fetch page: {str(e)}", "url": url}  

soup = BeautifulSoup(html, "html.parser")  
text = soup.get_text(separator=" ", strip=True)  

# Title  
title = ""  
if soup.title:  
    title = soup.title.string or ""  
if not title:  
    h1 = soup.find("h1")  
    title = h1.get_text(strip=True) if h1 else url  

# Description  
desc_tag = (  
    soup.find("meta", attrs={"name": "description"}) or  
    soup.find("meta", property="og:description")  
)  
description = (desc_tag.get("content", "") if desc_tag else "")[:500]  

# Price  
price = extract_price(soup, text)  

# Category  
category = detect_category(title, description, url)  

return {  
    "title": title[:200].strip(),  
    "description": description.strip(),  
    "price": price,  
    "category": category,  
    "url": url,  
}
