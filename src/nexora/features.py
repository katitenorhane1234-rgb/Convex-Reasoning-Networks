"""
src/nexora/features.py
======================
Converts raw product data (title, price, description, category)
into a fixed-dimensional tensor suitable for CRN input.

This is a deterministic, rule-based encoder — no pre-trained LLM
weights are required. The encoding is honest about what it is:
a heuristic numerical representation of product attributes.
"""
from __future__ import annotations

import hashlib
import re
from typing import Optional

import torch
from torch import Tensor

# CRN input dimension expected by the default nexora checkpoint
NEXORA_INPUT_DIM = 32


CATEGORY_LIST = [
    "electronics", "fashion", "beauty", "home", "sports",
    "food", "toys", "automotive", "books", "health",
    "jewellery", "garden", "pet", "office", "travel",
    "baby", "music", "gaming", "art", "other",
]

SENTIMENT_WORDS_POS = {"premium", "best", "top", "pro", "ultra", "advanced", "new", "original"}
SENTIMENT_WORDS_NEG = {"cheap", "basic", "simple", "generic", "used", "refurbished"}


def _hash_feature(text: str, n_dims: int, seed: int = 0) -> Tensor:
    """Deterministic hashing trick: map text → n_dims floats in [-1, 1]."""
    feats = []
    for i in range(n_dims):
        h = hashlib.md5(f"{seed}:{i}:{text[:256]}".encode()).hexdigest()
        val = (int(h[:8], 16) / 0xFFFFFFFF) * 2 - 1   # in [-1, 1]
        feats.append(val)
    return torch.tensor(feats, dtype=torch.float32)


def encode_product(
    title: str,
    description: str = "",
    price: Optional[float] = None,
    category: str = "other",
    url: str = "",
) -> Tensor:
    """
    Encode product attributes into a float32 tensor of shape (NEXORA_INPUT_DIM,).

    Encoding breakdown (32 dims total):
        [0]       price_norm          normalised log-price (or 0 if unknown)
        [1]       has_price           binary flag
        [2]       title_len_norm      normalised title length
        [3]       desc_len_norm       normalised description length
        [4]       sentiment_score     simple lexicon score
        [5:25]    category_onehot     20-dim one-hot (category)
        [25:32]   title_hash          7-dim hash of title
    """
    feats = torch.zeros(NEXORA_INPUT_DIM)

    # 0: price (log-normalised)
    if price is not None and price > 0:
        import math
        feats[0] = math.log1p(price) / 10.0   # rough normalisation
        feats[1] = 1.0
    else:
        feats[0] = 0.0
        feats[1] = 0.0

    # 2–3: text lengths
    feats[2] = min(len(title) / 100.0, 1.0)
    feats[3] = min(len(description) / 500.0, 1.0)

    # 4: sentiment
    words = set((title + " " + description).lower().split())
    pos = len(words & SENTIMENT_WORDS_POS)
    neg = len(words & SENTIMENT_WORDS_NEG)
    feats[4] = (pos - neg) / max(pos + neg + 1, 1)

    # 5–24: category one-hot
    cat_lower = category.lower().strip()
    cat_idx = CATEGORY_LIST.index(cat_lower) if cat_lower in CATEGORY_LIST else len(CATEGORY_LIST) - 1
    feats[5 + cat_idx] = 1.0

    # 25–31: title hash (7 dims)
    feats[25:32] = _hash_feature(title, n_dims=7, seed=42)

    return feats


def encode_product_batch(products: list[dict]) -> Tensor:
    """Encode a list of product dicts → (N, NEXORA_INPUT_DIM)."""
    return torch.stack([
        encode_product(
            title=p.get("title", ""),
            description=p.get("description", ""),
            price=p.get("price"),
            category=p.get("category", "other"),
            url=p.get("url", ""),
        )
        for p in products
    ])
