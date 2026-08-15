"""src/nexora/__init__.py"""
from .adapter import NexoraCRNAdapter
from .features import encode_product, NEXORA_INPUT_DIM
from .generator import MarketingGenerator
from .crawler import fetch_product

__all__ = [
    "NexoraCRNAdapter",
    "encode_product",
    "NEXORA_INPUT_DIM",
    "MarketingGenerator",
    "fetch_product",
]
