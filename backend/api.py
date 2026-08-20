"""
backend/api.py
==============
FastAPI backend for Nexora.

Pipeline:
    POST /analyze  →  crawler  →  features  →  CRN  →  generator  →  JSON

Start:
    uvicorn backend.api:app --reload --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Allow imports from Nexora/src/
ROOT_DIR = Path(__file__).resolve().parent.parent
NEXORA_SRC = ROOT_DIR / "Nexora" / "src"

if str(NEXORA_SRC) not in sys.path:
    sys.path.insert(0, str(NEXORA_SRC))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl, field_validator

from nexora.crawler import fetch_product
from nexora.adapter import NexoraCRNAdapter
from nexora.generator import MarketingGenerator

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Nexora AI Engine",
    description="Convex Reasoning Network-powered marketing campaign generator.",
    version="1.0.0",
    docs_url="/docs",
)

# CORS — GitHub Pages origin + localhost for dev
ALLOWED_ORIGINS = [
    "https://katitenorhane1234-rgb.github.io",
    "http://localhost:3000",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

# ---------------------------------------------------------------------------
# Startup — load model once
# ---------------------------------------------------------------------------

_adapter: NexoraCRNAdapter | None = None
_generator: MarketingGenerator | None = None


@app.on_event("startup")
async def startup_event() -> None:
    global _adapter, _generator
    checkpoint_path = Path(__file__).parent.parent / "checkpoints" / "nexora_crn_best.pt"
    _adapter   = NexoraCRNAdapter(checkpoint_path=checkpoint_path)
    _generator = MarketingGenerator()

    status = "trained checkpoint" if _adapter.is_trained else "random initialisation (train first)"
    print(f"[Nexora] CRN loaded — {status}")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    url: str

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("URL cannot be empty")
        if not (v.startswith("http://") or v.startswith("https://")):
            raise ValueError("URL must start with http:// or https://")
        return v


class AnalyzeResponse(BaseModel):
    success: bool
    product: dict
    crn: dict
    campaign: dict
    duration_ms: int
    errors: list[str]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    trained = _adapter.is_trained if _adapter else False
    return {
        "status": "ok",
        "crn_status": "trained" if trained else "untrained",
        "model": "ConvexReasoningNetwork",
        "version": "1.0.0",
    }


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(body: AnalyzeRequest) -> AnalyzeResponse:
    t0 = time.monotonic()
    errors: list[str] = []

    # 1. Fetch product page
    product = await fetch_product(body.url)
    if "error" in product:
        raise HTTPException(status_code=422, detail=product["error"])

    # 2. Run CRN
    try:
        crn_result = _adapter.run_inference(
            title=product["title"],
            description=product.get("description", ""),
            price=product.get("price"),
            category=product.get("category", "other"),
            url=body.url,
        )
    except Exception as exc:
        errors.append(f"CRN inference error: {str(exc)}")
        crn_result = {
            "crn_status": "error",
            "state_dimension": 0,
            "trajectory_length": 0,
            "final_state_norm": 0.0,
            "final_state": [],
            "checkpoint_info": {},
        }

    # 3. Generate campaign
    try:
        campaign = _generator.generate(product, crn_result)
    except Exception as exc:
        errors.append(f"Campaign generation error: {str(exc)}")
        campaign = {"error": str(exc), "generator": "failed"}

    duration_ms = int((time.monotonic() - t0) * 1000)

    # Strip internal final_state from response (too large, not useful to frontend)
    crn_public = {k: v for k, v in crn_result.items() if k != "final_state"}

    return AnalyzeResponse(
        success=len(errors) == 0,
        product=product,
        crn=crn_public,
        campaign=campaign,
        duration_ms=duration_ms,
        errors=errors,
    )
