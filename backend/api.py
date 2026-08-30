
"""
backend/api.py
==============
FastAPI backend for Nexora.

Pipeline:
    POST /analyze  →  crawler  →  features  →  CRN  →  generator  →  JSON

Start locally:
    uvicorn backend.api:app --host 0.0.0.0 --port 8000

Render start command:
    uvicorn backend.api:app --host 0.0.0.0 --port $PORT --workers 1
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

# ── Fix import paths BEFORE anything else ─────────────────────────────────
ROOT_DIR    = Path(__file__).resolve().parent.parent
NEXORA_SRC  = ROOT_DIR / "src"
if str(NEXORA_SRC) not in sys.path:
    sys.path.insert(0, str(NEXORA_SRC))

# ── Read Render's $PORT ────────────────────────────────────────────────────
PORT = int(os.environ.get("PORT", 8000))

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Nexora AI Engine",
    description="Convex Reasoning Network-powered marketing campaign generator.",
    version="1.0.0",
    docs_url="/docs",
)

# ── Serve index.html at "/" ────────────────────────────────────────────────
@app.get("/")
async def home():
    index = ROOT_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return JSONResponse({"service": "Nexora AI Engine", "status": "ok"})

# ── CORS ───────────────────────────────────────────────────────────────────
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
# Model state — loaded lazily so startup is instant
# ---------------------------------------------------------------------------

_adapter   = None
_generator = None
_model_loading = False
_model_error: str | None = None


def _load_model_sync() -> None:
    """
    Load CRN and generator synchronously.
    Called once in a background task so the port opens immediately.
    """
    global _adapter, _generator, _model_loading, _model_error
    try:
        from nexora.adapter   import NexoraCRNAdapter
        from nexora.generator import MarketingGenerator

        checkpoint_path = ROOT_DIR / "checkpoints" / "nexora_crn_best.pt"
        _adapter   = NexoraCRNAdapter(checkpoint_path=checkpoint_path)
        _generator = MarketingGenerator()

        status = "trained checkpoint" if _adapter.is_trained else "random initialisation"
        print(f"[Nexora] CRN loaded — {status}", flush=True)
    except Exception as exc:
        _model_error = str(exc)
        print(f"[Nexora] Model load failed: {exc}", flush=True)
    finally:
        _model_loading = False


@app.on_event("startup")
async def startup_event() -> None:
    """
    Port opens IMMEDIATELY — model loads in background.
    This prevents Render's port-scan timeout.
    """
    global _model_loading
    _model_loading = True
    print("[Nexora] Starting background model load...", flush=True)

    # Run blocking load in a thread so the event loop stays free
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _load_model_sync)


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
    success:     bool
    product:     dict
    crn:         dict
    campaign:    dict
    duration_ms: int
    errors:      list[str]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    """
    Always responds instantly — even while model is still loading.
    Used by Render's health-check and the frontend.
    """
    if _model_loading:
        return {
            "status":     "loading",
            "crn_status": "initialising",
            "model":      "ConvexReasoningNetwork",
            "version":    "1.0.0",
        }
    if _model_error:
        return {
            "status":     "error",
            "crn_status": "failed",
            "error":      _model_error,
            "version":    "1.0.0",
        }
    trained = _adapter.is_trained if _adapter else False
    return {
        "status":     "ok",
        "crn_status": "trained" if trained else "untrained",
        "model":      "ConvexReasoningNetwork",
        "version":    "1.0.0",
    }


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(body: AnalyzeRequest) -> AnalyzeResponse:
    t0 = time.monotonic()
    errors: list[str] = []

    # Guard: model still loading
    if _model_loading:
        raise HTTPException(
            status_code=503,
            detail="Nexora AI is still initialising. Please retry in a few seconds."
        )
    if _model_error:
        raise HTTPException(
            status_code=503,
            detail=f"Nexora AI failed to load: {_model_error}"
        )
    if _adapter is None or _generator is None:
        raise HTTPException(status_code=503, detail="Nexora AI not ready.")

    # 1. Fetch product page
    try:
        from nexora.crawler import fetch_product
        product = await fetch_product(body.url)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not fetch URL: {exc}")

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
            "crn_status":        "error",
            "state_dimension":   0,
            "trajectory_length": 0,
            "final_state_norm":  0.0,
            "final_state":       [],
            "checkpoint_info":   {},
        }

    # 3. Generate campaign
    try:
        campaign = _generator.generate(product, crn_result)
    except Exception as exc:
        errors.append(f"Campaign generation error: {str(exc)}")
        campaign = {"error": str(exc), "generator": "failed"}

    duration_ms = int((time.monotonic() - t0) * 1000)

    # Remove large internal field from response
    crn_public = {k: v for k, v in crn_result.items() if k != "final_state"}

    return AnalyzeResponse(
        success=len(errors) == 0,
        product=product,
        crn=crn_public,
        campaign=campaign,
        duration_ms=duration_ms,
        errors=errors,
    )


# ---------------------------------------------------------------------------
# Entry point — used by: python backend/api.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.api:app",
        host="0.0.0.0",
        port=PORT,
        reload=False,
    )
    
