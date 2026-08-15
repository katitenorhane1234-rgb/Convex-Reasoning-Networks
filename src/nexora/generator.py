"""
src/nexora/generator.py
=======================
MarketingGenerator — converts CRN latent state + product info into campaign JSON.

Two modes:
  1. LLM mode  (requires ANTHROPIC_API_KEY env var)
     Uses Claude to generate natural-language marketing copy.
     The CRN final_state is included as a structured context vector.

  2. Local prototype mode  (no API key needed)
     Deterministic template-based generation.
     Clearly labelled as "local_prototype".
     Used for testing and when no LLM key is available.
"""
from __future__ import annotations

import os
import json
from typing import Any

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")


class MarketingGenerator:

    def generate(
        self,
        product: dict,
        crn_result: dict,
    ) -> dict:
        """
        Parameters
        ----------
        product    : extracted product info (title, price, category, …)
        crn_result : output of NexoraCRNAdapter.run_inference()

        Returns
        -------
        campaign dict
        """
        if ANTHROPIC_API_KEY:
            return self._generate_with_llm(product, crn_result)
        return self._generate_local(product, crn_result)

    # ------------------------------------------------------------------
    def _generate_with_llm(self, product: dict, crn_result: dict) -> dict:
        """Generate campaign copy using Claude, with CRN state as context."""
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

            system = (
                "You are Nexora AI, a marketing campaign generator. "
                "You receive structured product data and a CRN (Convex Reasoning Network) "
                "latent representation vector. Use this information to generate a professional "
                "marketing campaign. Return ONLY valid JSON with the structure shown."
            )

            crn_summary = {
                "state_dimension": crn_result["state_dimension"],
                "trajectory_length": crn_result["trajectory_length"],
                "state_norm": round(crn_result["final_state_norm"], 4),
                "status": crn_result["crn_status"],
            }

            user_msg = f"""
Product:
{json.dumps(product, indent=2)}

CRN Latent Representation:
{json.dumps(crn_summary, indent=2)}

Generate a marketing campaign JSON with this structure:
{{
  "hook": "one compelling opening line",
  "cta": "call to action",
  "caption": "social media caption (2-3 sentences)",
  "hashtags": ["tag1", "tag2", "tag3", "tag4", "tag5"],
  "video_concept": "brief description of video ad concept",
  "target_audience": "description of ideal customer",
  "platforms": ["TikTok", "Instagram"],
  "tone": "energetic|professional|casual|luxury",
  "generator": "nexora_llm"
}}
"""
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1000,
                system=system,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw = response.content[0].text.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            return json.loads(raw)

        except Exception as exc:
            # LLM failed — fall back to local generator, report reason
            result = self._generate_local(product, crn_result)
            result["llm_error"] = str(exc)
            return result

    # ------------------------------------------------------------------
    def _generate_local(self, product: dict, crn_result: dict) -> dict:
        """
        LOCAL PROTOTYPE GENERATOR.

        Deterministic template-based campaign. Not AI-generated.
        Useful for testing the full pipeline without any API keys.
        """
        title = product.get("title", "this product")
        category = product.get("category", "product")
        price = product.get("price")
        price_str = f"${price:.0f}" if price else "great price"

        crn_norm = round(crn_result.get("final_state_norm", 0.0), 3)
        crn_status = crn_result.get("crn_status", "unknown")

        hooks = [
            f"You've been looking for {title} — and here it is.",
            f"The {category} world changed forever. Meet {title}.",
            f"Stop scrolling. {title} is exactly what you need.",
        ]
        ctaList = [
            "Shop now →",
            "Get yours today →",
            "Limited stock — order now →",
        ]

        # Use CRN state norm to deterministically pick template variation
        idx = int(crn_norm * 10) % len(hooks)

        return {
            "hook": hooks[idx],
            "cta": ctaList[idx],
            "caption": (
                f"Introducing {title} — the {category} solution you've been waiting for. "
                f"Available now at {price_str}. Don't miss out."
            ),
            "hashtags": [
                f"#{category.replace(' ', '')}",
                "#NewDrop",
                "#MustHave",
                "#ShopNow",
                "#Nexora",
            ],
            "video_concept": (
                f"30-second product reveal: open with close-up of {title}, "
                f"show key features, end with CTA overlay."
            ),
            "target_audience": f"People interested in {category}, price-conscious shoppers seeking quality.",
            "platforms": ["TikTok", "Instagram", "Facebook"],
            "tone": "energetic",
            "generator": "local_prototype",
            "crn_influence_note": (
                f"CRN latent norm={crn_norm} (status: {crn_status}) "
                f"used to select template variation {idx}."
            ),
        }
