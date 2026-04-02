# app/vlm_pipeline/vlm_extractor.py
import json
import logging
from typing import Dict, Any, List
import os
import base64

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Tu es un extracteur de données bibliographiques. Analyse cette mosaïque de tranches de livres. Chaque livre a un numéro au-dessus. Retourne UNIQUEMENT un objet JSON valide avec la structure suivante : {"books": [{"id": 1, "title": "...", "author": "...", "publisher": "..."}]}. Remplace par null si illisible. Ne génère aucun texte hors du JSON."""

async def extract_metadata_from_mosaic(mosaic_base64: str) -> Dict[str, List[Dict[str, Any]]]:
    if not genai:
        logger.error("google-genai SDK not installed")
        return {"books": []}
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY not set")
        return {"books": []}
    try:
        client = genai.Client(api_key=api_key)
        image_part = types.Part.from_bytes(
            data=base64.b64decode(mosaic_base64),
            mime_type="image/jpeg"
        )
        # Using generate_content synchronously as per plan code snippet, 
        # but the function is async. The plan code is what I should follow.
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=[SYSTEM_PROMPT, image_part],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            )
        )
        text = response.text
        if text.startswith("```json"):
            text = text[7:-3].strip()
        elif text.startswith("```"):
             # Handle other code blocks if necessary, but flash 2.0 with json mime type should be cleaner
             pass
             
        return json.loads(text)
    except Exception as e:
        logger.error(f"VLM extraction failed: {e}")
        return {"books": []}
