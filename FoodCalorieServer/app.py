from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import io
import base64
import requests
import json

app = FastAPI()

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llava"

SYSTEM_PROMPT = """You are a nutrition assistant.
Given a FOOD PHOTO, identify the most likely food item(s) and estimate total calories visible.
Return ONLY valid JSON with keys:
- label: short description string
- calories: integer total calories estimate
If no food is visible, return label="no food" and calories=0.
Be conservative and reasonable.
"""

@app.post("/analyze")
async def analyze(image: UploadFile = File(...)):
    try:
        raw = await image.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")

        # Resize for faster inference (dev)
        img.thumbnail((768, 768))

        # Encode as base64 JPEG for Ollama
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        prompt = (
            SYSTEM_PROMPT +
            "\nReturn JSON only. Do not include markdown.\n"
        )

        payload = {
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "images": [b64],
        }

        r = requests.post(OLLAMA_URL, json=payload, timeout=180)
        r.raise_for_status()
        text = r.json().get("response", "").strip()

        # Try to parse JSON from the model output (robust)
        # If it returns extra text, extract the first {...} block.
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start:end+1]

        data = json.loads(text)

        label = str(data.get("label", "no food"))
        calories = int(data.get("calories", 0))

        return {"label": label, "calories": calories}

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/health")
def health():
    return {"ok": True}
