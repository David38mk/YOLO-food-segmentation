"""Quantitative LLaVA evaluation on the Emteq ground-truth set.

Pipeline per image (mirrors the paper's prototype): detect food with the
fine-tuned YOLOv8 model, mask everything outside the detected boxes, send the
masked image to a local Ollama LLaVA model, and compare its answer against the
labels in data/ground_truth/ground_truth.csv.

A prediction counts as correct when at least one ground-truth food item is
mentioned in LLaVA's answer (case-insensitive substring match on the main noun),
which matches how the ground truth was written (free-text food descriptions).

Requires: pip install ultralytics requests
          ollama pull llava   (and `ollama serve` running)
Run from repo root: python src/evaluate_llava.py
"""

import base64
import csv
import os

import cv2
import numpy as np
import requests
from ultralytics import YOLO

MODEL_PATH = "models/best.pt"
IMG_DIR = "data/FoodID_Dataset"
GT_CSV = "data/ground_truth/ground_truth.csv"
OLLAMA_URL = "http://localhost:11434/api/generate"
CONF_THRESHOLD = 0.25
PROMPT = (
    "This image shows food with the background blacked out. "
    "Name the food items you see, as a short comma separated list."
)

# generic words that would match almost any answer and inflate accuracy
STOPWORDS = {"with", "and", "the", "a", "of", "milk", "bar", "ice", "meat"}


def mask_outside_boxes(image, boxes):
    masked = np.zeros_like(image)
    for x1, y1, x2, y2 in boxes.astype(int):
        masked[y1:y2, x1:x2] = image[y1:y2, x1:x2]
    return masked


def ask_llava(image):
    # downscale: vision encoding dominates CPU inference time
    h, w = image.shape[:2]
    scale = 512 / max(h, w)
    if scale < 1:
        image = cv2.resize(image, (int(w * scale), int(h * scale)))
    ok, buf = cv2.imencode(".jpg", image)
    if not ok:
        return None
    payload = {
        "model": "llava",
        "prompt": PROMPT,
        "images": [base64.b64encode(buf.tobytes()).decode()],
        "stream": False,
        "options": {"num_predict": 60},
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=300)
    r.raise_for_status()
    return r.json()["response"]


def keywords(gt_text):
    words = set()
    for item in gt_text.replace("/", ",").split(","):
        for w in item.strip().lower().split():
            w = w.strip("().?")
            if len(w) > 3 and w not in STOPWORDS:
                words.add(w)
    return words


RESULTS_CSV = "data/ground_truth/llava_eval_results.csv"


def main():
    # resumable: skip images already in the results file, append new ones
    done = set()
    if os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, newline="", encoding="utf-8") as f:
            done = {r["image"] for r in csv.DictReader(f)}

    model = YOLO(MODEL_PATH)
    with open(GT_CSV, newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r["Ground Truth"].strip() != "?"]

    limit = int(os.environ.get("EVAL_LIMIT", "0"))  # 0 = no limit
    processed = 0
    new_file = not done
    out = open(RESULTS_CSV, "a", newline="", encoding="utf-8")
    writer = csv.writer(out)
    if new_file:
        writer.writerow(["image", "ground_truth", "llava_answer", "correct"])

    for row in rows:
        name = row["Image #"].strip()
        if name in done:
            continue
        if limit and processed >= limit:
            break
        # ultralytics patches cv2.imread to raise on missing files instead of
        # returning None, so guard with an existence check as well
        path = os.path.join(IMG_DIR, name)
        if not os.path.exists(path):
            continue
        image = cv2.imread(path)
        if image is None:
            continue

        pred = model.predict(image, conf=CONF_THRESHOLD, verbose=False)[0]
        if len(pred.boxes) == 0:
            writer.writerow([name, row["Ground Truth"], "NO DETECTION", False])
        else:
            masked = mask_outside_boxes(image, pred.boxes.xyxy.cpu().numpy())
            answer = ask_llava(masked)
            hit = any(k in answer.lower() for k in keywords(row["Ground Truth"]))
            writer.writerow([name, row["Ground Truth"], answer.strip().replace("\n", " "), hit])
        out.flush()
        processed += 1
        print(f"[{len(done) + processed}] {name}", flush=True)

    out.close()

    # summary over everything accumulated so far
    with open(RESULTS_CSV, newline="", encoding="utf-8") as f:
        results = list(csv.DictReader(f))
    total = len(results)
    detected = sum(1 for r in results if r["llava_answer"] != "NO DETECTION")
    correct = sum(1 for r in results if r["correct"] == "True")
    print(f"\nImages evaluated:        {total}")
    print(f"Food detected (YOLO):    {detected} ({detected / max(total, 1):.1%})")
    print(f"LLaVA correct food type: {correct}/{detected} ({correct / max(detected, 1):.1%} of detected)")


if __name__ == "__main__":
    main()
