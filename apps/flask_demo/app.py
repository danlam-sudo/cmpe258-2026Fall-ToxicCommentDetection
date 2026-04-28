import json
import os
from pathlib import Path

import torch
from flask import Flask, render_template, request
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def resolve_bundle_dir() -> Path:
    default_dir = (
        Path(__file__).resolve().parents[2]
        / "notebooks"
        / "distilbert"
        / "distilbert_exp07_demo_bundle"
    )
    raw = os.environ.get("BUNDLE_DIR", str(default_dir))
    return Path(raw).expanduser().resolve()


def validate_bundle_dir(bundle_dir: Path) -> None:
    if not bundle_dir.exists() or not bundle_dir.is_dir():
        raise FileNotFoundError(
            f"BUNDLE_DIR does not exist or is not a directory: {bundle_dir}"
        )
    required = ["config.json", "labels.json", "thresholds.json"]
    missing = [name for name in required if not (bundle_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Bundle missing required files {missing} in {bundle_dir}"
        )


def load_bundle(bundle_dir: Path):
    validate_bundle_dir(bundle_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(bundle_dir), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(bundle_dir), local_files_only=True
    )
    model.eval()
    with open(bundle_dir / "labels.json", "r") as f:
        labels = json.load(f)
    with open(bundle_dir / "thresholds.json", "r") as f:
        thresholds = json.load(f)
    return tokenizer, model, labels, thresholds


def predict_one(text: str, tokenizer, model, labels, thresholds, max_length: int = 192):
    enc = tokenizer(
        [text],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    # DistilBERT forward does not accept token_type_ids.
    enc = {k: v for k, v in enc.items() if k in ("input_ids", "attention_mask")}
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    outputs = []
    for idx, label in enumerate(labels):
        prob = float(probs[idx])
        threshold = float(thresholds[label])
        outputs.append(
            {
                "label": label,
                "probability": round(prob, 4),
                "threshold": round(threshold, 4),
                "prediction": int(prob >= threshold),
            }
        )
    return outputs


app = Flask(__name__)
BUNDLE_DIR = resolve_bundle_dir()
TOKENIZER, MODEL, LABELS, THRESHOLDS = load_bundle(BUNDLE_DIR)


@app.get("/")
def index():
    return render_template(
        "index.html",
        text="",
        predictions=None,
        bundle_dir=str(BUNDLE_DIR),
        error=None,
    )


@app.post("/predict")
def predict():
    text = request.form.get("text", "").strip()
    if not text:
        return render_template(
            "index.html",
            text="",
            predictions=None,
            bundle_dir=str(BUNDLE_DIR),
            error="Please enter some text first.",
        )
    predictions = predict_one(text, TOKENIZER, MODEL, LABELS, THRESHOLDS)
    return render_template(
        "index.html",
        text=text,
        predictions=predictions,
        bundle_dir=str(BUNDLE_DIR),
        error=None,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="127.0.0.1", port=port, debug=False)
