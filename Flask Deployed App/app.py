# =============================================================================
# Plant Disease Detection — Complete Flask Application
# Model   : Custom 4-block VGG-style CNN (PyTorch)
# Author  : Nadimpalli Siddhartha Kumar Raju
# =============================================================================

# ── 1. IMPORTS ────────────────────────────────────────────────────────────────
import os
import json
import time
import uuid
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from flask import (
    Flask,
    redirect,
    render_template,
    request,
    jsonify,
    url_for,
    flash,
)
import pandas as pd
import gdown
import CNN  # CNN.py must be in the same directory

# ── 2. LOGGING ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── 3. CONFIG ─────────────────────────────────────────────────────────────────
class Config:
    # Google Drive model file
    GDRIVE_FILE_ID  = "1QRLcceLPKax9cdUlJW59853i-dYqZqTw"
    GDRIVE_URL      = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}"
    MODEL_PATH      = "plant_disease_model_1_latest.pt"

    # Data files
    DISEASE_CSV     = "disease_info.csv"
    SUPPLEMENT_CSV  = "supplement_info.csv"

    # Upload settings
    UPLOAD_FOLDER       = "static/uploads"
    ALLOWED_EXTENSIONS  = {"png", "jpg", "jpeg", "webp", "bmp"}
    MAX_CONTENT_MB      = 10

    # Model
    NUM_CLASSES         = 39
    IMG_SIZE            = 224
    CONFIDENCE_WARN     = 60.0   # warn user if confidence < this %
    TOP_K               = 3      # return top-K predictions

    # App
    SECRET_KEY          = "plant-disease-secret-2024"
    MAX_HISTORY         = 20     # keep last N predictions in memory

cfg = Config()


# ── 4. APP INIT ───────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = cfg.SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = cfg.MAX_CONTENT_MB * 1024 * 1024

os.makedirs(cfg.UPLOAD_FOLDER, exist_ok=True)

# In-memory prediction history (resets on server restart)
prediction_history: list[dict] = []


# ── 5. IMAGE TRANSFORM ────────────────────────────────────────────────────────
def preprocess_image(image_path: str) -> torch.Tensor:
    """
    Opens image, resizes to 224×224, converts to normalised tensor.
    Handles RGBA / palette images by converting to RGB first.
    Returns tensor of shape (1, 3, 224, 224).
    """
    image = Image.open(image_path).convert("RGB")
    image = image.resize((cfg.IMG_SIZE, cfg.IMG_SIZE), Image.BILINEAR)
    tensor = TF.to_tensor(image)                         # (3, 224, 224), [0,1]
    tensor = TF.normalize(                               # ImageNet stats
        tensor,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    return tensor.unsqueeze(0)                           # (1, 3, 224, 224)


# ── 6. MODEL LOADER ───────────────────────────────────────────────────────────
def load_model() -> torch.nn.Module:
    """
    Downloads model from Google Drive if not present locally, then loads it.
    Returns model in eval mode on CPU.
    """
    if not os.path.exists(cfg.MODEL_PATH):
        log.info("Model not found locally — downloading from Google Drive ...")
        gdown.download(cfg.GDRIVE_URL, cfg.MODEL_PATH, quiet=False)
        log.info("Download complete.")
    else:
        log.info(f"Model found at {cfg.MODEL_PATH} — skipping download.")

    model = CNN.CNN(cfg.NUM_CLASSES)
    state = torch.load(cfg.MODEL_PATH, map_location=torch.device("cpu"))
    model.load_state_dict(state)
    model.eval()
    log.info("Model loaded successfully.")
    return model


# ── 7. PREDICTION ─────────────────────────────────────────────────────────────
def predict(image_path: str) -> dict:
    """
    Runs inference on a single image.
    Returns dict with:
        index        : int   — predicted class index
        disease_name : str
        confidence   : float — top-1 confidence in %
        low_conf     : bool  — True if confidence < threshold
        top_k        : list  — [{index, disease_name, confidence}, ...]
        inference_ms : float — wall-clock time in ms
    """
    t0 = time.perf_counter()

    tensor = preprocess_image(image_path)
    with torch.no_grad():
        output = MODEL(tensor)
        probs  = torch.softmax(output[0], dim=0)

    top_probs, top_idxs = torch.topk(probs, k=cfg.TOP_K)

    top_k_results = [
        {
            "index":        idx.item(),
            "disease_name": IDX_TO_CLASS.get(idx.item(), "Unknown"),
            "confidence":   round(prob.item() * 100, 2),
        }
        for prob, idx in zip(top_probs, top_idxs)
    ]

    best         = top_k_results[0]
    inference_ms = round((time.perf_counter() - t0) * 1000, 1)

    return {
        "index":        best["index"],
        "disease_name": best["disease_name"],
        "confidence":   best["confidence"],
        "low_conf":     best["confidence"] < cfg.CONFIDENCE_WARN,
        "top_k":        top_k_results,
        "inference_ms": inference_ms,
    }


# ── 8. HELPERS ────────────────────────────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in cfg.ALLOWED_EXTENSIONS
    )


def safe_filename(original: str) -> str:
    """Generates a unique filename preserving the original extension."""
    ext = Path(original).suffix.lower()
    return f"{uuid.uuid4().hex}{ext}"


def get_disease_data(idx: int) -> dict:
    """Returns disease info row as dict for a given class index."""
    row = DISEASE_INFO.iloc[idx]
    return {
        "title":       row.get("disease_name",  "Unknown"),
        "description": row.get("description",   ""),
        "steps":       row.get("Possible Steps",""),
        "image_url":   row.get("image_url",     ""),
    }


def get_supplement_data(idx: int) -> dict:
    """Returns supplement info row as dict for a given class index."""
    row = SUPPLEMENT_INFO.iloc[idx]
    return {
        "name":      row.get("supplement name",  ""),
        "image_url": row.get("supplement image", ""),
        "buy_link":  row.get("buy link",         "#"),
    }


def add_to_history(filename: str, pred: dict) -> None:
    """Appends a prediction to the in-memory history list."""
    prediction_history.append({
        "timestamp":    datetime.now().strftime("%d %b %Y, %I:%M %p"),
        "filename":     filename,
        "disease":      pred["disease_name"],
        "confidence":   pred["confidence"],
        "low_conf":     pred["low_conf"],
        "inference_ms": pred["inference_ms"],
        "top_k":        pred["top_k"],
    })
    # Keep only the most recent N predictions
    if len(prediction_history) > cfg.MAX_HISTORY:
        prediction_history.pop(0)


# ── 9. ROUTES ─────────────────────────────────────────────────────────────────

# ── 9a. Home ──────────────────────────────────────────────────────────────────
@app.route("/")
def home_page():
    stats = {
        "total_predictions": len(prediction_history),
        "diseases_covered":  cfg.NUM_CLASSES,
    }
    return render_template("home.html", stats=stats)


# ── 9b. AI Engine (upload page) ───────────────────────────────────────────────
@app.route("/index")
def ai_engine_page():
    return render_template("index.html")


# ── 9c. Submit (prediction) ───────────────────────────────────────────────────
@app.route("/submit", methods=["POST"])
def submit():
    # Validate file presence
    if "image" not in request.files:
        flash("No file was uploaded. Please select an image.", "error")
        return redirect(url_for("ai_engine_page"))

    file = request.files["image"]

    if file.filename == "":
        flash("No file selected. Please choose a leaf image.", "error")
        return redirect(url_for("ai_engine_page"))

    if not allowed_file(file.filename):
        flash(
            f"Invalid file type. Allowed: {', '.join(cfg.ALLOWED_EXTENSIONS).upper()}",
            "error",
        )
        return redirect(url_for("ai_engine_page"))

    # Save file
    filename  = safe_filename(file.filename)
    file_path = os.path.join(cfg.UPLOAD_FOLDER, filename)
    file.save(file_path)
    log.info(f"Image saved: {file_path}")

    # Run prediction
    try:
        pred = predict(file_path)
    except Exception as e:
        log.error(f"Prediction failed: {e}")
        flash("Prediction failed. Please try again with a different image.", "error")
        return redirect(url_for("ai_engine_page"))

    # Fetch disease and supplement info
    disease    = get_disease_data(pred["index"])
    supplement = get_supplement_data(pred["index"])

    # Log to history
    add_to_history(file.filename, pred)

    log.info(
        f"Predicted: {pred['disease_name']}  "
        f"conf={pred['confidence']}%  "
        f"time={pred['inference_ms']}ms"
    )

    return render_template(
        "submit.html",
        # Disease info
        title       = disease["title"],
        desc        = disease["description"],
        prevent     = disease["steps"],
        image_url   = disease["image_url"],
        # Prediction meta
        pred        = pred["index"],
        confidence  = pred["confidence"],
        low_conf    = pred["low_conf"],
        inference_ms= pred["inference_ms"],
        top_k       = pred["top_k"],
        # Uploaded image path (for preview)
        uploaded_image = url_for("static", filename=f"uploads/{filename}"),
        # Supplement
        sname       = supplement["name"],
        simage      = supplement["image_url"],
        buy_link    = supplement["buy_link"],
    )


# ── 9d. Supplement Marketplace ────────────────────────────────────────────────
@app.route("/market")
def market():
    return render_template(
        "market.html",
        supplement_image = list(SUPPLEMENT_INFO["supplement image"]),
        supplement_name  = list(SUPPLEMENT_INFO["supplement name"]),
        disease          = list(DISEASE_INFO["disease_name"]),
        buy              = list(SUPPLEMENT_INFO["buy link"]),
    )


# ── 9e. Prediction History ────────────────────────────────────────────────────
@app.route("/history")
def history():
    return render_template(
        "history.html",
        history = list(reversed(prediction_history)),  # newest first
    )


# ── 9f. API: Predict (JSON endpoint) ─────────────────────────────────────────
@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    JSON API endpoint.
    POST multipart/form-data with field 'image'.
    Returns JSON prediction result.
    """
    if "image" not in request.files:
        return jsonify({"error": "No image field in request"}), 400

    file = request.files["image"]
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    filename  = safe_filename(file.filename)
    file_path = os.path.join(cfg.UPLOAD_FOLDER, filename)
    file.save(file_path)

    try:
        pred       = predict(file_path)
        disease    = get_disease_data(pred["index"])
        supplement = get_supplement_data(pred["index"])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "disease_name":  pred["disease_name"],
        "confidence":    pred["confidence"],
        "low_confidence":pred["low_conf"],
        "inference_ms":  pred["inference_ms"],
        "top_k":         pred["top_k"],
        "description":   disease["description"],
        "steps":         disease["steps"],
        "supplement":    supplement,
    })


# ── 9g. Static pages ──────────────────────────────────────────────────────────
@app.route("/contact")
def contact():
    return render_template("contact-us.html")


@app.route("/mobile-device")
def mobile_device_detected_page():
    return render_template("mobile-device.html")


# ── 9h. Health check ──────────────────────────────────────────────────────────
@app.route("/health")
def health():
    return jsonify({
        "status":          "ok",
        "model_loaded":    MODEL is not None,
        "total_classes":   cfg.NUM_CLASSES,
        "predictions_run": len(prediction_history),
    })


# ── 9i. Error handlers ────────────────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    return render_template("404.html"), 404


@app.errorhandler(413)
def file_too_large(e):
    flash(
        f"File too large. Maximum allowed size is {cfg.MAX_CONTENT_MB} MB.",
        "error",
    )
    return redirect(url_for("ai_engine_page"))


@app.errorhandler(500)
def server_error(e):
    log.error(f"Server error: {e}")
    return render_template("500.html"), 500


# ── 10. STARTUP ───────────────────────────────────────────────────────────────
def load_csv(path: str, encoding: str = "cp1252") -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required data file not found: {path}\n"
            "Place disease_info.csv and supplement_info.csv in the app root."
        )
    return pd.read_csv(path, encoding=encoding)


# Load data at startup — fail fast if files are missing
log.info("Loading data files ...")
DISEASE_INFO    = load_csv(cfg.DISEASE_CSV)
SUPPLEMENT_INFO = load_csv(cfg.SUPPLEMENT_CSV)
log.info(f"  disease_info.csv    → {len(DISEASE_INFO)} rows")
log.info(f"  supplement_info.csv → {len(SUPPLEMENT_INFO)} rows")

# Load class mapping (from training output) or fall back to hardcoded dict
IDX_MAP_PATH = "results/idx_to_class.json"
if os.path.exists(IDX_MAP_PATH):
    with open(IDX_MAP_PATH) as f:
        IDX_TO_CLASS: dict[int, str] = {
            int(k): v for k, v in json.load(f).items()
        }
    log.info(f"Class mapping loaded from {IDX_MAP_PATH}")
else:
    # Fallback: build from disease_info.csv
    IDX_TO_CLASS = {
        int(row["index"]): row["disease_name"]
        for _, row in DISEASE_INFO.iterrows()
    }
    log.warning(
        f"{IDX_MAP_PATH} not found — built class map from disease_info.csv"
    )

# Load CNN model
log.info("Loading model ...")
MODEL = load_model()
log.info("Startup complete. Ready to serve.")


# ── 11. ENTRY POINT ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(
        debug=False,    # set True only during local development
        host="0.0.0.0",
        port=5000,
    )
