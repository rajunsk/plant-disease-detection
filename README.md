# 🌿 PlantGuard AI — Plant Disease Detection

A full-stack deep learning web application that detects plant diseases from leaf images in real time.


---

## 📌 Demo

| Upload leaf | Get diagnosis | Browse treatments |
|---|---|---|
| Drag & drop any leaf image | AI returns disease name + confidence % | View supplement recommendations |

---

## 🧠 Model Architecture

Custom **4-block VGG-style CNN** built in PyTorch from scratch:

```
Input (3 × 224 × 224)
  └── Block 1: Conv2d(3→32)   → BN → ReLU → Conv2d(32→32)   → BN → ReLU → MaxPool  →  32 × 112 × 112
  └── Block 2: Conv2d(32→64)  → BN → ReLU → Conv2d(64→64)   → BN → ReLU → MaxPool  →  64 × 56 × 56
  └── Block 3: Conv2d(64→128) → BN → ReLU → Conv2d(128→128) → BN → ReLU → MaxPool  → 128 × 28 × 28
  └── Block 4: Conv2d(128→256)→ BN → ReLU → Conv2d(256→256) → BN → ReLU → MaxPool  → 256 × 14 × 14
  └── AdaptiveAvgPool2d(7×7) → Flatten (12544)
  └── Dropout(0.4) → Linear(12544→1024) → ReLU → Dropout(0.4) → Linear(1024→39)
Output: Logits (39 classes)
```

**Key design choices:**
- Double conv per block learns richer features before spatial downsampling
- BatchNorm after every conv for stable, fast training
- AdaptiveAvgPool2d makes the model input-size agnostic
- He (Kaiming) initialisation optimal for ReLU activations
- Label smoothing (0.1) reduces overconfidence

---

## 📊 Results

| Metric | Score |
|--------|-------|
| Test Accuracy | 92% |
| Macro Precision | ~91% |
| Macro Recall | ~91% |
| Inference Time | < 1 second (CPU) |
| Classes | 39 |
| Dataset | PlantVillage |

---

## 🌾 Supported Plants & Diseases

14 plant species · 27 disease categories · 12 healthy classes

| Plant | Conditions |
|-------|-----------|
| Apple | Scab, Black Rot, Cedar Rust, Healthy |
| Blueberry | Healthy |
| Cherry | Powdery Mildew, Healthy |
| Corn | Gray Leaf Spot, Common Rust, Northern Leaf Blight, Healthy |
| Grape | Black Rot, Black Measles, Leaf Blight, Healthy |
| Orange | Citrus Greening (HLB) |
| Peach | Bacterial Spot, Healthy |
| Pepper | Bacterial Spot, Healthy |
| Potato | Early Blight, Late Blight, Healthy |
| Raspberry | Healthy |
| Soybean | Healthy |
| Squash | Powdery Mildew |
| Strawberry | Leaf Scorch, Healthy |
| Tomato | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy |

---

## 🗂️ Project Structure

```
plant-disease-detection/
├── app.py                          # Flask application (all routes + inference)
├── CNN.py                          # Model architecture
├── plant_disease_cnn.py            # Training pipeline (run once to train)
├── requirements.txt
├── disease_info.csv                # 39 disease descriptions + treatment steps
├── supplement_info.csv             # 39 supplement recommendations + buy links
├── plant_disease_model_1_latest.pt # Trained weights (auto-downloaded from Drive)
├── results/
│   ├── idx_to_class.json           # Class index mapping
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   └── test_metrics.json
├── static/
│   └── uploads/                    # User-uploaded images (auto-created)
└── templates/
    ├── base.html                   # Shared layout + nav
    ├── home.html                   # Landing page
    ├── index.html                  # Upload page
    ├── submit.html                 # Results page
    ├── market.html                 # Supplement marketplace
    ├── history.html                # Prediction history
    ├── contact-us.html             # Contact / about
    ├── mobile-device.html          # Mobile warning
    ├── 404.html
    └── 500.html
```

---

## 🚀 Setup & Run

### 1. Clone the repository
```bash
git clone https://github.com/rajunsk/plant-disease-detection.git
cd plant-disease-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
python app.py
```
The model weights are **automatically downloaded** from Google Drive on first run.
Open `http://localhost:5000` in your browser.

### 4. (Optional) Retrain the model
```bash
# Download PlantVillage dataset from:
# https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
# Extract to a folder named "PlantVillage/" then:
python plant_disease_cnn.py
```

---

## 🌐 API Endpoint

A JSON API is available for programmatic access:

```bash
curl -X POST http://localhost:5000/api/predict \
     -F "image=@your_leaf.jpg"
```

**Response:**
```json
{
  "disease_name": "Tomato___Early_blight",
  "confidence": 94.27,
  "low_confidence": false,
  "inference_ms": 312.4,
  "top_k": [
    {"index": 30, "disease_name": "Tomato___Early_blight", "confidence": 94.27},
    {"index": 38, "disease_name": "Tomato___healthy",      "confidence": 3.11},
    {"index": 31, "disease_name": "Tomato___Late_blight",  "confidence": 1.94}
  ],
  "description": "Early blight is one of the most common tomato diseases...",
  "steps": "Prune or stake plants to improve air circulation...",
  "supplement": {
    "name": "NATIVO FUNGICIDE",
    "image_url": "...",
    "buy_link": "https://..."
  }
}
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Model | PyTorch, Custom CNN |
| Backend | Python, Flask |
| Data | Pandas, NumPy |
| Image processing | Pillow, torchvision |
| Model download | gdown |
| Evaluation | scikit-learn, matplotlib, seaborn |
| Frontend | HTML5, CSS3, Vanilla JS |
| Deployment | gunicorn |

---

## 📈 Training Details

| Parameter | Value |
|-----------|-------|
| Dataset | PlantVillage (54,305 images) |
| Train / Val / Test | 80% / 10% / 10% |
| Batch size | 32 |
| Optimizer | Adam (lr=1e-3, wd=1e-4) |
| Scheduler | StepLR (step=7, γ=0.5) |
| Augmentation | Flip, Rotate, ColorJitter, Affine |
| Epochs | 30 (early stopping, patience=5) |
| Loss | CrossEntropyLoss (label_smoothing=0.1) |

---

## 👨‍💻 Author

**Nadimpalli Siddhartha Kumar Raju**
B.Tech CSE (Data Science), Kalasalingam Academy of Research and Education, 2022–2026

📧 nskraju45@gmail.com | 📱 +91 9441194020 | 📍 Bhimavaram, India

*Actively seeking Data Analyst / Data Scientist roles.*

---

## 📄 License

This project is open for educational and portfolio use.
Dataset: [PlantVillage on Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
