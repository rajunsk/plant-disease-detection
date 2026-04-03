# =============================================================================
# Plant Disease Detection — Complete CNN Training Pipeline
# Dataset : PlantVillage (39 classes)
# Model   : Custom 4-block VGG-style CNN (PyTorch)
# Author  : Nadimpalli Siddhartha Kumar Raju
# Redg No : 99220040647
# =============================================================================

# ── 1. IMPORTS ────────────────────────────────────────────────────────────────
import os
import time
import copy
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# ── 2. CONFIG ─────────────────────────────────────────────────────────────────
class Config:
    # Paths
    DATA_DIR        = "PlantVillage"          # root folder of dataset
    MODEL_SAVE_PATH = "plant_disease_model_1_latest.pt"
    RESULTS_DIR     = "results"

    # Data
    IMG_SIZE        = 224
    TRAIN_SPLIT     = 0.80
    VAL_SPLIT       = 0.10
    TEST_SPLIT      = 0.10
    NUM_CLASSES     = 39

    # Training
    BATCH_SIZE      = 32
    NUM_EPOCHS      = 30
    LEARNING_RATE   = 1e-3
    WEIGHT_DECAY    = 1e-4
    LR_STEP_SIZE    = 7          # StepLR: decay every N epochs
    LR_GAMMA        = 0.5        # multiply LR by this at each step

    # Early stopping
    PATIENCE        = 5          # stop if val loss doesn't improve for N epochs

    # Misc
    SEED            = 42
    NUM_WORKERS     = 4
    DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = Config()
os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

torch.manual_seed(cfg.SEED)
np.random.seed(cfg.SEED)
print(f"Using device: {cfg.DEVICE}")


# ── 3. DATA TRANSFORMS ────────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.2,
        hue=0.1
    ),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],   # ImageNet stats (good default)
        std=[0.229, 0.224, 0.225]
    ),
])

val_test_transform = transforms.Compose([
    transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ── 4. DATASET & DATALOADERS ──────────────────────────────────────────────────
def load_data(data_dir: str):
    """
    Loads PlantVillage from data_dir, splits 80/10/10 into train/val/test.
    Returns three DataLoaders and the class-to-index mapping.
    """
    full_dataset = datasets.ImageFolder(
        root=data_dir,
        transform=train_transform
    )

    n_total = len(full_dataset)
    n_train = int(n_total * cfg.TRAIN_SPLIT)
    n_val   = int(n_total * cfg.VAL_SPLIT)
    n_test  = n_total - n_train - n_val

    train_set, val_set, test_set = random_split(
        full_dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(cfg.SEED)
    )

    # Val and test should NOT use augmentation
    val_set.dataset  = copy.deepcopy(full_dataset)
    val_set.dataset.transform  = val_test_transform
    test_set.dataset = copy.deepcopy(full_dataset)
    test_set.dataset.transform = val_test_transform

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True
    )

    class_to_idx = full_dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    print(f"\nDataset split:")
    print(f"  Train : {n_train:,} images")
    print(f"  Val   : {n_val:,} images")
    print(f"  Test  : {n_test:,} images")
    print(f"  Classes: {len(class_to_idx)}")

    return train_loader, val_loader, test_loader, idx_to_class


# ── 5. CNN MODEL ──────────────────────────────────────────────────────────────
class CNN(nn.Module):
    """
    4-block VGG-style CNN.
    Each block: Conv → BN → ReLU → Conv → BN → ReLU → MaxPool
    Channels:   32 → 64 → 128 → 256
    Dense head: AdaptiveAvgPool → Dropout → FC(1024) → Dropout → FC(K)
    """

    def __init__(self, K: int):
        super(CNN, self).__init__()

        def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )

        self.conv_layers = nn.Sequential(
            conv_block(3,   32),    # 224 → 112
            conv_block(32,  64),    # 112 → 56
            conv_block(64,  128),   # 56  → 28
            conv_block(128, 256),   # 28  → 14
        )

        # AdaptiveAvgPool: always outputs 7×7 regardless of input size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))  # 256×7×7 = 12544

        self.dense_layers = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(256 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(1024, K),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """He (Kaiming) initialisation — optimal for ReLU networks."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = self.adaptive_pool(x)
        x = x.flatten(1)           # (batch, 12544)
        x = self.dense_layers(x)   # (batch, K)
        return x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── 6. TRAINING LOOP ──────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds        = outputs.argmax(dim=1)
        correct      += (preds == labels).sum().item()
        total        += images.size(0)

    epoch_loss = running_loss / total
    epoch_acc  = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss    = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds        = outputs.argmax(dim=1)
        correct      += (preds == labels).sum().item()
        total        += images.size(0)

    return running_loss / total, correct / total


def train(model, train_loader, val_loader, cfg):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg.LR_STEP_SIZE,
        gamma=cfg.LR_GAMMA
    )

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   []
    }

    best_val_acc  = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_ctr  = 0

    print(f"\n{'Epoch':>6} {'Train Loss':>12} {'Train Acc':>10} "
          f"{'Val Loss':>10} {'Val Acc':>10} {'LR':>10}")
    print("-" * 62)

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, cfg.DEVICE
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, cfg.DEVICE
        )
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        lr_now = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        print(f"{epoch:>6}  {train_loss:>11.4f}  {train_acc*100:>9.2f}%"
              f"  {val_loss:>9.4f}  {val_acc*100:>9.2f}%  {lr_now:.2e}"
              f"  ({elapsed:.1f}s)")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc   = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, cfg.MODEL_SAVE_PATH)
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= cfg.PATIENCE:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement for {cfg.PATIENCE} epochs).")
                break

    print(f"\nBest Val Accuracy: {best_val_acc * 100:.2f}%")
    print(f"Model saved to   : {cfg.MODEL_SAVE_PATH}")

    model.load_state_dict(best_model_wts)
    return model, history


# ── 7. EVALUATION ─────────────────────────────────────────────────────────────
@torch.no_grad()
def get_all_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        probs   = torch.softmax(outputs, dim=1)
        preds   = probs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

    return (
        np.array(all_preds),
        np.array(all_labels),
        np.array(all_probs)
    )


def print_evaluation_report(preds, labels, idx_to_class, split="Test"):
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    acc  = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average='macro', zero_division=0)
    rec  = recall_score(labels, preds, average='macro', zero_division=0)
    f1   = f1_score(labels, preds, average='macro', zero_division=0)

    print(f"\n{'='*55}")
    print(f" {split} Set Results")
    print(f"{'='*55}")
    print(f"  Accuracy  : {acc  * 100:.2f}%")
    print(f"  Precision : {prec * 100:.2f}%  (macro)")
    print(f"  Recall    : {rec  * 100:.2f}%  (macro)")
    print(f"  F1 Score  : {f1   * 100:.2f}%  (macro)")
    print(f"{'='*55}\n")
    print(classification_report(labels, preds,
                                 target_names=class_names,
                                 zero_division=0))

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def plot_confusion_matrix(preds, labels, idx_to_class, save_path):
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    cm = confusion_matrix(labels, preds)

    fig, ax = plt.subplots(figsize=(20, 18))
    sns.heatmap(
        cm,
        annot=True, fmt='d',
        xticklabels=class_names,
        yticklabels=class_names,
        cmap='Blues',
        ax=ax,
        linewidths=0.3,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label",      fontsize=12)
    ax.set_title("Confusion Matrix — Plant Disease CNN", fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=7)
    plt.yticks(rotation=0,  fontsize=7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved → {save_path}")


def plot_training_curves(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    axes[0].plot(epochs, history["train_loss"], 'b-o', label='Train',    ms=4)
    axes[0].plot(epochs, history["val_loss"],   'r-o', label='Val',      ms=4)
    axes[0].set_title("Loss per epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy
    train_acc_pct = [a * 100 for a in history["train_acc"]]
    val_acc_pct   = [a * 100 for a in history["val_acc"]]
    axes[1].plot(epochs, train_acc_pct, 'b-o', label='Train', ms=4)
    axes[1].plot(epochs, val_acc_pct,   'r-o', label='Val',   ms=4)
    axes[1].set_title("Accuracy per epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle("Plant Disease CNN — Training Curves", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training curves saved → {save_path}")


# ── 8. INFERENCE (single image) ───────────────────────────────────────────────
def predict_image(image_path: str,
                  model: nn.Module,
                  idx_to_class: dict,
                  device: torch.device,
                  top_k: int = 3):
    """
    Predicts plant disease from a single image file.
    Returns (predicted_class, confidence_%, top_k list).
    """
    from PIL import Image as PILImage

    image = PILImage.open(image_path).convert("RGB")
    tensor = val_test_transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(tensor)
        probs  = torch.softmax(output[0], dim=0)

    top_probs, top_idxs = torch.topk(probs, k=top_k)
    top_results = [
        {
            "class":      idx_to_class[idx.item()],
            "confidence": round(prob.item() * 100, 2)
        }
        for prob, idx in zip(top_probs, top_idxs)
    ]

    best_class = top_results[0]["class"]
    best_conf  = top_results[0]["confidence"]

    print(f"\nPrediction for: {image_path}")
    for i, r in enumerate(top_results, 1):
        print(f"  {i}. {r['class']:<45} {r['confidence']:>6.2f}%")

    if best_conf < 60:
        print("  ⚠  Low confidence — consider re-uploading a clearer image.")

    return best_class, best_conf, top_results


# ── 9. MAIN ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # 9a. Load data
    train_loader, val_loader, test_loader, idx_to_class = load_data(cfg.DATA_DIR)

    # Save class mapping for Flask app
    with open(os.path.join(cfg.RESULTS_DIR, "idx_to_class.json"), "w") as f:
        json.dump(idx_to_class, f, indent=2)
    print("Class mapping saved → results/idx_to_class.json")

    # 9b. Build model
    model = CNN(K=cfg.NUM_CLASSES).to(cfg.DEVICE)
    print(f"\nModel parameters: {count_parameters(model):,}")

    # 9c. Train
    model, history = train(model, train_loader, val_loader, cfg)

    # 9d. Plot training curves
    plot_training_curves(
        history,
        save_path=os.path.join(cfg.RESULTS_DIR, "training_curves.png")
    )

    # 9e. Evaluate on test set
    test_preds, test_labels, test_probs = get_all_predictions(
        model, test_loader, cfg.DEVICE
    )

    metrics = print_evaluation_report(
        test_preds, test_labels, idx_to_class, split="Test"
    )

    # 9f. Save metrics to JSON
    with open(os.path.join(cfg.RESULTS_DIR, "test_metrics.json"), "w") as f:
        json.dump({k: round(float(v), 4) for k, v in metrics.items()}, f, indent=2)

    # 9g. Confusion matrix
    plot_confusion_matrix(
        test_preds, test_labels, idx_to_class,
        save_path=os.path.join(cfg.RESULTS_DIR, "confusion_matrix.png")
    )

    # 9h. Quick inference demo
    # predict_image("sample_leaf.jpg", model, idx_to_class, cfg.DEVICE)

    print("\nAll done. Outputs saved in results/")
