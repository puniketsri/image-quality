## src/train.py
## Full training loop for the Image Quality Gate model.
## Run from project root: python src/train.py

# SSL fix for Mac — must be at the very top before any other imports
import ssl, certifi, os
ssl._create_default_https_context = ssl.create_default_context
os.environ["SSL_CERT_FILE"]      = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import mlflow
import sys
sys.path.insert(0, "src")
from model   import QualityGateModel
from dataset import QualityDataset

# ══════════════════════════════════════════════════════════════════
# CONFIG — change these values to experiment
# ══════════════════════════════════════════════════════════════════
BATCH_SIZE   = 16      # images per batch (reduce to 8 if Mac runs slow)
NUM_EPOCHS   = 20      # how many full passes through the data
LR           = 1e-4    # learning rate (0.0001)
WEIGHT_DECAY = 1e-4    # L2 regularisation strength
DATA_DIR     = "data/augmented"
MODEL_DIR    = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


def train():
    # ── 1. DEVICE ──────────────────────────────────────────────────────
    # Use GPU if available, otherwise CPU
    # On your MacBook Air this will print: Training on: cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # ── 2. DATA ────────────────────────────────────────────────────────
    dataset    = QualityDataset(DATA_DIR)
    train_size = int(0.8 * len(dataset))
    val_size   = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,    # shuffle each epoch — important
        num_workers=0    # 0 is safest on Mac
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,   # no need to shuffle validation
        num_workers=0
    )

    # ── 3. MODEL ───────────────────────────────────────────────────────
    model = QualityGateModel().to(device)

    # Freeze EfficientNet backbone — only train the two heads
    # This prevents overfitting on your small dataset
    for param in model.features.parameters():
        param.requires_grad = False

    trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"Trainable parameters: {trainable:,}")

    # ── 4. LOSS FUNCTIONS ──────────────────────────────────────────────
    # MSELoss: for the quality score (predicting a number 0–1)
    regression_loss = nn.MSELoss()

    # CrossEntropyLoss: for the failure class (predicting which category)
    classification_loss = nn.CrossEntropyLoss()

    # ── 5. OPTIMIZER ───────────────────────────────────────────────────
    # AdamW: updates weights using gradients + momentum
    # Only pass parameters that need updating (requires_grad=True)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    # ── 6. SCHEDULER ───────────────────────────────────────────────────
    # Reduces learning rate smoothly from LR → 0 over NUM_EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS
    )

    # ── 7. MLFLOW TRACKING ─────────────────────────────────────────────
    # Records every metric so you can see charts at localhost:5000
    mlflow.set_experiment("image-quality-gate")

    with mlflow.start_run():
        mlflow.log_params({
            "batch_size": BATCH_SIZE,
            "lr":         LR,
            "epochs":     NUM_EPOCHS,
            "model":      "efficientnet-b0-frozen",
        })

        best_val_loss = float("inf")

        # ── 8. EPOCH LOOP ──────────────────────────────────────────────────
        for epoch in range(NUM_EPOCHS):

            # ════ TRAINING PHASE ═══════════════════════════════════════
            model.train()       # activates Dropout
            train_loss = 0.0

            for imgs, signals, scores, labels in train_loader:

                # Move data to same device as model
                imgs    = imgs.to(device)
                signals = signals.to(device)
                scores  = scores.to(device)
                labels  = labels.to(device)

                # ── THE FIVE STEPS (repeat for every batch) ──────────────

                # Step 1: Clear gradients from previous iteration
                optimizer.zero_grad()

                # Step 2: Forward pass — get predictions
                pred_score, pred_class = model(imgs, signals)

                # Step 3: Compute loss (how wrong are the predictions)
                loss_s = regression_loss(pred_score, scores)
                loss_c = classification_loss(pred_class, labels)
                loss   = loss_s + loss_c    # combine both losses

                # Step 4: Backward pass — compute gradients
                loss.backward()

                # Step 5: Update weights using gradients
                optimizer.step()

                train_loss += loss.item()

            # ════ VALIDATION PHASE ═════════════════════════════════════
            model.eval()        # deactivates Dropout
            val_loss = 0.0
            correct  = 0
            total    = 0

            with torch.no_grad():   # no gradients needed — saves memory
                for imgs, signals, scores, labels in val_loader:
                    imgs    = imgs.to(device)
                    signals = signals.to(device)
                    scores  = scores.to(device)
                    labels  = labels.to(device)

                    pred_score, pred_class = model(imgs, signals)

                    loss_s = regression_loss(pred_score, scores)
                    loss_c = classification_loss(pred_class, labels)
                    val_loss += (loss_s + loss_c).item()

                    # Count how many class predictions are correct
                    predicted = pred_class.argmax(dim=1)
                    correct  += (predicted == labels).sum().item()
                    total    += labels.size(0)

            # ════ LOG METRICS ══════════════════════════════════════════
            avg_train = train_loss / len(train_loader)
            avg_val   = val_loss   / len(val_loader)
            accuracy  = correct / total

            print(
                f"Epoch {epoch+1:02d}/{NUM_EPOCHS} | "
                f"train={avg_train:.4f} | "
                f"val={avg_val:.4f} | "
                f"acc={accuracy:.3f}"
            )

            mlflow.log_metrics({
                "train_loss": avg_train,
                "val_loss":   avg_val,
                "val_acc":    accuracy,
            }, step=epoch)

            # Reduce learning rate after each epoch
            scheduler.step()

            # ════ SAVE BEST MODEL ══════════════════════════════════════
            # Save whenever validation loss improves
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                save_path = MODEL_DIR / "best_model.pth"
                torch.save(model.state_dict(), save_path)
                print(f"  ✓ Saved best model (val={avg_val:.4f})")

    print(f"\nTraining complete.")
    print(f"Best val_loss: {best_val_loss:.4f}")
    print(f"Model saved at: models/best_model.pth")


if __name__ == "__main__":
    train()
