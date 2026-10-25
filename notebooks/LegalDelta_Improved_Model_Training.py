# ============================================================
# LegalDelta-BERT — One Improved Training Run
# XLM-RoBERTa + standardized regression + pairwise ranking loss
#
# Frozen split: 228 train / 50 validation / 79 test
# Test set is NOT loaded or evaluated in this cell.
# ============================================================

import os
import json
import random
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error

# -------------------- 1. Configuration --------------------

SEED = 42
BASE_MODEL = "xlm-roberta-base"
MAX_LENGTH = 256
BATCH_SIZE = 8
EPOCHS = 6
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
RANK_WEIGHT = 0.25

BASE = "/content/drive/MyDrive/LegalDelta_LMP"
OUT = os.path.join(BASE, "LegalDelta_improved_candidate")
os.makedirs(OUT, exist_ok=True)

# Keep the recovered checkpoint untouched.
assert torch.cuda.is_available(), (
    "GPU not available. Select Runtime > Change runtime type > T4 GPU."
)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda")

# -------------------- 2. Load only train and validation --------------------

train_df = pd.read_csv(os.path.join(BASE, "train_source_grouped.csv"))
val_df = pd.read_csv(os.path.join(BASE, "validation_source_grouped.csv"))

for df in (train_df, val_df):
    df["LMP score"] = pd.to_numeric(df["LMP score"], errors="raise")
    assert df["LMP score"].between(1, 10).all()

assert len(train_df) == 228, f"Unexpected training size: {len(train_df)}"
assert len(val_df) == 50, f"Unexpected validation size: {len(val_df)}"
assert set(train_df["Source"]).isdisjoint(set(val_df["Source"])), (
    "Train/validation source overlap detected."
)

# Standardization parameters are learned from TRAINING data only.
target_mean = float(train_df["LMP score"].mean())
target_std = float(train_df["LMP score"].std(ddof=0))
assert target_std > 0

train_df = train_df.copy()
val_df = val_df.copy()

train_df["target"] = (train_df["LMP score"] - target_mean) / target_std
val_df["target"] = (val_df["LMP score"] - target_mean) / target_std

print("GPU:", torch.cuda.get_device_name(0))
print(f"Train: {len(train_df)} | Validation: {len(val_df)}")
print(f"Training target mean/std: {target_mean:.4f} / {target_std:.4f}")
print("Test set remains untouched.")

# -------------------- 3. Tokenization and data loaders --------------------

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

class PairDataset(Dataset):
    def __init__(self, frame):
        self.original = frame["Original"].astype(str).tolist()
        self.simplified = frame["Simplified"].astype(str).tolist()
        self.targets = frame["target"].astype(float).tolist()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        item = tokenizer(
            self.original[index],
            self.simplified[index],
            truncation=True,
            max_length=MAX_LENGTH,
        )
        item["labels"] = self.targets[index]
        return item

def collate_fn(features):
    labels = torch.tensor(
        [f["labels"] for f in features], dtype=torch.float32
    )
    inputs = [{k: v for k, v in f.items() if k != "labels"}
              for f in features]
    batch = tokenizer.pad(inputs, padding=True, return_tensors="pt")
    batch["labels"] = labels
    return batch

train_loader = DataLoader(
    PairDataset(train_df),
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    generator=torch.Generator().manual_seed(SEED),
)

val_loader = DataLoader(
    PairDataset(val_df),
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
)

# -------------------- 4. Model --------------------

class LegalDeltaRegressor(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model)
        hidden_size = self.encoder.config.hidden_size
        self.dropout1 = nn.Dropout(0.2)
        self.hidden = nn.Linear(hidden_size, 256)
        self.activation = nn.GELU()
        self.dropout2 = nn.Dropout(0.2)
        self.regressor = nn.Linear(256, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        cls = outputs.last_hidden_state[:, 0, :]
        x = self.dropout1(cls)
        x = self.activation(self.hidden(x))
        x = self.dropout2(x)
        return self.regressor(x).squeeze(-1)

model = LegalDeltaRegressor(BASE_MODEL).to(device)

# -------------------- 5. Combined regression + ranking loss --------------------

def combined_loss(predictions, targets):
    regression_loss = F.smooth_l1_loss(predictions, targets)

    # Compare each unique pair within the batch whose target scores differ.
    target_diff = targets[:, None] - targets[None, :]
    pred_diff = predictions[:, None] - predictions[None, :]

    upper_triangle = torch.triu(
        torch.ones_like(target_diff, dtype=torch.bool), diagonal=1
    )
    valid = upper_triangle & (target_diff != 0)

    if valid.any():
        direction = torch.sign(target_diff[valid])
        ranking_loss = F.softplus(-direction * pred_diff[valid]).mean()
    else:
        ranking_loss = predictions.new_tensor(0.0)

    total = regression_loss + RANK_WEIGHT * ranking_loss
    return total, regression_loss.detach(), ranking_loss.detach()

# -------------------- 6. Optimizer and scheduler --------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps,
)

scaler = torch.amp.GradScaler("cuda", enabled=True)

# -------------------- 7. Validation --------------------

def evaluate_validation():
    model.eval()
    predictions, targets = [], []

    with torch.no_grad():
        for batch in val_loader:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                preds = model(**batch)

            predictions.extend(preds.float().cpu().numpy().tolist())
            targets.extend(labels.cpu().numpy().tolist())

    # Convert standardized values back to the original LMP scale.
    predictions = np.asarray(predictions) * target_std + target_mean
    targets = np.asarray(targets) * target_std + target_mean

    return {
        "rmse": float(np.sqrt(mean_squared_error(targets, predictions))),
        "mae": float(mean_absolute_error(targets, predictions)),
        "pearson": float(pearsonr(targets, predictions).statistic)
            if np.std(predictions) > 0 else 0.0,
        "spearman": float(spearmanr(targets, predictions).statistic)
            if np.std(predictions) > 0 else 0.0,
        "prediction_mean": float(predictions.mean()),
        "prediction_std": float(predictions.std()),
    }

# -------------------- 8. Train once; select by validation RMSE --------------------

best_rmse = float("inf")
best_state = None
best_epoch = None
history = []

print("\nStarting the single improved training run...")

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    running_regression = 0.0
    running_ranking = 0.0

    for batch in train_loader:
        labels = batch.pop("labels").to(device)
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            predictions = model(**batch)
            loss, reg_loss, rank_loss = combined_loss(predictions, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running_loss += loss.item()
        running_regression += reg_loss.item()
        running_ranking += rank_loss.item()

    val_metrics = evaluate_validation()

    record = {
        "epoch": epoch,
        "train_loss": running_loss / len(train_loader),
        "train_regression_loss": running_regression / len(train_loader),
        "train_ranking_loss": running_ranking / len(train_loader),
        **{f"val_{k}": v for k, v in val_metrics.items()},
    }
    history.append(record)

    print(
        f"Epoch {epoch}/{EPOCHS} | "
        f"Train loss {record['train_loss']:.4f} | "
        f"Val RMSE {val_metrics['rmse']:.4f} | "
        f"Val MAE {val_metrics['mae']:.4f} | "
        f"Val Pearson {val_metrics['pearson']:.4f}"
    )

    if np.isfinite(val_metrics["rmse"]) and val_metrics["rmse"] < best_rmse:
        best_rmse = val_metrics["rmse"]
        best_epoch = epoch
        best_state = {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
        }

        # Save the best candidate as training proceeds.
        torch.save(best_state, os.path.join(OUT, "best_model_state.pt"))

        tokenizer.save_pretrained(os.path.join(OUT, "tokenizer"))
        model.encoder.config.save_pretrained(os.path.join(OUT, "encoder_config"))

# -------------------- 9. Restore best epoch and save artifacts --------------------

assert best_state is not None, "No valid checkpoint was selected."

model.load_state_dict(best_state)
model.to(device)

final_val_metrics = evaluate_validation()

pd.DataFrame(history).to_csv(
    os.path.join(OUT, "training_history.csv"), index=False
)

metadata = {
    "project": "LegalDelta-BERT",
    "base_model": BASE_MODEL,
    "seed": SEED,
    "max_length": MAX_LENGTH,
    "epochs_requested": EPOCHS,
    "best_epoch": best_epoch,
    "learning_rate": LEARNING_RATE,
    "batch_size": BATCH_SIZE,
    "ranking_loss_weight": RANK_WEIGHT,
    "regression_loss": "SmoothL1",
    "target_mean_train": target_mean,
    "target_std_train": target_std,
    "train_size": len(train_df),
    "validation_size": len(val_df),
    "best_validation_metrics": final_val_metrics,
    "test_evaluated": False,
}

with open(os.path.join(OUT, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print("\n========== RUN COMPLETE ==========")
print("Best epoch:", best_epoch)
print("Best validation metrics:")
for key, value in final_val_metrics.items():
    print(f"{key}: {value:.4f}")

print("\nSaved candidate to:", OUT)
print("Best weights: best_model_state.pt")
print("Training history: training_history.csv")
print("Metadata: metadata.json")
print("Test set was not loaded or evaluated.")