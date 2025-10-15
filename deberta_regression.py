# deberta_price_regression_csv.py

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW

from tqdm import tqdm
import pandas as pd
import numpy as np

# -----------------------------
# 1. CONFIGURATION
# -----------------------------
MODEL_NAME = "microsoft/deberta-v3-base"   # try "deberta-v3-small" for smaller GPU
CSV_PATH = "train.csv"                  # path to your CSV file
TEXT_COLUMN = "catalog_content"
PRICE_COLUMN = "price"

MAX_LEN = 128
BATCH_SIZE = 8
EPOCHS = 0
LR = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# 2. LOAD DATA FROM CSV
# -----------------------------
print(f"Loading dataset from {CSV_PATH}...")
df = pd.read_csv(CSV_PATH)

# Clean up data (drop missing values, reset index)
df = df[[TEXT_COLUMN, PRICE_COLUMN]].dropna().reset_index(drop=True)

print(f"Loaded {len(df)} rows.")

# Optional: log-transform prices if large variation
# df[PRICE_COLUMN] = np.log1p(df[PRICE_COLUMN])

# Split train/validation (80/20)
train_df = df.sample(frac=0.7, random_state=42)
rem_df = df.drop(train_df.index)

val_df = rem_df.sample(frac=0.3, random_state=42)
held_out_test_df =rem_df.drop(val_df.index)




# -----------------------------
# 3. TOKENIZER & DATASET CLASS
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class ProductDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.texts = df[TEXT_COLUMN].tolist()
        self.prices = df[PRICE_COLUMN].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        price = torch.tensor(self.prices[idx], dtype=torch.float)
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "price": price
        }

train_dataset = ProductDataset(train_df, tokenizer, MAX_LEN)
val_dataset = ProductDataset(val_df, tokenizer, MAX_LEN)
held_out_dataset = ProductDataset(held_out_test_df, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
heldout_loader =  DataLoader(held_out_dataset, batch_size=BATCH_SIZE, shuffle=False)


# -----------------------------
# 4. MODEL DEFINITION
# -----------------------------
class DebertaForRegression(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        price = self.regressor(cls_emb)
        return price.squeeze(-1)


# -----------------------------
# 5. TRAINING & VALIDATION
# -----------------------------
model = DebertaForRegression(MODEL_NAME).to(DEVICE)
optimizer = AdamW(model.parameters(), lr=LR)
criterion = nn.MSELoss()


def compute_smape(true_vals, pred_vals):
    t = np.asarray(true_vals, dtype=float)
    p = np.asarray(pred_vals, dtype=float)
    denom = (np.abs(t) + np.abs(p)) / 2.0
    denom = np.where(denom == 0, 1e-8, denom)
    return np.mean(np.abs(p - t) / denom) * 100.0



def evaluate(model, loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            price = batch["price"].to(DEVICE)
            output = model(input_ids, attention_mask)
            preds.extend(output.cpu().numpy())
            targets.extend(price.cpu().numpy())
            print("total preds so far... ", len(preds))
    preds, targets = np.array(preds), np.array(targets)
    rmse = np.sqrt(np.mean((preds - targets) ** 2))
    print("rmse is ", rmse)
    smape = compute_smape(targets, preds)
    print("smape is ", smape)
    return rmse


model.train()

for epoch in range(EPOCHS):
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        price = batch["price"].to(DEVICE)

        output = model(input_ids, attention_mask)
        loss = criterion(output, price)
        print("loss is ", loss.item())
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    val_rmse = evaluate(model, val_loader)
    print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val RMSE = {val_rmse:.4f}")



torch.save(model.state_dict(), 'deberat_regression.pt')
# -----------------------------
# 6. TEST / INFERENCE EXAMPLE
# -----------------------------
#df = pd.read_csv('test.csv')
#enc = tokenizer(held_out_test_df[TEXT_COLUMN], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(DEVICE)

model.eval()
#with torch.no_grad():
#    preds = model(enc["input_ids"], enc["attention_mask"]).cpu().numpy()
print("working on model evaluation...")
evaluate(model, heldout_loader)



def compute_smape(true_vals, pred_vals):
    t = np.asarray(true_vals, dtype=float)
    p = np.asarray(pred_vals, dtype=float)
    denom = (np.abs(t) + np.abs(p)) / 2.0
    denom = np.where(denom == 0, 1e-8, denom)
    return np.mean(np.abs(p - t) / denom) * 100.0

print(compute_smape(held_out_test_df[PRICE_COLUMN], preds))

'''
print("\nSample Predictions:")
for text, pred in zip(test_texts, preds):
    print(f"{text}  -->  Predicted Price: ${pred:.2f}")
'''
