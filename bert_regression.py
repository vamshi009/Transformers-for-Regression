# STEP 1: Install required libraries (run this only once)
# !pip install transformers datasets scikit-learn pandas

# STEP 2: Import libraries
import pandas as pd
from datasets import Dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import mean_squared_error
import numpy as np
import torch

# STEP 3: Load your custom dataset
df = pd.read_csv("train.csv")  # Replace with your CSV path
print(df.head())

# STEP 4: Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# STEP 5: Tokenization
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(example):
    return tokenizer(example["catalog_content"], truncation=True, padding="max_length", max_length=128)

tokenized_dataset = dataset.map(tokenize)

# STEP 6: Rename target column to 'labels' and set format
tokenized_dataset = tokenized_dataset.rename_column("price", "labels")
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# STEP 7: Split into train and test sets
train_test_split = tokenized_dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]

# STEP 8: Load BERT model for regression
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=1,
    problem_type="regression"
)

# STEP 9: Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    learning_rate=2e-5,
    logging_dir="./logs",
    save_total_limit=1
)

# STEP 10: Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# STEP 11: Train the model
trainer.train()

trainer.save_model('textualregression')
# STEP 12: Predict and evaluate
preds = trainer.predict(test_dataset)

predictions = preds.predictions.flatten()
true_values = preds.label_ids

mse = mean_squared_error(true_values, predictions)
print("\n📊 Mean Squared Error (MSE):", mse)
print("\n🔍 Sample predictions:")
for i in range(5):
    print(f"Predicted: {predictions[i]:.2f}, Actual: {true_values[i]:.2f}")



def compute_smape(true_vals, pred_vals):
    t = np.asarray(true_vals, dtype=float)
    p = np.asarray(pred_vals, dtype=float)
    denom = (np.abs(t) + np.abs(p)) / 2.0
    denom = np.where(denom == 0, 1e-8, denom)
    return np.mean(np.abs(p - t) / denom) * 100.0

print(compute_smape(true_values, predictions))
