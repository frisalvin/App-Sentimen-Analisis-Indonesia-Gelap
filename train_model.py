import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

# =====================
# 1. Load & Persiapkan Data
# =====================
data = pd.read_csv("data/Hasil_Labelling_Data.csv")
data.dropna(subset=['steming_data', 'Sentiment'], inplace=True)
data = data[['steming_data', 'Sentiment']]

label_map = {"Negatif": 0, "Netral": 1, "Positif": 2}
data['label'] = data['Sentiment'].map(label_map)

train_texts, val_texts, train_labels, val_labels = train_test_split(
    data['steming_data'],
    data['label'],
    test_size=0.2,
    stratify=data['label'],
    random_state=42
)

# =====================
# 2. Oversampling Data Train
# =====================
train_df = pd.DataFrame({'text': train_texts, 'label': train_labels})
max_count = train_df['label'].value_counts().max()

oversampled = []
for label in train_df['label'].unique():
    subset = train_df[train_df['label'] == label]
    resampled = resample(subset, replace=True, n_samples=max_count, random_state=42)
    oversampled.append(resampled)

balanced_train_df = pd.concat(oversampled).sample(frac=1, random_state=42).reset_index(drop=True)

# =====================
# 3. Tokenizer dan Encoding
# =====================
MODEL_NAME = "indobenchmark/indobert-base-p1"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def encode(texts):
    return tokenizer(
        texts.tolist(),
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )

train_encodings = encode(balanced_train_df['text'])
val_encodings = encode(val_texts)

# =====================
# 4. Dataset Class
# =====================
class IndoBERTDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels.reset_index(drop=True)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

train_dataset = IndoBERTDataset(train_encodings, balanced_train_df['label'])
val_dataset = IndoBERTDataset(val_encodings, val_labels.reset_index(drop=True))

# =====================
# 5. Class Weights & Model
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.array([0, 1, 2]),
    y=balanced_train_df['label'].values
)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
model.to(device)

loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)

# =====================
# 6. Custom Trainer
# =====================
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

# =====================
# 7. TrainingArguments
# =====================
training_args = TrainingArguments(
    output_dir="./results_indobert",
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    save_total_limit=1
)

# =====================
# 8. Trainer dan Pelatihan
# =====================
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()
trainer.evaluate()

# =====================
# 9. Save Model & Tokenizer
# =====================
model.save_pretrained("model")
tokenizer.save_pretrained("model/tokenizer")

print("✅ Model dan tokenizer IndoBERT berhasil disimpan.")