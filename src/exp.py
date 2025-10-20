# import torch
# if torch.cuda.is_available():
#     print("CUDA is available. Using GPU.")
# else:
#     print("CUDA is not available. Using CPU.")
# import pickle, os

# path = os.path.join("data", "adversarial_test", "politifact_test_adv_A.pkl")
# with open(path, "rb") as f:
#     data = pickle.load(f)

# print(type(data))
# if isinstance(data, list):
#     print("Length:", len(data))
#     print("First item:", data[0])
# elif isinstance(data, dict):
#     print("Keys:", list(data.keys())[:10])
#     print("First item type:", type(next(iter(data.values()))))
# import pickle
# import os

# # Path to your pickle file
# pkl_file = os.path.join("data", "adversarial_test", "politifact_test_adv_A.pkl")
# # pkl_file = os.path.join("data", "news_articles", "politifact_test")
# # Check if the file exists
# if not os.path.exists(pkl_file):
#     raise FileNotFoundError(f"{pkl_file} not found!")

# # Load the pickle file
# with open(pkl_file, "rb") as f:
#     data = pickle.load(f)

# # Inspect the type of the data
# print(f"Type of data: {type(data)}")

# # Ensure it's a dictionary
# if isinstance(data, dict):
#     print(f"Keys in the dict: {list(data.keys())}")
    
#     news_list = data.get("news", [])
#     labels_list = data.get("labels", [])

#     if news_list and labels_list:
#         # Filter news with label == 1 (level 1)
#         level1_news = [news for news, label in zip(news_list, labels_list) if label == 1]
#         print(f"Total number of news with level 1: {len(level1_news)}\n")

#         # Print first 5 level 1 news articles
#         print("First 5 news articles with level 1:")
#         for i, article in enumera
# evaluate_roberta_variants.py
# .......THIS IS THE BEST WORKING CODE AS OF NOW.........
# import os
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from transformers import RobertaTokenizer, RobertaModel
# from sklearn.metrics import precision_recall_fscore_support, accuracy_score
# import numpy as np
# import pickle

# # ---------------- CONFIG ----------------
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# MODEL_PATH = "checkpoints/politifact_iter9.pt"  # your trained checkpoint
# DATASET = "politifact"
# DATA_DIR = "data/adversarial_test"  # folder containing A, B, C, D variants
# ORIGINAL_PATH = f"data/news_articles/{DATASET}_test.pkl"
# VARIANTS = ["Original", "A", "B", "C", "D"]

# # ---------------- DATASET ----------------
# class NewsDataset(Dataset):
#     def __init__(self, texts, labels, tokenizer, max_len=128):
#         self.texts = texts
#         self.labels = labels
#         self.tokenizer = tokenizer
#         self.max_len = max_len

#     def __getitem__(self, idx):
#         text = self.texts[idx]
#         label = self.labels[idx]
#         encoding = self.tokenizer.encode_plus(
#             text,
#             add_special_tokens=True,
#             max_length=self.max_len,
#             padding="max_length",
#             truncation=True,
#             return_tensors="pt"
#         )
#         return {
#             "input_ids": encoding["input_ids"].flatten(),
#             "attention_mask": encoding["attention_mask"].flatten(),
#             "labels": torch.tensor(label, dtype=torch.long),
#         }

#     def __len__(self):
#         return len(self.texts)

# # ---------------- MODEL ----------------
# class RobertaClassifier(nn.Module):
#     def __init__(self, n_classes=4):
#         super(RobertaClassifier, self).__init__()
#         self.roberta = RobertaModel.from_pretrained("roberta-base")
#         self.dropout = nn.Dropout(0.5)
#         self.fc_out = nn.Linear(self.roberta.config.hidden_size, n_classes)
#         self.binary_transform = nn.Linear(self.roberta.config.hidden_size, 2)

#     def forward(self, input_ids, attention_mask):
#         outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
#         pooled_output = outputs[1]
#         pooled_output = self.dropout(pooled_output)
#         return self.fc_out(pooled_output), self.binary_transform(pooled_output)

# # ---------------- HELPERS ----------------
# def load_pickle(path):
#     with open(path, "rb") as f:
#         return pickle.load(f)

# def create_loader(texts, labels, tokenizer, batch_size=1, max_len=128):
#     dataset = NewsDataset(texts, labels, tokenizer, max_len)
#     return DataLoader(dataset, batch_size=batch_size, shuffle=False)

# def evaluate_model(model, loader):
#     model.eval()
#     all_labels, all_preds = [], []

#     with torch.no_grad():
#         for batch in loader:
#             input_ids = batch["input_ids"].to(DEVICE)
#             attention_mask = batch["attention_mask"].to(DEVICE)
#             labels = batch["labels"].to(DEVICE)

#             _, out = model(input_ids, attention_mask)
#             _, preds = torch.max(out, dim=1)

#             all_labels.extend(labels.cpu().numpy())
#             all_preds.extend(preds.cpu().numpy())

#     # Compute metrics
#     acc = accuracy_score(all_labels, all_preds)
#     prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro")
#     f1_by_class = precision_recall_fscore_support(all_labels, all_preds, average=None, labels=[0, 1])
#     f1_r, f1_f = f1_by_class[2][0], f1_by_class[2][1]

#     return acc, prec, rec, f1, f1_r, f1_f

# # ---------------- MAIN ----------------
# if __name__ == "__main__":
#     tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

#     print(f"[INFO] Loading trained model from {MODEL_PATH} ...")
#     model = RobertaClassifier(n_classes=4).to(DEVICE)
#     model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
#     model.eval()

#     for variant in VARIANTS:
#         if variant == "Original":
#             file_path = ORIGINAL_PATH
#         else:
#             file_path = os.path.join(DATA_DIR, f"{DATASET}_test_adv_{variant}.pkl")

#         if not os.path.exists(file_path):
#             print(f"[WARN] Missing file for variant {variant}: {file_path}")
#             continue

#         data = load_pickle(file_path)
#         news = data.get("news", [])
#         labels = data.get("labels", [])

#         loader = create_loader(news, labels, tokenizer, batch_size=1, max_len=128)

#         acc, prec, rec, f1, f1r, f1f = evaluate_model(model, loader)

#         print(f"LLM-Augmented (RoBERTa backbone, real news: fine-grained labels set to all-0)")
#         print(f"------------- Test Variant: {variant} -------------")
#         print(f"Accuracy: {acc:.4f}")
#         print(f"Precision: {prec:.4f}")
#         print(f"Recall: {rec:.4f}")
#         print(f"F1 (Macro): {f1:.4f}")
#         print(f"F1-Real: {f1r:.4f}")
#         print(f"F1-Fake: {f1f:.4f}")
#         print("------------------------------------------------------\n")

# evaluate_roberta_variants.py

# evaluate_roberta_variants_with_log.py

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np
import pickle
from datetime import datetime

# ---------------- CONFIG ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "checkpoints/politifact_iter9.pt"  # your trained checkpoint
DATASET = "politifact"
DATA_DIR = "data/adversarial_test"  # folder containing A, B, C, D variants
ORIGINAL_PATH = f"data/news_articles/{DATASET}_test.pkl"
VARIANTS = ["Original", "A", "B", "C", "D"]

# Log directory & file (matches your requested location/name)
LOG_DIR = "logs/logs_archive_all4_adv"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "log_politifact_sheepdog.iter9")

# ---------------- DATASET ----------------
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }

    def __len__(self):
        return len(self.texts)

# ---------------- MODEL ----------------
class RobertaClassifier(nn.Module):
    def __init__(self, n_classes=4):
        super(RobertaClassifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.dropout = nn.Dropout(0.5)
        self.fc_out = nn.Linear(self.roberta.config.hidden_size, n_classes)
        self.binary_transform = nn.Linear(self.roberta.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        return self.fc_out(pooled_output), self.binary_transform(pooled_output)

# ---------------- HELPERS ----------------
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def create_loader(texts, labels, tokenizer, batch_size=1, max_len=128):
    dataset = NewsDataset(texts, labels, tokenizer, max_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

def evaluate_model(model, loader, repeats=10):
    """Run evaluation multiple times to collect distributions like your sample output."""
    all_acc, all_prec, all_rec, all_f1, all_f1r, all_f1f = [], [], [], [], [], []

    for _ in range(repeats):
        model.eval()
        all_labels, all_preds = [], []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)

                _, out = model(input_ids, attention_mask)
                _, preds = torch.max(out, dim=1)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        # Compute metrics
        acc = accuracy_score(all_labels, all_preds)
        prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro")
        f1_by_class = precision_recall_fscore_support(all_labels, all_preds, average=None, labels=[0, 1])
        # safe indexing: if only one class present, handle gracefully
        try:
            f1_r, f1_f = f1_by_class[2][0], f1_by_class[2][1]
        except Exception:
            # fallback if weird shape: compute per-class f1 with labels param differently
            f1_per_label = precision_recall_fscore_support(all_labels, all_preds, average=None, labels=[0,1])
            f1_r = f1_per_label[2][0] if len(f1_per_label[2]) > 0 else 0.0
            f1_f = f1_per_label[2][1] if len(f1_per_label[2]) > 1 else 0.0

        all_acc.append(acc)
        all_prec.append(prec)
        all_rec.append(rec)
        all_f1.append(f1)
        all_f1r.append(f1_r)
        all_f1f.append(f1_f)

    return all_acc, all_prec, all_rec, all_f1, all_f1r, all_f1f

# ---------------- MAIN ----------------
if __name__ == "__main__":
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    # load model (same as your working code)
    print(f"[INFO] Loading trained model from {MODEL_PATH} ...")
    model = RobertaClassifier(n_classes=4).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # Open log file in write mode (overwrite). Use "a" to append instead.
    with open(LOG_FILE, "w", encoding="utf-8") as logf:
        # optional header
        header = (
            f"=== Evaluation Log (politifact_iter9) ===\n"
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )
        print(header, end="")        # print header to terminal
        logf.write(header)           # write header to log

        for variant in VARIANTS:
            if variant == "Original":
                file_path = ORIGINAL_PATH
            else:
                file_path = os.path.join(DATA_DIR, f"{DATASET}_test_adv_{variant}.pkl")

            if not os.path.exists(file_path):
                warn_msg = f"[WARN] Missing file for variant {variant}: {file_path}\n"
                print(warn_msg, end="")
                logf.write(warn_msg)
                continue

            data = load_pickle(file_path)
            news = data.get("news", [])
            labels = data.get("labels", [])

            loader = create_loader(news, labels, tokenizer, batch_size=1, max_len=128)

            all_acc, all_prec, all_rec, all_f1, all_f1r, all_f1f = evaluate_model(model, loader)

            # Build output exactly as your desired format
            output_lines = []
            output_lines.append("LLM-Augmented (RoBERTa backbone, real news: fine-grained labels set to all-0)\n")
            output_lines.append(f"-------------Test Variant: {variant} -------------\n")
            output_lines.append(f"All Acc.s:{all_acc}\n")
            output_lines.append(f"All Prec.s:{all_prec}\n")
            output_lines.append(f"All Rec.s:{all_rec}\n")
            output_lines.append(f"All F1.s:{all_f1}\n")
            output_lines.append(f"All F1-R:{all_f1r}\n")
            output_lines.append(f"All F1-F:{all_f1f}\n")
            output_lines.append(f"Average acc.: {np.mean(all_acc)} \n")
            output_lines.append(f"Average Prec / Rec / F1 (macro): {np.mean(all_prec)}, {np.mean(all_rec)}, {np.mean(all_f1)} \n")
            output_lines.append(f"Average F1 by class (Real / Fake): {np.mean(all_f1r)}, {np.mean(all_f1f)} \n\n")

            # Print to terminal and write same text to log file
            for line in output_lines:
                print(line, end="")   # print without extra newline
                logf.write(line)

    print(f"[INFO] Saved evaluation output to: {LOG_FILE}")
