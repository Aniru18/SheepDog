

# # prediction.py
# import torch
# import torch.nn as nn
# from transformers import RobertaTokenizer, RobertaModel

# # ---------------- CONFIG ----------------
# MODEL_PATH = "checkpoints/politifact_iter9.pt"  # trained model checkpoint
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ---------------- MODEL DEFINITION ----------------
# class RobertaClassifier(nn.Module):
#     def __init__(self, n_classes=4):
#         super(RobertaClassifier, self).__init__()
#         self.roberta = RobertaModel.from_pretrained("roberta-base")
#         self.dropout = nn.Dropout(0.5)
#         self.fc_out = nn.Linear(self.roberta.config.hidden_size, n_classes)
#         self.binary_transform = nn.Linear(self.roberta.config.hidden_size, 2)

#     def forward(self, input_ids, attention_mask):
#         outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
#         pooled_output = outputs[1]  # [CLS]
#         pooled_output = self.dropout(pooled_output)
#         return self.fc_out(pooled_output), self.binary_transform(pooled_output)

# # ---------------- PREDICTION FUNCTION ----------------
# def predict_news(model, tokenizer, text, max_len=128):
#     model.eval()
    
#     encoding = tokenizer.encode_plus(
#         text,
#         add_special_tokens=True,
#         max_length=max_len,
#         padding="max_length",
#         truncation=True,
#         return_tensors="pt"
#     )

#     input_ids = encoding["input_ids"].to(DEVICE)
#     attention_mask = encoding["attention_mask"].to(DEVICE)

#     with torch.no_grad():
#         _, binary_output = model(input_ids, attention_mask)
#         probs = torch.softmax(binary_output, dim=1)[0]

#     pred_label = torch.argmax(probs).item()
#     confidence = probs[pred_label].item()
#     label_str = "Real News ✅" if pred_label == 0 else "Fake News ❌"

#     return label_str, confidence, probs.cpu().tolist()

# # ---------------- MAIN ----------------
# if __name__ == "__main__":
#     print("[INFO] Loading tokenizer and model...")
#     tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
#     model = RobertaClassifier(n_classes=4).to(DEVICE)
#     model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
#     model.eval()

#     print("\n🔍 Type or paste your news article below.")
#     print("💡 Type 'exit' and press Enter to quit.\n")

#     while True:
#         print("\nEnter news article (press Enter twice to classify):\n")

#         # Read multi-line input
#         lines = []
#         while True:
#             line = input()
#             if line.strip().lower() == "exit":
#                 print("\n👋 Exiting...")
#                 exit(0)
#             if line.strip() == "":
#                 break
#             lines.append(line)

#         news_text = " ".join(lines).strip()

#         if not news_text:
#             print("❗No text entered. Try again or type 'exit'.")
#             continue

#         # Predict
#         label_str, confidence, probs = predict_news(model, tokenizer, news_text)

#         print("\n🧠 Prediction Result:", label_str)
#         print(f"📊 Confidence: {confidence:.4f} ({confidence * 100:.2f}%)")
#         print(f"   Class probabilities (Real, Fake): {probs}")
# prediction.py
import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel

# ---------------- CONFIG ----------------
MODEL_PATH = "checkpoints/politifact_iter9.pt"  # trained model checkpoint
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fine-grained class names (must match your 'classes' list in PKL)
FG_CLASSES = [
    "Lack of credible sources",           # Class 0
    "False or misleading information",    # Class 1
    "Biased opinion",                     # Class 2
    "Inconsistencies with reputable sources"  # Class 3
]

# ---------------- MODEL DEFINITION ----------------
class RobertaClassifier(nn.Module):
    def __init__(self, n_classes=4):
        super(RobertaClassifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.dropout = nn.Dropout(0.5)
        self.fc_out = nn.Linear(self.roberta.config.hidden_size, n_classes)  # fine-grained head
        self.binary_transform = nn.Linear(self.roberta.config.hidden_size, 2)  # real/fake head

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # [CLS]
        pooled_output = self.dropout(pooled_output)
        # Return BOTH heads: fine-grained logits, binary logits
        return self.fc_out(pooled_output), self.binary_transform(pooled_output)

# ---------------- PREDICTION FUNCTION ----------------
def predict_news(model, tokenizer, text, max_len=128):
    model.eval()
    
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)

    with torch.no_grad():
        fg_logits, binary_output = model(input_ids, attention_mask)

        # --- Binary prediction (Real / Fake) ---
        probs_binary = torch.softmax(binary_output, dim=1)[0]  # [2]
        pred_label = torch.argmax(probs_binary).item()
        confidence = probs_binary[pred_label].item()
        label_str = "Real News ✅" if pred_label == 0 else "Fake News ❌"

        # --- Fine-grained attributions (4 classes) ---
        # Use sigmoid because this is a multi-label head
        fg_probs = torch.sigmoid(fg_logits)[0]  # [4]

    # Convert tensors to plain Python lists
    probs_binary_list = probs_binary.cpu().tolist()     # [P(real), P(fake)]
    fg_probs_list = fg_probs.cpu().tolist()             # [p(class0), ..., p(class3)]

    return label_str, confidence, probs_binary_list, fg_probs_list

# ---------------- MAIN ----------------
if __name__ == "__main__":
    print("[INFO] Loading tokenizer and model...")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaClassifier(n_classes=4).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    print("\n🔍 Type or paste your news article below.")
    print("💡 Type 'exit' and press Enter to quit.\n")

    while True:
        print("\nEnter news article (press Enter twice to classify):\n")

        # Read multi-line input
        lines = []
        while True:
            line = input()
            if line.strip().lower() == "exit":
                print("\n👋 Exiting...")
                exit(0)
            if line.strip() == "":
                break
            lines.append(line)

        news_text = " ".join(lines).strip()

        if not news_text:
            print("❗No text entered. Try again or type 'exit'.")
            continue

        # Predict
        label_str, confidence, probs_binary, fg_probs = predict_news(model, tokenizer, news_text)

        # ----- Print binary prediction -----
        print("\n🧠 Prediction Result:", label_str)
        print(f"📊 Confidence: {confidence:.4f} ({confidence * 100:.2f}%)")
        print(f"   Class probabilities (Real, Fake): {probs_binary}")

        # ----- Print fine-grained attributions -----
        print("\n🔎 Fine-Grained Veracity Attributions (per class):")
        for idx, (cls_name, p) in enumerate(zip(FG_CLASSES, fg_probs)):
            print(f"   Class {idx} - {cls_name}: {p:.4f} ({p * 100:.2f}%)")

        # Optional: highlight top contributing issues (sorted)
        print("\n✨ Most likely issues (sorted):")
        sorted_indices = sorted(range(len(fg_probs)), key=lambda i: fg_probs[i], reverse=True)
        for i in sorted_indices:
            print(f"   {FG_CLASSES[i]} → {fg_probs[i]:.4f} ({fg_probs[i] * 100:.2f}%)")
