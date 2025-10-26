# prediction.py
import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel

# ---------------- CONFIG ----------------
MODEL_PATH = "checkpoints/politifact_iter9.pt"  # trained model checkpoint
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- MODEL DEFINITION ----------------
class RobertaClassifier(nn.Module):
    def __init__(self, n_classes=4):
        super(RobertaClassifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.dropout = nn.Dropout(0.5)
        self.fc_out = nn.Linear(self.roberta.config.hidden_size, n_classes)
        self.binary_transform = nn.Linear(self.roberta.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # [CLS] token representation
        pooled_output = self.dropout(pooled_output)
        return self.fc_out(pooled_output), self.binary_transform(pooled_output)

# ---------------- PREDICTION FUNCTION ----------------
def predict_news(model, tokenizer, text, max_len=128):
    """Predicts whether a given news text is Real or Fake."""
    model.eval()

    # Tokenize input
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
        _, binary_output = model(input_ids, attention_mask)
        probs = torch.softmax(binary_output, dim=1)
        pred_label = torch.argmax(probs, dim=1).item()

    # 0 -> Real, 1 -> Fake
    return "Real News ✅" if pred_label == 0 else "Fake News ❌"

# ---------------- MAIN ----------------
if __name__ == "__main__":
    print("[INFO] Loading tokenizer and model...")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaClassifier(n_classes=4).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    print("\n🔍 Enter/paste the news article below (press Enter twice to predict):\n")

    # Read multi-line input
    lines = []
    while True:
        line = input()
        if line.strip() == "":
            break
        lines.append(line)
    news_text = " ".join(lines).strip()

    if not news_text:
        print("❗No input provided. Please run again and paste a news text.")
    else:
        prediction = predict_news(model, tokenizer, news_text)
        print("\n🧠 Prediction Result:", prediction)
