import os
import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.append(os.getcwd())
import torch
import torch.nn.functional as F
from transformers import RobertaTokenizer
from src.sheepdog import RobertaClassifier, NewsDataset
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------- PARAMETERS -----------------
MODEL_PATH = "checkpoints/politifact_iter1.pt"  # change if needed
MAX_LEN = 512
BATCH_SIZE = 1  # for single prediction

# ----------------- LOAD MODEL -----------------
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaClassifier(n_classes=4).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()


# ----------------- PREDICTION FUNCTION -----------------
def predict_news(news_text):
    """
    Input: news_text (str)
    Output: dict with label, confidence, reason
    """
    # Tokenize
    encoding = tokenizer.encode_plus(
        news_text,
        add_special_tokens=True,
        max_length=MAX_LEN,
        pad_to_max_length=True,
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # Forward pass
    with torch.no_grad():
        logits, fg_logits = model(input_ids=input_ids, attention_mask=attention_mask)

        # Binary label prediction
        probs = F.softmax(logits, dim=-1)
        pred_class = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, pred_class].item()

        # Fine-grained reason
        fg_probs = torch.sigmoid(fg_logits)[0]  # fine-grained
        # Find the top contributing fine-grained reason
        top_idx = torch.argmax(fg_probs).item()
        reason = f"Top indicator index: {top_idx}, score: {fg_probs[top_idx]:.3f}"

        # Map prediction to real/fake
        class_map = {0: "Real", 1: "Fake", 2: "Mostly Real", 3: "Mostly Fake"}  # adjust according to your dataset
        label = class_map.get(pred_class, "Unknown")

    return {
        "label": label,
        "confidence": confidence,
        "reason": reason
    }


# ----------------- USER INTERFACE -----------------
if __name__ == "__main__":
    print("Paste your news article below (end with an empty line):")
    lines = []
    while True:
        line = input()
        if line.strip() == "":
            break
        lines.append(line)
    news_text = " ".join(lines)

    result = predict_news(news_text)
    print("\n----- Prediction Result -----")
    print(f"Label: {result['label']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Reason: {result['reason']}")
