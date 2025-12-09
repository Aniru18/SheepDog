# import pickle
# from collections import Counter

# # Path to your test data file
# # data_path = "data/news_articles/politifact_test.pkl"
# data_path = "data/adversarial_test/politifact_test_adv_A.pkl"
# # Load the pickle file
# with open(data_path, "rb") as f:
#     data = pickle.load(f)

# # Peek at structure
# print("Type of loaded data:", type(data))
# if isinstance(data, dict):
#     print("Keys:", data.keys())

# # Check if it has 'labels' or 'label' or similar
# labels = None
# for key in ['labels', 'label', 'y', 'targets']:
#     if key in data:
#         labels = data[key]
#         print(f"\nFound label field: '{key}'")
#         break

# # If data is a list of dicts
# if labels is None and isinstance(data, list):
#     if isinstance(data[0], dict) and 'label' in data[0]:
#         labels = [d['label'] for d in data]
#         print("\nExtracted labels from list of dicts using 'label' key.")

# # Now analyze the labels
# if labels is not None:
#     print("\nTotal samples:", len(labels))
#     label_counts = Counter(labels)
#     print("Label distribution:", label_counts)
#     print("Unique labels:", sorted(label_counts.keys()))
# else:
#     print("\n❌ Could not find any label field in politifact_test.pkl. Please print one sample to inspect structure.")
# ------------------------- FULL TRAINING SCRIPT WITH WIKIPEDIA KB -------------------------

import torch, os, sys, warnings, wikipedia, spacy, numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaTokenizer, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import precision_recall_fscore_support as score, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------- KB & NLP UTILITIES -------------------------
nlp = spacy.load("en_core_web_sm")
wiki_cache = {}

def extract_entities(text):
    doc = nlp(text)
    return [ent.text for ent in doc.ents]

def query_wikipedia(entity, sentences=5):
    if entity in wiki_cache:
        return wiki_cache[entity]
    try:
        summary = wikipedia.summary(entity, sentences=sentences, auto_suggest=True)
    except wikipedia.DisambiguationError as e:
        summary = wikipedia.summary(e.options[0], sentences=sentences)
    except wikipedia.PageError:
        summary = ""
    except Exception:
        summary = ""
    wiki_cache[entity] = summary
    return summary

def verifier(article_text, evidence_texts):
    if not evidence_texts:
        return 0.0
    all_texts = [article_text] + evidence_texts
    vectorizer = TfidfVectorizer().fit_transform(all_texts)
    sim_matrix = cosine_similarity(vectorizer[0:1], vectorizer[1:])
    return sim_matrix.mean()

# ------------------------- DATASET -------------------------
class NewsDatasetAug(Dataset):
    def __init__(self, texts, aug_texts1, aug_texts2, labels, fg_label, aug_fg1, aug_fg2, tokenizer, max_len):
        self.texts = texts
        self.aug_texts1 = aug_texts1
        self.aug_texts2 = aug_texts2
        self.labels = labels
        self.fg_label = fg_label
        self.aug_fg1 = aug_fg1
        self.aug_fg2 = aug_fg2
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, item):
        text = self.texts[item]
        aug_text1 = self.aug_texts1[item]
        aug_text2 = self.aug_texts2[item]
        label = self.labels[item]
        fg_label = self.fg_label[item]
        aug_fg1 = self.aug_fg1[item]
        aug_fg2 = self.aug_fg2[item]

        def encode(txt):
            return self.tokenizer.encode_plus(
                txt, add_special_tokens=True, max_length=self.max_len,
                padding='max_length', truncation=True, return_tensors='pt'
            )

        encoding = encode(text)
        aug1_encoding = encode(aug_text1)
        aug2_encoding = encode(aug_text2)

        return {
            'news_text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'input_ids_aug1': aug1_encoding['input_ids'].flatten(),
            'input_ids_aug2': aug2_encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'attention_mask_aug1': aug1_encoding['attention_mask'].flatten(),
            'attention_mask_aug2': aug2_encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long),
            'fg_label': torch.FloatTensor(fg_label),
            'fg_label_aug1': torch.FloatTensor(aug_fg1),
            'fg_label_aug2': torch.FloatTensor(aug_fg2)
        }

    def __len__(self):
        return len(self.texts)

def create_train_loader(contents, contents_aug1, contents_aug2, labels, fg_label, aug_fg1, aug_fg2, tokenizer, max_len, batch_size):
    ds = NewsDatasetAug(contents, contents_aug1, contents_aug2, labels, fg_label, aug_fg1, aug_fg2, tokenizer, max_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, item):
        text = self.texts[item]
        label = self.labels[item]
        encoding = self.tokenizer.encode_plus(
            text, add_special_tokens=True, max_length=self.max_len,
            padding='max_length', truncation=True, return_tensors='pt'
        )
        return {
            'news_text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

    def __len__(self):
        return len(self.texts)

def create_eval_loader(contents, labels, tokenizer, max_len, batch_size):
    ds = NewsDataset(contents, labels, tokenizer, max_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

# ------------------------- MODEL -------------------------
class RobertaClassifier(nn.Module):
    def __init__(self, n_classes):
        super(RobertaClassifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.dropout = nn.Dropout(0.5)
        self.fc_out = nn.Linear(self.roberta.config.hidden_size, n_classes)
        self.binary_transform = nn.Linear(self.roberta.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        return self.fc_out(pooled_output), self.binary_transform(pooled_output)

# ------------------------- TRAINING -------------------------
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def train_model(tokenizer, max_len, n_epochs, batch_size, dataset_name, iterations, load_articles, load_reframing):
    torch.cuda.empty_cache()
    x_train, x_test, x_test_res, y_train, y_test = load_articles(dataset_name)
    model = RobertaClassifier(n_classes=4).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = 10000
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    scaler = torch.cuda.amp.GradScaler()

    for iter_num in range(iterations):
        model.train()
        x_train_res1, x_train_res2, y_train_fg, y_train_fg_m, y_train_fg_t = load_reframing(dataset_name)
        train_loader = create_train_loader(x_train, x_train_res1, x_train_res2, y_train, y_train_fg, y_train_fg_m, y_train_fg_t, tokenizer, max_len, batch_size)
        for epoch in range(n_epochs):
            avg_loss = []
            for batch in tqdm(train_loader):
                # ----------- move inputs to device -----------
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                input_ids_aug1 = batch["input_ids_aug1"].to(device)
                attention_mask_aug1 = batch["attention_mask_aug1"].to(device)
                input_ids_aug2 = batch["input_ids_aug2"].to(device)
                attention_mask_aug2 = batch["attention_mask_aug2"].to(device)

                targets = batch["labels"].to(device)
                fg_labels = batch["fg_label"].to(device)
                fg_labels_aug1 = batch["fg_label_aug1"].to(device)
                fg_labels_aug2 = batch["fg_label_aug2"].to(device)

                # ----------- KB Consistency -----------
                batch_texts = batch['news_text']
                batch_scores = []
                for text in batch_texts:
                    entities = extract_entities(text)
                    evidence_texts = [query_wikipedia(ent) for ent in entities]
                    score_kb = verifier(text, evidence_texts)
                    batch_scores.append(score_kb)
                consistency_scores = torch.tensor(batch_scores, dtype=torch.float32).unsqueeze(1).to(device)

                # ----------- Forward Pass -----------
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    out_labels, out_labels_bi = model(input_ids, attention_mask)
                    out_labels_aug1, out_labels_bi_aug1 = model(input_ids_aug1, attention_mask_aug1)
                    out_labels_aug2, out_labels_bi_aug2 = model(input_ids_aug2, attention_mask_aug2)

                    fg_criterion = nn.BCEWithLogitsLoss()
                    finegrain_loss = (fg_criterion(out_labels, fg_labels) +
                                      fg_criterion(out_labels_aug1, fg_labels_aug1) +
                                      fg_criterion(out_labels_aug2, fg_labels_aug2)) / 3

                    sup_criterion = nn.CrossEntropyLoss()
                    sup_loss = sup_criterion(out_labels_bi, targets)

                    out_probs = F.softmax(out_labels_bi, dim=-1)
                    cons_criterion = nn.KLDivLoss(reduction='batchmean')
                    cons_loss = 0.5 * cons_criterion(F.log_softmax(out_labels_bi_aug1, dim=-1), out_probs) + \
                                0.5 * cons_criterion(F.log_softmax(out_labels_bi_aug2, dim=-1), out_probs)

                    # Integrate KB by weighting supervised loss
                    loss = sup_loss * consistency_scores.mean() + cons_loss + finegrain_loss

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                avg_loss.append(loss.item())

            print(f"Iter {iter_num} | Epoch {epoch+1} | Train Loss: {np.mean(avg_loss):.4f}")

        # ----------- Evaluation -----------
        test_loader = create_eval_loader(x_test, y_test, tokenizer, max_len, batch_size)
        all_preds, all_labels = [], []
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                _, out = model(input_ids, attention_mask)
                _, preds = torch.max(out, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        acc = accuracy_score(all_labels, all_preds)
        prec, rec, f1, _ = score(all_labels, all_preds, average='macro')
        print(f"---- Evaluation Iter {iter_num} Epoch {epoch+1} ---- Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}\n")

        torch.save(model.state_dict(), f'checkpoints/{dataset_name}_iter{iter_num}_kb.pt')

# ------------------------- MAIN -------------------------
if __name__ == "__main__":
    set_seed(0)
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    # Example placeholders for your dataset loader functions
    iterations = 1
    max_len = 128
    batch_size = 1
    n_epochs = 1
    dataset_name = "politifact"

    # load_articles and load_reframing should be defined externally
    # Example: train_model(tokenizer, max_len, n_epochs, batch_size, dataset_name, iterations, load_articles, load_reframing)
