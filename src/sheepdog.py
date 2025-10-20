#with evaluation code
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaTokenizer, AdamW, get_linear_schedule_with_warmup
import argparse
import numpy as np
import sys, os
from tqdm import tqdm
import warnings
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score

sys.path.append(os.getcwd())
from utils.load_data import load_articles, load_reframing

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------- ARGUMENTS -----------------
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', default='politifact', type=str)
parser.add_argument('--model_name', default='Pretrained-LM', type=str)
parser.add_argument('--iters', default=10, type=int)
parser.add_argument('--batch_size', default=1, type=int)  # Reduced for 4GB GPU
parser.add_argument('--n_epochs', default=5, type=int)
parser.add_argument('--max_len', default=128, type=int)   # Reduced from 512
args = parser.parse_args()

torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.cuda.manual_seed_all(0)

# ----------------- DATASETS -----------------
class NewsDatasetAug(Dataset):
    def __init__(self, texts, aug_texts1, aug_texts2, labels, fg_label, aug_fg1, aug_fg2, tokenizer, max_len):
        self.texts = texts
        self.aug_texts1 = aug_texts1
        self.aug_texts2 = aug_texts2
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.labels = labels
        self.fg_label = fg_label
        self.aug_fg1 = aug_fg1
        self.aug_fg2 = aug_fg2

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
            'input_ids': encoding['input_ids'].flatten(),
            'input_ids_aug1': aug1_encoding['input_ids'].flatten(),
            'input_ids_aug2': aug2_encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'attention_mask_aug1': aug1_encoding['attention_mask'].flatten(),
            'attention_mask_aug2': aug2_encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long),
            'fg_label': torch.FloatTensor(fg_label),
            'fg_label_aug1': torch.FloatTensor(aug_fg1),
            'fg_label_aug2': torch.FloatTensor(aug_fg2),
        }

    def __len__(self):
        return len(self.texts)

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

# ----------------- MODEL -----------------
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

# ----------------- DATA LOADERS -----------------
def create_train_loader(contents, contents_aug1, contents_aug2, labels, fg_label, aug_fg1, aug_fg2, tokenizer, max_len, batch_size):
    ds = NewsDatasetAug(contents, contents_aug1, contents_aug2, labels, fg_label, aug_fg1, aug_fg2, tokenizer, max_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)  # num_workers=0 for Windows

def create_eval_loader(contents, labels, tokenizer, max_len, batch_size):
    ds = NewsDataset(contents, labels, tokenizer, max_len)
    return DataLoader(ds, batch_size=batch_size, num_workers=0)

# ----------------- TRAINING -----------------
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def train_model(tokenizer, max_len, n_epochs, batch_size, dataset_name, iter_num):
    torch.cuda.empty_cache()
    scaler = torch.cuda.amp.GradScaler()
    x_train, x_test, x_test_res, y_train, y_test = load_articles(dataset_name)

    model = RobertaClassifier(n_classes=4).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = 10000
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    train_losses, train_accs = [], []

    # Evaluation function (GitHub-style)
    def evaluate_model_on_loader(loader):
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                _, out = model(input_ids, attention_mask)
                _, preds = torch.max(out, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        acc = accuracy_score(all_labels, all_preds)
        prec, rec, f1, _ = score(all_labels, all_preds, average='macro')
        return acc, prec, rec, f1

    for epoch in range(n_epochs):
        model.train()
        x_train_res1, x_train_res2, y_train_fg, y_train_fg_m, y_train_fg_t = load_reframing(dataset_name)
        train_loader = create_train_loader(x_train, x_train_res1, x_train_res2, y_train, 
                                           y_train_fg, y_train_fg_m, y_train_fg_t, tokenizer, max_len, batch_size)

        avg_loss, avg_acc = [], []

        for batch_data in tqdm(train_loader):
            input_ids = batch_data["input_ids"].to(device)
            attention_mask = batch_data["attention_mask"].to(device)
            input_ids_aug1 = batch_data["input_ids_aug1"].to(device)
            attention_mask_aug1 = batch_data["attention_mask_aug1"].to(device)
            input_ids_aug2 = batch_data["input_ids_aug2"].to(device)
            attention_mask_aug2 = batch_data["attention_mask_aug2"].to(device)
            targets = batch_data["labels"].to(device)
            fg_labels = batch_data["fg_label"].to(device)
            fg_labels_aug1 = batch_data["fg_label_aug1"].to(device)
            fg_labels_aug2 = batch_data["fg_label_aug2"].to(device)

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

                loss = sup_loss + cons_loss + finegrain_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            avg_loss.append(loss.item())
            _, pred = out_labels_bi.max(dim=-1)
            correct = pred.eq(targets).sum().item()
            avg_acc.append(correct / len(targets))

        train_losses.append(np.mean(avg_loss))
        train_accs.append(np.mean(avg_acc))
        print(f"Iter {iter_num:03d} | Epoch {epoch+1:03d} | Train Acc: {np.mean(avg_acc):.4f}")

        # ------------------- GitHub-style Evaluation -------------------
        test_loader = create_eval_loader(x_test, y_test, tokenizer, max_len, batch_size)
        test_loader_res = create_eval_loader(x_test_res, y_test, tokenizer, max_len, batch_size)

        acc_orig, prec_orig, rec_orig, f1_orig = evaluate_model_on_loader(test_loader)
        acc_res, prec_res, rec_res, f1_res = evaluate_model_on_loader(test_loader_res)

        print(f"---- Evaluation after Epoch {epoch+1} ----")
        print(f"Original Test Set -> Acc: {acc_orig:.4f}, Prec: {prec_orig:.4f}, Rec: {rec_orig:.4f}, F1: {f1_orig:.4f}")
        print(f"Restyled Test Set -> Acc: {acc_res:.4f}, Prec: {prec_res:.4f}, Rec: {rec_res:.4f}, F1: {f1_res:.4f}\n")

    torch.save(model.state_dict(), f'checkpoints/{dataset_name}_iter{iter_num}.pt')
    return model


# ----------------- MAIN -----------------
if __name__ == "__main__":
    set_seed(0)
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    iterations = args.iters
    max_len = args.max_len
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    dataset_name = args.dataset_name

    for iter_num in range(iterations):
        model = train_model(tokenizer, max_len, n_epochs, batch_size, dataset_name, iter_num)
