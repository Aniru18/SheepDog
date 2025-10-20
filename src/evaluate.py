# #..................before git hub style implematation

# import torch
# import numpy as np
# from sklearn.metrics import precision_recall_fscore_support as score
# from sklearn.metrics import accuracy_score
# from transformers import RobertaTokenizer
# from torch.utils.data import DataLoader
# import os, sys
# from tqdm import tqdm
# sys.path.append(os.getcwd())
# from utils.load_data import load_articles
# from src.sheepdog import RobertaClassifier, NewsDataset  # import from your training file
# import pickle
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ----------------- EVALUATION DATA LOADER -----------------
# def create_eval_loader(contents, labels, tokenizer, max_len, batch_size):
#     ds = NewsDataset(texts=contents, labels=np.array(labels), tokenizer=tokenizer, max_len=max_len)
#     return DataLoader(ds, batch_size=batch_size, num_workers=0)


# # ----------------- EVALUATION FUNCTION -----------------
# def evaluate_model(model, data_loader):
#     model.eval()
#     all_preds, all_labels = [], []
#     with torch.no_grad():
#         for batch in data_loader:
#             input_ids = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#             labels = batch['labels'].to(device)
#             _, out = model(input_ids, attention_mask)
#             _, preds = torch.max(out, dim=1)
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())
#     return np.array(all_labels), np.array(all_preds)


# # ----------------- MAIN -----------------
# if __name__ == "__main__":
#     dataset_name = "politifact"
#     model_path = f"checkpoints/{dataset_name}_iter1.pt"  # Load your saved model
#     batch_size = 4
#     max_len = 512

#     tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
#     x_train, x_test, x_test_res, y_train, y_test = load_articles(dataset_name)

#     # Load model
#     model = RobertaClassifier(n_classes=4).to(device)
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.eval()

#     # Variants: Original + Adversarial A
#     # variants = {
#     #     "Original": (x_test, y_test),
#     #     "A": (x_test_res, y_test)
#     # }
#     # Load base dataset (original test set)
#     x_train, x_test, x_test_res, y_train, y_test = load_articles(dataset_name)

#     # Load all adversarial variants from data/adversarial_test/
# # ----------------- LOAD ADVERSARIAL VARIANT -----------------
#     def load_adv_variant(name, variant, original_labels):
#         """
#         Load adversarial test variant, keeping the original labels from y_test.
#         """
#         path = os.path.join("data", "adversarial_test", f"{name}_test_adv_{variant}.pkl")
#         if not os.path.exists(path):
#             print(f"⚠️ Missing file: {path}")
#             return [], []

#         with open(path, "rb") as f:
#             data = pickle.load(f)

#         # Expected structure: {'news': [...]} or {'news': [...], 'labels': [...]}
#         if isinstance(data, dict) and "news" in data:
#             contents = data["news"]
#             if "labels" in data:
#                 labels = data["labels"]  # use labels from pickle if present
#             else:
#                 labels = original_labels[:len(contents)]  # use corresponding original labels
#         else:
#             raise ValueError(f"Unexpected format: {type(data)} or missing 'news' key in {path}")

#         return contents, labels


#     # Variants dictionary for evaluation
#     variants = {
#     "Original": (x_test, y_test),
#     "A": load_adv_variant(dataset_name, "A", y_test),
#     "B": load_adv_variant(dataset_name, "B", y_test),
#     "C": load_adv_variant(dataset_name, "C", y_test),
#     "D": load_adv_variant(dataset_name, "D", y_test),
#     }


#     for var_name, (x_var, y_var) in variants.items():
#         all_acc, all_prec, all_rec, all_f1 = [], [], [], []
#         split_size = len(x_var) // 10  # 10 splits to match GitHub log

#         for i in range(10):
#             start = i * split_size
#             end = len(x_var) if i == 9 else (i+1) * split_size
#             x_split = x_var[start:end]
#             y_split = y_var[start:end]

#             test_loader = create_eval_loader(x_split, y_split, tokenizer, max_len, batch_size)
#             y_true, y_pred = evaluate_model(model, test_loader)

#             acc = (y_true == y_pred).mean()
#             prec, rec, f1, _ = score(y_true, y_pred, average='macro')

#             all_acc.append(acc)
#             all_prec.append(prec)
#             all_rec.append(rec)
#             all_f1.append(f1)

#         # Print results in GitHub-style format
#         print(f"LLM-Augmented (RoBERTa backbone, real news: fine-grained labels set to all-0)")
#         print(f"-------------Test Variant: {var_name} -------------")
#         print(f"All Acc.s:{all_acc}")
#         print(f"All Prec.s:{all_prec}")
#         print(f"All Rec.s:{all_rec}")
#         print(f"All F1.s:{all_f1}")
#         print(f"Average acc.: {np.mean(all_acc)} ")
#         print(f"Average Prec / Rec / F1 (macro): {np.mean(all_prec)}, {np.mean(all_rec)}, {np.mean(all_f1)} \n")
# #..................after git hub style implematation
import os
import sys
sys.path.append(os.getcwd())
import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support as score
from transformers import RobertaTokenizer
from torch.utils.data import DataLoader
from src.sheepdog import RobertaClassifier, NewsDataset
from utils.load_data import load_articles

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------- EVAL DATA LOADER -----------------
def create_eval_loader(contents, labels, tokenizer, max_len, batch_size):
    ds = NewsDataset(texts=contents, labels=np.array(labels), tokenizer=tokenizer, max_len=max_len)
    return DataLoader(ds, batch_size=batch_size, num_workers=0)

# ----------------- EVAL FUNCTION -----------------
def evaluate_model(model, data_loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            _, out = model(input_ids, attention_mask)
            _, preds = torch.max(out, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_labels), np.array(all_preds)

# ----------------- MAIN -----------------
if __name__ == "__main__":
    dataset_name = "politifact"
    model_path = f"checkpoints/{dataset_name}_iter1.pt"
    batch_size = 4
    max_len = 512

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    x_train, x_test, x_test_res, y_train, y_test = load_articles(dataset_name)

    # Load model
    model = RobertaClassifier(n_classes=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Only Original + Restyle (like GitHub)
    variants = {
        "Original": (x_test, y_test),
        "Restyle": (x_test_res, y_test)
    }

    for var_name, (x_var, y_var) in variants.items():
        all_acc, all_prec, all_rec, all_f1 = [], [], [], []
        split_size = len(x_var) // 10  # GitHub splits

        for i in range(10):
            start = i * split_size
            end = len(x_var) if i == 9 else (i + 1) * split_size
            x_split = x_var[start:end]
            y_split = y_var[start:end]

            test_loader = create_eval_loader(x_split, y_split, tokenizer, max_len, batch_size)
            y_true, y_pred = evaluate_model(model, test_loader)

            acc = (y_true == y_pred).mean()
            prec, rec, f1, _ = score(y_true, y_pred, average='macro')

            all_acc.append(acc)
            all_prec.append(prec)
            all_rec.append(rec)
            all_f1.append(f1)

        # Print results like GitHub logs
        print(f"-------------Test Variant: {var_name} -------------")
        print(f"All Acc.s:{all_acc}")
        print(f"All Prec.s:{all_prec}")
        print(f"All Rec.s:{all_rec}")
        print(f"All F1.s:{all_f1}")
        print(f"Average acc.: {np.mean(all_acc)} ")
        print(f"Average Prec / Rec / F1 (macro): {np.mean(all_prec)}, {np.mean(all_rec)}, {np.mean(all_f1)} \n")
