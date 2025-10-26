import pickle
from collections import Counter

# Path to your test data file
# data_path = "data/news_articles/politifact_test.pkl"
data_path = "data/adversarial_test/politifact_test_adv_A.pkl"
# Load the pickle file
with open(data_path, "rb") as f:
    data = pickle.load(f)

# Peek at structure
print("Type of loaded data:", type(data))
if isinstance(data, dict):
    print("Keys:", data.keys())

# Check if it has 'labels' or 'label' or similar
labels = None
for key in ['labels', 'label', 'y', 'targets']:
    if key in data:
        labels = data[key]
        print(f"\nFound label field: '{key}'")
        break

# If data is a list of dicts
if labels is None and isinstance(data, list):
    if isinstance(data[0], dict) and 'label' in data[0]:
        labels = [d['label'] for d in data]
        print("\nExtracted labels from list of dicts using 'label' key.")

# Now analyze the labels
if labels is not None:
    print("\nTotal samples:", len(labels))
    label_counts = Counter(labels)
    print("Label distribution:", label_counts)
    print("Unique labels:", sorted(label_counts.keys()))
else:
    print("\n❌ Could not find any label field in politifact_test.pkl. Please print one sample to inspect structure.")
