# import torch
# if torch.cuda.is_available():
#     print("CUDA is available. Using GPU.")
# else:
#     print("CUDA is not available. Using CPU.")
import pickle, os

path = os.path.join("data", "adversarial_test", "politifact_test_adv_A.pkl")
with open(path, "rb") as f:
    data = pickle.load(f)

print(type(data))
if isinstance(data, list):
    print("Length:", len(data))
    print("First item:", data[0])
elif isinstance(data, dict):
    print("Keys:", list(data.keys())[:10])
    print("First item type:", type(next(iter(data.values()))))
