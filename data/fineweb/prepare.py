import os
import numpy as np
import tiktoken
from datasets import load_dataset

# ----------------------------
# Config
# ----------------------------
OUT_DIR = os.path.dirname(__file__)
DATASET_NAME = "HuggingFaceFW/fineweb"
SPLIT = "train"

# Tu peux limiter pour faire un petit dataset local
TARGET_TOKENS = 100_000_000   # ex: 100M tokens
VAL_RATIO = 0.01              # 1% validation

# Optionnel : choisir un sous-ensemble / sample si dispo
# Exemple:
# ds = load_dataset(DATASET_NAME, name="sample-10BT", split=SPLIT, streaming=True)
# Sinon:
ds = load_dataset(DATASET_NAME, split=SPLIT, streaming=True)

enc = tiktoken.get_encoding("gpt2")
eot = enc.eot_token  # token de séparation de documents

train_path = os.path.join(OUT_DIR, "train.bin")
val_path = os.path.join(OUT_DIR, "val.bin")

train_f = open(train_path, "wb")
val_f = open(val_path, "wb")

n_train = 0
n_val = 0
n_total = 0
doc_count = 0

def write_tokens(f, tokens):
    arr = np.array(tokens, dtype=np.uint16)
    arr.tofile(f)

try:
    for example in ds:
        text = example.get("text", None)
        if not text:
            continue

        ids = enc.encode_ordinary(text)
        ids.append(eot)  # séparation entre docs

        if len(ids) == 0:
            continue

        # split train/val simple au niveau document
        if np.random.random() < VAL_RATIO:
            write_tokens(val_f, ids)
            n_val += len(ids)
        else:
            write_tokens(train_f, ids)
            n_train += len(ids)

        n_total += len(ids)
        doc_count += 1

        if doc_count % 1000 == 0:
            print(
                f"docs={doc_count:,} | total={n_total:,} | "
                f"train={n_train:,} | val={n_val:,}"
            )

        if n_total >= TARGET_TOKENS:
            break

finally:
    train_f.close()
    val_f.close()

print(f"Done.")
print(f"Documents processed: {doc_count:,}")
print(f"train.bin tokens: {n_train:,}")
print(f"val.bin tokens:   {n_val:,}")