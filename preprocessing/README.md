# Preprocessing — Jigsaw Toxic Comment Classification

**Kaggle CSVs live in the repo’s `data/` directory** (project root). Code is in **`text_preprocessing.py`**.

Each model notebook should call **exactly one** function:

| Model | Function | Returns |
|-------|----------|---------|
| CNN + GloVe | `preprocess_for_cnn(...)` | `CNNPreprocessResult` |
| BiLSTM + attention | `preprocess_for_bilstm(...)` | `BiLSTMPreprocessResult` |
| BERT | `preprocess_for_bert(...)` | `TransformerPreprocessResult` |
| DistilBERT | `preprocess_for_distilbert(...)` | `TransformerPreprocessResult` |

Run Jupyter with the **project root** as the working directory.

```python
from preprocessing.text_preprocessing import preprocess_for_cnn, CNNPreprocessResult

result = preprocess_for_cnn(
    validation_fraction=0.1,
    random_state=42,
    max_len=256,
    min_freq=2,
    max_vocab=50_000,
)
# result.X_train, result.X_val, result.y_train, result.y_val, result.vocab, result.train_df, result.val_df
```

---

## `preprocess_for_cnn` / `preprocess_for_bilstm`

**Parameters (keyword-only):**

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `csv_path` | repo `data/train.csv` | Override path to training CSV |
| `validation_fraction` | `0.1` | Validation share |
| `random_state` | `42` | Split seed |
| `max_len` | `256` | Word-level truncate/pad length |
| `min_freq` | `2` | Min word count on **train** to enter vocab |
| `max_vocab` | `50_000` | Cap vocab size (includes `<pad>` / `<unk>`) |
| `max_train_samples` / `max_val_samples` | `None` | After split, keep only first *N* rows (fast smoke tests) |

**CNN result:** `train_df`, `val_df`, `vocab`, `X_train`, `X_val`, `y_train`, `y_val`  
(`X_*` are `int64` `(N, max_len)`; `y_*` are `float32` `(N, 6)`.)

**BiLSTM result:** same fields plus `length_train`, `length_val` (`int64`, one length per row after truncate).

**GloVe:** load and align embedding rows to `vocab` word ids in the model notebook (not done inside preprocessing).

---

## `preprocess_for_bert` / `preprocess_for_distilbert`

Requires **`pip install transformers`** (and usually **`torch`**).

**Parameters:**

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `csv_path` | repo `data/train.csv` | Override CSV path |
| `validation_fraction` | `0.1` | Validation share |
| `random_state` | `42` | Split seed |
| `pretrained_model_name` | `bert-base-uncased` / `distilbert-base-uncased` | HF checkpoint |
| `max_length` | `512` | Subword truncate/pad length |
| `return_tensors` | `"pt"` | HF batch format (`None`, `"np"`, `"pt"`, etc.) |
| `max_train_samples` / `max_val_samples` | `None` | After split, tokenize only first *N* rows (saves time and RAM) |

**Result:** `train_df`, `val_df`, `tokenizer`, `train_encodings`, `val_encodings`, `y_train`, `y_val`  
Encodings include `input_ids` and `attention_mask` (and BERT may add `token_type_ids`).

---

## Shared constants

- **`LABEL_COLUMNS`** — six label names in CSV order  
- **`PAD_TOKEN`** / **`UNK_TOKEN`** — reserved in `vocab` for word models (`<pad>` → 0, `<unk>` → 1)

---

## Conceptual steps (handled inside each function)

1. Load `train.csv`  
2. Stratified train/validation split (by count of positive labels, 0–6)  
3. Normalize `comment_text`  
4. **Word models:** fit vocab on train only → padded id matrices (+ lengths for BiLSTM)  
5. **Transformers:** `AutoTokenizer` encode train/val with `padding="max_length"` and `truncation=True`

Class weights, focal loss, threshold tuning, and GloVe file loading are **not** included here.

---

## Related

- **`eda/EDA.ipynb`** — exploratory analysis  
- **Root `README.md`** — data download location  
