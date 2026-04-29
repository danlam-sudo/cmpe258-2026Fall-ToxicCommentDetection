"""
One function per model family — load, split, normalize, and encode.

    from preprocessing.text_preprocessing import (
        LABEL_COLUMNS,
        PAD_TOKEN,
        UNK_TOKEN,
        preprocess_for_cnn,
        preprocess_for_bilstm,
        preprocess_for_bert,
        preprocess_for_distilbert,
        CNNPreprocessResult,
        BiLSTMPreprocessResult,
        TransformerPreprocessResult,
    )

BERT/DistilBERT require ``pip install transformers`` (and usually ``torch``).
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Six binary toxicity heads (order matches train.csv).
LABEL_COLUMNS: tuple[str, ...] = (
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
)

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_TRAIN_PATH = _REPO_ROOT / "data" / "train.csv"


# --- result objects (what each preprocess_for_* returns) --------------------------------


@dataclass
class CNNPreprocessResult:
    """CNN + GloVe: padded word-id matrices aligned to ``vocab``."""

    train_df: pd.DataFrame
    val_df: pd.DataFrame
    vocab: dict[str, int]
    X_train: np.ndarray
    X_val: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray


@dataclass
class BiLSTMPreprocessResult:
    """BiLSTM: same as CNN plus sequence lengths for packing / masking."""

    train_df: pd.DataFrame
    val_df: pd.DataFrame
    vocab: dict[str, int]
    X_train: np.ndarray
    X_val: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    length_train: np.ndarray
    length_val: np.ndarray


@dataclass
class TransformerPreprocessResult:
    """BERT / DistilBERT: HF tokenizer outputs + labels."""

    train_df: pd.DataFrame
    val_df: pd.DataFrame
    tokenizer: Any
    train_encodings: Any
    val_encodings: Any
    y_train: np.ndarray
    y_val: np.ndarray


# --- private helpers --------------------------------------------------------------------


def _normalize_comment_text(text: str | float | None) -> str:
    if text is None:
        return ""
    if isinstance(text, float) and math.isnan(text):
        return ""
    s = unicodedata.normalize("NFKC", str(text))
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _tokenize_words(text: str) -> list[str]:
    if not text:
        return []
    return text.replace("\n", " ").split()


def _load_train_dataframe(csv_path: Path | str | None) -> pd.DataFrame:
    path = Path(csv_path) if csv_path is not None else _DEFAULT_TRAIN_PATH
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Place Kaggle CSVs under data/ (see README).")
    df = pd.read_csv(path)
    missing = [c for c in ("id", "comment_text", *LABEL_COLUMNS) if c not in df.columns]
    if missing:
        raise ValueError(f"train.csv missing columns: {missing}")
    return df


def _train_validation_split(
    df: pd.DataFrame,
    *,
    validation_fraction: float,
    random_state: int,
    use_iterative_stratify: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1 (exclusive).")
    if use_iterative_stratify:
        try:
            from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
        except ImportError as e:
            raise ImportError(
                "Iterative stratification requires `iterative-stratification`. "
                "Install with: pip install iterative-stratification"
            ) from e

        y = df[list(LABEL_COLUMNS)].values
        idx = np.arange(len(df))
        splitter = MultilabelStratifiedShuffleSplit(
            n_splits=1,
            test_size=validation_fraction,
            random_state=random_state,
        )
        train_idx, val_idx = next(splitter.split(idx, y))
        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()
        return train_df, val_df

    y_strat = df[list(LABEL_COLUMNS)].sum(axis=1).astype(int)
    return train_test_split(
        df,
        test_size=validation_fraction,
        random_state=random_state,
        shuffle=True,
        stratify=y_strat,
    )


def _with_normalized_comment_text(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["comment_text"] = out["comment_text"].map(_normalize_comment_text)
    return out


def _labels_array(df: pd.DataFrame) -> np.ndarray:
    return df[list(LABEL_COLUMNS)].values.astype(np.float32)


def _dataset_balance_stats(df: pd.DataFrame) -> dict[str, int]:
    y = df[list(LABEL_COLUMNS)].values
    out: dict[str, int] = {
        "n_rows": int(len(df)),
        "n_clean": int((y.sum(axis=1) == 0).sum()),
    }
    for i, label in enumerate(LABEL_COLUMNS):
        out[f"pos_{label}"] = int(y[:, i].sum())
    return out


def _print_balance_stats(title: str, stats: dict[str, int]) -> None:
    pieces = [
        f"rows={stats['n_rows']}",
        f"clean={stats['n_clean']}",
    ]
    pieces.extend(f"{label}={stats[f'pos_{label}']}" for label in LABEL_COLUMNS)
    print(f"[preprocess] {title}: " + ", ".join(pieces))


def _rebalance_train_dataframe(
    train_df: pd.DataFrame,
    *,
    clean_to_toxic_ratio: float,
    rare_labels: Sequence[str],
    rare_oversample_factor: float,
    max_copies_per_row: int,
    rebalance_random_state: int,
) -> pd.DataFrame:
    if clean_to_toxic_ratio < 0:
        raise ValueError("clean_to_toxic_ratio must be >= 0.")
    if rare_oversample_factor < 0:
        raise ValueError("rare_oversample_factor must be >= 0.")
    if max_copies_per_row < 1:
        raise ValueError("max_copies_per_row must be >= 1.")
    invalid_labels = [c for c in rare_labels if c not in LABEL_COLUMNS]
    if invalid_labels:
        raise ValueError(f"Invalid rare_labels: {invalid_labels}")

    rng = np.random.default_rng(rebalance_random_state)
    work = train_df.reset_index(drop=True).copy()
    y = work[list(LABEL_COLUMNS)].values
    label_sum = y.sum(axis=1)
    clean_idx = np.flatnonzero(label_sum == 0)
    toxic_idx = np.flatnonzero(label_sum > 0)

    if len(toxic_idx) == 0:
        return work

    target_clean = int(round(clean_to_toxic_ratio * len(toxic_idx)))
    if len(clean_idx) > target_clean:
        keep_clean = rng.choice(clean_idx, size=target_clean, replace=False)
    else:
        keep_clean = clean_idx
    keep_idx = np.concatenate([toxic_idx, keep_clean])
    keep_idx.sort()
    rebalanced = work.iloc[keep_idx].copy().reset_index(drop=True)

    if rare_oversample_factor <= 0 or len(rare_labels) == 0:
        return rebalanced

    rare_mask = rebalanced[list(rare_labels)].sum(axis=1).values > 0
    rare_pos_idx = np.flatnonzero(rare_mask)
    if len(rare_pos_idx) == 0:
        return rebalanced

    n_to_add = int(round((rare_oversample_factor - 1.0) * len(rare_pos_idx)))
    if n_to_add <= 0:
        return rebalanced

    counts = np.ones(len(rebalanced), dtype=np.int64)
    added: list[int] = []
    shuffled_rare = rng.permutation(rare_pos_idx).tolist()
    cursor = 0
    while len(added) < n_to_add:
        if cursor >= len(shuffled_rare):
            shuffled_rare = rng.permutation(rare_pos_idx).tolist()
            cursor = 0
        idx = int(shuffled_rare[cursor])
        cursor += 1
        if counts[idx] >= max_copies_per_row:
            if np.all(counts[rare_pos_idx] >= max_copies_per_row):
                break
            continue
        counts[idx] += 1
        added.append(idx)

    if added:
        extra = rebalanced.iloc[added].copy()
        rebalanced = pd.concat([rebalanced, extra], ignore_index=True)
        order = rng.permutation(len(rebalanced))
        rebalanced = rebalanced.iloc[order].reset_index(drop=True)
    return rebalanced


def _fit_word_vocabulary_from_texts(
    texts: Iterable[str],
    *,
    min_freq: int,
    max_vocab: int | None,
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(_tokenize_words(text))

    vocab: dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for word, _count in counts.most_common():
        if _count < min_freq:
            break
        if word in vocab:
            continue
        if max_vocab is not None and len(vocab) >= max_vocab:
            break
        vocab[word] = len(vocab)
    return vocab


def _text_to_word_ids(text: str, vocab: dict[str, int]) -> list[int]:
    unk = vocab[UNK_TOKEN]
    return [vocab.get(w, unk) for w in _tokenize_words(text)]


def _texts_to_padded_word_ids(
    texts: Iterable[str],
    vocab: dict[str, int],
    max_len: int,
) -> np.ndarray:
    pad_id = vocab[PAD_TOKEN]
    seqs = [_text_to_word_ids(t, vocab) for t in texts]
    out = np.full((len(seqs), max_len), pad_id, dtype=np.int64)
    for i, seq in enumerate(seqs):
        seq = seq[:max_len]
        if seq:
            out[i, : len(seq)] = seq
    return out


def _word_sequence_lengths(
    texts: Iterable[str],
    vocab: dict[str, int],
    max_len: int,
) -> np.ndarray:
    lengths = []
    for t in texts:
        n = min(len(_text_to_word_ids(t, vocab)), max_len)
        lengths.append(n)
    return np.array(lengths, dtype=np.int64)


def _word_model_pipeline(
    *,
    csv_path: Path | str | None,
    validation_fraction: float,
    random_state: int,
    max_len: int,
    min_freq: int,
    max_vocab: int | None,
    include_lengths: bool,
    max_train_samples: int | None,
    max_val_samples: int | None,
    use_iterative_stratify: bool,
    rebalance_train: bool,
    clean_to_toxic_ratio: float,
    rare_labels: Sequence[str],
    rare_oversample_factor: float,
    max_copies_per_row: int,
    rebalance_random_state: int,
    print_diagnostics: bool,
) -> CNNPreprocessResult | BiLSTMPreprocessResult:
    df = _load_train_dataframe(csv_path)
    train_df, val_df = _train_validation_split(
        df,
        validation_fraction=validation_fraction,
        random_state=random_state,
        use_iterative_stratify=use_iterative_stratify,
    )
    train_df = _with_normalized_comment_text(train_df)
    val_df = _with_normalized_comment_text(val_df)
    train_stats_before = _dataset_balance_stats(train_df)
    val_stats_before = _dataset_balance_stats(val_df)
    if rebalance_train:
        train_df = _rebalance_train_dataframe(
            train_df,
            clean_to_toxic_ratio=clean_to_toxic_ratio,
            rare_labels=rare_labels,
            rare_oversample_factor=rare_oversample_factor,
            max_copies_per_row=max_copies_per_row,
            rebalance_random_state=rebalance_random_state,
        )
    if print_diagnostics:
        _print_balance_stats("train_before", train_stats_before)
        _print_balance_stats("train_after", _dataset_balance_stats(train_df))
        _print_balance_stats("val_unchanged", val_stats_before)
    if max_train_samples is not None:
        train_df = train_df.iloc[:max_train_samples].reset_index(drop=True)
    if max_val_samples is not None:
        val_df = val_df.iloc[:max_val_samples].reset_index(drop=True)

    vocab = _fit_word_vocabulary_from_texts(
        train_df["comment_text"],
        min_freq=min_freq,
        max_vocab=max_vocab,
    )
    X_train = _texts_to_padded_word_ids(train_df["comment_text"], vocab, max_len)
    X_val = _texts_to_padded_word_ids(val_df["comment_text"], vocab, max_len)
    y_train = _labels_array(train_df)
    y_val = _labels_array(val_df)

    if not include_lengths:
        return CNNPreprocessResult(
            train_df=train_df,
            val_df=val_df,
            vocab=vocab,
            X_train=X_train,
            X_val=X_val,
            y_train=y_train,
            y_val=y_val,
        )
    return BiLSTMPreprocessResult(
        train_df=train_df,
        val_df=val_df,
        vocab=vocab,
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        length_train=_word_sequence_lengths(train_df["comment_text"], vocab, max_len),
        length_val=_word_sequence_lengths(val_df["comment_text"], vocab, max_len),
    )


def _transformer_pipeline(
    *,
    csv_path: Path | str | None,
    validation_fraction: float,
    random_state: int,
    pretrained_model_name: str,
    max_length: int,
    return_tensors: str | None,
    max_train_samples: int | None,
    max_val_samples: int | None,
    use_iterative_stratify: bool,
    rebalance_train: bool,
    clean_to_toxic_ratio: float,
    rare_labels: Sequence[str],
    rare_oversample_factor: float,
    max_copies_per_row: int,
    rebalance_random_state: int,
    print_diagnostics: bool,
) -> TransformerPreprocessResult:
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "Transformer preprocessing requires `transformers` (and usually `torch`). "
            "Install with: pip install transformers torch"
        ) from e

    df = _load_train_dataframe(csv_path)
    train_df, val_df = _train_validation_split(
        df,
        validation_fraction=validation_fraction,
        random_state=random_state,
        use_iterative_stratify=use_iterative_stratify,
    )
    train_df = _with_normalized_comment_text(train_df)
    val_df = _with_normalized_comment_text(val_df)
    train_stats_before = _dataset_balance_stats(train_df)
    val_stats_before = _dataset_balance_stats(val_df)
    if rebalance_train:
        train_df = _rebalance_train_dataframe(
            train_df,
            clean_to_toxic_ratio=clean_to_toxic_ratio,
            rare_labels=rare_labels,
            rare_oversample_factor=rare_oversample_factor,
            max_copies_per_row=max_copies_per_row,
            rebalance_random_state=rebalance_random_state,
        )
    if print_diagnostics:
        _print_balance_stats("train_before", train_stats_before)
        _print_balance_stats("train_after", _dataset_balance_stats(train_df))
        _print_balance_stats("val_unchanged", val_stats_before)
    if max_train_samples is not None:
        train_df = train_df.iloc[:max_train_samples].reset_index(drop=True)
    if max_val_samples is not None:
        val_df = val_df.iloc[:max_val_samples].reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    train_texts = train_df["comment_text"].tolist()
    val_texts = val_df["comment_text"].tolist()

    train_encodings = tokenizer(
        train_texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors=return_tensors,
    )
    val_encodings = tokenizer(
        val_texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors=return_tensors,
    )

    return TransformerPreprocessResult(
        train_df=train_df,
        val_df=val_df,
        tokenizer=tokenizer,
        train_encodings=train_encodings,
        val_encodings=val_encodings,
        y_train=_labels_array(train_df),
        y_val=_labels_array(val_df),
    )


# --- public: one entry point per model ---------------------------------------------------


def preprocess_for_cnn(
    *,
    csv_path: Path | str | None = None,
    validation_fraction: float = 0.1,
    random_state: int = 42,
    max_len: int = 256,
    min_freq: int = 2,
    max_vocab: int | None = 50_000,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    use_iterative_stratify: bool = False,
    rebalance_train: bool = False,
    clean_to_toxic_ratio: float = 3.0,
    rare_labels: Sequence[str] = ("severe_toxic", "threat", "identity_hate"),
    rare_oversample_factor: float = 1.0,
    max_copies_per_row: int = 3,
    rebalance_random_state: int = 42,
    print_diagnostics: bool = False,
) -> CNNPreprocessResult:
    """
    Load ``train.csv``, stratified train/val split, normalize text, fit word
    vocabulary on train only, build padded word-id matrices for CNN + GloVe.

    Align pretrained GloVe rows to ``result.vocab`` ids in your model code.
    Optional ``max_train_samples`` / ``max_val_samples`` cap rows after the split
    (for fast smoke tests).
    """
    out = _word_model_pipeline(
        csv_path=csv_path,
        validation_fraction=validation_fraction,
        random_state=random_state,
        max_len=max_len,
        min_freq=min_freq,
        max_vocab=max_vocab,
        include_lengths=False,
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
        use_iterative_stratify=use_iterative_stratify,
        rebalance_train=rebalance_train,
        clean_to_toxic_ratio=clean_to_toxic_ratio,
        rare_labels=rare_labels,
        rare_oversample_factor=rare_oversample_factor,
        max_copies_per_row=max_copies_per_row,
        rebalance_random_state=rebalance_random_state,
        print_diagnostics=print_diagnostics,
    )
    assert isinstance(out, CNNPreprocessResult)
    return out


def preprocess_for_bilstm(
    *,
    csv_path: Path | str | None = None,
    validation_fraction: float = 0.1,
    random_state: int = 42,
    max_len: int = 256,
    min_freq: int = 2,
    max_vocab: int | None = 50_000,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    use_iterative_stratify: bool = False,
    rebalance_train: bool = False,
    clean_to_toxic_ratio: float = 3.0,
    rare_labels: Sequence[str] = ("severe_toxic", "threat", "identity_hate"),
    rare_oversample_factor: float = 1.0,
    max_copies_per_row: int = 3,
    rebalance_random_state: int = 42,
    print_diagnostics: bool = False,
) -> BiLSTMPreprocessResult:
    """
    Same pipeline as CNN plus ``length_train`` / ``length_val`` for
    ``pack_padded_sequence`` or attention masking.
    """
    out = _word_model_pipeline(
        csv_path=csv_path,
        validation_fraction=validation_fraction,
        random_state=random_state,
        max_len=max_len,
        min_freq=min_freq,
        max_vocab=max_vocab,
        include_lengths=True,
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
        use_iterative_stratify=use_iterative_stratify,
        rebalance_train=rebalance_train,
        clean_to_toxic_ratio=clean_to_toxic_ratio,
        rare_labels=rare_labels,
        rare_oversample_factor=rare_oversample_factor,
        max_copies_per_row=max_copies_per_row,
        rebalance_random_state=rebalance_random_state,
        print_diagnostics=print_diagnostics,
    )
    assert isinstance(out, BiLSTMPreprocessResult)
    return out


def preprocess_for_bert(
    *,
    csv_path: Path | str | None = None,
    validation_fraction: float = 0.1,
    random_state: int = 42,
    pretrained_model_name: str = "bert-base-uncased",
    max_length: int = 512,
    return_tensors: str | None = "pt",
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    use_iterative_stratify: bool = False,
    rebalance_train: bool = False,
    clean_to_toxic_ratio: float = 3.0,
    rare_labels: Sequence[str] = ("severe_toxic", "threat", "identity_hate"),
    rare_oversample_factor: float = 1.0,
    max_copies_per_row: int = 3,
    rebalance_random_state: int = 42,
    print_diagnostics: bool = False,
) -> TransformerPreprocessResult:
    """
    Load, split, normalize, then tokenize with a BERT ``AutoTokenizer``.

    ``train_encodings`` / ``val_encodings`` include ``input_ids`` and
    ``attention_mask`` (and ``token_type_ids`` if the model provides them).
    Optional ``max_train_samples`` / ``max_val_samples`` cap rows after the split.
    """
    return _transformer_pipeline(
        csv_path=csv_path,
        validation_fraction=validation_fraction,
        random_state=random_state,
        pretrained_model_name=pretrained_model_name,
        max_length=max_length,
        return_tensors=return_tensors,
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
        use_iterative_stratify=use_iterative_stratify,
        rebalance_train=rebalance_train,
        clean_to_toxic_ratio=clean_to_toxic_ratio,
        rare_labels=rare_labels,
        rare_oversample_factor=rare_oversample_factor,
        max_copies_per_row=max_copies_per_row,
        rebalance_random_state=rebalance_random_state,
        print_diagnostics=print_diagnostics,
    )


def preprocess_for_distilbert(
    *,
    csv_path: Path | str | None = None,
    validation_fraction: float = 0.1,
    random_state: int = 42,
    pretrained_model_name: str = "distilbert-base-uncased",
    max_length: int = 512,
    return_tensors: str | None = "pt",
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    use_iterative_stratify: bool = False,
    rebalance_train: bool = False,
    clean_to_toxic_ratio: float = 3.0,
    rare_labels: Sequence[str] = ("severe_toxic", "threat", "identity_hate"),
    rare_oversample_factor: float = 1.0,
    max_copies_per_row: int = 3,
    rebalance_random_state: int = 42,
    print_diagnostics: bool = False,
) -> TransformerPreprocessResult:
    """
    Same as ``preprocess_for_bert`` but defaults to DistilBERT weights.
    """
    return _transformer_pipeline(
        csv_path=csv_path,
        validation_fraction=validation_fraction,
        random_state=random_state,
        pretrained_model_name=pretrained_model_name,
        max_length=max_length,
        return_tensors=return_tensors,
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
        use_iterative_stratify=use_iterative_stratify,
        rebalance_train=rebalance_train,
        clean_to_toxic_ratio=clean_to_toxic_ratio,
        rare_labels=rare_labels,
        rare_oversample_factor=rare_oversample_factor,
        max_copies_per_row=max_copies_per_row,
        rebalance_random_state=rebalance_random_state,
        print_diagnostics=print_diagnostics,
    )
