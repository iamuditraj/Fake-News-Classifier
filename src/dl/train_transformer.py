"""
Phase 2 – Transformer Fine-Tuning for Fake News Detection
==========================================================
Section 1: Imports & Dataset Preparation
Section 2: Evaluation Metrics & Training Configuration
Section 3: CLI Interface & Main Execution

This module bridges the Phase 1 data pipeline (pandas-based) into
HuggingFace's `datasets` / `transformers` ecosystem so we can
fine-tune a pre-trained transformer on the WELFake corpus.

Usage
-----
    python src/dl/train_transformer.py
    python src/dl/train_transformer.py --model roberta-base --epochs 5 --batch 4
"""

# ── Standard library ────────────────────────────────────────────────
import argparse
import json
import os
import sys

# ── Third-party: data handling ──────────────────────────────────────
import numpy as np
import pandas as pd
from scipy.special import softmax

# ── Third-party: evaluation ────────────────────────────────────────
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# ── HuggingFace: datasets ──────────────────────────────────────────
from datasets import Dataset

# ── HuggingFace: transformers ───────────────────────────────────────
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# ── Phase 1 reuse ──────────────────────────────────────────────────
# Add the parent `src/` directory to sys.path so we can import data.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data import load_dataset as load_welfake, get_splits


# ── Constants ───────────────────────────────────────────────────────
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 512


# ── Dataset Preparation ────────────────────────────────────────────

def prepare_datasets(csv_path: str = None, model_name: str = None):
    """
    End-to-end dataset preparation pipeline.

    1. Loads the cleaned WELFake DataFrame via Phase 1's ``load_dataset()``.
    2. Splits into train / val / test via ``get_splits()``.
    3. Converts each split from a pandas DataFrame into a HuggingFace
       ``Dataset``, renames ``label`` → ``labels`` (the column name
       expected by HuggingFace ``Trainer``), and sets the format to
       PyTorch tensors.

    Parameters
    ----------
    csv_path : str, optional
        Path to the WELFake CSV.  Defaults to the path baked into
        ``data.load_dataset()``.
    model_name : str, optional
        HuggingFace model identifier for tokenizer selection.
        Defaults to the module-level ``MODEL_NAME``.

    Returns
    -------
    tuple[Dataset, Dataset, Dataset]
        ``(train_ds, val_ds, test_ds)`` – tokenized and formatted for
        PyTorch / HuggingFace Trainer.
    """
    _model = model_name or MODEL_NAME

    # Step 1 – Load & clean via Phase 1
    print("Loading data via Phase 1 pipeline...")
    df = load_welfake(csv_path) if csv_path else load_welfake()
    print(f"  Total samples after cleaning: {len(df):,}")

    # Step 2 – Stratified split (80 / 10 / 10)
    print("Splitting into train / val / test...")
    train_df, val_df, test_df = get_splits(df)
    print(f"  Train: {len(train_df):,}  |  Val: {len(val_df):,}  |  Test: {len(test_df):,}")

    # Step 3 – Convert each split
    train_ds = _df_to_hf_dataset(train_df)
    val_ds   = _df_to_hf_dataset(val_df)
    test_ds  = _df_to_hf_dataset(test_df)

    # Step 4 – Tokenize
    tokenizer = load_tokenizer(_model)
    print(f"Tokenizing with {_model} (max_length={MAX_LENGTH})...")

    train_ds = train_ds.map(lambda batch: tokenize(batch, tokenizer), batched=True)
    val_ds   = val_ds.map(lambda batch: tokenize(batch, tokenizer), batched=True)
    test_ds  = test_ds.map(lambda batch: tokenize(batch, tokenizer), batched=True)

    # Step 5 – Set format for PyTorch
    torch_columns = ["input_ids", "attention_mask", "labels"]
    train_ds.set_format(type="torch", columns=torch_columns)
    val_ds.set_format(type="torch", columns=torch_columns)
    test_ds.set_format(type="torch", columns=torch_columns)

    print("Dataset preparation complete ✓")
    return train_ds, val_ds, test_ds


def _df_to_hf_dataset(df: pd.DataFrame) -> Dataset:
    """
    Convert a pandas DataFrame to a HuggingFace ``Dataset``.

    - Resets the index to avoid HuggingFace indexing issues.
    - Renames ``label`` → ``labels`` (required by ``Trainer``).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ``input_text`` and ``label``.

    Returns
    -------
    Dataset
        A HuggingFace Dataset with columns ``input_text`` and ``labels``.
    """
    df = df.reset_index(drop=True)
    ds = Dataset.from_pandas(df)
    ds = ds.rename_column("label", "labels")
    return ds


# ── Tokenization ───────────────────────────────────────────────────

def load_tokenizer(model_name: str = None):
    """
    Load the pre-trained tokenizer.

    Parameters
    ----------
    model_name : str, optional
        HuggingFace model identifier.  Defaults to ``MODEL_NAME``.

    Returns
    -------
    transformers.PreTrainedTokenizerFast
    """
    return AutoTokenizer.from_pretrained(model_name or MODEL_NAME)


def tokenize(batch, tokenizer):
    """
    Tokenize a batch of examples from the dataset.

    Uses ``padding='max_length'`` so every sequence in the batch is
    padded to exactly ``MAX_LENGTH`` tokens – this guarantees uniform
    tensor shapes without requiring a custom data collator.

    Parameters
    ----------
    batch : dict
        A batch dict produced by ``Dataset.map(batched=True)``.
        Must contain the key ``input_text``.
    tokenizer : transformers.PreTrainedTokenizerFast
        The tokenizer loaded via ``load_tokenizer()``.

    Returns
    -------
    dict
        Tokenized fields: ``input_ids``, ``attention_mask``
        (and ``token_type_ids`` if applicable).
    """
    return tokenizer(
        batch["input_text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )


# ── Evaluation Metrics ─────────────────────────────────────────────

def compute_metrics(eval_pred):
    """
    Compute evaluation metrics for the HuggingFace ``Trainer``.

    This function is passed directly to ``Trainer(compute_metrics=...)``.
    The ``Trainer`` provides raw logits (not probabilities), so we apply
    ``scipy.special.softmax`` to convert them before computing ROC-AUC.

    Metrics computed
    ----------------
    - **accuracy** – overall correctness
    - **macro_f1** – class-balanced F1 (robust to imbalance)
    - **roc_auc**  – area under the ROC curve (uses positive-class prob)

    Parameters
    ----------
    eval_pred : transformers.EvalPrediction
        Named tuple with ``.predictions`` (raw logits) and
        ``.label_ids`` (ground-truth labels).

    Returns
    -------
    dict[str, float]
        Metric name → value mapping consumed by ``Trainer``.
    """
    logits, labels = eval_pred

    # Convert raw logits → probabilities via softmax
    probs = softmax(logits, axis=1)

    # Predicted class = argmax of logits
    preds = np.argmax(logits, axis=1)

    # Probability of the positive class (label = 1) for ROC-AUC
    pos_probs = probs[:, 1]

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="macro"),
        "roc_auc":  roc_auc_score(labels, pos_probs),
    }


# ── Training Configuration ─────────────────────────────────────────
#
# Tuned for an NVIDIA RTX 4060 (8 GB VRAM).
#
# Memory budget breakdown for roberta-base (125 M params):
#   • Model weights (fp16)       ≈ 250 MB
#   • Optimizer states (AdamW)   ≈ 1.0 GB
#   • Activations (batch=8, 512) ≈ 3-4 GB
#   • Headroom                   ≈ 2-3 GB
#
# Effective batch size = per_device_batch × gradient_accumulation
#                      = 8 × 4 = 32
# ────────────────────────────────────────────────────────────────────

def get_training_args(
    output_dir: str = None,
    epochs: int = 3,
    batch_size: int = 8,
) -> TrainingArguments:
    """
    Build a ``TrainingArguments`` object configured for local training
    on an RTX 4060 (8 GB VRAM).

    Key choices
    -----------
    - ``per_device_train_batch_size`` defaults to 8 (keeps VRAM < 7 GB).
    - ``gradient_accumulation_steps`` auto-scales to keep effective
      batch size at 32 (``32 // batch_size``).
    - ``fp16=True`` halves activation memory and speeds up Ampere cores.
    - Evaluation and checkpoint saving happen once per epoch.
    - Logging is sent to **Weights & Biases** (``wandb``).

    Parameters
    ----------
    output_dir : str, optional
        Directory for checkpoints and logs.  Defaults to
        ``<project_root>/models/transformer``.
    epochs : int, optional
        Number of training epochs (default: 3).
    batch_size : int, optional
        Per-device train batch size (default: 8).

    Returns
    -------
    TrainingArguments
    """
    if output_dir is None:
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        output_dir = os.path.join(project_root, "models", "transformer")

    # Keep effective batch ≈ 32 regardless of per-device size
    grad_accum = max(1, 32 // batch_size)

    return TrainingArguments(
        # ── Output ──────────────────────────────────────────────
        output_dir=output_dir,
        overwrite_output_dir=True,

        # ── Training schedule ───────────────────────────────────
        num_train_epochs=epochs,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,

        # ── Batch / memory (RTX 4060 – 8 GB) ───────────────────
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,  # eval is forward-only
        gradient_accumulation_steps=grad_accum,     # effective ≈ 32
        fp16=True,                                  # half-precision on Ampere

        # ── Evaluation & saving (per epoch) ─────────────────────
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,                  # keep only 2 best checkpoints

        # ── Logging ─────────────────────────────────────────────
        logging_dir=os.path.join(output_dir, "logs"),
        logging_strategy="steps",
        logging_steps=50,
        report_to="wandb",

        # ── Misc ────────────────────────────────────────────────
        seed=42,
        dataloader_num_workers=2,
        remove_unused_columns=True,
    )


# ── CLI Interface ──────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the transformer training script.

    Flags
    -----
    --model   : HuggingFace model identifier (default: distilbert-base-uncased)
    --data    : Path to the WELFake CSV file
    --output  : Directory for checkpoints & saved model
    --epochs  : Number of training epochs (default: 3)
    --batch   : Per-device train batch size (default: 8)
    """
    parser = argparse.ArgumentParser(
        description="Phase 2 – Fine-tune a transformer for fake-news detection.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME,
        help=f"HuggingFace model identifier (default: {MODEL_NAME})",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to the WELFake CSV. Defaults to the path in data.py.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Directory for checkpoints & final model (default: models/transformer).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Per-device train batch size (default: 8).",
    )
    return parser.parse_args()


# ── Main Execution ─────────────────────────────────────────────────

def main():
    args = parse_args()

    model_name = args.model
    print(f"\n{'='*60}")
    print(f"  Phase 2 – Transformer Training")
    print(f"  Model : {model_name}")
    print(f"  Epochs: {args.epochs}  |  Batch: {args.batch}")
    print(f"{'='*60}\n")

    # ── 1. Prepare datasets ────────────────────────────────────────
    train_ds, val_ds, test_ds = prepare_datasets(
        csv_path=args.data,
        model_name=model_name,
    )

    # ── 2. Load model ──────────────────────────────────────────────
    print(f"Loading model: {model_name}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
    )
    print(f"  Parameters: {model.num_parameters():,}")

    # ── 3. Build training args ─────────────────────────────────────
    training_args = get_training_args(
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch,
    )
    print(f"  Output dir : {training_args.output_dir}")
    print(f"  Effective batch: {args.batch} × {training_args.gradient_accumulation_steps}"
          f" = {args.batch * training_args.gradient_accumulation_steps}")
    print(f"  fp16: {training_args.fp16}")

    # ── 4. Initialize Trainer ──────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    # ── 5. Train ───────────────────────────────────────────────────
    print("\nStarting training...")
    trainer.train()

    # ── 6. Evaluate on validation set ──────────────────────────────
    print("\nRunning final evaluation on validation set...")
    eval_results = trainer.evaluate()

    print(f"\n--- VALIDATION METRICS ---")
    for key, value in eval_results.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    # ── 7. Save best model + tokenizer ─────────────────────────────
    save_dir = os.path.join(training_args.output_dir, "best_model")
    os.makedirs(save_dir, exist_ok=True)

    print(f"\nSaving best model to {save_dir}...")
    trainer.save_model(save_dir)

    tokenizer = load_tokenizer(model_name)
    tokenizer.save_pretrained(save_dir)
    print(f"  Tokenizer saved alongside model ✓")

    # ── 8. Export metrics to JSON ──────────────────────────────────
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)

    metrics_path = os.path.join(results_dir, "transformer_metrics.json")

    # Flatten HF's eval_ prefix for consistency with Phase 1 output
    export_metrics = {
        k.replace("eval_", ""): v
        for k, v in eval_results.items()
    }
    export_metrics["model_name"] = model_name
    export_metrics["epochs"] = args.epochs
    export_metrics["batch_size"] = args.batch

    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                existing_data = json.load(f)
            if not isinstance(existing_data, list):
                existing_data = [existing_data]
        except Exception:
            existing_data = []
    else:
        existing_data = []
        
    existing_data.append(export_metrics)

    with open(metrics_path, "w") as f:
        json.dump(existing_data, f, indent=4)
    print(f"  Metrics appended to {metrics_path} ✓")

    print(f"\n{'='*60}")
    print("  Training complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
