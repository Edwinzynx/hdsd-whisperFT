"""Fine-tune openai/whisper-small on HDSD with AdaLoRA (PEFT).

Run from the HDSD/ directory:
    python train_whisper_adalora.py

Outputs
-------
whisper-adalora-hdsd/          Seq2SeqTrainer checkpoints
whisper-adalora-hdsd/final_adapter/   PEFT adapter weights only (load separately
                                      from the frozen whisper-small base)
"""

# ── 0. Imports ────────────────────────────────────────────────────────────────

import os
import sys
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset

from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from peft import AdaLoraConfig, TaskType, get_peft_model

import jiwer

from hdsd_augment import augment as hdsd_augment


# ── 1. Paths & Hyper-parameters ───────────────────────────────────────────────

BASE_DIR   = Path(__file__).parent.resolve()   # .../HDSD/
TRAIN_CSV  = BASE_DIR / "train.csv"
VAL_CSV    = BASE_DIR / "val.csv"
TEST_CSV   = BASE_DIR / "test.csv"
OUTPUT_DIR = BASE_DIR / "whisper-adalora-hdsd"

MODEL_ID   = "openai/whisper-small"
LANGUAGE   = "hi"
TASK       = "transcribe"
SAMPLE_RATE = 16_000

# AdaLoRA
ADALORA_INIT_R   = 12
ADALORA_TARGET_R = 8
ADALORA_ALPHA    = 32
ADALORA_DROPOUT  = 0.05

# Training
TRAIN_BATCH_SIZE   = 8
GRAD_ACCUM_STEPS   = 2          # effective batch = 16
LEARNING_RATE      = 1e-4
WARMUP_STEPS       = 500
MAX_STEPS          = 4_000
EVAL_STEPS         = 500
SAVE_STEPS         = 500
LOGGING_STEPS      = 25
EVAL_BATCH_SIZE    = 8
GENERATION_MAX_LEN = 225
NUM_WORKERS        = 4

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

EARLY_STOPPING_PATIENCE = 5   # eval rounds without WER improvement → stop training

# Logging backend: prefer WandB (richer dashboards); fall back to TensorBoard
# which ships with transformers and needs no account.
try:
    import wandb  # noqa: F401
    REPORT_TO = ["wandb", "tensorboard"]
    print("WandB detected — logging to WandB + TensorBoard")
except ImportError:
    REPORT_TO = ["tensorboard"]
    print("WandB not found — logging to TensorBoard  "
          "(install wandb and run `wandb login` to enable)")


# ── 1.5 Checkpoint Resume Guard ───────────────────────────────────────────────
# If a previous run was interrupted, pick up from the latest saved checkpoint.
# Trainer.train(resume_from_checkpoint=...) restores model weights, optimizer
# state, and step count — no re-running already-completed steps.

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_ckpts = sorted(
    [d for d in OUTPUT_DIR.glob("checkpoint-*") if d.is_dir()],
    key=lambda p: int(p.name.split("-")[-1]),
)
RESUME_FROM: str | None = str(_ckpts[-1]) if _ckpts else None
if RESUME_FROM:
    print(f"Resuming from checkpoint: {RESUME_FROM}")
else:
    print("No checkpoint found — starting fresh run")


# ── 2. Load CSVs ──────────────────────────────────────────────────────────────
# `transcript` column holds Devanagari text (what Whisper-small outputs for
# Hindi). `transcript_roman` (romanized) is present but not used for training.

def load_split(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Drop any rows with missing transcript or audio path
    df = df[df["transcript"].notna() & (df["transcript"].str.strip() != "")]
    df = df[df["path"].notna()]
    return df.reset_index(drop=True)


train_df = load_split(TRAIN_CSV)
val_df   = load_split(VAL_CSV)
test_df  = load_split(TEST_CSV)

print(f"Loaded  train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")


# ── 3. WhisperProcessor ───────────────────────────────────────────────────────

processor = WhisperProcessor.from_pretrained(MODEL_ID, language=LANGUAGE, task=TASK)
tokenizer = processor.tokenizer


# ── 4. Dataset ────────────────────────────────────────────────────────────────

class HDSDDataset(Dataset):
    """Loads a WAV file, optionally augments it, then returns Whisper features.

    On-the-fly augmentation is only applied when `apply_augment=True` (train
    split). Val/test always run the clean signal so eval WER is reproducible.
    """

    def __init__(self, df: pd.DataFrame, apply_augment: bool = False):
        self.records = df[["path", "transcript"]].values
        self.apply_augment = apply_augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        path, transcript = self.records[idx]

        # Load audio ──────────────────────────────────────────────────────────
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)          # stereo → mono (safety)
        if sr != SAMPLE_RATE:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

        # Augment (training only) ─────────────────────────────────────────────
        if self.apply_augment:
            audio = hdsd_augment(audio, SAMPLE_RATE)

        # Whisper log-mel features ────────────────────────────────────────────
        # feature_extractor pads/truncates to exactly 30 s (480 000 samples)
        # and returns shape (80, 3000).
        input_features = processor.feature_extractor(
            audio, sampling_rate=SAMPLE_RATE, return_tensors="np"
        ).input_features[0]                     # (80, 3000)

        # Tokenise transcript ─────────────────────────────────────────────────
        labels = tokenizer(transcript.strip()).input_ids

        return {"input_features": input_features, "labels": labels}


train_dataset = HDSDDataset(train_df, apply_augment=True)
val_dataset   = HDSDDataset(val_df,   apply_augment=False)
test_dataset  = HDSDDataset(test_df,  apply_augment=False)


# ── 5. AdaLoRA Model ─────────────────────────────────────────────────────────

base_model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)

# Required when gradient_checkpointing=True with PEFT
base_model.config.use_cache = False

# Force Hindi transcription during generate() calls (eval + inference)
base_model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
    language=LANGUAGE, task=TASK
)
base_model.config.suppress_tokens = []

adalora_config = AdaLoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    # Rank schedule: start wide, compress to target_r by tfinal
    init_r=ADALORA_INIT_R,
    target_r=ADALORA_TARGET_R,
    lora_alpha=ADALORA_ALPHA,
    lora_dropout=ADALORA_DROPOUT,
    # Apply to query + value projections in all encoder & decoder attention layers
    target_modules=["q_proj", "v_proj"],
    # Rank-adjustment schedule (steps)
    tinit=200,           # warm-up before adjustment starts
    tfinal=3_500,        # freeze ranks for the last 500 steps
    deltaT=100,          # re-evaluate every 100 steps
    beta1=0.85,
    beta2=0.85,
    orth_reg_weight=0.5,
    # Don't save the frozen base weights inside the adapter checkpoint
    modules_to_save=None,
)

model = get_peft_model(base_model, adalora_config)
model.enable_input_require_grads()   # needed for grad-checkpointing + PEFT
model.print_trainable_parameters()


# ── 6. Data Collator ─────────────────────────────────────────────────────────

@dataclass
class WhisperDataCollator:
    """Pads a batch of (input_features, labels) pairs for Seq2SeqTrainer.

    input_features: already fixed-size (80, 3000) — just stack into a tensor.
    labels:         variable-length token sequences → pad with -100 so the
                    cross-entropy loss ignores padding positions.
    """
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: list[dict]) -> dict:
        # ── input features ────────────────────────────────────────────────────
        input_features = torch.tensor(
            np.stack([f["input_features"] for f in features]), dtype=torch.float32
        )  # (B, 80, 3000)

        # ── labels ────────────────────────────────────────────────────────────
        label_ids = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_ids, return_tensors="pt", padding=True
        )
        # Replace pad positions with -100 (ignored by cross-entropy)
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1), -100
        )
        # Strip leading decoder-start token if the tokenizer prepended it —
        # Seq2SeqTrainer shifts labels internally.
        if (labels[:, 0] == self.decoder_start_token_id).all():
            labels = labels[:, 1:]

        return {"input_features": input_features, "labels": labels}


data_collator = WhisperDataCollator(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)


# ── 7. WER Metric ────────────────────────────────────────────────────────────

def compute_metrics(pred) -> dict:
    pred_ids  = pred.predictions
    label_ids = pred.label_ids

    # Restore pad positions that were masked to -100
    label_ids = np.where(label_ids == -100, tokenizer.pad_token_id, label_ids)

    pred_strs  = tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
    label_strs = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Normalise whitespace before scoring
    pred_strs  = [s.strip() for s in pred_strs]
    label_strs = [s.strip() for s in label_strs]

    wer_score = 100.0 * jiwer.wer(label_strs, pred_strs)
    return {"wer": round(wer_score, 2)}


# ── 7.5 CSV Logger Callback ───────────────────────────────────────────────────
# Writes every logged metric (train loss, eval WER, learning rate, …) to
# OUTPUT_DIR/training_log.csv so progress is readable without TensorBoard.
# pd.DataFrame handles the sparse-key case (train rows lack eval_wer etc.)
# by filling those cells with NaN automatically.

class CSVLoggerCallback(TrainerCallback):
    def __init__(self, output_dir: Path) -> None:
        self.path = Path(output_dir) / "training_log.csv"
        self._rows: list[dict] = []

    def on_log(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        logs: dict | None = None,
        **kwargs,
    ) -> None:
        if not logs:
            return
        row: dict = {"step": state.global_step, "epoch": round(state.epoch or 0.0, 3)}
        row.update(logs)
        self._rows.append(row)
        # Re-write whole file on each event so it's always a valid CSV
        # mid-training; at logging_steps=25 this is cheap.
        pd.DataFrame(self._rows).to_csv(self.path, index=False)


# ── 7.7 Whisper-compatible Trainer + AdaLoRA Rank-Update Callback ────────────
#
# Two bugs fixed here:
#
# Bug 1 — PeftModelForSeq2SeqLM forward dispatch:
#   PEFT's SEQ_2_SEQ_LM wrapper calls self.base_model(input_ids=input_ids, ...)
#   even when input_ids=None.  Whisper's forward uses input_features=, not
#   input_ids=, so that kwarg causes "TypeError: multiple values for input_ids".
#   Fix: override compute_loss to call model.base_model.model() directly.
#   The AdaLoRA adapters live inside q_proj/v_proj — they are still active.
#
# Bug 2 — AdaLoRA rank adaptation never triggered:
#   AdaLoRA only adjusts ranks when update_and_allocate(step) is called after
#   each optimizer step.  Without it the model trains as fixed-rank LoRA at
#   init_r=12 the whole time.  Fix: AdaLoraRankUpdateCallback.

class AdaLoraRankUpdateCallback(TrainerCallback):
    """Calls model.base_model.update_and_allocate(step) after every optimizer
    step so AdaLoRA's SVD-based rank pruning actually runs."""

    def on_step_end(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ) -> None:
        if model is not None and hasattr(model, "base_model"):
            model.base_model.update_and_allocate(state.global_step)


class HDSDSeq2SeqTrainer(Seq2SeqTrainer):
    """Seq2SeqTrainer subclass that routes the forward pass directly to the
    underlying WhisperForConditionalGeneration, bypassing PEFT's
    PeftModelForSeq2SeqLM.forward which injects input_ids= (Whisper has none)."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model.base_model.model(
            input_features=inputs["input_features"],
            labels=inputs["labels"],
        )
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


# ── 8. Training Arguments ─────────────────────────────────────────────────────

training_args = Seq2SeqTrainingArguments(
    output_dir=str(OUTPUT_DIR),

    # Batch / accumulation
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,

    # Optimiser
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    max_steps=MAX_STEPS,

    # Precision & memory
    fp16=True,
    gradient_checkpointing=True,

    # Evaluation & checkpointing
    evaluation_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,         # lower WER is better

    # Generation (used during predict_with_generate eval)
    predict_with_generate=True,
    generation_max_length=GENERATION_MAX_LEN,

    # Logging
    logging_steps=LOGGING_STEPS,
    logging_dir=str(OUTPUT_DIR / "logs"),
    report_to=REPORT_TO,

    # Misc
    remove_unused_columns=False,     # our Dataset uses dict keys, not HF columns
    dataloader_num_workers=NUM_WORKERS,
    seed=SEED,
    push_to_hub=False,
)


# ── 9. Pre-training Summary ───────────────────────────────────────────────────

_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
_frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
_total     = _trainable + _frozen
_fp16      = torch.cuda.is_available()
_eff_bs    = TRAIN_BATCH_SIZE * GRAD_ACCUM_STEPS

if torch.cuda.is_available():
    # Conservative breakdown: weights (fp16) + fp32 grads + Adam states + activations
    _w  = sum(p.numel() * 2 for p in model.parameters())          # fp16 weights
    _g  = _trainable * 4                                           # fp32 grads
    _o  = _trainable * 8                                           # Adam m + v (fp32)
    _a  = _eff_bs * 80 * 3000 * 2                                  # fp16 activations
    _est_gb  = (_w + _g + _o + _a) / 1024**3
    _avail   = torch.cuda.get_device_properties(0).total_memory / 1024**3
    _gpu_str = f"{torch.cuda.get_device_name(0)} ({_avail:.1f} GB total)"
else:
    _est_gb  = 0.0
    _gpu_str = "No CUDA — CPU only (training will be very slow)"

print("\n" + "=" * 64)
print("  PRE-TRAINING SUMMARY")
print("=" * 64)
print(f"  Trainable (AdaLoRA)   : {_trainable:>12,}  ({100*_trainable/_total:.2f}%)")
print(f"  Frozen  (base Whisper): {_frozen:>12,}  ({100*_frozen/_total:.2f}%)")
print(f"  Effective batch size  : {_eff_bs}  "
      f"({TRAIN_BATCH_SIZE} × {GRAD_ACCUM_STEPS} grad-accum steps)")
print(f"  FP16 active           : {_fp16}")
print(f"  Est. GPU memory       : {_est_gb:.2f} GB")
print(f"  GPU                   : {_gpu_str}")
print(f"  Logging backend       : {REPORT_TO}")
print(f"  CSV log               : {OUTPUT_DIR}/training_log.csv")
print(f"  Resume from           : {RESUME_FROM or 'scratch (step 0)'}")
print(f"  Early-stop patience   : {EARLY_STOPPING_PATIENCE} eval rounds  "
      f"(= {EARLY_STOPPING_PATIENCE * EVAL_STEPS} steps without WER improvement)")
print("=" * 64 + "\n")


# ── 10. Trainer + Training ────────────────────────────────────────────────────

trainer = HDSDSeq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE),
        CSVLoggerCallback(OUTPUT_DIR),
        AdaLoraRankUpdateCallback(),
    ],
)

print("── Starting training ────────────────────────────────────────────────")
trainer.train(resume_from_checkpoint=RESUME_FROM)


# ── 11. Save Adapter Weights ─────────────────────────────────────────────────
# Saves only the LoRA delta weights (~few MB). The frozen whisper-small base
# model is loaded separately at inference time via PeftModel.from_pretrained().

adapter_dir = OUTPUT_DIR / "final_adapter"
model.save_pretrained(str(adapter_dir))
processor.save_pretrained(str(adapter_dir))

print(f"\n── Adapter saved to  {adapter_dir}")
print(
    "   Load at inference:\n"
    "     from peft import PeftModel\n"
    "     base = WhisperForConditionalGeneration.from_pretrained('openai/whisper-small')\n"
    f"     model = PeftModel.from_pretrained(base, '{adapter_dir}')\n"
)


# ── 12. Final Test-Set Evaluation ─────────────────────────────────────────────

print("\n── Evaluating on held-out test set ─────────────────────────────────")
test_results = trainer.predict(test_dataset)
test_wer = compute_metrics(test_results)["wer"]
print(f"   Test WER: {test_wer:.2f}%")
