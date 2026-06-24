"""colab_train.py
Whisper-small + AdaLoRA fine-tuning on HDSD (Hindi Dysarthric Speech).
Designed for Google Colab free tier (T4, ~12 hr session).

Fixes vs. train_whisper_adalora.py
───────────────────────────────────
1.  WandB disabled — report_to=["tensorboard"] always (no login required)
2.  eval_strategy (transformers ≥ 5.x; was evaluation_strategy)
3.  processing_class with inspect-based fallback to tokenizer= for 4.x
4.  AdaLoRA schedule: total_step=2000, tinit=50, tfinal=1500, MAX_STEPS=2000
5.  AdaLoraRankUpdateCallback: broad try/except guards update_and_allocate
6.  HDSDSeq2SeqTrainer: routes forward through model.base_model.model(
    input_features=...) to bypass PEFT's PeftModelForSeq2SeqLM which
    injects input_ids=None — a kwarg Whisper does not accept
7.  CSV path auto-fix: re-anchors hindi_sent/ paths to BASE_DIR so the
    script works wherever HDSD/ is unpacked (Drive, /content, etc.)
8.  EVAL_STEPS = SAVE_STEPS = 250 for checkpoint-on-disconnect safety
"""

import inspect
import os
import re
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

BASE_DIR   = Path(__file__).parent.resolve()
TRAIN_CSV  = BASE_DIR / "train.csv"
VAL_CSV    = BASE_DIR / "val.csv"
TEST_CSV   = BASE_DIR / "test.csv"
OUTPUT_DIR = BASE_DIR / "whisper-adalora-hdsd"

MODEL_ID    = "openai/whisper-small"
LANGUAGE    = "hi"
TASK        = "transcribe"
SAMPLE_RATE = 16_000

# AdaLoRA — rank schedule consistent with MAX_STEPS=2000
ADALORA_INIT_R    = 12
ADALORA_TARGET_R  = 8
ADALORA_ALPHA     = 32
ADALORA_DROPOUT   = 0.05
ADALORA_TINIT     = 50      # steps before rank pruning starts
ADALORA_TFINAL    = 1500    # freeze ranks after this step (500 steps to finish)
ADALORA_DELTAT    = 50      # re-evaluate rank budget every 50 steps

# Training — fits T4 (15 GB) within a 10-hour session
TRAIN_BATCH_SIZE  = 8
GRAD_ACCUM_STEPS  = 2       # effective batch = 16
LEARNING_RATE     = 1e-4
WARMUP_STEPS      = 200
MAX_STEPS         = 2_000
EVAL_STEPS        = 250     # checkpoint every 250 steps → survives disconnects
SAVE_STEPS        = 250
LOGGING_STEPS     = 25
EVAL_BATCH_SIZE   = 8
GENERATION_MAX_LEN = 225
NUM_WORKERS       = 2       # Colab: 2 vCPUs available

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

EARLY_STOPPING_PATIENCE = 6   # 6 × 250 = 1500 steps without WER improvement

REPORT_TO = ["tensorboard"]   # WandB disabled — no account/login required


# ── 2. CSV path auto-fix ──────────────────────────────────────────────────────
# CSVs carry absolute paths from the recording machine. Re-anchor every path
# that contains /hindi_sent/ to BASE_DIR so the script is location-agnostic.

def fix_paths(df: pd.DataFrame) -> pd.DataFrame:
    def _reanchor(p: str) -> str:
        m = re.search(r"(hindi_sent/.+)$", str(p))
        return str(BASE_DIR / m.group(1)) if m else p
    df = df.copy()
    df["path"] = df["path"].apply(_reanchor)
    return df


# ── 3. Load CSVs ──────────────────────────────────────────────────────────────

def load_split(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["transcript"].notna() & (df["transcript"].str.strip() != "")]
    df = df[df["path"].notna()]
    df = fix_paths(df)
    return df.reset_index(drop=True)


train_df = load_split(TRAIN_CSV)
val_df   = load_split(VAL_CSV)
test_df  = load_split(TEST_CSV)

print(f"Loaded  train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

# Spot-check that first audio path resolves
_first = Path(train_df["path"].iloc[0])
if not _first.exists():
    raise FileNotFoundError(
        f"Audio not found: {_first}\n"
        "Make sure HDSD/hindi_sent/ is under the same directory as this script."
    )
print(f"Audio path check OK: {_first}")


# ── 4. WhisperProcessor ───────────────────────────────────────────────────────

processor = WhisperProcessor.from_pretrained(MODEL_ID, language=LANGUAGE, task=TASK)
tokenizer = processor.tokenizer


# ── 5. Dataset ────────────────────────────────────────────────────────────────

class HDSDDataset(Dataset):
    """Loads WAV → optionally augments → returns Whisper log-mel features."""

    def __init__(self, df: pd.DataFrame, apply_augment: bool = False):
        self.records = df[["path", "transcript"]].values
        self.apply_augment = apply_augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        path, transcript = self.records[idx]
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SAMPLE_RATE:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        if self.apply_augment:
            audio = hdsd_augment(audio, SAMPLE_RATE)
        input_features = processor.feature_extractor(
            audio, sampling_rate=SAMPLE_RATE, return_tensors="np"
        ).input_features[0]                         # (80, 3000)
        labels = tokenizer(transcript.strip()).input_ids
        return {"input_features": input_features, "labels": labels}


train_dataset = HDSDDataset(train_df, apply_augment=True)
val_dataset   = HDSDDataset(val_df,   apply_augment=False)
test_dataset  = HDSDDataset(test_df,  apply_augment=False)


# ── 6. AdaLoRA Model ─────────────────────────────────────────────────────────

base_model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
base_model.config.use_cache = False
base_model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
    language=LANGUAGE, task=TASK
)
base_model.config.suppress_tokens = []

adalora_config = AdaLoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    init_r=ADALORA_INIT_R,
    target_r=ADALORA_TARGET_R,
    lora_alpha=ADALORA_ALPHA,
    lora_dropout=ADALORA_DROPOUT,
    target_modules=["q_proj", "v_proj"],
    tinit=ADALORA_TINIT,
    tfinal=ADALORA_TFINAL,
    deltaT=ADALORA_DELTAT,
    total_step=MAX_STEPS,       # must match MAX_STEPS; prevents KeyError in scheduler
    beta1=0.85,
    beta2=0.85,
    orth_reg_weight=0.5,
    modules_to_save=None,
)

model = get_peft_model(base_model, adalora_config)
model.enable_input_require_grads()   # required for grad-checkpointing + PEFT
model.print_trainable_parameters()


# ── 7. Data Collator ─────────────────────────────────────────────────────────

@dataclass
class WhisperDataCollator:
    """Pads labels; stacks fixed-size input_features into a batch tensor."""
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: list[dict]) -> dict:
        input_features = torch.tensor(
            np.stack([f["input_features"] for f in features]), dtype=torch.float32
        )  # (B, 80, 3000)
        label_ids = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_ids, return_tensors="pt", padding=True
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1), -100
        )
        if (labels[:, 0] == self.decoder_start_token_id).all():
            labels = labels[:, 1:]
        return {"input_features": input_features, "labels": labels}


data_collator = WhisperDataCollator(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)


# ── 8. WER Metric ────────────────────────────────────────────────────────────

def compute_metrics(pred) -> dict:
    pred_ids  = pred.predictions
    label_ids = pred.label_ids
    label_ids = np.where(label_ids == -100, tokenizer.pad_token_id, label_ids)
    pred_strs  = tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
    label_strs = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer_score = 100.0 * jiwer.wer(
        [s.strip() for s in label_strs],
        [s.strip() for s in pred_strs],
    )
    return {"wer": round(wer_score, 2)}


# ── 9. CSV Logger Callback ────────────────────────────────────────────────────

class CSVLoggerCallback(TrainerCallback):
    def __init__(self, output_dir: Path) -> None:
        self.path = Path(output_dir) / "training_log.csv"
        self._rows: list[dict] = []

    def on_log(
        self, args, state: TrainerState, control: TrainerControl,
        logs: dict | None = None, **kwargs,
    ) -> None:
        if not logs:
            return
        row: dict = {"step": state.global_step, "epoch": round(state.epoch or 0.0, 3)}
        row.update(logs)
        self._rows.append(row)
        pd.DataFrame(self._rows).to_csv(self.path, index=False)


# ── 10. AdaLoRA Rank-Update Callback ─────────────────────────────────────────
# update_and_allocate(step) triggers AdaLoRA's SVD-based rank pruning.
# Without it the model trains as fixed-rank LoRA at init_r the whole time.
# The broad except is intentional: at eval steps the internal budget table can
# be in an inconsistent state (KeyError / TypeError); skipping is non-fatal
# since ranks were already allocated at the previous training step.

class AdaLoraRankUpdateCallback(TrainerCallback):
    def on_step_end(
        self, args, state: TrainerState, control: TrainerControl,
        model=None, **kwargs,
    ) -> None:
        if model is not None and hasattr(model, "base_model"):
            try:
                model.base_model.update_and_allocate(state.global_step)
            except Exception:
                pass


# ── 11. HDSDSeq2SeqTrainer ────────────────────────────────────────────────────
# PEFT's PeftModelForSeq2SeqLM.forward always passes input_ids=None to the
# wrapped model.  Whisper's forward() does not accept input_ids (it uses
# input_features), so the kwarg collides → "multiple values for argument".
# Fix: call model.base_model.model(input_features=...) directly.
# AdaLoRA adapters are injected into q_proj/v_proj and remain active because
# the modified Linear layers live inside WhisperForConditionalGeneration.

class HDSDSeq2SeqTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model.base_model.model(
            input_features=inputs["input_features"],
            labels=inputs["labels"],
        )
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


# ── 12. Training Arguments ────────────────────────────────────────────────────

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_ckpts = sorted(
    [d for d in OUTPUT_DIR.glob("checkpoint-*") if d.is_dir()],
    key=lambda p: int(p.name.split("-")[-1]),
)
RESUME_FROM: str | None = str(_ckpts[-1]) if _ckpts else None
if RESUME_FROM:
    print(f"Resuming from checkpoint: {RESUME_FROM}")
else:
    print("Starting fresh run (no checkpoint found)")

training_args = Seq2SeqTrainingArguments(
    output_dir=str(OUTPUT_DIR),
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    max_steps=MAX_STEPS,
    fp16=True,
    gradient_checkpointing=True,
    eval_strategy="steps",          # fix: was evaluation_strategy (removed in 5.x)
    eval_steps=EVAL_STEPS,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=4,             # keep 4 checkpoints (1000 steps of history)
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    predict_with_generate=True,
    generation_max_length=GENERATION_MAX_LEN,
    logging_steps=LOGGING_STEPS,
    logging_dir=str(OUTPUT_DIR / "logs"),
    report_to=REPORT_TO,
    remove_unused_columns=False,
    dataloader_num_workers=NUM_WORKERS,
    seed=SEED,
    push_to_hub=False,
)


# ── 13. Pre-training Summary ──────────────────────────────────────────────────

_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
_total     = sum(p.numel() for p in model.parameters())

if torch.cuda.is_available():
    _gpu   = torch.cuda.get_device_name(0)
    _vram  = torch.cuda.get_device_properties(0).total_memory / 1024**3
    _gpustr = f"{_gpu} ({_vram:.1f} GB)"
else:
    _gpustr = "CPU (training will be very slow)"

_eff_bs   = TRAIN_BATCH_SIZE * GRAD_ACCUM_STEPS
_step_est = 6.0                         # conservative s/step on T4 for whisper-small
_eval_est = (MAX_STEPS // EVAL_STEPS) * 8   # 8 min per eval on T4
_total_h  = (_step_est * MAX_STEPS + _eval_est * 60) / 3600

print(f"\n{'='*62}")
print("  PRE-TRAINING SUMMARY")
print(f"{'='*62}")
print(f"  Trainable (AdaLoRA)  : {_trainable:>12,}  ({100*_trainable/_total:.2f}%)")
print(f"  Frozen (Whisper base): {_total-_trainable:>12,}  ({100*(1-_trainable/_total):.2f}%)")
print(f"  Effective batch size : {_eff_bs}  ({TRAIN_BATCH_SIZE}×{GRAD_ACCUM_STEPS} accum)")
print(f"  Max steps / eval int : {MAX_STEPS} / {EVAL_STEPS}")
print(f"  Est. wall time (T4)  : ~{_total_h:.1f} h")
print(f"  GPU                  : {_gpustr}")
print(f"  Resume from          : {RESUME_FROM or 'scratch (step 0)'}")
print(f"{'='*62}\n")


# ── 14. Build & Launch Trainer ────────────────────────────────────────────────
# processing_class= replaces tokenizer= in transformers ≥ 4.46.
# Use inspect to stay compatible with both 4.x and 5.x installs.

_callbacks = [
    EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE),
    CSVLoggerCallback(OUTPUT_DIR),
    AdaLoraRankUpdateCallback(),
]

_base_kwargs = dict(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=_callbacks,
)

_trainer_sig = inspect.signature(Seq2SeqTrainer.__init__).parameters
if "processing_class" in _trainer_sig:
    _base_kwargs["processing_class"] = processor.feature_extractor
else:
    _base_kwargs["tokenizer"] = processor.feature_extractor   # transformers < 4.46

trainer = HDSDSeq2SeqTrainer(**_base_kwargs)

print("── Starting training ───────────────────────────────────────────────")
trainer.train(resume_from_checkpoint=RESUME_FROM)


# ── 15. Save Adapter ──────────────────────────────────────────────────────────

adapter_dir = OUTPUT_DIR / "final_adapter"
model.save_pretrained(str(adapter_dir))
processor.save_pretrained(str(adapter_dir))
print(f"\nAdapter saved → {adapter_dir}")


# ── 16. Final Test-Set Evaluation ─────────────────────────────────────────────

print("\n── Evaluating on held-out test set ─────────────────────────────────")
test_results = trainer.predict(test_dataset)
test_wer = compute_metrics(test_results)["wer"]

print(f"\n{'='*62}")
print(f"  FINAL TEST WER : {test_wer:.2f}%")
print(f"  Target         : < 15.00%")
print(f"  Baseline (Whisper zero-shot on dysarthric Hindi): ~55–65%")
print(f"{'='*62}")

# Also write result to a file so it survives Colab session end
(OUTPUT_DIR / "test_wer.txt").write_text(f"test_wer={test_wer:.2f}%\n")
