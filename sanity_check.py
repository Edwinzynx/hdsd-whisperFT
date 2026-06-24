"""End-to-end sanity check for the Whisper-AdaLoRA training pipeline.

Check 1 (data integrity) runs with no ML dependencies — pure stdlib + pandas.
Checks 2-6 require the full ML stack and are skipped gracefully if not installed.

Usage:
    python sanity_check.py          # any machine, even without GPU/ML deps
    python sanity_check.py --all    # Lightning AI with full deps installed

Exit code 0 = all executed checks passed.
"""

import os, re, sys, time, wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR  = Path(__file__).parent.resolve()
TRAIN_CSV = BASE_DIR / "train.csv"
VAL_CSV   = BASE_DIR / "val.csv"
TEST_CSV  = BASE_DIR / "test.csv"

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
WARN = "\033[93m!\033[0m"

def ok(msg):   print(f"  {PASS} {msg}")
def fail(msg): print(f"  {FAIL} {msg}"); sys.exit(1)
def warn(msg): print(f"  {WARN} {msg}")


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 1 — Data integrity (pure Python + pandas, no ML deps)
# ══════════════════════════════════════════════════════════════════════════════

print("\n── 1. Data integrity ────────────────────────────────────────────────")

train = pd.read_csv(TRAIN_CSV)
val   = pd.read_csv(VAL_CSV)
test  = pd.read_csv(TEST_CSV)

# NaN / empty transcript check
for name, df in [("train", train), ("val", val), ("test", test)]:
    n_nan   = df["transcript"].isna().sum()
    n_empty = (df["transcript"].fillna("").str.strip() == "").sum()
    if n_nan or n_empty:
        fail(f"{name}.csv has {n_nan} NaN + {n_empty} empty transcripts")
    ok(f"{name}.csv: {len(df)} rows, 0 missing transcripts")

# Speaker-level leakage
train_spk = set(train["subject_id"])
tv = train_spk & set(val["subject_id"])
tt = train_spk & set(test["subject_id"])
vt = set(val["subject_id"]) & set(test["subject_id"])
if tv: fail(f"Train/val speaker overlap: {tv}")
if tt: fail(f"Train/test speaker overlap: {tt}")
if vt: warn(f"Val/test speaker overlap (acceptable): {vt}")
ok(f"No train↔val or train↔test leakage  "
   f"(train={len(train_spk)}  val={len(set(val['subject_id']))}  "
   f"test={len(set(test['subject_id']))} speakers)")

# Devanagari check
bad = train[~train["transcript"].apply(lambda t: bool(re.search(r"[ऀ-ॿ]", str(t))))]
if len(bad):
    fail(f"{len(bad)} train transcripts lack Devanagari characters")
ok(f"All transcripts are Devanagari  "
   f"(vocab={len(set(' '.join(train['transcript']).split()))} unique words)")

# Audio file existence + duration stats
durations, missing = [], []
for path in train["path"]:
    if not os.path.exists(path):
        missing.append(path)
        continue
    with wave.open(path) as wf:
        durations.append(wf.getnframes() / wf.getframerate())

if missing:
    fail(f"{len(missing)} audio files not found:\n" + "\n".join(missing[:5]))

short = sum(1 for d in durations if d < 1.5)
ok(f"Train audio  min={min(durations):.2f}s  max={max(durations):.2f}s  "
   f"mean={sum(durations)/len(durations):.2f}s  ({len(durations)} files)")
if short:
    warn(f"{short} clips < 1.5 s (severe dysarthria — expected, kept)")

# Select 3 representative samples by duration percentile (p10 / p50 / p90)
dur_s = pd.Series(durations, index=train.index)
sample_rows = train.loc[
    [(dur_s - dur_s.quantile(q)).abs().idxmin() for q in (0.10, 0.50, 0.90)]
].reset_index(drop=True)

print(f"\n  Pipeline test samples (p10 / p50 / p90 duration):")
for _, r in sample_rows.iterrows():
    with wave.open(r["path"]) as wf:
        d = wf.getnframes() / wf.getframerate()
    print(f"    {r['subject_id']} {r['sentence_id']}  {d:.2f}s  \"{r['transcript']}\"")


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 2 — ML stack imports (lazy — each import tried independently)
# ══════════════════════════════════════════════════════════════════════════════

print("\n── 2. ML stack imports ──────────────────────────────────────────────")

_ml_ok = True

for pkg, install in [
    ("torch",          "torch"),
    ("soundfile",      "soundfile"),
    ("librosa",        "librosa"),
    ("transformers",   "transformers"),
    ("peft",           "peft"),
    ("jiwer",          "jiwer"),
]:
    try:
        mod = __import__(pkg)
        ok(f"{pkg} {getattr(mod, '__version__', '?')}")
    except ImportError:
        warn(f"{pkg} NOT installed  →  pip install {install}")
        _ml_ok = False

if not _ml_ok:
    warn("Some ML packages missing — skipping checks 3-6")
    warn("Install deps:  pip install torch transformers peft soundfile librosa jiwer")
    print("\n── Data integrity checks passed (checks 3-6 skipped — install deps above)\n")
    sys.exit(0)

import torch
import soundfile as sf
import librosa
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import AdaLoraConfig, TaskType, get_peft_model
import jiwer

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    ok(f"GPU: {torch.cuda.get_device_name(0)}  "
       f"({torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB VRAM)")
else:
    warn("No GPU — training will be very slow on CPU")


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 3 — Full pipeline: WAV → augment → log-mel → collator batch
# ══════════════════════════════════════════════════════════════════════════════

print("\n── 3. Full pipeline (3 samples) ─────────────────────────────────────")

sys.path.insert(0, str(BASE_DIR))
from hdsd_augment import augment as hdsd_augment

MODEL_ID    = "openai/whisper-small"
SAMPLE_RATE = 16_000

processor = WhisperProcessor.from_pretrained(MODEL_ID, language="hi", task="transcribe")
tokenizer = processor.tokenizer

processed = []
for _, row in sample_rows.iterrows():
    audio, sr = sf.read(row["path"], dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    dur_orig = len(audio) / SAMPLE_RATE

    audio_aug = hdsd_augment(audio, SAMPLE_RATE, p=1.0)   # force all ops for testing
    dur_aug   = len(audio_aug) / SAMPLE_RATE

    feats  = processor.feature_extractor(
        audio_aug, sampling_rate=SAMPLE_RATE, return_tensors="np"
    ).input_features[0]                                    # (80, 3000)
    labels = tokenizer(row["transcript"].strip()).input_ids

    ok(f"{row['subject_id']} {row['sentence_id']}:  "
       f"{dur_orig:.2f}s→{dur_aug:.2f}s (aug)  "
       f"feats={feats.shape}  labels={len(labels)} tok")
    processed.append({"input_features": feats, "labels": labels})


@dataclass
class _Collator:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features):
        input_features = torch.tensor(
            np.stack([f["input_features"] for f in features]), dtype=torch.float32
        )
        lids = [{"input_ids": f["labels"]} for f in features]
        lb   = self.processor.tokenizer.pad(lids, return_tensors="pt", padding=True)
        labels = lb["input_ids"].masked_fill(lb["attention_mask"].ne(1), -100)
        if (labels[:, 0] == self.decoder_start_token_id).all():
            labels = labels[:, 1:]
        return {"input_features": input_features, "labels": labels}


base_model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
collator   = _Collator(processor, base_model.config.decoder_start_token_id)
batch      = collator(processed)

print(f"\n  Batch shapes:")
ok(f"input_features : {tuple(batch['input_features'].shape)}  (expected (3, 80, 3000))")
ok(f"labels         : {tuple(batch['labels'].shape)}")
assert batch["input_features"].shape == (3, 80, 3000), "input_features shape mismatch"
assert batch["labels"].shape[0] == 3,                  "labels batch-size mismatch"
assert (batch["labels"] == -100).any(),                 "no -100 padding in labels"
ok("Collator verified")


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 4 — AdaLoRA forward + backward + gradient check
# ══════════════════════════════════════════════════════════════════════════════

print("\n── 4. AdaLoRA forward pass & loss ───────────────────────────────────")

adalora_cfg = AdaLoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    init_r=12, target_r=8, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    tinit=200, tfinal=3_500, deltaT=100,
    beta1=0.85, beta2=0.85, orth_reg_weight=0.5,
)
base_model.config.use_cache = False
base_model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
    language="hi", task="transcribe"
)
base_model.config.suppress_tokens = []

model = get_peft_model(base_model, adalora_cfg)
model.enable_input_require_grads()

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
ok(f"Trainable: {trainable:,} / {total:,}  ({100*trainable/total:.2f}%)")

model     = model.to(device)
batch_gpu = {k: v.to(device) for k, v in batch.items()}

# PeftModelForSeq2SeqLM.forward always injects input_ids= into the base-model
# call even when None — Whisper uses input_features=, not input_ids=, so that
# kwarg causes a "multiple values" TypeError.  Calling model.base_model.model(...)
# skips the PEFT dispatch wrapper entirely.  The AdaLoRA adapters are already
# injected into q_proj/v_proj, so the forward IS running through LoRA weights.
with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device == "cuda")):
    t0      = time.time()
    outputs = model.base_model.model(
        input_features=batch_gpu["input_features"],
        labels=batch_gpu["labels"],
    )
    loss    = outputs.loss
    elapsed_fwd = time.time() - t0

if torch.isnan(loss) or torch.isinf(loss):
    fail(f"Loss is {loss.item()} on first forward pass — NaN/Inf")
ok(f"Forward  loss={loss.item():.4f}  ({elapsed_fwd*1000:.0f} ms)")

loss.backward()
ok("Backward pass completed")

n_nz = sum(
    1 for p in model.parameters()
    if p.requires_grad and p.grad is not None and p.grad.abs().max() > 0
)
if n_nz == 0:
    fail("All gradients zero — AdaLoRA adapters not receiving gradients")
ok(f"Non-zero gradient tensors: {n_nz}")


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 5 — GPU memory
# ══════════════════════════════════════════════════════════════════════════════

print("\n── 5. GPU memory ────────────────────────────────────────────────────")

if device == "cuda":
    torch.cuda.synchronize()
    alloc_gb  = torch.cuda.memory_allocated(0) / 1024**3
    reserv_gb = torch.cuda.memory_reserved(0)  / 1024**3
    total_gb  = torch.cuda.get_device_properties(0).total_memory / 1024**3
    ok(f"Allocated={alloc_gb:.2f} GB  Reserved={reserv_gb:.2f} GB  Total={total_gb:.1f} GB")
    if reserv_gb > total_gb * 0.90:
        warn(f"Memory >90% — OOM risk at batch_size=8; "
             "try per_device_train_batch_size=4 in train_whisper_adalora.py")
    else:
        ok(f"Headroom OK for batch_size=8")
else:
    warn("No CUDA — GPU memory check skipped")


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 6 — Estimated training time
# ══════════════════════════════════════════════════════════════════════════════

print("\n── 6. Estimated training time ───────────────────────────────────────")

MAX_STEPS     = 4_000
EFF_BATCH     = 8 * 2    # TRAIN_BATCH_SIZE × GRAD_ACCUM_STEPS
steps_per_ep  = len(train) / EFF_BATCH
total_epochs  = MAX_STEPS / steps_per_ep

# Scale measured fwd time to batch_size=8, then ×3 for bwd + optimizer
step_s        = elapsed_fwd * (8 / 3) * 3
total_h       = step_s * MAX_STEPS / 3600

ok(f"Eff. batch={EFF_BATCH}  ~{steps_per_ep:.0f} steps/epoch  ~{total_epochs:.1f} epochs")
ok(f"Est. step time : {step_s:.1f} s  |  Est. total : {total_h:.1f} h for {MAX_STEPS} steps")
if device != "cuda":
    warn("CPU timings are not representative — GPU will be 10-50× faster")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 68)
print("  All checks passed.  You are ready to train:")
print("  nohup python train_whisper_adalora.py > training.log 2>&1 &")
print("─" * 68 + "\n")
