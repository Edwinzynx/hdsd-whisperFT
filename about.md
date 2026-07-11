Let's go through exactly what happens during your training.

---

# Your dataset

```
Training samples      : 1633
Validation samples    : 159
Test samples          : 210
```

Each training sample contains

```
Audio (.wav)
        ↓
Whisper Feature Extractor
        ↓
80 × 3000 log-Mel spectrogram

Transcript
        ↓
Tokenizer
        ↓
Token IDs
```

So after preprocessing, each example looks like

```
{
    input_features : (80,3000)
    labels : [50258, 8485, 228, ...]
}
```

---

# DataLoader

You set

```python
per_device_train_batch_size = 1
```

So the dataloader gives the model

```
Batch 1

Example 1
```

then

```
Batch 2

Example 2
```

then

```
Batch 3

Example 3
```

Only **one audio sample** is on the GPU at any time.

This is necessary because your GTX1650 has only **4GB VRAM**.

---

# Data Collator

Suppose two transcripts are

```
Example 1

आज मौसम अच्छा है

↓

[50258, 4211, 3456, 1234]
```

```
Example 2

आज अच्छा

↓

[50258, 4211, 1234]
```

The collator pads them

```
Example 1

4211
3456
1234

Example 2

4211
1234
-100
```

Notice

```
-100
```

These are ignored while computing the loss.

---

# Forward pass

For one batch

```
Audio
      ↓

Whisper Encoder

      ↓

Encoder Hidden States

      ↓

Whisper Decoder

      ↓

Predicted Tokens
```

The decoder predicts one token at a time.

Suppose

Reference

```
आपके
हिंदी
पसन्द
...
```

Prediction

```
आपके
हिंदी
करने
...
```

Cross-entropy loss is computed between prediction and reference.

---

# Backpropagation

Normally,

```
Forward

↓

Loss

↓

Backward

↓

Optimizer Step
```

Since you have

```python
gradient_accumulation_steps = 8
```

this changes.

---

# Gradient accumulation

Instead of updating after every sample,

Step 1

```
Example 1

↓

Loss

↓

Gradient

↓

Store gradient
```

No optimizer step.

---

Step 2

```
Example 2

↓

Gradient

↓

Add to previous gradient
```

Still no optimizer step.

---

This continues

```
Example 3
Example 4
Example 5
Example 6
Example 7
Example 8
```

Now the accumulated gradient represents

```
8 training examples
```

Only then

```
optimizer.step()

optimizer.zero_grad()
```

is executed.

---

So your effective batch is

```
GPU batch

1 sample
```

but

```
Optimizer batch

8 samples
```

Equivalent to

```
Batch size = 8
```

without requiring enough GPU memory to hold 8 examples simultaneously.

---

# How many optimizer updates?

You have

```
1633 samples
```

Each optimizer update uses

```
8 samples
```

Therefore

```
1633 / 8

≈ 204 updates
```

per epoch.

---

# Epoch

One epoch means

```
Model sees

all 1633 audio files

exactly once
```

Then

```
Validation

↓

WER

↓

CER

↓

Checkpoint
```

because you configured

```python
evaluation_strategy="steps"
```

(or `"epoch"` if you use that configuration).

---

# Validation

Training is **disabled**.

No gradients.

For each validation sample

```
Audio

↓

Whisper generate()

↓

Prediction

↓

Compare with reference
```

Then

```
WER

CER
```

are computed.

---

# Best model

Suppose

```
Epoch 1

WER = 0.62
```

Saved.

---

```
Epoch 2

WER = 0.55
```

Better.

Checkpoint becomes best.

---

```
Epoch 3

WER = 0.58
```

Worse.

Checkpoint isn't selected.

---

At the end,

```python
load_best_model_at_end=True
```

automatically restores

```
Epoch 2 weights
```

rather than the final epoch if they performed best.

---

# What exactly is being updated?

Because you're doing **full fine-tuning**:

```
Whisper Encoder
✓ updated

Whisper Decoder
✓ updated

Attention weights
✓ updated

Feed-forward layers
✓ updated

Embeddings
✓ updated

Output projection
✓ updated
```

Every trainable parameter in Whisper-small learns from your HDSD dataset.

---

# Overall training pipeline

```text
HDSD Training Set (1633 samples)
            │
            ▼
Load one audio sample
            │
            ▼
Whisper Feature Extractor
            │
            ▼
Input Features (80 × 3000)
            │
            ▼
Whisper Encoder
            │
            ▼
Encoder Hidden States
            │
            ▼
Whisper Decoder
            │
            ▼
Predicted Token IDs
            │
            ▼
Cross-Entropy Loss
            │
            ▼
Backpropagation
            │
            ▼
Accumulate Gradients (8 samples)
            │
            ▼
Optimizer Step
            │
            ▼
Repeat until all 1633 samples are processed
            │
            ▼
Validation on 159 samples
            │
            ▼
Compute WER and CER
            │
            ▼
Save Checkpoint
            │
            ▼
Repeat for all epochs
            │
            ▼
Load the checkpoint with the lowest validation WER
```

This is the complete training workflow your notebook implements.
