# Whisper Fine-Tuning for Hindi Dysarthric Speech Recognition

Fine-tuning OpenAI's **Whisper-Small** model on the **Hindi Dysarthric Speech Dataset (HDSD)** to improve automatic speech recognition (ASR) for speakers with dysarthria.

---

## Overview

Automatic Speech Recognition systems are typically trained on speech from healthy speakers and often perform poorly on dysarthric speech due to articulation impairments. This project investigates whether fine-tuning Whisper-Small on dysarthric Hindi speech can significantly improve transcription accuracy.

The project includes:

- Data preprocessing and normalization
- Speaker-independent train/validation/test split
- Whisper-Small fine-tuning
- Quantitative evaluation using Word Error Rate (WER) and Character Error Rate (CER)
- Training history and qualitative prediction analysis

---

## Dataset

**Dataset:** Hindi Dysarthric Speech Dataset (HDSD)

Configuration used:

- Language: Hindi
- Microphone: M2 (close-talk microphone)
- Normalization:
  - `ei → e`
  - `ii → i`

Dataset split:

| Split | Samples |
|--------|---------:|
| Train | 1633 |
| Validation | 159 |
| Test | 210 |

A **speaker-independent split** was used to ensure speakers in the test set were never seen during training.

> **Note**
>
> The HDSD dataset is **not included** in this repository due to licensing and size constraints.

---

## Methodology

### 1. Data Preprocessing

- Extracted utterance-level audio
- Selected only M2 microphone recordings
- Cleaned and normalized transcripts
- Converted Romanized transcripts to Devanagari
- Generated speaker-independent train/validation/test split

---

### 2. Baseline Evaluation

The pretrained Whisper-Small model was evaluated without any fine-tuning to establish baseline performance.

Metrics:

- Word Error Rate (WER)
- Character Error Rate (CER)

---

### 3. Fine-Tuning

Model:

- `openai/whisper-small`

Training:

- Supervised fine-tuning
- Hugging Face Transformers Trainer
- Mixed Precision (FP16)
- Gradient Accumulation
- Evaluation every fixed number of steps
- Best checkpoint selected based on **lowest validation WER**

---

### 4. Evaluation

Performance was evaluated on the held-out speaker-independent test set using:

- Word Error Rate (WER)
- Character Error Rate (CER)

Qualitative predictions are available in:

```
results/test_predictions.csv
```

---

# Results

## Baseline

| Metric | Value |
|---------|-------:|
| WER | 1.3996 |
| CER | 1.0809 |

---

## Fine-Tuned Model

| Metric | Value |
|---------|-------:|
| WER | 0.0235 |
| CER | 0.0188 |

---

## Improvement

| Metric | Improvement |
|---------|------------:|
| WER | **98.32%** |
| CER | **98.26%** |

Best validation checkpoint:

```
checkpoint-1200
```

---

## Repository Contents

### Notebook 1

`01_HDSD_Preprocessing_Baseline.ipynb`

Contains:

- Dataset preprocessing
- Transcript normalization
- Speaker split generation
- Baseline Whisper evaluation

Outputs:

- baseline metrics
- speaker split
- baseline predictions

---

### Notebook 2

`02_HDSD_FineTuning.ipynb`

Contains:

- Dataset preparation
- Feature extraction
- Whisper fine-tuning
- Evaluation
- Training history
- Final predictions

Outputs:

- Fine-tuned model
- Training history
- Test predictions
- Final metrics

---

## Results Files

### `training_history.csv`

Contains:

- training loss
- validation loss
- validation WER
- validation CER
- learning rate
- epoch
- training step

---

### `test_predictions.csv`

Contains the reference transcript and model prediction for every test utterance.

---

### `hdsd_baseline_results_rawTransliterated.json`

Stores baseline evaluation metrics.

---

### `hdsd_finetuned_results.json`

Stores:

- baseline metrics
- fine-tuned metrics
- percentage improvement
- best checkpoint
- validation WER

---

## Installation

```bash
git clone https://github.com/<username>/hdsd-whisperFT.git

cd hdsd-whisperFT

pip install -r requirements.txt
```

---

## Running the Project

Run the notebooks in order:

1.

```
01_HDSD_Preprocessing_Baseline.ipynb
```

2.

```
02_HDSD_FineTuning.ipynb
```

---

## Hardware

Training configuration:

- GPU: NVIDIA GTX 1650 (4 GB VRAM)
- RAM: 32 GB
- PyTorch
- Hugging Face Transformers

---

## Future Work

- Fine-tuning Whisper-Medium
- LoRA / PEFT fine-tuning
- Noise augmentation
- Cross-dataset evaluation
- Dysarthria severity-wise analysis

---

## Acknowledgements

- OpenAI Whisper
- Hugging Face Transformers
- Hindi Dysarthric Speech Dataset (HDSD)

---

## License

This repository contains **code only**.

The HDSD dataset must be obtained separately according to its original license and distribution policy.
Link: https://www.iitr.ac.in/manojtripathy/Advertisement/database_info-MT.pdf
