# HDSD: the three current Jupyter notebooks

This folder contains the current university Jupyter workflow: preprocessing and a pretrained baseline, full Whisper fine-tuning, and LoRA fine-tuning. Every preparation, training, generation, and scoring step is visible in the notebooks; no project helper module is required.

The files were copied from the live Jupyter project on 2026-10-15. Code and explanatory cells are unchanged. Saved outputs, execution counts, attachments, and execution metadata were removed from these public copies. The executed originals remain on the private cluster. [release.json](release.json) records file hashes and scientific code identities.

## Start here

| Notebook | What it does |
|---|---|
| [01 — Preprocessing and baseline](01_HDSD_Preprocessing_Baseline.ipynb) | Reads the supplied WAV/WORD files, converts the Roman annotations to Hindi, reproduces the fixed partitions, and evaluates pretrained Whisper-small. |
| [02 — Full fine-tuning](02_HDSD_Full_FineTuning.ipynb) | Updates all model weights, compares learning rates on development data, and repeats the selected configuration with three seeds. |
| [03 — LoRA fine-tuning](03_HDSD_PEFT_FineTuning.ipynb) | Freezes the pretrained weights and trains small attention adapters using the same data and selection procedure. |

All three also contain the completed study's final evaluation code. Read the [aggregate results](RESULTS.md) and [evaluation protocol](EVALUATION_PROTOCOL.md) to understand what the experiments support.

## Folder layout

```text
hdsd_simple/
├── 01_HDSD_Preprocessing_Baseline.ipynb
├── 02_HDSD_Full_FineTuning.ipynb
├── 03_HDSD_PEFT_FineTuning.ipynb
├── requirements.txt
├── README.md
├── EVALUATION_PROTOCOL.md
├── RESULTS.md
├── release.json
└── results/                 four aggregate score/resource tables
```

An authorized user supplies `HDSD/` privately. Running the notebooks creates `prepared/` for manifests and `runs/` for predictions, training state, checkpoints, and reports. These directories are ignored by Git and are not included in this export.

## Existing university Jupyter project

Open the existing `hdsd_simple` directory and select **HDSD Research**. Both learning-rate searches, all three selected seeds per method, and final evaluation are already complete. There is no need to retrain to read the results. Use one GPU notebook at a time.

The public notebooks preserve the live settings: `EXPERIMENT = "repeat"` in 02/03, and `RUN_FINAL_EVALUATION = False` in all three. Completed compatible experiments are checked and reused. Do not replace private `prepared/` or `runs/` files with empty directories when updating notebook code.

## Set up a separate machine

1. Clone this repository and enter `hdsd_simple/`.
2. Put your legally obtained HDSD release in `hdsd_simple/HDSD/`, or create a local symlink to it. The notebook expects `HDSD/hindi_sent/` and `HDSD/WORD/`. Both microphone files are checked during preprocessing, although the models use M2 only. Do not upload the dataset.
3. Create an isolated Python environment and register its Jupyter kernel. The study used Python 3.10.12, PyTorch 2.7.1+cu118, and an A100 MIG 3g.40gb allocation (approximately 39.25 GiB). The supplied configuration requires CUDA and BF16 support.

```bash
git clone https://github.com/Edwinzynx/hdsd-whisperFT.git
cd hdsd-whisperFT/hdsd_simple
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
python -m ipykernel install --user --name hdsd-research --display-name "HDSD Research"
```

Use an existing Jupyter server to open this directory and select that kernel. The requirements provide the kernel and model dependencies; they do not install a separate Jupyter server. The first model load needs access to Hugging Face to download the pinned Whisper-small revision.

4. Run notebook 01 from top to bottom, leaving final evaluation disabled. It creates the private manifest and development baseline. The exact supplied release produces 1,882 paired utterances, including 899 training and 44 primary development utterances.
5. In notebook 02, change `EXPERIMENT` to `"search"` before running. This compares three learning rates using primary development WER. Then change it to `"repeat"` and rerun to evaluate the selected configuration across seeds 42, 123, and 2026. Compatible finished runs are reused. `"pilot"` is available for an initial single run but is not required before the search.
6. Follow the same search-then-repeat sequence in notebook 03. Keep the shared settings fixed for this comparison. Full fine-tuning searches 5e-6/1e-5/2e-5; LoRA searches 5e-5/1e-4/2e-4 with rank 16, alpha 32, and dropout 0.05.

The search selects a learning rate on development data; it does not choose the best final-evaluation score. Each seed selects its checkpoint by minimum primary development WER. Equal scores keep the earlier epoch. Training saves resumable state at epoch boundaries; retain `best/`, `last/`, and `state.pt` privately.

## Final evaluation and reproducibility limits

The appended final sections use the separately frozen private `runs/final_v3/protocol.json` and `protocol.sha256`, along with exact selected checkpoints, saved development predictions, and the recorded environment. The JSON includes utterance references, so it is deliberately excluded from GitHub. The human-readable [protocol](EVALUATION_PROTOCOL.md) is published byte-for-byte because its hash is part of that frozen record.

The original cluster retains those inputs and can verify/reuse its completed evaluation. A fresh clone contains the evaluation implementation and aggregate evidence, but cannot replay the exact archived evaluation without those private artifacts. New training also produces new checkpoint hashes. Freeze a separate evaluation record for a new reproduction before generating holdout predictions; do not disable the identity checks or overwrite the archived protocol to make it fit. Keep `RUN_FINAL_EVALUATION = False` until the required private inputs exist.

For evaluation only on the original project, restart the kernel and run the earlier **Locate the project**, **Read/write files**, **Code identity**, **Read the frozen manifest**, and **Calculate word and character errors** cells. Then run the final section with `RUN_FINAL_EVALUATION = True`. It performs inference or verifies compatible cached predictions; no training cell is required.

## Data and interpretation

The analyzed pool has 1,882 paired utterances, 59 speakers, 30 prompts, and about 95.4 minutes per microphone. M1 is throat and M2 is close-talk. Adaptation uses 899 M2 recordings from 41 speakers and 21 prompts, about 45 minutes of audio. Paired microphones and repeated sessions are not additional independent speakers.

References use the supplied Roman WORD annotations and the explicit Hindi mapping in notebook 01, retaining annotated repetitions and omissions. The supplied labels were accepted for this study; no new listening verification is claimed. References and predictions receive identical NFC, punctuation, and whitespace normalization. WER counts word substitutions, deletions, and insertions; CER counts Unicode code points including spaces.

These are **exploratory fixed-holdout results** because the legacy test was inspected in earlier work. Report all three evaluation cases and all seeds. The primary comparison does not establish a clear full-versus-LoRA winner or equivalence. Augmentation is deferred. The study does not infer clinical severity from recognition errors.

HDSD remains subject to its supplied academic-use and non-redistribution agreement. This export contains source code, documentation, hashes, and aggregate numerical tables. Recordings, utterance annotations, manifests, per-recording predictions, model weights, credentials, and the private frozen JSON are excluded. The word conversion dictionary and pseudonymous partition IDs remain in the preprocessing code to specify the implemented method.
