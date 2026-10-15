# Hindi dysarthric speech recognition with Whisper

The current project compares pretrained Whisper-small, full fine-tuning, and LoRA on Hindi dysarthric speech. The three self-contained notebooks used on university Jupyter are in **[hdsd_simple/](hdsd_simple/README.md)**.

| Notebook | Purpose |
|---|---|
| [01 — Preprocessing and baseline](hdsd_simple/01_HDSD_Preprocessing_Baseline.ipynb) | Prepare the supplied annotations and fixed partitions; evaluate pretrained Whisper. |
| [02 — Full fine-tuning](hdsd_simple/02_HDSD_Full_FineTuning.ipynb) | Train all weights, select a learning rate on development data, and repeat across three seeds. |
| [03 — LoRA fine-tuning](hdsd_simple/03_HDSD_PEFT_FineTuning.ipynb) | Train attention adapters using the same selection and evaluation procedure. |

Start with the [setup and notebook instructions](hdsd_simple/README.md). The code is visible inside each notebook. Published copies have saved outputs cleared; the executed originals remain on the private cluster. This export adds no dataset files, transcripts, per-recording predictions, credentials, or model checkpoints. Supply an authorized local HDSD copy to run preprocessing and training.

## Completed evaluation

Both learning-rate searches and seeds 42/123/2026 are complete. Final evaluation covers 447 M2 recordings per model instance. Adapted WER values below are mean ± sample standard deviation across the three training seeds; lower is better.

| Evaluation case | Pretrained WER (%) | Full WER (%) | LoRA WER (%) |
|---|---:|---:|---:|
| Both speaker and prompt unfamiliar | 71.51 | 63.04 ± 6.97 | 63.31 ± 3.18 |
| Unfamiliar speaker, familiar prompt | 93.03 | 17.65 ± 4.32 | 44.79 ± 12.63 |
| Familiar speaker, unfamiliar prompt | 75.17 | 65.53 ± 3.31 | 62.78 ± 1.74 |

These are exploratory results: the legacy holdout was inspected in earlier work. The primary paired speaker interval spans both directions, so the results do not establish a clear full-versus-LoRA winner or equivalence. No augmentation is included.

Read the [results and limitations](hdsd_simple/RESULTS.md), [frozen evaluation procedure](hdsd_simple/EVALUATION_PROTOCOL.md), and [aggregate CSV tables](hdsd_simple/results/). The private frozen JSON contains references and is excluded; a fresh clone cannot replay the exact archived evaluation without the private inputs described in the setup guide.

## Earlier implementation

The root-level `01_HDSD_Preprocessing_Baseline.ipynb`, `02_HDSD_FineTuning.ipynb`, other scripts, and earlier result files are retained as historical work. Use `hdsd_simple/` for the current three-notebook study. Earlier scores used different preprocessing/evaluation and should not be mixed with this comparison. Earlier commits already tracked some manifest/transcript CSVs; adding ignore rules does not remove those files or their history.

HDSD is subject to the supplied academic-use and non-redistribution agreement. Keep recordings, annotations, generated manifests, individual predictions, and trained weights private. The current public export provides code, documentation, hashes, and aggregate numerical evidence.
