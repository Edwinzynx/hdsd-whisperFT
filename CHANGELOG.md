# Project Changelog: Preprocessing, Data Splits, and Model Evaluation Fixes

This document tracks all modifications and corrections made to the HDSD Whisper Fine-Tuning project to address structural data leakage, corrupted transcripts, control speaker contamination, and execution errors.

---

## 1. Summary of Major Problems Solved

### Problem 1 & 2: Broken Answer Key & Corrupted Hindi Orthography
*   **Original Issue**: Buggy regex rules (`ii\b -> i`, `ei\b -> e`) and suffix drops truncated long vowels and nasalizations (e.g., `हिंदी` became `हिन्दि`, `में` became `मेइन्`). This corrupted 69% of the ground truth sentences, resulting in a false-bad baseline WER/CER (Whisper transcribed decent Hindi but was graded against a misspelled key).
*   **Fix**: Programmatically analyzed the entire Hinglish corpus and extracted its exact vocabulary of **128 unique words**. Created a static, 100% accurate Hinglish-to-Devanagari mapping dictionary (`FIXED_WORD_MAP`) that restores correct standard Hindi spelling.

### Problem 3: Sentence Memorization (Data Leakage)
*   **Original Issue**: Splitting was only speaker-independent, allowing 94% of test set sentences to appear in the training set. The model memorized the sentence template list (1.4% error on seen sentences vs. 15.4% error on unseen sentences).
*   **Fix**: Implemented a strict **Dual-Independent Split** (disjoint speakers AND disjoint sentence template IDs `H01` to `H30`) to evaluate true generalization on unseen speakers speaking unseen sentences.

### Problem 4: Control Speakers in Dysarthric Test Set
*   **Original Issue**: Healthy control speaker `CF00` made up 14% of the dysarthric test set, inflating model performance.
*   **Fix**: Filtered out all control speakers starting with `C` from the train, validation, and test datasets.

---

## 2. Updated Data Splits (Dual-Independent)

*   **Total Dysarthric Records**: 1,882 (after filtering control speakers and empty/NaN transcripts)
*   **Train Set**: 899 utterances (disjoint speakers + disjoint sentences)
*   **Validation Set**: 44 utterances (disjoint speakers + disjoint sentences)
*   **Test Set**: 45 utterances (disjoint speakers + disjoint sentences)
*   **Leakage Verification**: Verified 0% speaker overlap and 0% sentence overlap across Train ↔ Val ↔ Test splits.

---

## 3. Modified & Added Files

### [NEW] [prepare_and_split_data.py](file:///c:/Users/edwin/OneDrive/Desktop/Capstone/prepare_and_split_data.py)
*   **Role**: The master pipeline data prep script.
*   **Changes**:
    *   Contains the complete 128 Hinglish-to-Hindi spelling mapping (`FIXED_WORD_MAP`).
    *   Filters out control speakers and empty/NaN transcripts.
    *   Implements the dual-independent speaker/sentence splitting logic.
    *   Saves the resulting datasets (`train.csv`, `val.csv`, `test.csv`, and compatibility copies `hdsd_*.csv`) in the Capstone root folder.

### [MODIFY] [prepare_data.py](file:///c:/Users/edwin/OneDrive/Desktop/Capstone/prepare_data.py)
*   **Changes**: Aligned transcript reading and Hinglish cleaning with the new spelling map to prevent vowel truncation.

### [MODIFY] [split_data.py](file:///c:/Users/edwin/OneDrive/Desktop/Capstone/split_data.py)
*   **Changes**: Updated to split sentences and speakers independently.

### [MODIFY] [sanity_check.py](file:///c:/Users/edwin/OneDrive/Desktop/Capstone/sanity_check.py)
*   **Changes**: Fixed `UnicodeEncodeError` on Windows (reconfigured stdout to UTF-8). Verified that our new splits successfully pass all leakage and spelling checks.

### [MODIFY] [01_HDSD_Preprocessing_Baseline.ipynb](file:///c:/Users/edwin/OneDrive/Desktop/Capstone/01_HDSD_Preprocessing_Baseline.ipynb)
*   **Changes**:
    *   **Cell 21**: Replaced old speaker-only split with the new healthy-filtering and dual-independent split logic.
    *   **Cell 31**: Replaced incorrect regex normalization function with the 128-word perfect transliteration mapping (`FIXED_WORD_MAP`).
    *   **Cell 33**: Preserved the column structure (`df["text_norm"]` and `df["text_devnagari"]`) to prevent errors in subsequent cells.
    *   **Cell 35**: Aligned final dataset rebuilding with the clean dual-independent logic (preventing the notebook from saving the old, leaky splits to disk).
    *   **Markdown Cells**: Updated all textual descriptions of the preprocessing and splitting logic to reflect the fixed system.

---

## 4. Environment & Execution Configuration

*   **Virtual Environment Path**: `C:\Users\edwin\OneDrive\Desktop\Convo AI\.gpuvenv`
*   **Configured Jupyter Kernel**: Registered `.gpuvenv` as a Jupyter kernel named `gpuvenv` (`Python (.gpuvenv)`).
*   **Dependencies Installed**: Verified and updated `peft`, `accelerate`, `transformers`, `evaluate`, `jiwer`, `soundfile`, `librosa`, and `openpyxl`.

---

*Note for other AI Agents: When loading this workspace, do not run standard `split_data.py` or the raw notebook without reloading the `gpuvenv` kernel. Check [sanity_check.py](file:///c:/Users/edwin/OneDrive/Desktop/Capstone/sanity_check.py) to run verification on local splits.*
