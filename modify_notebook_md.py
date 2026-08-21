import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))
notebook_path = os.path.join(BASE, "01_HDSD_Preprocessing_Baseline.ipynb")

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# New Markdown content
split_md_source = [
    "## Dual Speaker-and-Sentence-Independent Train / Validation / Test Split\n",
    "\n",
    "To obtain a realistic estimate of the model's performance and prevent overfitting, the dataset is split **both speaker-wise and sentence-wise** (dual-independent split):\n",
    "\n",
    "1. **Speaker Independence**: If utterances from the same speaker appear in both the training and testing sets, the model may memorize speaker-specific characteristics instead of learning to generalize to unseen speakers.\n",
    "2. **Sentence Independence**: Since the corpus uses a small pool of 30 sentence templates, if the same sentences are seen during training, the model can simply memorize the training sentence list rather than learning acoustic speech representation.\n",
    "\n",
    "Therefore, all control speakers (healthy speakers starting with 'C') are filtered out, and the remaining dysarthric speakers and sentences are split into disjoint sets (70% train, 15% val, 15% test). This ensures evaluation on completely unseen speakers speaking completely unseen sentences."
]

normalization_md_source = [
    "The data shows HDSD is using a phonetic Romanization scheme, not random spelling.\n",
    "\n",
    "### Corrected Dictionary-Based Transliteration Mapping\n",
    "\n",
    "To prevent vowels and nasalizations from being chopped off (e.g., preventing 'हिंदी' from becoming 'हिन्दि' and 'में' from becoming 'मेइन्'), we do not use heuristic rules like `ii->i` or `ei->e`. Instead, since the Hinglish vocabulary of the entire corpus consists of exactly 128 unique words, we define a complete 128-word dictionary mapping `FIXED_WORD_MAP` to map every Hinglish word to its standard Devanagari Hindi spelling with 100% accuracy."
]

problem_md_source = [
    "\n",
    "PROBLEM: The transcript is noisy. Heuristic normalizations (like chopping off vowels) result in corrupted Hindi references (e.g., 'हिन्दि' instead of 'हिंदी').\n",
    "\n",
    "SOLUTION: We build a perfect 128-word dictionary-based transliteration mapping layer that covers the entire corpus vocabulary. This ensures that the Whisper model is graded against 100% correct, standard Hindi spelling."
]

modified_cells = 0
for cell in nb["cells"]:
    cell_id = cell.get("id")
    if cell_id == "47eaa18a6b601bcf":
        cell["source"] = split_md_source
        modified_cells += 1
    elif cell_id == "e5d2e1f2da3520e0":
        cell["source"] = normalization_md_source
        modified_cells += 1
    elif cell_id == "31ea64b6fe64a417":
        cell["source"] = problem_md_source
        modified_cells += 1

print(f"Modified {modified_cells} markdown cells.")

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Saved notebook successfully.")
