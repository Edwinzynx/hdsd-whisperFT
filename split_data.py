import os
import pandas as pd
from sklearn.model_selection import train_test_split

BASE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE, "hdsd_manifest.csv"))

# Filter out controls completely
dysarthric = df[df["is_control"] == False].copy()
print(f"Dysarthric utterances: {len(dysarthric)}, speakers: {dysarthric['subject_id'].nunique()}")

# Sentence-independent split based on sentence_id
unique_sentences = sorted(dysarthric["sentence_id"].unique())
print(f"Unique sentence IDs: {len(unique_sentences)}")

train_sentences, temp_sentences = train_test_split(
    unique_sentences,
    test_size=0.30,
    random_state=42
)
val_sentences, test_sentences = train_test_split(
    temp_sentences,
    test_size=0.50,
    random_state=42
)

train_df = dysarthric[dysarthric["sentence_id"].isin(train_sentences)].reset_index(drop=True)
val_df   = dysarthric[dysarthric["sentence_id"].isin(val_sentences)].reset_index(drop=True)
test_df  = dysarthric[dysarthric["sentence_id"].isin(test_sentences)].reset_index(drop=True)

print(f"\nTrain: {len(train_df)} utterances ({train_df['subject_id'].nunique()} speakers, {len(train_sentences)} sentences)")
print(f"Val:   {len(val_df)} utterances ({val_df['subject_id'].nunique()} speakers, {len(val_sentences)} sentences)")
print(f"Test:  {len(test_df)} utterances ({test_df['subject_id'].nunique()} speakers, {len(test_sentences)} sentences)")

train_df.to_csv(os.path.join(BASE, "train.csv"), index=False)
val_df.to_csv(os.path.join(BASE, "val.csv"), index=False)
test_df.to_csv(os.path.join(BASE, "test.csv"), index=False)
print("\nSaved train.csv, val.csv, test.csv")
