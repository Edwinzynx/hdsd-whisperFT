import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

df = pd.read_csv("hdsd_manifest.csv")

# use only dysarthric speakers for training/eval
dysarthric = df[df["is_control"] == False].copy()
print(f"Dysarthric utterances: {len(dysarthric)}, speakers: {dysarthric['subject_id'].nunique()}")

# 70% train, 15% val, 15% test — split by speaker, not utterance
gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
train_idx, temp_idx = next(gss.split(dysarthric, groups=dysarthric["subject_id"]))

train_df = dysarthric.iloc[train_idx]
temp_df  = dysarthric.iloc[temp_idx]

gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df["subject_id"]))

val_df  = temp_df.iloc[val_idx]
test_df = temp_df.iloc[test_idx]

print(f"\nTrain: {train_df['subject_id'].nunique()} speakers, {len(train_df)} utterances")
print(f"Val:   {val_df['subject_id'].nunique()} speakers, {len(val_df)} utterances")
print(f"Test:  {test_df['subject_id'].nunique()} speakers, {len(test_df)} utterances")

train_df.to_csv("train.csv", index=False)
val_df.to_csv("val.csv",   index=False)
test_df.to_csv("test.csv", index=False)
print("\nSaved train.csv, val.csv, test.csv")
