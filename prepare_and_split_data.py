import os
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

BASE = os.path.dirname(os.path.abspath(__file__))
HDSD_ROOT = os.path.join(BASE, "hindi indic", "HDSD")
AUDIO_ROOT = os.path.join(HDSD_ROOT, "hindi_sent")
WORD_ROOT  = os.path.join(HDSD_ROOT, "WORD")

# Complete, perfect dictionary mapping for all 128 unique Hinglish words
FIXED_WORD_MAP = {
    "aapakei": "आपके", "karanei": "करने", "isakei": "इसके",
    "badalei": "बदले", "mein": "में", "isei": "इसे",
    "aisei": "ऐसे", "rakhein": "रखें", "kei": "के",
    "liyei": "लिए", "aagei": "आगे", "jaanei": "जाने",
    "kahatei": "कहते", "diijiyei": "दीजिए", "kiijiei": "कीजिए",
    "jaaei": "जाए", "rahei": "रहे", "padhtei": "पढ़ते",
    "hootaa": "होता", "karataa": "करता", "jaataa": "जाता",
    "rahataa": "रहता", "chalaa": "चला", "gayaa": "गया",
    "huaa": "हुआ", "thaa": "था", "liyaa": "लिया",
    "kahaa": "कहा", "deikhaa": "देखा", "bhuula": "भूला",
    "sakaa": "सका", "lagaa": "लगा", "aabhaara": "आभार",
    "dhanyavaada": "धन्यवाद", "svaagata": "स्वागत", "bhaarata": "भारत",
    "hindii": "हिंदी", "khushii": "खुशी", "jaldii": "जल्दी",
    "meirii": "मेरी", "abhii": "अभी", "bhii": "भी",
    "hii": "ही", "nahiin": "नहीं", "huiin": "हुईं",
    "jiivana": "जीवन", "krikeita": "क्रिकेट", "sandeisha": "संदेश",
    "sangharshha": "संघर्ष", "sahayooga": "सहयोग", "sahaaraa": "सहारा",
    "sanbhava": "संभव", "chikitsaa": "चिकित्सा", "unhoonnei": "उन्होंने",
    "looga": "लोग", "shaayada": "शायद", "shaanta": "शांत",
    "shuruu": "शुरू", "dhyaana": "ध्यान", "paalana": "पालन",
    "samajha": "समझ", "paakara": "पाकर", "maanaga": "मांग",
    "huuna": "हूं", "oora": "और", "aura": "और",
    "huii": "हुई", "jaarii": "जारी", "ilaaja": "इलाज",
    "kabhii": "कभी", "sabhii": "सभी",
    
    # Missing words identified during vocab analysis
    "hai": "है",
    "main": "मैं",
    "bahuta": "बहुत",
    "sei": "से",
    "kaa": "का",
    "naa": "ना",
    "hoo": "हो",
    "aisaa": "ऐसा",
    "isa": "इस",
    "joo": "जो",
    "pasanda": "पसंद",
    "ki": "कि",
    "kuchha": "कुछ",
    "aapa": "आप",
    "achchhaa": "अच्छा",
    "para": "पर",
    "artha": "अर्थ",
    "koo": "को",
    "yaha": "यह",
    "baatoon": "बातों",
    "madada": "मदद",
    "jhuutha": "झूठ",
    "kahataa": "कहता",
    "yahaan": "यहाँ",
    "tuma": "तुम",
    "heitu": "हेतु",
    "daura": "दौर",
    "doonoon": "दोनों",
    "antara": "अंतर",
    "kara": "कर",
    "aa": "आ",
    "dara": "दर",
    "usakaa": "उसका",
    "karoo": "करो",
    "usasei": "उससे",
    "kitanei": "कितने",
    "badaa": "बड़ा",
    "kheila": "खेल",
    "aapanei": "आपने",
    "phira": "फिर",
    "kaarya": "कार्य",
    "loo": "लो",
    "aba": "अब",
    "sakaa": "सका",
    "shuruu": "शुरू",
    "rahei": "रहे",
    "samaya": "समय",
    "saatha": "साथ",
    "bhaya": "भय",
    "baara": "बार",
    "kaaphii": "काफी",
    "baata": "बात",
    "vaha": "वह",
    "eika": "एक",
    "apanaa": "अपना",
    "aapakaa": "आपका",
    "magara": "मगर",
    "par": "पर",
    "pasand": "पसंद",
    "samayaa": "समय",
    "maang": "मांग",
    "hindi": "हिंदी"
}

def romanized_to_devanagari(text):
    words = text.strip().split()
    result = []
    for w in words:
        wl = w.lower()
        if wl in FIXED_WORD_MAP:
            result.append(FIXED_WORD_MAP[wl])
        else:
            print(f"Warning: word '{w}' not in map, using fallback ITRANS")
            result.append(transliterate(wl, sanscript.ITRANS, sanscript.DEVANAGARI))
    return " ".join(result)

def read_transcript(txt_path):
    words = []
    with open(txt_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                words.append(parts[2])
    return " ".join(words)

records = []

for speaker in sorted(os.listdir(AUDIO_ROOT)):
    speaker_audio = os.path.join(AUDIO_ROOT, speaker)
    speaker_word  = os.path.join(WORD_ROOT, speaker)

    if not os.path.isdir(speaker_audio):
        continue

    is_control = speaker.startswith("C")
    gender = "F" if "F" in speaker.lstrip("C")[:1] else "M"

    for wav_file in sorted(os.listdir(speaker_audio)):
        if not wav_file.endswith("_M2.wav"):
            continue

        parts = wav_file.replace(".wav", "").split("_")
        session_id  = parts[1]
        sentence_id = parts[2]

        wav_path = os.path.join(speaker_audio, wav_file)
        txt_file = os.path.join(speaker_word, f"{speaker}_{session_id}_{sentence_id}.txt")

        transcript_roman = ""
        transcript_deva  = ""

        if os.path.exists(txt_file):
            transcript_roman = read_transcript(txt_file)
            transcript_deva  = romanized_to_devanagari(transcript_roman)
        else:
            print(f"missing: {txt_file}")

        records.append({
            "path":             wav_path,
            "subject_id":       speaker,
            "session_id":       session_id,
            "sentence_id":      sentence_id,
            "gender":           gender,
            "is_control":       is_control,
            "transcript_roman": transcript_roman,
            "transcript":       transcript_deva,
        })

df = pd.DataFrame(records)

# Filter out controls completely
dysarthric = df[df["is_control"] == False].copy()

# Filter out empty/NaN transcripts
dysarthric = dysarthric[dysarthric["transcript"].notna() & (dysarthric["transcript"].str.strip() != "")].reset_index(drop=True)

print(f"\nTotal records: {len(df)}")
print(f"Controls and empty records filtered out. Remaining dysarthric records: {len(dysarthric)}")

# Split unique speakers (speaker-independent)
unique_speakers = sorted(dysarthric["subject_id"].unique())
train_speakers, temp_speakers = train_test_split(
    unique_speakers,
    test_size=0.30,
    random_state=42
)
val_speakers, test_speakers = train_test_split(
    temp_speakers,
    test_size=0.50,
    random_state=42
)

# Split unique sentences (sentence-independent)
unique_sentences = sorted(dysarthric["sentence_id"].unique())
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

# Filter splits by BOTH speaker and sentence (dual-independent)
train_df = dysarthric[dysarthric["subject_id"].isin(train_speakers) & dysarthric["sentence_id"].isin(train_sentences)].reset_index(drop=True)
val_df   = dysarthric[dysarthric["subject_id"].isin(val_speakers) & dysarthric["sentence_id"].isin(val_sentences)].reset_index(drop=True)
test_df  = dysarthric[dysarthric["subject_id"].isin(test_speakers) & dysarthric["sentence_id"].isin(test_sentences)].reset_index(drop=True)

print(f"Train utterances: {len(train_df)}")
print(f"Val utterances:   {len(val_df)}")
print(f"Test utterances:  {len(test_df)}")

# Verify zero leakage in speakers
overlap_spk_tv = set(train_df["subject_id"]) & set(val_df["subject_id"])
overlap_spk_tt = set(train_df["subject_id"]) & set(test_df["subject_id"])
assert len(overlap_spk_tv) == 0, f"Speaker leakage train-val: {overlap_spk_tv}"
assert len(overlap_spk_tt) == 0, f"Speaker leakage train-test: {overlap_spk_tt}"

# Verify zero leakage in sentences
overlap_se_tv = set(train_df["sentence_id"]) & set(val_df["sentence_id"])
overlap_se_tt = set(train_df["sentence_id"]) & set(test_df["sentence_id"])
assert len(overlap_se_tv) == 0, f"Sentence leakage train-val: {overlap_se_tv}"
assert len(overlap_se_tt) == 0, f"Sentence leakage train-test: {overlap_se_tt}"

print("Zero speaker leakage & Zero sentence leakage verified successfully!")

# Save splits to root directory
train_df.to_csv(os.path.join(BASE, "train.csv"), index=False)
val_df.to_csv(os.path.join(BASE, "val.csv"), index=False)
test_df.to_csv(os.path.join(BASE, "test.csv"), index=False)

# Copy to hdsd_*.csv files for notebook compatibility
train_df.to_csv(os.path.join(BASE, "hdsd_train.csv"), index=False)
val_df.to_csv(os.path.join(BASE, "hdsd_val.csv"), index=False)
test_df.to_csv(os.path.join(BASE, "hdsd_test.csv"), index=False)

# Also save the updated hdsd_manifest.csv
df.to_csv(os.path.join(BASE, "hdsd_manifest.csv"), index=False)

# Save speaker_split.json to keep it aligned
speaker_split = {
    "train_speakers": sorted(list(train_speakers)),
    "val_speakers": sorted(list(val_speakers)),
    "test_speakers": sorted(list(test_speakers))
}
import json
with open(os.path.join(BASE, "speaker_split.json"), "w") as f:
    json.dump(speaker_split, f, indent=2)

print("\nAll datasets split and saved successfully!")
