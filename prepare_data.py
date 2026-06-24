import os
import pandas as pd
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

BASE = os.path.dirname(os.path.abspath(__file__))
AUDIO_ROOT = os.path.join(BASE, "hindi_sent")
WORD_ROOT  = os.path.join(BASE, "WORD")

WORD_MAP = {
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
}

def romanized_to_devanagari(text):
    words = text.strip().split()
    result = []
    for w in words:
        if w in WORD_MAP:
            result.append(WORD_MAP[w])
        else:
            result.append(transliterate(w, sanscript.ITRANS, sanscript.DEVANAGARI))
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

print(f"\n--- Dataset Summary ---")
print(f"Total utterances     : {len(df)}")
print(f"Control speakers     : {df[df['is_control']]['subject_id'].nunique()}")
print(f"Dysarthric speakers  : {df[~df['is_control']]['subject_id'].nunique()}")
print(f"Missing transcripts  : {(df['transcript'] == '').sum()}")
print(f"\nSample rows:")
print(df[["subject_id", "is_control", "transcript_roman", "transcript"]].head(5).to_string())

df.to_csv("hdsd_manifest.csv", index=False)
print("\nSaved to hdsd_manifest.csv")
