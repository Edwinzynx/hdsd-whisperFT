import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))
notebook_path = os.path.join(BASE, "01_HDSD_Preprocessing_Baseline.ipynb")

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Define the new cell contents
split_cell_source = [
    "# Filter out control speakers and empty/NaN transcripts\n",
    "is_control = df[\"speaker\"].str.startswith(\"C\")\n",
    "dysarthric = df[~is_control].copy()\n",
    "dysarthric = dysarthric[dysarthric[\"text\"].notna() & (dysarthric[\"text\"].str.strip() != \"\")].reset_index(drop=True)\n",
    "\n",
    "dysarthric[\"sentence_id\"] = dysarthric[\"utterance\"].str.split(\"_\").str[2]\n",
    "\n",
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "# Split unique speakers (speaker-independent)\n",
    "unique_speakers = sorted(dysarthric[\"speaker\"].unique())\n",
    "train_speakers, temp_speakers = train_test_split(\n",
    "    unique_speakers,\n",
    "    test_size=0.30,\n",
    "    random_state=42\n",
    ")\n",
    "val_speakers, test_speakers = train_test_split(\n",
    "    temp_speakers,\n",
    "    test_size=0.50,\n",
    "    random_state=42\n",
    ")\n",
    "\n",
    "# Split unique sentences (sentence-independent)\n",
    "unique_sentences = sorted(dysarthric[\"sentence_id\"].unique())\n",
    "train_sentences, temp_sentences = train_test_split(\n",
    "    unique_sentences,\n",
    "    test_size=0.30,\n",
    "    random_state=42\n",
    ")\n",
    "val_sentences, test_sentences = train_test_split(\n",
    "    temp_sentences,\n",
    "    test_size=0.50,\n",
    "    random_state=42\n",
    ")\n",
    "\n",
    "# Filter splits by BOTH speaker and sentence (dual-independent)\n",
    "train_df = dysarthric[dysarthric[\"speaker\"].isin(train_speakers) & dysarthric[\"sentence_id\"].isin(train_sentences)].reset_index(drop=True)\n",
    "val_df   = dysarthric[dysarthric[\"speaker\"].isin(val_speakers) & dysarthric[\"sentence_id\"].isin(val_sentences)].reset_index(drop=True)\n",
    "test_df  = dysarthric[dysarthric[\"speaker\"].isin(test_speakers) & dysarthric[\"sentence_id\"].isin(test_sentences)].reset_index(drop=True)\n",
    "\n",
    "print(len(train_df), len(val_df), len(test_df))"
]

normalize_cell_source = [
    "from indic_transliteration import sanscript\n",
    "from indic_transliteration.sanscript import transliterate\n",
    "\n",
    "# Complete, perfect dictionary mapping for all 128 unique Hinglish words\n",
    "FIXED_WORD_MAP = {\n",
    "    \"aapakei\": \"आपके\", \"karanei\": \"करने\", \"isakei\": \"इसके\",\n",
    "    \"badalei\": \"बदले\", \"mein\": \"में\", \"isei\": \"इसे\",\n",
    "    \"aisei\": \"ऐसे\", \"rakhein\": \"रखें\", \"kei\": \"के\",\n",
    "    \"liyei\": \"लिए\", \"aagei\": \"आगे\", \"jaanei\": \"जाने\",\n",
    "    \"kahatei\": \"कहते\", \"diijiyei\": \"दीजिए\", \"kiijiei\": \"कीजिए\",\n",
    "    \"jaaei\": \"जाए\", \"rahei\": \"रहे\", \"padhtei\": \"पढ़ते\",\n",
    "    \"hootaa\": \"होता\", \"karataa\": \"करता\", \"jaataa\": \"जाता\",\n",
    "    \"rahataa\": \"रहता\", \"chalaa\": \"चला\", \"gayaa\": \"गया\",\n",
    "    \"huaa\": \"हुआ\", \"thaa\": \"था\", \"liyaa\": \"लिया\",\n",
    "    \"kahaa\": \"कहा\", \"deikhaa\": \"देखा\", \"bhuula\": \"भूला\",\n",
    "    \"sakaa\": \"सका\", \"lagaa\": \"लगा\", \"aabhaara\": \"आभार\",\n",
    "    \"dhanyavaada\": \"धन्यवाद\", \"svaagata\": \"स्वागत\", \"bhaarata\": \"भारत\",\n",
    "    \"hindii\": \"हिंदी\", \"khushii\": \"खुशी\", \"jaldii\": \"जल्दी\",\n",
    "    \"meirii\": \"मेरी\", \"abhii\": \"अभी\", \"bhii\": \"भी\",\n",
    "    \"hii\": \"ही\", \"nahiin\": \"नहीं\", \"huiin\": \"हुईं\",\n",
    "    \"jiivana\": \"जीवन\", \"krikeita\": \"क्रिकेट\", \"sandeisha\": \"संदेश\",\n",
    "    \"sangharshha\": \"संघर्ष\", \"sahayooga\": \"सहयोग\", \"sahaaraa\": \"सहारा\",\n",
    "    \"sanbhava\": \"संभव\", \"chikitsaa\": \"चिकित्सा\", \"unhoonnei\": \"उन्होंने\",\n",
    "    \"looga\": \"लोग\", \"shaayada\": \"शायद\", \"shaanta\": \"शांत\",\n",
    "    \"shuruu\": \"शुरू\", \"dhyaana\": \"ध्यान\", \"paalana\": \"पालन\",\n",
    "    \"samajha\": \"समझ\", \"paakara\": \"पाकर\", \"maanaga\": \"मांग\",\n",
    "    \"huuna\": \"हूं\", \"oora\": \"और\", \"aura\": \"और\",\n",
    "    \"huii\": \"हुई\", \"jaarii\": \"जारी\", \"ilaaja\": \"इलाज\",\n",
    "    \"kabhii\": \"कभी\", \"sabhii\": \"सभी\",\n",
    "    \"hai\": \"है\", \"main\": \"मैं\", \"bahuta\": \"बहुत\",\n",
    "    \"sei\": \"से\", \"kaa\": \"का\", \"naa\": \"ना\", \"hoo\": \"हो\",\n",
    "    \"aisaa\": \"ऐसा\", \"isa\": \"इस\", \"joo\": \"जो\", \"pasanda\": \"पसंद\",\n",
    "    \"ki\": \"कि\", \"kuchha\": \"कुछ\", \"aapa\": \"आप\", \"achchhaa\": \"अच्छा\",\n",
    "    \"para\": \"पर\", \"artha\": \"अर्थ\", \"koo\": \"को\", \"yaha\": \"यह\",\n",
    "    \"baatoon\": \"बातों\", \"madada\": \"मदद\", \"jhuutha\": \"झूठ\",\n",
    "    \"kahataa\": \"कहता\", \"yahaan\": \"यहाँ\", \"tuma\": \"तुम\",\n",
    "    \"heitu\": \"हेतु\", \"daura\": \"दौर\", \"doonoon\": \"दोनों\",\n",
    "    \"antara\": \"अंतर\", \"kara\": \"कर\", \"aa\": \"आ\", \"dara\": \"दर\",\n",
    "    \"usakaa\": \"उसका\", \"karoo\": \"करो\", \"usasei\": \"उससे\",\n",
    "    \"kitanei\": \"कितने\", \"badaa\": \"बड़ा\", \"kheila\": \"खेल\",\n",
    "    \"aapanei\": \"आपने\", \"phira\": \"फिर\", \"kaarya\": \"कार्य\",\n",
    "    \"loo\": \"लो\", \"aba\": \"अब\", \"sakaa\": \"सका\", \"shuruu\": \"शुरू\",\n",
    "    \"rahei\": \"रहे\", \"samaya\": \"समय\", \"saatha\": \"साथ\", \"bhaya\": \"भय\",\n",
    "    \"baara\": \"बार\", \"kaaphii\": \"काफी\", \"baata\": \"बात\", \"vaha\": \"वह\",\n",
    "    \"eika\": \"एक\", \"apanaa\": \"अपना\", \"aapakaa\": \"आपका\", \"magara\": \"मगर\",\n",
    "    \"par\": \"पर\", \"pasand\": \"पसंद\", \"samayaa\": \"समय\", \"maang\": \"मांग\",\n",
    "    \"hindi\": \"हिंदी\"\n",
    "}\n",
    "\n",
    "def normalize_hdsd(text):\n",
    "    words = text.strip().split()\n",
    "    result = []\n",
    "    for w in words:\n",
    "        wl = w.lower()\n",
    "        if wl in FIXED_WORD_MAP:\n",
    "            result.append(FIXED_WORD_MAP[wl])\n",
    "        else:\n",
    "            result.append(transliterate(wl, sanscript.ITRANS, sanscript.DEVANAGARI))\n",
    "    return \" \".join(result)\n"
]

apply_cell_source = [
    "df[\"text_norm\"] = df[\"text\"].apply(normalize_hdsd)\n",
    "\n",
    "df[\"text_devnagari\"] = df[\"text_norm\"].apply(\n",
    "    lambda x: transliterate(\n",
    "        x,\n",
    "        sanscript.ITRANS,\n",
    "        sanscript.DEVANAGARI\n",
    "    )\n",
    ")"
]

rebuild_split_cell_source = [
    "# Filter out control speakers and empty/NaN transcripts\n",
    "is_control = df[\"speaker\"].str.startswith(\"C\")\n",
    "dysarthric = df[~is_control].copy()\n",
    "dysarthric = dysarthric[dysarthric[\"text_devnagari\"].notna() & (dysarthric[\"text_devnagari\"].str.strip() != \"\")].reset_index(drop=True)\n",
    "\n",
    "dysarthric[\"sentence_id\"] = dysarthric[\"utterance\"].str.split(\"_\").str[2]\n",
    "\n",
    "# Filter splits by BOTH speaker and sentence (dual-independent)\n",
    "train_df = dysarthric[dysarthric[\"speaker\"].isin(train_speakers) & dysarthric[\"sentence_id\"].isin(train_sentences)].reset_index(drop=True)\n",
    "val_df   = dysarthric[dysarthric[\"speaker\"].isin(val_speakers) & dysarthric[\"sentence_id\"].isin(val_sentences)].reset_index(drop=True)\n",
    "test_df  = dysarthric[dysarthric[\"speaker\"].isin(test_speakers) & dysarthric[\"sentence_id\"].isin(test_sentences)].reset_index(drop=True)\n",
    "\n",
    "print(len(train_df))\n",
    "print(len(val_df))\n",
    "print(len(test_df))"
]

modified_cells = 0
for cell in nb["cells"]:
    cell_id = cell.get("id")
    if cell_id == "9f856cb1ce5458a7":
        cell["source"] = split_cell_source
        modified_cells += 1
    elif cell_id == "aaf07a99b3d44daa":
        cell["source"] = normalize_cell_source
        modified_cells += 1
    elif cell_id == "923ee2d8e96dfe8f":
        cell["source"] = apply_cell_source
        modified_cells += 1
    elif cell_id == "ce6103b5ecf66595":
        cell["source"] = rebuild_split_cell_source
        modified_cells += 1

print(f"Modified {modified_cells} cells.")

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Saved notebook successfully.")
