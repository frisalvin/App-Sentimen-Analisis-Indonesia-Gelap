import streamlit as st
import pandas as pd
import torch
import re
import os
import zipfile
import gdown
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertTokenizerFast
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# ==================== SETUP ==================== #
st.set_page_config(page_title="Analisis Sentimen Indonesia Gelap", layout="centered")

# ==================== UNDUH DAN EKSTRAK MODEL ==================== #
model_path = "model"
model_zip = "model.zip"
file_id = "1lQC6eYJuEwOXdQ6JnRnntutoMN5DBkSg"
gdrive_url = f"https://drive.google.com/uc?id={file_id}"

if not os.path.exists(model_path):
    with st.spinner("📦 Mengunduh model dari Google Drive..."):
        gdown.download(gdrive_url, model_zip, quiet=False)
        with zipfile.ZipFile(model_zip, 'r') as zip_ref:
            zip_ref.extractall()

# ==================== LOAD MODEL ==================== #
try:
    tokenizer = AutoTokenizer.from_pretrained(f"{model_path}/tokenizer")
    tokenizer_fast = BertTokenizerFast.from_pretrained(f"{model_path}/tokenizer")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to("cpu")
    model.eval()
except Exception as e:
    st.error(f"Gagal memuat model sentimen: {e}")
    st.stop()

# ==================== LOAD KAMUS ==================== #
kamus_path = "data/kamuskatabaku.xlsx"
if not os.path.exists(kamus_path):
    st.error(f"Kamus tidak ditemukan di: {kamus_path}")
    st.stop()
else:
    kamus_df = pd.read_excel(kamus_path)
    kamus_baku = dict(zip(kamus_df['tidak_baku'], kamus_df['kata_baku']))

# ==================== TOOLS ==================== #
stemmer = StemmerFactory().create_stemmer()

def bersihkan_teks(teks):
    teks = re.sub(r'https?://\S+', ' ', teks)
    teks = re.sub(r'<.*?>', ' ', teks)
    teks = re.sub(r'[^\w\s]', ' ', teks)
    teks = re.sub(r'\d+', ' ', teks)
    return teks.lower().strip()

def normalisasi(teks):
    return ' '.join([kamus_baku.get(kata, kata) for kata in teks.split()])

def stemming(teks):
    hasil = stemmer.stem(teks)
    if "perintah" in hasil and "pemerintah" in teks:
        hasil = hasil.replace("perintah", "pemerintah")
    return hasil

def prediksi_sentimen(teks):
    inputs = tokenizer(teks, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        skor = torch.softmax(outputs.logits, dim=1)
        idx = torch.argmax(skor, dim=1).item()
    label = {0: "Negatif", 1: "Netral", 2: "Positif"}
    return label[idx], skor[0][idx].item()

def kata_dominan(teks_asli, teks_stem):
    inputs = tokenizer_fast(teks_stem, return_tensors="pt", truncation=True, max_length=128, return_attention_mask=True)
    tokens = tokenizer_fast.convert_ids_to_tokens(inputs['input_ids'][0])
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        attentions = outputs.attentions[-1]
        avg_attention = attentions.mean(dim=1)
        token_scores = avg_attention[0][0]

    token_score_pairs = list(zip(tokens, token_scores.tolist()))
    token_score_pairs = token_score_pairs[1:]

    stem_tokens = []
    kata_saat_ini = ""
    for token, score in sorted(token_score_pairs, key=lambda x: x[1], reverse=True):
        if token in ["[PAD]", "[SEP]", "[CLS]"]:
            continue
        if token.startswith("##"):
            kata_saat_ini += token[2:]
        else:
            if kata_saat_ini:
                stem_tokens.append(kata_saat_ini)
            kata_saat_ini = token
    if kata_saat_ini:
        stem_tokens.append(kata_saat_ini)

    original_words = teks_asli.split()
    stemmed_words = teks_stem.split()

    mapping = {}
    for ori in original_words:
        stem_ori = stemmer.stem(ori)
        mapping[stem_ori] = ori

    hasil = []
    for token in stem_tokens:
        if token in mapping:
            hasil.append(mapping[token])
        else:
            hasil.append(token)
        if len(hasil) >= 2:
            break

    return hasil

# ==================== UI APP ==================== #

st.markdown("""
<h1 style='text-align:center'>🧠 Analisis Sentimen Isu <span style='color:#ff4b4b'>'Indonesia Gelap'</span></h1>
<hr>
""", unsafe_allow_html=True)

# ==== Input Manual Komentar ==== #
st.subheader("✍️ Analisis Komentar")
kalimat_input = st.text_area("Masukkan opini atau komentar Anda:")

if st.button("🔍 Analisis Sekarang"):
    if kalimat_input.strip():
        with st.spinner("Sedang menganalisis..."):
            teks_bersih = bersihkan_teks(kalimat_input)
            teks_norm = normalisasi(teks_bersih)
            teks_stem = stemming(teks_norm)

            hasil_sentimen, skor = prediksi_sentimen(teks_stem)
            kata_kunci = kata_dominan(teks_norm, teks_stem)

            with st.expander("🔎 Tahapan Preprocessing"):
                st.markdown("**Teks Asli:**")
                st.code(kalimat_input)
                st.markdown("**1. Pembersihan Teks:**")
                st.code(teks_bersih)
                st.markdown("**2. Normalisasi:**")
                st.code(teks_norm)
                st.markdown("**3. Stemming:**")
                st.code(teks_stem)

            warna = {"Positif": "#2ecc71", "Negatif": "#e74c3c", "Netral": "#f1c40f"}
            st.markdown(f"""
            <h3 style='text-align:center'>✅ Hasil Sentimen: <span style='color:{warna[hasil_sentimen]}'>{hasil_sentimen}</span></h3>
            <p style='text-align:center'>Confidence Score: <strong>{skor:.2%}</strong></p>
            <p style='text-align:center'><strong>Kata Dominan:</strong> {', '.join(kata_kunci)}</p>
            """, unsafe_allow_html=True)
    else:
        st.info("Silakan masukkan kalimat terlebih dahulu.")

# ==== Upload File untuk Analisis Massal ==== #
st.markdown("---")
st.subheader("📤 Analisis Komentar (CSV / Excel)")

uploaded_file = st.file_uploader("Unggah file CSV atau Excel", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        kolom_teks = st.selectbox("📌 Pilih kolom yang berisi komentar:", df.columns)

        if st.button("🔍 Analisis Komentar"):
            with st.spinner("Sedang menganalisis semua komentar..."):
                hasil_sentimen = []
                skor_sentimen = []
                kata_dominan_list = []

                for kalimat in df[kolom_teks]:
                    if pd.isna(kalimat):
                        hasil_sentimen.append("Tidak Valid")
                        skor_sentimen.append("0%")
                        kata_dominan_list.append("-")
                        continue
                    teks_bersih = bersihkan_teks(str(kalimat))
                    teks_norm = normalisasi(teks_bersih)
                    teks_stem = stemming(teks_norm)
                    label, skor = prediksi_sentimen(teks_stem)
                    kata_kunci = kata_dominan(teks_norm, teks_stem)

                    hasil_sentimen.append(label)
                    skor_sentimen.append(f"{skor*100:.2f}%")
                    kata_dominan_list.append(", ".join(kata_kunci))

                df["Hasil Sentimen"] = hasil_sentimen
                df["Skor (%)"] = skor_sentimen
                df["Kata Dominan"] = kata_dominan_list

                st.success("Analisis selesai ✅")
                st.dataframe(df[[kolom_teks, "Hasil Sentimen", "Skor (%)", "Kata Dominan"]])

                csv_output = df.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Unduh Hasil (CSV)", data=csv_output, file_name="hasil_sentimen.csv", mime="text/csv")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses file: {e}")
