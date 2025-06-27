import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import tempfile
from search.search_similar import find_top_3

st.set_page_config(page_title="Tìm kiếm nhạc cụ bộ dây", layout="wide")
st.title("🎻 Tìm kiếm tiếng nhạc cụ bộ dây")

uploaded = st.file_uploader("📂 Tải file .mp3", type=["mp3"])
if uploaded:
    # Lưu file tạm
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        f.write(uploaded.read())
        temp_path = f.name

    # Phát âm thanh input
    st.audio(uploaded, format='audio/mp3')
    st.subheader("🎵 Biểu đồ sóng âm thanh của file đã tải lên:")

    # Vẽ waveform
    # y, sr = librosa.load(temp_path, sr=22050)
    # fig, ax = plt.subplots(figsize=(10, 2))
    # librosa.display.waveshow(y, sr=sr, ax=ax)
    # ax.set_title("Waveform")
    # ax.set_xlabel("Thời gian (s)")
    # ax.set_ylabel("Biên độ")
    # st.pyplot(fig)

    st.info("🔍 Đang tìm kiếm 3 file giống nhất...")

    # Tìm top 3
    results = find_top_3(temp_path)
    st.success("✅ Kết quả:")

    for i, (fname, label, score) in enumerate(results, 1):
        st.markdown(f"**{i}. `{fname}` ({label}) — Similarity: `{score:.4f}`**")

        file_path = os.path.join("dataset", label, fname)
        if os.path.exists(file_path):
            # Phát file âm thanh
            with open(file_path, 'rb') as f:
                audio_bytes = f.read()
            st.audio(audio_bytes, format='audio/mp3')

            # Vẽ waveform
            # y2, sr2 = librosa.load(file_path, sr=22050)
            # fig2, ax2 = plt.subplots(figsize=(10, 2))
            # librosa.display.waveshow(y2, sr=sr2, ax=ax2)
            # ax2.set_title(f"Waveform: {fname}")
            # ax2.set_xlabel("Thời gian (s)")
            # ax2.set_ylabel("Biên độ")
            # st.pyplot(fig2)
        else:
            st.warning(f"⚠️ Không tìm thấy file: `{file_path}`")
