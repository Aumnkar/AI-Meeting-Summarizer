import streamlit as st
from transformers import pipeline
from pydub import AudioSegment
import tempfile
import datetime
import time

# ---- Audio chunking ----
def split_audio(uploaded_file, chunk_length_ms=30000):
    audio = AudioSegment.from_file(uploaded_file)
    chunks = []
    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i:i+chunk_length_ms]
        temp_chunk = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        chunk.export(temp_chunk.name, format="wav")
        chunks.append(temp_chunk.name)
    return chunks

def chunk_text(text, max_words=600):
    words = text.split()
    for i in range(0, len(words), max_words):
        yield " ".join(words[i:i+max_words])

# ---- Load models with cache ----
@st.cache_resource
def load_asr_model():
    return pipeline("automatic-speech-recognition", model="openai/whisper-small")

@st.cache_resource
def load_summarizer_model():
    return pipeline("summarization", model="facebook/bart-large-cnn")

# ---- Streamlit UI ----
st.title("🤖 AI Meeting Summarizer")

st.markdown("### 📤 Upload audio file (.mp3 or .wav)")
uploaded_file = st.file_uploader("Upload your audio file here", type=["mp3", "wav"])

if st.button("▶️ Process Audio", key="process_button"):
    if not uploaded_file:
        st.warning("⚠️ Please upload an audio file first!")
        st.stop()

    st.markdown("---")
    st.subheader("🌀 Phase 1: Splitting audio into chunks (30 sec)")
    audio_chunks = split_audio(uploaded_file)

    asr_model = load_asr_model()
    summarizer = load_summarizer_model()

    transcript = ""
    summary = ""

    st.subheader("🔊 Phase 2: Transcribing audio")
    progress_trans = st.progress(0)
    for i, chunk in enumerate(audio_chunks):
        result = asr_model(chunk)
        transcript += result["text"] + " "
        progress_trans.progress(int((i + 1) / len(audio_chunks) * 100))
        time.sleep(0.1)

    st.subheader("📡 Phase 3: Summarizing text")
    progress_sum = st.progress(0)
    chunks = list(chunk_text(transcript))
    for i, chunk in enumerate(chunks):
        sum_text = summarizer(chunk, max_length=150, min_length=30, do_sample=False)[0]["summary_text"]
        summary += sum_text + " "
        progress_sum.progress(int((i + 1) / len(chunks) * 100))
        time.sleep(0.1)

    st.markdown("---")
    st.subheader("📜 Transcript")
    st.text_area("Full Transcript", transcript, height=250)

    st.subheader("📌 Summary")
    st.text_area("Meeting Summary", summary, height=200)

    st.download_button(
        "📥 Download Transcript",
        transcript,
        file_name=f"transcript_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
    )
    st.download_button(
        "📥 Download Summary",
        summary,
        file_name=f"summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
    )
