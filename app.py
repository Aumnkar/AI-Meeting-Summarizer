import streamlit as st
import whisper
from transformers import pipeline
import tempfile
import os

# Load models once
@st.cache_resource
def load_models():
    whisper_model = whisper.load_model("base")
    summarizer_model = pipeline("summarization", model="facebook/bart-large-cnn")
    return whisper_model, summarizer_model

st.title("🎤 AI Meeting Summarizer")

uploaded_file = st.file_uploader("Upload meeting audio (.mp3, .wav)", type=["mp3", "wav"])

if uploaded_file:
    st.audio(uploaded_file)

    with st.spinner("Transcribing audio..."):
        temp_audio = tempfile.NamedTemporaryFile(delete=False)
        temp_audio.write(uploaded_file.read())
        temp_audio.close()

        whisper_model, summarizer_model = load_models()

        result = whisper_model.transcribe(temp_audio.name)
        transcript = result["text"]

    st.subheader("📝 Transcript")
    st.write(transcript)

    with st.spinner("Summarizing..."):
        summary = summarizer_model(transcript, max_length=130, min_length=30, do_sample=False)
        st.subheader("📄 Summary")
        st.write(summary[0]['summary_text'])

    # Optional: Extract action items
    st.subheader("✅ Action Items")
    keywords = ["need to", "should", "must", "assign", "due", "deadline", "follow up"]
    actions = [s for s in transcript.split(". ") if any(k in s.lower() for k in keywords)]

    if actions:
        for idx, item in enumerate(actions, 1):
            st.write(f"{idx}. {item}")
    else:
        st.info("No clear action items found.")

    # Clean up
    os.remove(temp_audio.name)
