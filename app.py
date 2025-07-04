import streamlit as st
from transformers import pipeline
import tempfile

# Load Whisper ASR model from HuggingFace
@st.cache_resource
def load_model():
    return pipeline("automatic-speech-recognition", model="openai/whisper-small")

asr_pipeline = load_model()

st.title("🎙️ AI Meeting Summarizer")

uploaded_file = st.file_uploader("Upload an audio file (.mp3, .wav)", type=["mp3", "wav"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as temp_audio:
        temp_audio.write(uploaded_file.read())
        temp_audio_path = temp_audio.name

    with st.spinner("Transcribing..."):
        result = asr_pipeline(temp_audio_path)
        transcript = result["text"]
        st.subheader("Transcript:")
        st.write(transcript)

        # Summarize
        with st.spinner("Summarizing..."):
            from transformers import pipeline as summary_pipeline
            summarizer = summary_pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

            # Split if transcript is too long
            if len(transcript) > 1000:
                chunks = [transcript[i:i+1000] for i in range(0, len(transcript), 1000)]
                summary = ""
                for chunk in chunks:
                    summary_text = summarizer(chunk, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
                    summary += summary_text + " "
            else:
                summary = summarizer(transcript, max_length=130, min_length=30, do_sample=False)[0]['summary_text']

            st.subheader("📝 Summary:")
            st.write(summary)
