import streamlit as st
import whisper
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import pyttsx3
import sounddevice as sd
import numpy as np
import tempfile
import wavio

# Initialize the text-to-speech engine
engine = pyttsx3.init()

def speak(message):
    try:
        engine.say(message)
        engine.runAndWait()
    except Exception as e:
        print(f"Text-to-speech error: {e}")

# Function to transcribe audio using Whisper
def transcribe_audio_with_whisper(audio_path):
    try:
        model = whisper.load_model("base")
        result = model.transcribe(audio_path)
        return result['text']
    except Exception as e:
        st.error(f"Error transcribing audio: {e}")
        return ""

# Function to query the LLM
def query_llm(transcription_text, question):
    try:
        tokenizer = AutoTokenizer.from_pretrained("stabilityai/stablelm-zephyr-3b")
        model = AutoModelForCausalLM.from_pretrained("stabilityai/stablelm-zephyr-3b")
        input_text = f"Here is the transcription of the audio:\n\n{transcription_text}\n\nQuestion: {question}"
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=2048)
        outputs = model.generate(inputs['input_ids'], max_new_tokens=150, num_return_sequences=1)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split("Question:")[1].strip() if "Question:" in response else response.strip()
    except Exception as e:
        st.error(f"Error querying the model: {e}")
        return ""

# Function to record audio from the microphone
def record_audio(duration=5, fs=44100):
    try:
        st.info("Recording...")
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float64')
        sd.wait()  # Wait until recording is finished
        st.success("Recording complete!")
        return np.squeeze(recording)
    except Exception as e:
        st.error(f"Error recording audio: {e}")
        return np.array([])

# Save the recording to a temporary file
def save_audio(audio, fs):
    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_file_path = temp_file.name
            wavio.write(temp_file_path, audio, fs, sampwidth=2)
        return temp_file_path
    except Exception as e:
        st.error(f"Error saving audio file: {e}")
        return ""

# Streamlit application
st.title("Audio Transcription and Q&A with Voice Interaction")

# Step 1: Upload audio file
uploaded_file = st.file_uploader("Upload an MP3 file", type=["mp3"])

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        temp_file_path = temp_file.name
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.read())

    # Transcribe the uploaded audio file
    with st.spinner("Transcribing audio..."):
        transcribed_text = transcribe_audio_with_whisper(temp_file_path)
    st.success("Transcription completed!")
    
    # Display transcription for reference
    st.subheader("Transcription")
    st.write(transcribed_text)
    
    # Ask a question about the transcription
    st.header("Ask a question about the transcription using your voice")
    
    if st.button("Record Question"):
        audio = record_audio(duration=5)  # Record for 5 seconds
        if audio.size == 0:
            st.error("No audio recorded.")
        else:
            audio_file_path = save_audio(audio, 44100)

            # Convert the speech to text
            with st.spinner("Converting speech to text..."):
                question = transcribe_audio_with_whisper(audio_file_path)
            st.success(f"Question recognized: {question}")

            # Generate the answer
            with st.spinner("Generating answer..."):
                answer = query_llm(transcribed_text, question)
            st.success("Answer generated!")
            st.write(answer)

            # Speak out the answer
            speak(answer)
