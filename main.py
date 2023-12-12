import streamlit as st

import torch

import inflect
import nltk

from transformers import pipeline
from nltk.tokenize import sent_tokenize

import numpy as np

from bark.generation import preload_models
from bark import generate_audio, SAMPLE_RATE

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def normalize_text(text):
    p = inflect.engine()
    # Handle numbers
    text = " ".join(
        [p.number_to_words(word) if word.isdigit() else word for word in text.split()]
    )
    # Expand common abbreviations
    text = text.replace("St.", "Street").replace("Dr.", "Doctor")
    return text


def tokenize_into_sentences(text):
    nltk.download("punkt")
    sentences = sent_tokenize(text)
    return sentences


def summarization_model(input_text):
    summarizer = pipeline(
        "summarization", model="facebook/bart-large-cnn", device=device
    )
    summarization = summarizer(
        input_text, max_length=130, min_length=30, do_sample=False
    )[0]["summary_text"]
    return summarization


def tts_model(output_text):
    preload_models()
    _output_text = normalize_text(output_text)
    _output_text = tokenize_into_sentences(_output_text)
    SPEAKER = "v2/en_speaker_6"
    silence = np.zeros(int(0.05 * SAMPLE_RATE))  # quarter second of silence

    pieces = []
    for sentence in _output_text:
        audio_array = generate_audio(sentence, history_prompt=SPEAKER)
        pieces += [audio_array, silence.copy()]
    st.audio(
        np.concatenate(pieces),
        format="audio/wav",
        start_time=0,
        sample_rate=SAMPLE_RATE,
    )


# Title of the web app
st.title("Text Summarizer - CS469")

# Create two columns for the input and output text
input_col, output_col = st.columns(2, gap="large")

with input_col:
    st.header("Input Text")
    input_text = st.text_area("Please enter your input", height=300, key="input_text")
    if st.button("Summarize"):
        # Calls the summarize_text function and updates the output_text in the session state
        if st.session_state.input_text != "":
            st.session_state.output_text = summarization_model(
                st.session_state.input_text
            )


with output_col:
    st.header("Output Text")
    if "output_text" not in st.session_state:
        st.session_state.output_text = ""
    output_text = st.text_area(
        "", value=st.session_state.output_text, height=300, key="output_text_display"
    )
    if st.button("🔊"):
        # Text-to-Speech Model
        tts_model(st.session_state.output_text)
        print()
