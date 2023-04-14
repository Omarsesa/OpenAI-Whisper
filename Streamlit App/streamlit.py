import streamlit as st
import whisper
import numpy as np
model = whisper.load_model("base")
print(
    f"Model is {'multilingual' if model.is_multilingual else 'English-only'} "
    f"and has {sum(np.prod(p.shape) for p in model.parameters()):,} parameters."
)

st.subheader("Please press the button and start talking ")
st.button("Start")
st.subheader("the result")







import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv
  
# Sampling frequency
freq = 44100
  
# Recording duration
duration = 5
  
# Start recorder with the given values 
# of duration and sample frequency
recording =st.progress( sd.rec(int(duration * freq), 
                   samplerate=freq, channels=2))
  
# Record audio for the given number of seconds
sd.wait(10)
  
# This will convert the NumPy array to an audio
# file with the given sampling frequency
write("recording0.wav", freq, recording)
  
# Convert the NumPy array to audio file

st.markdown("the Audio File Subtitles : ")
st.write(wv.write("recording1.wav", recording, freq, sampwidth=2))

import torch
torch.cuda.is_available()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
audio = whisper.load_audio("recording0.wav")
audio = whisper.pad_or_trim(audio)
mel = whisper.log_mel_spectrogram(audio).to(model.device)
_, probs = model.detect_language(mel)
st.markdown("the language you speak is :")
st.write(f"Detected language: {max(probs, key=probs.get)}")
options = whisper.DecodingOptions(language="en", without_timestamps=True, fp16 = False)
result = whisper.decode(model, mel, options)
import pyttsx3
text_speech=pyttsx3.init()
st.markdown("Translate the audio file into English ")
st.progress(text_speech.say(result.text))
text_speech.runAndWait()
st.write(result.text)