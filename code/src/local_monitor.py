import sounddevice as sd
import numpy as np
import whisper
import requests
import torch
from collections import deque
from classifier import ToxicityClassifier # Ensure the path is correct

# --- CONFIG ---
FS = 16000
WINDOW_SIZE = 3 
STEP_SIZE = 1 
BOT_URL = "http://127.0.0.1:5000/mute"
YOUR_USER_ID = 120329107438305280

# Strike System
violation_count = 0
MAX_STRIKES = 3

# Pre-fill a buffer with silence
audio_buffer = deque(maxlen=WINDOW_SIZE * FS)
audio_buffer.extend(np.zeros(WINDOW_SIZE * FS))

print("Loading Tiny Model & 3-Tier Classifier...")
model = whisper.load_model("tiny")
classifier = ToxicityClassifier()   

def stream_callback(indata, frames, time, status):
    audio_buffer.extend(indata.flatten())

print(f" Predictive Monitor Active (Target ID: {YOUR_USER_ID})")

with sd.InputStream(samplerate=FS, channels=1, callback=stream_callback):
    while True:
        sd.sleep(STEP_SIZE * 1000) 
        
        current_audio = np.array(audio_buffer).astype(np.float32)
        
        # Volume Check
        if np.max(np.abs(current_audio)) < 0.05:
            continue

        # Fast Transcription
        result = model.transcribe(current_audio, fp16=torch.cuda.is_available(), beam_size=1)
        text = result['text'].strip()
        
        if text:
            # Now returns two values: status and the %
            status, prob = classifier.predict_tier(text)
            
            # Print with percentage for debugging
            print(f" Heard: \"{text}\" | Decision: {status} ({prob*100:.1f}%)")
            
            if status == "Tier 2" or (status == "Tier 1" and prob > 0.70):
                print(f" ACTION: Muting for toxic behavior.")
                requests.post(BOT_URL, json={"user_id": YOUR_USER_ID, "reason": text})
            elif status == "Tier 1":
                print(f" IGNORED: General swearing.")