import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import sounddevice as sd
from scipy.io import wavfile

# Load the YAMNet model
model = hub.load('https://tfhub.dev/google/yamnet/1')
class_names = class_names_from_csv(model.class_map_path().numpy())

def ensure_sample_rate(original_sample_rate, waveform, desired_sample_rate=16000):
    """Resample waveform if required."""
    if original_sample_rate != desired_sample_rate:
        desired_length = int(round(float(len(waveform)) / original_sample_rate * desired_sample_rate))
        waveform = scipy.signal.resample(waveform, desired_length)
    return desired_sample_rate, waveform

def detect_gunshots(waveform):
    sample_rate = 16000  # Desired sample rate for YAMNet model
    sample_rate, waveform = ensure_sample_rate(sample_rate, waveform)
    waveform = waveform / np.max(np.abs(waveform))  # Normalize waveform

    scores, embeddings, spectrogram = model(waveform)
    scores_np = scores.numpy()
    infered_class = class_names[scores_np.mean(axis=0).argmax()]

    return infered_class

def audio_callback(indata, frames, time, status):
    if status:
        print("Error:", status)
        return

    waveform = indata.flatten().astype(np.float32)
    detected_class = detect_gunshots(waveform)
    print("Detected class:", detected_class)

def listen_and_detect():
    sample_rate = 16000
    duration = 10  # Number of seconds to listen for
    num_samples = sample_rate * duration

    print("Listening for audio...")
    with sd.InputStream(device=0, channels=1, samplerate=sample_rate, callback=audio_callback):
        sd.sleep(duration * 1000)
    print("Audio capture complete.")

listen_and_detect()
