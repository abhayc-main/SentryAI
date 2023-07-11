import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import sounddevice as sd
from scipy.io import wavfile
import csv
import os


def class_names_from_csv(class_map_csv_text):
    """Returns a list of class names corresponding to the score vector."""
    class_names = []
    with tf.io.gfile.GFile(class_map_csv_text) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            class_names.append(row['display_name'])
    return class_names


def ensure_sample_rate(original_sample_rate, waveform, desired_sample_rate=16000):
    """Resample waveform if required."""
    if original_sample_rate != desired_sample_rate:
        desired_length = int(round(float(len(waveform)) / original_sample_rate * desired_sample_rate))
        waveform = scipy.signal.resample(waveform, desired_length)
    return desired_sample_rate, waveform


def detect_gunshots(waveform):
    sample_rate = 16000  # Desired sample rate for YAMNet model
    if waveform.ndim > 1 and waveform.shape[1] > 1:
        waveform_mono = np.mean(waveform, axis=1)  # Convert stereo to mono
    else:
        waveform_mono = waveform
    sample_rate, waveform_mono = ensure_sample_rate(sample_rate, waveform_mono)
    waveform_mono = waveform_mono / np.max(np.abs(waveform_mono))  # Normalize waveform

    scores, embeddings, spectrogram = model(waveform_mono)
    scores_np = scores.numpy()
    infered_class = class_names[scores_np.mean(axis=0).argmax()]

    return infered_class


def post_process_threshold(scores, threshold):
    detected_classes = []
    for i, score in enumerate(scores):
        if score > threshold:
            detected_classes.append(class_names[i])
    return detected_classes


def test_samples_directory():
    sample_rate = 16000

    # Specify the directory path where the audio samples are located
    samples_directory = 'samples/'

    # Get the list of files in the samples directory
    file_list = os.listdir(samples_directory)

    # Process each file in the directory
    for filename in file_list:
        if filename.endswith('.wav'):
            filepath = os.path.join(samples_directory, filename)
            sample_rate, waveform = wavfile.read(filepath)
            waveform = waveform.astype(np.float32)

            detected_class = detect_gunshots(waveform)
            print(f"File: {filename} | Detected class: {detected_class}")
            if detected_class == "Explosion" or "Gunshot, gunfire" or "Cap gun":
                print("Firearm sound detected")
                
            # Perform post-processing
            scores = detect_gunshots(waveform)
            threshold = 0.5
            detected_classes = post_process_threshold(scores, threshold)
            print("Detected classes:", detected_classes)


# Load the YAMNet model
model = hub.load('https://tfhub.dev/google/yamnet/1')
class_names = class_names_from_csv(model.class_map_path().numpy())

# Call the function to test the samples directory
test_samples_directory()
