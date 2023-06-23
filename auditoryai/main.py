import tensorflow as tf
import tensorflow_hub as hub

# Load the YAMNet model from TensorFlow Hub
model_url = "https://tfhub.dev/google/yamnet/1"
model = hub.load(model_url)

audio_file = "samples/GunShotSnglShotIn PE1097906.wav"

def detect_gunshots(audio_data, sample_rate):
    # Resample the audio to 16 kHz if needed and convert to mono
    if sample_rate != 16000:
        audio_data = tf.audio.resample(audio_data, sample_rate, 16000)
    audio_data = tf.squeeze(audio_data)
    
    # Expand dimensions to match the expected shape of YAMNet
    audio_data = tf.expand_dims(audio_data, axis=-1)
    
    # Run inference on YAMNet model
    scores, embeddings, _ = model(audio_data)
    
    # Get the index of the "gunshot" class
    gunshot_index = tf.constant(404)
    
    # Check if the score for the "gunshot" class is above the threshold
    gunshot_score = scores[0][gunshot_index].numpy()
    threshold = 0.5  # Adjust the threshold as needed
    if gunshot_score > threshold:
        print("Gunshot detected!")
    else:
        print("No gunshot detected.")

# Example usage:
# audio_data: The audio waveform data as a TensorFlow tensor
# sample_rate: The sample rate of the audio data
detect_gunshots(audio_data, sample_rate)
