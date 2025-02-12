import pyaudio
import json
import queue
from vosk import Model, KaldiRecognizer

# Set the path to your downloaded Vosk model
MODEL_PATH = "vosk-model-en-us-0.22-lgraph"

# Load Vosk model
model = Model(MODEL_PATH)
recognizer = KaldiRecognizer(model, 16000)

# Queue to store audio data
audio_queue = queue.Queue()

# Setup PyAudio
p = pyaudio.PyAudio()

def callback(in_data, frame_count, time_info, status):
    """Callback function to process audio."""
    audio_queue.put(in_data)
    return (None, pyaudio.paContinue)

# Open audio stream
stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000,
                input=True, frames_per_buffer=8000, stream_callback=callback)
stream.start_stream()

print("Listening... Speak into the microphone.")

try:
    while True:
        data = audio_queue.get()
        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            print("You said:", result["text"])
except KeyboardInterrupt:
    print("\nStopping...")
    stream.stop_stream()
    stream.close()
    p.terminate()
