import torch
import whisper
import pyaudio
import numpy as np
import wave
import threading
import queue

# Constants
FORMAT = pyaudio.paInt16  # Audio format (16-bit)
CHANNELS = 1              # Mono audio
RATE = 16000              # Sampling rate (16kHz)
CHUNK = 1024              # Buffer size (frames per buffer)
SILENCE_THRESHOLD = 500   # Threshold to detect silence
MIN_AUDIO_LENGTH = 1.0    # Minimum audio length to process (in seconds)

# Load Whisper model
model = whisper.load_model("base")  # Use "tiny" or "base" for faster performance

# Queue to hold audio chunks
audio_queue = queue.Queue()

# Function to capture audio from the microphone
def capture_audio():
    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    print("Listening...")

    while True:
        # Read audio data from the stream
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio_queue.put(np.frombuffer(data, dtype=np.int16))

    stream.stop_stream()
    stream.close()
    p.terminate()

# Function to process audio and transcribe
def transcribe_audio():
    while True:
        # Collect audio chunks until silence is detected
        audio_frames = []
        while True:
            if not audio_queue.empty():
                frame = audio_queue.get()
                audio_frames.append(frame)
                # Check for silence
                if np.abs(frame).mean() < SILENCE_THRESHOLD:
                    break

        # Convert frames to a single numpy array
        audio_data = np.concatenate(audio_frames)

        # Skip if the audio is too short
        if len(audio_data) / RATE < MIN_AUDIO_LENGTH:
            continue

        # Transcribe the audio
        result = model.transcribe(audio_data.astype(np.float32) / 32768.0, fp16=False)
        print(f"Transcribed: {result['text']}")

# Start audio capture in a separate thread
audio_thread = threading.Thread(target=capture_audio)
audio_thread.daemon = True
audio_thread.start()

# Start transcription in the main thread
transcribe_audio()