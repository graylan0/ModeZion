import numpy as np
import re
import sounddevice as sd
import uuid
from scipy.io.wavfile import write as write_wav
from bark import generate_audio, SAMPLE_RATE
import os
from concurrent.futures import ThreadPoolExecutor

os.environ["SUNO_OFFLOAD_CPU"] = "True"
os.environ["SUNO_USE_SMALL_MODELS"] = "True"

def generate_audio_for_sentence(sentence):
    return generate_audio(sentence, history_prompt="v2/en_speaker_6")

def generate_response(message):
    # Split the message into sentences using a regular expression
    sentences = re.split('(?<=[.!?]) +', message)
    silence = np.zeros(int(0.25 * SAMPLE_RATE))  # quarter second of silence

    pieces = []

    # Use ThreadPoolExecutor to generate audio for each sentence in parallel
    with ThreadPoolExecutor() as executor:
        audio_arrays = list(executor.map(generate_audio_for_sentence, sentences))

    for audio_array in audio_arrays:
        pieces += [audio_array, silence.copy()]

    # Concatenate all audio pieces
    audio = np.concatenate(pieces)

    # Generate a random file name
    file_name = str(uuid.uuid4()) + ".wav"

    # Save the audio to a file in the current directory
    write_wav(file_name, SAMPLE_RATE, audio)

    # Play the audio using sounddevice
    sd.play(audio, samplerate=SAMPLE_RATE)
    sd.wait()  # Wait until audio playback is finished

    print(f"Audio generation completed and saved to {file_name}")

# Test the function with a message
generate_response("I like tacos and I cannot lie, the other robots cannot deny.")

