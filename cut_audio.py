from pydub import AudioSegment
import random

# Load your audio file (ensure the format is supported, like mp3, wav, etc.)
audio_file = "003.mp3"  # Replace with your file path
output_file = "cut_7s_audio.wav"  # The output file where you want the 7-second cut

# Load the audio file
audio = AudioSegment.from_file(audio_file)

# Total length of the audio in milliseconds (10 minutes = 600,000 milliseconds)
audio_length = len(audio)

# Ensure the audio is long enough (more than 7 seconds)
if audio_length >= 7000:
    # Choose a random start point for the 7 seconds
    start_time = random.randint(0, audio_length - 7000)  # Random start within the first 10 minutes
    end_time = start_time + 7000  # 7 seconds after the start time

    # Extract the 7 seconds of audio
    audio_7s = audio[start_time:end_time]

    # Export the 7-second audio to a new file
    audio_7s.export(output_file, format="wav")
    print(f"7-second segment extracted and saved as {output_file}")
else:
    print("Audio is shorter than 7 seconds!")
