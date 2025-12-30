import whisper
import sys

help_text = """
Usage: python test_audio.py [audio_file] [--help|-h]

If no audio file is provided, the script will transcribe 0.wav.
"""

if len(sys.argv) > 1 and sys.argv[1] in ["--help", "-h"]:
    print(help_text)
    sys.exit(0)

audio_file = sys.argv[1] if len(sys.argv) > 1 else "0.wav"

try:
    model = whisper.load_model("small")
    result = model.transcribe(audio_file)
    print(result["text"])
except Exception as e:
    print(f"Error transcribing audio file: {e}")
    sys.exit(1)
