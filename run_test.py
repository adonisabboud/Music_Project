import os
from transcribe import transcribe
import traceback
# 1. Type the exact name of your audio file here:
AUDIO_FILE = "tests/test_audio/Ya_msaharny.wav"


def main():
    if not os.path.exists(AUDIO_FILE):
        print(f"❌ Error: Could not find '{AUDIO_FILE}'")
        return

    print(f"🎵 Starting MVP Transcription for: {AUDIO_FILE}...\n")

    try:
        # 2. Call your newly wired pipeline with the output_path!
        output_xml = transcribe(
            audio_path=AUDIO_FILE,
            output_path="Ya_msaharny_transcription.musicxml",  # <-- Added this line!
            maqam_override="Rast on C"
        )

        print(f"\n✅ Success! Transcription complete.")
        print(f"📂 Output saved to: {output_xml}")
        print("🎹 Next Step: Drag and drop that .xml file into MuseScore or Sibelius!")

    except Exception as e:
        print(f"\n❌ An error occurred during transcription:\n{e}")
        traceback.print_exc()  # This will print the exact line if it fails again!


if __name__ == "__main__":
    main()