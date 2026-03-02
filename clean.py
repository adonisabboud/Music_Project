import torch
import librosa
import soundfile as sf
from df.enhance import enhance, init_df

input_file = "tests/test_audio/Ya_msaharny.wav"
output_file = "tests/test_audio/Ya_msaharny_DeepFilterNet.wav"

print("1. Loading DeepFilterNet AI...")
model, df_state, _ = init_df()

print(f"2. Loading audio with librosa at {df_state.sr()}Hz...")
y, sr = librosa.load(input_file, sr=df_state.sr())
audio_tensor = torch.from_numpy(y).unsqueeze(0)

print("3. Scrubbing room hiss...")
enhanced = enhance(model, df_state, audio_tensor)

print(f"4. Saving pristine audio to {output_file}...")
sf.write(output_file, enhanced.squeeze().numpy(), sr)
print("Done!")