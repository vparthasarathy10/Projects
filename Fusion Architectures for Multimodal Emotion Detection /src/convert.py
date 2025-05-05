import os
import subprocess

def convert_directory(mp4_folder, wav_folder):
    if not os.path.exists(wav_folder):
        os.makedirs(wav_folder)
    for file_name in os.listdir(mp4_folder):
        if file_name.endswith('.mp4'):
            base_name = os.path.splitext(file_name)[0]  # e.g. dia0_utt0
            mp4_path = os.path.join(mp4_folder, file_name)
            wav_path = os.path.join(wav_folder, base_name + '.wav')
            
            # ffmpeg command: convert to 16kHz mono WAV
            cmd = ["ffmpeg", "-y", "-i", mp4_path, "-ar", "16000", "-ac", "1", wav_path]
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"Converted {mp4_path} to {wav_path}")

# Adjust these paths to point to the local directories containing MP4 files.
train_mp4_folder = "./train_splits"  # Directory with train mp4
train_wav_folder = "./train_wav"     # Directory to save train wavs

dev_mp4_folder = "./dev_splits_complete" # Directory with dev mp4
dev_wav_folder = "./dev_wav"             # Directory to save dev wavs

convert_directory(train_mp4_folder, train_wav_folder)
convert_directory(dev_mp4_folder, dev_wav_folder)

print("Conversion complete!")
