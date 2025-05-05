# Emotion Detection in Conversational AI

This directory contains code and scripts for performing multi-modal emotion detection using text and audio data. The workflow involves two primary steps:

1. **Data Preparation**: Converting MP4 audio segments into WAV format using `convert.py`.
2. **Model Training and Evaluation**: Running `train.py` to train and compare different fusion strategies (text-only, early fusion, and gated fusion) using the MELD dataset.

## Repository Contents

- **convert.py**: A script that converts MP4 files into WAV format at 16 kHz mono.  
- **train.py**: A training script that:
  - Loads and preprocesses the MELD dataset (text and audio).
  - Trains three different models:
    - Text-Only model (baseline)
    - Early Fusion model (concatenation of text and audio embeddings)
    - Gated Fusion model (adaptive weighting of text and audio features)
  - Evaluates the trained models on validation and test sets.
  - Reports performance metrics (Accuracy and Weighted F1) and inference times.

These scripts assume the MELD dataset has been downloaded and placed alongside them in the same directory.

## Requirements

- Python 3.7+
- PyTorch (with GPU support recommended)
- Hugging Face Transformers
- Torchaudio
- Datasets (Hugging Face)
- NumPy, Pandas, TQDM
- ffmpeg (for audio conversion)
- (Optional) Google Colab environment for GPU and convenience

You can install the main dependencies with:
```bash
pip install transformers torchaudio einops datasets tqdm matplotlib

