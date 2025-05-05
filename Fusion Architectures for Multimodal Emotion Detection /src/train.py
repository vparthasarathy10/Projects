!pip install transformers torchaudio einops datasets tqdm matplotlib --quiet

import os
import pandas as pd
import numpy as np
import torch
import torchaudio
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import BertTokenizer, BertModel, Wav2Vec2Processor, Wav2Vec2Model
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import random
import time

from google.colab import drive
drive.mount('/content/drive')

DATA_ROOT = '/content/drive/MyDrive'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_csv_data(path):
    df = pd.read_csv(path)
    return df

def split_dev_into_val_test_by_dialogue(dev_df, split_ratio=0.5):
    unique_dialogues = dev_df['Dialogue_ID'].unique()
    unique_dialogues = np.sort(unique_dialogues)
    half = int(len(unique_dialogues) * split_ratio)
    val_dialogues = unique_dialogues[:half]
    test_dialogues = unique_dialogues[half:]

    val_df = dev_df[dev_df['Dialogue_ID'].isin(val_dialogues)].reset_index(drop=True)
    test_df = dev_df[dev_df['Dialogue_ID'].isin(test_dialogues)].reset_index(drop=True)
    return val_df, test_df

train_df = load_csv_data(os.path.join(DATA_ROOT, "train_sent_emo.csv"))
dev_df = load_csv_data(os.path.join(DATA_ROOT, "dev_sent_emo.csv"))

# Remove problematic utterances
train_df = train_df[~((train_df['Dialogue_ID'] == 125) & (train_df['Utterance_ID'] == 3))]
dev_df = dev_df[~((dev_df['Dialogue_ID'] == 110) & (dev_df['Utterance_ID'] == 7))]


val_df, test_df = split_dev_into_val_test_by_dialogue(dev_df, split_ratio=0.5)


train_wav_folder = os.path.join(DATA_ROOT, "train_wav")
dev_wav_folder = os.path.join(DATA_ROOT, "dev_wav")

emotion_list = ['neutral', 'joy', 'surprise', 'anger', 'sadness', 'disgust', 'fear']
emotion2id = {e: i for i, e in enumerate(emotion_list)}

def get_wav_path(split, did, uid):
    if split == 'train':
        return os.path.join(train_wav_folder, f"dia{did}_utt{uid}.wav")
    elif split == 'val':
        return os.path.join(dev_wav_folder, f"dia{did}_utt{uid}.wav")
    else:
        return os.path.join(dev_wav_folder, f"dia{did}_utt{uid}.wav")

class MELDDataset(Dataset):
    def __init__(self, df, split, tokenizer, max_text_len=64, augment_audio=False):
        self.df = df.reset_index(drop=True)
        self.split = split
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.augment_audio = augment_audio

    def __len__(self):
        return len(self.df)

    def audio_augment(self, audio):
        if random.random() < 0.5:
            gain = random.uniform(0.8, 1.2)
            audio = audio * gain
        return audio

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        did = row['Dialogue_ID']
        uid = row['Utterance_ID']
        text = row['Utterance']
        emo = row['Emotion']
        label = emotion2id[emo]

        # Process text here
        text_enc = self.tokenizer(text, truncation=True, max_length=self.max_text_len,
                                  padding='max_length', return_tensors='pt')

        # Load raw audio
        wav_path = get_wav_path(self.split, did, uid)
        speech, sr = torchaudio.load(wav_path)
        if self.augment_audio and self.split == 'train':
            speech = self.audio_augment(speech)

        # Return raw speech and text encodings; processor will be in collate_fn
        return {
            'input_ids': text_enc['input_ids'].squeeze(0),
            'attention_mask': text_enc['attention_mask'].squeeze(0),
            'speech': speech.squeeze(0),  # raw waveform
            'label': torch.tensor(label, dtype=torch.long)
        }

text_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
audio_processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')

def custom_collate_fn(batch):
    # batch is a list of dicts
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    labels = [item['label'] for item in batch]
    speech_list = [item['speech'].numpy() for item in batch]  # list of numpy arrays

    # Process audio as a batch
    audio_enc = audio_processor(speech_list, sampling_rate=16000, return_tensors='pt', padding=True)

    # Stack text and labels
    input_ids = torch.stack(input_ids, dim=0)
    attention_mask = torch.stack(attention_mask, dim=0)
    labels = torch.stack(labels, dim=0)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'audio_input_values': audio_enc['input_values'],
        'label': labels
    }

train_dataset = MELDDataset(train_df, 'train', text_tokenizer, augment_audio=True)
val_dataset   = MELDDataset(val_df, 'val', text_tokenizer)
test_dataset  = MELDDataset(test_df, 'test', text_tokenizer)

# Class imbalance
train_labels = train_df['Emotion'].map(lambda e: emotion2id[e]).values
class_counts = np.bincount(train_labels)
class_weights = 1.0 / class_counts
weights = class_weights[train_labels]
train_sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=8, sampler=train_sampler, num_workers=2, collate_fn=custom_collate_fn)
val_loader   = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2, collate_fn=custom_collate_fn)
test_loader  = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2, collate_fn=custom_collate_fn)

class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_model = BertModel.from_pretrained('bert-base-uncased')
    def forward(self, input_ids, attention_mask):
        out = self.text_model(input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:,0,:]

class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.audio_model = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h')
    def forward(self, input_values):
        out = self.audio_model(input_values).last_hidden_state
        audio_emb = out.mean(dim=1)
        return audio_emb

class TextOnlyModel(nn.Module):
    def __init__(self, hidden_dim=768, num_classes=len(emotion_list)):
        super().__init__()
        self.text_encoder = TextEncoder()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask, audio_input_values=None):
        h_text = self.text_encoder(input_ids, attention_mask)
        h = self.relu(self.fc1(h_text))
        logits = self.fc2(h)
        return logits

class EarlyFusionModel(nn.Module):
    def __init__(self, hidden_dim=768, num_classes=len(emotion_list)):
        super().__init__()
        self.text_encoder = TextEncoder()
        self.audio_encoder = AudioEncoder()
        self.fc1 = nn.Linear(hidden_dim*2, hidden_dim*2)
        self.fc2 = nn.Linear(hidden_dim*2, num_classes)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask, audio_input_values):
        h_text = self.text_encoder(input_ids, attention_mask)
        h_audio = self.audio_encoder(audio_input_values)
        h = torch.cat([h_text, h_audio], dim=-1)
        h = self.relu(self.fc1(h))
        h = self.relu(self.fc2(h))
        return h

class GatedFusionModel(nn.Module):
    def __init__(self, hidden_dim=768, num_classes=len(emotion_list)):
        super().__init__()
        self.text_encoder = TextEncoder()
        self.audio_encoder = AudioEncoder()

        self.gate = nn.Linear(hidden_dim*2, hidden_dim)
        self.sigmoid = nn.Sigmoid()

        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask, audio_input_values):
        h_text = self.text_encoder(input_ids, attention_mask)
        h_audio = self.audio_encoder(audio_input_values)

        gate_input = torch.cat([h_text, h_audio], dim=-1)
        g = self.sigmoid(self.gate(gate_input))

        fused = g * h_text + (1 - g) * h_audio
        h = self.relu(self.fc1(fused))
        logits = self.fc2(h)
        return logits

def evaluate_model(model, loader):
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            audio_input_values = batch['audio_input_values'].to(device)
            labels = batch['label'].to(device)

            if isinstance(model, TextOnlyModel):
                logits = model(input_ids, attention_mask)
            else:
                logits = model(input_ids, attention_mask, audio_input_values)

            pred = torch.argmax(logits, dim=-1)
            preds.extend(pred.cpu().tolist())
            trues.extend(labels.cpu().tolist())
    acc = accuracy_score(trues, preds)
    f1 = f1_score(trues, preds, average='weighted')
    return acc, f1

def train_model(model, train_loader, val_loader, epochs=3, lr=2e-5):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    best_val_f1 = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            audio_input_values = batch['audio_input_values'].to(device)
            labels = batch['label'].to(device)

            if isinstance(model, TextOnlyModel):
                logits = model(input_ids, attention_mask)
            else:
                logits = model(input_ids, attention_mask, audio_input_values)

            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        val_acc, val_f1 = evaluate_model(model, val_loader)

        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f} Val Acc={val_acc:.4f} Val F1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), "best_model.pt")

    model.load_state_dict(torch.load("best_model.pt"))
    return model

def measure_inference_time(model, loader, repeat=3):
    model.eval()
    start = time.time()
    for _ in range(repeat):
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            audio_input_values = batch['audio_input_values'].to(device)
            if isinstance(model, TextOnlyModel):
                _ = model(input_ids, attention_mask)
            else:
                _ = model(input_ids, attention_mask, audio_input_values)
    end = time.time()
    total_time = end - start
    avg_time_per_epoch = total_time / repeat
    return avg_time_per_epoch

text_model = TextOnlyModel()
early_model = EarlyFusionModel()
gated_model = GatedFusionModel()

print("Training Text-Only Model...")
text_model = train_model(text_model, train_loader, val_loader, epochs=3)
val_acc_t, val_f1_t = evaluate_model(text_model, val_loader)
test_acc_t, test_f1_t = evaluate_model(text_model, test_loader)
text_inference_time = measure_inference_time(text_model, val_loader)
print(f"Text-Only: Val Acc={val_acc_t:.4f}, Val F1={val_f1_t:.4f}, Test Acc={test_acc_t:.4f}, Test F1={test_f1_t:.4f}, Inference Time={text_inference_time:.4f}s")

print("Training Early Fusion Model...")
early_model = train_model(early_model, train_loader, val_loader, epochs=3)
val_acc_e, val_f1_e = evaluate_model(early_model, val_loader)
test_acc_e, test_f1_e = evaluate_model(early_model, test_loader)
early_inference_time = measure_inference_time(early_model, val_loader)
print(f"Early Fusion: Val Acc={val_acc_e:.4f}, Val F1={val_f1_e:.4f}, Test Acc={test_acc_e:.4f}, Test F1={test_f1_e:.4f}, Inference Time={early_inference_time:.4f}s")

print("Training Gated Fusion Model...")
gated_model = train_model(gated_model, train_loader, val_loader, epochs=3)
val_acc_g, val_f1_g = evaluate_model(gated_model, val_loader)
test_acc_g, test_f1_g = evaluate_model(gated_model, test_loader)
gated_inference_time = measure_inference_time(gated_model, val_loader)
print(f"Gated Fusion: Val Acc={val_acc_g:.4f}, Val F1={val_f1_g:.4f}, Test Acc={test_acc_g:.4f}, Test F1={test_f1_g:.4f}, Inference Time={gated_inference_time:.4f}s")