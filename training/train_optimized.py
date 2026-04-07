import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models import inception_v3, Inception_V3_Weights
from tqdm import tqdm
import matplotlib.pyplot as plt

# ─── Configuration ────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
REAL_DIR = os.path.join(DATA_DIR, "DFD_original sequences")
FAKE_DIR = os.path.join(DATA_DIR, "DFD_manipulated_sequences")

IMG_SIZE = 224
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-4

# Auto-detect GPU for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[*] Training on: {device.type.upper()}")

# ─── Frame Processing Functions ───────────────────────────────────────────
def crop_center_square(frame):
    """Crop the largest possible center square from a frame."""
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]

def extract_even_frames(video_path, seq_length=MAX_SEQ_LENGTH):
    """
    Extract perfectly spaced frames across the whole video instead of just the first N frames.
    Prevents the bug where AI only looked at the first 0.6 seconds of a deepfake.
    """
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if frame_count < seq_length:
        # If video is extremely short, take whatever we can
        frame_indices = np.arange(frame_count)
    else:
        # Sample evenly spaced frames across the entire video
        frame_indices = np.linspace(0, frame_count - 1, seq_length, dtype=int)
    
    frames = []
    current_frame = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if current_frame in frame_indices:
                frame = crop_center_square(frame)
                frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Normalize frame to [0, 1] then preprocess using InceptionV3 standards
                frame = frame.astype(np.float32) / 255.0
                
                # InceptionV3 standard PyTorch ImageNet Normalization
                # Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225]
                frame[:, :, 0] = (frame[:, :, 0] - 0.485) / 0.229
                frame[:, :, 1] = (frame[:, :, 1] - 0.456) / 0.224
                frame[:, :, 2] = (frame[:, :, 2] - 0.406) / 0.225
                
                # PyTorch wants Channels-First: (C, H, W)
                frame = np.transpose(frame, (2, 0, 1))
                frames.append(frame)
                
                if len(frames) == seq_length:
                    break
            current_frame += 1
    finally:
        cap.release()
        
    return torch.tensor(np.array(frames), dtype=torch.float32)

# ─── Feature Extractor ──────────────────────────────────────────────────
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # Load InceptionV3 without the classification head
        base_model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        base_model.fc = nn.Identity()  # Remove fully connected layer
        
        self.feature_extractor = base_model
        
        # Freeze Inception weights (we only want to train the Sequence model)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        self.feature_extractor.eval() # Ensure dropout & batchnorm are frozen

    def forward(self, x):
        with torch.no_grad():
            return self.feature_extractor(x)

# ─── Dataset & DataLoader (RAM Friendly) ────────────────────────────────
class DeepfakeDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
        self.extractor = FeatureExtractor().to(device)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        video_path, label = self.data_list[idx]
        
        # 1. Load the raw frames (T, C, H, W)
        frames = extract_even_frames(video_path)
        
        # 2. Extract features using InceptionV3
        frames = frames.to(device)
        features = self.extractor(frames) # Shape: (T, 2048)
        
        # 3. Create padding mask (1 for valid frame, 0 for pad)
        # If video was shorter than MAX_SEQ_LENGTH, pad it
        seq_len = features.shape[0]
        feature_sequence = torch.zeros((MAX_SEQ_LENGTH, NUM_FEATURES))
        feature_sequence[:seq_len] = features.cpu()

        mask = torch.zeros(MAX_SEQ_LENGTH, dtype=torch.bool)
        mask[:seq_len] = True
        
        return feature_sequence, mask, torch.tensor([label], dtype=torch.float32)

def gather_dataset():
    """Scans datset folders and assigns labels (0 = REAL, 1 = FAKE)"""
    data = []
    # Real
    if os.path.exists(REAL_DIR):
        print(f"[*] Scanning {REAL_DIR}...")
        for f in os.listdir(REAL_DIR):
            if f.endswith(('.mp4', '.avi', '.mov')):
                data.append((os.path.join(REAL_DIR, f), 0.0))
                
    # Fake
    if os.path.exists(FAKE_DIR):
        print(f"[*] Scanning {FAKE_DIR}...")
        for f in os.listdir(FAKE_DIR):
            if f.endswith(('.mp4', '.avi', '.mov')):
                data.append((os.path.join(FAKE_DIR, f), 1.0))
    return data

# ─── Sequence Model (Simplified LSTM) ───────────────────────────────────
class DeepfakeLSTM(nn.Module):
    def __init__(self):
        super(DeepfakeLSTM, self).__init__()
        # 64-unit LSTM replaces the overly deep, complex 3-layer GRU
        self.lstm = nn.LSTM(input_size=NUM_FEATURES, hidden_size=64, num_layers=1, batch_first=True)
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask):
        # We need lengths for PyTorch packed sequences, derived from mask
        # mask is (Batch, SeqLen) where True = valid. Sum along dim 1 = length.
        lengths = mask.sum(dim=1).cpu()
        
        # Forward pass through LSTM
        lstm_out, (hn, cn) = self.lstm(x)
        
        # We want the output of the LAST valid timestep for each video in the batch
        # hn[-1] contains the final hidden state of the LSTM sequence!
        last_hidden = hn[-1]
        
        out = self.dropout1(last_hidden)
        out = self.relu(self.fc1(out))
        out = self.dropout2(out)
        out = self.fc2(out)
        return self.sigmoid(out)

# ─── Training Loop ──────────────────────────────────────────────────────
def train_model():
    print("[*] Gathering videos from disk...")
    all_data = gather_dataset()
    
    if len(all_data) == 0:
        print("[!] No videos found in dataset/ folder. Please extract videos to run training.")
        return
        
    print(f"[*] Found {len(all_data)} total videos.")
    
    # 80/20 train/test random split (Ideally group by original, but random for now)
    train_len = int(0.8 * len(all_data))
    test_len = len(all_data) - train_len
    train_data, test_data = random_split(all_data, [train_len, test_len])
    
    print(f"[*] Split: {train_len} Training, {test_len} Validation")

    train_loader = DataLoader(DeepfakeDataset(train_data), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(DeepfakeDataset(test_data), batch_size=BATCH_SIZE, shuffle=False)

    model = DeepfakeLSTM().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    best_val_loss = float('inf')
    early_stop_patience = 5
    early_stop_counter = 0

    print("\n🚀 Starting Training...")
    for epoch in range(EPOCHS):
        # TRAINING
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        
        for features, mask, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
            features, mask, labels = features.to(device), mask.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features, mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * features.size(0)
            preds = (outputs >= 0.51).float()
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total
        
        # VALIDATION
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        
        with torch.no_grad():
            for features, mask, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                features, mask, labels = features.to(device), mask.to(device), labels.to(device)
                
                outputs = model(features, mask)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * features.size(0)
                preds = (outputs >= 0.51).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total

        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_acc'].append(epoch_val_acc)

        print(f"Epoch {epoch+1} Results: Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | Val Acc: {epoch_val_acc:.4f}")

        # Checkpointing & Early Stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), "..", "backend", "deepfake_video_model.pth"))
            print("  ➡️ Saved new best model to backend/deepfake_video_model.pth")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_patience:
                print(f"🛑 Early stopping triggered after {epoch+1} epochs.")
                break

    # Save metrics plot
    save_plot(history)
    print("\n✅ Training Complete!")

def save_plot(history):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(14, 5))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_acc'], 'bo-', label='Training accuracy')
    plt.plot(epochs, history['val_acc'], 'ro-', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_loss'], 'bo-', label='Training loss')
    plt.plot(epochs, history['val_loss'], 'ro-', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plot_path = os.path.join(os.path.dirname(__file__), 'training_history.png')
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"[*] Training history chart saved to: {plot_path}")

if __name__ == "__main__":
    train_model()
