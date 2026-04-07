import nbformat as nbf
import os
import sys

# Get the contents of the training script
script_path = os.path.join("training", "train_optimized.py")
with open(script_path, "r", encoding="utf-8") as f:
    script_source = f.read()

# We will break the script into logical cells for the notebook.
# This assumes the script has # ─── section headers
sections = script_source.split("# ───")
imports_config = "# ───" + sections[1].split("_")[0] + "\n" + sections[1] # Approximate splitting, better if we just write the cells directly.

nb = nbf.v4.new_notebook()

# Cell 1: Markdown introduction
markdown_1 = """# DeepFake Detection Model — Optimized PyTorch Pipeline
This notebook serves as the interactive version of `train_optimized.py`. 
It walks through dataset exploration, frame extraction, model building, and training using PyTorch.
"""

# Cell 2: Imports
code_imports = """import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models import inception_v3, Inception_V3_Weights
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[*] Training on: {device.type.upper()}")"""

# Cell 3: Dataset Configuration
code_config = """DATA_DIR = os.path.join("..", "dataset")
REAL_DIR = os.path.join(DATA_DIR, "DFD_original sequences")
FAKE_DIR = os.path.join(DATA_DIR, "DFD_manipulated_sequences")

IMG_SIZE = 224
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-4

print("Dataset folders configured.")"""

# Cell 4: Data Exploration (Real vs Fake Counts)
code_explore = """real_videos = [f for f in os.listdir(REAL_DIR) if f.endswith(('.mp4', '.avi', '.mov'))] if os.path.exists(REAL_DIR) else []
fake_videos = [f for f in os.listdir(FAKE_DIR) if f.endswith(('.mp4', '.avi', '.mov'))] if os.path.exists(FAKE_DIR) else []

labels = ['REAL', 'FAKE']
counts = [len(real_videos), len(fake_videos)]

plt.figure(figsize=(8, 5))
plt.bar(labels, counts, color=['#10b981', '#ef4444'])
plt.title('Dataset Distribution (Real vs Fake)')
plt.ylabel('Number of Videos')
for i, count in enumerate(counts):
    plt.text(i, count + max(counts)*0.01, str(count), ha='center', va='bottom', fontweight='bold')
plt.show()"""

# Cell 5: Visualize some fake video frames
code_vis_frames = """def display_video_frames(video_path, num_frames=6):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        print("Invalid video")
        return
        
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    fig, axes = plt.subplots(1, num_frames, figsize=(20, 5))
    fig.suptitle(f"Frames evenly sampled from: {os.path.basename(video_path)}", fontsize=16)
    
    current_frame = 0
    extracted_frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        if current_frame in frame_indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            extracted_frames.append(frame)
            if len(extracted_frames) == num_frames: break
        current_frame += 1
    cap.release()
    
    for i, ax in enumerate(axes):
        ax.imshow(extracted_frames[i])
        ax.axis('off')
        ax.set_title(f"Frame {frame_indices[i]}")
    plt.show()

# Visualize the first FAKE video available
if fake_videos:
    sample_fake_video = os.path.join(FAKE_DIR, fake_videos[0])
    display_video_frames(sample_fake_video)
else:
    print("No FAKE dataset videos found to visualize.")"""

# Cell 6: Data Loading Logic
code_dataloading = """def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]

def extract_even_frames(video_path, seq_length=MAX_SEQ_LENGTH):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if frame_count < seq_length:
        frame_indices = np.arange(frame_count)
    else:
        frame_indices = np.linspace(0, frame_count - 1, seq_length, dtype=int)
    
    frames = []
    current_frame = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            if current_frame in frame_indices:
                frame = crop_center_square(frame)
                frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                frame = frame.astype(np.float32) / 255.0
                frame[:, :, 0] = (frame[:, :, 0] - 0.485) / 0.229
                frame[:, :, 1] = (frame[:, :, 1] - 0.456) / 0.224
                frame[:, :, 2] = (frame[:, :, 2] - 0.406) / 0.225
                
                frame = np.transpose(frame, (2, 0, 1))
                frames.append(frame)
                if len(frames) == seq_length: break
            current_frame += 1
    finally:
        cap.release()
    return torch.tensor(np.array(frames), dtype=torch.float32)"""

# Cell 7: Model Architecture
code_model = """class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        base_model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        base_model.fc = nn.Identity() 
        self.feature_extractor = base_model
        
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor.eval()

    def forward(self, x):
        with torch.no_grad():
            return self.feature_extractor(x)

class DeepfakeLSTM(nn.Module):
    def __init__(self):
        super(DeepfakeLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=NUM_FEATURES, hidden_size=64, num_layers=1, batch_first=True)
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask):
        lengths = mask.sum(dim=1).cpu()
        lstm_out, (hn, cn) = self.lstm(x)
        last_hidden = hn[-1]
        
        out = self.dropout1(last_hidden)
        out = self.relu(self.fc1(out))
        out = self.dropout2(out)
        out = self.fc2(out)
        return self.sigmoid(out)"""

# Cell 8: Dataset Class
code_dataset = """class DeepfakeDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
        self.extractor = FeatureExtractor().to(device)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        video_path, label = self.data_list[idx]
        frames = extract_even_frames(video_path)
        frames = frames.to(device)
        features = self.extractor(frames)
        
        seq_len = features.shape[0]
        feature_sequence = torch.zeros((MAX_SEQ_LENGTH, NUM_FEATURES))
        feature_sequence[:seq_len] = features.cpu()

        mask = torch.zeros(MAX_SEQ_LENGTH, dtype=torch.bool)
        mask[:seq_len] = True
        
        return feature_sequence, mask, torch.tensor([label], dtype=torch.float32)

all_data = [(os.path.join(REAL_DIR, f), 0.0) for f in real_videos] + \\
           [(os.path.join(FAKE_DIR, f), 1.0) for f in fake_videos]

train_len = int(0.8 * len(all_data))
test_len = len(all_data) - train_len

if len(all_data) > 0:
    train_data, test_data = random_split(all_data, [train_len, test_len])
    train_loader = DataLoader(DeepfakeDataset(train_data), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(DeepfakeDataset(test_data), batch_size=BATCH_SIZE, shuffle=False)
    print(f"[*] Data split complete: {train_len} Training, {test_len} Validation")
else:
    print("[!] No videos found. Cannot create loaders.")"""

# Cell 9: Training Loop (using tqdm.notebook for jupyter visualization)
code_train = """model = DeepfakeLSTM().to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
best_val_loss = float('inf')
early_stop_patience = 5
early_stop_counter = 0

print("\\n🚀 Starting PyTorch Training Loop...")

try:
    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        
        # Use tqdm for neat notebook progress bars
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for features, mask, labels in train_pbar:
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
            train_pbar.set_postfix(loss=loss.item())

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]")
        with torch.no_grad():
            for features, mask, labels in val_pbar:
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

        print(f"Results: Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | Val Acc: {epoch_val_acc:.4f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            # Save weights up a directory into the backend folder so API can see them
            torch.save(model.state_dict(), os.path.join("..", "backend", "deepfake_video_model.pth"))
            print("  ➡️ Saved new best model weights")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_patience:
                print(f"🛑 Early stopping triggered.")
                break
except NameError:
    print("Run the data loading cell first.")"""

# Cell 10: Plot Training History
code_plot = """# Plot Training & Validation Metrics
epochs_range = range(1, len(history['train_loss']) + 1)
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, history['train_acc'], 'bo-', label='Training accuracy')
plt.plot(epochs_range, history['val_acc'], 'ro-', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, history['train_loss'], 'bo-', label='Training loss')
plt.plot(epochs_range, history['val_loss'], 'ro-', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()"""

# Build Notebook
nb.cells.extend([
    nbf.v4.new_markdown_cell(markdown_1),
    nbf.v4.new_code_cell(code_imports),
    nbf.v4.new_code_cell(code_config),
    nbf.v4.new_markdown_cell("## 1. Dataset Exploration"),
    nbf.v4.new_code_cell(code_explore),
    nbf.v4.new_code_cell(code_vis_frames),
    nbf.v4.new_markdown_cell("## 2. Frame Extraction & Deep Model Setup"),
    nbf.v4.new_code_cell(code_dataloading),
    nbf.v4.new_code_cell(code_model),
    nbf.v4.new_markdown_cell("## 3. Data Loaders"),
    nbf.v4.new_code_cell(code_dataset),
    nbf.v4.new_markdown_cell("## 4. PyTorch Training Loop"),
    nbf.v4.new_code_cell(code_train),
    nbf.v4.new_code_cell(code_plot),
])

# Save notebook
out_path = os.path.join("training", "train_optimized.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"Successfully generated {out_path}")
