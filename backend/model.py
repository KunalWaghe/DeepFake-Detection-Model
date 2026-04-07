"""
DeepFake Detection — ML Inference Module (PyTorch)
Encapsulates InceptionV3 feature extraction and sequence model prediction.
"""

import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision.models import inception_v3, Inception_V3_Weights

# ─── Constants ───────────────────────────────────────────────────────────────
IMG_SIZE = 224
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048

# ─── PyTorch Models ─────────────────────────────────────────────────────────
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # Load InceptionV3 without the classification head
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
        # We need lengths to find the last valid output in the sequence
        # For simplicity in inference, since we pad up to max seq length
        # we can just use the final LSTM output 
        lstm_out, (hn, cn) = self.lstm(x)
        last_hidden = hn[-1]
        
        out = self.dropout1(last_hidden)
        out = self.relu(self.fc1(out))
        out = self.dropout2(out)
        out = self.fc2(out)
        return self.sigmoid(out)

class DeepfakeDetector:
    """Handles model loading, video processing, and PyTorch deepfake prediction."""

    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), "deepfake_video_model.pth")
        self.model_path = model_path
        
        # We force CPU for inference to use less RAM on the Free Tier cloud host
        self.device = torch.device('cpu') 
        
        self.feature_extractor = None
        self.sequence_model = None

    def load(self):
        """Load the PyTorch sequence model weights and build the InceptionV3 feature extractor."""
        print("[*] Loading PyTorch deepfake detection model...")
        
        # Load feature extractor
        self.feature_extractor = FeatureExtractor().to(self.device)
        self.feature_extractor.eval()
        
        # Load sequence model
        self.sequence_model = DeepfakeLSTM().to(self.device)
        if os.path.exists(self.model_path):
            self.sequence_model.load_state_dict(torch.load(self.model_path, map_location=self.device, weights_only=True))
            print("[OK] Sequence model weights loaded successfully!")
        else:
            print(f"[!] Warning: No trained weights found at {self.model_path}.")
            print("    The model will use random weights. Please train the model first.")
            
        self.sequence_model.eval()

    @staticmethod
    def _crop_center_square(frame: np.ndarray) -> np.ndarray:
        """Crop the center square from a frame."""
        y, x = frame.shape[0:2]
        min_dim = min(y, x)
        start_x = (x // 2) - (min_dim // 2)
        start_y = (y // 2) - (min_dim // 2)
        return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]

    @staticmethod
    def _extract_even_frames(video_path: str, seq_length: int = MAX_SEQ_LENGTH) -> torch.Tensor:
        """Extract perfectly spaced frames across the whole video."""
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if frame_count <= 0:
            return torch.tensor([])

        if frame_count < seq_length:
            frame_indices = np.arange(frame_count)
        else:
            frame_indices = np.linspace(0, frame_count - 1, seq_length, dtype=int)
        
        frames = []
        current_frame = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if current_frame in frame_indices:
                    frame = DeepfakeDetector._crop_center_square(frame)
                    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    frame = frame.astype(np.float32) / 255.0
                    frame[:, :, 0] = (frame[:, :, 0] - 0.485) / 0.229
                    frame[:, :, 1] = (frame[:, :, 1] - 0.456) / 0.224
                    frame[:, :, 2] = (frame[:, :, 2] - 0.406) / 0.225
                    
                    frame = np.transpose(frame, (2, 0, 1))
                    frames.append(frame)
                    
                    if len(frames) == seq_length:
                        break
                current_frame += 1
        finally:
            cap.release()
            
        if len(frames) == 0:
            return torch.tensor([])
            
        return torch.tensor(np.array(frames), dtype=torch.float32)

    def predict(self, video_path: str) -> dict:
        """
        Run deepfake detection on a video file.
        Returns: {"result": "FAKE"|"REAL", "confidence": float}
        """
        if self.sequence_model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # 1. Extract frames evenly from video
        frames = self._extract_even_frames(video_path)
        if len(frames) == 0:
            raise ValueError("Could not extract any frames from the video.")

        # 2. Extract features using InceptionV3
        frames = frames.to(self.device)
        with torch.no_grad():
            features = self.feature_extractor(frames) # Shape: (T, 2048)

        # 3. Create sequence tensor and mask
        seq_len = features.shape[0]
        feature_sequence = torch.zeros((1, MAX_SEQ_LENGTH, NUM_FEATURES), device=self.device)
        feature_sequence[0, :seq_len] = features

        mask = torch.zeros((1, MAX_SEQ_LENGTH), dtype=torch.bool, device=self.device)
        mask[0, :seq_len] = True

        # 4. Predict
        with torch.no_grad():
            prediction = self.sequence_model(feature_sequence, mask).item()

        confidence = float(prediction)
        result = "FAKE" if confidence >= 0.51 else "REAL"

        return {
            "result": result,
            "confidence": round(confidence * 100, 2),  # As percentage
        }
