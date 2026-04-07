# DeepFake Detector — AI-Powered Video Analysis

An AI-powered web application that analyzes videos to determine if they are authentic or deepfakes. The system extracts video frames, processes them using a pre-trained **InceptionV3** network for feature extraction, and feeds them into an **LSTM** sequence model built with **PyTorch** to make a final prediction.

## Features
- **Video Analysis**: Upload videos (MP4, AVI, MOV, MKV, WebM) to get a FAKE or REAL verdict.
- **Deep Learning Pipeline**: Combines InceptionV3 (feature extraction) with an LSTM (sequence modeling).
- **Modern Web Interface**: Beautiful, responsive frontend with drag-and-drop support and animations.
- **FastAPI Backend**: Efficient, asynchronous REST API for handling video uploads and model inference.

## Tech Stack
- **Frontend**: HTML5, CSS3 (Vanilla), JavaScript
- **Backend API**: FastAPI, Uvicorn, Python-Multipart
- **Machine Learning**: PyTorch, TorchVision (CPU-optimized for inference)
- **Computer Vision**: OpenCV (`opencv-python-headless`)

## Project Structure
```text
Deep-Fake-Detection-AI-Model/
├── backend/
│   ├── main.py                # FastAPI application & REST endpoints
│   ├── model.py               # DeepfakeDetector class, InceptionV3 + LSTM PyTorch models
│   ├── requirements.txt       # Backend dependencies
│   ├── Dockerfile             # Docker container definition
│   └── deepfake_video_model.pth # Trained model weights (generated via training)
├── frontend/
│   ├── index.html             # Main web interface
│   ├── styles.css             # UI styling & animations
│   └── script.js              # UI interaction logic and API requests
├── training/
│   ├── train_optimized.py     # Script to train the LSTM on video features
│   └── train_optimized.ipynb  # Jupyter notebook alternative for training
├── dataset/                   # Directory for training datasets (e.g. DFDC)
└── generate_notebook.py       # Helper utility script
```

## Setup & Installation

### 1. Prerequisites
- Python 3.8 or higher
- Git

### 2. Backend Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/KunalWaghe/Deep-Fake-Detection-Model.git
   cd Deep-Fake-Detection-Model/backend
   ```
2. Create and activate a virtual environment:
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the FastAPI backend server:
   ```bash
   python main.py
   # Or using uvicorn directly:
   # uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```
   The API will now be running at `http://localhost:8000`.

### 3. Frontend Setup
You can simply open `frontend/index.html` in your preferred web browser, or use a local development server:
```bash
cd ../frontend
python -m http.server 3000
```
Then navigate to `http://localhost:3000` in your web browser.

## Training the Model
If you want to train the model from scratch on your own dataset:
1. Download a deepfake dataset (e.g., DFDC) and organize it according to this structure:
   ```text
   dataset/
   ├── DFD_original sequences/     # Real, authentic videos
   └── DFD_manipulated_sequences/  # Fake, deepfake videos
   ```
2. Run the optimized training script:
   ```bash
   cd training
   python train_optimized.py
   ```
3. The script will automatically save the best performing model weights to `backend/deepfake_video_model.pth`.

## Contact & Author
- **Author**: Kunal Waghe
- **GitHub**: [KunalWaghe](https://github.com/KunalWaghe)
- **Issues**: Report bugs or request features at [Issues](https://github.com/KunalWaghe/Deep-Fake-Detection-Model/issues).