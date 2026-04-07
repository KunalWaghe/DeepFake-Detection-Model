FROM python:3.11-slim

# Install system dependencies for OpenCV
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download InceptionV3 weights during build (avoids cold-start download)
RUN python -c "from torchvision.models import inception_v3, Inception_V3_Weights; inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)"

# Copy application code and model
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
