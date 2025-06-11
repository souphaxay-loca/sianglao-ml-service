# Dockerfile for Sianglao ML Service - Development Mode
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directory for models if it doesn't exist
RUN mkdir -p saved_models

# Download models (this will take some time on first build)
RUN python download_models.py

# Expose the port the app runs on
EXPOSE 8000

# Set environment variables for development
ENV FLASK_ENV=development
ENV FLASK_DEBUG=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "app.py"]