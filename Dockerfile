# Use official lightweight Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=5000

# Set working directory
WORKDIR /app

# Install system dependencies (build-essential for compiling standard wheels if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download NLTK packages and huggingface transformer models during the image build phase
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt'); nltk.download('punkt_tab')"
RUN python -c "from transformers import BertTokenizer, BertForSequenceClassification; BertTokenizer.from_pretrained('prajjwal1/bert-tiny'); BertForSequenceClassification.from_pretrained('prajjwal1/bert-tiny')"

# Copy the application source code
COPY . .

# Create directories for models and data storage
RUN mkdir -p data/raw data/processed models artifacts/plots

# Expose port
EXPOSE 5000

# Run Flask backend with Gunicorn WSGI in production
CMD gunicorn --bind 0.0.0.0:$PORT src.app:app
