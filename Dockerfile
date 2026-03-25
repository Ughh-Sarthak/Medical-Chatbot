FROM python:3.12-slim

WORKDIR /app

# Install system dependencies needed for audio processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip uninstall -y pinecone-client pinecone-plugin-inference 2>/dev/null || true
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt
RUN pip install gunicorn

COPY . .

EXPOSE 8080

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--timeout", "120", "app:app"]