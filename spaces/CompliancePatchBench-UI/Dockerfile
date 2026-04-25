FROM python:3.13.5-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
COPY src/ ./src/

RUN pip3 install --no-cache-dir -r requirements.txt

# Hugging Face Spaces set PORT; default 7860 matches the API Space pattern.
EXPOSE 7860
HEALTHCHECK CMD curl -f http://127.0.0.1:7860/_stcore/health || exit 0

CMD ["sh", "-c", "streamlit run src/streamlit_app.py --server.port=${PORT:-7860} --server.address=0.0.0.0 --server.headless=true"]
