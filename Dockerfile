# Works on Hugging Face Spaces (Docker), Render, Railway, Google Cloud Run, Azure, etc.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PORT=8501
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
# Pre-download the embedding model at build time so the first visitor doesn't
# wait for it (the knowledge base itself is built automatically on first load).
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5', cache_dir='storage/models')" || true

EXPOSE 8501
CMD streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0 --server.headless=true
