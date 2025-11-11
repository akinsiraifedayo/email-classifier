FROM python:3.10-slim AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git wget && \
    rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --target=/app/deps
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

FROM python:3.10-slim
WORKDIR /app
COPY --from=builder /app/deps /usr/local/lib/python3.10/site-packages
COPY . .
RUN useradd -m appuser && chown -R appuser /app
USER appuser
EXPOSE 5000
ENV PORT=5000
ENV PYTHONUNBUFFERED=1
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5000", "email_tagger:app"]
