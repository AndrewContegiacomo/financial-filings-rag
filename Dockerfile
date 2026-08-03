# Python 3.12, not latest: ML libraries lag new releases by months, and
# Streamlit Cloud defaulting to 3.14 broke the torch import — the same
# failure this image exists to prevent.
FROM python:3.12-slim

WORKDIR /app

# Dependencies first, in their own layer: they change far less often than
# source, so edits to the code don't trigger a 200MB reinstall.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the embedding model into the image. Without this, the first query
# of every fresh container downloads ~90MB from HuggingFace — a cold
# start measured in minutes rather than seconds.
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('all-MiniLM-L6-v2')"

COPY . .

EXPOSE 8501

# Health check gives compose something real to wait on: the process
# starting is not the same as the app being able to serve.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
    CMD python -c "import urllib.request; \
        urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]