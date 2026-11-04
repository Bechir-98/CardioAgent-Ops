FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir chromadb==1.5.9
COPY . .
EXPOSE 8000
HEALTHCHECK --interval=5s --timeout=3s --retries=20 CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v2/heartbeat', timeout=2)" || exit 1
CMD ["chroma", "run", "--path", "/data/chroma", "--host", "0.0.0.0", "--port", "8000"]
