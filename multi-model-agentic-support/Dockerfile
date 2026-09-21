FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HOME=/app/cache/huggingface
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && useradd --create-home --uid 10001 agent
COPY src ./src
COPY scripts ./scripts
COPY data ./data
COPY reports ./reports
RUN mkdir -p /app/models /app/runtime /app/cache && chown -R agent:agent /app
USER agent
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)"
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
