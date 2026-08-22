# Single image reused across eval and serve, so what is evaluated is what runs.
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (better layer caching).
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -e .

# Bring in the rest (configs, scripts, sample data).
COPY . .

EXPOSE 8000

# Default: serve the API. Override the command to run eval instead.
CMD ["uvicorn", "bems_rag.serving.main:app", "--host", "0.0.0.0", "--port", "8000"]
