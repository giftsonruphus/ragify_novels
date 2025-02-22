FROM python:3.12-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY app/pyproject.toml .
RUN pip install --upgrade pip && pip install poetry && poetry install --no-root

# Copy application code
COPY app/ .

CMD ["poetry", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
