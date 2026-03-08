FROM python:3.11-slim

# Avoid buffered output (better logs on HF Spaces)
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install Python dependencies first (Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Install the hypothesis-engine package
RUN pip install --no-cache-dir -e .

# Create non-root user (HF Spaces runs containers as uid 1000)
RUN useradd -m -u 1000 user
RUN chown -R user:user /app
USER user

# HF Spaces expects port 7860
EXPOSE 7860

# Start the OpenEnv server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
