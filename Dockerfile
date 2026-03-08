FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir openenv-core==0.2.1 uvicorn

# Copy project
COPY . .
RUN pip install --no-cache-dir -e .

# Expose port for HF Spaces
EXPOSE 7860

# Run the OpenEnv server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
