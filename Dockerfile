FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY api.py .
COPY index.html .
COPY models/ ./models/
COPY data/ ./data/

# Expose port
EXPOSE 8000

# Run the FastAPI application
CMD ["python", "api.py"]
