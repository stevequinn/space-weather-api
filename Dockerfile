# Use official Python image
FROM python:3.11-slim

# Set work directory
WORKDIR /app

# Copy requirements and source code
COPY requirements.txt ./
COPY . ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create logs directory and set as a volume
RUN mkdir -p /app/logs
VOLUME ["/app/logs"]

# Expose FastAPI port 8200
EXPOSE 8200

# Run FastAPI app with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8200"]
