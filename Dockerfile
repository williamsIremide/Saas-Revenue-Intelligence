FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Create cache directory for Tranco
RUN mkdir -p .tranco

EXPOSE 8080

CMD ["python", "server.py"]