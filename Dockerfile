FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Install system dependencies (required for OpenCV in Python)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Hugging Face Spaces requires services to run on port 7860
EXPOSE 7860

# Start the Flask app using Gunicorn on port 7860
CMD ["gunicorn", "app.app:app", "--bind", "0.0.0.0:7860", "--timeout", "600"]
