# Use official Python runtime as a parent image
FROM python:3.9-slim

# Prevent Python from writing pyc files to disc and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . /app

# Expose Streamlit default port
EXPOSE 8501

# Default command to run the app
CMD ["streamlit", "run", "app/main.py", "--server.address=0.0.0.0", "--server.port=8501"]