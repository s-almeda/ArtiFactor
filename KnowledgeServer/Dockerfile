# Use a lightweight Python 3.9 image
FROM python:3.9-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements and install ALL dependencies in one step
COPY requirements.txt .

# Install everything in one go, using both PyPI and PyTorch CPU index
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install --default-timeout=100 --retries=5 \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

# Copy application files
COPY app/bootstrap.sh .
COPY app/gunicorn_config.py .
COPY app/helper_functions helper_functions/
COPY app/index.py .
COPY app/templates/ templates/
COPY app/static/ static/
COPY app/config.py .

# ADD THESE LINES FOR JOBS SYSTEM:
COPY app/jobs/__init__.py jobs/
COPY app/jobs/map_job_processor.py jobs/
COPY app/jobs/schema.sql jobs/
COPY app/jobs/worker.py jobs/

# Ensure bootstrap.sh is executable
RUN chmod +x bootstrap.sh

# CREATE JOBS CACHE DIRECTORY
RUN mkdir -p /app/jobs/cache

# Expose port 8080 for Flask
EXPOSE 8080

# Set environment variable to disable buffering
ENV PYTHONUNBUFFERED=1

# Start Flask with Gunicorn
ENTRYPOINT ["/bin/bash", "/app/bootstrap.sh"]