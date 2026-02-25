# Use a lightweight Python 3.9 image
FROM python:3.9-slim

# Set working directory inside the container
WORKDIR /app

# Runtime data contract (host vs container):
# - The app reads DB and images from /app/LOCALDB inside the container.
# - On local dev, docker_run.sh mounts ./app/LOCALDB -> /app/LOCALDB.
# - On server, docker_run.sh mounts /home/admin/LOCALDB -> /app/LOCALDB.
# Keep these host paths in sync with docker_run.sh/deploy_image_to_gcloud.sh.

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
COPY app/helperfunctions.py .
COPY app/voronoi_helper_functions.py .
COPY app/index.py .
COPY app/templates/ templates/
COPY app/static/ static/
COPY app/config.py .

# Ensure bootstrap.sh is executable
RUN chmod +x bootstrap.sh

# Expose port 8080 for Flask
EXPOSE 8080

# Set environment variable to disable buffering
ENV PYTHONUNBUFFERED=1

# Start Flask with Gunicorn
ENTRYPOINT ["/bin/bash", "/app/bootstrap.sh"]