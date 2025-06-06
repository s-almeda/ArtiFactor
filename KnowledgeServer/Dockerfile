# Use a lightweight Python 3.9 image
FROM python:3.9-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements and install dependencies first (better layer caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip
# Install PyTorch CPU version separately
RUN pip install --no-cache-dir torch==2.2.2 torchvision==0.17.2 -f https://download.pytorch.org/whl/cpu/
# Install the rest
RUN pip install --no-cache-dir --default-timeout=100 --retries=5 -r requirements.txt

RUN pip install --no-cache-dir "numpy==1.24.4"
RUN pip install --no-cache-dir "beautifulsoup4==4.13.4"
#RUN pip install --force-reinstall -v "numpy==1.25.2"
# # Download spaCy model
# RUN python -m spacy download en_core_web_sm

# Copy only the necessary files from the app folder
COPY app/bootstrap.sh .
COPY app/gunicorn_config.py .
COPY app/helperfunctions.py .
COPY app/index.py .

COPY app/templates/ templates/
COPY app/static/ static/


# Ensure bootstrap.sh is executable
RUN chmod +x bootstrap.sh

# Expose port 8080 for Flask
EXPOSE 8080

# Set environment variable to disable buffering (useful for logging)
ENV PYTHONUNBUFFERED=1

# Start Flask with Gunicorn
ENTRYPOINT ["/bin/bash", "/app/bootstrap.sh"]

