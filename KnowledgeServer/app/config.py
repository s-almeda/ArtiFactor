# config.py
import os

# Check if running in Docker
RUNNING_IN_DOCKER = os.environ.get('RUNNING_IN_DOCKER', 'false').lower() == 'true'

# Get the absolute path to the /app directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Database and image paths
DB_PATH = os.path.join(BASE_DIR, "LOCALDB", "knowledgebase.db")
IMAGES_PATH = os.path.join(BASE_DIR, "LOCALDB", "images")

# Model cache directories
if RUNNING_IN_DOCKER:
    MODEL_CACHE_DIR = "/root/.cache/torch/hub"
    TRANSFORMERS_CACHE_DIR = "/root/.cache/transformers"
else:
    MODEL_CACHE_DIR = os.path.join(BASE_DIR, ".cache", "torch", "hub")
    TRANSFORMERS_CACHE_DIR = os.path.join(BASE_DIR, ".cache", "transformers")
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    os.makedirs(TRANSFORMERS_CACHE_DIR, exist_ok=True)

print(f"✅ Running in Docker: {RUNNING_IN_DOCKER}")
print(f"✅ Model cache: {MODEL_CACHE_DIR}")
print(f"✅ Transformers cache: {TRANSFORMERS_CACHE_DIR}")
print(f"✅ Using database: {DB_PATH}")
print(f"✅ Using images: {IMAGES_PATH}")