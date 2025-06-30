# This script (generate_tsne.py) is developed by Bob Tianqi Wei.
"""
It generates a t-SNE plot of AI-generated images for any user from a local exported JSON file.
Just change USER_ID and JSON_FILENAME to use for different users.
"""
import json
import numpy as np
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.manifold import TSNE
import requests
import os
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# ====== USER CONFIGURATION ======
USER_ID = 'bob'  # Change this to the user you want
timestamp = None  # Optionally specify a version timestamp, or use latest if None
JSON_FILENAME = f'{USER_ID}-{USER_ID}-0.json'  # Change this to your exported file name
DOWNLOADS_PATH = os.path.expanduser('~/Downloads')
json_path = os.path.join(DOWNLOADS_PATH, JSON_FILENAME)

# ====== LOAD DATA ======
with open(json_path, 'r') as f:
    data = json.load(f)

# Get the latest or specified version key
def get_version_key(data, timestamp=None):
    keys = sorted(data.keys())
    if timestamp and timestamp in keys:
        return timestamp
    return keys[-1]

version_key = get_version_key(data, timestamp)
nodes = data[version_key]['nodes']

# Extract all AI-generated image nodes
image_nodes = [
    node for node in nodes
    if node['type'] == 'image' and node['data'].get('provenance') == 'ai'
]
image_urls = [node['data']['content'] for node in image_nodes]
prompts = [node['data'].get('prompt', '') for node in image_nodes]

print(f"Found {len(image_urls)} AI-generated images in version {version_key} for user {USER_ID}.")

# ====== DOWNLOAD IMAGES ======
imgs = []
valid_prompts = []
for url, prompt in zip(image_urls, prompts):
    try:
        resp = requests.get(url)
        img = Image.open(BytesIO(resp.content)).convert('RGB').resize((224, 224))
        imgs.append(np.array(img))
        valid_prompts.append(prompt)
    except Exception as e:
        print(f"Failed to download {url}: {e}")

if not imgs:
    print("No images to process.")
    exit(0)

imgs = np.array(imgs)

# ====== FEATURE EXTRACTION ======
model = VGG16(weights='imagenet', include_top=False, pooling='avg')
imgs_preprocessed = preprocess_input(imgs)
features = model.predict(imgs_preprocessed)

# ====== t-SNE DIMENSIONALITY REDUCTION ======
if len(features) < 2:
    print("Not enough images for t-SNE visualization.")
    exit(0)
perplexity = min(5, len(features)-1)
tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
features_2d = tsne.fit_transform(features)

# ====== PLOT AND SAVE t-SNE ======
plt.figure(figsize=(12, 10))
ax = plt.gca()
plt.scatter(features_2d[:, 0], features_2d[:, 1], alpha=0)
for (x, y), img_arr in zip(features_2d, imgs):
    imagebox = OffsetImage(img_arr, zoom=32/224)
    ab = AnnotationBbox(imagebox, (x, y), frameon=False)
    ax.add_artist(ab)
plt.title(f"t-SNE Visualization of {USER_ID}'s AI Generated Images (version: {version_key})")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.tight_layout()
# Save to Downloads
save_path = os.path.join(DOWNLOADS_PATH, f'{USER_ID}_tsne_result.png')
plt.savefig(save_path, dpi=200)
print(f"t-SNE plot saved to: {save_path}")
plt.show() 