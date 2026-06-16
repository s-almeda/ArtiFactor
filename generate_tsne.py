# This script (generate_tsne.py) is developed by Bob Tianqi Wei.
"""
Usage Instructions:
-------------------
This script visualizes image embeddings (AI and non-AI) using UMAP or t-SNE and ResNet50 or VGG16.
It supports image type filtering, colored borders, temporal connections, and flexible model/method selection.

First-time setup (install dependencies):
    pip install numpy pillow matplotlib tensorflow scikit-learn umap-learn requests

How to use:
1. Place your exported JSON file (e.g. bob-bob-0.json) in your Downloads folder.
2. Edit the following configuration options at the top of the script:

   - USER_ID: Set to the user you want to analyze.
   - JSON_FILENAME: Set to your exported file name if different.

   - IMAGE_FILTER: 'ai', 'non_ai', or 'all' (default: 'all')
   - SHOW_TEMPORAL_CONNECTIONS: True/False to show/hide time arrows
   - SHOW_BORDERS: True/False to show/hide colored borders
   - ANALYZE_ALL_VERSIONS: True/False to analyze all versions
   - USE_CURVED_LINES: True for curved arrows, False for straight

   - USE_UMAP: True for UMAP, False for t-SNE
   - USE_RESNET50: True for ResNet50, False for VGG16

3. Run the script:
    python3 generate_tsne.py

4. The output image will be saved to your Downloads folder, with a filename reflecting your settings (e.g. bob_umap_resnet50_all_result.png).

Feature summary:
- Flexible image filtering (AI, non-AI, all)
- Colored borders: blue for AI, gold for non-AI
- Temporal connections: arrows show chronological order
- Start/End labels at image center
- Choice of UMAP or t-SNE for dimensionality reduction
- Choice of ResNet50 or VGG16 for feature extraction
- Curved or straight arrows for time connections

For more details on UMAP vs t-SNE, see: https://pair-code.github.io/understanding-umap/
"""
import json
import numpy as np
from PIL import Image, ImageDraw
from io import BytesIO
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as vgg16_preprocess
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input as resnet50_preprocess
from sklearn.manifold import TSNE
import umap
import requests
import os
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from datetime import datetime
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch

# ====== USER CONFIGURATION ======
USER_ID = 'bob'  # Change this to the user you want
JSON_FILENAME = f'{USER_ID}-{USER_ID}-0.json'  # Change this to your exported file name
DOWNLOADS_PATH = os.path.expanduser('~/Downloads')
json_path = os.path.join(DOWNLOADS_PATH, JSON_FILENAME)

# ====== VISUALIZATION OPTIONS ======
IMAGE_FILTER = 'all'  # Options: 'ai', 'non_ai', 'all'
SHOW_TEMPORAL_CONNECTIONS = True  # Connect images chronologically
SHOW_BORDERS = True  # Show colored borders around images
ANALYZE_ALL_VERSIONS = False  # Analyze all versions for temporal analysis
USE_CURVED_LINES = True  # Use curved lines (False for straight lines)

# ====== MODEL SELECTION ======
USE_UMAP = True  # Use UMAP (False for t-SNE)
USE_RESNET50 = True  # Use ResNet50 (False for VGG16)

# ====== LOAD DATA ======
with open(json_path, 'r') as f:
    data = json.load(f)

def get_version_key(data, timestamp=None):
    keys = sorted(data.keys())
    if timestamp and timestamp in keys:
        return timestamp
    return keys[-1]

def add_border_to_image(img_array, border_color, border_width=10):
    """Add colored border to image array"""
    if not SHOW_BORDERS:
        return img_array
    
    # Convert numpy array to PIL Image
    img = Image.fromarray(img_array)
    draw = ImageDraw.Draw(img)
    
    # Draw border
    width, height = img.size
    for i in range(border_width):
        draw.rectangle([i, i, width-1-i, height-1-i], outline=border_color, width=1)
    
    return np.array(img)

def extract_images_from_version(data, version_key, filter_type='all'):
    """Extract images from a specific version based on filter type"""
    nodes = data[version_key]['nodes']
    
    if filter_type == 'ai':
        image_nodes = [
            node for node in nodes
            if node['type'] == 'image' and node['data'].get('provenance') == 'ai'
        ]
    elif filter_type == 'non_ai':
        image_nodes = [
            node for node in nodes
            if node['type'] == 'image' and node['data'].get('provenance') != 'ai'
        ]
    else:  # 'all'
        image_nodes = [
            node for node in nodes
            if node['type'] == 'image'
        ]
    
    return image_nodes

def download_and_process_images(image_nodes):
    """Download and process images with borders"""
    imgs = []
    prompts = []
    timestamps = []
    is_ai_list = []
    
    for node in image_nodes:
        try:
            url = node['data']['content']
            prompt = node['data'].get('prompt', '')
            is_ai = node['data'].get('provenance') == 'ai'
            
            # Extract timestamp from node ID
            node_id = node.get('id', '')
            timestamp = None
            
            # Try to extract timestamp from ID like "image-1741740135542"
            if '-' in node_id:
                try:
                    # Extract the numeric part after the last dash
                    timestamp_str = node_id.split('-')[-1]
                    if timestamp_str.isdigit():
                        # Convert milliseconds to seconds
                        timestamp = float(timestamp_str) / 1000
                    else:
                        # If not a timestamp, use current time
                        timestamp = datetime.now().timestamp()
                except:
                    timestamp = datetime.now().timestamp()
            else:
                timestamp = datetime.now().timestamp()
            
            resp = requests.get(url)
            img = Image.open(BytesIO(resp.content)).convert('RGB').resize((224, 224))
            img_array = np.array(img)
            
            # Add colored border
            border_color = (0, 100, 255) if is_ai else (255, 215, 0)  # Blue for AI, Gold for non-AI
            img_array = add_border_to_image(img_array, border_color)
            
            imgs.append(img_array)
            prompts.append(prompt)
            timestamps.append(timestamp)
            is_ai_list.append(is_ai)
            
        except Exception as e:
            print(f"Failed to download {url}: {e}")
    
    return imgs, prompts, timestamps, is_ai_list

# ====== MAIN PROCESSING ======
if ANALYZE_ALL_VERSIONS:
    # Analyze all versions for temporal analysis
    all_versions = sorted(data.keys())
    all_imgs = []
    all_prompts = []
    all_timestamps = []
    all_is_ai = []
    all_version_indices = []
    
    for i, version_key in enumerate(all_versions):
        image_nodes = extract_images_from_version(data, version_key, IMAGE_FILTER)
        if image_nodes:
            imgs, prompts, timestamps, is_ai_list = download_and_process_images(image_nodes)
            all_imgs.extend(imgs)
            all_prompts.extend(prompts)
            all_timestamps.extend(timestamps)
            all_is_ai.extend(is_ai_list)
            all_version_indices.extend([i] * len(imgs))
    
    imgs = all_imgs
    prompts = all_prompts
    timestamps = all_timestamps
    is_ai_list = all_is_ai
    version_indices = all_version_indices
    
else:
    # Analyze single version
    version_key = get_version_key(data)
    image_nodes = extract_images_from_version(data, version_key, IMAGE_FILTER)
    imgs, prompts, timestamps, is_ai_list = download_and_process_images(image_nodes)

if not imgs:
    print(f"No images found with filter '{IMAGE_FILTER}'.")
    exit(0)

print(f"Found {len(imgs)} images with filter '{IMAGE_FILTER}'.")
print(f"AI images: {sum(is_ai_list)}, Non-AI images: {len(is_ai_list) - sum(is_ai_list)}")

imgs = np.array(imgs)

# ====== FEATURE EXTRACTION ======
if USE_RESNET50:
    print("Using ResNet50 for feature extraction...")
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    imgs_preprocessed = resnet50_preprocess(imgs)
else:
    print("Using VGG16 for feature extraction...")
    model = VGG16(weights='imagenet', include_top=False, pooling='avg')
    imgs_preprocessed = vgg16_preprocess(imgs)

features = model.predict(imgs_preprocessed)

# ====== DIMENSIONALITY REDUCTION ======
if len(features) < 2:
    print("Not enough images for dimensionality reduction visualization.")
    exit(0)

if USE_UMAP:
    print("Using UMAP for dimensionality reduction...")
    # UMAP parameters optimized for better visualization
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(15, len(features)-1),
        min_dist=0.1,
        metric='cosine',
        random_state=42
    )
    features_2d = reducer.fit_transform(features)
    # Ensure output is numpy array
    if not isinstance(features_2d, np.ndarray):
        features_2d = np.array(features_2d)
else:
    print("Using t-SNE for dimensionality reduction...")
    perplexity = min(5, len(features)-1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    features_2d = tsne.fit_transform(features)

# ====== PLOT AND SAVE ======
plt.figure(figsize=(15, 12))
ax = plt.gca()

# Plot images
plt.scatter(features_2d[:, 0], features_2d[:, 1], alpha=0)
for (x, y), img_arr in zip(features_2d, imgs):
    imagebox = OffsetImage(img_arr, zoom=32/224)
    ab = AnnotationBbox(imagebox, (x, y), frameon=False)
    ax.add_artist(ab)

# Add temporal connections if enabled
if SHOW_TEMPORAL_CONNECTIONS and len(features_2d) > 1:
    # Sort by timestamp
    sorted_indices = np.argsort(timestamps)
    sorted_coords = features_2d[sorted_indices]
    sorted_timestamps = [timestamps[i] for i in sorted_indices]
    
    # Draw arrows connecting chronologically (center to center)
    for i in range(len(sorted_coords) - 1):
        start = sorted_coords[i]
        end = sorted_coords[i + 1]
        
        # Draw arrow (center to center)
        if USE_CURVED_LINES:
            arrow = FancyArrowPatch(
                start, end,
                arrowstyle='->',
                color='gray',
                alpha=0.8,
                linewidth=1,
                connectionstyle='arc3,rad=0.2'  # Create curved connection
            )
        else:
            arrow = FancyArrowPatch(
                start, end,
                arrowstyle='->',
                color='gray',
                alpha=0.8,
                linewidth=1
            )
        ax.add_patch(arrow)
        
        # Add Start and End labels at image center only
        if i == 0:
            ax.annotate('Start', xy=start, xytext=start,
                       fontsize=8, color='gray', weight='bold', 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
        if i == len(sorted_coords) - 2:
            ax.annotate('End', xy=end, xytext=end,
                       fontsize=8, color='gray', weight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

# Add legend
if SHOW_BORDERS:
    legend_elements = [
        patches.Patch(color='blue', label='AI Generated'),
        patches.Patch(color='gold', label='Non-AI Images')
    ]
    plt.legend(handles=legend_elements, loc='upper right')

# Title and labels
filter_text = IMAGE_FILTER.replace('_', ' ').title()
temporal_text = " (with temporal connections)" if SHOW_TEMPORAL_CONNECTIONS else ""
method_text = "UMAP" if USE_UMAP else "t-SNE"
model_text = "ResNet50" if USE_RESNET50 else "VGG16"
plt.title(f"{method_text} Visualization of {USER_ID}'s {filter_text} Images using {model_text}{temporal_text}")
plt.xlabel(f"{method_text} 1")
plt.ylabel(f"{method_text} 2")
plt.tight_layout()

# Save to Downloads
method_abbr = "umap" if USE_UMAP else "tsne"
model_abbr = "resnet50" if USE_RESNET50 else "vgg16"
save_path = os.path.join(DOWNLOADS_PATH, f'{USER_ID}_{method_abbr}_{model_abbr}_{IMAGE_FILTER}_result.png')
plt.savefig(save_path, dpi=200, bbox_inches='tight')
print(f"{method_text} plot saved to: {save_path}")
plt.show() 