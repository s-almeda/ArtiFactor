# templates/map_api.py
"""
This module defines the Flask blueprint for the map API:
endpoints for turning subsets of the art history dataset --> maps, with zones (clusters)

"""

from flask import Blueprint, jsonify, request, g, render_template
import json
import os
import helperfunctions as hf
from index import get_db
from config import IMAGES_PATH
from PIL import Image
import numpy as np
from config import BASE_DIR
MAPS_DIR = os.path.join(BASE_DIR, 'generated_maps')
os.makedirs(MAPS_DIR, exist_ok=True)

# Define the blueprint
map_api_bp = Blueprint('map_api', __name__)

@map_api_bp.route('/test', methods=['GET'])
def test():
    """Test endpoint to verify blueprint is working."""
    return jsonify({
        'success': True,
        'message': 'Map API is working!'
    })

@map_api_bp.route('/map-check')
def check_page():
    """Serve the API check page."""
    return render_template('map_api_check.html')


@map_api_bp.route('/generate_initial_map', methods=['GET'])
def handle_initial_map_request():
    """
    Handles a request for the initial map to populate the image-similarity-space.
    
    Expected URL parameters:
    - n: number of images (default: 50)
    - method: embedding method (default: 'clip')
    - disk: use disk images (default: 'true')
    - debug: enable debug output (default: 'false')
    - random: randomize selection (default: 'false')
    - clustering: enable clustering (default: 'false')
    - k: number of clusters (default: 5, only used if clustering=true)
    - cache: use cached results (default: 'false')
    
    Returns JSON response with map data and optional clustering info.
    """
    print("Received request for initial map generation...")
    
    try:
        # ---- PROCESS THE REQUEST ---- #
        n = int(request.args.get('n', 600))
        method = request.args.get('method', 'clip')
        use_disk = request.args.get('disk', 'true').lower() == 'true'
        debug = request.args.get('debug', 'false').lower() == 'true'
        random = request.args.get('random', 'false').lower() == 'true'
        enable_clustering = request.args.get('clustering', 'false').lower() == 'true'
        k = int(request.args.get('k', 5)) if enable_clustering else None
        cache = request.args.get('cache', 'false').lower() == 'true'

        # UMAP params with new defaults
        n_neighbors = int(request.args.get('n_neighbors', 500))
        min_dist = float(request.args.get('min_dist', 0.9))
        random_state = request.args.get('random_state', None)
        if random_state:
            random_state = int(random_state)

        print(f"Parameters: n={n}, method={method}, clustering={enable_clustering}, k={k}")
        
        def dprint(*args, **kwargs):
            if debug:
                print(*args, **kwargs)

        # Cache handling
        if cache:
            dprint(f"Using cache for map generation")
            min_dist_str = str(min_dist).replace('.', '_')
            cache_suffix = f"_n{n}_method_{method}_nn{n_neighbors}_dist{min_dist_str}"
            if enable_clustering:
                cache_suffix += f"_k{k}"
            cache_filename = f"initial_map_data{cache_suffix}.json"
            cache_path = os.path.join(MAPS_DIR, cache_filename)
            
            if os.path.exists(cache_path):
                dprint(f"Loading cached results from {cache_filename}")
                with open(cache_path, 'r') as f:
                    return jsonify(json.load(f))
        
        dprint(f"\n=== Starting map generation ===")
        
        db = get_db()
        
        # 1. Generate base map data (already normalized to [-1,1])
        base_data = generate_base_map_data(
            db, n, method, use_disk, random, dprint,
            n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state
        )
        if not base_data['success']:
            return jsonify(base_data)
        
        # 2. Build image points from processed data
        dprint(f"\nBuilding image points from processed data...")
        image_points = format_image_points(
            base_data['processed_data'], 
            base_data['coordinates_2d']
        )

        # 3. Build response
        map_response = {
            'success': True,
            'method': method,
            'count': len(image_points),
            'imagePoints': image_points,
            'not_found': base_data['stats']['not_found'],
            'cached_json': 'true' if cache else 'false',
            'precomputed_count': base_data['stats']['precomputed_count'],
            'umap_params': {
                'n_neighbors': n_neighbors,
                'min_dist': min_dist,
                'random_state': random_state
            }
        }
        
        # 4. Add clustering if requested (this adds similarityMap and local positions)
        if enable_clustering and k:
            dprint(f"\nAdding clustering (similarityMap) with k={k} to map data...")
            map_response = add_clustering_to_map_data(
                map_response, 
                base_data['coordinates_2d'], 
                k, 
                dprint
            )
        
        # 5. Save to cache
        if cache:
            dprint(f"Saving results to cache: {cache_filename}")
            with open(cache_path, 'w') as f:
                json.dump(map_response, f, indent=2)
            dprint(f"Saved results to {cache_filename}")
        
        # 6. Return the final map response
        dprint(f"\n=== Map generation complete ===")
        return jsonify(map_response)
    
    except Exception as e:
        print(f"Error during map generation: {e}")
        if request.args.get('debug', 'false').lower() == 'true':
            import traceback
            traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    
def generate_base_map_data(db, n, method, use_disk, random, dprint, n_neighbors=500, min_dist=0.9, random_state=None):
    """
    Extract embeddings and process data for n images.
    Returns dict with: embeddings, processed_data, stats, success, coordinates_2d
    """
    # 1. Get n image entries
    images = fetch_images(db, n, random)
    dprint(f"Found {len(images)} images")
    
    if len(images) < 2:
        return {'success': False, 'error': 'Not enough images found'}
    
    # 2. Query pre-computed embeddings
    image_ids = [row['image_id'] for row in images]
    if method == 'resnet':
        dprint(f"Querying pre-computed ResNet50 embeddings for {len(image_ids)} images...")
        precomputed_embeddings = query_resnet_embeddings(db, image_ids)
    else:
        dprint(f"Querying pre-computed CLIP embeddings for {len(image_ids)} images...")
        precomputed_embeddings = query_clip_embeddings(db, image_ids)
    dprint(f"Found {len(precomputed_embeddings)} pre-computed embeddings")
    
    # 3. Process all entries (get artist info for all)
    embeddings = []
    processed_data = []
    not_found = []
    #missing_embeddings = []
    
    for idx, row in enumerate(images):
        dprint(f"\n--- Processing {idx+1}/{len(images)}: {row['image_id']} ---")
        
        # Get artist info
        artist_names, artist_entries = get_artist_info(row, db)
        
        # Build the full data structure we need
        processed_entry = {
            'image_entry': row,  # Changed from 'row' to 'image_entry' for clarity
            'artist_names': artist_names,
            'artist_entries': artist_entries
        }
        
        # Check if we have pre-computed embedding
        image_id = row['image_id']
        if image_id in precomputed_embeddings:
            dprint(f"✓ Using pre-computed embedding")
            embedding = precomputed_embeddings[image_id]
            embeddings.append(embedding)
            processed_data.append(processed_entry)
        else:
            dprint(f"⚠ Missing embedding for {image_id}")
            not_found.append(image_id)
    
    if len(embeddings) < 2:
        return {'success': False, 'error': 'Not enough valid embeddings', 'not_found': not_found}
    
    # 4. Reduce dimensions with UMAP
    dprint(f"\nRunning UMAP on {len(embeddings)} embeddings...")
    embeddings_array = np.array(embeddings)

    umap_params = {}
    if n_neighbors: umap_params['n_neighbors'] = int(n_neighbors)
    if min_dist: umap_params['min_dist'] = float(min_dist)  
    if random_state is not None and random_state != '':
        umap_params['random_state'] = int(random_state)

    coordinates_2d = hf.reduce_to_2d_umap(embeddings_array, **umap_params)
    
    # Normalize coordinates to [-1, 1] range
    coords_min = coordinates_2d.min(axis=0)
    coords_max = coordinates_2d.max(axis=0)
    coordinates_2d_normalized = 2 * (coordinates_2d - coords_min) / (coords_max - coords_min) - 1
    
    dprint(f"✓ UMAP complete, normalized to [-1, 1]")
    
    # Package up stats
    stats = {
        'not_found': not_found,
        'precomputed_count': len(precomputed_embeddings)#,
        #'computed_on_fly': 0  # We're not computing on fly anymore
    }
    
    return {
        'success': True,
        'coordinates_2d': coordinates_2d_normalized,  # Already normalized
        'processed_data': processed_data,
        'stats': stats
    }


def add_clustering_to_map_data(map_response, coordinates_2d, k, dprint):
    """
    Add clustering information to existing map response.
    """
    dprint(f"\nApplying k-means clustering with k={k}...")
    
    # Apply k-means clustering
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(coordinates_2d)
    
    # Generate zones with proper radius calculation
    zones = []
    for cluster_id in range(k):
        cluster_mask = cluster_labels == cluster_id
        cluster_coords = coordinates_2d[cluster_mask]
        
        if len(cluster_coords) > 0:
            # Calculate center
            center = cluster_coords.mean(axis=0)
            
            # Calculate radius (distance to furthest point * 1.2 for padding)
            distances = np.sqrt(((cluster_coords - center) ** 2).sum(axis=1))
            radius = np.percentile(distances, 90) * 1.2 if len(distances) > 0 else 0.1
            
            zones.append({
                'cluster_id': int(cluster_id),
                'label': f'Zone {cluster_id + 1}',
                'center': {
                    'x': float(center[0]),
                    'y': float(center[1])
                },
                'radius': float(radius),
                'point_count': int(cluster_mask.sum())
            })
    
    # Add cluster info and local positions to each image point
    for idx, point in enumerate(map_response['imagePoints']):
        cluster_id = int(cluster_labels[idx])
        zone = zones[cluster_id]
        
        # Calculate local position relative to zone center
        global_x = point['x']
        global_y = point['y']
        local_x = (global_x - zone['center']['x']) / zone['radius'] if zone['radius'] > 0 else 0
        local_y = (global_y - zone['center']['y']) / zone['radius'] if zone['radius'] > 0 else 0
        
        # Normalize to unit circle if outside
        local_dist = np.sqrt(local_x**2 + local_y**2)
        if local_dist > 1:
            local_x /= local_dist
            local_y /= local_dist
        
        point['clusterInfo'] = {
            'cluster_id': cluster_id,
            'local_position': {
                'x': float(local_x),
                'y': float(local_y)
            }
        }
    
    # Add similarity map to response (not clustering)
    map_response['similarityMap'] = {
        'enabled': True,
        'k': k,
        'zones': zones
    }
    
    dprint(f"✓ Added similarity map with {len(zones)} zones")
    
    return map_response

@map_api_bp.route('/add_clusters_to_map', methods=['POST'])
def handle_add_clusters_to_map():
    """
    Handles a request to add clustering to existing map data.
    
    Expected request JSON:
    {
        "mapData": {...},  // existing map response data
        "k": 5            // number of clusters
    }
    
    Returns JSON response with clustering added to the map data.
    """
    print("Received request to add clusters to existing map...")
    
    try:
        # ---- PROCESS THE REQUEST ---- #
        if not request.json:
            return jsonify({"error": "No JSON data provided"}), 400
        
        map_data = request.json.get('mapData')
        k = request.json.get('k')
        debug = request.json.get('debug', False)
        
        if not map_data:
            return jsonify({"error": "No mapData provided"}), 400
        if not k:
            return jsonify({"error": "No k value provided"}), 400
        
        print(f"Adding clustering with k={k} to map with {map_data.get('count', 0)} points")
        
        def dprint(*args, **kwargs):
            if debug:
                print(*args, **kwargs)
        
        # Extract coordinates from existing map data
        coordinates_2d = extract_coordinates_from_map_data(map_data)
        if coordinates_2d is None:
            return jsonify({"error": "Could not extract coordinates from map data"}), 400
        
        # Add clustering to the map data
        clustered_map_data = add_clustering_to_map_data(map_data, coordinates_2d, k, dprint)
        
        return jsonify(clustered_map_data)
    
    except Exception as e:
        print(f"Error adding clusters to map: {e}")
        if request.json and request.json.get('debug'):
            import traceback
            traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def extract_coordinates_from_map_data(map_data):
    """
    Extract 2D coordinates from existing map data.
    
    Args:
        map_data: existing map response dict with imagePoints
    
    Returns:
        numpy array of coordinates or None if extraction fails
    """
    try:
        image_points = map_data.get('imagePoints', [])
        if not image_points:
            return None
        
        coordinates = []
        for point in image_points:
            x = point.get('x')
            y = point.get('y') 
            if x is not None and y is not None:
                coordinates.append([float(x), float(y)])
            else:
                return None
        
        return np.array(coordinates)
    
    except Exception as e:
        print(f"Error extracting coordinates: {e}")
        return None

def query_resnet_embeddings(db, image_ids):
            """
            Query pre-computed ResNet50 embeddings for given image IDs.
            Returns dict mapping image_id -> embedding array.
            """
            if not image_ids:
                return {}

            placeholders = ','.join(['?' for _ in image_ids])
            query = f"""
                SELECT image_id, embedding 
                FROM vec_image_features 
                WHERE image_id IN ({placeholders})
            """

            cursor = db.execute(query, image_ids)
            results = cursor.fetchall()

            embeddings = {}
            for row in results:
                embedding_data = row['embedding']
                if isinstance(embedding_data, bytes):
                    embedding = np.frombuffer(embedding_data, dtype=np.float32)
                elif isinstance(embedding_data, str):
                    embedding = np.array(json.loads(embedding_data))
                else:
                    embedding = np.array(embedding_data)
                embeddings[row['image_id']] = embedding

            return embeddings

def query_clip_embeddings(db, image_ids):
    """
    Query pre-computed CLIP embeddings for given image IDs.
    Returns dict mapping image_id -> embedding array.
    """
    if not image_ids:
        return {}
    
    placeholders = ','.join(['?' for _ in image_ids])
    query = f"""
        SELECT image_id, embedding 
        FROM vec_clip_features 
        WHERE image_id IN ({placeholders})
    """
    
    cursor = db.execute(query, image_ids)
    results = cursor.fetchall()
    
    embeddings = {}
    for row in results:
        # Handle binary embedding data from SQLite vector extension
        embedding_data = row['embedding']
        if isinstance(embedding_data, bytes):
            # Convert binary data to numpy array
            # This assumes the binary format matches what your vector extension uses
            embedding = np.frombuffer(embedding_data, dtype=np.float32)
        elif isinstance(embedding_data, str):
            # Fallback for JSON string format
            embedding = np.array(json.loads(embedding_data))
        else:
            # Direct array/list
            embedding = np.array(embedding_data)
        
        embeddings[row['image_id']] = embedding
    
    return embeddings


def insert_clip_embedding(image_id, embedding, db):
    """
    Insert a CLIP embedding into the vec_clip_features table.
    """
    # Convert numpy array to binary format for SQLite vector extension
    if isinstance(embedding, np.ndarray):
        embedding_binary = embedding.astype(np.float32).tobytes()
    else:
        embedding_binary = np.array(embedding, dtype=np.float32).tobytes()
    
    query = """
        INSERT OR REPLACE INTO vec_clip_features (image_id, embedding, created_at)
        VALUES (?, ?, datetime('now'))
    """
    
    db.execute(query, (image_id, embedding_binary))
    db.commit()

def fetch_images(db, n, random=False):
    """Get n, optionally random, images that have descriptions."""
    if random:
        cursor = db.execute("""
            SELECT * FROM image_entries 
            WHERE descriptions IS NOT NULL
            ORDER BY RANDOM()
            LIMIT ?
        """, (n,))
    else:
        cursor = db.execute("""
            SELECT * FROM image_entries 
            WHERE descriptions IS NOT NULL
            LIMIT ?
        """, (n,))
    return cursor.fetchall()


def load_image_from_row(row, use_disk, dprint):
    """Try to load image from disk first, then URLs."""
    # Try disk
    if use_disk and row['filename']:
        try:
            from config import IMAGES_PATH
            path = os.path.join(IMAGES_PATH, row['filename'])
            if os.path.exists(path):
                return Image.open(path).convert('RGB')
        except Exception as e:
            dprint(f"Disk load failed: {e}")
    
    # Try URLs
    if row['image_urls']:
        try:
            urls = json.loads(row['image_urls'])
            for size in ['large', 'larger', 'medium', 'small']:
                if size in urls and hf.check_image_url(urls[size]):
                    return hf.url_to_image(urls[size])
        except Exception as e:
            dprint(f"URL load failed: {e}")
    
    return None


def get_artist_info(row, db):
    """Look up artist names in text_entries table."""
    artist_names = []
    artist_entries = []
    
    if row['artist_names']:
        try:
            artist_names = json.loads(row['artist_names'])
            for name in artist_names:
                matches = hf.find_exact_matches(name, db, artists_only=True)
                if matches:
                    artist_entries.append(matches[0])
        except json.JSONDecodeError:
            pass
    
    return artist_names, artist_entries


def format_image_points(processed_data, coordinates_2d):
    """Format the final image points for response, with simplified artist_entries."""
    points = []
    for data, coord in zip(processed_data, coordinates_2d):
        row = data['image_entry']
        # Only keep artist name and entry_id for each artist entry
        artist_entries_simple = [
            {
                'name': entry.get('value'),
                'entryId': entry.get('entry_id')
            }
            for entry in data.get('artist_entries', [])
        ]
        points.append({
            'entryId': row['image_id'],
            'x': float(coord[0]),
            'y': float(coord[1]),
            'artworkData': {
                'image_id': row['image_id'],
                'value': row['value'],
                'artist_names': row['artist_names'],
                'image_urls': row['image_urls'],
                'filename': row['filename'],
                'rights': row['rights'],
                'descriptions': row['descriptions'],
                'relatedKeywordIds': row['relatedKeywordIds'],
                'relatedKeywordStrings': row['relatedKeywordStrings']
            },
            'artistData': {
                #'names': data.get('artist_names', []), # names are included in entries
                'entries': artist_entries_simple
            }
        })
    return points


# Error handlers
@map_api_bp.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404

@map_api_bp.errorhandler(500)
def internal_error(error):
    return jsonify({'success': False, 'error': 'Internal server error'}), 500