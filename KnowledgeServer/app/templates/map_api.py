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
# from config import IMAGES_PATH
from PIL import Image
import numpy as np
from config import BASE_DIR
MAPS_DIR = os.path.join(BASE_DIR, 'generated_maps')
os.makedirs(MAPS_DIR, exist_ok=True)

def sort_vertices_clockwise(vertices):
    """
    Sort vertices of a polygon in clockwise order.
    
    Args:
        vertices: List of [x, y] coordinate pairs
        
    Returns:
        List of [x, y] coordinate pairs sorted clockwise
    """
    if len(vertices) < 3:
        return vertices
    
    # Calculate centroid
    cx = sum(v[0] for v in vertices) / len(vertices)
    cy = sum(v[1] for v in vertices) / len(vertices)
    
    # Sort by angle from centroid
    def angle_from_center(vertex):
        import math
        return math.atan2(vertex[1] - cy, vertex[0] - cx)
    
    # Sort clockwise (negative angle sort for clockwise)
    sorted_vertices = sorted(vertices, key=angle_from_center, reverse=True)
    return sorted_vertices

def calculate_centroid(vertices):
    """
    Calculate the centroid of a polygon defined by vertices.
    
    Args:
        vertices: List of [x, y] coordinate pairs
        
    Returns:
        [x, y] coordinate pair representing the centroid
    """
    if not vertices:
        return [0, 0]
    
    x_sum = sum(v[0] for v in vertices)
    y_sum = sum(v[1] for v in vertices)
    n = len(vertices)
    
    return [x_sum / n, y_sum / n]

def clip_infinite_voronoi_region(vor, point_idx, bounding_box):
    """
    Clip an infinite Voronoi region to a bounding box.
    
    Args:
        vor: scipy.spatial.Voronoi object
        point_idx: Index of the point whose region we're clipping
        bounding_box: Dict with keys 'min_x', 'max_x', 'min_y', 'max_y'
        
    Returns:
        List of [x, y] coordinate pairs representing the clipped region vertices
    """
    import numpy as np
    
    # Get the region for this point
    region_idx = vor.point_region[point_idx]
    region = vor.regions[region_idx]
    
    if not region or -1 in region:
        # This is an infinite region, create a bounded version
        # For simplicity, return the bounding box corners
        return [
            [bounding_box['min_x'], bounding_box['min_y']],
            [bounding_box['max_x'], bounding_box['min_y']],
            [bounding_box['max_x'], bounding_box['max_y']],
            [bounding_box['min_x'], bounding_box['max_y']]
        ]
    else:
        # Finite region, just return the vertices
        return [vor.vertices[i].tolist() for i in region]

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
    - min_dist: UMAP min_dist (default: 0.9)
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
        random_state = request.args.get('random_state', '42')
        if random_state and random_state.strip():
            random_state = int(random_state)
        else:
            random_state = 42

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
    
def generate_base_map_data(db, n, method, use_disk, random, dprint, n_neighbors=500, min_dist=0.9, random_state=42):
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
    if random_state is not None:
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

    # Use helper function for clustering
    cluster_labels = hf.apply_kmeans_clustering(coordinates_2d, k)

    # Generate zones with proper radius calculation
    zones = []
    for cluster_id in range(k):
        cluster_mask = cluster_labels == cluster_id
        cluster_coords = coordinates_2d[cluster_mask]

        if len(cluster_coords) > 0:
            center = cluster_coords.mean(axis=0)
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

        global_x = point['x']
        global_y = point['y']
        local_x = (global_x - zone['center']['x']) / zone['radius'] if zone['radius'] > 0 else 0
        local_y = (global_y - zone['center']['y']) / zone['radius'] if zone['radius'] > 0 else 0

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


@map_api_bp.route('/generate_voronoi_map', methods=['GET'])
def handle_voronoi_map_request():
    """
    Handles a request for a Voronoi diagram map.
    
    Expected URL parameters:
    - n: number of images (default: 50)
    - method: embedding method (default: 'clip')
    - disk: use disk images (default: 'true')
    - debug: enable debug output (default: 'false')
    - random: randomize selection (default: 'false')
    - min_dist: UMAP min_dist (default: 0.9)
    - cache: use cached results (default: 'false')
    - k: number of Voronoi regions (default: 10)
    - kmeans_iter: number of k-means iterations (default: 50)
    
    Returns simplified JSON response with Voronoi diagram data.
    """
    print("Received request for Voronoi map generation...")
    
    try:
        # Import scipy for Voronoi
        from scipy.spatial import Voronoi
        
        # ---- PROCESS THE REQUEST ---- #
        n = int(request.args.get('n', 100))
        method = request.args.get('method', 'clip')
        use_disk = request.args.get('disk', 'true').lower() == 'true'
        debug = request.args.get('debug', 'false').lower() == 'true'
        random = request.args.get('random', 'false').lower() == 'true'
        cache = request.args.get('cache', 'false').lower() == 'true'
        k = int(request.args.get('k', 10))  # Number of Voronoi regions

        # UMAP params
        n_neighbors = int(request.args.get('n_neighbors', 500))
        min_dist = float(request.args.get('min_dist', 0.9))
        random_state = request.args.get('random_state', '42')
        if random_state and random_state.strip():
            random_state = int(random_state)
        else:
            random_state = 42
        
        # K-means params
        kmeans_iter = int(request.args.get('kmeans_iter', 50))

        print(f"Parameters: n={n}, method={method}, voronoi=true, k={k}")
        
        def dprint(*args, **kwargs):
            if debug:
                print(*args, **kwargs)

        # Cache handling
        if cache:
            dprint(f"Using cache for Voronoi map generation")
            min_dist_str = str(min_dist).replace('.', '_')
            cache_suffix = f"_n{n}_method_{method}_nn{n_neighbors}_dist{min_dist_str}_k{k}_voronoi"
            cache_filename = f"voronoi_map_data{cache_suffix}.json"
            cache_path = os.path.join(MAPS_DIR, cache_filename)
            
            if os.path.exists(cache_path):
                dprint(f"Loading cached Voronoi results from {cache_filename}")
                with open(cache_path, 'r') as f:
                    return jsonify(json.load(f))

        dprint(f"\n=== Starting Voronoi map generation ===")
        
        db = get_db()
        
        # 1. Generate base map data
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

        # 3. Generate Voronoi diagram
        dprint(f"\nGenerating Voronoi diagram with k={k} regions...")
        voronoi_data = generate_voronoi_diagram(image_points, k, dprint, kmeans_iter=kmeans_iter)

        # 4. Build simplified response
        voronoi_response = {
            'success': True,
            'imagePoints': image_points,  # Now includes regionId
            'regions': format_voronoi_regions(voronoi_data),
            'generationParams': {
                'method': method,
                'k': k,
                'n': len(image_points),
                'umap_params': {
                    'n_neighbors': n_neighbors,
                    'min_dist': min_dist,
                    'random_state': random_state
                },
                'kmeans_params': {
                    'iterations': kmeans_iter
                },
                'algorithm': 'k-means + Voronoi'
            },
            'count': len(image_points)
        }
        
        # 5. Save to cache
        if cache:
            dprint(f"Saving Voronoi results to cache: {cache_filename}")
            with open(cache_path, 'w') as f:
                json.dump(voronoi_response, f, indent=2)
            dprint(f"Saved Voronoi results to {cache_filename}")
        
        # 6. Return the final response
        dprint(f"\n=== Voronoi map generation complete ===")
        return jsonify(voronoi_response)
    
    except Exception as e:
        print(f"Error during Voronoi map generation: {e}")
        if request.args.get('debug', 'false').lower() == 'true':
            import traceback
            traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# Simplify the generate_voronoi_diagram function to add regionId directly
def generate_voronoi_diagram(image_points, k, dprint, kmeans_iter=50):
    """
    Generate Voronoi diagram from image points using k-means clustering first.
    This function now adds regionId directly to image points.
    
    Args:
        image_points: List of image point dictionaries with x, y coordinates
        k: Number of clusters/regions
        dprint: Debug print function
        kmeans_iter: Number of k-means iterations (default: 50)
    """
    try:
        from scipy.spatial import Voronoi
        from scipy.cluster.vq import kmeans, vq
        import numpy as np
        
        dprint(f"Generating Voronoi diagram for {len(image_points)} points with k={k} regions...")
        
        # Extract coordinates as numpy array
        points = np.array([[p['x'], p['y']] for p in image_points], dtype=np.float64)
        dprint(f"Extracted coordinates shape: {points.shape}")
        
        # Step 1: Apply k-means clustering with k-means++ initialization
        dprint(f"Running k-means clustering with k={k} (using k-means++ initialization, {kmeans_iter} iterations)...")
        
        # Use scipy's kmeans which implements k-means++ initialization by default
        centroids, distortion = kmeans(points, k, iter=kmeans_iter, thresh=1e-05)
        dprint(f"K-means converged with distortion: {distortion:.4f}")
        dprint(f"Final centroids:\n{centroids}")
        
        # Assign each point to nearest centroid
        cluster_labels, distances = vq(points, centroids)
        dprint(f"Assigned {len(points)} points to {k} clusters")
        
        # Step 2: Create bounding box for Voronoi diagram
        min_x, min_y = points.min(axis=0)
        max_x, max_y = points.max(axis=0)
        
        # Add padding to bounding box
        padding = 0.2
        width = max_x - min_x
        height = max_y - min_y
        bounding_box = {
            'min_x': min_x - padding * width,
            'max_x': max_x + padding * width,
            'min_y': min_y - padding * height,
            'max_y': max_y + padding * height
        }
        dprint(f"Bounding box: {bounding_box}")
        
        # Step 3: Add boundary points to ensure all Voronoi regions are bounded
        boundary_margin = 0.5
        boundary_points = np.array([
            [bounding_box['min_x'] - boundary_margin * width, bounding_box['min_y'] - boundary_margin * height],
            [bounding_box['max_x'] + boundary_margin * width, bounding_box['min_y'] - boundary_margin * height],
            [bounding_box['max_x'] + boundary_margin * width, bounding_box['max_y'] + boundary_margin * height],
            [bounding_box['min_x'] - boundary_margin * width, bounding_box['max_y'] + boundary_margin * height]
        ])
        
        # Combine k-means centroids with boundary points for Voronoi generation
        voronoi_points = np.vstack([centroids, boundary_points])
        dprint(f"Creating Voronoi diagram from {len(centroids)} centroids + {len(boundary_points)} boundary points")
        
        # Step 4: Generate Voronoi diagram
        vor = Voronoi(voronoi_points)
        dprint(f"Voronoi diagram created with {len(vor.regions)} regions")
        
        # Step 5: Process Voronoi cells for the k centroids (ignore boundary point regions)
        cells = []
        for i in range(k):
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]
            
            if not region or -1 in region:
                vertices = clip_infinite_voronoi_region(vor, i, bounding_box)
            else:
                vertices = [vor.vertices[j].tolist() for j in region]
            
            if len(vertices) >= 3:
                vertices = sort_vertices_clockwise(vertices)
                centroid = calculate_centroid(vertices)
                
                # Get list of image IDs in this cluster
                image_ids = []
                for j, label in enumerate(cluster_labels):
                    if label == i:
                        image_ids.append(image_points[j]['entryId'])
                
                cells.append({
                    'id': i,
                    'vertices': vertices,
                    'centroid': centroid,
                    'imageIds': image_ids,
                    'pointCount': len(image_ids)
                })
        
        # Step 6: Add regionId to image points (simplified)
        for i, point in enumerate(image_points):
            cluster_id = int(cluster_labels[i])
            point['regionId'] = cluster_id
        
        voronoi_data = {
            'cells': cells,
            'k': k,
            'algorithm': 'k-means + Voronoi'
        }
        
        dprint(f"✓ Generated {len(cells)} bounded Voronoi regions from k-means centroids")
        
        return voronoi_data
        
    except Exception as e:
        dprint(f"Error generating Voronoi diagram: {e}")
        import traceback
        traceback.print_exc()
        return {
            'cells': [],
            'error': str(e),
            'algorithm': 'k-means + Voronoi (failed)'
        }

# Add a simple formatting function for regions
def format_voronoi_regions(voronoi_data):
    """Format Voronoi regions for simplified response."""
    regions = []
    for cell in voronoi_data.get('cells', []):
        regions.append({
            'id': cell['id'],
            'vertices': cell['vertices'],
            'centroid': cell['centroid'],
            'imageIds': cell['imageIds'],
            'pointCount': cell['pointCount']
        })
    return regions


@map_api_bp.route('/add_voronoi_to_map', methods=['POST'])
def handle_add_voronoi_to_map():
    """
    Add Voronoi regions to existing image points.
    
    Expected JSON:
    {
        "imagePoints": [...],  // Array of points with x, y coordinates
        "k": 10               // Number of regions
    }
    
    Expected URL parameters:
    - debug: enable debug output (default: 'false')
    - kmeans_iter: number of k-means iterations (default: 50)
    """
    try:
        if not request.json:
            return jsonify({"error": "No JSON data provided"}), 400
        
        image_points = request.json.get('imagePoints', [])
        k = request.json.get('k')
        kmeans_iter = int(request.args.get('kmeans_iter', 50))  # Get from URL params
        debug = request.args.get('debug', 'false').lower() == 'true'
        
        if not image_points:
            return jsonify({"error": "No imagePoints provided"}), 400
        if not k:
            return jsonify({"error": "No k value provided"}), 400
        
        def dprint(*args, **kwargs):
            if debug:
                print(*args, **kwargs)
        
        dprint(f"Adding Voronoi regions to {len(image_points)} points with k={k}, kmeans_iter={kmeans_iter}")
        
        # Clear any existing regionId
        for point in image_points:
            point.pop('regionId', None)
        
        # Generate Voronoi diagram
        voronoi_data = generate_voronoi_diagram(image_points, k, dprint, kmeans_iter=kmeans_iter)
        
        # Build simplified response
        response = {
            'success': True,
            'imagePoints': image_points,  # Now includes regionId
            'regions': format_voronoi_regions(voronoi_data),
            'generationParams': {
                'k': k,
                'kmeans_params': {
                    'iterations': kmeans_iter
                },
                'algorithm': 'k-means + Voronoi'
            },
            'count': len(image_points)
        }
        
        return jsonify(response)
    
    except Exception as e:
        if debug:
            import traceback
            traceback.print_exc()
        return jsonify({"error": str(e)}), 500




# Error handlers
@map_api_bp.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404

@map_api_bp.errorhandler(500)
def internal_error(error):
    return jsonify({'success': False, 'error': 'Internal server error'}), 500