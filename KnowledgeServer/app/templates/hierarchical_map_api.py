# templates/hierarchical_map_api.py
"""
This module defines the Flask blueprint for the hierarchical map API:
endpoints for generating Voronoi-based hierarchical maps directly from art history dataset

"""

from flask import Blueprint, jsonify, request, g, render_template
from index import get_db
import os
import json
import hashlib
import numpy as np
import traceback
import math
from config import BASE_DIR
import helperfunctions as hf
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from voronoi_helper_functions import sort_vertices_clockwise, calculate_centroid, clip_infinite_voronoi_region, create_optimal_pairs_compactness
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from shapely.strtree import STRtree  # Import STRtree for spatial indexing
from timeout_decorator import timeout, TimeoutError

MAPS_DIR = os.path.join(BASE_DIR, 'generated_maps')
os.makedirs(MAPS_DIR, exist_ok=True)






# Define the blueprint
hierarchical_map_api_bp = Blueprint('hierarchical_map_api', __name__)

@hierarchical_map_api_bp.route('/hierarchical-test', methods=['GET'])
def test():
    """Test endpoint to verify blueprint is working."""
    return jsonify({
        'success': True,
        'message': 'Hierarchical Map API is working!'
    })

@hierarchical_map_api_bp.route('/hierarchical-check')
def hierarchical_check_page():
    """Serve the hierarchical API check page."""
    return render_template('hierarchical_check.html')

@hierarchical_map_api_bp.route('/generate_hierarchical_voronoi_map', methods=['GET'])
@timeout(300)  # 5 minutes timeout for long-running map generation
def handle_hierarchical_voronoi_map_request():
    """
    Handles a request for a hierarchical Voronoi diagram map.
    This combines the base map generation with Voronoi diagram creation in one step.
    
    Expected URL parameters:
    - n: number of images (default: 100)
    - method: embedding method (default: 'clip')
    - disk: use disk images (default: 'true')
    - debug: enable debug output (default: 'false')
    - random: randomize selection (default: 'false')
    - min_dist: UMAP min_dist (default: 0.9)
    - n_neighbors: UMAP n_neighbors (default: 500)
    - random_state: UMAP random_state (default: 42)
    - cache: use cached results (default: 'false')
    - k: number of Voronoi regions (default: 10)
    - kmeans_iter: number of k-means iterations (default: 50)
    
    Returns JSON response with hierarchical Voronoi map data.
    """
    print("Received request for hierarchical Voronoi map generation...")
    
    try:
        # Import scipy for Voronoi
        from scipy.spatial import Voronoi
        
        # ---- PROCESS THE REQUEST ---- #
        n = int(request.args.get('n', 100))
        method = request.args.get('method', 'clip')
        # use_disk = request.args.get('disk', 'true').lower() == 'true'
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

        print(f"Parameters: n={n}, method={method}, hierarchical_voronoi=true, k={k}")
        
        def dprint(*args, **kwargs):
            if debug:
                print(*args, **kwargs)

        # Cache handling
        if cache:
            # Create comprehensive cache key including all parameters that affect the result
            cache_key = f"hierarchical_voronoi_n{n}_method{method}_k{k}_nn{n_neighbors}_dist{min_dist}_rs{random_state}_iter{kmeans_iter}_random{random}"
            cache_file = os.path.join(MAPS_DIR, f"{cache_key}.json")
            if os.path.exists(cache_file):
                dprint(f"Loading from cache: {cache_file}")
                try:
                    with open(cache_file, 'r') as f:
                        cached_data = json.load(f)
                        cached_data['cached'] = True
                        dprint(f"✓ Successfully loaded cached hierarchical map")
                        return jsonify(cached_data)
                except Exception as e:
                    dprint(f"⚠ Failed to load cache file {cache_file}: {e}")
                    # Continue with generation if cache loading fails

        dprint(f"\n=== Starting hierarchical Voronoi map generation ===")
        
        db = get_db()
        
        # 1. Generate base map data
        base_data = generate_base_map_data(
            db, n, method, random, dprint,
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

        # 3. Generate hierarchical Voronoi diagram
        dprint(f"\nGenerating hierarchical Voronoi diagram with k={k} regions...")
        voronoi_data = generate_hierarchical_voronoi_diagram(image_points, k, dprint, kmeans_iter=kmeans_iter)

        # 4. Build comprehensive response
        map_response = {
            'success': True,
            'method': method,
            'count': len(image_points),
            'imagePoints': image_points,
            'voronoiData': voronoi_data,
            'hierarchicalMap': {
                'enabled': True,
                'k': k,
                'algorithm': 'k-means + Voronoi',
                'regions': format_voronoi_regions(voronoi_data)
            },
            'generationParams': {
                'n': n,
                'method': method,
                'k': k,
                'umap_params': {
                    'n_neighbors': n_neighbors,
                    'min_dist': min_dist,
                    'random_state': random_state
                },
                'kmeans_iter': kmeans_iter
            },
            'stats': base_data['stats'],
            'cached': False
        }
        
        # 5. Save to cache if requested
        if cache:
            try:
                dprint(f"Saving to cache: {cache_file}")
                # Ensure directory exists
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                with open(cache_file, 'w') as f:
                    json.dump(map_response, f, indent=2)
                dprint(f"✓ Successfully saved hierarchical map to cache")
            except Exception as e:
                dprint(f"⚠ Failed to save cache file {cache_file}: {e}")
        
        # 6. Return the final hierarchical map response
        dprint(f"\n=== Hierarchical Voronoi map generation complete ===")
        return jsonify(map_response)
    
    except TimeoutError:
        print("ERROR: Hierarchical map generation timed out after 5 minutes")
        return jsonify({
            'success': False,
            'error': 'Request timed out. The dataset is too large for the current timeout limit (5 minutes). Try reducing the number of images (n parameter) or consider using caching.',
            'timeout': True,
            'suggestions': [
                'Reduce the number of images (n parameter)',
                'Enable caching (cache=true) for repeated requests',
                'Try a smaller number of regions (k parameter)',
                'Use fewer UMAP iterations or simpler parameters'
            ]
        }), 408
    except Exception as e:
        print(f"Error during hierarchical map generation: {e}")
        if request.args.get('debug', 'false').lower() == 'true':
            traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc() if debug else None
        }), 500

def generate_base_map_data(db, n, method, random, dprint, n_neighbors=500, min_dist=0.9, random_state=42):
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
    
    for idx, row in enumerate(images):
        dprint(f"\n--- Processing {idx+1}/{len(images)}: {row['image_id']} ---")
        
        # Get artist info
        artist_names, artist_entries = get_artist_info(row, db)
        
        # Build the full data structure we need
        processed_entry = {
            'image_entry': row,
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
    from scipy.spatial import procrustes
    # Align coordinates to a square orientation using Procrustes analysis
    # Create a square reference with the same number of points, distributed in a square
    n_points = coordinates_2d.shape[0]
    # Distribute points in a square grid as reference
    side = int(np.ceil(np.sqrt(n_points)))
    ref_x, ref_y = np.meshgrid(
        np.linspace(-1, 1, side),
        np.linspace(-1, 1, side)
    )
    reference = np.vstack([ref_x.ravel(), ref_y.ravel()]).T[:n_points]

    # Apply Procrustes: mtx2 is the transformed coordinates_2d
    mtx1, mtx2, disparity = procrustes(reference, coordinates_2d)
    coordinates_2d_aligned = mtx2

    # Normalize coordinates to [-1, 1] range
    coords_min = coordinates_2d_aligned.min(axis=0)
    coords_max = coordinates_2d_aligned.max(axis=0)
    coordinates_2d_normalized = 2 * (coordinates_2d_aligned - coords_min) / (coords_max - coords_min) - 1

    dprint(f"✓ UMAP complete, normalized to [-1, 1]")
    
    # Package up stats
    stats = {
        'not_found': not_found,
        'precomputed_count': len(precomputed_embeddings)
    }
    
    return {
        'success': True,
        'coordinates_2d': coordinates_2d_normalized,
        'processed_data': processed_data,
        'stats': stats
    }

def generate_hierarchical_voronoi_diagram(image_points, k, dprint, kmeans_iter=50):
    """
    Generate hierarchical Voronoi diagram from image points using k-means clustering.
    This creates both the Voronoi regions and assigns each image point to its region.
    
    Args:
        image_points: List of image point dictionaries with x, y coordinates
        k: Number of clusters/regions
        dprint: Debug print function
        kmeans_iter: Number of k-means iterations (default: 50)
    """
    try:
        from scipy.spatial import Voronoi
        from scipy.cluster.vq import kmeans, vq
        
        dprint(f"Generating hierarchical Voronoi diagram for {len(image_points)} points with k={k} regions...")
        
        # Extract coordinates as numpy array
        points = np.array([[p['x'], p['y']] for p in image_points], dtype=np.float64)
        dprint(f"Extracted coordinates shape: {points.shape}")

        # STEP 0:  Compute normalized bounding box centered at (0,0), width/height=2, with padding --- #
        padding = 0.025  # 2.5% padding, adjust as needed
        min_x, min_y = points.min(axis=0)
        max_x, max_y = points.max(axis=0)
        span = max(max_x - min_x, max_y - min_y)
        span = max(span * (1 + padding), 2.0)  # Ensure at least 2 units

        # The box is centered at (0,0), so:
        bounding_box = {
            'min_x': -span / 2,
            'max_x': span / 2,
            'min_y': -span / 2,
            'max_y': span / 2
        }
        dprint(f"Normalized bounding box: {bounding_box}")



        
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
            region_idx = vor.point_region[i]  # Get region for centroid i
            region = vor.regions[region_idx]

            # --- Always crop/slice region polygon to the [-1, 1] bounding box --- #
            from shapely.geometry import Polygon, box as shapely_box
            crop_box = shapely_box(-1, -1, 1, 1)

            if region and -1 not in region:  # Finite region
                vertices = [vor.vertices[j].tolist() for j in region]
                vertices = sort_vertices_clockwise(vertices)
                # Crop polygon to bounding box
                poly = Polygon(vertices)
                cropped_poly = poly.intersection(crop_box)
                if cropped_poly.is_empty or not hasattr(cropped_poly, "exterior"):
                    # Fallback: use original vertices if crop fails
                    cropped_vertices = vertices
                else:
                    cropped_vertices = [list(coord) for coord in list(cropped_poly.exterior.coords)[:-1]]
                centroid = calculate_centroid(cropped_vertices)

                # Find images in this cluster
                cluster_mask = cluster_labels == i
                cluster_image_ids = [image_points[j]['entryId'] for j in range(len(image_points)) if cluster_mask[j]]

                cells.append({
                    'id': i,
                    'vertices': cropped_vertices,
                    'centroid': centroid,
                    'imageIds': cluster_image_ids,
                    'pointCount': int(cluster_mask.sum()),
                    'clusterLabel': f'Region {i + 1}'
                })
            else:
                # Handle infinite regions by clipping to bounding box
                vertices = clip_infinite_voronoi_region(vor, i, bounding_box)
                vertices = sort_vertices_clockwise(vertices)
                # Crop polygon to bounding box
                poly = Polygon(vertices)
                cropped_poly = poly.intersection(crop_box)
                if cropped_poly.is_empty or not hasattr(cropped_poly, "exterior"):
                    cropped_vertices = vertices
                else:
                    cropped_vertices = [list(coord) for coord in list(cropped_poly.exterior.coords)[:-1]]
                centroid = calculate_centroid(cropped_vertices)

                cluster_mask = cluster_labels == i
                cluster_image_ids = [image_points[j]['entryId'] for j in range(len(image_points)) if cluster_mask[j]]

                cells.append({
                    'id': i,
                    'vertices': cropped_vertices,
                    'centroid': centroid,
                    'imageIds': cluster_image_ids,
                    'pointCount': int(cluster_mask.sum()),
                    'clusterLabel': f'Region {i + 1}',
                    'clipped': True
                })
        
        # Step 6: Add hierarchical information to image points
        for i, point in enumerate(image_points):
            cluster_id = int(cluster_labels[i])
            region = cells[cluster_id]
            
            # Add regional information to each point
            point['hierarchicalInfo'] = {
                'regionId': cluster_id,
                'regionLabel': region['clusterLabel'],
                'regionCentroid': region['centroid'],
                'distanceToRegionCenter': float(distances[i])
            }
        
        # Step 7: Calculate hierarchical statistics
        hierarchical_stats = {
            'totalRegions': len(cells),
            'averagePointsPerRegion': len(image_points) / len(cells),
            'regionSizes': [cell['pointCount'] for cell in cells],
            'largestRegion': max(cells, key=lambda x: x['pointCount'])['pointCount'],
            'smallestRegion': min(cells, key=lambda x: x['pointCount'])['pointCount']
        }
        
        voronoi_data = {
            'cells': cells,
            'k': k,
            'algorithm': 'hierarchical k-means + Voronoi',
            'hierarchicalStats': hierarchical_stats,
            'boundingBox': bounding_box
        }
        
        dprint(f"✓ Generated {len(cells)} hierarchical Voronoi regions")
        dprint(f"Hierarchical stats: {hierarchical_stats}")
        
        return voronoi_data
        
    except Exception as e:
        dprint(f"Error generating hierarchical Voronoi diagram: {e}")
        traceback.print_exc()
        return {
            'cells': [],
            'error': str(e),
            'algorithm': 'hierarchical k-means + Voronoi (failed)'
        }

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
            embedding = np.frombuffer(embedding_data, dtype=np.float32)
        elif isinstance(embedding_data, str):
            # Fallback for JSON string format
            embedding = np.array(json.loads(embedding_data))
        else:
            # Direct array/list
            embedding = np.array(embedding_data)
        
        embeddings[row['image_id']] = embedding
    
    return embeddings

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

def get_artist_info(row, db):
    """Look up artist names in text_entries table."""
    artist_names = []
    artist_entries = []
    
    if row['artist_names']:
        try:
            artist_names = json.loads(row['artist_names'])
            for name in artist_names:
                cursor = db.execute("""
                    SELECT * FROM text_entries 
                    WHERE value = ? AND type = 'artist'
                """, (name,))
                artist_entry = cursor.fetchone()
                if artist_entry:
                    artist_entries.append(dict(artist_entry))
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
                'entries': artist_entries_simple
            }
        })
    return points

def format_voronoi_regions(voronoi_data):
    """Format Voronoi regions for simplified response."""
    regions = []
    for cell in voronoi_data.get('cells', []):
        regions.append({
            'id': cell['id'],
            'label': cell.get('clusterLabel', f"Region {cell['id'] + 1}"),
            'vertices': cell['vertices'],
            'centroid': cell['centroid'],
            'imageIds': cell['imageIds'],
            'pointCount': cell['pointCount'],
            'clipped': cell.get('clipped', False)
        })
    return regions


@hierarchical_map_api_bp.route('/merge_voronoi_regions', methods=['POST'])
@timeout(180)  # 3 minutes timeout for merging operations
def handle_voronoi_region_merge():
    """
    Merges optimal pairs of adjacent Voronoi regions into single regions.
    
    Expected request JSON:
    {
        "voronoiData": {...},  // Voronoi data from hierarchical map generation
        "imagePoints": [...],  // Image points with hierarchical info
        "debug": true/false    // Optional debug flag
    }
    
    Returns JSON response with merged regions and updated image points.
    """
    print("Received request for Voronoi region merging...")
    
    try:
        # ---- PROCESS THE REQUEST ---- #
        if not request.json:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        voronoi_data = request.json.get('voronoiData')
        image_points = request.json.get('imagePoints')
        debug = request.json.get('debug', False)
        pairing_strategy = request.json.get('pairingStrategy', 'compactness')
        cache = request.json.get('cache', False)
        
        if not voronoi_data:
            return jsonify({
                'success': False,
                'error': 'No voronoiData provided'
            }), 400
            
        if not image_points:
            return jsonify({
                'success': False,
                'error': 'No imagePoints provided'
            }), 400
        
        def dprint(*args, **kwargs):
            if debug:
                print(*args, **kwargs)
        
        # Cache handling for merge operations
        cache_file = None
        if cache:
            # Create cache key based on input data characteristics
            num_regions = len(voronoi_data.get('cells', []))
            num_points = len(image_points)
            
            # Create a hash of the voronoi data to ensure uniqueness
            voronoi_str = json.dumps(voronoi_data, sort_keys=True)
            voronoi_hash = hashlib.md5(voronoi_str.encode()).hexdigest()[:8]
            
            cache_key = f"merge_voronoi_regions{num_regions}_points{num_points}_strategy{pairing_strategy}_hash{voronoi_hash}"
            cache_file = os.path.join(MAPS_DIR, f"{cache_key}.json")
            
            if os.path.exists(cache_file):
                dprint(f"Loading merge result from cache: {cache_file}")
                try:
                    with open(cache_file, 'r') as f:
                        cached_data = json.load(f)
                        cached_data['cached'] = True
                        dprint(f"✓ Successfully loaded cached merge result")
                        return jsonify(cached_data)
                except Exception as e:
                    dprint(f"⚠ Failed to load cache file {cache_file}: {e}")
                    # Continue with merge if cache loading fails
        
        dprint(f"\n=== Starting Voronoi region merging with '{pairing_strategy}' strategy ===")
        
        # Step 1: Find adjacency pairs using existing function
        adjacency_result = find_voronoi_adjacency_pairs(voronoi_data, dprint, pairing_strategy)
        
        if not adjacency_result['success']:
            return jsonify(adjacency_result), 500
        
        # Step 2: Merge paired regions
        merge_result = merge_paired_voronoi_regions(
            voronoi_data, 
            image_points, 
            adjacency_result, 
            dprint
        )
        
        if not merge_result['success']:
            return jsonify(merge_result), 500
        
        # Build response with merged map data
        response = {
            'success': True,
            'originalVoronoiData': voronoi_data,
            'mergedVoronoiData': merge_result['mergedVoronoiData'],
            'originalImagePoints': image_points,
            'mergedImagePoints': merge_result['mergedImagePoints'],
            'mergeStats': merge_result['mergeStats'],
            'adjacencyData': adjacency_result['adjacencyData'],
            'cached': False
        }
        
        # Save to cache if requested
        if cache and cache_file:
            try:
                dprint(f"Saving merge result to cache: {cache_file}")
                # Ensure directory exists
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                with open(cache_file, 'w') as f:
                    json.dump(response, f, indent=2)
                dprint(f"✓ Successfully saved merge result to cache")
            except Exception as e:
                dprint(f"⚠ Failed to save cache file {cache_file}: {e}")
        
        dprint(f"\n=== Region merging complete ===")
        return jsonify(response)
    
    except TimeoutError:
        print("ERROR: Region merging timed out after 3 minutes")
        return jsonify({
            'success': False,
            'error': 'Region merging timed out. The current regions are too complex for the current timeout limit (3 minutes). Try using fewer regions or simpler pairing strategies.',
            'timeout': True,
            'suggestions': [
                'Use fewer initial regions (reduce k parameter)',
                'Enable caching (cache=true) for repeated operations'
            ]
        }), 408
    except Exception as e:
        print(f"Error during region merging: {e}")
        if request.json and request.json.get('debug'):
            traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc() if debug else None
        }), 500

def merge_paired_voronoi_regions(voronoi_data, image_points, adjacency_result, dprint):
    """
    Merge optimal pairs of adjacent regions into single regions.
    
    Args:
        voronoi_data: Original Voronoi data containing cells
        image_points: Original image points with hierarchical info
        adjacency_result: Result from find_voronoi_adjacency_pairs()
        dprint: Debug print function
    
    Returns:
        Dict with merged regions and updated image points
    """        
    try:
        cells = voronoi_data.get('cells', [])
        optimal_pairs = adjacency_result['adjacencyData']['optimalPairs']
        
        dprint(f"Merging {len(optimal_pairs)} optimal pairs from {len(cells)} original regions...")
        
        # Create mapping of region ID to cell data
        region_lookup = {cell['id']: cell for cell in cells}
        
        # Track which regions have been merged
        merged_region_ids = set()
        merged_cells = []
        region_id_mapping = {}  # old_region_id -> new_region_id
        
        # Step 1: Process optimal pairs - merge them into single regions
        for pair_idx, (region_a, region_b) in enumerate(optimal_pairs):
            if region_a in merged_region_ids or region_b in merged_region_ids:
                dprint(f"⚠ Skipping pair ({region_a}, {region_b}) - already merged")
                continue
            
            cell_a = region_lookup.get(region_a)
            cell_b = region_lookup.get(region_b)
            
            if not cell_a or not cell_b:
                dprint(f"⚠ Missing cell data for pair ({region_a}, {region_b})")
                continue
            
            try:
                # Create polygons from vertices
                poly_a = Polygon(cell_a['vertices'])
                poly_b = Polygon(cell_b['vertices'])
                
                # Check if they truly share a boundary
                intersection = poly_a.intersection(poly_b)
                touches = poly_a.touches(poly_b)
                
                # Only merge if they share a boundary
                if not touches:
                    dprint(f"⚠ Regions {region_a} and {region_b} don't seem to share a boundary, skipping merge")
                    continue
                
                # Verify that the intersection is a boundary (line), not just a point
                is_valid_intersection = False
                
                # Check for different intersection types
                if intersection.geom_type in ['LineString', 'MultiLineString']:
                    # Direct line intersection - valid for merging
                    is_valid_intersection = True
                    dprint(f"✓ Regions {region_a} and {region_b} share a {intersection.geom_type} boundary")
                
                elif intersection.geom_type == 'GeometryCollection':
                    # Check for line segments in the collection
                    for geom in intersection.geoms:
                        if geom.geom_type in ['LineString', 'MultiLineString']:
                            is_valid_intersection = True
                            break
                    
                    if is_valid_intersection:
                        dprint(f"✓ Regions {region_a} and {region_b} share line segments in a GeometryCollection")
                    else:
                        dprint(f"⚠ Regions {region_a} and {region_b} have a GeometryCollection intersection without line segments")
                
                # If no direct line intersection, check for shared vertices
                if not is_valid_intersection:
                    vertices_a = set(tuple(np.round(v, 6)) for v in poly_a.exterior.coords)
                    vertices_b = set(tuple(np.round(v, 6)) for v in poly_b.exterior.coords)
                    shared_vertices = vertices_a.intersection(vertices_b)
                    
                    if len(shared_vertices) >= 2:
                        is_valid_intersection = True
                        dprint(f"✓ Regions {region_a} and {region_b} share {len(shared_vertices)} vertices")
                    else:
                        dprint(f"⚠ Regions {region_a} and {region_b} don't share a proper boundary")
                
                # Skip if there's no valid shared boundary for merging
                if not is_valid_intersection:
                    dprint(f"⚠ No valid shared boundary for regions {region_a} and {region_b}, skipping merge")
                    continue
                
                # Merge the two polygons using unary_union
                # This properly dissolves internal boundaries while preserving exterior shapes
                # Get pre-merge properties for debugging
                area_a = poly_a.area
                area_b = poly_b.area
                perimeter_a = poly_a.length
                perimeter_b = poly_b.length
                dprint(f"    - Region {region_a}: Area={area_a:.3f}, Perimeter={perimeter_a:.3f}, Vertices={len(poly_a.exterior.coords)-1}")
                dprint(f"    - Region {region_b}: Area={area_b:.3f}, Perimeter={perimeter_b:.3f}, Vertices={len(poly_b.exterior.coords)-1}")
                
                # Merge the two polygons using unary_union
                # This properly dissolves internal boundaries while preserving exterior shapes
                merged_polygon = unary_union([poly_a, poly_b])
                dprint(f"    - Merged geometry type: {merged_polygon.geom_type}")
                
                # Extract outer boundary vertices
                if merged_polygon.geom_type == 'Polygon':
                    # Simple case: single polygon result
                    # Extract the exterior boundary, skipping the duplicate closing point
                    merged_vertices = list(merged_polygon.exterior.coords[:-1])  
                    merged_centroid = [merged_polygon.centroid.x, merged_polygon.centroid.y]
                    merged_area = merged_polygon.area
                    merged_perimeter = merged_polygon.length
                    
                    dprint(f"✓ Successfully merged regions {region_a} and {region_b}:")
                    dprint(f"    - Original total area: {area_a + area_b:.3f}, New merged area: {merged_area:.3f}")
                    dprint(f"    - Original total perimeter: {perimeter_a + perimeter_b:.3f}, New merged perimeter: {merged_perimeter:.3f}")
                    dprint(f"    - Original vertices: {len(poly_a.exterior.coords)-1 + len(poly_b.exterior.coords)-1}, New vertices: {len(merged_vertices)}")
                    
                elif merged_polygon.geom_type == 'MultiPolygon':
                    # This can happen if the polygons only touch at a point or if they're not properly connected
                    # Take the largest polygon as the merged result
                    largest_poly = max(merged_polygon.geoms, key=lambda p: p.area)
                    merged_vertices = list(largest_poly.exterior.coords[:-1])
                    merged_centroid = [largest_poly.centroid.x, largest_poly.centroid.y]
                    
                    # Report details on all components
                    total_components = len(merged_polygon.geoms)
                    component_details = [f"{i}: Area={p.area:.3f}, Vertices={len(p.exterior.coords)-1}" 
                                         for i, p in enumerate(merged_polygon.geoms)]
                    
                    dprint(f"⚠ Regions {region_a} and {region_b} merged into MultiPolygon with {total_components} components")
                    dprint(f"    - Components: {', '.join(component_details)}")
                    dprint(f"    - Using largest component: {len(merged_vertices)} vertices, area={largest_poly.area:.3f}")
                    
                else:
                    # Handle other geometry types (GeometryCollection, etc.)
                    dprint(f"⚠ Unexpected geometry type after merging: {merged_polygon.geom_type}")
                    
                    # Try to extract usable geometry
                    try:
                        # Check if we can convert to a MultiPolygon or get useful geometry
                        if hasattr(merged_polygon, 'geoms'):
                            # Extract all polygon components
                            polygon_parts = []
                            for geom in merged_polygon.geoms:
                                if geom.geom_type == 'Polygon':
                                    polygon_parts.append(geom)
                            
                            if polygon_parts:
                                largest_poly = max(polygon_parts, key=lambda p: p.area)
                                merged_vertices = list(largest_poly.exterior.coords[:-1])
                                merged_centroid = [largest_poly.centroid.x, largest_poly.centroid.y]
                                dprint(f"  ✓ Recovered a polygon from {merged_polygon.geom_type} with {len(merged_vertices)} vertices")
                            else:
                                dprint(f"  ✗ Could not recover any polygons from {merged_polygon.geom_type}")
                                continue
                        else:
                            # Couldn't find anything useful
                            dprint(f"  ✗ No useful geometry in {merged_polygon.geom_type}")
                            continue
                    except Exception as recovery_error:
                        dprint(f"  ✗ Failed to recover geometry: {str(recovery_error)}")
                        continue
                
                # Create new merged cell
                new_region_id = len(merged_cells)  # Use index as new ID
                merged_cell = {
                    'id': new_region_id,
                    'vertices': merged_vertices,
                    'centroid': merged_centroid,
                    'clusterLabel': f"Merged Region {new_region_id + 1}",
                    'pointCount': cell_a['pointCount'] + cell_b['pointCount'],
                    'imageIds': cell_a.get('imageIds', []) + cell_b.get('imageIds', []),
                    'originalRegions': [region_a, region_b],
                    'mergedFromPair': True
                }
                
                merged_cells.append(merged_cell)
                
                # Update mapping for both original regions
                region_id_mapping[region_a] = new_region_id
                region_id_mapping[region_b] = new_region_id
                
                # Mark as processed
                merged_region_ids.add(region_a)
                merged_region_ids.add(region_b)
                
                dprint(f"✓ Merged regions {region_a} and {region_b} into new region {new_region_id}")
                
            except Exception as e:
                dprint(f"⚠ Failed to merge regions {region_a} and {region_b}: {e}")
        
        # Step 2: Add unmerged regions as-is
        for cell in cells:
            region_id = cell['id']
            if region_id not in merged_region_ids:
                # Keep original region but update ID for consistency
                new_region_id = len(merged_cells)
                unmerged_cell = {
                    **cell,
                    'id': new_region_id,
                    'clusterLabel': cell.get('clusterLabel', f"Region {new_region_id + 1}"),
                    'mergedFromPair': False
                }
                merged_cells.append(unmerged_cell)
                region_id_mapping[region_id] = new_region_id
                dprint(f"✓ Kept unmerged region {region_id} as new region {new_region_id}")
        
        # Step 3: Update image points with new region assignments
        updated_image_points = []
        for point in image_points:
            updated_point = dict(point)  # Copy original point
            
            if 'hierarchicalInfo' in point:
                old_region_id = point['hierarchicalInfo']['regionId']
                if old_region_id in region_id_mapping:
                    new_region_id = region_id_mapping[old_region_id]
                    new_region_cell = merged_cells[new_region_id]
                    
                    # Update hierarchical info
                    updated_point['hierarchicalInfo'] = {
                        **point['hierarchicalInfo'],
                        'regionId': new_region_id,
                        'regionLabel': new_region_cell['clusterLabel'],
                        'regionCentroid': new_region_cell['centroid'],
                        'originalRegionId': old_region_id,
                        'wasMerged': new_region_cell['mergedFromPair']
                    }
                else:
                    dprint(f"⚠ No mapping found for region {old_region_id}")
            
            updated_image_points.append(updated_point)
        
        # Step 4: Create merged voronoi data structure
        merged_voronoi_data = {
            'cells': merged_cells,
            'k': len(merged_cells),
            'algorithm': 'hierarchical k-means + Voronoi + merge',
            'boundingBox': voronoi_data.get('boundingBox'),
            'mergeStats': {
                'originalRegions': len(cells),
                'mergedRegions': len(merged_cells),
                'optimalPairs': len(optimal_pairs),
                'mergedPairs': len([c for c in merged_cells if c.get('mergedFromPair', False)]),
                'unmergedRegions': len([c for c in merged_cells if not c.get('mergedFromPair', False)])
            }
        }
        
        merge_stats = merged_voronoi_data['mergeStats']
        dprint(f"Merge complete: {merge_stats}")
        
        return {
            'success': True,
            'mergedVoronoiData': merged_voronoi_data,
            'mergedImagePoints': updated_image_points,
            'mergeStats': merge_stats
        }
        
    except Exception as e:
        dprint(f"Error in region merging: {e}")
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }

# ===== PAIRING STRATEGY  =====






@hierarchical_map_api_bp.route('/adjacency-analysis', methods=['POST'])
def adjacency_analysis():
    """
    Perform adjacency analysis on the given Voronoi diagram data.
    
    Expects JSON body with the following fields:
    - voronoiData: List of Voronoi region polygons (as lists of [x, y] coordinates)
    - imagePoints: List of image points with 'entryId', 'x', 'y'
    
    Returns JSON response with adjacency information.
    """
    try:
        data = request.get_json()
        voronoi_data = data.get('voronoiData')
        image_points = data.get('imagePoints')
        
        if not voronoi_data or not image_points:
            return jsonify({'success': False, 'error': 'Invalid input data'}), 400
        
        # Convert Voronoi regions to Shapely polygons
        polygons = [Polygon(region) for region in voronoi_data]
        
        # Perform unary union to merge overlapping polygons
        merged_polygon = unary_union(polygons)
        
        # Find adjacent regions for each image point
        adjacency_list = []
        for point in image_points:
            point_geom = Point(point['x'], point['y'])
            
            # Find all polygons that contain this point
            containing_regions = [i for i, poly in enumerate(polygons) if poly.contains(point_geom)]
            
            adjacency_list.append({
                'entryId': point['entryId'],
                'adjacentRegions': containing_regions
            })
        
        return jsonify({
            'success': True,
            'adjacencyData': adjacency_list
        })
    
    except Exception as e:
        print(f"Error during adjacency analysis: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@hierarchical_map_api_bp.route('/analyze_voronoi_adjacency', methods=['POST'])
def handle_voronoi_adjacency_analysis():
    """
    Analyzes adjacency relationships in a Voronoi map and returns enhanced data
    with color-coded adjacent regions and highlighted boundaries.
    
    Expected request JSON:
    {
        "voronoiData": {...},  // Voronoi data from hierarchical map generation
        "debug": true/false    // Optional debug flag
    }
    
    Returns JSON response with adjacency analysis and visualization data.
    """
    print("Received request for Voronoi adjacency analysis...")
    
    try:
        # ---- PROCESS THE REQUEST ---- #
        if not request.json:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        voronoi_data = request.json.get('voronoiData')
        debug = request.json.get('debug', False)
        pairing_strategy = request.json.get('pairingStrategy', 'compactness')
        
        if not voronoi_data:
            return jsonify({
                'success': False,
                'error': 'No voronoiData provided'
            }), 400
        
        def dprint(*args, **kwargs):
            if debug:
                print(*args, **kwargs)
        
        dprint(f"\n=== Starting Voronoi adjacency analysis with '{pairing_strategy}' strategy ===")
        
        # Analyze adjacency relationships
        adjacency_result = analyze_voronoi_adjacency(voronoi_data, dprint, pairing_strategy)
        
        if not adjacency_result['success']:
            return jsonify(adjacency_result), 500
        
        # Build response with adjacency and visualization data
        response = {
            'success': True,
            'originalVoronoiData': voronoi_data,
            'adjacencyData': adjacency_result['adjacencyData'],
            'visualizationData': adjacency_result['visualizationData'],
            'stats': adjacency_result['stats']
        }
        
        dprint(f"\n=== Adjacency analysis complete ===")
        return jsonify(response)
    
    except Exception as e:
        print(f"Error during adjacency analysis: {e}")
        if request.json and request.json.get('debug'):
            traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc() if debug else None
        }), 500

def find_voronoi_adjacency_pairs(voronoi_data, dprint, pairing_strategy='compactness'):
    """
    Find and identify optimal pairs of adjacent Voronoi regions.
    
    Args:
        voronoi_data: Voronoi data containing cells with vertices
        dprint: Debug print function
        
        # fyi: pairing_strategy deprecated, using compactness everytime
    
    Returns:
        Dict with adjacency analysis results (no visualization data)
    """
    try:
        cells = voronoi_data.get('cells', [])
        k = len(cells)
        
        if k < 2:
            return {
                'success': False,
                'error': 'Need at least 2 regions for adjacency analysis'
            }
        
        dprint(f"Analyzing adjacency for {k} Voronoi regions...")
        
        # Step 1: Create Shapely polygons from Voronoi vertices, cropped to bounding box
        polygons = {}
        # Get bounding box from voronoi_data (should have min_x, max_x, min_y, max_y)
        bounding_box = voronoi_data.get('boundingBox')
        crop_box = None
        if bounding_box:
            from shapely.geometry import box as shapely_box
            crop_box = shapely_box(
                bounding_box['min_x'],
                bounding_box['min_y'],
                bounding_box['max_x'],
                bounding_box['max_y']
            )
        for cell in cells:
            region_id = cell['id']
            vertices = cell.get('vertices', [])
            if len(vertices) < 3:
                dprint(f"⚠ Region {region_id} has insufficient vertices ({len(vertices)}), skipping")
                continue
            try:
                # Create polygon from vertices
                polygon = Polygon(vertices)
                if crop_box is not None:
                    cropped_polygon = polygon.intersection(crop_box)
                else:
                    cropped_polygon = polygon
                if cropped_polygon.is_valid and not cropped_polygon.is_empty:
                    polygons[region_id] = cropped_polygon
                    dprint(f"✓ Created cropped polygon for region {region_id}")
                else:
                    dprint(f"⚠ Cropped polygon for region {region_id} is invalid or empty")
            except Exception as e:
                dprint(f"⚠ Failed to create/crop polygon for region {region_id}: {e}")
        
        if len(polygons) < 2:
            return {
                'success': False,
                'error': 'Need at least 2 valid polygons for adjacency analysis'
            }
        
        # Step 2: Build adjacency matrix using a spatial index for efficiency
        dprint(f"Building adjacency matrix for {len(polygons)} valid regions using spatial indexing...")
        adjacency_matrix = [[0 for _ in range(k)] for _ in range(k)]
        boundary_lengths = {}  # (region_i, region_j) -> length
        shared_boundaries = []
        
        # Create a list of polygon objects for the spatial index
        region_ids = list(polygons.keys())
        polygon_objects = list(polygons.values())
        
        # Define a small buffer for improved adjacency detection
        BUFFER_DISTANCE = 0.01
        
        # Create a spatial index for the polygons
        dprint(f"Creating spatial index for {len(polygon_objects)} polygons...")
        spatial_index = STRtree(polygon_objects)
        
        # Define a tolerance for "near adjacency" to account for floating point precision
        ADJACENCY_TOLERANCE = 0.01  # Small tolerance
        
        # For each polygon, query the spatial index for potential neighbors
        for i, region_i in enumerate(region_ids):
            poly_i = polygons[region_i]
            
            # Create a slightly expanded buffer for the polygon to find neighbors
            # This helps catch polygons that might be very close but not exactly touching
            buffered_poly = poly_i.buffer(BUFFER_DISTANCE)
            
            # Query the spatial index for polygons that intersect the buffer
            potential_neighbors_idx = spatial_index.query(buffered_poly)
            potential_neighbors = [polygon_objects[idx] for idx in potential_neighbors_idx]
            
            dprint(f"Region {region_i}: Found {len(potential_neighbors)} potential neighbors from spatial index")
            
            # Check each potential neighbor for actual adjacency
            for poly_j in potential_neighbors:
                # Skip self-intersection
                if poly_i == poly_j:
                    continue
                    
                # Find the region_j ID for this polygon
                j = polygon_objects.index(poly_j)
                region_j = region_ids[j]
                
                # Skip if we've already checked this pair
                if region_i >= region_j:
                    continue
                
                # Check if polygons touch (share a boundary)
                touches = poly_i.touches(poly_j)
                intersects = poly_i.intersects(poly_j)
                min_dist = poly_i.distance(poly_j)
                
                # Check for shared vertices (segments sharing endpoints)
                vertices_i = set(tuple(np.round(v, 6)) for v in poly_i.exterior.coords)
                vertices_j = set(tuple(np.round(v, 6)) for v in poly_j.exterior.coords)
                shared_vertices = vertices_i.intersection(vertices_j)
                
                # Check if there's an actual boundary intersection (most accurate check)
                has_shared_boundary = False
                if touches:
                    # Get the intersection to check if it's a boundary
                    # This works even for complex shared boundaries (series of line segments)
                    try:
                        intersection = poly_i.intersection(poly_j)
                        has_shared_boundary = intersection.geom_type in ['LineString', 'MultiLineString', 'GeometryCollection']
                        
                        if has_shared_boundary and hasattr(intersection, 'length'):
                            boundary_length = intersection.length
                            dprint(f"    - Regions share boundary with length: {boundary_length:.6f}")
                    except Exception:
                        pass
                
                # Regions are adjacent if:
                # 1. They touch according to Shapely AND share a boundary (not just a point)
                # 2. They share at least 2 vertices (a line segment)
                # 3. They're extremely close (just for floating point precision issues)
                is_adjacent = has_shared_boundary or len(shared_vertices) >= 2 or min_dist < ADJACENCY_TOLERANCE
                    
                # Add detailed debug info
                if len(shared_vertices) > 0:
                    dprint(f"    - Regions {region_i} and {region_j} share {len(shared_vertices)} vertices")
                
                if touches:
                    dprint(f"    - Regions {region_i} and {region_j} touch according to Shapely")
                
                if has_shared_boundary:
                    dprint(f"    - Regions {region_i} and {region_j} have a shared boundary")
                
                # For debugging: print vertices if not adjacent
                if is_adjacent:
                    adjacency_matrix[region_i][region_j] = 1
                    adjacency_matrix[region_j][region_i] = 1

                    # Step 3: Calculate boundary lengths via intersection.length
                    boundary_length = 0
                    try:
                        # Extract the shared boundary between polygons
                        intersection = poly_i.intersection(poly_j)
                        boundary_segments = []
                        
                        try:
                            # Extract boundary information based on the geometry type
                            # Case 1: Simple line segment boundary (most common)
                            if intersection.geom_type == 'LineString' or intersection.geom_type == 'LinearRing':
                                # Simple case: single line segment boundary
                                boundary_segments.append(list(intersection.coords))
                                boundary_length = intersection.length
                                dprint(f"    - Simple boundary: LineString (length: {boundary_length:.3f})")
                            
                            # Case 2: Multiple line segments forming boundary
                            elif intersection.geom_type == 'MultiLineString':
                                # Complex case: multiple line segments forming the boundary
                                segment_count = 0
                                for line in intersection.geoms:
                                    if line.length > 0:
                                        boundary_segments.append(list(line.coords))
                                        boundary_length += line.length
                                        segment_count += 1
                                dprint(f"    - Complex boundary: MultiLineString with {segment_count} segments (length: {boundary_length:.3f})")
                            
                            # Case 3: Mixed geometry collection - extract all line segments
                            elif intersection.geom_type == 'GeometryCollection':
                                # Mixed geometry collection - extract all line segments
                                dprint(f"    - Mixed boundary: GeometryCollection with {len(intersection.geoms)} parts")
                                line_found = False
                                for geom in intersection.geoms:
                                    if geom.geom_type == 'LineString' or geom.geom_type == 'LinearRing':
                                        boundary_segments.append(list(geom.coords))
                                        boundary_length += geom.length
                                        line_found = True
                                    elif geom.geom_type == 'MultiLineString':
                                        for line in geom.geoms:
                                            boundary_segments.append(list(line.coords))
                                            boundary_length += line.length
                                            line_found = True
                                    # Point contacts don't contribute to boundary length
                                
                                # If no line segments found, use a minimum length
                                if not line_found:
                                    boundary_length = 0.01
                                    dprint(f"    - No line segments in GeometryCollection, using minimum length")
                            
                            # Case 4: Point contact - not a true boundary, but record the points
                            elif intersection.geom_type in ['Point', 'MultiPoint']:
                                point_count = 1 if intersection.geom_type == 'Point' else len(intersection.geoms)
                                dprint(f"    - Point contact with {point_count} points")
                                
                                # Create small segments for visualization
                                if intersection.geom_type == 'Point':
                                    pt = list(intersection.coords)[0]
                                    boundary_segments.append([[pt[0], pt[1]], [pt[0], pt[1]]])
                                else:  # MultiPoint
                                    for point in intersection.geoms:
                                        pt = list(point.coords)[0]
                                        boundary_segments.append([[pt[0], pt[1]], [pt[0], pt[1]]])
                                
                                # Use a small length for point contacts
                                boundary_length = 0.01 * point_count
                            
                            # Case 5: Polygon or MultiPolygon (rare - overlapping regions)
                            elif intersection.geom_type in ['Polygon', 'MultiPolygon']:
                                # Extract the boundary of the overlap
                                boundary = intersection.boundary
                                if hasattr(boundary, 'coords'):  # LineString
                                    boundary_segments.append(list(boundary.coords))
                                    boundary_length = boundary.length
                                elif hasattr(boundary, 'geoms'):  # MultiLineString
                                    for line in boundary.geoms:
                                        boundary_segments.append(list(line.coords))
                                        boundary_length += line.length
                            
                            # For true boundaries, ensure they have a reasonable minimum length
                            MIN_BOUNDARY_LENGTH = 0.01
                            if boundary_length < MIN_BOUNDARY_LENGTH:
                                boundary_length = MIN_BOUNDARY_LENGTH
                            
                            # If we extracted any boundary information, include it
                            if boundary_segments:
                                dprint(f"    - Successfully extracted boundary (total length: {boundary_length:.3f})")
                            else:
                                dprint(f"    - No boundary segments found, using minimum length")
                                boundary_length = MIN_BOUNDARY_LENGTH
                                # Create a dummy segment for visualization
                                midpoint_i = np.array(poly_i.centroid.coords[0])
                                midpoint_j = np.array(poly_j.centroid.coords[0])
                                dummy_point = (midpoint_i + midpoint_j) / 2
                                boundary_segments.append([[dummy_point[0], dummy_point[1]], [dummy_point[0], dummy_point[1]]])
                            
                        except Exception as boundary_error:
                            # Log the error but continue with a minimum length
                            dprint(f"    ⚠ Error extracting boundary: {str(boundary_error)}")
                            boundary_length = MIN_BOUNDARY_LENGTH
                        
                                # Always create a boundary record, even if extraction had issues
                        # This ensures adjacency is preserved for merging
                        boundary_info = {
                            'regionIds': [region_i, region_j],
                            'boundarySegments': boundary_segments if boundary_segments else [[[0, 0], [0, 0]]],
                            'length': boundary_length,
                            'intersectionType': intersection.geom_type
                        }
                        shared_boundaries.append(boundary_info)
                        boundary_lengths[(region_i, region_j)] = boundary_length
                        dprint(f"    - Added boundary: {intersection.geom_type}, length: {boundary_length:.3f}")

                    except Exception as e:
                        dprint(f"⚠ Failed to extract boundary for regions {region_i}-{region_j}: {str(e)}")
                        # Even if extraction fails, we'll still consider them adjacent with a minimum length
                        MIN_BOUNDARY_LENGTH = 0.01
                        boundary_length = MIN_BOUNDARY_LENGTH
                        
                        # Create a fallback boundary visualization
                        try:
                            # Get centroids for both polygons for fallback visualization
                            midpoint_i = np.array(poly_i.centroid.coords[0])
                            midpoint_j = np.array(poly_j.centroid.coords[0])
                            dummy_point = (midpoint_i + midpoint_j) / 2
                            boundary_segments = [[[midpoint_i[0], midpoint_i[1]], [midpoint_j[0], midpoint_j[1]]]]
                        except:
                            # Complete fallback if even that fails
                            boundary_segments = [[[0, 0], [0, 0]]]
                        
                        # Add fallback boundary record
                        boundary_info = {
                            'regionIds': [region_i, region_j],
                            'boundarySegments': boundary_segments,
                            'length': boundary_length,
                            'intersectionType': 'error-fallback'
                        }
                        shared_boundaries.append(boundary_info)
                        boundary_lengths[(region_i, region_j)] = boundary_length
                        dprint(f"    - Using fallback boundary length {boundary_length} despite error")

                    dprint(f"✓ Regions {region_i} and {region_j} are adjacent (boundary length: {boundary_length:.3f})")
                else:
                    # Print debug info for non-adjacent pairs
                    dprint(f"✗ Regions {region_i} and {region_j} are NOT adjacent.")
                    dprint(f"    - poly_i.touches(poly_j): {touches}")
                    dprint(f"    - poly_i.intersects(poly_j): {intersects}")
                    dprint(f"    - poly_i.distance(poly_j): {min_dist:.10f}")
                    dprint(f"    - Vertices region {region_i}: {list(poly_i.exterior.coords)}")
                    dprint(f"    - Vertices region {region_j}: {list(poly_j.exterior.coords)}")
                    # Optionally, print shared points
                    shared_points = set(tuple(np.round(v, 8)) for v in poly_i.exterior.coords) & set(tuple(np.round(v, 8)) for v in poly_j.exterior.coords)
                    dprint(f"    - Shared vertices (rounded): {shared_points}")

        # Step 4: Create optimal pairs using selected pairing strategy
        dprint(f"Creating optimal pairs using '{pairing_strategy}' strategy...")

        # Select pairing strategy function
        
        optimal_pairs = create_optimal_pairs_compactness(
            region_ids, boundary_lengths, polygons, dprint
        )
        
        # Calculate basic statistics
        total_possible_adjacencies = (k * (k - 1)) // 2
        total_adjacencies = len(shared_boundaries)
        adjacency_percentage = (total_adjacencies / total_possible_adjacencies * 100) if total_possible_adjacencies > 0 else 0
        
        # Print detailed summary of adjacencies and boundary lengths
        dprint(f"\nADJACENCY ANALYSIS SUMMARY:")
        dprint(f"- Total regions: {k}")
        dprint(f"- Total possible region pairs: {total_possible_adjacencies}")
        dprint(f"- Total adjacent region pairs: {total_adjacencies} ({adjacency_percentage:.1f}%)")
        
        # Show boundary length distribution
        if boundary_lengths:
            lengths = list(boundary_lengths.values())
            min_length = min(lengths)
            max_length = max(lengths)
            avg_length = sum(lengths) / len(lengths)
            dprint(f"- Boundary lengths: min={min_length:.3f}, max={max_length:.3f}, avg={avg_length:.3f}")
            
            # Count how many have minimum length
            min_count = sum(1 for l in lengths if l == 0.01)
            if min_count > 0:
                dprint(f"- {min_count} boundaries ({min_count/len(lengths)*100:.1f}%) have minimum length (0.01)")
            
        # Show all boundary lengths for debugging
        dprint("\nDetailed boundary lengths:")
        for (r1, r2), length in sorted(boundary_lengths.items(), key=lambda x: x[1]):
            dprint(f"- Regions {r1}-{r2}: {length:.3f}")
        
        # Convert boundary_lengths keys from tuples to strings for JSON serialization
        boundary_lengths_serializable = {
            f"{min(key)}-{max(key)}": v for key, v in boundary_lengths.items()
        }
        
        return {
            'success': True,
            'adjacencyData': {
                'adjacencyMatrix': adjacency_matrix,
                'optimalPairs': optimal_pairs,
                'boundaryLengths': boundary_lengths_serializable,
                'sharedBoundaries': shared_boundaries
            },
            'basicStats': {
                'totalRegions': k,
                'validPolygons': len(polygons),
                'totalAdjacencies': total_adjacencies,
                'optimalPairs': len(optimal_pairs),
                'unpairedRegions': len(region_ids) - (len(optimal_pairs) * 2),
                'totalPossibleAdjacencies': total_possible_adjacencies,
                'adjacencyPercentage': round(adjacency_percentage, 1)
            },
            'regionIds': region_ids
        }
        
    except Exception as e:
        dprint(f"Error in adjacency analysis: {e}")
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }

def apply_adjacency_visualization_colors(adjacency_result, dprint):
    """
    Apply color assignments and cosmetics based on optimal pairs.
    
    Args:
        adjacency_result: Result from find_voronoi_adjacency_pairs()
        dprint: Debug print function
    
    Returns:
        Dict with visualization data and enhanced statistics
    """
    try:
        if not adjacency_result['success']:
            return adjacency_result
        
        adjacency_data = adjacency_result['adjacencyData']
        basic_stats = adjacency_result['basicStats']
        region_ids = adjacency_result['regionIds']
        optimal_pairs = adjacency_data['optimalPairs']
        shared_boundaries = adjacency_data['sharedBoundaries']
        
        dprint(f"Applying visualization colors for {len(optimal_pairs)} optimal pairs...")
        
        # Color palette for paired regions
        pair_colors = [
            '#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57', '#ff9ff3', 
            '#54a0ff', '#5f27cd', '#00d2d3', '#ff6348', '#ff7675', '#74b9ff'
        ]
        
        region_colors = {}
        boundary_colors = {}
        
        # Function to darken a hex color
        def darken_color(hex_color, factor=0.7):
            # Remove # if present
            hex_color = hex_color.lstrip('#')
            # Convert to RGB
            r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            # Darken
            r, g, b = int(r * factor), int(g * factor), int(b * factor)
            # Convert back to hex
            return f'#{r:02x}{g:02x}{b:02x}'
        
        # Assign colors: paired regions → same color, unpaired → gray
        for idx, (region_a, region_b) in enumerate(optimal_pairs):
            color = pair_colors[idx % len(pair_colors)]
            
            # Both regions in the pair get the same color
            region_colors[str(region_a)] = color
            region_colors[str(region_b)] = color
        
        # Assign default gray to unpaired regions
        default_color = '#cccccc'  # Light gray
        for region_id in region_ids:
            if str(region_id) not in region_colors:
                region_colors[str(region_id)] = default_color
        
        # Create a mapping of paired regions for quick lookup
        paired_with = {}
        for region_a, region_b in optimal_pairs:
            paired_with[region_a] = region_b
            paired_with[region_b] = region_a
        
        # Color boundaries: paired → darker region color, unpaired → black
        dprint(f"Assigning boundary colors based on optimal pairs...")
        for boundary in shared_boundaries:
            region_a, region_b = boundary['regionIds']
            boundary_key = f"{min(region_a, region_b)}-{max(region_a, region_b)}"
            
            # Check if these two regions are paired together
            if (region_a in paired_with and paired_with[region_a] == region_b):
                # This is a shared boundary between paired regions - use darker version of their color
                base_color = region_colors[str(region_a)]  # Both regions have same color
                boundary_colors[boundary_key] = darken_color(base_color, 0.6)
                boundary['isPaired'] = True
                dprint(f"✓ Paired boundary {region_a}-{region_b}: {boundary_colors[boundary_key]}")
            else:
                # This is a boundary between non-paired regions - use black
                boundary_colors[boundary_key] = '#000000'
                boundary['isPaired'] = False
                dprint(f"✓ Non-paired boundary {region_a}-{region_b}: black")
        
        # Enhanced statistics with visualization info
        enhanced_stats = {
            **basic_stats,
            'sharedBoundaries': len(shared_boundaries),
            'pairedBoundaries': len([b for b in shared_boundaries if b.get('isPaired', False)]),
            'unpairedBoundaries': len([b for b in shared_boundaries if not b.get('isPaired', False)])
        }
        
        dprint(f"Visualization color assignment complete: {enhanced_stats}")
        
        return {
            'success': True,
            'adjacencyData': adjacency_data,
            'visualizationData': {
                'regionColors': region_colors,
                'boundaryColors': boundary_colors
            },
            'stats': enhanced_stats
        }
        
    except Exception as e:
        dprint(f"Error in visualization color assignment: {e}")
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }

def analyze_voronoi_adjacency(voronoi_data, dprint, pairing_strategy='compactness'):
    """
    Main adjacency analysis function that coordinates finding pairs and applying visualization.
    
    Args:
        voronoi_data: Voronoi data containing cells with vertices
        dprint: Debug print function
        pairing_strategy: Strategy for pairing regions
    
    Returns:
        Dict with adjacency analysis results and visualization data
    """
    # Step 1: Find and identify pairs
    pair_result = find_voronoi_adjacency_pairs(voronoi_data, dprint, pairing_strategy)
    
    if not pair_result['success']:
        return pair_result
    
    # Step 2: Apply cosmetics and visualization colors
    return apply_adjacency_visualization_colors(pair_result, dprint)



# Error handlers
@hierarchical_map_api_bp.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404

@hierarchical_map_api_bp.errorhandler(500)
def internal_error(error):
    return jsonify({'success': False, 'error': 'Internal server error'}), 500
