# templates/hierarchical_map_api.py
"""
This module defines the Flask blueprint for the hierarchical map API:
endpoints for generating Voronoi-based hierarchical maps directly from art history dataset

"""

from flask import Blueprint, jsonify, request, g, render_template
from index import get_db
import os
import json
import numpy as np
import traceback
import math
from config import BASE_DIR
import helperfunctions as hf
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union

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

        print(f"Parameters: n={n}, method={method}, hierarchical_voronoi=true, k={k}")
        
        def dprint(*args, **kwargs):
            if debug:
                print(*args, **kwargs)

        # Cache handling
        if cache:
            cache_key = f"hierarchical_voronoi_n{n}_method{method}_nn{n_neighbors}_dist{min_dist}_k{k}_rs{random_state}"
            cache_file = os.path.join(MAPS_DIR, f"{cache_key}.json")
            if os.path.exists(cache_file):
                dprint(f"Loading from cache: {cache_file}")
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                    cached_data['cached'] = True
                    return jsonify(cached_data)

        dprint(f"\n=== Starting hierarchical Voronoi map generation ===")
        
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
            dprint(f"Saving to cache: {cache_file}")
            with open(cache_file, 'w') as f:
                json.dump(map_response, f, indent=2)
        
        # 6. Return the final hierarchical map response
        dprint(f"\n=== Hierarchical Voronoi map generation complete ===")
        return jsonify(map_response)
    
    except Exception as e:
        print(f"Error during hierarchical map generation: {e}")
        if request.args.get('debug', 'false').lower() == 'true':
            traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc() if debug else None
        }), 500

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
    
    # Normalize coordinates to [-1, 1] range
    coords_min = coordinates_2d.min(axis=0)
    coords_max = coordinates_2d.max(axis=0)
    coordinates_2d_normalized = 2 * (coordinates_2d - coords_min) / (coords_max - coords_min) - 1
    
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
            
            if region and -1 not in region:  # Finite region
                vertices = [vor.vertices[j].tolist() for j in region]
                vertices = sort_vertices_clockwise(vertices)
                centroid = calculate_centroid(vertices)
                
                # Find images in this cluster
                cluster_mask = cluster_labels == i
                cluster_image_ids = [image_points[j]['entryId'] for j in range(len(image_points)) if cluster_mask[j]]
                
                cells.append({
                    'id': i,
                    'vertices': vertices,
                    'centroid': centroid,
                    'imageIds': cluster_image_ids,
                    'pointCount': int(cluster_mask.sum()),
                    'clusterLabel': f'Region {i + 1}'
                })
            else:
                # Handle infinite regions by clipping to bounding box
                vertices = clip_infinite_voronoi_region(vor, i, bounding_box)
                vertices = sort_vertices_clockwise(vertices)
                centroid = calculate_centroid(vertices)
                
                cluster_mask = cluster_labels == i
                cluster_image_ids = [image_points[j]['entryId'] for j in range(len(image_points)) if cluster_mask[j]]
                
                cells.append({
                    'id': i,
                    'vertices': vertices,
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

# Error handlers
@hierarchical_map_api_bp.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404

@hierarchical_map_api_bp.errorhandler(500)
def internal_error(error):
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

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
        
        if not voronoi_data:
            return jsonify({
                'success': False,
                'error': 'No voronoiData provided'
            }), 400
        
        def dprint(*args, **kwargs):
            if debug:
                print(*args, **kwargs)
        
        dprint(f"\n=== Starting Voronoi adjacency analysis ===")
        
        # Analyze adjacency relationships
        adjacency_result = analyze_voronoi_adjacency(voronoi_data, dprint)
        
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

def analyze_voronoi_adjacency(voronoi_data, dprint):
    """
    Analyze adjacency relationships between Voronoi regions using Shapely.
    
    Args:
        voronoi_data: Voronoi data containing cells with vertices
        dprint: Debug print function
    
    Returns:
        Dict with adjacency analysis results and visualization data
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
        
        # Step 1: Create Shapely polygons for each region
        polygons = {}
        for cell in cells:
            region_id = cell['id']
            vertices = cell.get('vertices', [])
            
            if len(vertices) < 3:
                dprint(f"⚠ Region {region_id} has insufficient vertices ({len(vertices)}), skipping")
                continue
            
            try:
                # Create polygon from vertices
                polygon = Polygon(vertices)
                if polygon.is_valid:
                    polygons[region_id] = polygon
                    dprint(f"✓ Created polygon for region {region_id}")
                else:
                    dprint(f"⚠ Invalid polygon for region {region_id}")
            except Exception as e:
                dprint(f"⚠ Failed to create polygon for region {region_id}: {e}")
        
        if len(polygons) < 2:
            return {
                'success': False,
                'error': 'Need at least 2 valid polygons for adjacency analysis'
            }
        
        # Step 2: Build adjacency matrix and calculate boundary lengths
        dprint(f"Building adjacency matrix for {len(polygons)} valid regions...")
        adjacency_matrix = [[0 for _ in range(k)] for _ in range(k)]
        boundary_lengths = {}  # (region_i, region_j) -> length
        shared_boundaries = []
        
        region_ids = list(polygons.keys())
        for i, region_i in enumerate(region_ids):
            for j, region_j in enumerate(region_ids):
                if i < j:  # Avoid duplicate checks
                    poly_i = polygons[region_i]
                    poly_j = polygons[region_j]
                    
                    # Check if polygons touch (share a boundary)
                    if poly_i.touches(poly_j):
                        adjacency_matrix[region_i][region_j] = 1
                        adjacency_matrix[region_j][region_i] = 1
                        
                        # Extract shared boundary and calculate length
                        boundary_length = 0
                        try:
                            intersection = poly_i.intersection(poly_j)
                            boundary_segments = []
                            
                            if hasattr(intersection, 'coords'):
                                # LineString boundary
                                boundary_coords = list(intersection.coords)
                                boundary_segments = [boundary_coords]
                                boundary_length = intersection.length
                            elif hasattr(intersection, 'geoms'):
                                # Multiple segments
                                for geom in intersection.geoms:
                                    if hasattr(geom, 'coords'):
                                        boundary_segments.append(list(geom.coords))
                                        boundary_length += geom.length
                            
                            if boundary_segments:
                                shared_boundaries.append({
                                    'regionIds': [region_i, region_j],
                                    'boundarySegments': boundary_segments,
                                    'length': boundary_length
                                })
                                boundary_lengths[(region_i, region_j)] = boundary_length
                                
                        except Exception as e:
                            dprint(f"⚠ Failed to extract boundary for regions {region_i}-{region_j}: {e}")
                        
                        dprint(f"✓ Regions {region_i} and {region_j} are adjacent (boundary length: {boundary_length:.3f})")
        
        # Step 3: Create optimal pairs based on longest shared boundaries
        dprint(f"Creating optimal pairs based on longest shared boundaries...")
        paired_regions = set()
        optimal_pairs = []
        
        # For each region, find its best pairing partner
        for region_id in region_ids:
            if region_id in paired_regions:
                continue
                
            best_partner = None
            best_length = 0
            
            # Find the region this one shares its longest boundary with
            for other_region in region_ids:
                if other_region == region_id or other_region in paired_regions:
                    continue
                    
                # Check both possible key orders
                boundary_key1 = (min(region_id, other_region), max(region_id, other_region))
                if boundary_key1 in boundary_lengths:
                    length = boundary_lengths[boundary_key1]
                    if length > best_length:
                        best_length = length
                        best_partner = other_region
            
            if best_partner is not None:
                optimal_pairs.append([region_id, best_partner])
                paired_regions.add(region_id)
                paired_regions.add(best_partner)
                dprint(f"✓ Paired regions {region_id} and {best_partner} (shared boundary length: {best_length:.3f})")
        
        # Step 4: Create color assignments for optimal pairs
        # Step 4: Create color assignments for optimal pairs
        dprint(f"Creating color assignments for {len(optimal_pairs)} optimal pairs...")
        
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
        
        # Assign colors to optimal pairs
        for idx, (region_a, region_b) in enumerate(optimal_pairs):
            color = pair_colors[idx % len(pair_colors)]
            
            # Both regions in the pair get the same color
            region_colors[str(region_a)] = color
            region_colors[str(region_b)] = color
        
        # Assign default colors to unpaired regions
        default_color = '#cccccc'  # Light gray
        for region_id in region_ids:
            if str(region_id) not in region_colors:
                region_colors[str(region_id)] = default_color
        
        # Step 5: Assign boundary colors based on pairing
        dprint(f"Assigning boundary colors based on optimal pairs...")
        
        # Create a mapping of paired regions for quick lookup
        paired_with = {}
        for region_a, region_b in optimal_pairs:
            paired_with[region_a] = region_b
            paired_with[region_b] = region_a
        
        # Color each boundary
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
        
        # Step 6: Calculate statistics
        total_possible_adjacencies = (k * (k - 1)) // 2
        total_adjacencies = len(shared_boundaries)
        adjacency_percentage = (total_adjacencies / total_possible_adjacencies * 100) if total_possible_adjacencies > 0 else 0
        
        stats = {
            'totalRegions': k,
            'validPolygons': len(polygons),
            'totalAdjacencies': total_adjacencies,
            'optimalPairs': len(optimal_pairs),
            'unpairedRegions': len(region_ids) - (len(optimal_pairs) * 2),
            'totalPossibleAdjacencies': total_possible_adjacencies,
            'adjacencyPercentage': round(adjacency_percentage, 1),
            'sharedBoundaries': len(shared_boundaries),
            'pairedBoundaries': len([b for b in shared_boundaries if b.get('isPaired', False)]),
            'unpairedBoundaries': len([b for b in shared_boundaries if not b.get('isPaired', False)])
        }
        
        dprint(f"Optimal pairing stats: {stats}")
        
        # Convert boundary_lengths keys from tuples to strings for JSON serialization
        boundary_lengths_serializable = {
            f"{min(k)}-{max(k)}": v for k, v in boundary_lengths.items()
        }
        
        return {
            'success': True,
            'adjacencyData': {
                'adjacencyMatrix': adjacency_matrix,
                'optimalPairs': optimal_pairs,
                'sharedBoundaries': shared_boundaries,
                'boundaryLengths': boundary_lengths_serializable
            },
            'visualizationData': {
                'regionColors': region_colors,
                'boundaryColors': boundary_colors
            },
            'stats': stats
        }
        
    except Exception as e:
        dprint(f"Error in adjacency analysis: {e}")
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }
