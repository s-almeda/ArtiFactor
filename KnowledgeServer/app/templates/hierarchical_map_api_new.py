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
import traceback
from config import BASE_DIR
from helper_functions import helperfunctions as hf
# For scipy imports, we'll do them locally where needed
try:
    from timeout_decorator import timeout, TimeoutError
except ImportError:
    # Fallback if timeout_decorator is not available
    def timeout(seconds):
        def decorator(func):
            return func
        return decorator
    class TimeoutError(Exception):
        pass

# Import from our helper module
from app.helper_functions.voronoi_helper_functions import (
    # Voronoi geometry helpers
    sort_vertices_clockwise, calculate_centroid, clip_infinite_voronoi_region,
    
    # Adjacency analysis
    find_voronoi_adjacency_pairs, apply_adjacency_visualization_colors, analyze_voronoi_adjacency,
    
    # Region merging
    create_optimal_pairs_compactness, merge_paired_voronoi_regions, 
    
    # Voronoi generation
    generate_hierarchical_voronoi_diagram,
    
    # Data formatting and preparation
    format_image_points, format_voronoi_regions, generate_base_map_data,
    
    # Database helpers
    fetch_images, get_artist_info, query_resnet_embeddings, query_clip_embeddings
)

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

# These functions have been moved to voronoi_helper_functions.py:
# - generate_base_map_data
# - generate_hierarchical_voronoi_diagram
# - format_image_points
# - format_voronoi_regions
# - query_resnet_embeddings
# - query_clip_embeddings
# - fetch_images
# - get_artist_info

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

# These functions have been moved to voronoi_helper_functions.py:
# - merge_paired_voronoi_regions
# - create_optimal_pairs_compactness

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

# These functions have been moved to voronoi_helper_functions.py:
# - find_voronoi_adjacency_pairs
# - apply_adjacency_visualization_colors
# - analyze_voronoi_adjacency

# Error handlers
@hierarchical_map_api_bp.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Not found'
    }), 404

@hierarchical_map_api_bp.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500
