# map_api_v3 third iteration of the map api routes we neeed uwu

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
import voronoi_helper_functions as vhf

from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from shapely.strtree import STRtree  # Import STRtree for spatial indexing
from timeout_decorator import timeout, TimeoutError

MAPS_DIR = os.path.join(BASE_DIR, 'generated_maps')
os.makedirs(MAPS_DIR, exist_ok=True)


# Define the blueprint
map_api_v3_bp = Blueprint('map_api_v3', __name__)   


@map_api_v3_bp.route('/map-check-v3')
def map_check_v3_page():
    """Serve the v3 map API check page."""
    return render_template('map_api_v3.html')




@map_api_v3_bp.route('/handle_map_request_v3', methods=['POST'])
def handle_map_request_v3():
    """
    Handle requests for the 4-level hierarchical map with customizable parameters.
    """
    try:
        data = request.get_json()
        
        # Extract parameters
        debug = data.get('debug', True)
        
        def dprint(*args, **kwargs):
            if debug:
                print(*args, **kwargs)
        
        dprint("Received data:", data)
        
        # Parse all parameters upfront
        num_keywords = int(data.get('numKeywords', 100))
        weights = data.get('weights', {
            'clip': 0.6,
            'resnet': 0.0,
            'keyword_semantic': 0.4,
            'keyword_bias': 0.7,
            'debug': debug
        })
        
        # Extract base UMAP params (n_neighbors will be calculated dynamically)
        base_umap_params = data.get('umap', {})
        base_umap_params.pop('n_neighbors', None)  # Remove n_neighbors, we'll calculate it
        base_umap_params.setdefault('min_dist', 0.9)
        base_umap_params.setdefault('random_state', None)
        
        compression_params = data.get('compression', {
            'threshold_percentile': 90, 
            'compression_factor': 0.3
        })
        padding_factor = data.get('padding_factor', 0.1)  # Default padding factor
        
        db = get_db()
        dprint("Database connection established.")

        # === RAW DATA PROCESSING PIPELINE (keeping everything as numpy/efficient structures) ===
        
        # 1. Get keywords (returns list of dicts with artwork IDs)
        keywords_raw = vhf.get_salient_keywords(db, 0, 500, num_keywords)
        dprint(f"Retrieved {len(keywords_raw)} keywords")

        # 2. Get embeddings (returns numpy arrays and artwork IDs)
        embeddings_data = vhf.get_keyword_biased_embeddings(db, keywords_raw, weights=weights)
        embeddings_np = embeddings_data['embeddings']  # numpy array
        artwork_ids = embeddings_data['artworks']  # list of IDs
        n_artworks = embeddings_np.shape[0]
        dprint(f"Embeddings shape: {embeddings_np.shape}")

        # 3. Global dimensionality reduction with dynamic n_neighbors
        global_n_neighbors = hf.calculate_n_neighbors(n_artworks)
        dprint(f"Using n_neighbors={global_n_neighbors} for {n_artworks} artworks")
        
        uncompressed_coords = hf.reduce_to_2d_umap(
            embeddings_np,
            n_neighbors=global_n_neighbors,
            **base_umap_params
        )
        coordinates_2d = hf.soft_radial_compression(uncompressed_coords, **compression_params)
        dprint("2D coordinates computed")

        # 4. Clustering
        n_clusters = data.get('n_clusters', len(keywords_raw))
        cluster_labels = hf.apply_kmeans_clustering(coordinates_2d, n_clusters)
        dprint(f"Clustering complete with {n_clusters} clusters")

        # 5. Build raw cluster structure (keeps numpy arrays)
        clusters_raw = vhf.build_raw_clusters(
            cluster_labels, 
            coordinates_2d, 
            embeddings_np, 
            artwork_ids,  # Just pass IDs, not full metadata
            n_clusters
        )
        dprint(f"Created {len(clusters_raw)} non-empty clusters")

        # 6. Generate Voronoi cells (returns vertices as numpy arrays)
        voronoi_data = vhf.generate_voronoi_cells(clusters_raw, coordinates_2d)
        dprint("Generated Voronoi cells")

        # 7. Generate per-cluster UMAP coordinates (returns numpy arrays)
        per_cluster_coords = vhf.generate_per_cluster_umap(
            clusters_raw, 
            base_umap_params,
            voronoi_data,
            padding_factor=padding_factor
        )
        dprint("Generated per-cluster UMAP coordinates")

        # === JSON FORMATTING (only fetch metadata and format at the very end) ===
        
        # 8. Fetch artwork metadata only now
        artworks_metadata = vhf.fetch_artwork_metadata_batch(artwork_ids, db)
        dprint(f"Fetched metadata for {len(artworks_metadata)} artworks")
        
        # 9. Format everything for JSON response
        response_data = format_map_response(
            clusters_raw,
            voronoi_data,
            per_cluster_coords,
            artworks_metadata,
            n_clusters
        )
        
        return jsonify(response_data)
        
    except Exception as e:
        print("Exception occurred:", traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc() if debug else None
        }), 500
    


def format_map_response(clusters_raw, voronoi_data, per_cluster_coords, artworks_metadata, n_clusters):
    """
    Convert all raw data to the final JSON-safe response format with smart cluster labeling.
    """
    # Format clusters
    formatted_clusters = []
    
    for cluster_id in sorted(clusters_raw.keys()):
        cluster = clusters_raw[cluster_id]
        voronoi = voronoi_data.get(cluster_id, {'vertices': []})
        coords = per_cluster_coords.get(cluster_id, [])
        
        # Generate smart cluster label based on metadata
        cluster_label = generate_smart_cluster_label(
            cluster['artwork_ids'], 
            artworks_metadata, 
            cluster_id
        )
        
        # Format cluster data
        formatted_cluster = {
            'cluster_label': cluster_label,
            'representative_artwork_id': cluster['representative_id'],
            'centroid': {
                'x': float(cluster['centroid'][0]),
                'y': float(cluster['centroid'][1])
            },
            'voronoi_vertices': [[float(x), float(y)] for x, y in voronoi['vertices']],
            'artworks_map': []
        }
        
        # Add artwork coordinates
        for i, artwork_id in enumerate(cluster['artwork_ids']):
            if i < len(coords):
                formatted_cluster['artworks_map'].append({
                    'id': artwork_id,
                    'coords': {
                        'x': float(coords[i][0]),
                        'y': float(coords[i][1])
                    }
                })
        
        formatted_clusters.append(formatted_cluster)
    
    return {
        'clusters': formatted_clusters,
        'artworks': artworks_metadata
    }

def generate_smart_cluster_label(artwork_ids, metadata, cluster_id):
        """Generate a meaningful label for a cluster based on its artworks
            TODO (later) use LLM/VLM to gen meaningful labels
        """
        if not artwork_ids:
            return f"Cluster {cluster_id + 1}"
        
        # Collect artists from cluster
        artists = []
        for art_id in artwork_ids:
            if art_id in metadata:
                artist = metadata[art_id].get('artist', '')
                if artist and artist != 'Unknown Artist':
                    artists.append(artist)
        
        # If all same artist, use artist name
        unique_artists = list(set(artists))
        if len(unique_artists) == 1:
            return unique_artists[0]
        
        # If 2-3 dominant artists (>30% each), list them
        if artists:
            from collections import Counter
            artist_counts = Counter(artists)
            total = len(artists)
            dominant_artists = [
                artist for artist, count in artist_counts.most_common(3)
                if count / total > 0.3
            ]
            if dominant_artists:
                return ' & '.join(dominant_artists[:2])
        
        # Otherwise, use cluster number with size
        return f"Cluster {cluster_id + 1} ({len(artwork_ids)} artworks)"