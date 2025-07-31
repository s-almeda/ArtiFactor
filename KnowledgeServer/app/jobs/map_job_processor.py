# map_job_processor.py
# Contains process_map_job for use by both Flask API and worker

import os
import json
import traceback

# Use sqlean for easy extension loading and sqlite_vec for vector extension
import sqlean as sqlite3
import helperfunctions as hf
import voronoi_helper_functions as vhf
from config import BASE_DIR, DB_PATH

import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')


def get_db_connection():
    import sqlite_vec
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        print('[Worker] sqlite-vec extension loaded successfully')
    except Exception as e:
        print(f'[Worker] Failed to load sqlite-vec extension: {e}')
    return conn

MAPS_DIR = os.path.join(BASE_DIR, 'generated_maps')
os.makedirs(MAPS_DIR, exist_ok=True)

def keyword_based_cluster_count(n_keywords, n_artworks):
    """
    Simple heuristic: use keywords but ensure at least 5 artworks per cluster
    """
    max_viable_clusters = n_artworks // 5  # At least 5 artworks per cluster
    return min(n_keywords, max_viable_clusters)


def process_map_job(job_id, request_params, update_job_status):
    """
    Process a map generation job. This function is importable by both Flask and worker.
    update_job_status: function(job_id, status, progress_message=None, cache_key=None, error_message=None)
    """
    try:
        # Update status to processing
        update_job_status(job_id, 'processing', 'Starting map generation...')
        data = request_params
        debug = data.get('debug', True)
        def dprint(*args, **kwargs):
            if debug:
                print(*args, **kwargs)
        dprint("Received data:", data)
        # Generate cache key
        cache_key = hf.generate_cache_key(data)
        cache_file = os.path.join(MAPS_DIR, f"{cache_key}.json")
        # Update progress
        update_job_status(job_id, 'processing', 'Retrieving keywords...')
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
        # Open DB connection
        db = get_db_connection()
        dprint("Database connection established.")
        # === RAW DATA PROCESSING PIPELINE ===
        # 1. Get keywords
        keywords_raw = vhf.get_salient_keywords(db, 0, 500, num_keywords)
        dprint(f"Retrieved {len(keywords_raw)} keywords")
        update_job_status(job_id, 'processing', f'Retrieving artworks and their embeddings for {len(keywords_raw)} different artists and categories...')
        # 2. Get embeddings

        embeddings_data = vhf.get_keyword_biased_embeddings(db, keywords_raw, weights=weights)
        embeddings_np = embeddings_data['embeddings']
        artwork_ids = embeddings_data['artworks']
        n_artworks = embeddings_np.shape[0]
        dprint(f"Embeddings shape: {embeddings_np.shape}")
        update_job_status(
            job_id,
            'processing',
            f'Processing {embeddings_np.shape[1]}-dimensional embeddings for {n_artworks} artworks...'
        )
        # 3. Global dimensionality reduction

        global_n_neighbors = hf.calculate_n_neighbors(n_artworks)
        dprint(f"Using n_neighbors={global_n_neighbors} for {n_artworks} artworks")
        uncompressed_coords = hf.reduce_to_2d_umap(
            embeddings_np,
            n_neighbors=global_n_neighbors,
            **base_umap_params
        )
        coordinates_2d = hf.soft_radial_compression(uncompressed_coords, **compression_params)
        dprint("2D coordinates computed")
        update_job_status(job_id, 'processing', f'Flattened {n_artworks} artworks into 2 dimensions! Now grouping similar artworks together...')
        # 4. Clustering
        n_clusters = data.get('n_clusters', keyword_based_cluster_count(len(keywords_raw), n_artworks))
        cluster_labels = hf.apply_kmeans_clustering(coordinates_2d, n_clusters)
        dprint(f"Clustering complete with {n_clusters} clusters")
        update_job_status(job_id, 'processing', 'Clusering complete! Creating map shapes...')
        # 5. Build raw cluster structure
        clusters_raw = vhf.build_raw_clusters(
            cluster_labels, 
            coordinates_2d, 
            embeddings_np, 
            artwork_ids,  # Just pass IDs, not full metadata
            n_clusters
        )
        dprint(f"Created {len(clusters_raw)} non-empty clusters")
        #update_job_status(job_id, 'processing', 'Generating Voronoi cells...')
        # 6. Generate Voronoi cells
        voronoi_data = vhf.generate_voronoi_cells(clusters_raw, coordinates_2d)
        dprint("Generated Voronoi cells")
        update_job_status(job_id, 'processing', 'Recomputing the local positions of artworks within each neighborhood...')
        # 7. Generate per-cluster UMAP coordinates
        per_cluster_coords = vhf.generate_per_cluster_umap(
            clusters_raw, 
            base_umap_params,
            voronoi_data,
            padding_factor=padding_factor
        )
        dprint("Generated per-cluster UMAP coordinates")
        # === HIERARCHICAL LEVELS ===
        update_job_status(job_id, 'processing', 'Creating regions from neighborhoods...')

        level2_data, level3_data = vhf.generate_level2_level3(clusters_raw, voronoi_data, dprint) 

        dprint(f"Generated hierarchical levels - level2: {len(level2_data['clusters'])} clusters, level3: {len(level3_data['clusters'])} clusters")
        update_job_status(
            job_id,
            'processing',
            f"Gathered {len(level2_data['clusters'])} regions into {len(level3_data['clusters'])} countries..."
        )
        # === JSON FORMATTING ===

        # 8. Fetch artwork metadata only now
        artworks_metadata = vhf.fetch_artwork_metadata_batch(artwork_ids, db)
        dprint(f"Fetched metadata for {len(artworks_metadata)} artworks")

        # 9. Format everything for JSON response
        response_data = format_map_response(
            clusters_raw,
            voronoi_data,
            per_cluster_coords,
            artworks_metadata,
            n_clusters,
            level2_data,
            level3_data
        )
        update_job_status(job_id, 'processing', 'Finishing touches...')
        # Before saving to cache, add these fields:
        response_data['cache_key'] = cache_key
        response_data['cached'] = False
        response_data['success'] = True
        update_job_status(job_id, 'processing', 'Saving results...')
        # Save to cache
        with open(cache_file, 'w') as f:
            json.dump(response_data, f, separators=(',', ':'))
        # Mark as completed
        update_job_status(job_id, 'completed', 'Map generation complete!', cache_key)
        db.close()
    except Exception as e:
        error_msg = f"Job failed: {str(e)}"
        print(f"Job {job_id} failed:", traceback.format_exc())
        update_job_status(job_id, 'failed', error_message=error_msg)

def format_map_response(clusters_raw, voronoi_data, per_cluster_coords, artworks_metadata, n_clusters, level2_data, level3_data):
    """
    Convert all raw data to the final JSON-safe response format with hierarchical levels.
    """
    # Format level 1 (existing clusters)
    level1_clusters = []
    for cluster_id in sorted(clusters_raw.keys()):
        cluster = clusters_raw[cluster_id]
        voronoi = voronoi_data.get(cluster_id, {'vertices': []})
        coords = per_cluster_coords.get(cluster_id, [])
        
        cluster_label = generate_smart_cluster_label(
            cluster['artwork_ids'], 
            artworks_metadata, 
            cluster_id
        )
        
        formatted_cluster = {
            'cluster_id': cluster_id,
            'cluster_label': cluster_label,
            'representative_artworks': cluster.get('representative_ids', [])[:3],  # Top 3
            'centroid': {
                'x': float(cluster['centroid'][0]),
                'y': float(cluster['centroid'][1])
            },
            'voronoi_vertices': [[float(x), float(y)] for x, y in voronoi['vertices']],
            'artworks_map': []
        }
        
        for i, artwork_id in enumerate(cluster['artwork_ids']):
            if i < len(coords):
                formatted_cluster['artworks_map'].append({
                    'id': artwork_id,
                    'coords': {
                        'x': float(coords[i][0]),
                        'y': float(coords[i][1])
                    }
                })
        
        level1_clusters.append(formatted_cluster)
    
    # Format level 2 and 3 from the new data structure
    level2_clusters = format_hierarchical_level(level2_data['clusters'], level2_data['voronoi'], artworks_metadata)
    level3_clusters = format_hierarchical_level(level3_data['clusters'], level3_data['voronoi'], artworks_metadata)
    
    return {
        'level_1': level1_clusters,
        'level_2': level2_clusters,
        'level_3': level3_clusters,
        'artworks': artworks_metadata
    }

def format_hierarchical_level(clusters_dict, voronoi_dict, artworks_metadata):
    """Format a hierarchical level from dict format."""
    formatted_clusters = []
    
    for cluster_id in sorted(clusters_dict.keys()):
        cluster = clusters_dict[cluster_id]
        voronoi = voronoi_dict.get(cluster_id, {'vertices': []})
        
        # Generate label (reuse the same logic)
        cluster_label = generate_smart_cluster_label(
            cluster['artwork_ids'], 
            artworks_metadata, 
            cluster_id
        )
        
        formatted_cluster = {
            'cluster_id': cluster_id,
            'cluster_label': cluster_label,
            'representative_artworks': cluster.get('representative_ids', [])[:3],
            'centroid': {
                'x': float(cluster['centroid'][0]),
                'y': float(cluster['centroid'][1])
            },
            'voronoi_vertices': [[float(x), float(y)] for x, y in voronoi['vertices']]
        }
        
        # Add child clusters if this is a merged region
        if 'child_clusters' in cluster and cluster['child_clusters']:
            formatted_cluster['child_clusters'] = cluster['child_clusters']
        
        formatted_clusters.append(formatted_cluster)
    
    return formatted_clusters

def generate_smart_cluster_label(artwork_ids, metadata, cluster_id):
    """Generate a meaningful label for a cluster based on its artworks"""
    if not artwork_ids:
        return f"Cluster {cluster_id + 1}"
    artists = []
    for art_id in artwork_ids:
        if art_id in metadata:
            artist = metadata[art_id].get('artist', '')
            if artist and artist != 'Unknown Artist':
                artists.append(artist)
    unique_artists = list(set(artists))
    if len(unique_artists) == 1:
        return unique_artists[0]
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
    return f"Cluster {cluster_id + 1} ({len(artwork_ids)} artworks)"
