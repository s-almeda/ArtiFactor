"""
Hierarchical Voronoi Map API
This module defines functions for creating hierarchical voronoi maps with multiple levels of clustering.
"""

import json
import numpy as np
from flask import Blueprint, jsonify, request, render_template
from index import get_db
from sklearn.metrics.pairwise import cosine_distances
from scipy.cluster.hierarchy import linkage, cut_tree
from scipy.spatial import Voronoi
from scipy.spatial.distance import squareform
import umap


def extract_combined_embeddings(sample_size=1000, clip_weight=2.0, keyword_weight=1.0, artist_weight=3.0):
    """
    Extract and combine embeddings from the database for hierarchical mapping.
    
    Args:
        sample_size: Number of images to sample (default: 1000)
        clip_weight: Weight for CLIP embeddings (default: 2.0)
        keyword_weight: Weight for keyword embeddings (default: 1.0)
        artist_weight: Weight for artist embeddings (default: 3.0)
    
    Returns:
        List of dictionaries with combined embeddings and metadata
    """
    db = get_db()
    
    # Get images with required data
    query = """
        SELECT image_id, value, artist_names, relatedKeywordIds
        FROM image_entries 
        WHERE artist_names IS NOT NULL 
        AND relatedKeywordIds IS NOT NULL
        AND artist_names != '[]'
        AND relatedKeywordIds != '[]'
        ORDER BY RANDOM()
        LIMIT ?
    """
    
    cursor = db.execute(query, (sample_size,))
    images = cursor.fetchall()
    
    if not images:
        return []
    
    # Get all image IDs for CLIP embeddings
    image_ids = [row['image_id'] for row in images]
    clip_embeddings = _get_clip_embeddings(db, image_ids)
    
    # Collect all artist and keyword IDs we need
    all_artist_ids = set()
    all_keyword_ids = set()
    
    for row in images:
        # Parse artist names to get artist IDs
        try:
            artist_names = json.loads(row['artist_names'])
            if artist_names:
                primary_artist = artist_names[0]
                artist_id = _lookup_artist_id(db, primary_artist)
                if artist_id:
                    all_artist_ids.add(artist_id)
        except (json.JSONDecodeError, IndexError):
            continue
            
        # Parse keyword IDs
        try:
            keyword_ids = json.loads(row['relatedKeywordIds'])
            all_keyword_ids.update(keyword_ids)
        except json.JSONDecodeError:
            continue
    
    # Get all artist and keyword embeddings in batch
    artist_embeddings = _get_value_embeddings(db, list(all_artist_ids))
    keyword_embeddings = _get_value_embeddings(db, list(all_keyword_ids))
    
    # Process each image
    images_data = []
    for row in images:
        try:
            # Get CLIP embedding
            clip_embedding = clip_embeddings.get(row['image_id'])
            if clip_embedding is None:
                continue
                
            # Get artist embedding
            artist_names = json.loads(row['artist_names'])
            primary_artist = artist_names[0] if artist_names else None
            if not primary_artist:
                continue
                
            artist_id = _lookup_artist_id(db, primary_artist)
            artist_embedding = artist_embeddings.get(artist_id) if artist_id else None
            if artist_embedding is None:
                continue
            
            # Get keyword embeddings
            keyword_ids = json.loads(row['relatedKeywordIds'])
            valid_keyword_embeddings = []
            for keyword_id in keyword_ids:
                if keyword_id in keyword_embeddings:
                    valid_keyword_embeddings.append(keyword_embeddings[keyword_id])
            
            if not valid_keyword_embeddings:
                continue
                
            # Average keyword embeddings
            avg_keyword_embedding = np.mean(valid_keyword_embeddings, axis=0)
            
            # Combine embeddings with weights
            combined_embedding = np.concatenate([
                clip_embedding * clip_weight,
                avg_keyword_embedding * keyword_weight,
                artist_embedding * artist_weight
            ])
            
            images_data.append({
                'image_id': row['image_id'],
                'combined_embedding': combined_embedding,
                'artist_name': primary_artist,
                'artwork_title': row['value']
            })
            
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            # Skip malformed entries
            continue
    
    return images_data


def _get_clip_embeddings(db, image_ids):
    """Get CLIP embeddings for given image IDs."""
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
        embedding_data = row['embedding']
        if isinstance(embedding_data, bytes):
            embedding = np.frombuffer(embedding_data, dtype=np.float32)
        else:
            try:
                embedding = np.array(embedding_data, dtype=np.float32)
            except (ValueError, TypeError):
                continue
        embeddings[row['image_id']] = embedding
    
    return embeddings


def _get_value_embeddings(db, entry_ids):
    """Get value embeddings for given entry IDs (artists and keywords)."""
    if not entry_ids:
        return {}
    
    placeholders = ','.join(['?' for _ in entry_ids])
    query = f"""
        SELECT id, embedding 
        FROM vec_value_features 
        WHERE id IN ({placeholders})
    """
    
    cursor = db.execute(query, entry_ids)
    results = cursor.fetchall()
    
    embeddings = {}
    for row in results:
        embedding_data = row['embedding']
        if isinstance(embedding_data, bytes):
            embedding = np.frombuffer(embedding_data, dtype=np.float32)
        else:
            try:
                embedding = np.array(embedding_data, dtype=np.float32)
            except (ValueError, TypeError):
                continue
        embeddings[row['id']] = embedding
    
    return embeddings


def _lookup_artist_id(db, artist_name):
    """Look up artist entry_id from text_entries table."""
    query = """
        SELECT entry_id 
        FROM text_entries 
        WHERE value = ? AND isArtist = 1
    """
    
    cursor = db.execute(query, (artist_name,))
    result = cursor.fetchone()
    return result['entry_id'] if result else None


def get_2d_positions_umap(sampled_data, random_state=42, n_neighbors=15, min_dist=0.1):
    """
    Transform high-dimensional embeddings to 2D coordinates using UMAP.
    
    Args:
        sampled_data: List of dictionaries with 'combined_embedding' key
        random_state: Random state for reproducibility
        n_neighbors: UMAP parameter controlling local vs global structure
        min_dist: UMAP parameter controlling how tightly points are packed
    
    Returns:
        numpy array of 2D coordinates, shape (n_samples, 2)
    """
    # Stack all embeddings into a matrix
    embeddings_matrix = np.vstack([data['combined_embedding'] for data in sampled_data])
    
    # Create UMAP reducer
    umap_reducer = umap.UMAP(
        n_components=2,
        random_state=random_state,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric='cosine'  # Use cosine distance for embeddings
    )
    
    # Transform to 2D
    positions_2d = umap_reducer.fit_transform(embeddings_matrix)
    
    # Normalize to [-1, 1] range for consistent visualization
    positions_2d = 2 * (positions_2d - positions_2d.min(axis=0)) / (positions_2d.max(axis=0) - positions_2d.min(axis=0)) - 1
    
    return positions_2d


def compute_hierarchical_clusters(positions_2d, level_1_clusters=5, level_2_clusters=15, level_3_clusters=45, method='ward'):
    """
    Compute hierarchical clustering on 2D positions to create spatially coherent tree structure.
    
    Args:
        positions_2d: 2D coordinates array, shape (n_samples, 2) 
        level_1_clusters: Number of level 1 clusters (major groups)
        level_2_clusters: Number of level 2 clusters (medium groups)
        level_3_clusters: Number of level 3 clusters (fine groups)
        method: Linkage method ('ward', 'complete', 'average', 'single')
    
    Returns:
        Dictionary with cluster assignments for each level
    """
    # Compute distance matrix using Euclidean distance on 2D positions
    # This ensures spatially coherent clusters for Voronoi regions
    distance_matrix = cosine_distances(positions_2d)  # Could also use euclidean_distances
    
    # Perform hierarchical clustering
    # Note: For ward method, we can use the raw 2D data directly
    if method == 'ward':
        linkage_matrix = linkage(positions_2d, method=method)
    else:
        # For other methods, use precomputed distances
        condensed_distances = squareform(distance_matrix, checks=False)
        linkage_matrix = linkage(condensed_distances, method=method)
    
    # Cut tree at different heights to get hierarchical zoom levels
    # Level 3 clusters are children of Level 2, which are children of Level 1
    level_1_labels = cut_tree(linkage_matrix, n_clusters=level_1_clusters).flatten()
    level_2_labels = cut_tree(linkage_matrix, n_clusters=level_2_clusters).flatten()
    level_3_labels = cut_tree(linkage_matrix, n_clusters=level_3_clusters).flatten()
    
    return {
        'level_1': level_1_labels,
        'level_2': level_2_labels,
        'level_3': level_3_labels,
        'linkage_matrix': linkage_matrix
    }


def generate_nested_voronoi_regions(positions_2d, hierarchical_clusters, level_1_clusters=5, level_2_clusters=15, level_3_clusters=45):
    """
    Generate hierarchical Voronoi regions where Level 2 regions are subsets of Level 1,
    and Level 3 regions are subsets of Level 2.
    
    Args:
        positions_2d: 2D coordinates array, shape (n_samples, 2)
        hierarchical_clusters: Dict with cluster assignments from compute_hierarchical_clusters
        level_1_clusters: Number of level 1 clusters
        level_2_clusters: Number of level 2 clusters  
        level_3_clusters: Number of level 3 clusters
    
    Returns:
        Dictionary with Voronoi regions for each level
    """
    # Get cluster assignments
    level_1_labels = hierarchical_clusters['level_1']
    level_2_labels = hierarchical_clusters['level_2'] 
    level_3_labels = hierarchical_clusters['level_3']
    
    # LEVEL 1: Standard Voronoi on full space
    level_1_regions = _generate_voronoi_for_level(positions_2d, level_1_labels, level_1_clusters)
    
    # LEVEL 2: Voronoi within each Level 1 region
    level_2_regions = []
    for l1_region in level_1_regions:
        l1_cluster_id = l1_region['cluster_id']
        
        # Find all points in this Level 1 cluster
        l1_mask = level_1_labels == l1_cluster_id
        l1_positions = positions_2d[l1_mask]
        l1_level2_labels = level_2_labels[l1_mask]
        
        if len(l1_positions) > 1:
            # Find unique Level 2 clusters within this Level 1 cluster
            unique_l2_clusters = np.unique(l1_level2_labels)
            
            # Generate Voronoi regions for Level 2 clusters within this L1 region
            l2_regions_in_l1 = _generate_constrained_voronoi(
                l1_positions, l1_level2_labels, unique_l2_clusters, l1_region['vertices']
            )
            
            # Add parent info and append
            for l2_region in l2_regions_in_l1:
                l2_region['parent_cluster_id'] = l1_cluster_id
                level_2_regions.append(l2_region)
    
    # LEVEL 3: Voronoi within each Level 2 region  
    level_3_regions = []
    for l2_region in level_2_regions:
        l2_cluster_id = l2_region['cluster_id']
        
        # Find all points in this Level 2 cluster
        l2_mask = level_2_labels == l2_cluster_id
        l2_positions = positions_2d[l2_mask]
        l2_level3_labels = level_3_labels[l2_mask]
        
        if len(l2_positions) > 1:
            # Find unique Level 3 clusters within this Level 2 cluster
            unique_l3_clusters = np.unique(l2_level3_labels)
            
            # Generate Voronoi regions for Level 3 clusters within this L2 region
            l3_regions_in_l2 = _generate_constrained_voronoi(
                l2_positions, l2_level3_labels, unique_l3_clusters, l2_region['vertices']
            )
            
            # Add parent info and append
            for l3_region in l3_regions_in_l2:
                l3_region['parent_cluster_id'] = l2_cluster_id
                level_3_regions.append(l3_region)
    
    return {
        'level_1': level_1_regions,
        'level_2': level_2_regions,
        'level_3': level_3_regions
    }


def _generate_voronoi_for_level(positions, cluster_labels, n_clusters):
    """Generate standard Voronoi regions for a clustering level."""
    regions = []
    
    # Calculate centroids for each cluster
    centroids = []
    cluster_info = []
    
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        if np.any(cluster_mask):
            cluster_points = positions[cluster_mask]
            centroid = np.mean(cluster_points, axis=0)
            centroids.append(centroid)
            cluster_info.append({
                'cluster_id': cluster_id,
                'point_count': np.sum(cluster_mask),
                'centroid': centroid
            })
    
    if len(centroids) < 3:
        # Not enough clusters for Voronoi - return simple regions
        for info in cluster_info:
            regions.append({
                'vertices': _create_simple_region(info['centroid']),
                'centroid': info['centroid'].tolist(),
                'cluster_id': info['cluster_id'],
                'point_count': info['point_count']
            })
        return regions
    
    # Create Voronoi diagram
    centroids = np.array(centroids)
    
    # Add boundary points to ensure bounded regions
    bounds = _get_data_bounds(positions)
    boundary_points = _create_boundary_points(bounds)
    
    voronoi_points = np.vstack([centroids, boundary_points])
    vor = Voronoi(voronoi_points)
    
    # Extract regions for the original centroids
    for i, info in enumerate(cluster_info):
        region_idx = vor.point_region[i]
        vertex_indices = vor.regions[region_idx]
        
        if len(vertex_indices) > 0 and -1 not in vertex_indices:
            vertices = vor.vertices[vertex_indices]
            # Clip to reasonable bounds
            vertices = _clip_vertices(vertices, bounds)
        else:
            # Fallback for infinite regions
            vertices = _create_simple_region(info['centroid'])
        
        regions.append({
            'vertices': vertices.tolist() if hasattr(vertices, 'tolist') else vertices,
            'centroid': info['centroid'].tolist() if hasattr(info['centroid'], 'tolist') else info['centroid'],
            'cluster_id': int(info['cluster_id']),
            'point_count': int(info['point_count'])
        })
    
    return regions


def _generate_constrained_voronoi(positions, cluster_labels, unique_clusters, boundary_vertices):
    """Generate Voronoi regions constrained within a boundary polygon."""
    regions = []
    
    if len(unique_clusters) <= 1:
        # Only one cluster - return the entire boundary as the region
        if len(unique_clusters) == 1:
            cluster_mask = cluster_labels == unique_clusters[0]
            centroid = np.mean(positions[cluster_mask], axis=0)
            regions.append({
                'vertices': boundary_vertices,
                'centroid': centroid.tolist() if hasattr(centroid, 'tolist') else centroid,
                'cluster_id': int(unique_clusters[0]),
                'point_count': int(np.sum(cluster_mask))
            })
        return regions
    
    # Calculate centroids for each cluster
    centroids = []
    cluster_info = []
    
    for cluster_id in unique_clusters:
        cluster_mask = cluster_labels == cluster_id
        if np.any(cluster_mask):
            cluster_points = positions[cluster_mask]
            centroid = np.mean(cluster_points, axis=0)
            centroids.append(centroid)
            cluster_info.append({
                'cluster_id': cluster_id,
                'point_count': np.sum(cluster_mask),
                'centroid': centroid
            })
    
    # For simplicity, divide the boundary region based on proximity to centroids
    # This is a simplified version - could be improved with proper constrained Voronoi
    for info in cluster_info:
        # Create a simple region around each centroid
        # For now, just use the boundary - in a full implementation,
        # you'd compute the intersection of Voronoi cells with the boundary polygon
        vertices = _create_simple_region_in_boundary(info['centroid'], boundary_vertices)
        
        regions.append({
            'vertices': vertices,
            'centroid': info['centroid'].tolist() if hasattr(info['centroid'], 'tolist') else info['centroid'],
            'cluster_id': int(info['cluster_id']),
            'point_count': int(info['point_count'])
        })
    
    return regions


def _get_data_bounds(positions):
    """Get bounding box of the data."""
    return {
        'min_x': float(np.min(positions[:, 0])) - 0.2,
        'max_x': float(np.max(positions[:, 0])) + 0.2,
        'min_y': float(np.min(positions[:, 1])) - 0.2,
        'max_y': float(np.max(positions[:, 1])) + 0.2
    }


def _create_boundary_points(bounds):
    """Create boundary points for Voronoi diagram."""
    margin = 0.5
    return np.array([
        [bounds['min_x'] - margin, bounds['min_y'] - margin],
        [bounds['max_x'] + margin, bounds['min_y'] - margin],
        [bounds['max_x'] + margin, bounds['max_y'] + margin],
        [bounds['min_x'] - margin, bounds['max_y'] + margin],
        # Add mid-edge points for better boundaries
        [(bounds['min_x'] + bounds['max_x'])/2, bounds['min_y'] - margin],
        [bounds['max_x'] + margin, (bounds['min_y'] + bounds['max_y'])/2],
        [(bounds['min_x'] + bounds['max_x'])/2, bounds['max_y'] + margin],
        [bounds['min_x'] - margin, (bounds['min_y'] + bounds['max_y'])/2]
    ])


def _clip_vertices(vertices, bounds):
    """Clip vertices to reasonable bounds."""
    return np.clip(vertices, 
                   [bounds['min_x'], bounds['min_y']], 
                   [bounds['max_x'], bounds['max_y']])


def _create_simple_region(centroid, size=0.2):
    """Create a simple square region around a centroid."""
    return [
        [centroid[0] - size, centroid[1] - size],
        [centroid[0] + size, centroid[1] - size],
        [centroid[0] + size, centroid[1] + size],
        [centroid[0] - size, centroid[1] + size]
    ]


def _create_simple_region_in_boundary(centroid, boundary_vertices):
    """Create a simple region within boundary constraints."""
    # For now, just return a small region around the centroid
    # In a full implementation, this would properly intersect with the boundary
    return _create_simple_region(centroid, size=0.1)


def format_for_frontend(sampled_data, positions_2d, hierarchical_clusters, hierarchical_regions, 
                       level_1_clusters=5, level_2_clusters=15, level_3_clusters=45,
                       clip_weight=2.0, keyword_weight=1.0, artist_weight=3.0,
                       clustering_method="ward", umap_params=None):
    """
    Format hierarchical clustering and Voronoi data for frontend consumption.
    
    Args:
        sampled_data: List of image data dictionaries
        positions_2d: 2D coordinates from UMAP
        hierarchical_clusters: Dict with cluster assignments for each level
        hierarchical_regions: Dict with Voronoi regions for each level
        level_1_clusters: Number of level 1 clusters
        level_2_clusters: Number of level 2 clusters  
        level_3_clusters: Number of level 3 clusters
        clip_weight, keyword_weight, artist_weight: Embedding weights used
        clustering_method: Method used for hierarchical clustering
        umap_params: UMAP parameters used
    
    Returns:
        Dictionary formatted for frontend consumption
    """
    if umap_params is None:
        umap_params = {'n_components': 2, 'random_state': 42}
    
    # Format image points with hierarchical cluster info
    image_points = []
    for idx, (data, coord) in enumerate(zip(sampled_data, positions_2d)):
        point = {
            'entryId': data['image_id'],
            'x': float(coord[0]),
            'y': float(coord[1]),
            'artworkData': {
                'image_id': data['image_id'],
                'value': data.get('artwork_title', ''),
                'artist_names': [data['artist_name']] if data['artist_name'] else []
            },
            'artistData': {
                'name': data['artist_name']
            }
        }
        
        # Add hierarchical cluster info if available
        if hierarchical_clusters:
            level_1_labels = hierarchical_clusters.get('level_1', [])
            level_2_labels = hierarchical_clusters.get('level_2', [])
            level_3_labels = hierarchical_clusters.get('level_3', [])
            
            point['hierarchicalClusters'] = {
                'level_1': int(level_1_labels[idx]) if idx < len(level_1_labels) else 0,
                'level_2': int(level_2_labels[idx]) if idx < len(level_2_labels) else 0,
                'level_3': int(level_3_labels[idx]) if idx < len(level_3_labels) else 0
            }
        
        image_points.append(point)
    
    # Format regions for each level
    regions = {
        'level_1': _format_regions_for_level(hierarchical_regions.get('level_1', []), 1),
        'level_2': _format_regions_for_level(hierarchical_regions.get('level_2', []), 2),
        'level_3': _format_regions_for_level(hierarchical_regions.get('level_3', []), 3)
    }
    
    frontend_data = {
        'count': len(sampled_data),
        'generationParams': {
            'algorithm': "UMAP + hierarchical + Voronoi",
            'hierarchical_params': {
                'method': clustering_method,
                'level_1_clusters': level_1_clusters,
                'level_2_clusters': level_2_clusters,
                'level_3_clusters': level_3_clusters
            },
            'umap_params': umap_params,
            'embedding_weights': {
                'clip_weight': clip_weight,
                'keyword_weight': keyword_weight,
                'artist_weight': artist_weight
            }
        },
        'imagePoints': image_points,
        'regions': regions,
        'success': True
    }
    
    return frontend_data


def _format_regions_for_level(regions_list, level):
    """Format Voronoi regions for a specific hierarchical level."""
    formatted_regions = []
    
    for idx, region in enumerate(regions_list):
        if region and 'vertices' in region:
            # Ensure all values are JSON serializable
            formatted_region = {
                'id': f"level_{level}_region_{idx}",
                'level': int(level),
                'vertices': region['vertices'],  # Should already be lists
                'centroid': [float(x) for x in region.get('centroid', [0, 0])],
                'area': float(region.get('area', 0)),
                'point_count': int(region.get('point_count', 0))
            }
            formatted_regions.append(formatted_region)
    
    return formatted_regions


# Define the blueprint for hierarchical maps
hierarchical_map_api_bp = Blueprint('hierarchical_map', __name__)

@hierarchical_map_api_bp.route('/test', methods=['GET'])
def test_hierarchical():
    """Test endpoint to verify hierarchical map blueprint is working."""
    return jsonify({
        'success': True,
        'message': 'Hierarchical Map API is working!'
    })

@hierarchical_map_api_bp.route('/hierarchical-check')
def hierarchical_check_page():
    """Serve the hierarchical map API check page."""
    return render_template('hierarchical_map_check.html')

@hierarchical_map_api_bp.route('/generate_hierarchical_voronoi', methods=['GET'])
def handle_hierarchical_voronoi_request():
    """
    Generate hierarchical Voronoi maps with 3 levels of clustering.
    
    Expected URL parameters:
    - sample_size: number of images to sample (default: 500)
    - level_1_clusters: number of level 1 clusters (default: 5)
    - level_2_clusters: number of level 2 clusters (default: 15)  
    - level_3_clusters: number of level 3 clusters (default: 45)
    - clip_weight: weight for CLIP embeddings (default: 2.0)
    - keyword_weight: weight for keyword embeddings (default: 1.0)
    - artist_weight: weight for artist embeddings (default: 3.0)
    - clustering_method: method for hierarchical clustering (default: 'ward')
    - umap_random_state: random state for UMAP (default: 42)
    - debug: enable debug output (default: 'false')
    
    Returns JSON response with hierarchical Voronoi map data.
    """
    print("Received request for hierarchical Voronoi map generation...")
    
    try:
        # Parse parameters
        sample_size = int(request.args.get('sample_size', 500))
        level_1_clusters = int(request.args.get('level_1_clusters', 5))
        level_2_clusters = int(request.args.get('level_2_clusters', 15))
        level_3_clusters = int(request.args.get('level_3_clusters', 45))
        clip_weight = float(request.args.get('clip_weight', 2.0))
        keyword_weight = float(request.args.get('keyword_weight', 1.0))
        artist_weight = float(request.args.get('artist_weight', 3.0))
        clustering_method = request.args.get('clustering_method', 'ward')
        umap_random_state = int(request.args.get('umap_random_state', 42))
        debug = request.args.get('debug', 'false').lower() == 'true'
        
        def dprint(msg):
            if debug:
                print(f"[DEBUG] {msg}")
        
        dprint(f"Parameters: sample_size={sample_size}, levels=({level_1_clusters},{level_2_clusters},{level_3_clusters})")
        dprint(f"Weights: clip={clip_weight}, keyword={keyword_weight}, artist={artist_weight}")
        
        # STEP 1: Extract combined embeddings
        dprint("Step 1: Extracting combined embeddings...")
        step1_data = extract_combined_embeddings(
            sample_size=sample_size,
            clip_weight=clip_weight,
            keyword_weight=keyword_weight,
            artist_weight=artist_weight
        )
        
        if not step1_data:
            return jsonify({
                'success': False,
                'error': 'No valid embeddings found',
                'step': 1
            })
        
        dprint(f"Step 1 complete: {len(step1_data)} images with embeddings")
        
        # STEP 2: Skip stratified sampling for now
        dprint("Step 2: Skipping stratified sampling...")
        step2_data = step1_data
        dprint(f"Step 2 complete: {len(step2_data)} images (no sampling applied)")
        
        # STEP 3: Get 2D positions with UMAP
        dprint("Step 3: Computing 2D positions with UMAP...")
        step3_positions = get_2d_positions_umap(
            sampled_data=step2_data,
            random_state=umap_random_state
        )
        dprint(f"Step 3 complete: {step3_positions.shape} 2D positions")
        
        # STEP 4: Compute hierarchical clusters
        dprint("Step 4: Computing hierarchical clusters on 2D positions...")
        step4_clusters = compute_hierarchical_clusters(
            positions_2d=step3_positions,
            level_1_clusters=level_1_clusters,
            level_2_clusters=level_2_clusters,
            level_3_clusters=level_3_clusters,
            method=clustering_method
        )
        dprint(f"Step 4 complete: hierarchical clusters assigned")
        
        # STEP 5: Generate nested Voronoi regions
        dprint("Step 5: Generating nested Voronoi regions...")
        step5_regions = generate_nested_voronoi_regions(
            positions_2d=step3_positions,
            hierarchical_clusters=step4_clusters,
            level_1_clusters=level_1_clusters,
            level_2_clusters=level_2_clusters,
            level_3_clusters=level_3_clusters
        )
        dprint(f"Step 5 complete: {len(step5_regions['level_1'])}+{len(step5_regions['level_2'])}+{len(step5_regions['level_3'])} regions")
        
        # STEP 6: Format for frontend
        dprint("Step 6: Formatting for frontend...")
        step6_frontend = format_for_frontend(
            sampled_data=step2_data,
            positions_2d=step3_positions,
            hierarchical_clusters=step4_clusters,
            hierarchical_regions=step5_regions,
            level_1_clusters=level_1_clusters,
            level_2_clusters=level_2_clusters,
            level_3_clusters=level_3_clusters,
            clip_weight=clip_weight,
            keyword_weight=keyword_weight,
            artist_weight=artist_weight,
            clustering_method=clustering_method,
            umap_params={'n_components': 2, 'random_state': umap_random_state}
        )
        
        dprint(f"Step 6 complete: formatted {step6_frontend['count']} points with {len(step6_frontend['regions']['level_1'])} L1 regions")
        
        return jsonify(step6_frontend)
        
    except Exception as e:
        print(f"Error in hierarchical Voronoi generation: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'debug_info': traceback.format_exc() if debug else None
        })