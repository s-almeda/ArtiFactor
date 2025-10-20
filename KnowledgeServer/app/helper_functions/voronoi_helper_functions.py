"""
Helper functions for Voronoi-based hierarchical maps

This module contains geometric and computational utilities for:
- Polygon vertex processing and sorting
- Centroid calculation
- Clipping infinite Voronoi regions
- Region adjacency and optimal pairing calculations
"""

import math
import numpy as np
from shapely.geometry import Polygon, Point
import ast
import json
import traceback
from helper_functions import helperfunctions as hf  # helper functions including preprocess_text






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

def create_optimal_pairs_compactness(region_ids, boundary_lengths, polygons, dprint):
    """
    Strategy 4: Pairing based on merged region compactness.
    Prioritizes pairs that would create more compact (circle-like) merged regions.
    
    Args:
        region_ids: List of region IDs
        boundary_lengths: Dict mapping (region_a, region_b) tuples to boundary lengths
        polygons: Dict mapping region_id to Polygon objects
        dprint: Debug print function
    
    Returns:
        List of [region_a, region_b] pairs
    """
    # New strategy: merge all adjacent region pairs, prioritize by average point proximity (compactness)

    pair_candidates = []
    # Build a lookup for region points (centroids or all points if available)
    # For now, use centroids from polygons
    region_centroids = {rid: np.array(polygons[rid].centroid.coords[0]) for rid in region_ids if rid in polygons}

    # Find all adjacent pairs
    for i, region_a in enumerate(region_ids):
        for region_b in region_ids[i+1:]:
            if region_a == region_b:
                continue
            boundary_key = (min(region_a, region_b), max(region_a, region_b))
            if boundary_key in boundary_lengths:
                # Get boundary length for this pair
                bl = boundary_lengths[boundary_key]
                
                # Calculate average centroid distance (proxy for compactness)
                if region_a in region_centroids and region_b in region_centroids:
                    dist = np.linalg.norm(region_centroids[region_a] - region_centroids[region_b])
                else:
                    dist = float('inf')
                
                # Prefer longer boundaries (more significant adjacency)
                # For very similar centroid distances, boundary length becomes the deciding factor
                # This is achieved by using boundary length as a tie-breaker
                # We normalize the boundary length to be between 0 and 1
                normalized_bl = min(bl / 5.0, 1.0)  # Cap at 1.0, assuming most boundaries are < 5.0 units
                
                # Score is primarily based on distance but adjusted slightly by boundary length
                # Lower score = better candidate (prioritize close centroids with substantial boundaries)
                score = dist * (1.05 - normalized_bl * 0.1)  # Small adjustment factor based on boundary
                
                pair_candidates.append({
                    'pair': [region_a, region_b],
                    'distance': dist,
                    'boundary_length': bl,
                    'score': score
                })
                
                dprint(f"Candidate pair: {region_a}-{region_b}, centroid_dist={dist:.4f}, boundary_length={bl:.4f}, score={score:.4f}")

    # Sort all pairs by score (lowest/best score first)
    pair_candidates.sort(key=lambda x: x['score'])

    # Greedily select pairs, ensuring no region is merged more than once per pass
    paired_regions = set()
    optimal_pairs = []
    for candidate in pair_candidates:
        a, b = candidate['pair']
        if a in paired_regions or b in paired_regions:
            continue
        optimal_pairs.append([a, b])
        paired_regions.add(a)
        paired_regions.add(b)
        dprint(f"✓ Paired regions {a} and {b} (centroid distance: {candidate['distance']:.4f}, boundary: {candidate['boundary_length']:.4f}, score: {candidate['score']:.4f})")

    return optimal_pairs

def heuristic_cluster_count(n_points):
   """
   Rule of thumb: roughly sqrt(n/2) clusters
   """
   return max(3, min(15, int(np.sqrt(n_points / 2))))


def get_salient_keywords(db, min_works=5, max_works=500, n=None):
    """
    Calculate keyword salience based on marginal contribution to artwork coverage.
    Uses greedy selection to maximize total coverage while minimizing overlap.
    Return: [{keyword, entry_id, count, salience_score, image_ids}] sorted by selection order.
    If n is provided, only the top n results are returned.
    """
    
    # Step 1: Get all keywords and their associated images
    cursor = db.execute("""
        SELECT entry_id, value as keyword, images
        FROM text_entries
        WHERE images IS NOT NULL AND TRIM(images) != '' 
    """)
    keyword_rows = cursor.fetchall()

    # Step 2: Build mapping of keyword to unique image ids
    keyword_candidates = []
    all_image_ids = set()
    
    for row in keyword_rows:
        entry_id = row['entry_id']
        keyword = row['keyword']
        try:
            image_ids = set(ast.literal_eval(row['images']))
        except Exception:
            image_ids = set()
        if not image_ids:
            continue
            
        count = len(image_ids)
        if count < min_works or count > max_works:
            continue
            
        keyword_candidates.append({
            'keyword': keyword,
            'entry_id': entry_id,
            'image_ids': image_ids,
            'count': count
        })
        all_image_ids.update(image_ids)

    total_artworks = len(all_image_ids)
    if total_artworks == 0:
        return []

    # Step 3: Calculate initial salience scores for all candidates
    for item in keyword_candidates:
        # True distinctiveness = how unique this keyword's artworks are
        distinctiveness = item['count'] / total_artworks if total_artworks else 0
        item['initial_salience'] = item['count'] * distinctiveness

    # Step 4: Greedy selection with overlap penalty
    selected_keywords = []
    covered_artworks = set()
    remaining_candidates = keyword_candidates.copy()
    
    # Determine how many keywords to select
    target_count = n if n is not None else len(keyword_candidates)
    
    while len(selected_keywords) < target_count and remaining_candidates:
        best_score = -1
        best_idx = -1
        best_keyword = None
        
        for i, candidate in enumerate(remaining_candidates):
            # Calculate marginal contribution (new artworks only)
            new_artworks = candidate['image_ids'] - covered_artworks
            new_count = len(new_artworks)
            
            if new_count == 0:
                # This keyword adds no new artworks
                marginal_score = 0
            else:
                # Score based on new artworks added
                # Higher weight for keywords that add substantial new coverage
                coverage_boost = new_count / total_artworks
                
                # Bonus for keywords that had high initial salience
                initial_salience_factor = candidate['initial_salience'] / (candidate['count'] ** 2 / total_artworks)
                
                marginal_score = new_count * coverage_boost * initial_salience_factor
            
            if marginal_score > best_score:
                best_score = marginal_score
                best_idx = i
                best_keyword = candidate
        
        # Select the best keyword
        if best_idx >= 0 and best_score > 0:
            selected_keyword = remaining_candidates.pop(best_idx)
            new_artworks = selected_keyword['image_ids'] - covered_artworks
            
            # Update the selected keyword with final stats
            selected_keyword['new_artworks_added'] = len(new_artworks)
            selected_keyword['salience_score'] = best_score
            selected_keyword['total_coverage'] = len(covered_artworks | selected_keyword['image_ids'])
            
            selected_keywords.append(selected_keyword)
            covered_artworks.update(selected_keyword['image_ids'])
            
            print(f"Selected '{selected_keyword['keyword']}': +{len(new_artworks)} new artworks, "
                  f"total coverage: {len(covered_artworks)}/{total_artworks} "
                  f"({100*len(covered_artworks)/total_artworks:.1f}%)")
        else:
            # No more keywords contribute new artworks
            break
    
    # Step 5: Clean up the return format
    results = []
    for item in selected_keywords:
        results.append({
            'keyword': item['keyword'],
            'entry_id': item['entry_id'],
            'count': item['count'],
            'salience_score': item['salience_score'],
            'image_ids': item['image_ids'],  # Include for downstream deduplication
            'new_artworks_added': item['new_artworks_added'],
            'coverage_contribution': item['new_artworks_added'] / total_artworks
        })
    
    print(f"\nFinal selection: {len(results)} keywords covering {len(covered_artworks)}/{total_artworks} artworks "
          f"({100*len(covered_artworks)/total_artworks:.1f}% coverage)")
    
    return results



def get_keyword_biased_embedding(artwork_id, main_keyword_id, db, weights={
   'clip': 0.5,
   'resnet': 0.0,
   'keyword_semantic': 0.5,  # Combined text semantic weight
   'keyword_bias': 0.7       # How much to bias toward main keyword vs other keywords (0.7 = 70% main, 30% others)
}):
   """
   Create a multimodal embedding for an artwork biased by its main keyword and other keywords.
   Creates a unified text semantic embedding that combines main and other keywords.
   
   Args:
       artwork_id: image_id from image_entries table
       main_keyword_id: entry_id from text_entries table (the salient keyword we're clustering around)
       db: database connection
       weights: dict specifying weight for each embedding type:
           - clip: weight for CLIP visual embedding (1024d)
           - resnet: weight for ResNet visual embedding (2048d)
           - keyword_semantic: weight for unified text semantic embedding (384d)
           - keyword_bias: how much to weight main keyword vs others (0-1, default 0.7)
   Returns:
       numpy array: normalized combined embedding with consistent shape, or None if required components missing
   """
   
   # Use get() instead of pop() to avoid modifying the weights dict
   debug = weights.get('debug', False)
   def dprint(*args, **kwargs):
         if debug:
              print(*args, **kwargs)
   # Extract keyword bias parameter
   keyword_bias = weights.get('keyword_bias', 0.7)
   
   # Validate and normalize weights
   active_weights = {k: v for k, v in weights.items() if v > 0 and k in ['clip', 'resnet', 'keyword_semantic']}
   if not active_weights:
       dprint("No active weights, returning None")
       return None

   total_weight = sum(active_weights.values())
   normalized_weights = {k: v/total_weight for k, v in active_weights.items()}
   # dprint(f"Normalized weights: {normalized_weights}")
   # dprint(f"Keyword bias: {keyword_bias} (main) / {1-keyword_bias} (others)")

   # Store components
   embedding_components = []
   missing_required = []

   # 1. CLIP embedding (1024d)
   if 'clip' in normalized_weights:
       clip_embs = hf.get_clip_embeddings(db, [artwork_id])
       if artwork_id in clip_embs:
           clip_emb = np.array(clip_embs[artwork_id])
           if np.linalg.norm(clip_emb) > 0:
               clip_norm = clip_emb / np.linalg.norm(clip_emb)
               embedding_components.append(normalized_weights['clip'] * clip_norm)
               # dprint(f"CLIP shape: {clip_norm.shape}")
           else:
               dprint("CLIP embedding is zero vector")
               missing_required.append('clip')
       else:
           dprint("No CLIP embedding found")
           missing_required.append('clip')

   # 2. ResNet embedding (2048d)
   if 'resnet' in normalized_weights:
       resnet_embs = hf.get_resnet_embeddings(db, [artwork_id])
       if artwork_id in resnet_embs:
           resnet_emb = np.array(resnet_embs[artwork_id])
           if np.linalg.norm(resnet_emb) > 0:
               resnet_norm = resnet_emb / np.linalg.norm(resnet_emb)
               embedding_components.append(normalized_weights['resnet'] * resnet_norm)
               # dprint(f"ResNet shape: {resnet_norm.shape}")
           else:
               dprint("ResNet embedding is zero vector")
               missing_required.append('resnet')
       else:
           dprint("No ResNet embedding found")
           missing_required.append('resnet')

   # 3. Unified keyword semantic embedding (384d)
   if 'keyword_semantic' in normalized_weights:
       # Get main keyword value embedding
       main_value_embs = hf.get_value_embeddings(db, [main_keyword_id])
       if main_keyword_id not in main_value_embs:
           dprint(f"No main keyword value embedding found for {main_keyword_id}")
           missing_required.append('keyword_semantic')
       else:
           main_value_emb = np.array(main_value_embs[main_keyword_id])
           if np.linalg.norm(main_value_emb) == 0:
               dprint("Main keyword value embedding is zero vector")
               missing_required.append('keyword_semantic')
           else:
               # Start with main keyword embedding
               main_value_norm = main_value_emb / np.linalg.norm(main_value_emb)
               
               # Try to get other keywords
               cursor = db.execute("""
                   SELECT relatedKeywordIds FROM image_entries WHERE image_id = ?
               """, (artwork_id,))
               row = cursor.fetchone()
               
               keyword_semantic_emb = main_value_norm  # Default to just main keyword
               
               if row and row['relatedKeywordIds']:
                   try:
                       related_keyword_ids = json.loads(row['relatedKeywordIds'])
                       other_keyword_ids = [kid for kid in related_keyword_ids if kid != main_keyword_id]
                       
                       if other_keyword_ids:
                           other_value_embs = hf.get_value_embeddings(db, other_keyword_ids)
                           valid_other_embs = [np.array(other_value_embs[kid]) 
                                             for kid in other_keyword_ids 
                                             if kid in other_value_embs and np.linalg.norm(np.array(other_value_embs[kid])) > 0]
                           
                           if valid_other_embs:
                               # Average other keyword embeddings
                               other_avg = np.mean(valid_other_embs, axis=0)
                               other_norm = other_avg / np.linalg.norm(other_avg)
                               
                               # Blend main and other keywords based on bias
                               keyword_semantic_emb = (keyword_bias * main_value_norm + 
                                                     (1 - keyword_bias) * other_norm)
                               keyword_semantic_emb = keyword_semantic_emb / np.linalg.norm(keyword_semantic_emb)
                               
                               # dprint(f"Blended {len(valid_other_embs)} other keywords with main keyword")
                           else:
                               dprint("No valid other keyword embeddings, using only main keyword")
                       else:
                           dprint("No other keywords, using only main keyword")
                   except json.JSONDecodeError:
                       dprint("Malformed relatedKeywordIds JSON, using only main keyword")
               else:
                   dprint("No relatedKeywordIds, using only main keyword")
               
               # Add the unified keyword semantic embedding
               embedding_components.append(normalized_weights['keyword_semantic'] * keyword_semantic_emb)
               # dprint(f"Keyword semantic shape: {keyword_semantic_emb.shape}")

   # Check if any required components are missing
   if missing_required:
       # dprint(f"Missing required components: {missing_required}, returning None")
       return None

   # Concatenate all components
   if not embedding_components:
       # dprint("No embedding components found, returning None")
       return None

   try:
       combined_embedding = np.concatenate(embedding_components)
   except Exception as e:
       # dprint(f"Error concatenating components: {e}")
       return None

   # Final normalization
   if np.linalg.norm(combined_embedding) == 0:
       # dprint("Combined embedding is zero vector, returning None")
       return None
   
   final_emb = combined_embedding / np.linalg.norm(combined_embedding)
   # dprint(f"Final embedding shape: {final_emb.shape}")
   return final_emb


def get_keyword_biased_embeddings(db, salient_keywords, weights=None):
    """
    Returns embeddings and artwork IDs in efficient numpy/list format.
    
    Args:
        db: Database connection
        salient_keywords: List of keyword dicts from get_salient_keywords
        weights: Optional dict of weights for embedding components
    
    Returns:
        {
            'embeddings': numpy array of shape (n_artworks, embedding_dim),
            'artworks': list of artwork IDs (strings),
            'keyword_map': dict mapping artwork_id -> keyword_entry_id
        }
    """
    
    # Set default weights if not provided
    if weights is None:
        weights = {
            'clip': 0.6,
            'resnet': 0.0,
            'keyword_semantic': 0.4,
            'keyword_bias': 0.7
        }

    # Simple data structures
    artwork_ids = []
    embeddings_list = []
    keyword_map = {}  # artwork_id -> keyword_entry_id
    
    debug = weights.get('debug', False)

    # Summary counters
    total_artworks = 0
    total_keywords = len(salient_keywords)
    multi_keyword_count = 0
    single_keyword_count = 0
    duplicate_count = 0
    unique_count = 0

    # Deduplication set
    seen_artworks = set()

    # Process each salient keyword
    for keyword_idx, keyword in enumerate(salient_keywords):
        entry_id = keyword['entry_id']
        keyword_name = keyword['keyword']

        # Use image_ids from keyword if available
        if 'image_ids' in keyword:
            artwork_ids_for_keyword = keyword['image_ids']
        else:
            # Fallback to database query
            cursor = db.execute(
                "SELECT images FROM text_entries WHERE entry_id = ?",
                (entry_id,)
            )
            row = cursor.fetchone()
            if not row or not row[0]:
                continue
            artwork_ids_for_keyword = json.loads(row[0])

        n_multi = 0
        n_single = 0
        processed = 0

        # Process each artwork
        for artwork_id in artwork_ids_for_keyword:
            if artwork_id in seen_artworks:
                duplicate_count += 1
                continue  # Skip duplicates

            embedding = get_keyword_biased_embedding(
                artwork_id=artwork_id,
                main_keyword_id=entry_id,
                db=db,
                weights=weights
            )

            if embedding is not None:
                artwork_ids.append(artwork_id)
                embeddings_list.append(embedding)
                keyword_map[artwork_id] = entry_id
                processed += 1
                seen_artworks.add(artwork_id)
                unique_count += 1

                # Count if artwork has multiple keywords (other than main)
                cursor = db.execute("SELECT relatedKeywordIds FROM image_entries WHERE image_id = ?", (artwork_id,))
                row = cursor.fetchone()
                if row and row['relatedKeywordIds']:
                    try:
                        related_keyword_ids = json.loads(row['relatedKeywordIds'])
                        if isinstance(related_keyword_ids, list) and len(related_keyword_ids) > 1:
                            n_multi += 1
                        else:
                            n_single += 1
                    except Exception:
                        n_single += 1
                else:
                    n_single += 1

        total_artworks += processed
        multi_keyword_count += n_multi
        single_keyword_count += n_single

        if debug:
            print(f"Keyword {keyword_idx+1}/{total_keywords} '{keyword_name}': {processed} artworks processed, {n_multi} with multiple keywords, {n_single} with only one keyword.")

    # Convert to numpy array
    embeddings_np = np.array(embeddings_list) if embeddings_list else np.array([])

    if debug:
        print(f"\nProcessed {unique_count} unique artworks (skipped {duplicate_count} duplicates) across {total_keywords} keywords.")
        print(f"Deduplication rate: {duplicate_count/(unique_count + duplicate_count)*100:.1f}% duplicates removed")
        print(f"Artworks with multiple keywords: {multi_keyword_count}")
        print(f"Artworks with only one keyword: {single_keyword_count}")
        if embeddings_np.size > 0:
            print(f"Final embeddings shape: {embeddings_np.shape}")

    return {
        'embeddings': embeddings_np,
        'artworks': artwork_ids,
        'keyword_map': keyword_map
    }

def generate_level2_level3(clusters_raw, voronoi_data, dprint):
    """
    Generate level 2 and 3 by iteratively merging adjacent regions.
    Works with dict format where cluster_id is the key.
    """
    # Start with copies of level 1 data
    current_clusters = clusters_raw.copy()
    current_voronoi = voronoi_data.copy()
    merge_history = []
    
    # Keep merging until we have ≤10 regions
    while len(current_clusters) > 10:
        # Find adjacent pairs
        adjacency_result = find_voronoi_adjacency_pairs(current_voronoi, dprint)
        
        if not adjacency_result['success'] or len(adjacency_result['adjacencyData']['optimalPairs']) == 0:
            dprint("No more pairs to merge")
            break
        
        # Get the optimal pairs
        optimal_pairs = adjacency_result['adjacencyData']['optimalPairs']
        
        # Merge the pairs
        merged_clusters, merged_voronoi = merge_paired_voronoi_regions(
            current_clusters,
            current_voronoi,
            optimal_pairs,
            dprint
        )
        
        if len(merged_clusters) >= len(current_clusters):
            dprint("No successful merges, stopping")
            break
            
        current_clusters = merged_clusters
        current_voronoi = merged_voronoi
        
        # Save state after adding representatives
        current_clusters_copy = current_clusters.copy()
        add_representative_artworks(current_clusters_copy)
        merge_history.append({
            'clusters': current_clusters_copy,
            'voronoi': current_voronoi.copy()
        })
        
        dprint(f"Merge iteration complete: {len(current_clusters)} clusters remaining")
    
    # Select level 2 and 3 from merge history
    if len(merge_history) >= 2:
        # Use first merge for level 2, last merge for level 3
        level2_data = merge_history[0]  # First merge result
        level3_data = merge_history[-1]  # Final merge result
    elif len(merge_history) == 1:
        # If only one merge, use original clusters for level 2
        level2_clusters = clusters_raw.copy()
        add_representative_artworks(level2_clusters)
        level2_data = {
            'clusters': level2_clusters,
            'voronoi': voronoi_data
        }
        level3_data = merge_history[0]  # Use the only merge for level 3
    else:
        # No successful merges - return copies with representatives added
        level2_clusters = current_clusters.copy()
        level3_clusters = current_clusters.copy()
        add_representative_artworks(level2_clusters)
        add_representative_artworks(level3_clusters)
        level2_data = {
            'clusters': level2_clusters,
            'voronoi': current_voronoi
        }
        level3_data = {
            'clusters': level3_clusters,
            'voronoi': current_voronoi
        }
        dprint("No successful merges, using original data for all levels")
    
    return level2_data, level3_data


def merge_paired_voronoi_regions(clusters_raw, voronoi_data, optimal_pairs, dprint):
    """
    Merge optimal pairs of adjacent regions into single regions.
    Works with dict format where cluster_id is the key.
    
    Args:
        clusters_raw: dict of cluster_id -> cluster data
        voronoi_data: dict of cluster_id -> voronoi cell data
        optimal_pairs: list of (cluster_id1, cluster_id2) tuples to merge
        dprint: Debug print function
    
    Returns:
        Merged clusters_raw and voronoi_data in the same dict format
    """
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    import numpy as np
    
    merged_clusters = {}
    merged_voronoi = {}
    merged_ids = set()
    
    # Get next available ID
    max_id = max(list(clusters_raw.keys()) + list(voronoi_data.keys()))
    next_id = max_id + 1
    
    # Process optimal pairs
    for region_a, region_b in optimal_pairs:
        if region_a in merged_ids or region_b in merged_ids:
            dprint(f"⚠ Skipping pair ({region_a}, {region_b}) - already merged")
            continue
        
        if region_a not in clusters_raw or region_b not in clusters_raw:
            dprint(f"⚠ Missing cluster data for pair ({region_a}, {region_b})")
            continue
            
        if region_a not in voronoi_data or region_b not in voronoi_data:
            dprint(f"⚠ Missing voronoi data for pair ({region_a}, {region_b})")
            continue
        
        try:
            # Create polygons from vertices
            poly_a = Polygon(voronoi_data[region_a]['vertices'])
            poly_b = Polygon(voronoi_data[region_b]['vertices'])
            
            # Check if they truly share a boundary
            if not poly_a.touches(poly_b):
                dprint(f"⚠ Regions {region_a} and {region_b} don't share a boundary, skipping")
                continue
            
            # Merge the polygons
            merged_polygon = unary_union([poly_a, poly_b])
            
            # Handle different geometry types
            if merged_polygon.geom_type == 'Polygon':
                merged_vertices = list(merged_polygon.exterior.coords[:-1])
                merged_centroid = np.array([merged_polygon.centroid.x, merged_polygon.centroid.y])
            elif merged_polygon.geom_type == 'MultiPolygon':
                # Take the largest polygon
                largest_poly = max(merged_polygon.geoms, key=lambda p: p.area)
                merged_vertices = list(largest_poly.exterior.coords[:-1])
                merged_centroid = np.array([largest_poly.centroid.x, largest_poly.centroid.y])
                dprint(f"⚠ MultiPolygon result, using largest component")
            else:
                dprint(f"⚠ Unexpected geometry type: {merged_polygon.geom_type}, skipping")
                continue
            
            # Merge cluster data
            cluster_a = clusters_raw[region_a]
            cluster_b = clusters_raw[region_b]
            
            merged_clusters[next_id] = {
                'artwork_ids': cluster_a['artwork_ids'] + cluster_b['artwork_ids'],
                'indices': np.concatenate([cluster_a['indices'], cluster_b['indices']]),
                'centroid': merged_centroid,
                'coordinates': np.vstack([cluster_a['coordinates'], cluster_b['coordinates']]),
                'size': cluster_a['size'] + cluster_b['size'],
                'child_clusters': [region_a, region_b]  # Track what was merged
            }
            
            # Merge voronoi data
            merged_voronoi[next_id] = {
                'vertices': merged_vertices
            }
            
            merged_ids.add(region_a)
            merged_ids.add(region_b)
            dprint(f"✓ Merged regions {region_a} and {region_b} into new region {next_id}")
            next_id += 1
            
        except Exception as e:
            dprint(f"⚠ Failed to merge regions {region_a} and {region_b}: {e}")
            import traceback
            traceback.print_exc()
    
    # Add unmerged regions
    for cluster_id in clusters_raw:
        if cluster_id not in merged_ids:
            merged_clusters[cluster_id] = clusters_raw[cluster_id].copy()
            merged_voronoi[cluster_id] = voronoi_data[cluster_id].copy()
    
    dprint(f"Merge complete: {len(clusters_raw)} regions -> {len(merged_clusters)} regions")
    
    return merged_clusters, merged_voronoi

def build_raw_clusters(cluster_labels, coordinates_2d, embeddings, artwork_ids, n_clusters):
    """
    Build cluster structure keeping everything as numpy arrays.
    
    Args:
        cluster_labels: numpy array of cluster assignments
        coordinates_2d: numpy array of 2D coordinates (global layout)
        embeddings: numpy array of high-dimensional embeddings
        artwork_ids: list of artwork IDs (not full metadata)
        n_clusters: number of clusters
    
    Returns:
        dict of cluster_id -> cluster_data
    """
    clusters = {}
    

    for cluster_id in range(n_clusters):
        # Get mask for this cluster
        cluster_mask = cluster_labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        if len(cluster_indices) == 0:  # Skip empty clusters
            continue
        # Extract cluster-specific data
        cluster_coords_2d = coordinates_2d[cluster_mask]
        cluster_embeddings = embeddings[cluster_mask]
        cluster_artwork_ids = [artwork_ids[i] for i in cluster_indices]
        # Find centroid in 2D space (for Voronoi)
        centroid_2d = np.mean(cluster_coords_2d, axis=0)
        clusters[cluster_id] = {
            'artwork_ids': cluster_artwork_ids,
            'indices': cluster_indices,
            'centroid': centroid_2d,  # Keep as numpy array
            # representative_ids will be added below
            'coordinates': cluster_coords_2d,
            'embeddings': cluster_embeddings,
            'size': len(cluster_indices)
        }
    clusters = add_representative_artworks(clusters)
    return clusters

def find_voronoi_adjacency_pairs(voronoi_data, dprint):
    """
    Find and identify optimal pairs of adjacent Voronoi regions.
    Works with dict format where cluster_id is the key.
    
    Args:
        voronoi_data: dict of cluster_id -> {'vertices': [...]}
        dprint: Debug print function
    
    Returns:
        Dict with adjacency analysis results
    """
    from shapely.geometry import Polygon
    from shapely.strtree import STRtree
    import numpy as np
    
    try:
        cluster_ids = list(voronoi_data.keys())
        k = len(cluster_ids)
        
        if k < 2:
            return {
                'success': False,
                'error': 'Need at least 2 regions for adjacency analysis'
            }
        
        dprint(f"Analyzing adjacency for {k} Voronoi regions...")
        
        # Step 1: Create Shapely polygons from Voronoi vertices
        polygons = {}
        for cluster_id in cluster_ids:
            vertices = voronoi_data[cluster_id]['vertices']
            if len(vertices) < 3:
                dprint(f"⚠ Region {cluster_id} has insufficient vertices ({len(vertices)}), skipping")
                continue
            try:
                polygon = Polygon(vertices)
                if polygon.is_valid and not polygon.is_empty:
                    polygons[cluster_id] = polygon
                else:
                    dprint(f"⚠ Polygon for region {cluster_id} is invalid or empty")
            except Exception as e:
                dprint(f"⚠ Failed to create polygon for region {cluster_id}: {e}")
        
        if len(polygons) < 2:
            return {
                'success': False,
                'error': 'Need at least 2 valid polygons for adjacency analysis'
            }
        
        # Step 2: Find adjacent pairs
        boundary_lengths = {}
        adjacent_pairs = []
        
        # Create spatial index for efficiency
        polygon_list = list(polygons.values())
        polygon_ids = list(polygons.keys())
        spatial_index = STRtree(polygon_list)
        
        # Check each polygon for neighbors
        for i, cluster_id in enumerate(polygon_ids):
            poly = polygons[cluster_id]
            
            # Find potential neighbors
            potential_neighbors_idx = spatial_index.query(poly.buffer(0.01))
            
            for idx in potential_neighbors_idx:
                neighbor_id = polygon_ids[idx]
                
                # Skip self and already checked pairs
                if neighbor_id <= cluster_id:
                    continue
                
                neighbor_poly = polygons[neighbor_id]
                
                # Check if they touch
                if poly.touches(neighbor_poly):
                    # Calculate boundary length
                    try:
                        intersection = poly.intersection(neighbor_poly)
                        if hasattr(intersection, 'length'):
                            boundary_length = max(intersection.length, 0.01)
                        else:
                            boundary_length = 0.01
                    except:
                        boundary_length = 0.01
                    
                    adjacent_pairs.append((cluster_id, neighbor_id))
                    boundary_lengths[(cluster_id, neighbor_id)] = boundary_length
                    dprint(f"✓ Regions {cluster_id} and {neighbor_id} are adjacent (boundary: {boundary_length:.3f})")
        
        # Step 3: Create optimal pairs
        optimal_pairs = create_optimal_pairs_compactness(
            polygon_ids, boundary_lengths, polygons, dprint
        )
        
        dprint(f"\nADJACENCY SUMMARY:")
        dprint(f"- Total regions: {k}")
        dprint(f"- Adjacent pairs: {len(adjacent_pairs)}")
        dprint(f"- Optimal pairs for merging: {len(optimal_pairs)}")
        
        return {
            'success': True,
            'adjacencyData': {
                'optimalPairs': optimal_pairs,
                'boundaryLengths': boundary_lengths,
                'adjacentPairs': adjacent_pairs
            },
            'basicStats': {
                'totalRegions': k,
                'totalAdjacencies': len(adjacent_pairs),
                'optimalPairs': len(optimal_pairs)
            }
        }
        
    except Exception as e:
        dprint(f"Error in adjacency analysis: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }

def add_representative_artworks(clusters):
    """
    For each cluster in the clusters dict, find the top 3 representative artwork IDs (closest to centroid)
    and add them as 'representative_ids' (list) in the cluster dict.
    """
    for _, cluster in clusters.items():
        coords = cluster['coordinates']
        centroid = cluster['centroid']
        artwork_ids = cluster['artwork_ids']
        if len(artwork_ids) == 0:
            cluster['representative_ids'] = []
            continue
        distances = np.linalg.norm(coords - centroid, axis=1)
        top_3_indices = np.argsort(distances)[:3]
        representative_ids = [artwork_ids[i] for i in top_3_indices]
        cluster['representative_ids'] = representative_ids
    return clusters

def generate_voronoi_cells(clusters_raw, coordinates_2d, bounding_box=None):
    """
    Generate Voronoi cells from cluster centroids.
    
    Args:
        clusters_raw: dict of cluster_id -> cluster data with 'centroid' as numpy array
        coordinates_2d: global coordinates for bounds checking
        bounding_box: optional dict with min_x, max_x, min_y, max_y
    
    Returns:
        dict of cluster_id -> {'vertices': numpy array of shape (n_vertices, 2)}
    """
    from scipy.spatial import Voronoi
    from shapely.geometry import Polygon, box as shapely_box
    
    # Extract centroids in cluster order
    cluster_ids = sorted(clusters_raw.keys())
    centroids = np.array([clusters_raw[cid]['centroid'] for cid in cluster_ids])
    
    # Use data bounds if no bounding box specified
    if bounding_box is None:
        # Get bounds from actual data
        min_coords = coordinates_2d.min(axis=0)
        max_coords = coordinates_2d.max(axis=0)
        # Add small margin
        margin = 0.05
        width = max_coords[0] - min_coords[0]
        height = max_coords[1] - min_coords[1]
        bounding_box = {
            'min_x': min_coords[0] - margin * width,
            'max_x': max_coords[0] + margin * width,
            'min_y': min_coords[1] - margin * height,
            'max_y': max_coords[1] + margin * height
        }
    
    # Add boundary points
    boundary_margin = 0.5
    width = bounding_box['max_x'] - bounding_box['min_x']
    height = bounding_box['max_y'] - bounding_box['min_y']
    
    boundary_points = np.array([
        [bounding_box['min_x'] - boundary_margin * width, bounding_box['min_y'] - boundary_margin * height],
        [bounding_box['max_x'] + boundary_margin * width, bounding_box['min_y'] - boundary_margin * height],
        [bounding_box['max_x'] + boundary_margin * width, bounding_box['max_y'] + boundary_margin * height],
        [bounding_box['min_x'] - boundary_margin * width, bounding_box['max_y'] + boundary_margin * height]
    ])
    
    # Combine centroids with boundary points
    voronoi_points = np.vstack([centroids, boundary_points])
    
    # Generate Voronoi diagram
    vor = Voronoi(voronoi_points)
    
    # Create crop box (use actual data bounds)
    crop_box = shapely_box(
        bounding_box['min_x'], 
        bounding_box['min_y'], 
        bounding_box['max_x'], 
        bounding_box['max_y']
    )
    
    # Process each cluster's Voronoi cell
    voronoi_data = {}
    
    for i, cluster_id in enumerate(cluster_ids):
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        
        if region and -1 not in region:  # Finite region
            vertices = vor.vertices[region]
        else:
            # Handle infinite regions
            vertices = clip_infinite_voronoi_region(vor, i, bounding_box)
        
        # Sort vertices clockwise
        vertices = sort_vertices_clockwise(vertices)
        
        # Crop to bounds
        poly = Polygon(vertices)
        cropped_poly = poly.intersection(crop_box)
        
        if not cropped_poly.is_empty and hasattr(cropped_poly, "exterior"):
            # Extract vertices, removing the duplicate last point
            cropped_vertices = np.array(cropped_poly.exterior.coords)[:-1]
        else:
            cropped_vertices = np.array(vertices)
        
        voronoi_data[cluster_id] = {
            'vertices': cropped_vertices  # Keep as numpy array
        }
    
    return voronoi_data


def fit_coords_to_voronoi_cell(coords, voronoi_vertices, padding_factor=0.1):
    """
    Use elastic transformation to naturally stretch coordinates to fill polygon shape
    GUARANTEED to keep all points inside the polygon with padding
    """
    from shapely.geometry import Polygon, Point, LineString
    import numpy as np
    
    poly = Polygon(voronoi_vertices)
    bounds = poly.bounds  # (minx, miny, maxx, maxy)
    
    # Ensure coords are normalized to [0,1]
    min_vals = coords.min(axis=0)
    max_vals = coords.max(axis=0)
    range_vals = np.where(max_vals - min_vals == 0, 1, max_vals - min_vals)
    norm_coords = (coords - min_vals) / range_vals
    
    # Calculate absolute padding in coordinate units
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    padding_x = width * padding_factor
    padding_y = height * padding_factor
    
    fitted_coords = []
    for coord in norm_coords:
        # Map to polygon bounds first
        x_raw = bounds[0] + coord[0] * width
        y_raw = bounds[1] + coord[1] * height
        
        # Find polygon width at this y-level using the ORIGINAL polygon
        y_line = LineString([(bounds[0] - 1, y_raw), (bounds[2] + 1, y_raw)])
        intersections = poly.boundary.intersection(y_line)
        
        # Extract x-coordinates from intersections
        x_coords = []
        if hasattr(intersections, 'geoms'):
            for geom in intersections.geoms:
                if hasattr(geom, 'x'):
                    x_coords.append(geom.x)
        elif hasattr(intersections, 'x'):
            x_coords.append(intersections.x)
        
        if len(x_coords) >= 2:
            # Use actual polygon width at this y-level
            min_x, max_x = min(x_coords), max(x_coords)
            
            # Apply padding by shrinking the available width
            local_width = max_x - min_x
            padded_min_x = min_x + padding_x
            padded_max_x = max_x - padding_x
            
            # Make sure we still have width left after padding
            if padded_max_x <= padded_min_x:
                # Fallback to center if padding is too large
                x_target = (min_x + max_x) / 2
            else:
                # Map x-coordinate within the padded width
                x_target = padded_min_x + coord[0] * (padded_max_x - padded_min_x)
        else:
            # Fallback to padded bounding box
            x_target = (bounds[0] + padding_x) + coord[0] * (width - 2 * padding_x)
        
        # Apply y-padding
        padded_y_min = bounds[1] + padding_y
        padded_y_max = bounds[3] - padding_y
        
        if padded_y_max <= padded_y_min:
            y_target = (bounds[1] + bounds[3]) / 2
        else:
            y_target = padded_y_min + coord[1] * (padded_y_max - padded_y_min)
        
        # Final safety check - ensure point is inside original polygon
        point = Point(x_target, y_target)
        if not poly.contains(point):
            # Move toward centroid
            centroid = poly.centroid
            # Calculate how far to move (aggressive correction)
            move_factor = 0.3
            x_target += move_factor * (centroid.x - x_target)
            y_target += move_factor * (centroid.y - y_target)
            
            # Check again
            point = Point(x_target, y_target)
            if not poly.contains(point):
                # Ultimate fallback - place near centroid
                centroid = poly.centroid
                x_target = centroid.x + np.random.normal(0, 0.01)
                y_target = centroid.y + np.random.normal(0, 0.01)
        
        fitted_coords.append([x_target, y_target])
    
    return np.array(fitted_coords)

def apply_global_alignment(umap_coords, global_coords, alignment_strength=0.3):
    """
    Bias UMAP coordinates toward their global relative positions
    """
    # Normalize both coordinate sets to [0,1]
    def normalize_coords(coords):
        min_vals = coords.min(axis=0)
        max_vals = coords.max(axis=0)
        range_vals = max_vals - min_vals
        # Handle edge case where all points are the same
        range_vals = np.where(range_vals == 0, 1, range_vals)
        return (coords - min_vals) / range_vals
    
    norm_umap = normalize_coords(umap_coords)
    norm_global = normalize_coords(global_coords)
    
    # Blend: mostly UMAP structure, some global bias
    aligned = (1 - alignment_strength) * norm_umap + alignment_strength * norm_global
    
    return aligned


def generate_per_cluster_umap(clusters_raw, base_umap_params, voronoi_data, padding_factor=0.05):
    """
    Generate per-cluster UMAP coordinates with polygon-aware initialization.
    
    Args:
        clusters_raw: dict of cluster_id -> cluster data
        base_umap_params: dict of UMAP parameters (excluding n_neighbors)
        voronoi_data: dict of cluster_id -> {'vertices': numpy array}
    
    Returns:
        dict of cluster_id -> numpy array of coordinates
    """
    import umap
    from shapely.geometry import Polygon

    per_cluster_coords = {}
    
    for cluster_id, cluster in clusters_raw.items():
        n_artworks = cluster['size']
        # 0-point cluster: skip
        if n_artworks == 0:
            continue
        # Single artwork case - place at Voronoi cell centroid
        if n_artworks == 1:
            vertices = voronoi_data[cluster_id]['vertices']
            poly = Polygon(vertices)
            centroid = poly.centroid
            per_cluster_coords[cluster_id] = np.array([[centroid.x, centroid.y]])
            continue
        # For 2 or 3 artworks, distribute randomly around the centroid
        if n_artworks <= 3:
            vertices = voronoi_data[cluster_id]['vertices']
            poly = Polygon(vertices)
            centroid = poly.centroid
            # Small random offsets within the polygon bounds
            coords = []
            for _ in range(n_artworks):
                for __ in range(20):
                    angle = np.random.uniform(0, 2 * np.pi)
                    radius = np.random.uniform(0.01, 0.08) * poly.length  # scale by perimeter
                    x = centroid.x + radius * np.cos(angle)
                    y = centroid.y + radius * np.sin(angle)
                    point = Point(x, y)
                    if poly.contains(point):
                        coords.append([x, y])
                        break
            else:
                # fallback: use centroid
                coords.append([centroid.x, centroid.y])
            per_cluster_coords[cluster_id] = np.array(coords)
            continue
        # Get polygon for this cluster
        vertices = voronoi_data[cluster_id]['vertices']
        poly = Polygon(vertices)
        # Calculate dynamic n_neighbors for this cluster
        cluster_n_neighbors = hf.calculate_n_neighbors(
            n_artworks, 
            min_neighbors=3, 
            max_neighbors=15
        )
        # Sample initial positions within the polygon
        init_positions = sample_points_in_polygon(poly, n_artworks)
        # Apply global alignment to initialization
        global_coords_normalized = normalize_coords_to_polygon(
            cluster['coordinates'], vertices
        )
        # Blend polygon sampling with global awareness
        alignment_strength = 0.4
        blended_init = (1 - alignment_strength) * init_positions + alignment_strength * global_coords_normalized
        # Only call fit_transform if n_artworks >= 2
        if n_artworks > 3:
            cluster_umap = umap.UMAP(
                n_neighbors=cluster_n_neighbors,
                init=blended_init,
                n_jobs=-1,
                **{k: v for k, v in base_umap_params.items() if k != 'parallel'}
            )
            try:
                raw_umap_coords = cluster_umap.fit_transform(cluster['embeddings'])
            except Exception as e:
                import traceback
                print(f"[generate_per_cluster_umap] UMAP fit_transform failed for cluster {cluster_id} (size={n_artworks}): {e}")
                traceback.print_exc()
                # Fallback: use blended_init if possible, else centroid
                if 'blended_init' in locals() and blended_init.shape[0] == n_artworks:
                    raw_umap_coords = blended_init
                else:
                    centroid = poly.centroid
                    raw_umap_coords = np.array([[centroid.x, centroid.y] for _ in range(n_artworks)])
            # Apply global alignment to the UMAP output
            aligned_coords = apply_global_alignment(
                raw_umap_coords, 
                cluster['coordinates'],  # global coords for this cluster
                alignment_strength=0.3
            )
            # Fit the aligned coordinates to the Voronoi polygon
            fitted_coords = fit_coords_to_voronoi_cell(
                aligned_coords, 
                vertices, 
                padding_factor=padding_factor
            )
            per_cluster_coords[cluster_id] = fitted_coords
    return per_cluster_coords


def sample_points_in_polygon(polygon, n_points):
    """Sample n points uniformly within a polygon"""
    bounds = polygon.bounds
    points = []
    max_attempts = n_points * 20  # Prevent infinite loops
    attempts = 0
    
    while len(points) < n_points and attempts < max_attempts:
        x = np.random.uniform(bounds[0], bounds[2])
        y = np.random.uniform(bounds[1], bounds[3])
        if polygon.contains(Point(x, y)):
            points.append([x, y])
        attempts += 1
    
    # Fill any remaining points near centroid
    centroid = polygon.centroid
    while len(points) < n_points:
        # Add small random offset from centroid
        offset_x = np.random.normal(0, 0.02)
        offset_y = np.random.normal(0, 0.02)
        points.append([centroid.x + offset_x, centroid.y + offset_y])
    
    return np.array(points)


def normalize_coords_to_polygon(global_coords, voronoi_vertices):
    """Normalize global coordinates to fit within polygon bounds"""
    from shapely.geometry import Polygon
    
    poly = Polygon(voronoi_vertices)
    bounds = poly.bounds
    
    # Normalize global coords to [0,1]
    min_vals = global_coords.min(axis=0)
    max_vals = global_coords.max(axis=0)
    range_vals = np.where(max_vals - min_vals == 0, 1, max_vals - min_vals)
    norm_coords = (global_coords - min_vals) / range_vals
    
    # Map to polygon bounds
    mapped_coords = []
    for coord in norm_coords:
        x = bounds[0] + coord[0] * (bounds[2] - bounds[0])
        y = bounds[1] + coord[1] * (bounds[3] - bounds[1])
        mapped_coords.append([x, y])
    
    return np.array(mapped_coords)


def fetch_artwork_metadata_batch(artwork_ids, db):
    """
    Efficiently fetch metadata for all artwork IDs in one query.

    Args:
        artwork_ids: list of artwork IDs to fetch
        db: database connection

    Returns:
        dict of artwork_id -> metadata dict (with full nested descriptions)
    """
    import json

    if not artwork_ids:
        return {}

    placeholders = ','.join(['?' for _ in artwork_ids])

    cursor = db.execute(f"""
        SELECT 
            image_id,
            value as title,
            artist_names,
            image_urls,
            filename,
            rights,
            descriptions,
            relatedKeywordIds,
            relatedKeywordStrings
        FROM image_entries
        WHERE image_id IN ({placeholders})
    """, artwork_ids)

    metadata = {}
    for row in cursor:
        try:
            artist_names = json.loads(row['artist_names']) if row['artist_names'] else []
            image_urls = json.loads(row['image_urls']) if row['image_urls'] else {}
            descriptions = json.loads(row['descriptions']) if row['descriptions'] else {}
            related_keywords = json.loads(row['relatedKeywordStrings']) if row['relatedKeywordStrings'] else []

            thumbnail_url = (
                image_urls.get('small') or 
                image_urls.get('medium_rectangle') or 
                image_urls.get('normalized') or 
                ''
            )

            full_url = (
                image_urls.get('large') or 
                image_urls.get('normalized') or 
                thumbnail_url
            )

            metadata[row['image_id']] = {
                'title': row['title'] or 'Untitled',
                'artist': artist_names[0] if artist_names else 'Unknown Artist',
                'artist_names': artist_names,
                'image_urls': image_urls,
                'thumbnail_url': thumbnail_url,
                'url': full_url,
                'descriptions': descriptions,  # full raw descriptions by source
                'rights': row['rights'] or '',
                'keywords': related_keywords[:10]
            }

        except Exception as e:
            print(f"Error parsing metadata for {row['image_id']}: {e}")
            metadata[row['image_id']] = {
                'title': row.get('value', 'Untitled'),
                'artist': 'Unknown Artist',
                'artist_names': [],
                'image_urls': {},
                'thumbnail_url': '',
                'url': '',
                'descriptions': {},
                'rights': row.get('rights', ''),
                'keywords': []
            }

    return metadata
