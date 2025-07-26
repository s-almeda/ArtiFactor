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
from shapely.geometry import Polygon

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
    import numpy as np
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
