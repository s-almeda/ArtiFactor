import numpy as np
from shapely.geometry import Point, Polygon
from helper_functions.helperfunctions import base64_to_image, url_to_image, find_most_similar_images, extract_img_features
from helper_functions.helperfunctions import find_most_similar_texts, find_most_similar_clip
from helper_functions.helperfunctions import extract_text_features, extract_clip_multimodal_features

# Added debug statements to log function inputs and outputs for better traceability.

def find_containing_region(position, regions, priority_region_ids=None):
    print(f"DEBUG: find_containing_region called with position={position}, priority_region_ids={priority_region_ids}")
    """
    Find which region contains the given position.
    Check priority regions first for efficiency.
    
    Args:
        position: [x, y] coordinates
        regions: List of region dictionaries with 'id' and 'vertices'
        priority_region_ids: List of region IDs to check first (from anchor artworks)
    
    Returns:
        region_id (string) or None if not found
    """
    query_point = Point(position)
    
    # First check priority regions (where anchor artworks are located)
    if priority_region_ids:
        priority_regions = [r for r in regions if str(r['id']) in priority_region_ids]
        for region in priority_regions:
            try:
                region_polygon = Polygon(region['vertices'])
                if region_polygon.contains(query_point):
                    print(f"Found containing region (priority): {region['id']}")
                    return str(region['id'])
            except Exception as e:
                print(f"Error checking priority region {region['id']}: {e}")
                continue
    
    # Then check all other regions
    for region in regions:
        region_id = str(region['id'])
        if priority_region_ids and region_id in priority_region_ids:
            continue  # Already checked above
        
        try:
            region_polygon = Polygon(region['vertices'])
            if region_polygon.contains(query_point):
                print(f"Found containing region: {region_id}")
                return region_id
        except Exception as e:
            print(f"Error checking region {region_id}: {e}")
            continue
    
    print(f"No containing region found for position {position}")
    print(f"DEBUG: find_containing_region returning None")
    return None

def apply_distance_constraints(position, anchor_position, min_distance, max_distance):
    print(f"DEBUG: apply_distance_constraints called with position={position}, anchor_position={anchor_position}, min_distance={min_distance}, max_distance={max_distance}")
    """
    Ensure position is within min/max distance from anchor.
    
    Args:
        position: Target [x, y] position
        anchor_position: Anchor [x, y] position  
        min_distance: Minimum allowed distance
        max_distance: Maximum allowed distance
    
    Returns:
        Constrained [x, y] position
    """
    anchor_pos = np.array(anchor_position)
    target_pos = np.array(position)
    
    # Calculate current distance
    distance = np.linalg.norm(target_pos - anchor_pos)
    
    if distance < min_distance:
        # Too close - move away from anchor
        if distance == 0:
            # If exactly at anchor, move in random direction
            direction = np.random.random(2) - 0.5
            direction = direction / np.linalg.norm(direction)
        else:
            direction = (target_pos - anchor_pos) / distance
        
        constrained_pos = anchor_pos + direction * min_distance
        print(f"Applied min distance constraint: {distance:.4f} -> {min_distance}")
        
    elif distance > max_distance:
        # Too far - move closer to anchor
        direction = (target_pos - anchor_pos) / distance
        constrained_pos = anchor_pos + direction * max_distance
        print(f"Applied max distance constraint: {distance:.4f} -> {max_distance}")
        
    else:
        # Within bounds
        constrained_pos = target_pos
    
    print(f"DEBUG: apply_distance_constraints returning constrained_pos={constrained_pos}")
    return constrained_pos

def place_query_image_triangulated(query_image, artwork_positions, artwork_to_region_map, region_vertices, db, 
                                 min_distance=0.1, max_distance=0.5, similarity_weight=0.7):
    """
    Place query image using simple weighted triangulation.
    
    Args:
        query_image: image as base64 or url
        artwork_positions: Dict of artwork_id -> [x, y] coordinates  
        artwork_to_region_map: Dict of artwork_id -> region_id
        region_vertices: Dict of region_id -> list of [x, y] polygon vertices
        db: Database connection
        min_distance: Minimum distance from most similar artwork
        max_distance: Maximum distance from most similar artwork  
        similarity_weight: 0.0 = centroid, 1.0 = closest to anchor1
    
    Returns:
        Dict with position, regionId, and anchor information
    """
    try:
        print("Processing query image input...")
        
        # 1. Handle input type: PIL image, URL, or base64
        if hasattr(query_image, 'size'):  # PIL Image
            query_img = query_image
        elif isinstance(query_image, str) and query_image.startswith('http'):
            print("Loading image from URL...")
            query_img = url_to_image(query_image)
        elif isinstance(query_image, str):
            # Handle data URL format if present
            if query_image.startswith('data:image'):
                query_image = query_image.split(',')[1]
            print("Decoding base64 image...")
            query_img = base64_to_image(query_image)
        else:
            return {"error": "Unsupported query image format"}

        if query_img is None:
            return {"error": "Failed to decode query image"}
        
        print("Extracting features from query image...")
        query_features = extract_img_features(query_img)
        
        # 2. Find most similar images (with artwork ID filtering for current map level)
        artwork_ids_list = list(artwork_positions.keys())
        print(f"Finding most similar images from {len(artwork_ids_list)} artworks...")
        similar_images = find_most_similar_images(
            query_features, 
            db, 
            top_k=10,  # Get more candidates in case some don't have positions
            artwork_ids=artwork_ids_list
        )
        
        if len(similar_images) < 3:
            return {
                "error": f"Need at least 3 similar artworks in database, found {len(similar_images)}",
                "found_artworks": len(similar_images)
            }

        print(f"Found {len(similar_images)} similar artworks")
        
        # 3. Get anchor positions - need to find them in the artwork_positions
        anchors = []
        for img in similar_images:
            if img['image_id'] in artwork_positions:
                anchors.append({
                    'id': img['image_id'],
                    'position': np.array(artwork_positions[img['image_id']]),
                    'distance': img['distance'],
                    'similarity': 1.0 / (1.0 + img['distance'])
                })
        
        if len(anchors) < 3:
            # Fallback: try to find positions for similar artworks by searching through regions
            print("Some anchors not in artwork_positions, searching regions...")
            for img in similar_images:
                if img['image_id'] not in artwork_positions:
                    # Search through regions to find this artwork
                    for region_id, vertices in region_vertices.items():
                        # This is a fallback - we don't have artworksMap here
                        # so we'll skip artworks not in artwork_positions
                        continue
            
            if len(anchors) < 3:
                return {
                    "error": f"Only found {len(anchors)} anchors with known positions",
                    "found_anchors": len(anchors)
                }

        # Use top 3 anchors
        anchor1, anchor2, anchor3 = anchors[:3]
        
        print(f"Using anchors: {anchor1['id']}, {anchor2['id']}, {anchor3['id']}")
        
        # 4. Simple weighted triangulation
        pos1, pos2, pos3 = anchor1['position'], anchor2['position'], anchor3['position']
        
        # Calculate centroid of the 3 anchors
        centroid = (pos1 + pos2 + pos3) / 3
        
        # Weighted position: similarity_weight controls how close to anchor1 vs centroid
        target_position = similarity_weight * pos1 + (1 - similarity_weight) * centroid
        
        print(f"Target position before constraints: {target_position}")
        print(f"Anchor1: {pos1}, Centroid: {centroid}, Weight: {similarity_weight}")
        
        # 5. Apply distance constraints
        constrained_position = apply_distance_constraints(
            target_position, pos1, min_distance, max_distance
        )
        
        # 6. Find containing region using geometric search
        # Build regions list from region_vertices for the search function
        regions_list = [
            {'id': region_id, 'vertices': vertices} 
            for region_id, vertices in region_vertices.items()
        ]
        
        # Get priority region IDs from anchors
        priority_region_ids = []
        for anchor in anchors[:3]:
            region_id = artwork_to_region_map.get(anchor['id'])
            if region_id:
                priority_region_ids.append(str(region_id))
        
        assigned_region = find_containing_region(
            constrained_position, regions_list, priority_region_ids
        )
        
        if not assigned_region:
            # Fallback: assign to region of most similar artwork
            assigned_region = artwork_to_region_map.get(anchor1['id'])
            print(f"Using fallback region assignment: {assigned_region}")
        
        print(f"Final position: {constrained_position}")
        print(f"Assigned region: {assigned_region}")

        return {
            "success": True,
            "position": constrained_position.tolist(),
            "regionId": assigned_region,
            "wasConstrained": not np.allclose(target_position, constrained_position, atol=1e-6),
            "confidence": anchor1['similarity'],
            "anchors": [
                {
                    "id": anchor['id'], 
                    "similarity": anchor['similarity'], 
                    "position": anchor['position'].tolist(), 
                    "distance": anchor['distance']
                }
                for anchor in anchors[:3]
            ],
            "parameters": {
                "min_distance": min_distance,
                "max_distance": max_distance, 
                "similarity_weight": similarity_weight,
                "centroid": centroid.tolist(),
                "target_before_constraints": target_position.tolist()
            }
        }
        
    except Exception as e:
        import traceback
        print(f"Exception occurred: {e}")
        traceback.print_exc()
        return {
            "error": f"Error in triangulation: {str(e)}",
            "traceback": traceback.format_exc()
        }

def calculate_triangulated_position(anchors, artwork_to_region_map, region_vertices, min_distance, max_distance, similarity_weight):
    print(f"DEBUG: calculate_triangulated_position called with anchors={anchors}, min_distance={min_distance}, max_distance={max_distance}, similarity_weight={similarity_weight}")
    """
    Helper function to calculate triangulated position from anchors.
    Reuses existing logic from place_query_image_triangulated.
    """
    if len(anchors) < 3:
        raise ValueError("Need at least 3 anchors for triangulation")
    
    # Use top 3 anchors
    anchor1, anchor2, anchor3 = anchors[:3]
    pos1, pos2, pos3 = anchor1['position'], anchor2['position'], anchor3['position']
    
    # Calculate centroid
    centroid = (pos1 + pos2 + pos3) / 3
    
    # Weighted position
    target_position = similarity_weight * pos1 + (1 - similarity_weight) * centroid
    
    # Apply distance constraints
    constrained_position = apply_distance_constraints(
        target_position, pos1, min_distance, max_distance
    )
    
    # Find containing region
    regions_list = [
        {'id': region_id, 'vertices': vertices} 
        for region_id, vertices in region_vertices.items()
    ]
    
    priority_region_ids = []
    for anchor in anchors[:3]:
        region_id = artwork_to_region_map.get(anchor['id'])
        if region_id:
            priority_region_ids.append(str(region_id))
    
    assigned_region = find_containing_region(
        constrained_position, regions_list, priority_region_ids
    )
    
    if not assigned_region:
        # Fallback: assign to region of most similar artwork
        assigned_region = artwork_to_region_map.get(anchor1['id'])
    
    print(f"DEBUG: calculate_triangulated_position returning position={constrained_position.tolist()}, regionId={assigned_region}")
    return {
        'position': constrained_position.tolist(),
        'regionId': assigned_region,
        'wasConstrained': not np.allclose(target_position, constrained_position, atol=1e-6)
    }

def place_query_image_multimodal(query_image, prompt_text, artwork_positions, artwork_to_region_map, region_vertices, db, 
                                 min_distance=0.1, max_distance=0.5, similarity_weight=0.7):
    print(f"DEBUG: place_query_image_multimodal called with query_image={query_image}, prompt_text={prompt_text}")
    """
    Place query image using multimodal triangulation (CLIP and visual-only).

    Args:
        query_image: image as base64 or URL
        prompt_text: text description for multimodal placement
        artwork_positions: Dict of artwork_id -> [x, y] coordinates
        artwork_to_region_map: Dict of artwork_id -> region_id
        region_vertices: Dict of region_id -> list of [x, y] polygon vertices
        db: Database connection
        min_distance: Minimum distance from most similar artwork
        max_distance: Maximum distance from most similar artwork
        similarity_weight: 0.0 = centroid, 1.0 = closest to anchor1

    Returns:
        Dict with primary placement, alternative placements, and anchor information
    """
    try:
        print("Processing multimodal query...")

        # Step 1: Process inputs
        if hasattr(query_image, 'size'):  # PIL Image
            query_img = query_image
        elif isinstance(query_image, str) and query_image.startswith('http'):
            print("Loading image from URL...")
            query_img = url_to_image(query_image)
        elif isinstance(query_image, str):
            if query_image.startswith('data:image'):
                query_image = query_image.split(',')[1]
            print("Decoding base64 image...")
            query_img = base64_to_image(query_image)
        else:
            return {"error": "Unsupported query image format"}

        if query_img is None:
            return {"error": "Failed to decode query image"}

        print("Extracting features from query image and text...")

        # Step 2: Generate embeddings
        image_features = extract_img_features(query_img)        # ResNet visual features
        clip_features = extract_clip_multimodal_features(query_img, prompt_text)  # CLIP multimodal features

        # Step 3: Run similarity searches
        artwork_ids_list = list(artwork_positions.keys())
        print(f"Running similarity searches among {len(artwork_ids_list)} artworks...")

        clip_results = find_most_similar_clip(clip_features, db, top_k=3, artwork_ids=artwork_ids_list)
        image_results = find_most_similar_images(image_features, db, top_k=3, artwork_ids=artwork_ids_list)

        # Debugging unfiltered searches
        print("=== TESTING WITHOUT ARTWORK ID FILTERING ===")

        print("Running CLIP search without artwork_ids filter...")
        clip_results_unfiltered = find_most_similar_clip(clip_features, db, top_k=10, artwork_ids=None)
        print(f"CLIP results (unfiltered): {len(clip_results_unfiltered)}")
        for result in clip_results_unfiltered[:5]:
            print(f"  CLIP: {result}")

        print("Running image search without artwork_ids filter...")
        image_results_unfiltered = find_most_similar_images(image_features, db, top_k=10, artwork_ids=None)
        print(f"Image results (unfiltered): {len(image_results_unfiltered)}")
        for result in image_results_unfiltered[:5]:
            print(f"  Image: {result}")

        print("=== CHECKING ID OVERLAPS ===")
        all_vector_ids = set()
        for result in clip_results_unfiltered + image_results_unfiltered:
            vector_id = result.get('image_id') or result.get('entry_id')
            if vector_id:
                all_vector_ids.add(vector_id)

        artwork_position_ids = set(artwork_positions.keys())
        overlap = all_vector_ids.intersection(artwork_position_ids)

        print(f"Vector table IDs (sample): {list(all_vector_ids)[:10]}")
        print(f"Artwork position IDs (sample): {list(artwork_position_ids)[:10]}")
        print(f"ID overlap found: {len(overlap)} matches")

        if overlap:
            print(f"✅ Overlapping IDs: {list(overlap)[:5]}")
            valid_clip_results = [r for r in clip_results_unfiltered 
                                if (r.get('image_id') or r.get('entry_id')) in overlap]
            valid_image_results = [r for r in image_results_unfiltered 
                                 if (r.get('image_id') or r.get('entry_id')) in overlap]
            clip_results = valid_clip_results
            image_results = valid_image_results
        else:
            print("❌ NO ID OVERLAP - Vector tables and artwork positions use different ID formats")

        # Step 4: Build anchor collection with proper key handling
        all_anchors = []

        def add_anchors(results, anchor_type):
            for result in results:
                artwork_id = result.get('entry_id') or result.get('image_id')
                if artwork_id and artwork_id in artwork_positions:
                    all_anchors.append({
                        'id': artwork_id,
                        'position': np.array(artwork_positions[artwork_id]),
                        'type': anchor_type,
                        'distance': result['distance'],
                        'similarity': 1.0 / (1.0 + result['distance'])
                    })

        add_anchors(clip_results, 'clip')
        add_anchors(image_results, 'image')

        if len(all_anchors) < 3:
            return {
                "error": f"Need at least 3 anchors total, found {len(all_anchors)}",
                "found_anchors": len(all_anchors)
            }

        print(f"Found {len(all_anchors)} total anchors")

        # Step 5: Calculate placements
        def get_anchors_by_type(anchor_type):
            return [a for a in all_anchors if a['type'] == anchor_type]

        def calculate_placement(anchors):
            if len(anchors) >= 3:
                return calculate_triangulated_position(anchors[:3], artwork_to_region_map, region_vertices, min_distance, max_distance, similarity_weight)
            return None

        # Primary placement (CLIP-based with fallback)
        clip_anchors = get_anchors_by_type('clip')
        if len(clip_anchors) >= 3:
            primary_anchors = clip_anchors[:3]
        else:
            primary_anchors = sorted(all_anchors, key=lambda x: (x['type'] != 'clip', x['distance']))[:3]
            print(f"Using fallback primary anchors: CLIP={len(clip_anchors)}, total={len(primary_anchors)}")

        primary_placement = calculate_placement(primary_anchors)

        if not primary_placement:
            return {
                "error": "Failed to calculate primary placement. Ensure sufficient anchors are available.",
                "anchorCounts": {
                    "clip": len(clip_anchors),
                    "image": len(get_anchors_by_type('image'))
                }
            }

        # Alternative placements
        alternative_placements = {}

        # Visual-only placement
        image_anchors = get_anchors_by_type('image')
        visual_only_placement = calculate_placement(image_anchors)
        if visual_only_placement:
            alternative_placements['visualOnly'] = visual_only_placement

        # Step 6: Return result in expected API format
        print(f"DEBUG: place_query_image_multimodal returning success={True}, position={primary_placement['position']}, regionId={primary_placement['regionId']}")
        return {
            "success": True,
            "position": primary_placement['position'],
            "regionId": primary_placement['regionId'],
            "wasConstrained": primary_placement.get('wasConstrained', False),
            "confidence": primary_anchors[0]['similarity'],
            "anchors": [
                {
                    "artworkId": anchor['id'],
                    "similarity": anchor['similarity'],
                    "position": anchor['position'].tolist(),
                    "distance": anchor['distance'],
                    "type": anchor['type']
                }
                for anchor in all_anchors[:9]  # Return up to 9 anchors
            ],
            "alternativePlacements": alternative_placements,
            "parameters": {
                "min_distance": min_distance,
                "max_distance": max_distance,
                "similarity_weight": similarity_weight
            },
            "anchorCounts": {
                "clip": len(clip_anchors),
                "image": len(image_anchors)
            }
        }

    except Exception as e:
        import traceback
        print(f"Exception in multimodal placement: {e}")
        traceback.print_exc()
        return {
            "error": f"Error in multimodal placement: {str(e)}",
            "traceback": traceback.format_exc()
        }