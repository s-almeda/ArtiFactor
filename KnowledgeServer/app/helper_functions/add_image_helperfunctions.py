import numpy as np
from shapely.geometry import Point, Polygon
from helper_functions.helperfunctions import base64_to_image, url_to_image, find_most_similar_images, extract_img_features
from helper_functions.helperfunctions import find_similar_artworks_by_text, find_most_similar_clip
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

# def place_query_image_triangulated(query_image, artwork_positions, artwork_to_region_map, region_vertices, db, 
#                                  min_distance=0.1, max_distance=0.5, similarity_weight=0.7):
#     """
#     Place query image using simple weighted triangulation.
    
#     Args:
#         query_image: image as base64 or url
#         artwork_positions: Dict of artwork_id -> [x, y] coordinates  
#         artwork_to_region_map: Dict of artwork_id -> region_id
#         region_vertices: Dict of region_id -> list of [x, y] polygon vertices
#         db: Database connection
#         min_distance: Minimum distance from most similar artwork
#         max_distance: Maximum distance from most similar artwork  
#         similarity_weight: 0.0 = centroid, 1.0 = closest to anchor1
    
#     Returns:
#         Dict with position, regionId, and anchor information
#     """
#     try:
#         print("Processing query image input...")
        
#         # 1. Handle input type: PIL image, URL, or base64
#         if hasattr(query_image, 'size'):  # PIL Image
#             query_img = query_image
#         elif isinstance(query_image, str) and query_image.startswith('http'):
#             print("Loading image from URL...")
#             query_img = url_to_image(query_image)
#         elif isinstance(query_image, str):
#             # Handle data URL format if present
#             if query_image.startswith('data:image'):
#                 query_image = query_image.split(',')[1]
#             print("Decoding base64 image...")
#             query_img = base64_to_image(query_image)
#         else:
#             return {"error": "Unsupported query image format"}

#         if query_img is None:
#             return {"error": "Failed to decode query image"}
        
#         print("Extracting features from query image...")
#         query_features = extract_img_features(query_img)
        
#         # 2. Find most similar images (with artwork ID filtering for current map level)
#         artwork_ids_list = list(artwork_positions.keys())
#         print(f"Finding most similar images from {len(artwork_ids_list)} artworks...")
#         similar_images = find_most_similar_images(
#             query_features, 
#             db, 
#             top_k=10,  # Get more candidates in case some don't have positions
#             artwork_ids=artwork_ids_list
#         )
        
#         if len(similar_images) < 3:
#             return {
#                 "error": f"Need at least 3 similar artworks in database, found {len(similar_images)}",
#                 "found_artworks": len(similar_images)
#             }

#         print(f"Found {len(similar_images)} similar artworks")
        
#         # 3. Get anchor positions - need to find them in the artwork_positions
#         anchors = []
#         for img in similar_images:
#             if img['image_id'] in artwork_positions:
#                 anchors.append({
#                     'id': img['image_id'],
#                     'position': np.array(artwork_positions[img['image_id']]),
#                     'distance': img['distance'],
#                     'similarity': 1.0 / (1.0 + img['distance'])
#                 })
        
#         if len(anchors) < 3:
#             # Fallback: try to find positions for similar artworks by searching through regions
#             print("Some anchors not in artwork_positions, searching regions...")
#             for img in similar_images:
#                 if img['image_id'] not in artwork_positions:
#                     # Search through regions to find this artwork
#                     for region_id, vertices in region_vertices.items():
#                         # This is a fallback - we don't have artworksMap here
#                         # so we'll skip artworks not in artwork_positions
#                         continue
            
#             if len(anchors) < 3:
#                 return {
#                     "error": f"Only found {len(anchors)} anchors with known positions",
#                     "found_anchors": len(anchors)
#                 }

#         # Use top 3 anchors
#         anchor1, anchor2, anchor3 = anchors[:3]
        
#         print(f"Using anchors: {anchor1['id']}, {anchor2['id']}, {anchor3['id']}")
        
#         # 4. Simple weighted triangulation
#         pos1, pos2, pos3 = anchor1['position'], anchor2['position'], anchor3['position']
        
#         # Calculate centroid of the 3 anchors
#         centroid = (pos1 + pos2 + pos3) / 3
        
#         # Weighted position: similarity_weight controls how close to anchor1 vs centroid
#         target_position = similarity_weight * pos1 + (1 - similarity_weight) * centroid
        
#         print(f"Target position before constraints: {target_position}")
#         print(f"Anchor1: {pos1}, Centroid: {centroid}, Weight: {similarity_weight}")
        
#         # 5. Apply distance constraints
#         constrained_position = apply_distance_constraints(
#             target_position, pos1, min_distance, max_distance
#         )
        
#         # 6. Find containing region using geometric search
#         # Build regions list from region_vertices for the search function
#         regions_list = [
#             {'id': region_id, 'vertices': vertices} 
#             for region_id, vertices in region_vertices.items()
#         ]
        
#         # Get priority region IDs from anchors
#         priority_region_ids = []
#         for anchor in anchors[:3]:
#             region_id = artwork_to_region_map.get(anchor['id'])
#             if region_id:
#                 priority_region_ids.append(str(region_id))
        
#         assigned_region = find_containing_region(
#             constrained_position, regions_list, priority_region_ids
#         )
        
#         if not assigned_region:
#             # Fallback: assign to region of most similar artwork
#             assigned_region = artwork_to_region_map.get(anchor1['id'])
#             print(f"Using fallback region assignment: {assigned_region}")
        
#         print(f"Final position: {constrained_position}")
#         print(f"Assigned region: {assigned_region}")

#         return {
#             "success": True,
#             "position": constrained_position.tolist(),
#             "regionId": assigned_region,
#             "wasConstrained": not np.allclose(target_position, constrained_position, atol=1e-6),
#             "confidence": anchor1['similarity'],
#             "anchors": [
#                 {
#                     "id": anchor['id'], 
#                     "similarity": anchor['similarity'], 
#                     "position": anchor['position'].tolist(), 
#                     "distance": anchor['distance']
#                 }
#                 for anchor in anchors[:3]
#             ],
#             "parameters": {
#                 "min_distance": min_distance,
#                 "max_distance": max_distance, 
#                 "similarity_weight": similarity_weight,
#                 "centroid": centroid.tolist(),
#                 "target_before_constraints": target_position.tolist()
#             }
#         }
        
#     except Exception as e:
#         import traceback
#         print(f"Exception occurred: {e}")
#         traceback.print_exc()
#         return {
#             "error": f"Error in triangulation: {str(e)}",
#             "traceback": traceback.format_exc()
#         }

# def calculate_triangulated_position(anchors, artwork_to_region_map, region_vertices, min_distance, max_distance, similarity_weight):
#     print(f"DEBUG: calculate_triangulated_position called with anchors={anchors}, min_distance={min_distance}, max_distance={max_distance}, similarity_weight={similarity_weight}")
#     """
#     Helper function to calculate triangulated position from anchors.
#     Reuses existing logic from place_query_image_triangulated.
#     """
#     if len(anchors) < 3:
#         raise ValueError("Need at least 3 anchors for triangulation")
    
#     # Use top 3 anchors
#     anchor1, anchor2, anchor3 = anchors[:3]
#     pos1, pos2, pos3 = anchor1['position'], anchor2['position'], anchor3['position']
    
#     # Calculate centroid
#     centroid = (pos1 + pos2 + pos3) / 3
    
#     # Weighted position
#     target_position = similarity_weight * pos1 + (1 - similarity_weight) * centroid
    
#     # Apply distance constraints
#     constrained_position = apply_distance_constraints(
#         target_position, pos1, min_distance, max_distance
#     )
    
#     # Find containing region
#     regions_list = [
#         {'id': region_id, 'vertices': vertices} 
#         for region_id, vertices in region_vertices.items()
#     ]
    
#     priority_region_ids = []
#     for anchor in anchors[:3]:
#         region_id = artwork_to_region_map.get(anchor['id'])
#         if region_id:
#             priority_region_ids.append(str(region_id))
    
#     assigned_region = find_containing_region(
#         constrained_position, regions_list, priority_region_ids
#     )
    
#     if not assigned_region:
#         # Fallback: assign to region of most similar artwork
#         assigned_region = artwork_to_region_map.get(anchor1['id'])
    
#     print(f"DEBUG: calculate_triangulated_position returning position={constrained_position.tolist()}, regionId={assigned_region}")
#     return {
#         'position': constrained_position.tolist(),
#         'regionId': assigned_region,
#         'wasConstrained': not np.allclose(target_position, constrained_position, atol=1e-6)
#     }

def get_image_nearest_neighbors_multimodal(query_image, query_text, db, top_k=3, artwork_ids=None):
    """
    Get baseline image search results for a query image and/or text.
    
    Args:
        query_image: Image as base64 or URL (optional if query_text provided)
        query_text: Text prompt for multimodal search (optional if query_image provided)
        db: Database connection
        top_k: Number of top results to return
        artwork_ids: Optional list of artwork IDs to filter results to only a specific subset of artworks

    Returns:
        Dict with clip_results, image_results, and text_results, each containing list of matches
        Or dict with error message if both inputs are missing or other error occurs
    """
    try:
        # Validate inputs
        if query_image is None and query_text is None:
            return {"error": "At least one of query_image or query_text must be provided"}

        # Initialize result containers
        image_results = None
        text_results = None
        clip_results = None
        query_img = None

        # Process image if provided
        if query_image is not None:
            print("Processing query image...")
            if hasattr(query_image, 'size'):
                query_img = query_image
            elif isinstance(query_image, str) and query_image.startswith('http'):
                query_img = url_to_image(query_image)
            elif isinstance(query_image, str):
                if query_image.startswith('data:image'):
                    query_image = query_image.split(',')[1]
                query_img = base64_to_image(query_image)
            else:
                return {"error": "Unsupported query image format"}

            if query_img is None:
                return {"error": "Failed to decode query image"}
            
            print("Extracting image features...")
            image_features = extract_img_features(query_img)
            image_results = find_most_similar_images(image_features, db, top_k=top_k, artwork_ids=artwork_ids)

        # Process text if provided
        if query_text is not None:
            print("Processing query text...")
            text_features = extract_text_features(query_text)
            text_results = find_similar_artworks_by_text(text_features, db, top_k=top_k, artwork_ids=artwork_ids)
            
            # Also find exact matches for the query text
            print("Finding exact matches for query text...")
            from helper_functions.helperfunctions import find_exact_matches
            exact_matches = find_exact_matches(query_text, db, artists_only=False, search_aliases=True)
            
            # Filter exact matches by artwork_ids if provided and convert to expected format
            if exact_matches:
                exact_text_results = []
                for match in exact_matches:
                    image_id = match.get('image_id')
                    if artwork_ids is None or image_id in artwork_ids:
                        exact_text_results.append({
                            'image_id': image_id,
                            'distance': 0.0,  # Exact match gets distance 0
                            'match_type': 'exact'
                        })
                
                # Prepend exact matches to text results (they should come first due to distance 0)
                if exact_text_results:
                    print(f"Found {len(exact_text_results)} exact matches")
                    if text_results:
                        text_results = exact_text_results + text_results
                    else:
                        text_results = exact_text_results

        # Run CLIP search only if both image and text are provided
        if query_img is not None and query_text is not None:
            print("Running multimodal CLIP search...")
            clip_features = extract_clip_multimodal_features(query_img, query_text)
            clip_results = find_most_similar_clip(clip_features, db, top_k=top_k, artwork_ids=artwork_ids)
            # Normalize CLIP results to use image_id instead of entry_id
            if clip_results:
                clip_results = [{**result, 'image_id': result.pop('entry_id') if 'entry_id' in result else result.get('image_id')} 
                              for result in clip_results]

        if artwork_ids:
            print(f"Results filtered to {len(artwork_ids)} artwork IDs")
        
        # Construct response with available results
        response = {}
        if clip_results is not None:
            response["clip_results"] = clip_results
        if image_results is not None:
            response["image_results"] = image_results
        if text_results is not None:
            response["text_results"] = text_results

        if not response:
            return {"error": "No search results found"}

        return response

    except Exception as e:
        import traceback
        print(f"Exception in get_image_nearest_neighbors_multimodal: {e}")
        traceback.print_exc()
        return {
            "error": f"Error in baseline image search: {str(e)}",
            "traceback": traceback.format_exc()
        }


def place_query_image_multimodal(query_image, prompt_text, artwork_positions, artwork_to_region_map, region_vertices, db, 
                                 min_distance=0.1, max_distance=0.5):
    """
    Places a query image using multimodal similarity (image, text, CLIP).
    Args:
        query_image: Image, URL, or base64 string
        prompt_text: Text prompt
        artwork_positions: Dict of artwork_id -> [x, y]
        artwork_to_region_map: Dict of artwork_id -> region_id
        region_vertices: Dict of region_id -> vertices
        db: Similarity search database
        min_distance, max_distance: Placement distance bounds
    Returns:
        dict with position, regionId, confidence, anchors, parameters, anchorCounts, error/traceback if any
        Each anchor provides:

        The artwork's ID (artworkId)
        Its position in the coordinate space (position)
        Similarity scores (similarity, adjustedSimilarity)
        The distance metric used for similarity (distance)
        The type of similarity used (type: "clip", "image", or "text")
        Anchors help determine where to place the query image by providing strong reference points in the artwork space. The function uses the top anchors to calculate the most appropriate placement for the query image.


        The returned result includes up to 9 anchors in the "anchors" list (all_candidates[:9]), 
        each with similarity, adjustedSimilarity, position, distance, and type.
    """
    try:
        # Step 1: Process inputs (same as before)
        if hasattr(query_image, 'size'):
            query_img = query_image
        elif isinstance(query_image, str) and query_image.startswith('http'):
            query_img = url_to_image(query_image)
        elif isinstance(query_image, str):
            if query_image.startswith('data:image'):
                query_image = query_image.split(',')[1]
            query_img = base64_to_image(query_image)
        else:
            return {"error": "Unsupported query image format"}

        if query_img is None:
            return {"error": "Failed to decode query image"}

        # Step 2: Generate embeddings
        image_features = extract_img_features(query_img)
        text_features = extract_text_features(prompt_text)
        clip_features = extract_clip_multimodal_features(query_img, prompt_text)

        # Step 3: Run similarity searches
        clip_results = find_most_similar_clip(clip_features, db, top_k=10, artwork_ids=artwork_positions.keys())
        image_results = find_most_similar_images(image_features, db, top_k=10, artwork_ids=artwork_positions.keys())
        text_results = find_similar_artworks_by_text(text_features, db, top_k=10, artwork_ids=artwork_positions.keys())

        # Step 4: Build and score all candidates
        all_candidates = []

        # Add CLIP results with adjustment
        for result in clip_results:
            artwork_id = result.get('entry_id') or result.get('image_id')
            if artwork_id and artwork_id in artwork_positions:
                base_similarity = 1.0 / (1.0 + result['distance'])
                adjusted_similarity = base_similarity * 0.7  # Reduce CLIP dominance
                all_candidates.append({
                    'id': artwork_id,
                    'position': np.array(artwork_positions[artwork_id]),
                    'type': 'clip',
                    'distance': result['distance'],
                    'base_similarity': base_similarity,
                    'adjusted_similarity': adjusted_similarity
                })

        # Add image results with boost
        for result in image_results:
            artwork_id = result.get('entry_id') or result.get('image_id')
            if artwork_id and artwork_id in artwork_positions:
                base_similarity = 1.0 / (1.0 + result['distance'])
                adjusted_similarity = base_similarity * 8.0  # Boost ResNet scores
                all_candidates.append({
                    'id': artwork_id,
                    'position': np.array(artwork_positions[artwork_id]),
                    'type': 'image',
                    'distance': result['distance'],
                    'base_similarity': base_similarity,
                    'adjusted_similarity': adjusted_similarity
                })

        # Add text results
        for result in text_results:
            artwork_id = result.get('image_id')
            if artwork_id and artwork_id in artwork_positions:
                base_similarity = 1.0 / (1.0 + result['distance'])
                adjusted_similarity = base_similarity  # Keep text scores as-is
                all_candidates.append({
                    'id': artwork_id,
                    'position': np.array(artwork_positions[artwork_id]),
                    'type': 'text',
                    'distance': result['distance'],
                    'base_similarity': base_similarity,
                    'adjusted_similarity': adjusted_similarity
                })

        if len(all_candidates) < 2:
            return {
                "error": f"Need at least 2 similar artworks, found {len(all_candidates)}",
                "found_candidates": len(all_candidates)
            }

        # Step 5: Sort by adjusted similarity and pick top 2
        all_candidates.sort(key=lambda x: x['adjusted_similarity'], reverse=True)
        
        anchor1 = all_candidates[0]  # Most similar
        anchor2 = all_candidates[1]  # Second most similar

        # Step 6: Calculate placement position with distance scaling
        primary_pos = anchor1['position']
        secondary_pos = anchor2['position']
        
        # Scale placement distance based on similarity score
        min_distance = 0.005   # Very close for perfect matches
        max_distance = 0.1    # Farther for poor matches (remember, coords are normalized to [0.0, 1.0])
        max_similarity = anchor1['adjusted_similarity']
        scaled_distance = min_distance + (max_distance - min_distance) * (1.0 - max_similarity)
        scaled_distance = max(min_distance, min(scaled_distance, max_distance))  # Clamp to bounds

        # Calculate direction from primary to secondary
        direction_vector = secondary_pos - primary_pos
        direction_length = np.linalg.norm(direction_vector)
        if direction_length > 0:
            direction_unit = direction_vector / direction_length
        else:
            direction_unit = np.random.random(2) - 0.5
            direction_unit = direction_unit / np.linalg.norm(direction_unit)

        placement_position = primary_pos + direction_unit * scaled_distance

        # Step 7: Find containing region
        regions_list = [
            {'id': region_id, 'vertices': vertices} 
            for region_id, vertices in region_vertices.items()
        ]
        primary_region = artwork_to_region_map.get(anchor1['id'])
        priority_regions = [str(primary_region)] if primary_region else []
        assigned_region = find_containing_region(
            placement_position, regions_list, priority_regions
        )
        if not assigned_region:
            assigned_region = primary_region

        # Step 8: Return result
        return {
            "success": True,
            "position": placement_position.tolist(),
            "regionId": assigned_region,
            "confidence": anchor1['adjusted_similarity'],
            "anchors": [
                {
                    "artworkId": candidate['id'],
                    "similarity": candidate['base_similarity'],
                    "adjustedSimilarity": candidate['adjusted_similarity'],
                    "position": candidate['position'].tolist(),
                    "distance": candidate['distance'],
                    "type": candidate['type']
                }
                for candidate in all_candidates[:9]
            ],
            "parameters": {
                "min_distance": min_distance,
                "max_distance": max_distance,
                "scaled_distance": scaled_distance
            },
            "anchorCounts": {
                "clip": len([c for c in all_candidates if c['type'] == 'clip']),
                "image": len([c for c in all_candidates if c['type'] == 'image']),
                "text": len([c for c in all_candidates if c['type'] == 'text'])
            }
        }

    except Exception as e:
        import traceback
        print(f"Exception in simplified multimodal placement: {e}")
        traceback.print_exc()
        return {
            "error": f"Error in simplified placement: {str(e)}",
            "traceback": traceback.format_exc()
        }
