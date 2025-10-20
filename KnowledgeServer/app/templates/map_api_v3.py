# map_api_v3 third iteration of the map api routes we neeed uwu

from flask import Blueprint, jsonify, request, render_template
import os
import json, base64


from config import BASE_DIR
from helper_functions import helperfunctions as hf
from helper_functions.add_image_helperfunctions import get_image_nearest_neighbors_multimodal, place_query_image_multimodal
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# import helper_functions.voronoi_helper_functions as vhf

# 
MAPS_DIR = os.path.join(BASE_DIR, 'generated_maps')
os.makedirs(MAPS_DIR, exist_ok=True)


# Define the blueprint
map_api_v3_bp = Blueprint('map_api_v3', __name__)   


@map_api_v3_bp.route('/map-check')
def map_check_v3_page():
    """Serve the v3 map API check page."""
    return render_template('map_api_v3.html')

@map_api_v3_bp.route('/submit_map_job', methods=['POST'])
def submit_map_job():
    """Submit a map generation job and return job_id immediately"""
    try:
        data = request.get_json()
        debug = data.get('debug', True)
        
        def dprint(*args, **kwargs):
            if debug:
                print(*args, **kwargs)

        # unless 'regenerate' is explicitely set to 'true', check first for "good_map_data.json" in the maps dir
        regenerate = data.get('regenerate', False)
        dprint(f"Regenerate flag is set to: {regenerate}")
        if not regenerate:      
            dprint("Regenerate flag not set, checking for existing good_map_data.json...")
            #look for good_map_Data.json specifically, this is our canonical map from now on
            canonical_map = os.path.join(MAPS_DIR, "good_map_data.json")
            if os.path.exists(canonical_map):
                dprint("Found existing target map data, returning cached data.")
                try:
                    with open(canonical_map, 'r') as f:
                        cached_data = json.load(f)
                        cached_data['cached'] = True
                        cached_data['cache_key'] = 'good_map_data'
                        return jsonify({
                        'job_id': None,
                        'status': 'completed',
                        'result': cached_data
                    })
                except Exception as e:
                    dprint(f"⚠ Failed to load good map data, will process as job: {e}")

        # Generate cache key to check if result already exists
        dprint("Generating cache key for the submitted job...")
        cache_key = hf.generate_cache_key(data)
        cache_file = os.path.join(MAPS_DIR, f"{cache_key}.json")
        
        # If cached, return result immediately
        if os.path.exists(cache_file):
            dprint(f"Result already cached: {cache_key}")
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                    cached_data['cached'] = True
                    cached_data['cache_key'] = cache_key
                    return jsonify({
                        'job_id': None,
                        'status': 'completed',
                        'result': cached_data
                    })
            except Exception as e:
                dprint(f"⚠ Failed to load cache, will process as job: {e}")
        
        # Create job for processing
        from jobs import create_job
        job_id = create_job(data)
        
        dprint(f"Created job {job_id}")
        
        return jsonify({
            'job_id': job_id,
            'status': 'pending',
            'message': 'Job submitted successfully'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
    
@map_api_v3_bp.route('/job_status/<job_id>', methods=['GET'])
def get_job_status_endpoint(job_id):
    """Get the current status of a job"""
    from jobs import get_job_status
    
    job_status = get_job_status(job_id)
    
    if not job_status:
        return jsonify({'error': 'Job not found'}), 404
    
    response = {
        'job_id': job_id,
        'status': job_status['status'],
        'message': job_status['progress_message'],
        'created_at': job_status['created_at']
    }
    
    # If completed, include cache key for result retrieval
    if job_status['status'] == 'completed' and job_status['cache_key']:
        response['cache_key'] = job_status['cache_key']
    
    # If failed, include error
    if job_status['status'] == 'failed' and job_status['error_message']:
        response['error'] = job_status['error_message']
    
    return jsonify(response)

@map_api_v3_bp.route('/get_result/<cache_key>', methods=['GET'])
def get_cached_result(cache_key):
    """Get the completed result from cache"""
    cache_file = os.path.join(MAPS_DIR, f"{cache_key}.json")
    
    if not os.path.exists(cache_file):
        return jsonify({'error': 'Result not found'}), 404
    
    try:
        with open(cache_file, 'r') as f:
            result = json.load(f)
            return jsonify(result)
    except Exception as e:
        return jsonify({'error': f'Failed to load result: {str(e)}'}), 500


# -- all the map job handling has been moved to jobs/mapjob_processor.py!

@map_api_v3_bp.route('/api/baseline-image-search', methods=['POST'])
def baseline_image_search():
    """
    API endpoint to perform a baseline search for artworks using a query image and/or query text + CLIP / ResNet /MiniLM.
    if no query text, skip clip/text/exact match based search for artworks
    is no query image, skip image based search for artworks
    if neither, return error
    Expected JSON payload:
    {
        "queryImage": "data:image/jpeg;base64,/9j/4AAQ..." OR (preferably) an imageURL,
        "queryText": "A beautiful landscape painting with mountains", OPTIONAL
        "topK": 3  # Optional, default is 3 each, for a total of 9 if running all 3 functions,
        "artwork_ids": ["123", "456", "789"],  # Optional, filter to these artwork IDs
        "useArtworksJson": true, # if true, will only search within the 'artwork_ids.json' file that is stored in the backend server in the generated_maps folder
        "includeFullMetadata": true  # Optional, default is true. If false, returns lightweight results with just image_id, distance, and search_type
    }
    Returns:
    {
        "success": true,
        "clip_matches": [...],  // Multimodal matches (image + text)
        "image_matches": [...], // Visual similarity matches (ResNet)
        "text_matches": [...],  // Semantic text matches (MiniLM)
        "all_matches": [...]    // Combined deduplicated list for backward compatibility (only when includeFullMetadata=true)
    }
    """ 
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400

        data = request.get_json()
        query_image = data.get("queryImage")
        query_text = data.get("queryText")
        top_k = data.get("topK", 3)
        include_full_metadata = data.get("includeFullMetadata", True)

        if not query_image and not query_text:
            return jsonify({"error": "Missing a queryImage or queryText"}), 400

        from index import get_db
        db = get_db()

        artwork_ids = None
        if data.get("artwork_ids"):
            if not isinstance(data["artwork_ids"], list) or not all(isinstance(id, str) for id in data["artwork_ids"]):
                return jsonify({"error": "artwork_ids must be a list of strings"}), 400
            artwork_ids = data["artwork_ids"]

        elif data.get("useArtworksJson", False):
            # Load artwork IDs from the JSON file
            artwork_ids_file = os.path.join(MAPS_DIR, "artwork_ids.json")
            if not os.path.exists(artwork_ids_file):
                return jsonify({"error": "Artwork IDs file not found"}), 404
            
            with open(artwork_ids_file, "r", encoding="utf-8") as f:
                artwork_data = json.load(f)
                artwork_ids = artwork_data.get("artwork_ids", [])

        # Get baseline search results
        matches = get_image_nearest_neighbors_multimodal(query_image, query_text, db, top_k=top_k, artwork_ids=artwork_ids)
        
        print("Debug - Raw matches:", matches)  # Debug print
        
        # If lightweight results requested, return raw results with minimal processing
        if not include_full_metadata:
            clip_results = []
            image_results = []
            text_results = []

            # Process CLIP results (multimodal)
            if 'clip_results' in matches:
                for match in matches.get('clip_results', []):
                    image_id = match.get("artworkId") or match.get("image_id")
                    if image_id:
                        clip_results.append({
                            "image_id": image_id,
                            "distance": match.get("distance"),
                            "search_type": "clip"
                        })

            # Process image results (visual similarity)
            if 'image_results' in matches:
                for match in matches.get('image_results', []):
                    image_id = match.get("artworkId") or match.get("image_id")
                    if image_id:
                        image_results.append({
                            "image_id": image_id,
                            "distance": match.get("distance"),
                            "search_type": "image"
                        })

            # Process text results (semantic similarity)
            if 'text_results' in matches:
                for match in matches.get('text_results', []):
                    image_id = match.get("artworkId") or match.get("image_id")
                    if image_id:
                        text_results.append({
                            "image_id": image_id,
                            "distance": match.get("distance"),
                            "search_type": "text"
                        })

            print(f"Debug - Lightweight Results: {len(clip_results)} clip, {len(image_results)} image, {len(text_results)} text")

            # Return lightweight results
            return jsonify({
                "success": True,
                "clip_matches": clip_results,
                "image_matches": image_results,
                "text_matches": text_results,
                "summary": {
                    "clip_count": len(clip_results),
                    "image_count": len(image_results),
                    "text_count": len(text_results)
                }
            }), 200

        # Full metadata processing (existing logic)
        clip_results = []
        image_results = []
        text_results = []
        all_results = []
        seen_ids = set()

        IMAGES_PATH = os.path.join(BASE_DIR, "images")

        # Helper function to enrich match data with database info
        def enrich_match_with_db_data(match, search_type):
            image_id = match.get("artworkId") or match.get("image_id")
            
            query = "SELECT * FROM image_entries WHERE image_id = ?"
            cursor = db.execute(query, [image_id])
            db_row = cursor.fetchone()

            if not db_row:
                print(f"Debug - No database entry found for ID: {image_id}")
                return None

            db_row_dict = dict(db_row)
            image_urls = hf.safe_json_loads(db_row_dict.get('image_urls', '{}'), default={})
            image_url = None

            for size in ['large', 'medium', 'larger', 'small', 'square', 'tall']:
                url = image_urls.get(size)
                if url and hf.check_image_url(url):
                    image_url = url
                    break

            if not image_url and db_row_dict.get('filename'):
                try:
                    image_path = os.path.join(IMAGES_PATH, db_row_dict['filename'])
                    with open(image_path, "rb") as image_file:
                        image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
                    image_url = f"data:image/jpeg;base64,{image_base64}"
                except FileNotFoundError:
                    image_url = "https://upload.wikimedia.org/wikipedia/commons/a/a3/Image-not-found.png"

            return {
                "image_id": db_row_dict.get("image_id"),
                "value": db_row_dict.get("value", "Unknown"),
                "distance": match.get("distance") or match.get("similarity"),
                "search_type": search_type,  # Add search type to each result
                "image_url": image_url,
                "artist_names": db_row_dict.get("artist_names", "").split(", ") if db_row_dict.get("artist_names") else [],
                "image_urls": image_urls,
                "filename": db_row_dict.get("filename"),
                "rights": db_row_dict.get("rights", "Unknown"),
                "descriptions": hf.safe_json_loads(db_row_dict.get("descriptions", "{}"), default={}),
                "relatedKeywordIds": hf.safe_json_loads(db_row_dict.get("relatedKeywordIds", "[]"), default=[]),
                "relatedKeywordStrings": hf.safe_json_loads(db_row_dict.get("relatedKeywordStrings", "[]"), default=[])
            }

        # Process CLIP results (multimodal)
        if 'clip_results' in matches:
            for match in matches.get('clip_results', []):
                enriched = enrich_match_with_db_data(match, 'clip')
                if enriched:
                    clip_results.append(enriched)
                    if enriched['image_id'] not in seen_ids:
                        all_results.append(enriched)
                        seen_ids.add(enriched['image_id'])

        # Process image results (visual similarity)
        if 'image_results' in matches:
            for match in matches.get('image_results', []):
                enriched = enrich_match_with_db_data(match, 'image')
                if enriched:
                    image_results.append(enriched)
                    if enriched['image_id'] not in seen_ids:
                        all_results.append(enriched)
                        seen_ids.add(enriched['image_id'])

        # Process text results (semantic similarity)
        if 'text_results' in matches:
            for match in matches.get('text_results', []):
                enriched = enrich_match_with_db_data(match, 'text')
                if enriched:
                    text_results.append(enriched)
                    if enriched['image_id'] not in seen_ids:
                        all_results.append(enriched)
                        seen_ids.add(enriched['image_id'])

        print(f"Debug - Full Metadata Results: {len(clip_results)} clip, {len(image_results)} image, {len(text_results)} text")

        # Return full metadata results
        return jsonify({
            "success": True,
            "clip_matches": clip_results,   # Multimodal matches
            "image_matches": image_results,  # Visual matches
            "text_matches": text_results,    # Semantic text matches
            "all_matches": all_results,      # Combined deduplicated list
            "summary": {
                "total_unique": len(all_results),
                "clip_count": len(clip_results),
                "image_count": len(image_results),
                "text_count": len(text_results)
            }
        }), 200
        
    except Exception as e:
        import traceback
        return jsonify({
            "error": f"Server error: {str(e)}",
            "traceback": traceback.format_exc()
        }), 500
    

# @map_api_v3_bp.route('/api/place-query-multimodal', methods=['POST'])
# def place_query_multimodal():
#     """
#     API endpoint to place a query image using multimodal similarity (CLIP, ResNet, MiniLM).

#     Expected JSON payload:
#     {
#         "queryImage": as an imageURL 
#         "promptText": "A beautiful landscape painting with mountains",
#         "regions": [...],
#         "params": {
#             "minDistance": 0.1,
#             "maxDistance": 0.5,
#             "similarityWeight": 0.7
#         }
       
#     }

#     Returns:
#     {
#         "success": true,
#         "position": [0.23, 0.45],
#         "regionId": "region_1",
#         "confidence": 0.87,
#         "anchors": [
#             {"artworkId": "123", "distance": 0.1, "type": "clip", "position": [...]},
#             {"artworkId": "456", "distance": 0.3, "type": "image", "position": [...]},
#             {"artworkId": "789", "distance": 0.2, "type": "text", "position": [...]}  
#         ],
#         "alternativePlacements": {
#             "visualOnly": {"position": [0.18, 0.52], "regionId": "region_2"},
#             "textOnly": {"position": [0.31, 0.38], "regionId": "region_1"}
#         }
#     }
#     """
#     try:
#         print("Received request for /api/place-query-multimodal")
#         if not request.is_json:
#             return jsonify({"error": "Request must be JSON"}), 400

#         data = request.get_json()
#         required_fields = ['queryImage', 'promptText', 'regions']
#         missing_fields = [field for field in required_fields if field not in data]
#         if missing_fields:
#             return jsonify({"error": f"Missing required fields: {missing_fields}"}), 400

#         query_image = data['queryImage']
#         prompt_text = data['promptText']
#         regions = data['regions']
#         params = data.get('params', {})

#         if not regions:
#             return jsonify({"error": "No regions provided"}), 400

#         artwork_positions = {}
#         artwork_to_region = {}
#         region_vertices = {}

#         for region in regions:
#             region_id = str(region['id'])
#             region_vertices[region_id] = region['vertices']

#             for artwork in region.get('artworksMap', []):
#                 artwork_id = artwork['id']
#                 artwork_positions[artwork_id] = [
#                     artwork['coords']['x'],
#                     artwork['coords']['y']
#                 ]
#                 artwork_to_region[artwork_id] = region_id

#         from index import get_db
#         db = get_db()

#         min_distance = params.get('minDistance', 0.1)
#         max_distance = params.get('maxDistance', 0.5)
#         similarity_weight = params.get('similarityWeight', 0.7)

#         # search will automatically filter to only the artworks in artwork_positions fyi
#         result = place_query_image_multimodal(
#             query_image=query_image,
#             prompt_text=prompt_text,
#             artwork_positions=artwork_positions,
#             artwork_to_region_map=artwork_to_region,
#             region_vertices=region_vertices,
#             db=db,
#             min_distance=min_distance,
#             max_distance=max_distance,
#             similarity_weight=similarity_weight
#         )

#         if "error" in result:
#             return jsonify(result), 400
#         return jsonify(result), 200

#     except Exception as e:
#         import traceback
#         return jsonify({
#             "error": f"Server error: {str(e)}",
#             "traceback": traceback.format_exc()
#         }), 500


@map_api_v3_bp.route('/api/get_similar_artworks_by_text', methods=['POST'])
def get_similar_artworks_by_text():
    """
    API endpoint to find the most similar artworks based on a query text.

    Expected JSON payload:
    {
        "queryText": "A beautiful landscape painting with mountains",
        "topK": 5  # Optional, default is 5
    }

    Returns:
    {
        "success": true,
        "matches": [
            {"image_id": "123", "similarity": 0.95},
            {"image_id": "456", "similarity": 0.89},
            ...
        ]
    }
    """
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400

        data = request.get_json()
        query_text = data.get("queryText")
        top_k = data.get("topK", 5)

        if not query_text:
            return jsonify({"error": "Missing required field: queryText"}), 400

        from index import get_db
        db = get_db()

        # Extract text features using MiniLM
        text_features = hf.extract_text_features(query_text)

        # Find similar artworks using the helper function
        matches = hf.find_similar_artworks_by_text(text_features, db, top_k=top_k)

        return jsonify({"success": True, "matches": matches}), 200

    except Exception as e:
        import traceback
        return jsonify({
            "error": f"Server error: {str(e)}",
            "traceback": traceback.format_exc()
        }), 500
    

#NEW DEMO MAPS ROUTE!

@map_api_v3_bp.route('/demo_maps', methods=['GET'])
def get_demo_maps():
    """
    Serve demo maps (visual, keyword, balanced) in a single response.
    Optional query param: ?size=small or ?size=large (default: large)
    Loads files from MAPS_DIR/v9_demo_maps/
    """
    try:
        demo_maps_dir = os.path.join(MAPS_DIR, 'v9_demo_maps')
        
        # Get size parameter from query string (default: large)
        size = request.args.get('size', 'large').lower()
        if size not in ['small', 'large']:
            size = 'large'
        
        # Choose filenames based on size
        prefix = '50_' if size == 'small' else '250_'
        map_files = {
            'visual': f'{prefix}visual_similarity.json',
            'keyword': f'{prefix}keyword_similarity.json', 
            'balanced': f'{prefix}balanced.json'
        }
        
        maps_data = {}
        for key, filename in map_files.items():
            file_path = os.path.join(demo_maps_dir, filename)
            if not os.path.exists(file_path):
                return jsonify({
                    'success': False,
                    'error': f'Demo map file not found: {filename}'
                }), 404
            try:
                with open(file_path, 'r') as f:
                    map_data = json.load(f)
                    map_data['map_type'] = key
                    map_data['cached'] = True
                    map_data['cache_key'] = f'demo_{key}_{size}'
                    maps_data[key] = map_data
            except json.JSONDecodeError as e:
                return jsonify({
                    'success': False,
                    'error': f'Invalid JSON in {filename}: {str(e)}'
                }), 500
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': f'Error loading {filename}: {str(e)}'
                }), 500
        
        # Return all maps
        return jsonify(maps_data)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to load demo maps: {str(e)}'
        }), 500

# =-- old version keepingn for legacy code for now 
# @map_api_v3_bp.route('/api/place-query-image', methods=['POST'])
# def place_query_image():
#     """
#     API endpoint to place a query image in the map using visual similarity triangulation.
    
#     Expected JSON payload:
#     {
#         "queryImage": "data:image/jpeg;base64,/9j/4AAQ..." OR as an imageURL,
#         "regions": [...],
#         "params": {
#             "minDistance": 0.1,  # Optional, minimum distance from nearest neighbor
#             "maxDistance": 0.5,  # Optional, maximum distance from nearest neighbor
#             "similarityWeight": 0.7  # Optional, weight for the most similar artwork
#         }
#     }
    
#     Returns:
#     {
#         "success": true,
#         "position": [0.23, 0.45],
#         "regionId": "region_1",
#         "confidence": 0.87,
#         "anchors": [...]
#     }
#     """
#     try:
#         # Validate request
#         print("Received request for /api/place-query-image")  # Debug print
#         if not request.is_json:
#             print("Request is not JSON")  # Debug print
#             return jsonify({"error": "Request must be JSON"}), 400
        
#         data = request.get_json()

#         print(f"Received request with {len(data.get('regions', []))} regions")  # Debug print
        
#         # Check required fields
#         required_fields = ['queryImage', 'regions']
#         missing_fields = [field for field in required_fields if field not in data]
#         if missing_fields:
#             print(f"Missing required fields: {missing_fields}")  # Debug print
#             return jsonify({
#                 "error": f"Missing required fields: {missing_fields}"
#             }), 400
        
#         # Extract data
#         query_image = data['queryImage']
#         regions = data['regions']
#         params = data.get('params', {})

#         print("Extracted all required fields from request")  # Debug print
        
#         # Validate data
#         if not regions:
#             print("No regions provided")  # Debug print
#             return jsonify({"error": "No regions provided"}), 400
        
#         if not query_image:
#             print("No query image provided")  # Debug print
#             return jsonify({"error": "No query image provided"}), 400
        
#         # Build redundant data structures from regions
#         artwork_positions = {}
#         artwork_to_region = {}
#         region_vertices = {}

#         for region in regions:
#             region_id = str(region['id'])
#             region_vertices[region_id] = region['vertices']
            
#             for artwork in region.get('artworksMap', []):
#                 artwork_id = artwork['id']
#                 # Combine region centroid with artwork offset
#                 artwork_positions[artwork_id] = [
#                     artwork['coords']['x'],
#                     artwork['coords']['y']
#                 ]
#                 artwork_to_region[artwork_id] = region_id

#         print(f"Region summary: {len(region_vertices)} regions, {len(artwork_positions)} artworks")
#         print(f"Sample region IDs: {list(region_vertices.keys())[:5]}")
#         print(f"Sample artwork count per region: {[len(r.get('artworksMap', [])) for r in regions[:3]]}")
        
#         from index import get_db
#         # Get database connection (assuming it's available globally or via app context)
#         db = get_db()  # You'll need to implement this based on your setup
        
#         print("Database connection acquired")  # Debug print
        
#         # Extract optional parameters
#         min_distance = params.get('minDistance', 0.1)
#         max_distance = params.get('maxDistance', 0.5)
#         similarity_weight = params.get('similarityWeight', 0.7)

#         print(f"Using params: minDistance={min_distance}, maxDistance={max_distance}, similarityWeight={similarity_weight}")  # Debug print
        
#         # Process the request
#         result = place_query_image_triangulated(
#             query_image=query_image,
#             artwork_positions=artwork_positions,
#             artwork_to_region_map=artwork_to_region,
#             region_vertices=region_vertices,
#             db=db,
#             min_distance=min_distance,
#             max_distance=max_distance,
#             similarity_weight=similarity_weight
#         )
        
#         print(f"Result from place_query_image_triangulated: {result}")  # Debug print
        
#         # Return result
#         if "error" in result:
#             print("Error in triangulation result")  # Debug print
#             return jsonify(result), 400
#         else:
#             print("Returning successful result")  # Debug print
#             return jsonify(result), 200
            
#     except Exception as e:
#         import traceback
#         print(f"Exception occurred: {e}")  # Debug print
#         return jsonify({
#             "error": f"Server error: {str(e)}",
#             "traceback": traceback.format_exc()
#         }), 500