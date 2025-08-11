# map_api_v3 third iteration of the map api routes we neeed uwu

from flask import Blueprint, jsonify, request, render_template
import os
import json


from config import BASE_DIR
from helper_functions import helperfunctions as hf
from helper_functions.add_image_helperfunctions import place_query_image_triangulated, place_query_image_multimodal
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# import helper_functions.voronoi_helper_functions as vhf

# 
MAPS_DIR = os.path.join(BASE_DIR, 'generated_maps')
os.makedirs(MAPS_DIR, exist_ok=True)


# Define the blueprint
map_api_v3_bp = Blueprint('map_api_v3', __name__)   


@map_api_v3_bp.route('/map-check-v3')
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
        
        # Generate cache key to check if result already exists
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



@map_api_v3_bp.route('/api/place-query-multimodal', methods=['POST'])
def place_query_multimodal():
    """
    API endpoint to place a query image using multimodal similarity (CLIP, ResNet, MiniLM).
    
    Expected JSON payload:
    {
        "queryImage": "data:image/jpeg;base64,/9j/4AAQ..." OR as an imageURL,
        "promptText": "A beautiful landscape painting with mountains",
        "regions": [...],
        "params": {
            "minDistance": 0.1,
            "maxDistance": 0.5,
            "similarityWeight": 0.7
        }
    }
    
    Returns:
    {
        "success": true,
        "position": [0.23, 0.45],  // Primary CLIP-based placement
        "regionId": "region_1",
        "confidence": 0.87,
        "anchors": [
            {"artworkId": "123", "distance": 0.1, "type": "clip", "position": [...]},
            {"artworkId": "456", "distance": 0.3, "type": "image", "position": [...]},
            {"artworkId": "789", "distance": 0.2, "type": "text", "position": [...]}
        ],
        "alternativePlacements": {
            "visualOnly": {"position": [0.18, 0.52], "regionId": "region_2"},
            "textOnly": {"position": [0.31, 0.38], "regionId": "region_1"}
        }
    }
    """
    try:
        print("Received request for /api/place-query-multimodal")
        if not request.is_json:
            print("Request is not JSON")
            return jsonify({"error": "Request must be JSON"}), 400
        
        data = request.get_json()
        print(f"Received multimodal request with {len(data.get('regions', []))} regions")
        
        # Check required fields
        required_fields = ['queryImage', 'promptText', 'regions']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            print(f"Missing required fields: {missing_fields}")
            return jsonify({
                "error": f"Missing required fields: {missing_fields}"
            }), 400
        
        # Extract data
        query_image = data['queryImage']
        prompt_text = data['promptText']
        regions = data['regions']
        params = data.get('params', {})

        print("Extracted all required fields from multimodal request")
        
        # Validate data
        if not regions:
            print("No regions provided")
            return jsonify({"error": "No regions provided"}), 400
        
        if not query_image:
            print("No query image provided")
            return jsonify({"error": "No query image provided"}), 400
            
        if not prompt_text:
            print("No prompt text provided")
            return jsonify({"error": "No prompt text provided"}), 400
        
        # Build data structures from regions (same as legacy route)
        artwork_positions = {}
        artwork_to_region = {}
        region_vertices = {}

        for region in regions:
            region_id = str(region['id'])
            region_vertices[region_id] = region['vertices']
            
            for artwork in region.get('artworksMap', []):
                artwork_id = artwork['id']
                artwork_positions[artwork_id] = [
                    artwork['coords']['x'],
                    artwork['coords']['y']
                ]
                artwork_to_region[artwork_id] = region_id

        print(f"Region summary: {len(region_vertices)} regions, {len(artwork_positions)} artworks")
        
        from index import get_db
        db = get_db()
        print("Database connection acquired")
        
        # Extract optional parameters
        min_distance = params.get('minDistance', 0.1)
        max_distance = params.get('maxDistance', 0.5)
        similarity_weight = params.get('similarityWeight', 0.7)

        print(f"Using params: minDistance={min_distance}, maxDistance={max_distance}, similarityWeight={similarity_weight}")
        
        # Process the multimodal request
        result = place_query_image_multimodal(
            query_image=query_image,
            prompt_text=prompt_text,
            artwork_positions=artwork_positions,
            artwork_to_region_map=artwork_to_region,
            region_vertices=region_vertices,
            db=db,
            min_distance=min_distance,
            max_distance=max_distance,
            similarity_weight=similarity_weight
        )
        
        print(f"Result from place_query_image_multimodal: {result}")
        
        # Return result
        if "error" in result:
            print("Error in multimodal placement")
            return jsonify(result), 400
        else:
            print("Returning successful multimodal result")
            return jsonify(result), 200
            
    except Exception as e:
        import traceback
        print(f"Exception in multimodal route: {e}")
        return jsonify({
            "error": f"Server error: {str(e)}",
            "traceback": traceback.format_exc()
        }), 500


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


# =-- old version keepingn for legacy code for now 
@map_api_v3_bp.route('/api/place-query-image', methods=['POST'])
def place_query_image():
    """
    API endpoint to place a query image in the map using visual similarity triangulation.
    
    Expected JSON payload:
    {
        "queryImage": "data:image/jpeg;base64,/9j/4AAQ..." OR as an imageURL,
        "regions": [...],
        "params": {
            "minDistance": 0.1,  # Optional, minimum distance from nearest neighbor
            "maxDistance": 0.5,  # Optional, maximum distance from nearest neighbor
            "similarityWeight": 0.7  # Optional, weight for the most similar artwork
        }
    }
    
    Returns:
    {
        "success": true,
        "position": [0.23, 0.45],
        "regionId": "region_1",
        "confidence": 0.87,
        "anchors": [...]
    }
    """
    try:
        # Validate request
        print("Received request for /api/place-query-image")  # Debug print
        if not request.is_json:
            print("Request is not JSON")  # Debug print
            return jsonify({"error": "Request must be JSON"}), 400
        
        data = request.get_json()

        print(f"Received request with {len(data.get('regions', []))} regions")  # Debug print
        
        # Check required fields
        required_fields = ['queryImage', 'regions']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            print(f"Missing required fields: {missing_fields}")  # Debug print
            return jsonify({
                "error": f"Missing required fields: {missing_fields}"
            }), 400
        
        # Extract data
        query_image = data['queryImage']
        regions = data['regions']
        params = data.get('params', {})

        print("Extracted all required fields from request")  # Debug print
        
        # Validate data
        if not regions:
            print("No regions provided")  # Debug print
            return jsonify({"error": "No regions provided"}), 400
        
        if not query_image:
            print("No query image provided")  # Debug print
            return jsonify({"error": "No query image provided"}), 400
        
        # Build redundant data structures from regions
        artwork_positions = {}
        artwork_to_region = {}
        region_vertices = {}

        for region in regions:
            region_id = str(region['id'])
            region_vertices[region_id] = region['vertices']
            
            for artwork in region.get('artworksMap', []):
                artwork_id = artwork['id']
                # Combine region centroid with artwork offset
                artwork_positions[artwork_id] = [
                    artwork['coords']['x'],
                    artwork['coords']['y']
                ]
                artwork_to_region[artwork_id] = region_id

        print(f"Region summary: {len(region_vertices)} regions, {len(artwork_positions)} artworks")
        print(f"Sample region IDs: {list(region_vertices.keys())[:5]}")
        print(f"Sample artwork count per region: {[len(r.get('artworksMap', [])) for r in regions[:3]]}")
        
        from index import get_db
        # Get database connection (assuming it's available globally or via app context)
        db = get_db()  # You'll need to implement this based on your setup
        
        print("Database connection acquired")  # Debug print
        
        # Extract optional parameters
        min_distance = params.get('minDistance', 0.1)
        max_distance = params.get('maxDistance', 0.5)
        similarity_weight = params.get('similarityWeight', 0.7)

        print(f"Using params: minDistance={min_distance}, maxDistance={max_distance}, similarityWeight={similarity_weight}")  # Debug print
        
        # Process the request
        result = place_query_image_triangulated(
            query_image=query_image,
            artwork_positions=artwork_positions,
            artwork_to_region_map=artwork_to_region,
            region_vertices=region_vertices,
            db=db,
            min_distance=min_distance,
            max_distance=max_distance,
            similarity_weight=similarity_weight
        )
        
        print(f"Result from place_query_image_triangulated: {result}")  # Debug print
        
        # Return result
        if "error" in result:
            print("Error in triangulation result")  # Debug print
            return jsonify(result), 400
        else:
            print("Returning successful result")  # Debug print
            return jsonify(result), 200
            
    except Exception as e:
        import traceback
        print(f"Exception occurred: {e}")  # Debug print
        return jsonify({
            "error": f"Server error: {str(e)}",
            "traceback": traceback.format_exc()
        }), 500