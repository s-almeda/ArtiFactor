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
