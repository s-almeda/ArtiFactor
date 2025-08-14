# templates/database_requests.py
"""
This module defines the Flask blueprint for database requests:
endpoints for retrieving entries from the image_entries and text_entries tables
"""

from flask import Blueprint, jsonify, request, g
from index import get_db
from helper_functions import helperfunctions as hf  # helper functions including preprocess_text
import json

# Define the blueprint
database_requests_bp = Blueprint('database_requests', __name__)

@database_requests_bp.route('/text_entry_by_name/<query>', methods=['GET'])
def get_text_entry_by_name(query):
    """
    Retrieve text entries by keyword (exact match, sluggified).
    Optionally restrict to artists and search aliases if artist_only=true is passed as a URL parameter.

    Args:
        query: Keyword to search for (string)

    URL Params:
        artist_only: (optional, bool) If true, restrict search to artists and search aliases.

    Returns:
        JSON response with matching entries or an error message
    """
    try:
        db = get_db()
        artist_only = request.args.get('artist_only', 'false').lower() == 'true'
        slug = hf.slugify(query, ' ')
        if artist_only:
            matches = hf.find_exact_matches(slug, db, artists_only=True, search_aliases=True)
        else:
            matches = hf.find_exact_matches(slug, db, artists_only=False, search_aliases=False)
        if matches:
            return jsonify({
                'success': True,
                'data': matches
            })
        else:
            return jsonify({
                'success': False,
                'error': f'No entry found with name {slug}'
            }), 404
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@database_requests_bp.route('/text/<entry_id>', methods=['GET'])
def get_text_entry(entry_id):
    """
    Retrieve a single text entry by its ID.
    
    Args:
        entry_id: ID of the text entry
        
    Returns:
        JSON response with the text entry data or an error message
    """
    try:
        db = get_db()
        entry = hf.retrieve_by_id(entry_id, db, entry_type="text")
        
        if entry:
            return jsonify({
                'success': True,
                'data': entry
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Text entry with ID {entry_id} not found'
            }), 404
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@database_requests_bp.route('/artwork/<image_id>', methods=['GET'])
def get_artwork_entry(image_id):
    """
    Retrieve a single image entry by its ID.
    
    Args:
        image_id: ID of the image entry
        
    Returns:
        JSON response with the image entry data or an error message
    """
    try:
        db = get_db()
        entry = hf.retrieve_by_id(image_id, db, entry_type="image")
        
        if entry:
            # Process any JSON fields stored as strings
            for field in ['image_urls', 'artist_names', 'relatedKeywordIds', 'relatedKeywordStrings', 'descriptions']:
                if field in entry and isinstance(entry[field], str):
                    try:
                        entry[field] = json.loads(entry[field])
                    except (json.JSONDecodeError, TypeError):
                        pass  # Keep as string if not valid JSON
            
            return jsonify({
                'success': True,
                'data': entry
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Artwork entry with ID {image_id} not found'
            }), 404
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@database_requests_bp.route('/database_request', methods=['POST'])
def batch_database_request():
    """
    Retrieve multiple entries by their IDs.
    
    Expected request JSON:
    {
        "entry_type": "image" or "text",
        "ids": [list of entry IDs]
    }
    
    Returns:
        JSON response with the requested entries or an error message
    """
    try:
        request_data = request.get_json()
        if not request_data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided in request'
            }), 400
        
        entry_type = request_data.get('entry_type', 'image')
        ids = request_data.get('ids', [])
        
        if not ids:
            return jsonify({
                'success': False,
                'error': 'No IDs provided in request'
            }), 400
            
        if not isinstance(ids, list):
            return jsonify({
                'success': False,
                'error': 'IDs must be provided as a list'
            }), 400
        
        db = get_db()
        results = []
        
        for id_value in ids:
            entry = hf.retrieve_by_id(id_value, db, entry_type=entry_type)
            if entry:
                # Process JSON fields for image entries
                if entry_type == "image":
                    for field in ['image_urls', 'artist_names', 'relatedKeywordIds', 'relatedKeywordStrings', 'descriptions']:
                        if field in entry and isinstance(entry[field], str):
                            try:
                                entry[field] = json.loads(entry[field])
                            except (json.JSONDecodeError, TypeError):
                                pass  # Keep as string if not valid JSON
                                
                results.append(entry)
        
        return jsonify({
            'success': True,
            'data': results,
            'count': len(results),
            'requested': len(ids),
            'entry_type': entry_type
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Error handlers
@database_requests_bp.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Not found'
    }), 404

@database_requests_bp.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500
