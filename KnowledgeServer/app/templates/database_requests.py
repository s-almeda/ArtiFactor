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


@database_requests_bp.route('/artworks', methods=['GET'])
def get_artworks_paginated():
    """
    Retrieve artworks with pagination support.
    
    URL Parameters:
        limit: (optional, int, default=20) Number of artworks to return
        offset: (optional, int, default=0) Number of artworks to skip
        order_by: (optional, str, default='id') Field to order by (id, title, artist, etc.)
        order: (optional, str, default='ASC') Sort order (ASC or DESC)
    
    Returns:
        JSON response with paginated artwork data matching ArtworkData interface
    """
    try:
        db = get_db()
        
        # Get pagination parameters
        limit = min(int(request.args.get('limit', 20)), 100)  # Cap at 100 for performance
        offset = int(request.args.get('offset', 0))
        order_by = request.args.get('order_by', 'id')
        order = request.args.get('order', 'ASC').upper()
        
        # Validate order parameter
        if order not in ['ASC', 'DESC']:
            order = 'ASC'
        
        # Get total count for pagination info
        count_query = "SELECT COUNT(*) as total FROM image_entries"
        cursor = db.execute(count_query)
        total_count = cursor.fetchone()['total']
        
        # Validate order_by parameter against actual schema
        allowed_order_fields = ['image_id', 'value', 'artist_names', 'filename', 'rights']
        if order_by == 'id':  # Map 'id' to actual column name
            order_by = 'image_id'
        elif order_by == 'title':  # Map 'title' to actual column name  
            order_by = 'value'
        elif order_by == 'artist':  # Map 'artist' to actual column name
            order_by = 'artist_names'
        elif order_by not in allowed_order_fields:
            order_by = 'image_id'
        
        # Build the main query
        query = f"""
        SELECT 
            image_id as entryId,
            value as title,
            image_urls,
            descriptions,
            artist_names,
            rights,
            relatedKeywordStrings as keywords
        FROM image_entries 
        ORDER BY {order_by} {order}
        LIMIT ? OFFSET ?
        """
        
        cursor = db.execute(query, (limit, offset))
        results = cursor.fetchall()
        
        # Process the results to match ArtworkData interface
        artworks = []
        for row in results:
            artwork = dict(row)
            
            # Process JSON fields
            json_fields = ['image_urls', 'descriptions', 'artist_names', 'keywords']
            for field in json_fields:
                if field in artwork and isinstance(artwork[field], str):
                    try:
                        artwork[field] = json.loads(artwork[field])
                    except (json.JSONDecodeError, TypeError):
                        # Set defaults based on field type
                        if field == 'image_urls':
                            artwork[field] = {}
                        elif field == 'descriptions':
                            artwork[field] = {}
                        elif field in ['artist_names', 'keywords']:
                            artwork[field] = []
            
            # Ensure all required fields exist with proper defaults
            if 'image_urls' not in artwork or artwork['image_urls'] is None:
                artwork['image_urls'] = {}
            if 'descriptions' not in artwork or artwork['descriptions'] is None:
                artwork['descriptions'] = {}
            if 'artist_names' not in artwork or artwork['artist_names'] is None:
                artwork['artist_names'] = []
            if 'keywords' not in artwork or artwork['keywords'] is None:
                artwork['keywords'] = []
            if 'rights' not in artwork or artwork['rights'] is None:
                artwork['rights'] = ''
            if 'title' not in artwork or artwork['title'] is None:
                artwork['title'] = ''
            
            # Extract artist from artist_names array (first artist or empty string)
            if artwork['artist_names'] and len(artwork['artist_names']) > 0:
                artwork['artist'] = artwork['artist_names'][0]
            else:
                artwork['artist'] = ''
            
            # Smart selection for thumbnail_url (smallest available)
            thumbnail_priority = ['small', 'square', 'medium', 'large']
            thumbnail_url = ''
            for size in thumbnail_priority:
                if size in artwork['image_urls'] and artwork['image_urls'][size]:
                    thumbnail_url = artwork['image_urls'][size]
                    break
            if not thumbnail_url and artwork['image_urls']:
                # Fallback to first available if none of the priority sizes exist
                thumbnail_url = next(iter(artwork['image_urls'].values()), '')
            artwork['thumbnail_url'] = thumbnail_url
            
            # Smart selection for url (largest available)
            url_priority = ['large', 'medium', 'medium_rectangle', 'tall']
            url = ''
            for size in url_priority:
                if size in artwork['image_urls'] and artwork['image_urls'][size]:
                    url = artwork['image_urls'][size]
                    break
            if not url and artwork['image_urls']:
                # Fallback to first available if none of the priority sizes exist
                url = next(iter(artwork['image_urls'].values()), '')
            artwork['url'] = url
                
            artworks.append(artwork)
        
        # Calculate pagination metadata
        has_next = (offset + limit) < total_count
        has_prev = offset > 0
        total_pages = (total_count + limit - 1) // limit  # Ceiling division
        current_page = (offset // limit) + 1
        
        return jsonify({
            'success': True,
            'data': artworks,
            'pagination': {
                'total_count': total_count,
                'current_page': current_page,
                'total_pages': total_pages,
                'limit': limit,
                'offset': offset,
                'has_next': has_next,
                'has_prev': has_prev,
                'returned_count': len(artworks)
            }
        })
        
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': f'Invalid parameter value: {str(e)}'
        }), 400
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
