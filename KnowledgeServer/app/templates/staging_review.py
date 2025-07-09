# staging_review.py

from flask import Blueprint, render_template, request, jsonify, current_app, send_file
import json
import os
import glob
from datetime import datetime
from index import get_db
from typing import Dict, List, Optional, Any

# Path to the staging directory (relative to app root)
STAGING_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'LOCALDB', 'staging'))

def get_latest_staging_file() -> Optional[str]:
    """Return the path to the most recent staging_data_*.json file in the staging dir, or None if not found."""
    files = glob.glob(os.path.join(STAGING_DIR, 'staging_data_*.json'))
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]

staging_review_bp = Blueprint('staging_review', __name__, url_prefix='/staging_review')

@staging_review_bp.route('/')
def staging_review_page():
    """Main staging review page"""
    return render_template('staging_review.html')

@staging_review_bp.route('/load_staging_data')
def load_staging_data():
    """Load the staging JSON file and return summary data"""
    try:
        staging_file_path = get_latest_staging_file()
        if not staging_file_path or not os.path.exists(staging_file_path):
            return jsonify({
                'success': False,
                'error': 'Staging data file not found'
            })
        with open(staging_file_path, 'r', encoding='utf-8') as f:
            staging_data = json.load(f)
        # Debug: print out keywords and descriptions for first artist if present
        if staging_data.get('artists'):
            first_artist = staging_data['artists'][0]
            # Try all possible keyword fields
            print('DEBUG: First artist keywords:', first_artist.get('keywords'))
            print('DEBUG: First artist RelatedKeywordStrings:', first_artist.get('RelatedKeywordStrings'))
            print('DEBUG: First artist RelatedKeywordIds:', first_artist.get('RelatedKeywordIds'))
            print('DEBUG: First artist descriptions:', first_artist.get('descriptions'))
            if first_artist.get('artworks'):
                first_artwork = first_artist['artworks'][0]
                print('DEBUG: First artwork keywords:', first_artwork.get('keywords'))
                print('DEBUG: First artwork RelatedKeywordStrings:', first_artwork.get('RelatedKeywordStrings'))
                print('DEBUG: First artwork RelatedKeywordIds:', first_artwork.get('RelatedKeywordIds'))
                print('DEBUG: First artwork descriptions:', first_artwork.get('descriptions'))
        processed_data = process_staging_data(staging_data)
        return jsonify({
            'success': True,
            'data': processed_data
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@staging_review_bp.route('/get_artist_details/<artist_slug>')
def get_artist_details(artist_slug):
    """Get detailed information about a specific artist and their artworks"""
    try:
        staging_file_path = get_latest_staging_file()
        if not staging_file_path or not os.path.exists(staging_file_path):
            return jsonify({
                'success': False,
                'error': 'Staging data file not found'
            })
        with open(staging_file_path, 'r', encoding='utf-8') as f:
            staging_data = json.load(f)
        # Find the artist in staging data
        artist_data = None
        for artist in staging_data.get('artists', []):
            if artist.get('slug') == artist_slug:
                artist_data = artist
                break
        if not artist_data:
            return jsonify({
                'success': False,
                'error': 'Artist not found in staging data'
            })
        # Get database info for this artist
        db = get_db()
        db_artist = None
        if artist_data.get('is_existing'):
            db_artist = db.execute(
                'SELECT * FROM text_entries WHERE entry_id = ? AND isArtist = 1',
                (artist_data.get('existing_id'),)
            ).fetchone()
        # Get existing artworks for this artist (from DB)
        db_existing_artworks = []
        if db_artist:
            artist_name = db_artist['value']
            db_existing_artworks = db.execute('''
                SELECT ie.*, te.value as title 
                FROM image_entries ie
                JOIN text_entries te ON ie.image_id = te.entry_id
                WHERE ie.artist_names LIKE ?
            ''', (f'%{artist_name}%',)).fetchall()

        # Build a set of image_ids from staging to avoid duplicates
        staged_image_ids = set()
        for aw in artist_data.get('artworks', []):
            if aw.get('image_id'):
                staged_image_ids.add(str(aw['image_id']))

        # Process artworks: combine staged and DB, mark status
        processed_artworks = []
        # 1. Staged artworks (new or update)
        for artwork in artist_data.get('artworks', []):
            artwork_info = {
                'staging_data': artwork,
                'status': 'new' if not artwork.get('is_existing') else 'update',
                'existing_data': None
            }
            if artwork.get('is_existing'):
                # Find existing artwork in database
                existing_artwork = db.execute(
                    'SELECT * FROM image_entries WHERE image_id = ?',
                    (artwork.get('existing_id'),)
                ).fetchone()
                if existing_artwork:
                    artwork_info['existing_data'] = dict(existing_artwork)
            processed_artworks.append(artwork_info)
        # 2. DB-only artworks (not in staging)
        for row in db_existing_artworks:
            row_dict = dict(row)
            if str(row_dict.get('image_id')) not in staged_image_ids:
                # Try to get keywords if present (assume comma-separated string or JSON array)
                keywords = []
                if 'keywords' in row_dict and row_dict['keywords']:
                    try:
                        if isinstance(row_dict['keywords'], str):
                            if row_dict['keywords'].startswith('['):
                                import ast
                                keywords = ast.literal_eval(row_dict['keywords'])
                            else:
                                keywords = [k.strip() for k in row_dict['keywords'].split(',') if k.strip()]
                        elif isinstance(row_dict['keywords'], list):
                            keywords = row_dict['keywords']
                    except Exception:
                        pass
                    # Extract date from descriptions if available
                    date_value = ''
                    if 'descriptions' in row_dict and row_dict['descriptions']:
                        try:
                            import ast
                            descriptions = ast.literal_eval(row_dict['descriptions']) if isinstance(row_dict['descriptions'], str) else row_dict['descriptions']
                            if isinstance(descriptions, dict) and 'wikiart' in descriptions and 'date' in descriptions['wikiart']:
                                date_value = descriptions['wikiart']['date']
                        except Exception:
                            pass
                    
                processed_artworks.append({
                    'staging_data': {
                        'value': row_dict.get('value', ''),  # Use 'value' instead of 'title'
                        'date': date_value,  # Extract from descriptions.wikiart.date
                        'image_id': row_dict.get('image_id', ''),
                        'image_urls': {'medium': row_dict.get('image_url', '')},
                        'keywords': keywords,
                        'is_existing': True
                    },
                    'status': 'db',
                    'existing_data': row_dict
                })
        return jsonify({
            'success': True,
            'artist': {
                'staging_data': artist_data,
                'existing_data': dict(db_artist) if db_artist else None,
                'artworks': processed_artworks
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@staging_review_bp.route('/approve_artist', methods=['POST'])
def approve_artist():
    """Approve changes for a specific artist"""
    try:
        data = request.get_json()
        updated_artist = data.get('artist')
        if not updated_artist or 'slug' not in updated_artist:
            return jsonify({'success': False, 'error': 'No artist data provided'})
        artist_slug = updated_artist['slug']
        staging_file_path = get_latest_staging_file()
        if not staging_file_path or not os.path.exists(staging_file_path):
            return jsonify({'success': False, 'error': 'Staging data file not found'})
        with open(staging_file_path, 'r', encoding='utf-8') as f:
            staging_data = json.load(f)
        found = False
        for idx, artist in enumerate(staging_data.get('artists', [])):
            if artist.get('slug') == artist_slug:
                # Update the artist with all new fields from admin
                staging_data['artists'][idx] = updated_artist
                found = True
                break
        if not found:
            return jsonify({'success': False, 'error': 'Artist not found in staging data'})
        # Save the updated staging data
        with open(staging_file_path, 'w', encoding='utf-8') as f:
            json.dump(staging_data, f, indent=2, ensure_ascii=False)
        return jsonify({
            'success': True,
            'message': f'Artist {artist_slug} approved and staging data updated.'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

def process_staging_data(staging_data: Dict) -> Dict:
    """Process staging data to add database comparison information"""
    db = get_db()
    processed_data = staging_data.copy()
    
    # Process each artist
    for artist in processed_data.get('artists', []):
        # Check if artist exists in database
        if artist.get('is_existing') and artist.get('existing_id'):
            db_artist = db.execute(
                'SELECT * FROM text_entries WHERE entry_id = ? AND isArtist = 1',
                (artist.get('existing_id'),)
            ).fetchone()
            if db_artist:
                artist['db_info'] = dict(db_artist)
                # Get existing artworks count by matching artist_names JSON array
                artist_name = db_artist['value']
                existing_artworks_count = db.execute(
                    'SELECT COUNT(*) FROM image_entries WHERE artist_names LIKE ?',
                    (f'%{artist_name}%',)
                ).fetchone()[0]
                artist['existing_artworks_count'] = existing_artworks_count
            else:
                artist['db_info'] = None
                artist['existing_artworks_count'] = 0
        else:
            artist['db_info'] = None
            artist['existing_artworks_count'] = 0
        # Count new artworks for this artist
        new_artworks_count = sum(1 for artwork in artist.get('artworks', []) 
                                if not artwork.get('is_existing'))
        artist['new_artworks_count'] = new_artworks_count
    
    # Add summary statistics
    processed_data['summary'] = {
        'total_artists': len(processed_data.get('artists', [])),
        'new_artists': sum(1 for artist in processed_data.get('artists', []) 
                          if not artist.get('is_existing')),
        'existing_artists': sum(1 for artist in processed_data.get('artists', []) 
                               if artist.get('is_existing')),
        'total_new_artworks': sum(artist.get('new_artworks_count', 0) 
                                 for artist in processed_data.get('artists', [])),
        'total_artworks_to_update': sum(1 for artwork in processed_data.get('artworks', []) 
                                       if artwork.get('is_existing'))
    }
    
    return processed_data

@staging_review_bp.route('/get_staging_summary')
def get_staging_summary():
    """Get a summary of the staging data for the dashboard"""
    try:
        staging_file_path = get_latest_staging_file()
        if not staging_file_path or not os.path.exists(staging_file_path):
            return jsonify({
                'success': False,
                'error': 'Staging data file not found'
            })
        with open(staging_file_path, 'r', encoding='utf-8') as f:
            staging_data = json.load(f)
        metadata = staging_data.get('metadata', {})
        summary = {
            'timestamp': metadata.get('timestamp'),
            'total_processed': metadata.get('total_processed', 0),
            'total_artists': metadata.get('total_artists', 0),
            'total_artworks': metadata.get('total_artworks', 0),
            'new_artists_count': metadata.get('new_artists_count', 0),
            'new_artworks_count': metadata.get('new_artworks_count', 0),
            'limit': metadata.get('limit', 0),
            'download_enabled': metadata.get('download_enabled', False)
        }
        return jsonify({
            'success': True,
            'summary': summary
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@staging_review_bp.route('/reject_artist', methods=['POST'])
def reject_artist():
    """Reject changes for a specific artist"""
    try:
        data = request.get_json()
        artist_slug = data.get('artist_slug')
        staging_file_path = get_latest_staging_file()
        if not staging_file_path or not os.path.exists(staging_file_path):
            return jsonify({'success': False, 'error': 'Staging data file not found'})
        with open(staging_file_path, 'r', encoding='utf-8') as f:
            staging_data = json.load(f)
        # Remove the artist from the staging data (or mark as skipped)
        new_artists = []
        rejected = False
        for artist in staging_data.get('artists', []):
            if artist.get('slug') == artist_slug:
                rejected = True
                continue  # skip this artist
            new_artists.append(artist)
        staging_data['artists'] = new_artists
        with open(staging_file_path, 'w', encoding='utf-8') as f:
            json.dump(staging_data, f, indent=2, ensure_ascii=False)
        return jsonify({
            'success': True,
            'message': f'Rejected changes for artist {artist_slug}'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })
    
# New: List all per-artist JSON files
@staging_review_bp.route('/list_artist_files')
def list_artist_files():
    """Return a list of all per-artist staging JSON files (sorted by name)"""
    files = glob.glob(os.path.join(STAGING_DIR, 'staging_artist_*.json'))
    files.sort()
    artist_files = []
    for f in files:
        try:
            with open(f, 'r', encoding='utf-8') as jf:
                data = json.load(jf)
                slug = data.get('metadata', {}).get('slug') or os.path.basename(f)
                name = data.get('metadata', {}).get('artist') or slug
                artist_files.append({
                    'filename': os.path.basename(f),
                    'slug': slug,
                    'name': name
                })
        except Exception as e:
            continue
    return jsonify({'success': True, 'files': artist_files})

# New: Load a specific artist JSON file by filename
@staging_review_bp.route('/load_artist_data/<filename>')
def load_artist_data(filename):
    """Load a specific per-artist JSON file by filename"""
    safe_name = os.path.basename(filename)
    file_path = os.path.join(STAGING_DIR, safe_name)
    if not os.path.exists(file_path):
        return jsonify({'success': False, 'error': 'File not found'})
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return jsonify({'success': True, 'data': data})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# New: Remove a specific artist JSON file by filename
@staging_review_bp.route('/remove_artist_file/<filename>', methods=['DELETE'])
def remove_artist_file(filename):
    """Remove a specific per-artist JSON file by filename"""
    safe_name = os.path.basename(filename)
    file_path = os.path.join(STAGING_DIR, safe_name)
    if not os.path.exists(file_path):
        return jsonify({'success': False, 'error': 'File not found'})
    try:
        os.remove(file_path)
        return jsonify({'success': True, 'message': f'Artist file {safe_name} removed successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# New: Save processed artist data to a clean JSON file
@staging_review_bp.route('/save_processed_artist', methods=['POST'])
def save_processed_artist():
    """Save processed artist data to a clean, database-ready JSON file"""
    try:
        data = request.get_json()
        processed_data = data.get('processed_data')
        original_filename = data.get('original_filename')
        
        if not processed_data or not original_filename:
            return jsonify({'success': False, 'error': 'Missing required data'})
        
        # Create the processed JSON structure
        clean_data = create_clean_artist_json(processed_data)
        
        # Save to the same filename in staging directory
        safe_name = os.path.basename(original_filename)
        file_path = os.path.join(STAGING_DIR, safe_name)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(clean_data, f, indent=2, ensure_ascii=False)
        
        return jsonify({
            'success': True,
            'message': f'Processed artist data saved to {safe_name}'
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

def create_clean_artist_json(form_data):
    """Transform form data into clean, database-ready JSON structure"""
    artist_data = form_data.get('artist', {})
    artworks_data = form_data.get('artworks', [])
    
    # Check if artist is skipped
    if artist_data.get('skipped') or form_data.get('skipped'):
        # Create minimal structure for skipped artist
        return {
            'metadata': {
                'timestamp': form_data.get('timestamp', ''),
                'artist': artist_data.get('name', ''),
                'slug': artist_data.get('slug', ''),
                'artwork_count': 0,
                'processed': True,
                'skipped': True
            },
            'artist': {
                'name': artist_data.get('name', ''),
                'slug': artist_data.get('slug', ''),
                'skipped': True,
                'descriptions': {},
                'RelatedKeywordIds': '[]',
                'RelatedKeywordStrings': '[]',
                'artworks': []
            }
        }
    
    # Process artist keywords
    artist_keywords = []
    artist_keyword_ids = []
    
    # Check if keywords were sent from the form processing
    if 'RelatedKeywordStrings' in form_data and 'RelatedKeywordIds' in form_data:
        try:
            artist_keywords = json.loads(form_data['RelatedKeywordStrings'])
            artist_keyword_ids = json.loads(form_data['RelatedKeywordIds'])
        except (json.JSONDecodeError, TypeError):
            pass
    
    # If no keywords were processed from form, try to get from existing data
    if not artist_keywords and artist_data.get('RelatedKeywordStrings'):
        try:
            existing_keywords = json.loads(artist_data.get('RelatedKeywordStrings', '[]'))
            if existing_keywords:
                artist_keywords = existing_keywords
        except (json.JSONDecodeError, TypeError):
            pass
    
    # Also preserve existing keyword IDs if available
    if not artist_keyword_ids and artist_data.get('RelatedKeywordIds'):
        try:
            existing_keyword_ids = json.loads(artist_data.get('RelatedKeywordIds', '[]'))
            if existing_keyword_ids:
                artist_keyword_ids = existing_keyword_ids
        except (json.JSONDecodeError, TypeError):
            pass
    
    # Ensure artist has an entry_id for consistency
    artist_entry_id = artist_data.get('existing_id')
    if not artist_entry_id:
        # Generate new ID for new artist
        import time
        import random
        import string
        timestamp = int(time.time())
        random_part = ''.join(random.choices(string.ascii_lowercase + string.digits, k=16))
        artist_entry_id = f"{timestamp:x}{random_part}"
        artist_data['generated_entry_id'] = artist_entry_id
    
    # Process artworks
    processed_artworks = []
    
    for awidx, artwork in enumerate(artworks_data):
        if artwork.get('removed'):
            continue
            
        # Process artwork keywords
        artwork_keywords = []
        artwork_keyword_ids = []
        
        # Check if keywords were sent from the form processing
        if 'relatedKeywordStrings' in artwork and 'relatedKeywordIds' in artwork:
            try:
                artwork_keywords = json.loads(artwork['relatedKeywordStrings'])
                artwork_keyword_ids = json.loads(artwork['relatedKeywordIds'])
            except (json.JSONDecodeError, TypeError):
                pass
        
        # If no keywords were processed from form, preserve existing keywords
        if not artwork_keywords and artwork.get('relatedKeywordStrings'):
            try:
                existing_keywords = json.loads(artwork.get('relatedKeywordStrings', '[]'))
                if existing_keywords:
                    artwork_keywords = existing_keywords
            except (json.JSONDecodeError, TypeError):
                pass
        
        # Also preserve existing keyword IDs if available
        if not artwork_keyword_ids and artwork.get('relatedKeywordIds'):
            try:
                existing_keyword_ids = json.loads(artwork.get('relatedKeywordIds', '[]'))
                if existing_keyword_ids:
                    artwork_keyword_ids = existing_keyword_ids
            except (json.JSONDecodeError, TypeError):
                pass
        
        # IMPORTANT: Ensure artist's entry_id is in relatedKeywordIds and artist name is in relatedKeywordStrings
        artist_entry_id = artist_data.get('existing_id') or artist_data.get('generated_entry_id')
        artist_name = artist_data.get('name', '')
        
        if not artist_entry_id:
            # Generate new ID for new artist if not already generated
            import time
            import random
            import string
            timestamp = int(time.time())
            random_part = ''.join(random.choices(string.ascii_lowercase + string.digits, k=16))
            artist_entry_id = f"{timestamp:x}{random_part}"
            artist_data['generated_entry_id'] = artist_entry_id
        
        # Ensure artist is first in the keyword lists
        if artist_entry_id not in artwork_keyword_ids:
            artwork_keyword_ids.insert(0, artist_entry_id)
        if artist_name not in artwork_keywords:
            artwork_keywords.insert(0, artist_name)
        
        # Build artwork entry
        artwork_entry = {
            'value': artwork.get('value', ''),
            'image_id': artwork.get('image_id', ''),
            'is_existing': artwork.get('is_existing', False),
            'existing_id': artwork.get('existing_id'),
            'artist_names': [artist_data.get('name', '')],
            'image_urls': artwork.get('image_urls', {}),
            'filename': artwork.get('filename', ''),
            'rights': artwork.get('rights', ''),
            'descriptions': artwork.get('descriptions', {}),
            'relatedKeywordIds': json.dumps(artwork_keyword_ids),
            'relatedKeywordStrings': json.dumps(artwork_keywords)
        }
        
        processed_artworks.append(artwork_entry)
    
    # Build artist entry that matches the original JSON structure
    artist_entry = {
        'name': artist_data.get('name', ''),
        'slug': artist_data.get('slug', ''),
        'is_existing': artist_data.get('is_existing', False),
        'existing_id': artist_data.get('existing_id'),
        'generated_entry_id': artist_data.get('generated_entry_id'),  # Include generated ID
        'artist_aliases': artist_data.get('artist_aliases', []),
        'descriptions': artist_data.get('descriptions', {}),
        'RelatedKeywordIds': json.dumps(artist_keyword_ids),
        'RelatedKeywordStrings': json.dumps(artist_keywords),
        'artworks': processed_artworks
    }
    
    # Create final clean JSON structure that matches the original format
    clean_data = {
        'metadata': {
            'timestamp': form_data.get('timestamp', ''),
            'artist': artist_entry['name'],
            'slug': artist_entry['slug'],
            'artwork_count': len(processed_artworks),
            'processed': True
        },
        'artist': artist_entry
    }
    
    return clean_data

# New: Generate final SQL-ready JSON from all processed artist files
@staging_review_bp.route('/generate_final_sql_json', methods=['POST'])
def generate_final_sql_json():
    """Generate final SQL-ready JSON from all processed artist files"""
    try:
        # Get all processed artist files
        files = glob.glob(os.path.join(STAGING_DIR, 'staging_artist_*.json'))
        if not files:
            return jsonify({'success': False, 'error': 'No artist files found'})
        
        # Structure for final SQL commands
        final_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_files_processed': len(files),
                'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            'text_entries': [],  # For INSERT/UPDATE to text_entries (artists)
            'image_entries': [],  # For INSERT/UPDATE to image_entries (artworks)
            'stats': {
                'total_artists': 0,
                'new_artists': 0,
                'existing_artists': 0,
                'total_artworks': 0,
                'new_artworks': 0,
                'existing_artworks': 0,
                'skipped_artists': 0
            }
        }
        
        # Process each artist file
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    artist_data = json.load(f)
                
                artist_info = artist_data.get('artist', {})
                
                # Skip if artist is marked as skipped
                if artist_info.get('skipped'):
                    final_data['stats']['skipped_artists'] += 1
                    continue
                
                # Process artist for text_entries
                artist_entry = create_text_entry_from_artist(artist_info)
                final_data['text_entries'].append(artist_entry)
                
                # Update stats
                final_data['stats']['total_artists'] += 1
                if artist_info.get('is_existing'):
                    final_data['stats']['existing_artists'] += 1
                else:
                    final_data['stats']['new_artists'] += 1
                
                # Process artworks for image_entries
                artworks = artist_info.get('artworks', [])
                for artwork in artworks:
                    artwork_entry = create_image_entry_from_artwork(artwork, artist_info)
                    final_data['image_entries'].append(artwork_entry)
                    
                    # Update stats
                    final_data['stats']['total_artworks'] += 1
                    if artwork.get('is_existing'):
                        final_data['stats']['existing_artworks'] += 1
                    else:
                        final_data['stats']['new_artworks'] += 1
                        
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                continue
        
        # Save final JSON
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        final_filename = f'final_sql_ready_{timestamp}.json'
        final_path = os.path.join(STAGING_DIR, final_filename)
        
        with open(final_path, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False)
        
        return jsonify({
            'success': True,
            'filename': final_filename,
            'stats': final_data['stats'],
            'message': f'Final SQL JSON generated with {final_data["stats"]["total_artists"]} artists and {final_data["stats"]["total_artworks"]} artworks'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

# New: Download final SQL JSON file
@staging_review_bp.route('/download_final_json/<filename>')
def download_final_json(filename):
    """Download the final SQL-ready JSON file"""
    try:
        safe_filename = os.path.basename(filename)
        file_path = os.path.join(STAGING_DIR, safe_filename)
        
        if not os.path.exists(file_path):
            return jsonify({'success': False, 'error': 'File not found'}), 404
        
        return send_file(file_path, as_attachment=True, download_name=safe_filename)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

def create_text_entry_from_artist(artist_info):
    """Create a text_entries row structure from artist info"""
    # Safely parse keyword arrays from JSON strings
    try:
        related_keyword_ids = json.loads(artist_info.get('RelatedKeywordIds', '[]'))
    except (json.JSONDecodeError, TypeError):
        related_keyword_ids = []
    
    try:
        related_keyword_strings = json.loads(artist_info.get('RelatedKeywordStrings', '[]'))
    except (json.JSONDecodeError, TypeError):
        related_keyword_strings = []
    
    # Get or generate artist entry_id
    artist_entry_id = artist_info.get('existing_id')
    if not artist_entry_id:
        artist_entry_id = generate_new_id()
        # Store generated ID back to artist_info for use in artwork processing
        artist_info['generated_entry_id'] = artist_entry_id
    
    return {
        'entry_id': artist_entry_id,
        'value': artist_info.get('name', ''),
        'images': [artwork.get('image_id') for artwork in artist_info.get('artworks', []) if artwork.get('image_id')],
        'isArtist': 1,
        'type': 'artist',
        'artist_aliases': artist_info.get('artist_aliases', []),
        'descriptions': artist_info.get('descriptions', {}),
        'relatedKeywordIds': json.dumps(related_keyword_ids),  # Store as JSON string for DB
        'relatedKeywordStrings': json.dumps(related_keyword_strings),  # Store as JSON string for DB
        'sql_action': 'UPDATE' if artist_info.get('is_existing') else 'INSERT'
    }

def create_image_entry_from_artwork(artwork, artist_info):
    """Create an image_entries row structure from artwork info"""
    # Safely parse keyword arrays from JSON strings
    try:
        related_keyword_ids = json.loads(artwork.get('relatedKeywordIds', '[]'))
    except (json.JSONDecodeError, TypeError):
        related_keyword_ids = []
    
    try:
        related_keyword_strings = json.loads(artwork.get('relatedKeywordStrings', '[]'))
    except (json.JSONDecodeError, TypeError):
        related_keyword_strings = []
    
    # IMPORTANT: Ensure artist's entry_id is in relatedKeywordIds and artist name is in relatedKeywordStrings
    artist_entry_id = artist_info.get('existing_id') or artist_info.get('generated_entry_id')
    artist_name = artist_info.get('name', '')
    
    if not artist_entry_id:
        # Generate new ID for new artist if not already generated
        artist_entry_id = generate_new_id()
    
    # Ensure artist is first in the keyword lists
    if artist_entry_id not in related_keyword_ids:
        related_keyword_ids.insert(0, artist_entry_id)
    if artist_name not in related_keyword_strings:
        related_keyword_strings.insert(0, artist_name)
    
    return {
        'image_id': artwork.get('image_id') or generate_new_id(),
        'value': artwork.get('value', ''),
        'artist_names': [artist_name],
        'image_urls': artwork.get('image_urls', {}),
        'filename': artwork.get('filename', ''),
        'rights': artwork.get('rights', ''),
        'descriptions': artwork.get('descriptions', {}),
        'relatedKeywordIds': json.dumps(related_keyword_ids),  # Store as JSON string for DB
        'relatedKeywordStrings': json.dumps(related_keyword_strings),  # Store as JSON string for DB
        'sql_action': 'UPDATE' if artwork.get('is_existing') else 'INSERT'
    }

def generate_new_id():
    """Generate a new MongoDB-style ID for new entries"""
    import time
    import random
    import string
    
    timestamp = int(time.time())
    random_part = ''.join(random.choices(string.ascii_lowercase + string.digits, k=16))
    return f"{timestamp:x}{random_part}"