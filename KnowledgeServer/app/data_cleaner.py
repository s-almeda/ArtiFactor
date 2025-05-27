from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for
import json
import sqlite3

data_cleaner_bp = Blueprint('data_cleaner', __name__)

@data_cleaner_bp.route('/data_cleaner')
def data_cleaner():
    """Main data cleaner page"""
    return render_template('data_cleaner.html')

@data_cleaner_bp.route('/check_orphaned_images', methods=['POST'])
def check_orphaned_images():
    """Check for text_entries with invalid image_id references"""
    try:
        from index import get_db  # Adjust this import based on your app structure
        db = get_db()
        
        # Get all valid image_ids from image_entries
        valid_image_ids = set()
        cursor = db.execute("SELECT image_id FROM image_entries")
        for row in cursor.fetchall():
            valid_image_ids.add(row['image_id'])
        
        # Check text_entries for invalid image references
        faulty_entries = []
        cursor = db.execute("SELECT entry_id, value, images FROM text_entries WHERE images IS NOT NULL AND images != ''")
        
        for row in cursor.fetchall():
            entry_id = row['entry_id']
            value = row['value']
            images_json = row['images']
            
            try:
                # Parse the JSON array of image_ids
                if images_json:
                    image_ids = json.loads(images_json)
                    if isinstance(image_ids, list):
                        # Find invalid image_ids
                        invalid_ids = []
                        valid_ids = []
                        
                        for img_id in image_ids:
                            if img_id not in valid_image_ids:
                                invalid_ids.append(img_id)
                            else:
                                valid_ids.append(img_id)
                        
                        # If there are invalid IDs, add to faulty entries
                        if invalid_ids:
                            faulty_entries.append({
                                'entry_id': entry_id,
                                'value': value,
                                'invalid_image_ids': invalid_ids,
                                'valid_image_ids': valid_ids,
                                'total_image_refs': len(image_ids)
                            })
            except json.JSONDecodeError:
                # Handle malformed JSON
                faulty_entries.append({
                    'entry_id': entry_id,
                    'value': value,
                    'invalid_image_ids': ['MALFORMED_JSON'],
                    'valid_image_ids': [],
                    'total_image_refs': 0,
                    'json_error': True
                })
        
        return jsonify({
            'success': True,
            'faulty_entries': faulty_entries,
            'total_faulty': len(faulty_entries),
            'total_valid_images': len(valid_image_ids)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@data_cleaner_bp.route('/fix_orphaned_images', methods=['POST'])
def fix_orphaned_images():
    """Fix text_entries by removing invalid image_id references"""
    try:
        from index import get_db
        db = get_db()
        
        # Get the faulty entries data from the request
        faulty_entries = request.json.get('faulty_entries', [])
        
        if not faulty_entries:
            return jsonify({
                'success': False,
                'error': 'No faulty entries provided'
            })
        
        fixed_count = 0
        
        for entry in faulty_entries:
            entry_id = entry['entry_id']
            valid_image_ids = entry['valid_image_ids']
            
            # Handle malformed JSON entries
            if entry.get('json_error'):
                # Set images to empty array for malformed JSON
                new_images_json = json.dumps([])
            else:
                # Keep only valid image IDs
                new_images_json = json.dumps(valid_image_ids) if valid_image_ids else json.dumps([])
            
            # Update the database
            db.execute(
                "UPDATE text_entries SET images = ? WHERE entry_id = ?",
                (new_images_json, entry_id)
            )
            fixed_count += 1
        
        # Commit the changes
        db.commit()
        
        return jsonify({
            'success': True,
            'fixed_count': fixed_count,
            'message': f'Successfully fixed {fixed_count} entries'
        })
        
    except Exception as e:
        db.rollback()  # Rollback on error
        return jsonify({
            'success': False,
            'error': str(e)
        })

@data_cleaner_bp.route('/get_database_stats', methods=['GET'])
def get_database_stats():
    """Get basic database statistics"""
    try:
        from index import get_db
        db = get_db()
        
        # Count total entries
        text_cursor = db.execute("SELECT COUNT(*) as count FROM text_entries")
        text_count = text_cursor.fetchone()['count']
        
        image_cursor = db.execute("SELECT COUNT(*) as count FROM image_entries")
        image_count = image_cursor.fetchone()['count']
        
        # Count entries with image references
        text_with_images_cursor = db.execute(
            "SELECT COUNT(*) as count FROM text_entries WHERE images IS NOT NULL AND images != '' AND images != '[]'"
        )
        text_with_images_count = text_with_images_cursor.fetchone()['count']
        
        return jsonify({
            'success': True,
            'stats': {
                'total_text_entries': text_count,
                'total_image_entries': image_count,
                'text_entries_with_images': text_with_images_count
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@data_cleaner_bp.route('/check_malformed_json', methods=['POST'])
def check_malformed_json():
    """Check for Python-style lists/strings that should be JSON in images column"""
    try:
        from index import get_db
        import ast
        
        db = get_db()
        
        malformed_entries = []
        cursor = db.execute("SELECT entry_id, value, images FROM text_entries WHERE images IS NOT NULL AND images != ''")
        
        for row in cursor.fetchall():
            entry_id = row['entry_id']
            value = row['value']
            images_text = row['images']
            
            if not images_text:
                continue
                
            # Try to parse as JSON first
            try:
                json.loads(images_text)
                # If it parses as JSON, it's fine
                continue
            except json.JSONDecodeError:
                # JSON parsing failed, now check if it's a Python-style list/string
                try:
                    # Try to evaluate as Python literal (safe evaluation)
                    python_obj = ast.literal_eval(images_text)
                    
                    # Check if it's a list (which is what we expect for images)
                    if isinstance(python_obj, list):
                        # Convert to proper JSON
                        proper_json = json.dumps(python_obj)
                        
                        malformed_entries.append({
                            'entry_id': entry_id,
                            'value': value,
                            'current_text': images_text,
                            'converted_json': proper_json,
                            'python_object': python_obj,
                            'type': 'python_list'
                        })
                    elif isinstance(python_obj, str):
                        # Single string that should be in an array
                        proper_json = json.dumps([python_obj])
                        
                        malformed_entries.append({
                            'entry_id': entry_id,
                            'value': value,
                            'current_text': images_text,
                            'converted_json': proper_json,
                            'python_object': [python_obj],
                            'type': 'python_string'
                        })
                    else:
                        # Some other Python object
                        malformed_entries.append({
                            'entry_id': entry_id,
                            'value': value,
                            'current_text': images_text,
                            'converted_json': None,
                            'python_object': python_obj,
                            'type': 'unknown_python_object',
                            'error': f'Unexpected type: {type(python_obj)}'
                        })
                        
                except (ValueError, SyntaxError):
                    # Not valid Python either - truly malformed
                    malformed_entries.append({
                        'entry_id': entry_id,
                        'value': value,
                        'current_text': images_text,
                        'converted_json': None,
                        'python_object': None,
                        'type': 'truly_malformed',
                        'error': 'Cannot parse as JSON or Python'
                    })
        
        return jsonify({
            'success': True,
            'malformed_entries': malformed_entries,
            'total_malformed': len(malformed_entries)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@data_cleaner_bp.route('/fix_malformed_json', methods=['POST'])
def fix_malformed_json():
    """Fix malformed JSON by converting Python-style to proper JSON"""
    try:
        from index import get_db
        db = get_db()
        
        malformed_entries = request.json.get('malformed_entries', [])
        
        if not malformed_entries:
            return jsonify({
                'success': False,
                'error': 'No malformed entries provided'
            })
        
        fixed_count = 0
        skipped_count = 0
        
        for entry in malformed_entries:
            entry_id = entry['entry_id']
            converted_json = entry.get('converted_json')
            
            # Only fix entries that have a valid conversion
            if converted_json is not None:
                db.execute(
                    "UPDATE text_entries SET images = ? WHERE entry_id = ?",
                    (converted_json, entry_id)
                )
                fixed_count += 1
            else:
                skipped_count += 1
        
        # Commit the changes
        db.commit()
        
        return jsonify({
            'success': True,
            'fixed_count': fixed_count,
            'skipped_count': skipped_count,
            'message': f'Successfully fixed {fixed_count} entries, skipped {skipped_count} unfixable entries'
        })
        
    except Exception as e:
        db.rollback()
        return jsonify({
            'success': False,
            'error': str(e)
        })


@data_cleaner_bp.route('/check_artists_without_images', methods=['POST'])
def check_artists_without_images():
    """Check for artists (isArtist=1) that have no images"""
    try:
        print("=== DEBUG: Starting artists without images check ===")
        from index import get_db
        db = get_db()
        
        # Find artists with no images or empty images
        artists_without_images = []
        cursor = db.execute("""
            SELECT entry_id, value, images, artist_aliases 
            FROM text_entries 
            WHERE isArtist = 1 
            AND (images IS NULL OR images = '' OR images = '[]')
        """)
        
        for row in cursor.fetchall():
            entry_id = row['entry_id']
            value = row['value']
            images = row['images']
            artist_aliases = row['artist_aliases']
            
            print(f"DEBUG: Found artist without images: {entry_id} - {value}")
            
            # Parse aliases if available
            aliases = []
            if artist_aliases:
                try:
                    aliases = json.loads(artist_aliases)
                except json.JSONDecodeError:
                    aliases = []
            
            artists_without_images.append({
                'entry_id': entry_id,
                'value': value,
                'images': images,
                'aliases': aliases
            })
        
        print(f"DEBUG: Found {len(artists_without_images)} artists without images")
        
        return jsonify({
            'success': True,
            'artists_without_images': artists_without_images,
            'total_artists_without_images': len(artists_without_images)
        })
        
    except Exception as e:
        print(f"ERROR in check_artists_without_images: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@data_cleaner_bp.route('/remove_artists_without_images', methods=['POST'])
def remove_artists_without_images():
    """Remove artists that have no images"""
    try:
        print("=== DEBUG: Starting remove artists without images ===")
        from index import get_db
        db = get_db()
        
        # Get the artist entries to remove
        artists_to_remove = request.json.get('artists_to_remove', [])
        print(f"DEBUG: Received {len(artists_to_remove)} artists to remove")
        
        if not artists_to_remove:
            return jsonify({
                'success': False,
                'error': 'No artists provided for removal'
            })
        
        removed_count = 0
        
        for artist in artists_to_remove:
            entry_id = artist['entry_id']
            value = artist.get('value', 'Unknown')
            
            print(f"DEBUG: Removing artist {entry_id} - {value}")
            
            # Delete the artist entry
            cursor = db.execute(
                "DELETE FROM text_entries WHERE entry_id = ? AND isArtist = 1",
                (entry_id,)
            )
            print(f"DEBUG: Delete query affected {cursor.rowcount} rows for artist {entry_id}")
            
            if cursor.rowcount > 0:
                removed_count += 1
        
        # Commit the changes
        db.commit()
        print(f"DEBUG: Committed changes, removed {removed_count} artists")
        
        return jsonify({
            'success': True,
            'removed_count': removed_count,
            'message': f'Successfully removed {removed_count} artists without images'
        })
        
    except Exception as e:
        print(f"ERROR in remove_artists_without_images: {str(e)}")
        try:
            db.rollback()
            print("DEBUG: Database rollback completed")
        except:
            print("DEBUG: Database rollback failed or not needed")
        return jsonify({
            'success': False,
            'error': str(e)
        })