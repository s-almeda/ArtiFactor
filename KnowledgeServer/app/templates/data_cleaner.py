from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for
import json
import sqlite3
import ast
import traceback

data_cleaner_bp = Blueprint('data_cleaner', __name__)

@data_cleaner_bp.route('/validate_admin_password', methods=['POST'])
def validate_admin_password():
    """Validate admin password for data cleaner access"""
    try:
        import os
        data = request.get_json()
        password = data.get('password', '')
        
        # Get the expected password from environment variable
        expected_password = os.environ.get('STAGING_ADMIN_PASSWORD', 'default_admin_pass')
        
        if password == expected_password:
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'error': 'Invalid password'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

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
    """Check for Python-style lists/strings that should be JSON in ALL JSON columns"""
    try:
        print("=== DEBUG: Starting comprehensive malformed JSON check ===")
        from index import get_db
        import ast
        
        db = get_db()
        
        malformed_entries = []
        
        # Define JSON columns for each table
        text_json_columns = ['images', 'artist_aliases', 'descriptions', 'relatedKeywordIds', 'relatedKeywordStrings']
        image_json_columns = ['artist_names', 'image_urls', 'descriptions', 'relatedKeywordIds', 'relatedKeywordStrings']
        
        # Check text_entries table
        print("DEBUG: Checking text_entries table...")
        text_cursor = db.execute("SELECT entry_id, value, images, artist_aliases, descriptions, relatedKeywordIds, relatedKeywordStrings FROM text_entries")
        
        for row in text_cursor.fetchall():
            entry_id = row['entry_id']
            value = row['value']
            
            for column in text_json_columns:
                json_text = row[column]
                if json_text:
                    result = check_json_column(entry_id, value, column, json_text, 'text_entries')
                    if result:
                        malformed_entries.append(result)
        
        # Check image_entries table
        print("DEBUG: Checking image_entries table...")
        image_cursor = db.execute("SELECT image_id, value, artist_names, image_urls, descriptions, relatedKeywordIds, relatedKeywordStrings FROM image_entries")
        
        for row in image_cursor.fetchall():
            image_id = row['image_id']
            value = row['value']
            
            for column in image_json_columns:
                json_text = row[column]
                if json_text:
                    result = check_json_column(image_id, value, column, json_text, 'image_entries')
                    if result:
                        malformed_entries.append(result)
        
        print(f"DEBUG: Total malformed entries found: {len(malformed_entries)}")
        
        return jsonify({
            'success': True,
            'malformed_entries': malformed_entries,
            'total_malformed': len(malformed_entries)
        })
        
    except Exception as e:
        print(f"ERROR in check_malformed_json: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

def check_json_column(entry_id, value, column_name, json_text, table_name):
    """Helper function to check a single JSON column"""
    try:
        # Try to parse as JSON first
        json.loads(json_text)
        # If it parses as JSON, it's fine
        return None
    except json.JSONDecodeError:
        print(f"DEBUG: {table_name}.{column_name} - JSON parse failed for {entry_id}")
        # JSON parsing failed, now check if it's a Python-style list/string
        try:
            import ast
            # Try to evaluate as Python literal (safe evaluation)
            python_obj = ast.literal_eval(json_text)
            
            # Check if it's a list or dict (which we expect for JSON columns)
            if isinstance(python_obj, (list, dict)):
                # Convert to proper JSON
                proper_json = json.dumps(python_obj)
                
                return {
                    'entry_id': entry_id,
                    'table': table_name,
                    'column': column_name,
                    'value': value,
                    'current_text': json_text,
                    'converted_json': proper_json,
                    'python_object': python_obj,
                    'type': f'python_{type(python_obj).__name__}'
                }
            elif isinstance(python_obj, str):
                # Single string that should be in an array (for array columns)
                if column_name in ['images', 'artist_names', 'artist_aliases', 'relatedKeywordIds', 'relatedKeywordStrings']:
                    proper_json = json.dumps([python_obj])
                    python_obj = [python_obj]
                else:
                    # For dict columns, wrap in quotes
                    proper_json = json.dumps(python_obj)
                
                return {
                    'entry_id': entry_id,
                    'table': table_name,
                    'column': column_name,
                    'value': value,
                    'current_text': json_text,
                    'converted_json': proper_json,
                    'python_object': python_obj,
                    'type': 'python_string'
                }
            else:
                # Some other Python object
                return {
                    'entry_id': entry_id,
                    'table': table_name,
                    'column': column_name,
                    'value': value,
                    'current_text': json_text,
                    'converted_json': None,
                    'python_object': python_obj,
                    'type': 'unknown_python_object',
                    'error': f'Unexpected type: {type(python_obj)}'
                }
                
        except (ValueError, SyntaxError):
            # Not valid Python either - truly malformed
            return {
                'entry_id': entry_id,
                'table': table_name,
                'column': column_name,
                'value': value,
                'current_text': json_text,
                'converted_json': None,
                'python_object': None,
                'type': 'truly_malformed',
                'error': 'Cannot parse as JSON or Python'
            }



@data_cleaner_bp.route('/fix_malformed_json', methods=['POST'])
def fix_malformed_json():
    """Fix malformed JSON by converting Python-style to proper JSON in ALL columns"""
    try:
        print("=== DEBUG: Starting comprehensive malformed JSON fix ===")
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
            table = entry['table']
            column = entry['column']
            converted_json = entry.get('converted_json')
            
            print(f"DEBUG: Processing {table}.{column} for entry {entry_id}")
            
            # Only fix entries that have a valid conversion
            if converted_json is not None:
                # Determine the correct ID column name
                id_column = 'entry_id' if table == 'text_entries' else 'image_id'
                
                # Update the specific column in the specific table
                query = f"UPDATE {table} SET {column} = ? WHERE {id_column} = ?"
                cursor = db.execute(query, (converted_json, entry_id))
                
                print(f"DEBUG: Updated {table}.{column} for {entry_id}, affected {cursor.rowcount} rows")
                fixed_count += 1
            else:
                print(f"DEBUG: Skipping unfixable entry {entry_id} in {table}.{column}")
                skipped_count += 1
        
        # Commit the changes
        db.commit()
        print(f"DEBUG: Committed changes, fixed {fixed_count} entries, skipped {skipped_count}")
        
        return jsonify({
            'success': True,
            'fixed_count': fixed_count,
            'skipped_count': skipped_count,
            'message': f'Successfully fixed {fixed_count} entries across all tables, skipped {skipped_count} unfixable entries'
        })
        
    except Exception as e:
        print(f"ERROR in fix_malformed_json: {str(e)}")
        try:
            db.rollback()
            print("DEBUG: Database rollback completed")
        except:
            print("DEBUG: Database rollback failed or not needed")
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


@data_cleaner_bp.route('/check_invalid_image_urls', methods=['POST'])
def check_invalid_image_urls():
    """Check for image entries with invalid or broken image URLs"""
    try:
        print("=== DEBUG: Starting invalid image URLs check ===")
        from index import get_db
        import requests
        from urllib.parse import urlparse
        
        db = get_db()
        
        invalid_entries = []
        cursor = db.execute("SELECT image_id, value, image_urls FROM image_entries WHERE image_urls IS NOT NULL AND image_urls != ''")
        
        total_checked = 0
        for row in cursor.fetchall():
            total_checked += 1
            image_id = row['image_id']
            value = row['value']
            image_urls_text = row['image_urls']
            
            print(f"DEBUG: Checking image {image_id}: {value}")
            
            try:
                # Parse the JSON
                image_urls = json.loads(image_urls_text)
                
                if not isinstance(image_urls, dict):
                    print(f"DEBUG: Image {image_id} - image_urls is not a dictionary")
                    invalid_entries.append({
                        'image_id': image_id,
                        'value': value,
                        'image_urls_text': image_urls_text,
                        'error': 'image_urls is not a JSON object/dictionary',
                        'invalid_urls': [],
                        'valid_urls': {},
                        'type': 'not_dict'
                    })
                    continue
                
                # Check each URL in the dictionary
                invalid_urls = []
                valid_urls = {}
                
                expected_sizes = ['large', 'large_rectangle', 'larger', 'medium', 'medium_rectangle', 
                                'normalized', 'small', 'square', 'tall']
                
                for size, url in image_urls.items():
                    if not url or not isinstance(url, str):
                        invalid_urls.append({'size': size, 'url': url, 'error': 'Empty or non-string URL'})
                        continue
                    
                    # Basic URL validation
                    try:
                        parsed = urlparse(url)
                        if not parsed.scheme or not parsed.netloc:
                            invalid_urls.append({'size': size, 'url': url, 'error': 'Invalid URL format'})
                            continue
                    except Exception as e:
                        invalid_urls.append({'size': size, 'url': url, 'error': f'URL parse error: {str(e)}'})
                        continue
                    
                    # Check if URL is accessible (with timeout)
                    try:
                        response = requests.head(url, timeout=5, allow_redirects=True)
                        if response.status_code == 200:
                            valid_urls[size] = url
                            print(f"DEBUG: Image {image_id} - {size} URL is valid")
                        else:
                            invalid_urls.append({'size': size, 'url': url, 'error': f'HTTP {response.status_code}'})
                            print(f"DEBUG: Image {image_id} - {size} URL returned {response.status_code}")
                    except requests.exceptions.Timeout:
                        invalid_urls.append({'size': size, 'url': url, 'error': 'Request timeout'})
                        print(f"DEBUG: Image {image_id} - {size} URL timed out")
                    except requests.exceptions.RequestException as e:
                        invalid_urls.append({'size': size, 'url': url, 'error': f'Request failed: {str(e)}'})
                        print(f"DEBUG: Image {image_id} - {size} URL failed: {str(e)}")
                
                # If there are any invalid URLs, add to the list
                if invalid_urls:
                    invalid_entries.append({
                        'image_id': image_id,
                        'value': value,
                        'image_urls_text': image_urls_text,
                        'invalid_urls': invalid_urls,
                        'valid_urls': valid_urls,
                        'total_urls': len(image_urls),
                        'type': 'broken_urls'
                    })
                
            except json.JSONDecodeError as e:
                print(f"DEBUG: Image {image_id} - JSON decode error: {str(e)}")
                invalid_entries.append({
                    'image_id': image_id,
                    'value': value,
                    'image_urls_text': image_urls_text,
                    'error': f'JSON decode error: {str(e)}',
                    'invalid_urls': [],
                    'valid_urls': {},
                    'type': 'json_error'
                })
        
        print(f"DEBUG: Checked {total_checked} images, found {len(invalid_entries)} with invalid URLs")
        
        return jsonify({
            'success': True,
            'invalid_entries': invalid_entries,
            'total_invalid': len(invalid_entries),
            'total_checked': total_checked
        })
        
    except Exception as e:
        print(f"ERROR in check_invalid_image_urls: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@data_cleaner_bp.route('/fix_invalid_image_urls', methods=['POST'])
def fix_invalid_image_urls():
    """Fix image entries by removing invalid URLs and keeping only valid ones"""
    try:
        print("=== DEBUG: Starting fix invalid image URLs ===")
        from index import get_db
        db = get_db()
        
        invalid_entries = request.json.get('invalid_entries', [])
        action = request.json.get('action', 'clean')  # 'clean' or 'remove'
        
        print(f"DEBUG: Received {len(invalid_entries)} entries to fix with action: {action}")
        
        if not invalid_entries:
            return jsonify({
                'success': False,
                'error': 'No invalid entries provided'
            })
        
        fixed_count = 0
        removed_count = 0
        
        for entry in invalid_entries:
            image_id = entry['image_id']
            value = entry.get('value', 'Unknown')
            
            print(f"DEBUG: Processing image {image_id} - {value}")
            
            if action == 'remove':
                # Remove the entire image entry
                cursor = db.execute("DELETE FROM image_entries WHERE image_id = ?", (image_id,))
                print(f"DEBUG: Removed image entry {image_id}, affected {cursor.rowcount} rows")
                removed_count += 1
            else:
                # Clean URLs - keep only valid ones
                valid_urls = entry.get('valid_urls', {})
                
                if valid_urls:
                    # Update with only valid URLs
                    new_image_urls_json = json.dumps(valid_urls)
                    cursor = db.execute(
                        "UPDATE image_entries SET image_urls = ? WHERE image_id = ?",
                        (new_image_urls_json, image_id)
                    )
                    print(f"DEBUG: Cleaned URLs for image {image_id}, kept {len(valid_urls)} valid URLs")
                    fixed_count += 1
                else:
                    # No valid URLs, set to empty object
                    cursor = db.execute(
                        "UPDATE image_entries SET image_urls = ? WHERE image_id = ?",
                        ('{}', image_id)
                    )
                    print(f"DEBUG: No valid URLs for image {image_id}, set to empty object")
                    fixed_count += 1
        
        # Commit the changes
        db.commit()
        print(f"DEBUG: Committed changes")
        
        if action == 'remove':
            message = f'Successfully removed {removed_count} image entries with invalid URLs'
        else:
            message = f'Successfully cleaned {fixed_count} image entries, keeping only valid URLs'
        
        return jsonify({
            'success': True,
            'fixed_count': fixed_count,
            'removed_count': removed_count,
            'message': message
        })
        
    except Exception as e:
        print(f"ERROR in fix_invalid_image_urls: {str(e)}")
        try:
            db.rollback()
            print("DEBUG: Database rollback completed")
        except:
            print("DEBUG: Database rollback failed or not needed")
        return jsonify({
            'success': False,
            'error': str(e)
        })

# Add these routes to your data_cleaner.py file

@data_cleaner_bp.route('/check_related_keywords', methods=['POST'])
def check_related_keywords():
    """Check for invalid RelatedKeywordIds and mismatched RelatedKeywordStrings"""
    try:
        print("=== DEBUG: Starting related keywords check ===")
        from index import get_db
        db = get_db()
        
        # Get all valid entry_ids from text_entries
        valid_entry_ids = set()
        entry_values = {}  # Map entry_id to value
        cursor = db.execute("SELECT entry_id, value FROM text_entries")
        for row in cursor.fetchall():
            entry_id = row['entry_id']
            valid_entry_ids.add(entry_id)
            entry_values[entry_id] = row['value']
        
        print(f"DEBUG: Found {len(valid_entry_ids)} valid entry IDs")
        
        issues = []
        
        # Check text_entries
        print("DEBUG: Checking text_entries table...")
        text_cursor = db.execute("""
            SELECT entry_id, value, relatedKeywordIds, relatedKeywordStrings 
            FROM text_entries 
            WHERE (relatedKeywordIds IS NOT NULL AND relatedKeywordIds != '' AND relatedKeywordIds != '[]')
               OR (relatedKeywordStrings IS NOT NULL AND relatedKeywordStrings != '' AND relatedKeywordStrings != '[]')
        """)
        
        for row in text_cursor.fetchall():
            entry_id = row['entry_id']
            value = row['value']
            keyword_ids_text = row['relatedKeywordIds']
            keyword_strings_text = row['relatedKeywordStrings']
            
            result = check_keyword_integrity(
                entry_id, value, keyword_ids_text, keyword_strings_text, 
                valid_entry_ids, entry_values, 'text_entries'
            )
            if result:
                issues.append(result)
        
        # Check image_entries
        print("DEBUG: Checking image_entries table...")
        image_cursor = db.execute("""
            SELECT image_id, value, relatedKeywordIds, relatedKeywordStrings 
            FROM image_entries 
            WHERE (relatedKeywordIds IS NOT NULL AND relatedKeywordIds != '' AND relatedKeywordIds != '[]')
               OR (relatedKeywordStrings IS NOT NULL AND relatedKeywordStrings != '' AND relatedKeywordStrings != '[]')
        """)
        
        for row in image_cursor.fetchall():
            image_id = row['image_id']
            value = row['value']
            keyword_ids_text = row['relatedKeywordIds']
            keyword_strings_text = row['relatedKeywordStrings']
            
            result = check_keyword_integrity(
                image_id, value, keyword_ids_text, keyword_strings_text, 
                valid_entry_ids, entry_values, 'image_entries'
            )
            if result:
                issues.append(result)
        
        print(f"DEBUG: Found {len(issues)} entries with keyword issues")
        
        return jsonify({
            'success': True,
            'issues': issues,
            'total_issues': len(issues),
            'total_valid_entries': len(valid_entry_ids)
        })
        
    except Exception as e:
        print(f"ERROR in check_related_keywords: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        })

def check_keyword_integrity(entry_id, value, keyword_ids_text, keyword_strings_text, 
                           valid_entry_ids, entry_values, table_name):
    """Helper function to check keyword integrity for a single entry"""
    try:
        # Create reverse lookup: value -> entry_id for finding missing IDs
        value_to_entry_id = {v: k for k, v in entry_values.items()}
        
        # Parse keyword IDs
        current_ids = []
        if keyword_ids_text and keyword_ids_text != '[]':
            try:
                current_ids = json.loads(keyword_ids_text)
                if not isinstance(current_ids, list):
                    current_ids = []
            except json.JSONDecodeError:
                print(f"DEBUG: JSON decode error for relatedKeywordIds in {table_name} {entry_id}")
                current_ids = []
        
        # Parse keyword strings
        current_strings = []
        if keyword_strings_text and keyword_strings_text != '[]':
            try:
                current_strings = json.loads(keyword_strings_text)
                if not isinstance(current_strings, list):
                    current_strings = []
            except json.JSONDecodeError:
                print(f"DEBUG: JSON decode error for relatedKeywordStrings in {table_name} {entry_id}")
                current_strings = []
        
        # Step 1: Separate valid and invalid IDs
        valid_ids = []
        invalid_ids = []
        for kid in current_ids:
            if kid in valid_entry_ids:
                valid_ids.append(kid)
            else:
                invalid_ids.append(kid)
        
        # Step 2: Separate valid and invalid strings
        valid_strings = []
        invalid_strings = []
        for string in current_strings:
            if string in value_to_entry_id:
                valid_strings.append(string)
            else:
                invalid_strings.append(string)
        
        # Step 3: Build the corrected lists
        corrected_ids = set(valid_ids)  # Start with valid IDs
        corrected_strings = set()
        
        # Add strings for all valid IDs
        for vid in valid_ids:
            if vid in entry_values:
                corrected_strings.add(entry_values[vid])
        
        # Add IDs for all valid strings
        for string in valid_strings:
            if string in value_to_entry_id:
                corrected_ids.add(value_to_entry_id[string])
                corrected_strings.add(string)
        
        # Convert back to sorted lists for consistency
        corrected_ids_list = sorted(list(corrected_ids))
        corrected_strings_list = sorted(list(corrected_strings))
        
        # Check if current arrays match the corrected ones
        current_ids_set = set(current_ids)
        current_strings_set = set(current_strings)
        
        ids_match = current_ids_set == corrected_ids
        strings_match = current_strings_set == corrected_strings
        
        # Only return an issue if there are problems to fix
        if not ids_match or not strings_match or invalid_ids or invalid_strings:
            return {
                'entry_id': entry_id,
                'table': table_name,
                'value': value,
                'current_ids': current_ids,
                'current_strings': current_strings,
                'corrected_ids': corrected_ids_list,
                'corrected_strings': corrected_strings_list,
                'invalid_ids': invalid_ids,
                'invalid_strings': invalid_strings,
                'valid_ids': valid_ids,
                'valid_strings': valid_strings,
                'ids_match': ids_match,
                'strings_match': strings_match
            }
        
        return None
        
    except Exception as e:
        print(f"ERROR in check_keyword_integrity for {entry_id}: {str(e)}")
        return {
            'entry_id': entry_id,
            'table': table_name,
            'value': value,
            'error': str(e),
            'current_ids': [],
            'current_strings': [],
            'corrected_ids': [],
            'corrected_strings': [],
            'invalid_ids': [],
            'invalid_strings': [],
            'valid_ids': [],
            'valid_strings': [],
            'ids_match': False,
            'strings_match': False
        }

@data_cleaner_bp.route('/fix_related_keywords', methods=['POST'])
def fix_related_keywords():
    """Fix RelatedKeywordIds and RelatedKeywordStrings"""
    try:
        print("=== DEBUG: Starting fix related keywords ===")
        from index import get_db
        db = get_db()
        
        issues = request.json.get('issues', [])
        
        if not issues:
            return jsonify({
                'success': False,
                'error': 'No issues provided'
            })
        
        fixed_count = 0
        
        for issue in issues:
            entry_id = issue['entry_id']
            table = issue['table']
            corrected_ids = issue.get('corrected_ids', [])
            corrected_strings = issue.get('corrected_strings', [])
            
            print(f"DEBUG: Fixing {table} entry {entry_id}")
            print(f"DEBUG: Current IDs: {issue.get('current_ids', [])}")
            print(f"DEBUG: Current strings: {issue.get('current_strings', [])}")
            print(f"DEBUG: Corrected IDs: {corrected_ids}")
            print(f"DEBUG: Corrected strings: {corrected_strings}")
            
            # Prepare the new JSON data
            new_ids_json = json.dumps(corrected_ids)
            new_strings_json = json.dumps(corrected_strings)
            
            # Determine the correct ID column name
            id_column = 'entry_id' if table == 'text_entries' else 'image_id'
            
            # Update the entry
            query = f"""
                UPDATE {table} 
                SET relatedKeywordIds = ?, relatedKeywordStrings = ? 
                WHERE {id_column} = ?
            """
            cursor = db.execute(query, (new_ids_json, new_strings_json, entry_id))
            
            print(f"DEBUG: Updated {table} entry {entry_id}, affected {cursor.rowcount} rows")
            fixed_count += 1
        
        # Commit the changes
        db.commit()
        print(f"DEBUG: Committed changes, fixed {fixed_count} entries")
        
        return jsonify({
            'success': True,
            'fixed_count': fixed_count,
            'message': f'Successfully fixed {fixed_count} entries with corrected keyword references'
        })
        
    except Exception as e:
        print(f"ERROR in fix_related_keywords: {str(e)}")
        try:
            db.rollback()
            print("DEBUG: Database rollback completed")
        except:
            print("DEBUG: Database rollback failed or not needed")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@data_cleaner_bp.route('/search_all_related_artworks', methods=['POST'])
def search_all_related_artworks():
    """Search for artworks related to all artists without images by checking relatedKeywordIds"""
    try:
        print("=== DEBUG: Starting bulk related artworks search ===")
        from index import get_db
        db = get_db()
        
        # First get all artists without images
        artists_cursor = db.execute("""
            SELECT entry_id, value, images, artist_aliases 
            FROM text_entries 
            WHERE isArtist = 1 
            AND (images IS NULL OR images = '' OR images = '[]')
        """)
        
        artists_with_matches = []
        artists_without_matches = []
        
        for artist_row in artists_cursor.fetchall():
            artist_entry_id = artist_row['entry_id']
            artist_name = artist_row['value']
            artist_aliases = artist_row['artist_aliases']
            
            # Parse aliases if available
            aliases = []
            if artist_aliases:
                try:
                    aliases = json.loads(artist_aliases)
                except json.JSONDecodeError:
                    aliases = []
            
            # Search for images that have this artist's entry_id in their relatedKeywordIds
            related_images = []
            cursor = db.execute("""
                SELECT image_id, value, artist_names, relatedKeywordIds 
                FROM image_entries 
                WHERE relatedKeywordIds LIKE ?
            """, (f'%"{artist_entry_id}"%',))
            
            for row in cursor.fetchall():
                image_id = row['image_id']
                value = row['value']
                artist_names = row['artist_names']
                related_keyword_ids = row['relatedKeywordIds']
                
                # Verify that the artist_entry_id is actually in the JSON array
                try:
                    keyword_ids = json.loads(related_keyword_ids) if related_keyword_ids else []
                    if artist_entry_id in keyword_ids:
                        related_images.append({
                            'image_id': image_id,
                            'value': value,
                            'artist_names': json.loads(artist_names) if artist_names else [],
                            'relatedKeywordIds': keyword_ids
                        })
                except json.JSONDecodeError:
                    continue
            
            artist_data = {
                'entry_id': artist_entry_id,
                'value': artist_name,
                'aliases': aliases,
                'related_images': related_images,
                'total_related': len(related_images)
            }
            
            if len(related_images) > 0:
                artists_with_matches.append(artist_data)
            else:
                artists_without_matches.append(artist_data)
        
        print(f"DEBUG: Found {len(artists_with_matches)} artists with matches, {len(artists_without_matches)} without matches")
        
        return jsonify({
            'success': True,
            'artists_with_matches': artists_with_matches,
            'artists_without_matches': artists_without_matches,
            'total_with_matches': len(artists_with_matches),
            'total_without_matches': len(artists_without_matches)
        })
        
    except Exception as e:
        print(f"ERROR in search_all_related_artworks: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@data_cleaner_bp.route('/execute_artist_updates', methods=['POST'])
def execute_artist_updates():
    """Execute updates: add image IDs to artists and remove artists with no matches"""
    try:
        print("=== DEBUG: Starting execute artist updates ===")
        from index import get_db
        db = get_db()
        
        data = request.json
        artists_with_matches = data.get('artists_with_matches', [])
        artists_without_matches = data.get('artists_without_matches', [])
        
        updated_artists = 0
        removed_artists = 0
        
        # Update artists with matches - add image IDs to their images array
        for artist in artists_with_matches:
            artist_entry_id = artist['entry_id']
            related_images = artist['related_images']
            
            if related_images:
                # Get current images array
                cursor = db.execute("SELECT images FROM text_entries WHERE entry_id = ?", (artist_entry_id,))
                row = cursor.fetchone()
                
                if row:
                    try:
                        current_images = json.loads(row['images']) if row['images'] and row['images'] != '[]' else []
                    except json.JSONDecodeError:
                        current_images = []
                    
                    # Add new image IDs (avoid duplicates)
                    new_image_ids = [img['image_id'] for img in related_images]
                    for img_id in new_image_ids:
                        if img_id not in current_images:
                            current_images.append(img_id)
                    
                    # Update the database
                    new_images_json = json.dumps(current_images)
                    db.execute(
                        "UPDATE text_entries SET images = ? WHERE entry_id = ?",
                        (new_images_json, artist_entry_id)
                    )
                    updated_artists += 1
                    print(f"DEBUG: Updated artist {artist_entry_id} with {len(new_image_ids)} new images")
        
        # Remove artists without matches
        for artist in artists_without_matches:
            artist_entry_id = artist['entry_id']
            cursor = db.execute(
                "DELETE FROM text_entries WHERE entry_id = ? AND isArtist = 1",
                (artist_entry_id,)
            )
            if cursor.rowcount > 0:
                removed_artists += 1
                print(f"DEBUG: Removed artist {artist_entry_id}")
        
        # Commit all changes
        db.commit()
        
        return jsonify({
            'success': True,
            'updated_artists': updated_artists,
            'removed_artists': removed_artists,
            'message': f'Successfully updated {updated_artists} artists with new images and removed {removed_artists} artists without any artwork'
        })
        
    except Exception as e:
        db.rollback()
        print(f"ERROR in execute_artist_updates: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@data_cleaner_bp.route('/check_artist_image_integrity', methods=['POST'])
def check_artist_image_integrity():
    """Check integrity between artists and their images"""
    try:
        print("=== DEBUG: Starting artist-image integrity check ===")
        from index import get_db
        db = get_db()
        
        integrity_issues = []
        
        # Get all artists with images
        cursor = db.execute("""
            SELECT entry_id, value, images 
            FROM text_entries 
            WHERE isArtist = 1 
            AND images IS NOT NULL 
            AND images != '' 
            AND images != '[]'
        """)
        
        for row in cursor.fetchall():
            artist_entry_id = row['entry_id']
            artist_name = row['value']
            images_json = row['images']
            
            try:
                image_ids = json.loads(images_json)
                if not isinstance(image_ids, list):
                    continue
                
                for image_id in image_ids:
                    # Check if image exists
                    image_cursor = db.execute("""
                        SELECT image_id, value, artist_names, relatedKeywordIds 
                        FROM image_entries 
                        WHERE image_id = ?
                    """, (image_id,))
                    
                    image_row = image_cursor.fetchone()
                    
                    if not image_row:
                        # Image doesn't exist
                        integrity_issues.append({
                            'type': 'missing_image',
                            'artist_entry_id': artist_entry_id,
                            'artist_name': artist_name,
                            'image_id': image_id,
                            'description': f"Artist '{artist_name}' references non-existent image {image_id}"
                        })
                    else:
                        # Image exists, check if artist is properly linked back
                        image_artist_names = json.loads(image_row['artist_names']) if image_row['artist_names'] else []
                        image_related_ids = json.loads(image_row['relatedKeywordIds']) if image_row['relatedKeywordIds'] else []
                        
                        # Check if artist name is in image's artist_names
                        if artist_name not in image_artist_names:
                            integrity_issues.append({
                                'type': 'missing_artist_name',
                                'artist_entry_id': artist_entry_id,
                                'artist_name': artist_name,
                                'image_id': image_id,
                                'image_value': image_row['value'],
                                'current_artist_names': image_artist_names,
                                'description': f"Image '{image_row['value']}' doesn't have artist '{artist_name}' in artist_names"
                            })
                        
                        # Check if artist entry_id is in image's relatedKeywordIds
                        if artist_entry_id not in image_related_ids:
                            integrity_issues.append({
                                'type': 'missing_related_keyword',
                                'artist_entry_id': artist_entry_id,
                                'artist_name': artist_name,
                                'image_id': image_id,
                                'image_value': image_row['value'],
                                'current_related_ids': image_related_ids,
                                'description': f"Image '{image_row['value']}' doesn't have artist entry_id '{artist_entry_id}' in relatedKeywordIds"
                            })
                            
            except json.JSONDecodeError:
                integrity_issues.append({
                    'type': 'malformed_json',
                    'artist_entry_id': artist_entry_id,
                    'artist_name': artist_name,
                    'images_json': images_json,
                    'description': f"Artist '{artist_name}' has malformed images JSON"
                })
        
        print(f"DEBUG: Found {len(integrity_issues)} integrity issues")
        
        return jsonify({
            'success': True,
            'integrity_issues': integrity_issues,
            'total_issues': len(integrity_issues)
        })
        
    except Exception as e:
        print(f"ERROR in check_artist_image_integrity: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@data_cleaner_bp.route('/fix_artist_image_integrity', methods=['POST'])
def fix_artist_image_integrity():
    """Fix artist-image integrity issues"""
    try:
        print("=== DEBUG: Starting artist-image integrity fix ===")
        from index import get_db
        db = get_db()
        
        # Get the integrity issues from the request
        data = request.json
        integrity_issues = data.get('integrity_issues', [])
        
        if not integrity_issues:
            return jsonify({
                'success': False,
                'error': 'No integrity issues provided'
            })
        
        fixed_count = 0
        
        for issue in integrity_issues:
            issue_type = issue['type']
            
            if issue_type == 'missing_image':
                # Remove the non-existent image ID from artist's images list
                artist_entry_id = issue['artist_entry_id']
                image_id = issue['image_id']
                
                # Get current images
                cursor = db.execute("SELECT images FROM text_entries WHERE entry_id = ?", (artist_entry_id,))
                row = cursor.fetchone()
                if row:
                    try:
                        current_images = json.loads(row['images'])
                        if image_id in current_images:
                            current_images.remove(image_id)
                            new_images_json = json.dumps(current_images)
                            db.execute("UPDATE text_entries SET images = ? WHERE entry_id = ?", 
                                     (new_images_json, artist_entry_id))
                            fixed_count += 1
                    except json.JSONDecodeError:
                        pass
                        
            elif issue_type == 'missing_artist_name':
                # Add artist name to image's artist_names
                image_id = issue['image_id']
                artist_name = issue['artist_name']
                
                cursor = db.execute("SELECT artist_names FROM image_entries WHERE image_id = ?", (image_id,))
                row = cursor.fetchone()
                if row:
                    try:
                        current_names = json.loads(row['artist_names']) if row['artist_names'] else []
                        if artist_name not in current_names:
                            current_names.append(artist_name)
                            new_names_json = json.dumps(current_names)
                            db.execute("UPDATE image_entries SET artist_names = ? WHERE image_id = ?",
                                     (new_names_json, image_id))
                            fixed_count += 1
                    except json.JSONDecodeError:
                        # Create new list with just this artist
                        new_names_json = json.dumps([artist_name])
                        db.execute("UPDATE image_entries SET artist_names = ? WHERE image_id = ?",
                                 (new_names_json, image_id))
                        fixed_count += 1
                        
            elif issue_type == 'missing_related_keyword':
                # Add artist entry_id to image's relatedKeywordIds
                image_id = issue['image_id']
                artist_entry_id = issue['artist_entry_id']
                
                cursor = db.execute("SELECT relatedKeywordIds FROM image_entries WHERE image_id = ?", (image_id,))
                row = cursor.fetchone()
                if row:
                    try:
                        current_ids = json.loads(row['relatedKeywordIds']) if row['relatedKeywordIds'] else []
                        if artist_entry_id not in current_ids:
                            current_ids.append(artist_entry_id)
                            new_ids_json = json.dumps(current_ids)
                            db.execute("UPDATE image_entries SET relatedKeywordIds = ? WHERE image_id = ?",
                                     (new_ids_json, image_id))
                            fixed_count += 1
                    except json.JSONDecodeError:
                        # Create new list with just this artist entry_id
                        new_ids_json = json.dumps([artist_entry_id])
                        db.execute("UPDATE image_entries SET relatedKeywordIds = ? WHERE image_id = ?",
                                 (new_ids_json, image_id))
                        fixed_count += 1
        
        # Commit all changes
        db.commit()
        
        return jsonify({
            'success': True,
            'fixed_count': fixed_count,
            'message': f'Successfully fixed {fixed_count} integrity issues'
        })
        
    except Exception as e:
        db.rollback()
        print(f"ERROR in fix_artist_image_integrity: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@data_cleaner_bp.route('/check_duplicate_images', methods=['POST'])
def check_duplicate_images():
    """Find potential duplicate artworks by the same artist with similar names"""
    try:
        from index import get_db
        import re
        import unicodedata
        import os
        from collections import defaultdict
        
        db = get_db()
        
        # Load previously marked "not duplicates" pairs
        not_duplicates_file = os.path.join(os.path.dirname(__file__), 'not_duplicates.txt')
        not_duplicate_pairs = set()
        
        if os.path.exists(not_duplicates_file):
            try:
                with open(not_duplicates_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and ',' in line:
                            parts = line.split(',')
                            if len(parts) >= 2:
                                # Store pairs in both directions for easy lookup
                                id1, id2 = parts[0].strip(), parts[1].strip()
                                not_duplicate_pairs.add((id1, id2))
                                not_duplicate_pairs.add((id2, id1))
                print(f"Loaded {len(not_duplicate_pairs)//2} not-duplicate pairs from file")
            except Exception as e:
                print(f"Warning: Could not load not_duplicates.txt: {e}")
                not_duplicate_pairs = set()
        
        def normalize_title(title):
            """Normalize title for comparison: lowercase, no accents, only alphanumeric"""
            if not title:
                return ""
            # Remove accents and diacritics
            title = unicodedata.normalize('NFD', title)
            title = ''.join(char for char in title if unicodedata.category(char) != 'Mn')
            # Convert to lowercase and keep only alphanumeric characters and spaces
            title = re.sub(r'[^a-zA-Z0-9\s]', '', title.lower())
            # Normalize whitespace
            title = ' '.join(title.split())
            return title
        
        # Get all image entries with their artist information
        cursor = db.execute("""
            SELECT ie.image_id, ie.value as title, ie.relatedKeywordIds, ie.descriptions, 
                   ie.image_urls, ie.artist_names, ie.rights, ie.relatedKeywordStrings
            FROM image_entries ie 
            WHERE ie.value IS NOT NULL AND ie.value != ''
            ORDER BY ie.image_id
        """)
        
        image_entries = cursor.fetchall()
        
        # Get artist information for all entries
        artist_lookup = {}
        cursor = db.execute("SELECT entry_id, value FROM text_entries WHERE value IS NOT NULL")
        for row in cursor.fetchall():
            artist_lookup[row['entry_id']] = row['value']
        
        # Group by artist and normalized title
        artist_artworks = defaultdict(lambda: defaultdict(list))
        
        for entry in image_entries:
            # Get artist name from artist_names field
            artist_name = None
            if entry['artist_names']:
                try:
                    artist_names = json.loads(entry['artist_names'])
                    if isinstance(artist_names, list) and artist_names:
                        # Use the first artist name
                        artist_name = artist_names[0]
                except:
                    pass
            
            if not artist_name:
                continue
                
            normalized_title = normalize_title(entry['title'])
            if not normalized_title:
                continue
                
            # Calculate additional information
            related_keywords_count = 0
            image_url_keys = []
            best_image_url = None
            
            try:
                if entry['relatedKeywordIds']:
                    keywords = json.loads(entry['relatedKeywordIds'])
                    if isinstance(keywords, list):
                        related_keywords_count = len(keywords)
            except:
                pass
                
            # Extract image URL keys and find best image URL
            try:
                if entry['image_urls']:
                    urls = json.loads(entry['image_urls'])
                    if isinstance(urls, dict):
                        image_url_keys = list(urls.keys())
                        # Try to find the best image URL (prefer medium size for display)
                        for preferred_key in ['medium', 'small', 'normalized', 'large']:
                            if preferred_key in urls:
                                best_image_url = urls[preferred_key]
                                break
                        # If no preferred key found, use the first available
                        if not best_image_url and urls:
                            best_image_url = list(urls.values())[0]
            except:
                pass
            
            artist_artworks[artist_name][normalized_title].append({
                'image_id': entry['image_id'],
                'value': entry['title'],
                'title': entry['title'],
                'artist_names': entry['artist_names'],
                'image_urls': entry['image_urls'],
                'image_url_keys': ', '.join(image_url_keys),
                'best_image_url': best_image_url,
                'rights': entry['rights'],
                'descriptions': entry['descriptions'],
                'relatedKeywordStrings': entry['relatedKeywordStrings'],
                'related_keywords': entry['relatedKeywordIds'],
                'related_keywords_count': related_keywords_count
            })
        
        # Find groups with multiple artworks (potential duplicates)
        duplicate_groups = []
        total_potential_duplicates = 0
        
        for artist_name, titles_dict in artist_artworks.items():
            for normalized_title, artworks in titles_dict.items():
                if len(artworks) > 1:  # More than one artwork with the same normalized title
                    # Filter out pairs that have been marked as "not duplicates"
                    filtered_artworks = []
                    
                    for artwork in artworks:
                        # Check if this artwork should be excluded based on not_duplicate_pairs
                        should_include = True
                        for other_artwork in artworks:
                            if artwork['image_id'] != other_artwork['image_id']:
                                pair_key = (str(artwork['image_id']), str(other_artwork['image_id']))
                                if pair_key in not_duplicate_pairs:
                                    should_include = False
                                    break
                        
                        if should_include:
                            filtered_artworks.append(artwork)
                    
                    # Only include groups that still have potential duplicates after filtering
                    if len(filtered_artworks) > 1:
                        duplicate_groups.append({
                            'artist_name': artist_name,
                            'normalized_title': normalized_title,
                            'count': len(filtered_artworks),
                            'items': filtered_artworks
                        })
                        total_potential_duplicates += len(filtered_artworks)
        
        # Sort by artist name and count (descending)
        duplicate_groups.sort(key=lambda x: (x['artist_name'], -x['count']))
        
        print(f"Found {len(duplicate_groups)} groups of potential duplicates containing {total_potential_duplicates} artworks")
        
        return jsonify({
            'success': True,
            'duplicate_groups': duplicate_groups,
            'total_duplicate_groups': len(duplicate_groups),
            'total_artworks': total_potential_duplicates
        })
        
    except Exception as e:
        print(f"ERROR in check_duplicate_images: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        })

@data_cleaner_bp.route('/process_duplicate_images', methods=['POST'])
def process_duplicate_images():
    """Merge selected duplicate artworks and record unselected pairs as not duplicates"""
    try:
        from index import get_db
        import os
        
        db = get_db()
        data = request.get_json()
        
        if not data or 'selected_groups' not in data:
            return jsonify({
                'success': False,
                'error': 'No selected groups provided'
            })
        
        selected_groups = data['selected_groups']
        all_groups = data.get('all_groups', [])  # We'll need this to track unselected pairs
        
        processed_groups = 0
        total_merged = 0
        total_removed = 0
        not_duplicate_pairs = []
        
        for group in selected_groups:
            items = group.get('items', [])
            if len(items) < 2:
                continue  # Need at least 2 items to merge
            
            # Find the "best" item to keep (most complete data)
            def score_completeness(item):
                score = 0
                if item.get('title') and item['title'].strip(): score += 2
                if item.get('descriptions') and item['descriptions'].strip(): score += 3
                if item.get('image_urls') and item['image_urls'].strip(): score += 2
                if item.get('wikiart_date') and item['wikiart_date'].strip(): score += 1
                if item.get('description') and item['description'].strip(): score += 3
                if item.get('related_keywords'): 
                    try:
                        keywords = json.loads(item['related_keywords'])
                        score += len(keywords) if isinstance(keywords, list) else 0
                    except:
                        pass
                # Prefer longer titles as they might be more descriptive
                if item.get('title'):
                    score += len(item['title']) * 0.01
                return score
            
            # Sort by completeness score (descending)
            items_sorted = sorted(items, key=score_completeness, reverse=True)
            keep_item = items_sorted[0]
            remove_items = items_sorted[1:]
            
            keep_id = keep_item['image_id']
            remove_ids = [item['image_id'] for item in remove_items]
            
            print(f"Merging group '{group['normalized_title']}' by {group['artist_name']}: keeping {keep_id}, removing {remove_ids}")
            
            # Merge data into the kept item
            merged_title = keep_item['title']
            merged_descriptions = keep_item.get('descriptions')
            merged_image_urls = keep_item.get('image_urls')
            merged_keywords = keep_item['related_keywords']
            
            # If the kept item is missing data, try to get it from others
            for item in remove_items:
                # Use data from other items if the kept item doesn't have it or if the other item has more complete data
                if not merged_title and item.get('title'):
                    merged_title = item['title']
                elif item.get('title') and len(item['title']) > len(merged_title or ''):
                    merged_title = item['title']  # Use longer, potentially more descriptive title
                    
                if not merged_descriptions and item.get('descriptions'):
                    merged_descriptions = item['descriptions']
                elif item.get('descriptions') and len(item['descriptions']) > len(merged_descriptions or ''):
                    merged_descriptions = item['descriptions']  # Use longer description
                    
                if not merged_image_urls and item.get('image_urls'):
                    merged_image_urls = item['image_urls']
                    
                if not merged_keywords and item.get('related_keywords'):
                    merged_keywords = item['related_keywords']
            
            # Update the kept item with merged data
            cursor = db.execute("""
                UPDATE image_entries 
                SET value = ?, descriptions = ?, image_urls = ?, relatedKeywordIds = ?
                WHERE image_id = ?
            """, (merged_title, merged_descriptions, merged_image_urls, merged_keywords, keep_id))
            
            # Update any text_entries that reference the removed images to point to the kept image
            for remove_id in remove_ids:
                # Find text entries that reference this image
                cursor = db.execute("SELECT entry_id, images FROM text_entries WHERE images LIKE ?", (f'%{remove_id}%',))
                text_entries = cursor.fetchall()
                
                for text_entry in text_entries:
                    try:
                        images_json = text_entry['images']
                        if images_json:
                            image_ids = json.loads(images_json)
                            if isinstance(image_ids, list):
                                # Replace the removed ID with the kept ID (if not already present)
                                updated_ids = []
                                for img_id in image_ids:
                                    if img_id == remove_id:
                                        if keep_id not in updated_ids:
                                            updated_ids.append(keep_id)
                                    elif img_id not in updated_ids:
                                        updated_ids.append(img_id)
                                
                                # Update the text entry
                                updated_json = json.dumps(updated_ids)
                                db.execute("UPDATE text_entries SET images = ? WHERE entry_id = ?", 
                                         (updated_json, text_entry['entry_id']))
                    except Exception as ref_error:
                        print(f"Warning: Could not update text entry {text_entry['entry_id']} reference: {ref_error}")
            
            # Remove the duplicate image entries
            for remove_id in remove_ids:
                db.execute("DELETE FROM image_entries WHERE image_id = ?", (remove_id,))
                total_removed += 1
            
            processed_groups += 1
            total_merged += len(remove_ids)
        
        # Record unselected pairs as "not duplicates" to avoid future review
        if all_groups:
            not_duplicates_file = os.path.join(os.path.dirname(__file__), 'not_duplicates.txt')
            
            # Create a set of selected group indices for quick lookup
            selected_group_indices = {group['group_index'] for group in selected_groups if 'group_index' in group}
            
            # For groups that were NOT selected for processing, record all pairs as "not duplicates"
            pairs_to_record = set()
            for group_index, group in enumerate(all_groups):
                if group_index not in selected_group_indices and len(group.get('items', [])) > 1:
                    items = group['items']
                    # Create pairs from all combinations in this unselected group
                    for i in range(len(items)):
                        for j in range(i + 1, len(items)):
                            id1, id2 = str(items[i]['image_id']), str(items[j]['image_id'])
                            # Store pairs in a consistent order (smaller ID first)
                            if id1 < id2:
                                pairs_to_record.add((id1, id2))
                            else:
                                pairs_to_record.add((id2, id1))
            
            # Write the pairs to the file
            if pairs_to_record:
                try:
                    with open(not_duplicates_file, 'a') as f:
                        for id1, id2 in pairs_to_record:
                            f.write(f"{id1},{id2}\n")
                    print(f"Recorded {len(pairs_to_record)} pairs as not duplicates")
                except Exception as e:
                    print(f"Warning: Could not write to not_duplicates.txt: {e}")
        
        db.commit()
        
        return jsonify({
            'success': True,
            'processed_groups': processed_groups,
            'total_artworks_processed': total_merged,
            'final_artworks_count': processed_groups,  # Number of unique artworks remaining after merge
            'total_removed': total_removed,
            'not_duplicate_pairs_recorded': len(pairs_to_record) if 'pairs_to_record' in locals() else 0,
            'message': f'Successfully processed {processed_groups} groups, merged {total_merged} duplicate artworks'
        })
        
    except Exception as e:
        db.rollback()
        print(f"ERROR in process_duplicate_images: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        })