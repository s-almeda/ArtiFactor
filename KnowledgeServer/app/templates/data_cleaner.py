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
        # Parse keyword IDs
        keyword_ids = []
        invalid_ids = []
        if keyword_ids_text and keyword_ids_text != '[]':
            try:
                keyword_ids = json.loads(keyword_ids_text)
                if not isinstance(keyword_ids, list):
                    keyword_ids = []
            except json.JSONDecodeError:
                print(f"DEBUG: JSON decode error for relatedKeywordIds in {table_name} {entry_id}")
                keyword_ids = []
        
        # Check for invalid IDs
        valid_ids = []
        for kid in keyword_ids:
            if kid in valid_entry_ids:
                valid_ids.append(kid)
            else:
                invalid_ids.append(kid)
        
        # Parse keyword strings
        keyword_strings = []
        if keyword_strings_text and keyword_strings_text != '[]':
            try:
                keyword_strings = json.loads(keyword_strings_text)
                if not isinstance(keyword_strings, list):
                    keyword_strings = []
            except json.JSONDecodeError:
                print(f"DEBUG: JSON decode error for relatedKeywordStrings in {table_name} {entry_id}")
                keyword_strings = []
        
        # Build correct strings list from valid IDs
        correct_strings = []
        for vid in valid_ids:
            if vid in entry_values:
                correct_strings.append(entry_values[vid])
        
        # Check if there are issues
        has_invalid_ids = len(invalid_ids) > 0
        strings_mismatch = set(keyword_strings) != set(correct_strings)
        
        if has_invalid_ids or strings_mismatch:
            return {
                'entry_id': entry_id,
                'table': table_name,
                'value': value,
                'keyword_ids': keyword_ids,
                'valid_ids': valid_ids,
                'invalid_ids': invalid_ids,
                'current_strings': keyword_strings,
                'correct_strings': correct_strings,
                'has_invalid_ids': has_invalid_ids,
                'strings_mismatch': strings_mismatch
            }
        
        return None
        
    except Exception as e:
        print(f"ERROR in check_keyword_integrity for {entry_id}: {str(e)}")
        return {
            'entry_id': entry_id,
            'table': table_name,
            'value': value,
            'error': str(e),
            'has_invalid_ids': True,
            'strings_mismatch': True
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
            valid_ids = issue.get('valid_ids', [])
            correct_strings = issue.get('correct_strings', [])
            
            print(f"DEBUG: Fixing {table} entry {entry_id}")
            
            # Prepare the new JSON data
            new_ids_json = json.dumps(valid_ids)
            new_strings_json = json.dumps(correct_strings)
            
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