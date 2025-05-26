# admin.py
from flask import Blueprint, request, jsonify, render_template_string
import json
import sqlite3
import requests
import uuid

admin_bp = Blueprint('admin', __name__, url_prefix='/admin')

def check_if_valid_image_url(url):
    try:
        response = requests.head(url, timeout=5)
        if response.status_code == 200:
            return True
        else:
            print(f"Invalid URL: {url} - Status Code: {response.status_code}")
            return False
    except requests.RequestException as e:
        print(f"Error checking URL: {url} - Exception: {e}")
        return False

def check_id_exists(db, table, id_column, id_value):
    """Check if an ID already exists in the database"""
    cursor = db.execute(f'SELECT COUNT(*) FROM {table} WHERE {id_column} = ?', (id_value,))
    count = cursor.fetchone()[0]
    return count > 0


def validate_entry(data, entry_type):
    """Validate entry data before insertion"""
    errors = []

    # Check for unique ID
    if entry_type == 'text':
        if not data.get('entry_id'):
            errors.append("Entry ID is required")
        elif check_id_exists(db, 'text_entries', 'entry_id', data['entry_id']):
            errors.append(f"Entry ID '{data['entry_id']}' already exists")
    elif entry_type == 'image':
        if not data.get('image_id'):
            errors.append("Image ID is required")
        elif check_id_exists(db, 'image_entries', 'image_id', data['image_id']):
            errors.append(f"Image ID '{data['image_id']}' already exists")
    
    
    # Common validations
    if not data.get('value'):
        errors.append("Value/Title is required")
    
    # Validate JSON fields
    json_fields = {
        'text': ['images', 'artist_aliases', 'descriptions', 'relatedKeywordIds', 'relatedKeywordStrings'],
        'image': ['image_urls', 'descriptions', 'relatedKeywordIds', 'relatedKeywordStrings']
    }
    
    for field in json_fields.get(entry_type, []):
        if data.get(field):
            try:
                parsed = json.loads(data[field])
                
                # Special validation for descriptions
                if field == 'descriptions' and parsed:
                    if not isinstance(parsed, dict):
                        errors.append(f"Descriptions must be an object with source keys")
                    else:
                        for source, desc_data in parsed.items():
                            if not isinstance(desc_data, dict):
                                errors.append(f"Description for source '{source}' must be an object")
                            elif 'description' not in desc_data:
                                errors.append(f"Description for source '{source}' must have a 'description' field")
                
                # Special validation for image_urls
                if field == 'image_urls' and entry_type == 'image':
                    if not isinstance(parsed, dict):
                        errors.append("image_urls must be a JSON object")
                    else:
                        valid_sizes = ['large', 'large_rectangle', 'larger', 'medium', 'small', 'square', 'tall']
                        for size, url in parsed.items():
                            if size not in valid_sizes:
                                errors.append(f"Invalid image size: {size}. Must be one of: {', '.join(valid_sizes)}")
                            elif not check_if_valid_image_url(url):
                                errors.append(f"Invalid or unreachable image URL for size '{size}': {url}")
                
            except json.JSONDecodeError:
                errors.append(f"Invalid JSON in {field}")
    
    # Image-specific validations
    if entry_type == 'image':
        if not data.get('rights'):
            errors.append("Image rights/copyright information is required")
        
        if not data.get('image_urls') or data.get('image_urls') == '{}':
            errors.append("At least one image URL is required")
    
    return errors

@admin_bp.route('/generate_uid')
def generate_uid():
    """Generate a unique ID"""
    # Generate a UUID and take first 24 characters of the hex representation
    uid = uuid.uuid4().hex[:24]
    # Prepend with 'm'
    return jsonify({'uid': 'm' + uid})


@admin_bp.route('/')
def admin_page():
    """Admin interface with validation feedback"""
    html = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Knowledge Base Admin</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; }
            .form-section { 
                margin: 20px 0; 
                padding: 20px; 
                background: white; 
                border-radius: 8px; 
                box-shadow: 0 2px 4px rgba(0,0,0,0.1); 
            }
            .input-group {
                display: flex;
                gap: 10px;
                align-items: center;
            }
            .input-group input {
                flex: 1;
            }
            input, textarea, select { 
                width: 100%; 
                margin: 5px 0; 
                padding: 8px; 
                border: 1px solid #ddd; 
                border-radius: 4px; 
                box-sizing: border-box;
            }
            textarea { min-height: 60px; font-family: monospace; }
            button { 
                padding: 10px 20px; 
                margin: 10px 0; 
                background: #007bff; 
                color: white; 
                border: none; 
                border-radius: 4px; 
                cursor: pointer; 
            }
            button:hover { background: #0056b3; }
            .generate-btn {
                padding: 8px 15px;
                margin: 0;
                background: #28a745;
                font-size: 14px;
                white-space: nowrap;
            }
            .generate-btn:hover { background: #218838; }
            .success { color: green; padding: 10px; background: #d4edda; border-radius: 4px; }
            .error { color: red; padding: 10px; background: #f8d7da; border-radius: 4px; }
            .validation-errors { 
                background: #fff3cd; 
                border: 1px solid #ffeaa7; 
                padding: 10px; 
                margin: 10px 0; 
                border-radius: 4px; 
            }
            .validation-errors ul { margin: 5px 0; padding-left: 20px; }
            .field-hint { font-size: 12px; color: #666; margin-top: 2px; }
            h2 { color: #333; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Knowledge Base Admin</h1>
            
            <div class="form-section">
                <h2>Add Text Entry</h2>
                <form id="textForm">
                    <div class="input-group">
                        <input type="text" name="entry_id" id="text_entry_id" placeholder="Entry ID (unique identifier)" required>
                        <button type="button" class="generate-btn" onclick="generateUID('text_entry_id')">Generate UID</button>
                    </div>
                    
                    <input type="text" name="value" placeholder="Value/Name" required>
                    
                    <textarea name="images" placeholder='["image_id_1", "image_id_2"]'></textarea>
                    <div class="field-hint">JSON array of image IDs</div>
                    
                    <select name="isArtist">
                        <option value="0">Not an Artist</option>
                        <option value="1">Is an Artist</option>
                    </select>
                    
                    <input type="text" name="type" placeholder="Type (e.g., artist, movement, technique)">
                    
                    <textarea name="artist_aliases" placeholder='["Alias 1", "Alias 2"]'></textarea>
                    <div class="field-hint">JSON array (only for artists)</div>
                    
                    <textarea name="descriptions" placeholder='{"artsy": {"description": "Main description", "category": "Painting"}}'></textarea>
                    <div class="field-hint">JSON object with source keys, each containing a description field</div>
                    
                    <div class="input-group">
                        <textarea name="relatedKeywordIds" id="text_relatedKeywordIds" placeholder='["entry_id_1", "entry_id_2"]'></textarea>
                        <button type="button" class="generate-btn" onclick="findRelatedKeywords('text')">Find Related</button>
                    </div>
                    <div class="field-hint">JSON array of related entry IDs</div>

                    <div class="input-group">
                        <textarea name="relatedKeywordStrings" id="text_relatedKeywordStrings" placeholder='["keyword 1", "keyword 2"]'></textarea>
                    </div>
                   
                    <div class="field-hint">JSON array of related keyword strings (auto-filled with IDs)</div>
                    
                    <button type="submit">Add Text Entry</button>
                </form>
                <div id="textResult"></div>
            </div>
            
            <div class="form-section">
                <h2>Add Image Entry</h2>
                <form id="imageForm">
                    <div class="input-group">
                        <input type="text" name="image_id" id="image_id" placeholder="Image ID (unique identifier)" required>
                        <button type="button" class="generate-btn" onclick="generateUID('image_id')">Generate UID</button>
                    </div>
                    
                    <input type="text" name="value" placeholder="Title" required>
                    
                    <input type="text" name="artist_names" placeholder="Artist Names (comma separated)">
                    
                    <textarea name="image_urls" placeholder='{"large": "https://...", "medium": "https://..."}' required></textarea>
                    <div class="field-hint">JSON object with size keys (large, medium, small, etc.) and URL values</div>
                    
                    <input type="text" name="filename" placeholder="Filename (optional)">
                    
                    <input type="text" name="rights" placeholder="Rights/Copyright (required)" required>
                    
                    <textarea name="descriptions" placeholder='{"artsy": {"description": "Artwork description", "medium": "Oil on canvas"}}'></textarea>
                    <div class="field-hint">JSON object with source keys</div>
                    
                    <div class="input-group">
                        <textarea name="relatedKeywordIds" id="text_relatedKeywordIds" placeholder='["entry_id_1", "entry_id_2"]'></textarea>
                        <button type="button" class="generate-btn" onclick="findRelatedKeywords('text')">Find Related</button>
                    </div>
                    <div class="field-hint">JSON array of related entry IDs</div>

                    <div class="input-group">
                        <textarea name="relatedKeywordStrings" id="text_relatedKeywordStrings" placeholder='["keyword 1", "keyword 2"]'></textarea>
                    </div>

                    <div class="field-hint">JSON array of related keyword strings (auto-filled with IDs)</div>

                    <button type="submit">Add Image Entry</button>
                </form>
                <div id="imageResult"></div>
            </div>
        </div>
        
        <script>

            
            async function findRelatedKeywords(formType) {
                // Gather text from the form
                let queryText = '';
                if (formType === 'text') {
                    const value = document.querySelector('#textForm input[name="value"]').value;
                    const type = document.querySelector('#textForm input[name="type"]').value;
                    queryText = `${value} ${type}`.trim();
                } else {
                    const title = document.querySelector('#imageForm input[name="value"]').value;
                    const artists = document.querySelector('#imageForm input[name="artist_names"]').value;
                    queryText = `${title} ${artists}`.trim();
                }
                
                if (!queryText) {
                    alert('Please fill in some information first (title/value)');
                    return;
                }
                
                try {
                    // Call the lookup_text endpoint
                    const response = await fetch('http://localhost:8080/lookup_text', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({ query: queryText, top_k: 5 })
                    });
                    
                    const results = await response.json();
                    
                    // Extract IDs and values
                    const ids = results.map(r => r.entry_id);
                    const strings = results.map(r => r.value);
                    
                    // Fill the form fields
                    if (formType === 'text') {
                        document.getElementById('text_relatedKeywordIds').value = JSON.stringify(ids);
                        document.getElementById('text_relatedKeywordStrings').value = JSON.stringify(strings);
                    } else {
                        document.getElementById('image_relatedKeywordIds').value = JSON.stringify(ids);
                        document.getElementById('image_relatedKeywordStrings').value = JSON.stringify(strings);
                    }
                    
                } catch (error) {
                    alert('Error finding related keywords: ' + error.message);
                }
            }


            async function generateUID(inputId) {
                try {
                    const response = await fetch('/admin/generate_uid');
                    const data = await response.json();
                    document.getElementById(inputId).value = data.uid;
                } catch (error) {
                    alert('Error generating UID: ' + error.message);
                }
            }
            
            async function submitForm(formId, endpoint, resultId) {
                const form = document.getElementById(formId);
                const formData = new FormData(form);
                const data = Object.fromEntries(formData);
                
                // Clear empty fields to avoid sending empty strings
                Object.keys(data).forEach(key => {
                    if (!data[key] || data[key].trim() === '') {
                        delete data[key];
                    }
                });
                
                try {
                    const response = await fetch(endpoint, {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify(data)
                    });
                    const result = await response.json();
                    
                    const resultDiv = document.getElementById(resultId);
                    if (result.success) {
                        resultDiv.innerHTML = '<div class="success">✅ ' + result.message + '</div>';
                        form.reset();
                    } else if (result.validation_errors) {
                        let html = '<div class="validation-errors"><strong>Please fix the following errors:</strong><ul>';
                        result.validation_errors.forEach(error => {
                            html += '<li>' + error + '</li>';
                        });
                        html += '</ul></div>';
                        resultDiv.innerHTML = html;
                    } else {
                        resultDiv.innerHTML = '<div class="error">❌ ' + result.error + '</div>';
                    }
                } catch (error) {
                    document.getElementById(resultId).innerHTML = 
                        '<div class="error">❌ Error: ' + error.message + '</div>';
                }
            }
            
            document.getElementById('textForm').onsubmit = (e) => {
                e.preventDefault();
                submitForm('textForm', '/admin/add_text', 'textResult');
            };
            
            document.getElementById('imageForm').onsubmit = (e) => {
                e.preventDefault();
                submitForm('imageForm', '/admin/add_image', 'imageResult');
            };
        </script>
    </body>
    </html>
    '''
    return render_template_string(html)


@admin_bp.route('/add_text', methods=['POST'])
def add_text_entry():
    """Add a new text entry with validation"""
    try:
        data = request.json
        
        # Validate the entry
        validation_errors = validate_entry(data, 'text')
        if validation_errors:
            return jsonify({'success': False, 'validation_errors': validation_errors})
        
        # Get database connection from app context
        from flask import g
        db = g.db
        
        # Insert into database
        db.execute('''
            INSERT INTO text_entries 
            (entry_id, value, images, isArtist, type, artist_aliases, 
             descriptions, relatedKeywordIds, relatedKeywordStrings)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data['entry_id'],
            data['value'],
            data.get('images', '[]'),
            int(data.get('isArtist', 0)),
            data.get('type', ''),
            data.get('artist_aliases', '[]'),
            data.get('descriptions', '{}'),
            data.get('relatedKeywordIds', '[]'),
            data.get('relatedKeywordStrings', '[]')
        ))
        db.commit()
        
        return jsonify({'success': True, 'message': f'Text entry "{data["value"]}" added successfully'})
    except sqlite3.IntegrityError:
        return jsonify({'success': False, 'error': 'Entry ID already exists'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@admin_bp.route('/add_image', methods=['POST'])
def add_image_entry():
    """Add a new image entry with validation"""
    try:
        data = request.json
        
        # Validate the entry
        validation_errors = validate_entry(data, 'image')
        if validation_errors:
            return jsonify({'success': False, 'validation_errors': validation_errors})
        
        # Get database connection from app context
        from flask import g
        db = g.db
        
        # Insert into database
        db.execute('''
            INSERT INTO image_entries 
            (image_id, value, artist_names, image_urls, filename, 
             rights, descriptions, relatedKeywordIds, relatedKeywordStrings)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data['image_id'],
            data['value'],
            data.get('artist_names', ''),
            data.get('image_urls', '{}'),
            data.get('filename', ''),
            data.get('rights', ''),
            data.get('descriptions', '{}'),
            data.get('relatedKeywordIds', '[]'),
            data.get('relatedKeywordStrings', '[]')
        ))
        db.commit()
        
        return jsonify({'success': True, 'message': f'Image entry "{data["value"]}" added successfully'})
    except sqlite3.IntegrityError:
        return jsonify({'success': False, 'error': 'Image ID already exists'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})