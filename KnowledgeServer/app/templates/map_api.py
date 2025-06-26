# templates/map_api.py
"""
This module defines the Flask blueprint for the map API:
endpoints for turning subsets of the art history dataset --> maps, with zones (clusters)

"""

from flask import Blueprint, jsonify, request, g, render_template
import json
import os
import helperfunctions as hf
from index import get_db
from config import IMAGES_PATH
from PIL import Image
import numpy as np
from config import BASE_DIR
MAPS_DIR = os.path.join(BASE_DIR, 'generated_maps')
os.makedirs(MAPS_DIR, exist_ok=True)

# Define the blueprint
map_api_bp = Blueprint('map_api', __name__)

@map_api_bp.route('/test', methods=['GET'])
def test():
    """Test endpoint to verify blueprint is working."""
    return jsonify({
        'success': True,
        'message': 'Map API is working!'
    })

@map_api_bp.route('/map-check')
def check_page():
    """Serve the API check page."""
    return render_template('map_api_check.html')


@map_api_bp.route('/generate_prototype_map', methods=['GET'])
def generate_prototype_map():
    """
    Generate 2D map from n image entries using CLIP embeddings.
    """
    try:
        # Parse parameters
        n = int(request.args.get('n', 50))
        method = request.args.get('method', 'clip')
        use_disk = request.args.get('disk', 'true').lower() == 'true'
        debug = request.args.get('debug', 'false').lower() == 'true'
        random = request.args.get('random', 'false').lower() == 'true'
        
        def dprint(*args, **kwargs):
            if debug:
                print(*args, **kwargs)

        # Check for cached results
        dprint(f"Checking for cached results for n={n}, method={method}, use_disk={use_disk}")
        cache_filename = f"initial_map_data_n{n}.json"
        cache_path = os.path.join(MAPS_DIR, cache_filename)
        
        if os.path.exists(cache_path):
            dprint(f"Loading cached results from {cache_filename}")
            with open(cache_path, 'r') as f:
                return jsonify(json.load(f))
        
        dprint(f"\n=== Starting generate_prototype_map ===")
        dprint(f"Parameters: n={n}, method={method}, use_disk={use_disk}")
        
        db = get_db()
        
        # 1. Get n image entries
        images = fetch_images(db, n, random)
        dprint(f"Found {len(images)} images")
        
        if len(images) < 2:
            return jsonify({'success': False, 'error': 'Not enough images found'}), 400
        
        # 2. Query pre-computed embeddings
        image_ids = [row['image_id'] for row in images]
        precomputed_embeddings = query_clip_embeddings(db, image_ids)
        dprint(f"Found {len(precomputed_embeddings)} pre-computed embeddings")
        
        # 3. Process all entries (get artist info for all)
        embeddings = []
        processed_data = []
        not_found = []
        missing_embeddings = []
        
        for idx, row in enumerate(images):
            dprint(f"\n--- Processing {idx+1}/{len(images)}: {row['image_id']} ---")
            
            # Get artist info (needed for response regardless of embedding status)
            artist_names, artist_entries = get_artist_info(row, db)
            
            processed_entry = {
                'row': row,
                'artist_names': artist_names,
                'artist_entries': artist_entries
            }
            
            # Check if we have pre-computed embedding
            image_id = row['image_id']
            if image_id in precomputed_embeddings:
                dprint(f"✓ Using pre-computed embedding")
                embedding = precomputed_embeddings[image_id]
                embeddings.append(embedding)
                processed_data.append(processed_entry)
            else:
                dprint(f"⚠ Missing embedding, will compute on-the-fly")
                missing_embeddings.append((idx, processed_entry))
        
        # 4. Handle missing embeddings on-the-fly
        for idx, processed_entry in missing_embeddings:
            row = processed_entry['row']
            dprint(f"\n--- Computing missing embedding for {row['image_id']} ---")
            
            # Load image
            img = load_image_from_row(row, use_disk, dprint)
            if img is None:
                not_found.append(row['image_id'])
                continue
            
            # Generate embedding
            try:
                if method == 'clip':
                    text = hf.convert_row_to_text(row)
                    embedding = hf.extract_clip_multimodal_features(img, text)
                else:
                    embedding = hf.extract_img_features(img)
                
                # Insert into database
                insert_clip_embedding(row['image_id'], embedding, db)
                dprint(f"✓ Saved embedding to database")
                
                embeddings.append(embedding)
                processed_data.append(processed_entry)
                
            except Exception as e:
                dprint(f"✗ Error generating embedding: {e}")
                not_found.append(row['image_id'])
                continue
        
        if len(embeddings) < 2:
            return jsonify({'success': False, 'error': 'Not enough valid embeddings', 'not_found': not_found}), 400
        
        # 5. Reduce dimensions with UMAP
        dprint(f"\nRunning UMAP on {len(embeddings)} embeddings...")
        coordinates_2d = hf.reduce_to_2d_umap(np.array(embeddings))
        
        # 6. Build response
        image_points = format_image_points(processed_data, coordinates_2d)
        
        response = {
            'success': True,
            'method': method,
            'count': len(image_points),
            'imagePoints': image_points,
            'not_found': not_found,
            'source': 'disk' if use_disk else 'url',
            'precomputed_count': len(precomputed_embeddings),
            'computed_on_fly': len(missing_embeddings) - len([x for x in not_found if x in [me[1]['row']['image_id'] for me in missing_embeddings]])
        }

        # Save to cache
        with open(cache_path, 'w') as f:
            json.dump(response, f, indent=2)
        dprint(f"Saved results to {cache_filename}")
        
        return jsonify(response)
        
    except Exception as e:
        if debug:
            import traceback
            traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


def query_clip_embeddings(db, image_ids):
    """
    Query pre-computed CLIP embeddings for given image IDs.
    Returns dict mapping image_id -> embedding array.
    """
    if not image_ids:
        return {}
    
    placeholders = ','.join(['?' for _ in image_ids])
    query = f"""
        SELECT image_id, embedding 
        FROM vec_clip_features 
        WHERE image_id IN ({placeholders})
    """
    
    cursor = db.execute(query, image_ids)
    results = cursor.fetchall()
    
    embeddings = {}
    for row in results:
        # Handle binary embedding data from SQLite vector extension
        embedding_data = row['embedding']
        if isinstance(embedding_data, bytes):
            # Convert binary data to numpy array
            # This assumes the binary format matches what your vector extension uses
            embedding = np.frombuffer(embedding_data, dtype=np.float32)
        elif isinstance(embedding_data, str):
            # Fallback for JSON string format
            embedding = np.array(json.loads(embedding_data))
        else:
            # Direct array/list
            embedding = np.array(embedding_data)
        
        embeddings[row['image_id']] = embedding
    
    return embeddings


def insert_clip_embedding(image_id, embedding, db):
    """
    Insert a CLIP embedding into the vec_clip_features table.
    """
    # Convert numpy array to binary format for SQLite vector extension
    if isinstance(embedding, np.ndarray):
        embedding_binary = embedding.astype(np.float32).tobytes()
    else:
        embedding_binary = np.array(embedding, dtype=np.float32).tobytes()
    
    query = """
        INSERT OR REPLACE INTO vec_clip_features (image_id, embedding, created_at)
        VALUES (?, ?, datetime('now'))
    """
    
    db.execute(query, (image_id, embedding_binary))
    db.commit()
def fetch_images(db, n, random=False):
    """Get n random images that have descriptions."""
    if random:
        cursor = db.execute("""
            SELECT * FROM image_entries 
            WHERE descriptions IS NOT NULL
            ORDER BY RANDOM()
            LIMIT ?
        """, (n,))
    else:
        cursor = db.execute("""
            SELECT * FROM image_entries 
            WHERE descriptions IS NOT NULL
            LIMIT ?
        """, (n,))
    return cursor.fetchall()


def load_image_from_row(row, use_disk, dprint):
    """Try to load image from disk first, then URLs."""
    # Try disk
    if use_disk and row['filename']:
        try:
            from config import IMAGES_PATH
            path = os.path.join(IMAGES_PATH, row['filename'])
            if os.path.exists(path):
                return Image.open(path).convert('RGB')
        except Exception as e:
            dprint(f"Disk load failed: {e}")
    
    # Try URLs
    if row['image_urls']:
        try:
            urls = json.loads(row['image_urls'])
            for size in ['large', 'larger', 'medium', 'small']:
                if size in urls and hf.check_image_url(urls[size]):
                    return hf.url_to_image(urls[size])
        except Exception as e:
            dprint(f"URL load failed: {e}")
    
    return None


def get_artist_info(row, db):
    """Look up artist names in text_entries table."""
    artist_names = []
    artist_entries = []
    
    if row['artist_names']:
        try:
            artist_names = json.loads(row['artist_names'])
            for name in artist_names:
                matches = hf.find_exact_matches(name, db, artists_only=True)
                if matches:
                    artist_entries.append(matches[0])
        except json.JSONDecodeError:
            pass
    
    return artist_names, artist_entries


def format_image_points(processed_data, coordinates_2d):
    """Format the final image points for response."""
    points = []
    for data, coord in zip(processed_data, coordinates_2d):
        points.append({
            'entryId': data['row']['image_id'],
            'x': float(coord[0]),
            'y': float(coord[1]),
            'artworkData': {
                'image_id': data['row']['image_id'],
                'value': data['row']['value'],
                'artist_names': data['row']['artist_names'],
                'image_urls': data['row']['image_urls'],
                'filename': data['row']['filename'],
                'rights': data['row']['rights'],
                'descriptions': data['row']['descriptions'],
                'relatedKeywordIds': data['row']['relatedKeywordIds'],
                'relatedKeywordStrings': data['row']['relatedKeywordStrings']
            },
            'artistData': {
                'names': data['artist_names'],
                'entries': data['artist_entries']
            }
        })
    return points



# Error handlers
@map_api_bp.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404

@map_api_bp.errorhandler(500)
def internal_error(error):
    return jsonify({'success': False, 'error': 'Internal server error'}), 500