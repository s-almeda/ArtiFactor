"""
create_text_coordinates.py
=========================
This script creates 2D coordinates for text entries using their vector embeddings from the existing
vec_description_features table. The coordinates are computed using UMAP dimensionality reduction
and stored in a new 'text_coordinates' table.

The script:
1. Connects to the SQLite database and retrieves all vector embeddings
2. Uses UMAP to reduce 384-dimensional embeddings to 2D coordinates
3. Creates a coordinates table and stores the results
4. Provides functionality to update coordinates when new entries are added

Usage:
    python create_text_coordinates.py [remake]
    - remake: Optional flag to recreate the coordinates table
"""

import sqlite3
import sys
import logging
import numpy as np
import sqlean as sqlite3
import sqlite_vec

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_text_coordinates_table(remake=False, method='umap'):
    """
    Create 2D coordinates for all text entries using their vector embeddings.
    
    Args:
        remake (bool): If True, recreate the coordinates table
        method (str): Dimensionality reduction method ('umap', 'pca', or 'tsne')
    """
    try:
        # Import the appropriate library based on method
        if method == 'umap':
            import umap
        elif method == 'pca':
            from sklearn.decomposition import PCA
        elif method == 'tsne':
            from sklearn.manifold import TSNE
        else:
            raise ValueError("Method must be 'umap', 'pca', or 'tsne'")
            
    except ImportError as e:
        logging.error(f"Required library not installed: {e}")
        logging.error("Install with: pip install umap-learn scikit-learn")
        return False

    # Connect to SQLite database
    conn = sqlite3.connect('LOCALDB/knowledgebase.db')
    
    # Enable sqlite-vec extension
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    
    cursor = conn.cursor()
    
    try:
        if remake:
            # Drop table if it exists
            cursor.execute('DROP TABLE IF EXISTS text_coordinates')
            logging.info("Dropped existing text_coordinates table.")

        # Create coordinates table with separate columns for each method
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS text_coordinates (
            entry_id TEXT PRIMARY KEY,
            umap_coords TEXT,  -- JSON: {"x": 1.23, "y": 4.56}
            pca_coords TEXT,   -- JSON: {"x": 1.23, "y": 4.56}
            tsne_coords TEXT,  -- JSON: {"x": 1.23, "y": 4.56}
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (entry_id) REFERENCES text_entries (entry_id)
        )
        ''')
        logging.info("Created text_coordinates table.")

        # Check if we need to compute coordinates for this method
        column_name = f"{method}_coords"
        if not remake:
            cursor.execute(f'SELECT COUNT(*) FROM text_coordinates WHERE {column_name} IS NOT NULL')
            existing_count = cursor.fetchone()[0]
            if existing_count > 0:
                logging.info(f"Coordinates already exist for method '{method}'. Use remake=True to regenerate.")
                return True

        # Retrieve all embeddings from vec_description_features
        cursor.execute('SELECT id, embedding FROM vec_description_features')
        results = cursor.fetchall()
        
        if not results:
            logging.error("No embeddings found in vec_description_features table.")
            return False
            
        logging.info(f"Retrieved {len(results)} embeddings from database.")

        # Extract entry IDs and convert embeddings from bytes to numpy arrays
        entry_ids = []
        embeddings = []
        
        for entry_id, embedding_bytes in results:
            entry_ids.append(entry_id)
            # Convert bytes back to numpy array
            embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
            embeddings.append(embedding)
        
        embeddings_array = np.array(embeddings)
        logging.info(f"Converted embeddings to array shape: {embeddings_array.shape}")

        # Apply dimensionality reduction
        logging.info(f"Applying {method} dimensionality reduction...")
        
        if method == 'umap':
            reducer = umap.UMAP(
                n_components=2, 
                random_state=42,
                n_neighbors=15,
                min_dist=0.1,
                metric='cosine'
            )
            coords_2d = reducer.fit_transform(embeddings_array)
            
        elif method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
            coords_2d = reducer.fit_transform(embeddings_array)
            
        elif method == 'tsne':
            reducer = TSNE(
                n_components=2, 
                random_state=42,
                perplexity=min(30, len(embeddings_array)-1),
                metric='cosine'
            )
            coords_2d = reducer.fit_transform(embeddings_array)

        logging.info(f"Generated 2D coordinates with shape: {coords_2d.shape}")

        # Insert coordinates into database
        logging.info("Inserting coordinates into database...")
        
        import json
        
        # Get column name for this method
        column_name = f"{method}_coords"
        
        # Insert/update coordinates for each entry
        for i, entry_id in enumerate(entry_ids):
            coord_json = json.dumps({
                "x": float(coords_2d[i, 0]), 
                "y": float(coords_2d[i, 1])
            })
            
            # Check if entry exists
            cursor.execute('SELECT entry_id FROM text_coordinates WHERE entry_id = ?', (entry_id,))
            exists = cursor.fetchone()
            
            if exists:
                # Update existing entry
                cursor.execute(f'''
                UPDATE text_coordinates 
                SET {column_name} = ?, updated_at = CURRENT_TIMESTAMP
                WHERE entry_id = ?
                ''', (coord_json, entry_id))
            else:
                # Insert new entry
                cursor.execute(f'''
                INSERT INTO text_coordinates (entry_id, {column_name})
                VALUES (?, ?)
                ''', (entry_id, coord_json))

        # Commit changes
        conn.commit()
        logging.info(f"Successfully inserted {len(entry_ids)} coordinate pairs using {method}.")
        
        # Print some sample coordinates
        cursor.execute(f'SELECT entry_id, {column_name} FROM text_coordinates WHERE {column_name} IS NOT NULL LIMIT 5')
        samples = cursor.fetchall()
        logging.info("Sample coordinates:")
        for entry_id, coord_json in samples:
            coord_data = json.loads(coord_json)
            logging.info(f"  {entry_id}: {coord_data}")
            
        return True
        
    except Exception as e:
        logging.error(f"Error creating coordinates: {e}")
        conn.rollback()
        return False
        
    finally:
        conn.close()
        logging.info("Closed database connection.")

def create_image_coordinates_table(remake=False, method='umap'):
    """
    Create 2D coordinates for all image entries using their vector embeddings.
    
    Args:
        remake (bool): If True, recreate the coordinates table
        method (str): Dimensionality reduction method ('umap', 'pca', or 'tsne')
    """
    try:
        # Import the appropriate library based on method
        if method == 'umap':
            import umap
        elif method == 'pca':
            from sklearn.decomposition import PCA
        elif method == 'tsne':
            from sklearn.manifold import TSNE
        else:
            raise ValueError("Method must be 'umap', 'pca', or 'tsne'")
            
    except ImportError as e:
        logging.error(f"Required library not installed: {e}")
        logging.error("Install with: pip install umap-learn scikit-learn")
        return False

    # Connect to SQLite database
    conn = sqlite3.connect('LOCALDB/knowledgebase.db')
    
    # Enable sqlite-vec extension
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    
    cursor = conn.cursor()
    
    try:
        if remake:
            # Drop table if it exists
            cursor.execute('DROP TABLE IF EXISTS image_coordinates')
            logging.info("Dropped existing image_coordinates table.")

        # Create coordinates table with separate columns for each method
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS image_coordinates (
            image_id TEXT PRIMARY KEY,
            umap_coords TEXT,  -- JSON: {"x": 1.23, "y": 4.56}
            pca_coords TEXT,   -- JSON: {"x": 1.23, "y": 4.56}
            tsne_coords TEXT,  -- JSON: {"x": 1.23, "y": 4.56}
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (image_id) REFERENCES image_entries (image_id)
        )
        ''')
        logging.info("Created image_coordinates table.")

        # Check if we need to compute coordinates for this method
        column_name = f"{method}_coords"
        if not remake:
            cursor.execute(f'SELECT COUNT(*) FROM image_coordinates WHERE {column_name} IS NOT NULL')
            existing_count = cursor.fetchone()[0]
            if existing_count > 0:
                logging.info(f"Coordinates already exist for method '{method}'. Use remake=True to regenerate.")
                return True

        # Retrieve all embeddings from vec_image_features
        cursor.execute('SELECT image_id, embedding FROM vec_image_features')
        results = cursor.fetchall()
        
        if not results:
            logging.error("No embeddings found in vec_image_features table.")
            return False
            
        logging.info(f"Retrieved {len(results)} embeddings from database.")

        # Extract image IDs and convert embeddings from bytes to numpy arrays
        image_ids = []
        embeddings = []
        
        for image_id, embedding_bytes in results:
            image_ids.append(image_id)
            # Convert bytes back to numpy array (ResNet50 features are 2048-dimensional)
            embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
            embeddings.append(embedding)
        
        embeddings_array = np.array(embeddings)
        logging.info(f"Converted embeddings to array shape: {embeddings_array.shape}")

        # Apply dimensionality reduction
        logging.info(f"Applying {method} dimensionality reduction...")
        
        if method == 'umap':
            reducer = umap.UMAP(
                n_components=2, 
                random_state=42,
                n_neighbors=15,
                min_dist=0.1,
                metric='cosine'
            )
            coords_2d = reducer.fit_transform(embeddings_array)
            
        elif method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
            coords_2d = reducer.fit_transform(embeddings_array)
            
        elif method == 'tsne':
            reducer = TSNE(
                n_components=2, 
                random_state=42,
                perplexity=min(30, len(embeddings_array)-1),
                metric='cosine'
            )
            coords_2d = reducer.fit_transform(embeddings_array)

        logging.info(f"Generated 2D coordinates with shape: {coords_2d.shape}")

        # Insert coordinates into database
        logging.info("Inserting coordinates into database...")
        
        import json
        
        # Get column name for this method
        column_name = f"{method}_coords"
        
        # Insert/update coordinates for each entry
        for i, image_id in enumerate(image_ids):
            coord_json = json.dumps({
                "x": float(coords_2d[i, 0]), 
                "y": float(coords_2d[i, 1])
            })
            
            # Check if entry exists
            cursor.execute('SELECT image_id FROM image_coordinates WHERE image_id = ?', (image_id,))
            exists = cursor.fetchone()
            
            if exists:
                # Update existing entry
                cursor.execute(f'''
                UPDATE image_coordinates 
                SET {column_name} = ?, updated_at = CURRENT_TIMESTAMP
                WHERE image_id = ?
                ''', (coord_json, image_id))
            else:
                # Insert new entry
                cursor.execute(f'''
                INSERT INTO image_coordinates (image_id, {column_name})
                VALUES (?, ?)
                ''', (image_id, coord_json))

        # Commit changes
        conn.commit()
        logging.info(f"Successfully inserted {len(image_ids)} coordinate pairs using {method}.")
        
        # Print some sample coordinates
        cursor.execute(f'SELECT image_id, {column_name} FROM image_coordinates WHERE {column_name} IS NOT NULL LIMIT 5')
        samples = cursor.fetchall()
        logging.info("Sample coordinates:")
        for image_id, coord_json in samples:
            coord_data = json.loads(coord_json)
            logging.info(f"  {image_id}: {coord_data}")
            
        return True
        
    except Exception as e:
        logging.error(f"Error creating coordinates: {e}")
        conn.rollback()
        return False
        
    finally:
        conn.close()
        logging.info("Closed database connection.")


# todo: change this to "triangulate_query_text_coordinates" and add to index.py or helperfunctions.py
def add_new_entry_coordinates(entry_id, method='umap', top_k=3):
    """
    Add coordinates for a new entry based on similar existing entries.
    This should be called after a new entry is added to vec_description_features.
    
    Args:
        entry_id (str): The entry ID of the new text
        method (str): The coordinate method to use ('umap', 'pca', 'tsne')
        top_k (int): Number of neighbors to use for placement
    """
    import json
    
    conn = sqlite3.connect('LOCALDB/knowledgebase.db')
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    
    cursor = conn.cursor()
    
    try:
        # Get the embedding for the new entry
        cursor.execute('SELECT embedding FROM vec_description_features WHERE id = ?', (entry_id,))
        result = cursor.fetchone()
        if not result:
            logging.error(f"No embedding found for entry_id: {entry_id}")
            return False
            
        new_embedding = np.frombuffer(result[0], dtype=np.float32)
        
        # Find similar entries using vector similarity
        cursor.execute('''
        SELECT id, distance 
        FROM vec_description_features 
        WHERE id != ?
        ORDER BY embedding <-> ? 
        LIMIT ?
        ''', (entry_id, new_embedding.tobytes(), top_k))
        
        similar_entries = cursor.fetchall()
        
        if not similar_entries:
            logging.warning(f"No similar entries found for {entry_id}")
            return False
            
        # Get coordinates of similar entries
        similar_ids = [row[0] for row in similar_entries]
        distances = [row[1] for row in similar_entries]
        
        column_name = f"{method}_coords"
        placeholders = ','.join(['?'] * len(similar_ids))
        cursor.execute(f'''
        SELECT entry_id, {column_name}
        FROM text_coordinates 
        WHERE entry_id IN ({placeholders}) AND {column_name} IS NOT NULL
        ''', similar_ids)
        
        coord_results = cursor.fetchall()
        
        if not coord_results:
            logging.error(f"No coordinates found for similar entries using method '{method}'")
            return False
            
        # Calculate weighted average position
        coords = []
        for _, coord_json in coord_results:
            coord_data = json.loads(coord_json)
            coords.append([coord_data['x'], coord_data['y']])
        
        coords = np.array(coords)
        # Convert distances to similarities (lower distance = higher similarity)
        similarities = 1 / (1 + np.array(distances[:len(coord_results)]))
        weights = similarities / np.sum(similarities)
        
        new_x = np.average(coords[:, 0], weights=weights)
        new_y = np.average(coords[:, 1], weights=weights)
        
        # Add small random offset to avoid exact overlaps
        new_x += np.random.normal(0, 0.05)
        new_y += np.random.normal(0, 0.05)
        
        # Create coordinate JSON
        coord_json = json.dumps({"x": float(new_x), "y": float(new_y)})
        
        # Check if entry exists in coordinates table
        cursor.execute('SELECT entry_id FROM text_coordinates WHERE entry_id = ?', (entry_id,))
        exists = cursor.fetchone()
        
        if exists:
            # Update existing entry
            cursor.execute(f'''
            UPDATE text_coordinates 
            SET {column_name} = ?, updated_at = CURRENT_TIMESTAMP
            WHERE entry_id = ?
            ''', (coord_json, entry_id))
        else:
            # Insert new entry
            cursor.execute(f'''
            INSERT INTO text_coordinates (entry_id, {column_name})
            VALUES (?, ?)
            ''', (entry_id, coord_json))
        
        conn.commit()
        logging.info(f"Added {method} coordinates for {entry_id}: {{'x': {new_x:.3f}, 'y': {new_y:.3f}}}")
        return True
        
    except Exception as e:
        logging.error(f"Error adding coordinates for {entry_id}: {e}")
        return False
        
    finally:
        conn.close()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Parse command line arguments
    remake = len(sys.argv) > 1 and sys.argv[1].lower() in ['remake', 'true', '1']
    method = sys.argv[2] if len(sys.argv) > 2 else 'umap'
    
    if method not in ['umap', 'pca', 'tsne']:
        logging.error("Method must be 'umap', 'pca', or 'tsne'")
        sys.exit(1)
    
    logging.info(f"Creating text coordinates using {method} (remake={remake})")
    
    success = create_text_coordinates_table(remake=remake, method=method)
    
    if success:
        logging.info("Script completed successfully!")
        
        # Show some statistics
        coords_dict = get_coordinates_dict(method)
        logging.info(f"Total {method} coordinates created: {len(coords_dict)}")
        
    else:
        logging.error("Script failed!")
        sys.exit(1)


def create_all_coordinates(remake=False):
    """Helper function to create all three coordinate types at once"""
    methods = ['umap', 'pca', 'tsne']
    
    for method in methods:
        logging.info(f"Creating {method} coordinates...")
        success = create_text_coordinates_table(remake=remake, method=method)
        if not success:
            logging.error(f"Failed to create {method} coordinates")
            return False
    
    logging.info("All coordinate methods completed!")
    return True