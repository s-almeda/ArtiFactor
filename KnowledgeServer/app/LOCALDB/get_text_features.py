"""
get_text_features.py
=====================
This script extracts text features from textual data stored in SQLite databases and saves them in a vectorized format.

The script processes text entries from the `text_entries` table in ONE sqlite database (`knowledgebase.db`) and creates two virtual tables (`vec_description_features` and `vec_value_features`) to store 
384-dimensional feature embeddings. If `remake` is True, the tables are recreated.

1. Connects to the respective SQLite database and enables the `sqlite-vec` extension.
2. Creates virtual tables (`vec_description_features` and `vec_value_features`) for storing text IDs and embeddings.
3. Loads a pre-trained SentenceTransformer model (`all-MiniLM-L6-v2`) for feature extraction.
4. Processes text data:
    - For `vec_description_features`: Combines `type`, `value`, `artist_aliases`, and `descriptions` fields.
    - For `vec_value_features`: Uses the `value` field.
5. Computes embeddings for the processed text and inserts them into the respective tables, skipping existing entries 
    if `remake` is False.
6. Logs progress, commits changes, and closes the database connection.

The script can be executed with command-line arguments to control which functions to run:
- `00`: Skip both functions.
- `01`: Run `create_value_text_vector_embeddings_text_db`.
- `10`: Run `create_description_text_vector_embeddings_text_db`.
- `11`: Run both functions.
"""



import sqlite3
import sys
import logging
import sqlean as sqlite3 #using sqlean to make using extensions easy,,
# then using the sqlite vector extension... https://alexgarcia.xyz/sqlite-vec/python.html
import sqlite_vec


def create_description_text_vector_embeddings_text_db(remake=False):
    from sentence_transformers import SentenceTransformer
    # Connect to SQLite database
    conn = sqlite3.connect('LOCALDB/knowledgebase.db')

    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    vec_version, = conn.execute("select vec_version()").fetchone()
    logging.info(f"vec_version={vec_version}")

    cursor = conn.cursor()

    if remake:
        # Drop table if it exists
        cursor.execute('DROP TABLE IF EXISTS vec_description_features')
        logging.info("Dropped existing vec_description_features table if it existed.")

    # Create virtual table with vector columns using sqlite-vec
    cursor.execute('''
    CREATE VIRTUAL TABLE IF NOT EXISTS vec_description_features USING vec0(
        id TEXT PRIMARY KEY,
        embedding float[384])
    ''')
    logging.info("Created virtual table vec_description_features.")

    # Load the model
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    logging.info("Loaded SentenceTransformer model.")

    # Retrieve entry_id, type, value, artist_aliases, and descriptions from the text_entries table
    cursor.execute('SELECT entry_id, type, value, artist_aliases, descriptions FROM text_entries')
    entries = cursor.fetchall()
    logging.info(f"Fetched {len(entries)} entries from the text_entries table.")

    for entry_id, type_, value, artist_aliases, descriptions in entries:
        if not remake:
            # Check if the entry_id is already present in the vec_description_features table
            cursor.execute('SELECT 1 FROM vec_description_features WHERE id = ?', (entry_id,))
            if cursor.fetchone():
                logging.info(f"Skipping entry_id {entry_id} as it already exists in vec_description_features table.")
                continue

        if descriptions:
            # Concatenate type, value, and artist_aliases to the front of the descriptions
            full_description = f"{type_}, {value}, {artist_aliases}, {descriptions}"
            logging.info(f"Generating features for: {full_description}")
            features_array = model.encode(full_description)
            logging.info(f"Generated features array for entry_id {entry_id}.")
            cursor.execute('''
            INSERT INTO vec_description_features (id, embedding)
            VALUES (?, ?)
            ''', (entry_id, features_array.tobytes()))
            logging.info(f"Inserted features for entry_id {entry_id} into vec_description_features table.")

    # Commit and close
    conn.commit()
    logging.info("Committed changes to the database.")
    conn.close()
    logging.info("Closed the database connection.")


def create_value_text_vector_embeddings_text_db(remake=False):
    from sentence_transformers import SentenceTransformer
    # Connect to SQLite database
    conn = sqlite3.connect('LOCALDB/knowledgebase.db')

    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    vec_version, = conn.execute("select vec_version()").fetchone()
    logging.info(f"vec_version={vec_version}")

    cursor = conn.cursor()

    if remake:
        # Drop table if it exists
        cursor.execute('DROP TABLE IF EXISTS vec_value_features')
        logging.info("Dropped existing vec_value_features table if it existed.")

    # Create virtual table with vector columns using sqlite-vec
    cursor.execute('''
    CREATE VIRTUAL TABLE IF NOT EXISTS vec_value_features USING vec0(
        id TEXT PRIMARY KEY,
        embedding float[384])
    ''')
    logging.info("Created virtual table vec_value_features.")

    # Load the model
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    logging.info("Loaded SentenceTransformer model.")

    # Retrieve entry_id and value from the text_entries table
    cursor.execute('SELECT entry_id, value FROM text_entries')
    entries = cursor.fetchall()
    logging.info(f"Fetched {len(entries)} entries from the text_entries table.")

    for entry_id, value in entries:
        if not remake:
            # Check if the entry_id is already present in the vec_value_features table
            cursor.execute('SELECT 1 FROM vec_value_features WHERE id = ?', (entry_id,))
            if cursor.fetchone():
                logging.info(f"Skipping entry_id {entry_id} as it already exists in vec_value_features table.")
                continue

        if value:
            logging.info(f"Generating features for value: {value}")
            features_array = model.encode(value)
            logging.info(f"Generated features array for entry_id {entry_id}.")
            cursor.execute('''
            INSERT INTO vec_value_features (id, embedding)
            VALUES (?, ?)
            ''', (entry_id, features_array.tobytes()))
            logging.info(f"Inserted features for entry_id {entry_id} into vec_value_features table.")

    # Commit and close
    conn.commit()
    logging.info("Committed changes to the database.")
    conn.close()
    logging.info("Closed the database connection.")

def main():
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 3:
        logging.info("Select the function to run:")
        logging.info("0: Run create_value_text_vector_embeddings_text_db")
        logging.info("1: Run create_description_text_vector_embeddings_text_db")
        logging.info("2: Run both functions")
        choice = input("Enter your choice (0, 1, or 2): ").strip()

        logging.info("Select the mode of operation:")
        logging.info("0: Remake entire table")
        logging.info("1: Update table for new values")
        mode = input("Enter your choice (0 or 1): ").strip()

        if mode not in {"0", "1"}:
            logging.error("Invalid mode choice. Please enter 0 or 1.")
            sys.exit(1)

        remake = mode == "0"

        if choice == "0":
            logging.info(f"Running create_value_text_vector_embeddings_text_db with remake={remake}...")
            create_value_text_vector_embeddings_text_db(remake=remake)
        elif choice == "1":
            logging.info(f"Running create_description_text_vector_embeddings_text_db with remake={remake}...")
            create_description_text_vector_embeddings_text_db(remake=remake)
        elif choice == "2":
            logging.info(f"Running both functions with remake={remake}...")
            create_description_text_vector_embeddings_text_db(remake=remake)
            create_value_text_vector_embeddings_text_db(remake=remake)
        else:
            logging.error("Invalid function choice. Please enter 0, 1, or 2.")
            sys.exit(1)

if __name__ == "__main__":
    main()