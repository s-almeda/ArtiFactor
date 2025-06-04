# # ========== DEPRECATED !! =============
# import sqlite3
# import os, sys
# import pandas as pd
# import csv
# import logging
# import requests
# #using sqlean to make using extensions easy,,
# import sqlean as sqlite3
# # then using the sqlite vector extension... https://alexgarcia.xyz/sqlite-vec/python.html
# import sqlite_vec
# import json

# # Grab API token for artsy. replace with new token in ~/.zshrc when this one expires (next exp date is 5/23)
# xapp_token = os.getenv("XAPP_TOKEN")
# if not xapp_token:
#     print("Error: XAPP_TOKEN environment variable is not set.")



# def initialize_text_db():
#     print("Step: Initializing Text Database...")
#     db_path = "LOCALDB/text.db"

#     if os.path.exists(db_path):
#         if input(f"'{db_path}' exists. Enter 'd' to delete or Enter to skip ").strip() == 'd':
#             os.remove(db_path)
#             print(f"Deleted: {db_path}")
#         else:
#             print("Skipped Text Database initialization.")
#             return

#     with sqlite3.connect(db_path) as conn:
#         conn.execute("""
#             CREATE TABLE IF NOT EXISTS text_entries (
#             entry_id                TEXT PRIMARY KEY,
#             value                   TEXT,
#             images                  TEXT,
#             isArtist                INTEGER,
#             type                    TEXT,
#             artist_aliases          TEXT,
#             descriptions            TEXT, -- store as JSON string 
#             relatedKeywordIds       TEXT,
#             relatedKeywordStrings   TEXT
#             )
#         """)
#     print("Text Database initialized successfully.")
#     #load from the csv
#     populate_textdb_with_genes()

# def initialize_image_db():
#     print("Step: Initializing Image Database...")
#     db_path = "LOCALDB/image.db"

#     if os.path.exists(db_path):
#         if input(f"'{db_path}' exists. Enter 'd' to delete or Enter to skip ").strip() == 'd':
#             os.remove(db_path)
#             print(f"Deleted: {db_path}")
#         else:
#             print("Skipped Image Database initialization.")
#             return

#     with sqlite3.connect(db_path) as conn:
#         conn.execute("""
#             CREATE TABLE IF NOT EXISTS image_entries (
#             image_id                TEXT PRIMARY KEY,
#             value                   TEXT,
#             artist_names            TEXT, -- store as JSON string (array)
#             image_urls              TEXT, -- store as JSON string
#             filename                TEXT,
#             rights                  TEXT,
#             descriptions            TEXT, -- store as JSON string
#             relatedKeywordIds       TEXT, -- store as JSON string (array)
#             relatedKeywordStrings   TEXT, -- store as JSON string (array)
#             )
#         """)
#     print("Image Database initialized successfully.")

# # Get images and populate the Image Database
# def get_images(url="https://api.artsy.net/api/artworks"):
#     print("Step: Fetching and Populating Images...")
#     # Ask the user if they want to start at a particular URL
#     user_input_url = input("Enter a starting URL (leave blank to use the default): ").strip()
#     if user_input_url:
#         url = user_input_url
#         print(f"Starting at user-provided URL: {url}")
#     else:
#         print(f"Using default URL: {url}")
#     image_db_path = "LOCALDB/image.db"


#     if not os.path.exists(image_db_path):
#         print(f"Error: '{image_db_path}' does not exist. Please initialize the Image Database first.")
#         return

#     headers = {"X-Xapp-Token": xapp_token}
#     counter = 0

#     try:
#         while url:
#             print(f"Fetching artworks from page {counter + 1}...")
#             response = requests.get(url, headers=headers)
#             if response.status_code == 200:
#                 data = response.json()
#                 artworks = data.get('_embedded', {}).get('artworks', [])

#                 for artwork in artworks:
#                     put_artwork_in_images_db(artwork, image_db_path)

#                 # Check for the "next" link
#                 next_url = data.get("_links", {}).get("next", {}).get("href")
#                 if next_url:
#                     url = next_url
#                     counter += 1
#                     if depth > 0 and counter >= depth:
#                         print(f"Reached depth limit of {depth}. Stopping pagination.")
#                         break
#                     else:
#                         print(f"PROCESSING PAGE #{counter+1} URL: {url}")
#                 else:
#                     url = None
#             else:
#                 print(f"API Connection Failed with status code {response.status_code}: {response.text}")
#                 break
#     except Exception as e:
#         print(f"Error connecting to API: {e}")

# def fill_images_list_for_each_keyword():
#     print("Step: Filling 'images' list for each keyword in the Text Database...")
#     text_db_path = "LOCALDB/text.db"
#     image_db_path = "LOCALDB/image.db"

#     if not os.path.exists(text_db_path):
#         print(f"Error: '{text_db_path}' does not exist. Please initialize the Text Database first.")
#         return

#     if not os.path.exists(image_db_path):
#         print(f"Error: '{image_db_path}' does not exist. Please initialize the Image Database first.")
#         return

#     try:
#         with sqlite3.connect(text_db_path) as text_conn, sqlite3.connect(image_db_path) as image_conn:
#             text_cursor = text_conn.cursor()
#             image_cursor = image_conn.cursor()

#             # Fetch all entry_ids from the text database
#             text_cursor.execute("SELECT entry_id FROM text_entries")
#             text_entries = text_cursor.fetchall()

#             for entry_id, in text_entries:
#                 # Find all rows in the image database where relatedKeywordIds contains the entry_id
#                 image_cursor.execute("SELECT image_id FROM image_entries WHERE relatedKeywordIds LIKE ?", (f"%{entry_id}%",))
#                 image_ids = [row[0] for row in image_cursor.fetchall()]

#                 # Update the 'images' column in the text database for the current entry_id
#                 text_cursor.execute("UPDATE text_entries SET images = ? WHERE entry_id = ?", (str(image_ids), entry_id))
#                 print(f"Updated 'images' for entry_id {entry_id}: {image_ids}")

#             text_conn.commit()
#         print("Successfully filled 'images' list for each keyword.")
#     except Exception as e:
#         print(f"An error occurred while filling 'images' list: {e}")

# # Function to handle artwork data and insert it into the Image Database
# def put_artwork_in_images_db(artwork_data, image_db_path):
#     try:
#         # Extract artwork ID
#         image_id = artwork_data.get("id")
#         if not image_id:
#             print("Error: Artwork data does not contain an 'id'.")
#             return '-1'
#         print("|-- processing artwork data:", image_id)

#         # Check for valid image rights
#         image_rights = artwork_data.get("image_rights", "")
#         if not image_rights or not any(keyword in image_rights.lower() for keyword in ["public", "cc", "domain", "national", "open", "museum"]):
#             print(f"Invalid or missing image rights for {image_id}: Rights: {image_rights}")
#             return '-1'

#         # Connect to the Image Database
#         with sqlite3.connect(image_db_path) as conn:
#             cursor = conn.cursor()

#             # Check if the artwork is already in the database
#             cursor.execute("SELECT image_id, relatedKeywordIds FROM image_entries WHERE image_id = ?", (image_id,))
#             result = cursor.fetchone()
#             if result:
#                 # get the list of relatedkeywordids already in this row
#                 print(f"Artwork {image_id} already exists in the database.")
#                 return image_id
#             print(f"Artwork {image_id} not found in the database. Inserting...")
#             # Prepare data for insertion
#             value = artwork_data.get("title", "")
#             artist_data = get_artists_for_artwork(image_id)  # Fetch artist data as tuples (artist_name, artist_id)
#             artist_names = ", ".join([artist_name for artist_name, _ in artist_data])  # Extract artist names
#             artist_ids = [artist_id for _, artist_id in artist_data]  # Extract artist IDs for relatedKeywordIds
#             descriptions = {}


#             if any(key in artwork_data for key in ["date", "category", "medium", "collecting_institution", "blurb", "additional_information"]):
#                 descriptions["artsy"] = json.dumps({
#                     "date": artwork_data.get("date", ""),
#                     "category": artwork_data.get("category", ""),
#                     "medium": artwork_data.get("medium", ""),
#                     "collecting_institution": artwork_data.get("collecting_institution", ""),
#                     "description": artwork_data.get("blurb", ""),
#                     "additional_information": artwork_data.get("additional_information", "")
#                 })
#             # Generate image URLs
#             image_urls = {}
#             image_template = artwork_data.get("_links", {}).get("image", {}).get("href", "")
#             image_versions = artwork_data.get("image_versions", [])

#             if image_template and image_versions:
#                 for version in image_versions:
#                     url = image_template.replace("{image_version}", version)
#                     if check_if_valid_image_url(url):
#                         image_urls[version] = url

#             # Attempt to download the image
#             filename = ""
#             for version in ["small", "square", "medium"]:
#                 if version in image_urls:
#                     # before downloading, check if its already there in the LOCALDB/images folder
#                     if os.path.exists(f"LOCALDB/images/{image_id}_{version}.jpg"):
#                         print(f"File already exists as {image_id}_{version}.jpg")
#                         filename = f"{image_id}_{version}.jpg"
#                         break
#                     print(f"Attempting to download {version} image for {image_id} from URL: {image_urls[version]}")
#                     try:
#                         response = requests.get(image_urls[version], stream=True, timeout=10)
#                         if response.status_code == 200:
#                             os.makedirs("LOCALDB/images", exist_ok=True)
#                             filename = f"{image_id}_{version}.jpg"
#                             filepath = os.path.join("LOCALDB/images", filename)
#                             with open(filepath, "wb") as f:
#                                 for chunk in response.iter_content(1024):
#                                     f.write(chunk)
#                             print(f"Downloaded image for {image_id} as {filename}")
#                             break
#                         else:
#                             print(f"Failed to download {version} image for {image_id}. Status code: {response.status_code}")
#                     except Exception as e:
#                         print(f"Failed to download {version} image for {image_id}: {e}")
#             else:
#                 print(f"No valid image could be downloaded for {image_id}.")
#                 return '-1'

#             # Prepare relatedKeywordIds 
#             related_keywords = get_related_keywords_for_artwork(image_id)
#             related_keyword_ids = artist_ids + [entry_id for entry_id, _ in related_keywords]
#             print(f"Got related keywords for {image_id}: {related_keyword_ids}")
#             # Skip inserting the artwork if there are no related keywords
#             if not related_keyword_ids:
#                 print(f"Skipping artwork {image_id} as it has no related keywords.")
#                 return '-1'
#             # Insert the new artwork into the database
#             cursor.execute("""
#                 INSERT INTO image_entries (
#                     image_id, value, artist_names, image_urls, filename, rights, descriptions, 
#                     relatedKeywordIds, relatedKeywordStrings, short_description
#                 )
#                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
#             """, (
#                 image_id,
#                 value,
#                 artist_names,  # Store artist names
#                 str(image_urls),  # Store image URLs as a string
#                 filename,  # Store the downloaded filename
#                 image_rights,  # Store the image rights
#                 json.dumps(descriptions),  # Convert descriptions to JSON string
#                 str(related_keyword_ids),  # Store relatedKeywordIds as a string
#                 "",  # empty for now, will fill later
#                 ""  # short_description (blank for now)
#             ))
#             conn.commit()

#         print(f"Inserted new artwork into the database: {image_id}")
#         return image_id

#     except Exception as e:
#         print(f"An error occurred while processing artwork {artwork_data.get('id', 'unknown')}: {e}")
#         return '-1'





# # Function to get artist names for an artwork and update the text database
# def get_artists_for_artwork(artwork_id):
#     print(f"|----- Fetching artist names for artwork_id: {artwork_id}")
#     text_db_path = "LOCALDB/text.db"

#     url = f"https://api.artsy.net/api/artists?artwork_id={artwork_id}"
#     headers = {"X-Xapp-Token": xapp_token}
#     artist_tuples = []

#     try:
#         response = requests.get(url, headers=headers)
#         if response.status_code == 200:
#             data = response.json()
#             artists = data.get("_embedded", {}).get("artists", [])

#             with sqlite3.connect(text_db_path) as conn:
#                 cursor = conn.cursor()

#                 for artist in artists:
#                     artist_id = artist.get("id")
#                     artist_name = artist.get("name")
#                     # Check if the artist is already in the text database
#                     cursor.execute("SELECT images FROM text_entries WHERE entry_id = ?", (artist_id,))
#                     result = cursor.fetchone()

#                     if result:
#                         # Artist exists, update the images column
#                         existing_images = result[0]
#                         images_list = eval(existing_images) if existing_images else []
#                         if artwork_id not in images_list:
#                             images_list.append(artwork_id)
#                             cursor.execute(
#                                 "UPDATE text_entries SET images = ? WHERE entry_id = ?",
#                                 (str(images_list), artist_id)
#                             )
#                             conn.commit()
#                         artist_tuples.append((artist_name, artist_id))
#                     else:
#                         # Artist does not exist, add them to the text database
#                         artist_last_name = artist_name.split()[-1] if artist_name else ""
#                         artist_first_name = artist_name.split()[0] if artist_name else ""
#                         artist_aliases = [
#                             {"name": artist_name, "sortable_name": artist.get("sortable_name", ""), 
#                              "last": artist_last_name, "first": artist_first_name, "slug": artist.get("slug", "")}
#                         ]
#                         descriptions = {
#                             "artsy": {
#                                 "birth": artist.get("birthday", ""),
#                                 "death": artist.get("deathday", ""),
#                                 "hometown": artist.get("hometown", ""),
#                                 "location": artist.get("location", ""),
#                                 "nationality": artist.get("nationality", ""),
#                                 "gender": artist.get("gender", ""),
#                                 "description": artist.get('biography', '')
#                             }
#                         }
#                         # Fetch related keywords for the artist
#                         related_keywords = get_related_keywords_for_artist(artist_id, artist_name)
#                         related_keyword_ids = [entry_id for entry_id, _ in related_keywords]

#                         if not related_keyword_ids:
#                             print(f"Skipping artist {artist_name} (ID: {artist_id}) as it has no related keywords.")
#                             continue

#                         cursor.execute("""
#                             INSERT INTO text_entries (
#                                 entry_id, value, images, isArtist, type, artist_aliases, descriptions, 
#                                 relatedKeywordIds, relatedKeywordStrings, short_description
#                             ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
#                         """, (
#                             artist_id, artist_name, str([artwork_id]), 1, "artist", 
#                             json.dumps(artist_aliases), json.dumps(descriptions), str(related_keyword_ids), "", ""
#                         ))
#                         conn.commit()
#                         print(f"Inserted artist {artist_name} (ID: {artist_id}) into the text database.")
#                         artist_tuples.append((artist_name, artist_id))

#             return artist_tuples
#         else:
#             print(f"Failed to fetch artists for artwork_id {artwork_id}. Status code: {response.status_code}")
#             return []
#     except Exception as e:
#         print(f"Error fetching artists for artwork_id {artwork_id}: {e}")
#         return []

# # Helper function to fetch related keywords for an artist
# def get_related_keywords_for_artist(artist_id, artist_name):
#     print(f"Fetching related keywords for artist_id: {artist_id}, name: {artist_name}")
#     text_db_path = "LOCALDB/text.db"
#     url = f"https://api.artsy.net/api/genes?artist_id={artist_id}"
#     headers = {"X-Xapp-Token": xapp_token}
#     result_list = []

#     try:
#         while url:
#             response = requests.get(url, headers=headers)
#             if response.status_code == 200:
#                 data = response.json()
#                 genes = data.get("_embedded", {}).get("genes", [])
                
#                 with sqlite3.connect(text_db_path) as conn:
#                     cursor = conn.cursor()

#                     for gene in genes:
#                         #print("found:  ", gene["name"])
#                         gene_id = gene.get("id")
#                         if not gene_id:
#                             continue

#                         # Check if the gene is already in the text database
#                         cursor.execute("SELECT value, relatedKeywordIds FROM text_entries WHERE entry_id = ?", (gene_id,))
#                         result = cursor.fetchone()

#                         if result:
#                             value, existing_ids = result
#                             existing_ids_list = eval(existing_ids) if existing_ids else []

#                             # Update relatedKeywordIds if not already present
#                             if artist_id not in existing_ids_list:
#                                 existing_ids_list.append(artist_id)
#                                 cursor.execute(
#                                     "UPDATE text_entries SET relatedKeywordIds = ? WHERE entry_id = ?",
#                                     (str(existing_ids_list), gene_id)
#                                 )
#                                 conn.commit()
#                             # Add to the result list
#                             result_list.append((gene_id, value))
#                         else:
#                             print("gene not found");
#                 # Check for the "next" link
#                 url = data.get("_links", {}).get("next", {}).get("href")
#             else:
#                 print(f"Failed to fetch genes for artist_id {artist_id}. Status code: {response.status_code}")
#                 break
#     except Exception as e:
#         print(f"Error fetching related keywords for artist_id {artist_id}: {e}")

#     return result_list





# # Helper function to fetch related keywords for an artwork
# def get_related_keywords_for_artwork(artwork_id):
#     print(f"Fetching related keywords for artwork_id: {artwork_id}")
#     text_db_path = "LOCALDB/text.db"

#     url = f"https://api.artsy.net/api/genes?artwork_id={artwork_id}"
#     headers = {"X-Xapp-Token": xapp_token}
#     result_list = []

#     try:
#         while url:
#             response = requests.get(url, headers=headers)
#             if response.status_code == 200:
#                 data = response.json()
#                 genes = data.get("_embedded", {}).get("genes", [])
                
#                 with sqlite3.connect(text_db_path) as conn:
#                     cursor = conn.cursor()

#                     for gene in genes:
#                         gene_id = gene.get("id")
#                         if not gene_id:
#                             continue

#                         # Check if the gene is already in the text database
#                         cursor.execute("SELECT value, relatedKeywordIds FROM text_entries WHERE entry_id = ?", (gene_id,))
#                         result = cursor.fetchone()

#                         if result:
#                             value, existing_ids = result
#                             existing_ids_list = eval(existing_ids) if existing_ids else []

#                             # Update relatedKeywordIds if not already present
#                             if artwork_id not in existing_ids_list:
#                                 existing_ids_list.append(artwork_id)
#                                 cursor.execute(
#                                     "UPDATE text_entries SET relatedKeywordIds = ? WHERE entry_id = ?",
#                                     (str(existing_ids_list), gene_id)
#                                 )
#                                 conn.commit()
#                             # Add to the result list
#                             result_list.append((gene_id, value))
#                 # Check for the "next" link
#                 url = data.get("_links", {}).get("next", {}).get("href")
#             else:
#                 print(f"Failed to fetch genes for artwork_id {artwork_id}. Status code: {response.status_code}")
#                 break
#     except Exception as e:
#         print(f"Error fetching related keywords for artwork_id {artwork_id}: {e}")

#     return result_list


# # Helper function to check if an image URL is valid
# def check_if_valid_image_url(url):
#     try:
#         response = requests.head(url, timeout=5)
#         if response.status_code == 200:
#             return True
#         else:
#             print(f"Invalid URL: {url} - Status Code: {response.status_code}")
#             return False
#     except requests.RequestException as e:
#         print(f"Error checking URL: {url} - Exception: {e}")
#         return False


# #create the populate_text_db_with_genescsv function
# def populate_textdb_with_genes():
#     print("Step: Populating Text Database with Genes CSV...")
#     db_path = "LOCALDB/text.db"
#     csv_path = "genes_cleaned.csv"

#     if not os.path.exists(db_path):
#         print(f"Error: '{db_path}' does not exist. Please initialize the Text Database first.")
#         return
#     if not os.path.exists(csv_path):
#         print(f"Error: '{csv_path}' does not exist. Please download the Genes CSV.")
#         return
    
#     # genes_cleaned.csv has the following columns: id,slug,gene name,gene family,description,automated
#     try:
#         df = pd.read_csv(csv_path)
#     except Exception as e:
#         print(f"Error reading '{csv_path}': {e}")
#         return
#     # Ensure required columns exist
#     required_columns = ["id", "gene name", "gene family", "description"]
#     if not all(col in df.columns for col in required_columns):
#         print(f"Error: Missing required columns in '{csv_path}'. Expected columns: {required_columns}")
#         return
#     # Process and insert data into the database
#     with sqlite3.connect(db_path) as conn:
#         for _, row in df.iterrows():
#             entry_id = row["id"]
#             value = row["gene name"]
#             type_ = row["gene family"]
#             descriptions = json.dumps({"artsy": row["description"]})
#             conn.execute("""
#                 INSERT INTO text_entries (entry_id, value, images, isArtist, type, artist_aliases, descriptions, relatedKeywordIds, relatedKeywordStrings, short_description)
#                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
#             """, (entry_id, value, "", 0, type_, "", descriptions, "", "", ""))
#     print("Text Database populated from genes_cleaned.csv successfully.")

# def update_relatedkeywordids_genes():
#     print("Step: Updating 'relatedKeywordIds' for genes in the Text Database...")
#     text_db_path = "LOCALDB/text.db"
#     image_db_path = "LOCALDB/image.db"

#     if not os.path.exists(text_db_path):
#         print(f"Error: '{text_db_path}' does not exist. Please initialize the Text Database first.")
#         return

#     if not os.path.exists(image_db_path):
#         print(f"Error: '{image_db_path}' does not exist. Please initialize the Image Database first.")
#         return

#     try:
#         with sqlite3.connect(text_db_path) as text_conn, sqlite3.connect(image_db_path) as image_conn:
#             text_cursor = text_conn.cursor()
#             image_cursor = image_conn.cursor()

#             # Fetch all genes (isArtist = 0) from the text database
#             text_cursor.execute("SELECT entry_id FROM text_entries WHERE isArtist = 0")
#             genes = text_cursor.fetchall()

#             for gene_id, in genes:
#                 compiled_related_ids = []

#                 # Check other entries in the text database
#                 text_cursor.execute("SELECT entry_id, relatedKeywordIds FROM text_entries")
#                 text_entries = text_cursor.fetchall()
#                 for entry_id, related_ids in text_entries:
#                     if related_ids:
#                         related_ids_list = eval(related_ids) if isinstance(related_ids, str) else related_ids
#                         if gene_id in related_ids_list:
#                             compiled_related_ids.append(entry_id)

#                 # Check entries in the image database
#                 image_cursor.execute("SELECT image_id, relatedKeywordIds FROM image_entries")
#                 image_entries = image_cursor.fetchall()
#                 for image_id, related_ids in image_entries:
#                     if related_ids:
#                         related_ids_list = eval(related_ids) if isinstance(related_ids, str) else related_ids
#                         if gene_id in related_ids_list:
#                             compiled_related_ids.append(image_id)

#                 # Update the gene's relatedKeywordIds in the text database
#                 text_cursor.execute(
#                     "UPDATE text_entries SET relatedKeywordIds = ? WHERE entry_id = ?",
#                     (str(compiled_related_ids), gene_id)
#                 )
#                 print(f"Updated 'relatedKeywordIds' for gene {gene_id}: {compiled_related_ids}")

#             text_conn.commit()
#         print("Successfully updated 'relatedKeywordIds' for all genes.")
#     except Exception as e:
#         print(f"An error occurred while updating 'relatedKeywordIds' for genes: {e}")


# def check_whats_in_there():
#     for db_name, db_path in {"Text Database": "LOCALDB/text.db", "Image Database": "LOCALDB/image.db"}.items():

#         print(f"\nChecking {db_name}...")
#         if not os.path.exists(db_path):
#             print(f"Error: '{db_path}' does not exist.")
#             continue
#         with sqlite3.connect(db_path) as conn:
                
#             conn.enable_load_extension(True)
#             sqlite_vec.load(conn)
#             conn.enable_load_extension(False)
#             vec_version, = conn.execute("select vec_version()").fetchone()
#             logging.info(f"vec_version={vec_version}")
            
#             cursor = conn.cursor()
#             tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
#             if not tables:
#                 print("No tables found.")
#                 continue
#             for table_name, in tables:
#                 if "vec" in table_name.lower():
#                     continue
#                 print(f"\nTable: {table_name}")
#                 schema = cursor.execute(f"PRAGMA table_info({table_name});").fetchall()
#                 for column in schema:
#                     print(column)
#                 row_count = cursor.execute(f"SELECT COUNT(*) FROM {table_name};").fetchone()[0]
#                 print(f"Row count: {row_count}")
#                 if row_count:
#                     print("Sample rows:", cursor.execute(f"SELECT * FROM {table_name} ORDER BY RANDOM() LIMIT 5;").fetchall())
#         print("Database check completed.")


# def update_related_keyword_strings_text_db():
#     print("Step: Updating 'relatedKeywordStrings' in the Text Database...")
#     text_db_path = "LOCALDB/text.db"

#     if not os.path.exists(text_db_path):
#         print(f"Error: '{text_db_path}' does not exist. Please initialize the Text Database first.")
#         return

#     try:
#         with sqlite3.connect(text_db_path) as conn:
#             cursor = conn.cursor()
#             text_cursor = conn.cursor()

#             query = "SELECT entry_id, relatedKeywordIds FROM text_entries"
#             cursor.execute(query)
#             rows = cursor.fetchall()

#             for entry_id, related_ids in rows:
#                 if not related_ids:
#                     continue

#                 try:
#                     related_ids_list = eval(related_ids) if isinstance(related_ids, str) else related_ids
#                     related_strings = []

#                     for related_id in related_ids_list:
#                         text_cursor.execute("SELECT value FROM text_entries WHERE entry_id = ?", (related_id,))
#                         result = text_cursor.fetchone()
#                         if result:
#                             print(f"Found related ID {related_id} in the text database, appending corresponding string: {result[0]}.")
#                             related_strings.append(result[0])
#                         else:
#                             print(f"Related ID {related_id} not found in the text database, removing from the row")
#                             related_ids_list.remove(related_id)

#                     cursor.execute(
#                         "UPDATE text_entries SET relatedKeywordStrings = ?, relatedKeywordIds = ? WHERE entry_id = ?",
#                         (", ".join(related_strings), str(related_ids_list), entry_id)
#                     )
#                 except Exception as e:
#                     print(f"Error processing entry_id {entry_id} in text_entries: {e}")

#             conn.commit()
#         print("Updated 'relatedKeywordStrings' in the Text Database successfully.")
#     except Exception as e:
#         print(f"An error occurred while updating the Text Database: {e}")


# def update_related_keyword_strings_image_db():
#     print("Step: Updating 'relatedKeywordStrings' in the Image Database...")
#     text_db_path = "LOCALDB/text.db"
#     image_db_path = "LOCALDB/image.db"

#     if not os.path.exists(text_db_path):
#         print(f"Error: '{text_db_path}' does not exist. Please initialize the Text Database first.")
#         return

#     if not os.path.exists(image_db_path):
#         print(f"Error: '{image_db_path}' does not exist. Please initialize the Image Database first.")
#         return

#     try:
#         with sqlite3.connect(text_db_path) as text_conn, sqlite3.connect(image_db_path) as image_conn:
#             text_cursor = text_conn.cursor()
#             cursor = image_conn.cursor()

#             query = "SELECT image_id, relatedKeywordIds FROM image_entries"
#             cursor.execute(query)
#             rows = cursor.fetchall()

#             for image_id, related_ids in rows:
#                 if not related_ids:
#                     continue

#                 try:
#                     related_ids_list = eval(related_ids) if isinstance(related_ids, str) else related_ids
#                     related_strings = []

#                     for related_id in related_ids_list:
#                         text_cursor.execute("SELECT value FROM text_entries WHERE entry_id = ?", (related_id,))
#                         result = text_cursor.fetchone()
#                         if result:
#                             print(f"Found related ID {related_id} in the text database, appending corresponding string: {result[0]}.")
#                             related_strings.append(result[0])
#                         else:
#                             print(f"Related ID {related_id} not found in the text database, removing from the row")
#                             related_ids_list.remove(related_id)

#                     cursor.execute(
#                         "UPDATE image_entries SET relatedKeywordStrings = ?, relatedKeywordIds = ? WHERE image_id = ?",
#                         (", ".join(related_strings), str(related_ids_list), image_id)
#                     )
#                 except Exception as e:
#                     print(f"Error processing image_id {image_id} in image_entries: {e}")

#             image_conn.commit()
#         print("Updated 'relatedKeywordStrings' in the Image Database successfully.")
#     except Exception as e:
#         print(f"An error occurred while updating the Image Database: {e}")


# def main():
#     global depth, row_limit
#     steps = [
#         ("Initialize Text Database", initialize_text_db),
#         ("Initialize Image Database", initialize_image_db),
#         #("Populate Text Database with Genes CSV", populate_textdb_with_genes),  
#         ("Get Images?", get_images),
#         ("Fill 'images' list for each keyword", fill_images_list_for_each_keyword),

#         ("Update 'relatedKeywordIds' for genes", update_relatedkeywordids_genes),
#         ("Update 'relatedKeywordStrings' in Text Database", update_related_keyword_strings_text_db),
#         ("Update 'relatedKeywordStrings' in Image Database", update_related_keyword_strings_image_db),
#         ("Check Database Contents", check_whats_in_there)
#     ]

#     print("Welcome to the SQLite Database Builder!")

#     # Parse command-line arguments
#     args = sys.argv[1:]
#     if len(args) >= 2:
#         try:
#             depth = int(args[0]) if args[0] else 0
#             if depth < 0:
#                 print("Depth cannot be negative. Using unlimited depth.")
#                 depth = 0
#         except ValueError:
#             print("Invalid input for depth. Using unlimited depth.")
#             depth = 0

#         try:
#             row_limit = int(args[1]) if args[1] else 0
#             if row_limit < 0:
#                 print("Number of rows cannot be negative. Processing all rows.")
#                 row_limit = 0
#         except ValueError:
#             print("Invalid input for rows. Processing all rows.")
#             row_limit = 0
#     else:
#         # Ask the user for the depth value if not provided
#         try:
#             user_input_depth = input("Enter the pagination depth (leave blank for unlimited): ").strip()
#             depth = int(user_input_depth) if user_input_depth else 0
#             if depth < 0:
#                 print("Depth cannot be negative. Using unlimited depth.")
#                 depth = 0
#         except ValueError:
#             print("Invalid input for depth. Using unlimited depth.")
#             depth = 0

#         try:
#             user_input_row_limit = input("Enter the number of rows to process in the database (leave blank for all rows): ").strip()
#             row_limit = int(user_input_row_limit) if user_input_row_limit else 0
#             if row_limit < 0:
#                 print("Number of rows cannot be negative. Processing all rows.")
#                 row_limit = 0
#         except ValueError:
#             print("Invalid input for rows. Processing all rows.")
#             row_limit = 0

#     print(f"Depth set to: {depth}, Rows set to: {row_limit}")

#     for step_name, step_function in steps:
#         while True:
#             print(f"\n{step_name}?")
#             user_input = input("Enter to Run | '1' to skip | 'q' to exit: ").strip().lower()
#             if user_input == '':
#                 step_function()
#                 break
#             elif user_input == '1':
#                 print(f"Skipping step: {step_name}")
#                 break
#             elif user_input == 'q':
#                 print("Exiting the process. Goodbye!")
#                 return
#             else:
#                 print("Invalid input. Press Enter to run, enter '1' to skip, or 'quit' to exit.")

# if __name__ == "__main__":
#     main()