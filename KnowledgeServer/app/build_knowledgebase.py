import sqlite3
import os, sys
import pandas as pd
import csv
import logging
import requests
#using sqlean to make using extensions easy,,
import sqlean as sqlite3
# then using the sqlite vector extension... https://alexgarcia.xyz/sqlite-vec/python.html
import sqlite_vec
import json



# Grab API token for artsy. replace with new token in ~/.zshrc when this one expires (next exp date is 5/23)
xapp_token = os.getenv("XAPP_TOKEN")
if not xapp_token:
    print("Error: XAPP_TOKEN environment variable is not set.")



def initialize_knowledgebase():
    print("Step: Initializing KnowledgeBase Database...")
    db_path = "LOCALDB/knowledgebase.db"

    if os.path.exists(db_path):
        if input(f"'{db_path}' exists. Enter 'd' to delete or Enter to skip: ").strip().lower() == 'd':
            os.remove(db_path)
            print(f"Deleted: {db_path}")
        else:
            print("Skipped KnowledgeBase Database initialization.")
            return

    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS image_entries (
                image_id                TEXT PRIMARY KEY,
                value                   TEXT,
                artist_names            TEXT,  -- JSON string array
                image_urls              TEXT,  -- JSON string dictionary
                filename                TEXT,
                rights                  TEXT,
                descriptions            TEXT,  -- JSON string dictionary; includes "synth" key if present
                relatedKeywordIds       TEXT,  -- JSON string array of text_entries.entry_id
                relatedKeywordStrings   TEXT   -- JSON string array of human-readable related keywords
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS text_entries (
                entry_id                TEXT PRIMARY KEY,
                value                   TEXT,
                images                  TEXT,  -- JSON string array of image_entries.image_id
                isArtist                INTEGER,
                type                    TEXT,
                artist_aliases          TEXT,  -- JSON string array (only if isArtist == 1)
                descriptions            TEXT,  -- JSON string dictionary; includes "synth" key if present
                relatedKeywordIds       TEXT,  -- JSON string array of text_entries.entry_id
                relatedKeywordStrings   TEXT   -- JSON string array of related keyword labels
            )
        """)

    print("KnowledgeBase Database initialized successfully.")


def get_images(url="https://api.artsy.net/api/artworks", max_depth=0):
    '''
    get_images(url, max_depth)
    ├── Loops through paginated Artsy API responses
    ├── For each artwork:
    │   └── put_artwork_in_images_db(conn, artwork) // PUT THE ARTWORK IN
    │       ├── get_artists_for_artwork(artwork_id, conn) // GET THE ARTWORK'S ARTIST (And add to text db)
    │       │   └── get_related_keywords_for_artist(artist_id, artist_name, conn) // GET THE ARTIST'S KEYWORDS
    │       ├── get_related_keywords_for_artwork(artwork_id, conn) // GET THE ARTWORK'S KEYWORDS
    │       ├── check_if_valid_image_url(url)
    │       ├── (downloads image to LOCALDB/images/)
    │       └── (inserts new image into images table in knowledgebase.db)    
    '''
    print("Step: Fetching and Populating Images...")
    # Ask the user if they want to start at a particular URL
    user_input_url = input("Enter a starting URL (leave blank to use the default): ").strip()
    if user_input_url:
        url = user_input_url
        print(f"Starting at user-provided URL: {url}")
        try:
            page_number = int(input("Enter the page number to start from (leave blank for 0): ").strip() or 0)
            if page_number < 0:
                print("Page number cannot be negative. Assuming we are starting from page 0.")
                page_number = 0
        except ValueError:
            print("Invalid input for page number. Assuming we are starting from page 0.")
            page_number = 0

        depth_counter = page_number
        if max_depth != 0:
            max_depth += depth_counter
    else:
        print(f"Using default URL: {url}")
        depth_counter = 0
    

    db_path = "LOCALDB/knowledgebase.db"

    if not os.path.exists(db_path):
        print(f"Error: '{db_path}' does not exist. Please initialize the KnowledgeBase Database first.")
        return

    headers = {"X-Xapp-Token": xapp_token}  # using global variable here


    try:
        with sqlite3.connect(db_path) as conn:
            while url:
                print(f"Fetching artworks from page {depth_counter + 1}...")
                response = requests.get(url, headers=headers)
                if response.status_code == 200:
                    data = response.json()
                    artworks = data.get('_embedded', {}).get('artworks', [])

                    for artwork in artworks:
                        put_artwork_in_images_db(conn, artwork)

                    next_url = data.get("_links", {}).get("next", {}).get("href")
                    if next_url:
                        url = next_url
                        depth_counter += 1
                        if max_depth > 0 and depth_counter >= max_depth:
                            print(f"Reached depth limit of {max_depth}. Stopping pagination.")
                            break
                        else:
                            print(f"PROCESSING PAGE #{depth_counter+1} URL: {url}")
                    else:
                        url = None
                else:
                    print(f"API Connection Failed with status code {response.status_code}: {response.text}")
                    break
    except Exception as e:
        print(f"Error connecting to API: {e}")


def put_artwork_in_images_db(conn, artwork_data):
    try:
        image_id = artwork_data.get("id")
        if not image_id:
            print("Error: Artwork data does not contain an 'id'.")
            return '-1'
        print(f"|-- processing artwork data: {image_id}")

        image_rights = artwork_data.get("image_rights", "")
        if not image_rights or not any(keyword in image_rights.lower() for keyword in ["public", "cc", "domain", "open"]): # museum, national
            print(f"Invalid or missing image rights for {image_id}: Rights: {image_rights}")
            return '-1'

        cursor = conn.cursor()

        cursor.execute("SELECT image_id FROM image_entries WHERE image_id = ?", (image_id,))
        if cursor.fetchone():
            print(f"Artwork {image_id} already exists in the database.")
            return image_id

        print(f"Artwork {image_id} not found in the database. Attempting to insert...")

        value = artwork_data.get("title", "")
        

        # Get artists for the artwork! 
        artist_data = get_artists_for_artwork(image_id, conn)
        artist_names = [artist_name for artist_name, _ in artist_data]
        artist_ids = [artist_id for _, artist_id in artist_data]
        if not artist_names or len(artist_names) < 1:
            print(f"Skipping artwork {image_id} as it has no associated artists.")
            return '-1'
        print(f"Found artists for {image_id}: {artist_names}")

        descriptions = {}
        if any(key in artwork_data for key in ["date", "category", "medium", "collecting_institution", "blurb", "additional_information"]):
            descriptions["artsy"] = {
                "date": artwork_data.get("date", ""),
                "category": artwork_data.get("category", ""),
                "medium": artwork_data.get("medium", ""),
                "collecting_institution": artwork_data.get("collecting_institution", ""),
                "description": artwork_data.get("blurb", ""),
                "additional_information": artwork_data.get("additional_information", "")
            }

        image_urls = {}
        image_template = artwork_data.get("_links", {}).get("image", {}).get("href", "")
        image_versions = artwork_data.get("image_versions", [])

        if image_template and image_versions:
            for version in image_versions:
                url = image_template.replace("{image_version}", version)
                if check_if_valid_image_url(url):
                    image_urls[version] = url

        filename = ""
        for version in ["small", "square", "medium"]:
            if version in image_urls:
                local_path = f"LOCALDB/images/{image_id}_{version}.jpg"
                if os.path.exists(local_path):
                    print(f"<< Image file for this artwork already exists as {image_id}_{version}.jpg >> ")
                    filename = f"{image_id}_{version}.jpg"
                    break
                print(f" << Attempting to download {version} image for {image_id} from URL: {image_urls[version]} >> ")
                try:
                    response = requests.get(image_urls[version], stream=True, timeout=10)
                    if response.status_code == 200:
                        os.makedirs("LOCALDB/images", exist_ok=True)
                        with open(local_path, "wb") as f:
                            for chunk in response.iter_content(1024):
                                f.write(chunk)
                        filename = f"{image_id}_{version}.jpg"
                        print(f"✅ Downloaded image for {image_id} as {filename} ✅ ")
                        break
                    else:
                        print(f"Failed to download {version} image for {image_id}. Status code: {response.status_code}")
                except Exception as e:
                    print(f"Failed to download {version} image for {image_id}: {e}")
        else:
            print(f"No valid image could be downloaded for {image_id}.")
            return '-1'

        # get_related_keywords_for_artwork !
        related_keywords = get_related_keywords_for_artwork(image_id, conn)
        related_keyword_ids = artist_ids + [entry_id for entry_id, _ in related_keywords]
        if not related_keyword_ids:
            print(f"Skipping artwork {image_id} as it has no related keywords.")
            return '-1'

        artist_names_json = json.dumps(artist_names)
        image_urls_json = json.dumps(image_urls)
        descriptions_json = json.dumps(descriptions)
        related_keyword_ids_json = json.dumps(related_keyword_ids)
        related_keyword_strings_json = json.dumps([kw_str for _, kw_str in related_keywords])

        cursor.execute("""
            INSERT INTO image_entries (
                image_id, value, artist_names, image_urls, filename, rights, descriptions,
                relatedKeywordIds, relatedKeywordStrings
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            image_id,
            value,
            artist_names_json,
            image_urls_json,
            filename,
            image_rights,
            descriptions_json,
            related_keyword_ids_json,
            related_keyword_strings_json
        ))
        conn.commit()
        print(f"Inserted new artwork into the database: {image_id}")
        return image_id

    except Exception as e:
        print(f"Error processing artwork {artwork_data.get('id', 'unknown')}: {e}")
        return '-1'


def get_artists_for_artwork(artwork_id, conn):
    print(f"|----- Fetching artist names for artwork_id: {artwork_id}")
    url = f"https://api.artsy.net/api/artists?artwork_id={artwork_id}"
    headers = {"X-Xapp-Token": xapp_token}
    artist_tuples = []

    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            artists = data.get("_embedded", {}).get("artists", [])

            cursor = conn.cursor()

            for artist in artists:
                artist_id = artist.get("id")
                artist_name = artist.get("name") or ""
                
                # Check if the artist is already in the text database
                cursor.execute("SELECT images FROM text_entries WHERE entry_id = ?", (artist_id,))
                result = cursor.fetchone()

                if result:
                    # Artist exists, update the images column
                    existing_images = result[0]
                    images_list = eval(existing_images) if existing_images else []
                    if artwork_id not in images_list:
                        images_list.append(artwork_id)
                        cursor.execute(
                            "UPDATE text_entries SET images = ? WHERE entry_id = ?",
                            (str(images_list), artist_id)
                        )
                        conn.commit()
                    artist_tuples.append((artist_name, artist_id))
                else:
                    # Artist does not exist, add them to the text database
                    artist_last_name = artist_name.split()[-1] if artist_name else ""
                    artist_first_name = artist_name.split()[0] if artist_name else ""
                    artist_aliases = [
                        {
                            "name": artist_name,
                            "sortable_name": artist.get("sortable_name", ""),
                            "last": artist_last_name,
                            "first": artist_first_name,
                            "slug": artist.get("slug", "")
                        }
                    ]
                    description = {
                        "artsy": {
                            "birth": artist.get("birthday", "").strip() if artist.get("birthday") else "",
                            "death": artist.get("deathday", "").strip() if artist.get("deathday") else "",
                            "hometown": artist.get("hometown", "").strip() if artist.get("hometown") else "",
                            "location": artist.get("location", "").strip() if artist.get("location") else "",
                            "nationality": artist.get("nationality", "").strip() if artist.get("nationality") else "",
                            "gender": artist.get("gender", "").strip() if artist.get("gender") else "",
                            "description": artist.get("biography", "").strip() if artist.get("biography") else ""
                        }
                    }
                    # Fetch related keywords using the same connection
                    related_keywords = get_related_keywords_for_artist(artist_id, artist_name, conn)
                    related_keyword_ids = [entry_id for entry_id, _ in related_keywords]

                    if not related_keyword_ids:
                        print(f"Skipping artist {artist_name} (ID: {artist_id}) as it has no related keywords.")
                        continue

                    cursor.execute("""
                        INSERT INTO text_entries (
                            entry_id, value, images, isArtist, type, artist_aliases, descriptions, 
                            relatedKeywordIds, relatedKeywordStrings
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        artist_id, artist_name, json.dumps([artwork_id]), 1, "artist", 
                        json.dumps(artist_aliases), json.dumps(description), json.dumps(related_keyword_ids), ""
                    ))
                    conn.commit()
                    print(f"|----- |----- Inserted artist {artist_name} (ID: {artist_id}) into the text database.")
                    artist_tuples.append((artist_name, artist_id))
            print("✅", end="\r")
            return artist_tuples
        else:
            print(f"Failed to fetch artists for artwork_id {artwork_id}. Status code: {response.status_code}")
            return []
    except Exception as e:
        print(f"Error fetching artists for artwork_id {artwork_id}: {e}")
        return []


# Function to get related keywords for an artist and update the text_entries table
def get_related_keywords_for_artist(artist_id, artist_name, conn):
    print(f"|-----|-----  Fetching related keywords for artist_id: {artist_id}, name: {artist_name}...", end= " ")
    result_list = []
    url = f"https://api.artsy.net/api/genes?artist_id={artist_id}"
    headers = {"X-Xapp-Token": xapp_token}

    try:
        while url:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                # get the genes that artsy says is related to this artist
                genes = data.get("_embedded", {}).get("genes", [])

                cursor = conn.cursor()

                for gene in genes: # for each of those genes, get its id...
                    gene_id = gene.get("id")
                    if not gene_id:
                        continue

                    # find it in the database
                    cursor.execute("SELECT value, relatedKeywordIds FROM text_entries WHERE entry_id = ?", (gene_id,))
                    row = cursor.fetchone()

                    # if it exists, we should record that this artist_id is related to it
                    if row:
                        value, existing_ids_json = row
                        try:
                            existing_ids_list = json.loads(existing_ids_json) if existing_ids_json else []
                        except json.JSONDecodeError:
                            existing_ids_list = []

                        if artist_id not in existing_ids_list:
                            existing_ids_list.append(artist_id)
                            cursor.execute(
                                "UPDATE text_entries SET relatedKeywordIds = ? WHERE entry_id = ?",
                                (json.dumps(existing_ids_list), gene_id)
                            )
                            conn.commit()
                        # and add this gene to the list of keywords that we'll add to the artist's row in text_entries,
                        # to record that this artist is related to this gene/keyword
                        result_list.append((gene_id, value))
                    else:
                        print(f"Gene {gene_id} not found in database.")

                url = data.get("_links", {}).get("next", {}).get("href")
            else:
                print(f"Failed to fetch genes for artist_id {artist_id}. Status code: {response.status_code}")
                break

    except Exception as e:
        print(f"Error fetching related keywords for artist_id {artist_id}: {e}")
    
    return result_list


# Function to get related keywords for an artwork and update the text database
def get_related_keywords_for_artwork(artwork_id, conn):
    print(f"|----- Fetching related keywords for artwork_id: {artwork_id}")
    url = f"https://api.artsy.net/api/genes?artwork_id={artwork_id}"
    headers = {"X-Xapp-Token": xapp_token}
    result_list = []

    try:
        while url:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                genes = data.get("_embedded", {}).get("genes", [])

                cursor = conn.cursor()

                for gene in genes:
                    gene_id = gene.get("id")
                    if not gene_id:
                        continue

                    cursor.execute("SELECT value, relatedKeywordIds FROM text_entries WHERE entry_id = ?", (gene_id,))
                    row = cursor.fetchone()

                    if row:
                        value, existing_ids_json = row
                        try:
                            existing_ids_list = json.loads(existing_ids_json) if existing_ids_json else []
                        except json.JSONDecodeError:
                            existing_ids_list = []

                        if artwork_id not in existing_ids_list:
                            existing_ids_list.append(artwork_id)
                            cursor.execute(
                                "UPDATE text_entries SET relatedKeywordIds = ? WHERE entry_id = ?",
                                (json.dumps(existing_ids_list), gene_id)
                            )
                            conn.commit()

                        result_list.append((gene_id, value))

                url = data.get("_links", {}).get("next", {}).get("href")
            else:
                print(f"Failed to fetch genes for artwork_id {artwork_id}. Status code: {response.status_code}")
                break
    except Exception as e:
        print(f"Error fetching related keywords for artwork_id {artwork_id}: {e}")
    print(" ✅", end="\r")
    return result_list


# Helper function to check if an image URL is valid
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



# ----- TEXT ENTRIES -----


def populate_textdb_with_genes():
    print("Step: Populating Text Database with Genes CSV...")
    db_path = "LOCALDB/knowledgebase.db"
    csv_path = "genes_cleaned.csv"

    if not os.path.exists(db_path):
        print(f"Error: '{db_path}' does not exist. Please initialize the Text Database first.")
        return
    if not os.path.exists(csv_path):
        print(f"Error: '{csv_path}' does not exist. Please download the Genes CSV.")
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading '{csv_path}': {e}")
        return

    required_columns = ["id", "gene name", "gene family", "description"]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns in '{csv_path}': {missing_cols}")
        return

    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            for _, row in df.iterrows():
                entry_id = row["id"]
                value = row["gene name"]
                type_ = row["gene family"]
                descriptions = json.dumps({"artsy": {"description": row["description"]}})
                cursor.execute("""
                    INSERT INTO text_entries (
                        entry_id, value, images, isArtist, type, artist_aliases,
                        descriptions, relatedKeywordIds, relatedKeywordStrings
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (entry_id, value, "", 0, type_, "", descriptions, "", ""))
            conn.commit()
        print("Text Database populated from genes_cleaned.csv successfully.")
    except Exception as e:
        print(f"Error inserting text data into knowledgebase.db: {e}")


def main():
    global max_depth, row_limit
    steps = [
        ("Initialize KnowledgeBase (text and images)", initialize_knowledgebase),
        ("Populate Text Database with Genes CSV", populate_textdb_with_genes),
        ("Get Images?", get_images),
    ]

    print("Welcome to the SQLite Database Builder!")

    # Parse command-line arguments
    args = sys.argv[1:]
    if len(args) >= 2:
        try:
            max_depth = int(args[0]) if args[0] else 0
            if max_depth < 0:
                print("Depth cannot be negative. Using unlimited depth.")
                max_depth = 0
        except ValueError:
            print("Invalid input for depth. Using unlimited depth.")
            max_depth = 0

        try:
            row_limit = int(args[1]) if args[1] else 0
            if row_limit < 0:
                print("Number of rows cannot be negative. Processing all rows.")
                row_limit = 0
        except ValueError:
            print("Invalid input for rows. Processing all rows.")
            row_limit = 0
    else:
        # Prompt user if not provided via command-line
        try:
            user_input_depth = input("Enter the pagination depth (leave blank for unlimited): ").strip()
            depth = int(user_input_depth) if user_input_depth else 0
            if depth < 0:
                print("Depth cannot be negative. Using unlimited depth.")
                depth = 0
        except ValueError:
            print("Invalid input for depth. Using unlimited depth.")
            depth = 0

        try:
            user_input_row_limit = input("Enter the number of rows to process in the database (leave blank for all rows): ").strip()
            row_limit = int(user_input_row_limit) if user_input_row_limit else 0
            if row_limit < 0:
                print("Number of rows cannot be negative. Processing all rows.")
                row_limit = 0
        except ValueError:
            print("Invalid input for rows. Processing all rows.")
            row_limit = 0

    print(f"Depth set to: {depth}, Rows set to: {row_limit}")

    for step_name, step_function in steps:
        while True:
            print(f"\n{step_name}?")
            user_input = input("Enter to skip | '1' to run | 'q' to exit: ").strip().lower()
            if user_input == '':
                print(f"Skipping step: {step_name}")
                break
            elif user_input == '1':
                step_function()
                break
            elif user_input == 'q':
                print("Exiting the process. Goodbye!")
                return
            else:
                print("Invalid input. Press Enter to skip, enter '1' to run, or 'q' to exit.")

if __name__ == "__main__":
    main()