import os
import sys
import json
import time
import requests
import sqlean as sqlite3
from bs4 import BeautifulSoup
from PIL import Image

BASE_URL = "https://comicbookplus.com"
IMAGE_HOST = "https://box01.comicbookplus.com"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"
}
DB_PATH = "comics.db"
IMAGES_DIR = "comic_images"
THUMBS_DIR = os.path.join(IMAGES_DIR, "thumbs")


# ---------------------------------------------------------------------------
# 0. DB INIT
# ---------------------------------------------------------------------------

def initialize_comics_db():
    print("Step: Initializing Comics Database...")

    if os.path.exists(DB_PATH):
        if input(f"'{DB_PATH}' exists. Enter 'd' to delete or Enter to skip: ").strip().lower() == 'd':
            os.remove(DB_PATH)
            print(f"Deleted: {DB_PATH}")
        else:
            print("Skipped Comics Database initialization.")
            return

    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(THUMBS_DIR, exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:

        # Pages — mirrors art history image_entries, plus book_id + page_number
        conn.execute("""
            CREATE TABLE IF NOT EXISTS image_entries (
                image_id                TEXT PRIMARY KEY,  -- eg: cbp_77488_p001
                value                   TEXT,              -- eg: "Momotaro - Page 1"
                artist_names            TEXT,              -- JSON string array of creator names
                image_urls              TEXT,              -- JSON string dict: {"full": url, "thumb": url}
                filename                TEXT,              -- eg: cbp_77488_p001.jpg
                rights                  TEXT,              -- "Public Domain / Comic Book Plus"
                descriptions            TEXT,              -- JSON dict with "comicbookplus" key
                relatedKeywordIds       TEXT,              -- JSON array: book_id + series_id + creator entry_ids
                relatedKeywordStrings   TEXT,              -- JSON array of human-readable labels
                book_id                 TEXT,              -- FK → book_entries.book_id
                page_number             INTEGER            -- 1-indexed
            )
        """)

        # Issues — new table with no art history equivalent
        conn.execute("""
            CREATE TABLE IF NOT EXISTS book_entries (
                book_id                 TEXT PRIMARY KEY,  -- eg: cbp_77488
                value                   TEXT,              -- full issue title
                series_id               TEXT,              -- FK → text_entries.entry_id
                issue_number            INTEGER,
                cover_date              TEXT,
                page_count              INTEGER,
                pages                   TEXT,              -- JSON array of image_ids in page order
                cover_image_id          TEXT,              -- image_id of page 1
                descriptions            TEXT,              -- JSON dict with "comicbookplus" key
                relatedKeywordIds       TEXT,              -- JSON array: series_id + creator entry_ids
                relatedKeywordStrings   TEXT
            )
        """)

        # Series, creators, publishers — reuses art history text_entries shape exactly
        # type values: "series" | "creator" | "publisher"
        conn.execute("""
            CREATE TABLE IF NOT EXISTS text_entries (
                entry_id                TEXT PRIMARY KEY,
                value                   TEXT,
                images                  TEXT,              -- JSON array of image_ids (all pages across all issues)
                isArtist                INTEGER,           -- 1 for creators, 0 for series/publisher
                type                    TEXT,              -- "series" | "creator" | "publisher"
                artist_aliases          TEXT,              -- JSON array (for creators, same shape as art history)
                descriptions            TEXT,              -- JSON dict with "comicbookplus" key
                relatedKeywordIds       TEXT,
                relatedKeywordStrings   TEXT
            )
        """)

    print("Comics Database initialized successfully.")


# ---------------------------------------------------------------------------
# 1. TOP LEVEL: scrape a collection page  (?cid=XXXX)
# ---------------------------------------------------------------------------

def scrape_collection(cid_url):
    """
    Entry point. Given a URL like https://comicbookplus.com/?cid=3503,
    scrapes the series metadata and then every book listed in the table.
    """
    print(f"\nStep: Scraping collection: {cid_url}")

    if not os.path.exists(DB_PATH):
        print(f"Error: '{DB_PATH}' does not exist. Please initialize the Comics Database first.")
        return

    soup = _get_soup(cid_url)
    if not soup:
        return

    with sqlite3.connect(DB_PATH) as conn:

        # --- Parse series metadata from .introtext ---
        series_id, series_name, publisher_id = _parse_and_store_series(conn, soup, cid_url)
        if not series_id:
            print("Failed to parse series metadata. Aborting.")
            return

        # --- Find all book links in .catlistings ---
        dlids = _parse_book_list(soup)
        print(f"Found {len(dlids)} books in collection.")

        for i, dlid in enumerate(dlids):
            book_url = f"{BASE_URL}/?dlid={dlid}"
            print(f"\n[{i+1}/{len(dlids)}] Scraping book dlid={dlid}")
            scrape_book(conn, book_url, series_id, series_name, publisher_id)
            time.sleep(1)  # be polite


# ---------------------------------------------------------------------------
# 2. SERIES: parse .introtext, upsert series + publisher text_entries
# ---------------------------------------------------------------------------

def _parse_and_store_series(conn, soup, cid_url):
    """
    Parses the .introtext block on a collection page.
    Upserts a series text_entry and a publisher text_entry.
    Returns (series_id, series_name, publisher_id).
    """
    introtext = soup.find("div", class_="introtext")
    if not introtext:
        print("Error: Could not find .introtext on collection page.")
        return None, None, None

    table = introtext.find("table")
    if not table:
        print("Error: Could not find metadata table in .introtext.")
        return None, None, None

    rows = table.find_all("tr")
    series_name = ""
    categories = []
    pub_history = ""
    description_text = ""

    for row in rows:
        head = row.find("td", class_="indexcardhead")
        if head:
            series_name = head.get_text(strip=True)
            continue

        cells = row.find_all("td")
        if len(cells) == 2:
            label = cells[0].get_text(strip=True)
            value = cells[1].get_text(strip=True)
            if "Categories" in label:
                categories = [a.get_text(strip=True) for a in cells[1].find_all("a")]
            elif "Publication History" in label:
                pub_history = value
        elif len(cells) == 1:
            # The long description paragraph
            text = cells[0].get_text(strip=True)
            if len(text) > 100:
                description_text = text

    # Extract CID from URL
    cid = cid_url.split("cid=")[-1].split("&")[0]
    series_id = f"cbp_series_{cid}"

    # Try to infer publisher name from description (often mentioned early)
    # We'll store it simply as a publisher entry keyed to the series
    # For now, derive publisher_id from series_id
    publisher_id = f"cbp_publisher_{cid}"
    publisher_name = series_name  # caller can refine; description often names publisher

    descriptions = {
        "comicbookplus": {
            "categories": categories,
            "publication_history": pub_history,
            "description": description_text,
            "source_url": cid_url
        }
    }

    cursor = conn.cursor()

    # Upsert series text_entry
    cursor.execute("SELECT entry_id FROM text_entries WHERE entry_id = ?", (series_id,))
    if not cursor.fetchone():
        cursor.execute("""
            INSERT INTO text_entries (
                entry_id, value, images, isArtist, type,
                artist_aliases, descriptions, relatedKeywordIds, relatedKeywordStrings
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            series_id, series_name, json.dumps([]), 0, "series",
            json.dumps([]), json.dumps(descriptions), json.dumps([]), json.dumps(categories)
        ))
        conn.commit()
        print(f"✅ Inserted series: {series_name} ({series_id})")
    else:
        print(f"☑️  Series already exists: {series_name} ({series_id})")

    # Upsert publisher text_entry (minimal — we'll enrich if we find more info)
    cursor.execute("SELECT entry_id FROM text_entries WHERE entry_id = ?", (publisher_id,))
    if not cursor.fetchone():
        cursor.execute("""
            INSERT INTO text_entries (
                entry_id, value, images, isArtist, type,
                artist_aliases, descriptions, relatedKeywordIds, relatedKeywordStrings
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            publisher_id, publisher_name, json.dumps([]), 0, "publisher",
            json.dumps([]), json.dumps({"comicbookplus": {"source_url": cid_url}}),
            json.dumps([series_id]), json.dumps([series_name])
        ))
        conn.commit()
        print(f"✅ Inserted publisher entry: {publisher_name} ({publisher_id})")

    return series_id, series_name, publisher_id


# ---------------------------------------------------------------------------
# 3. BOOK LIST: extract dlids from .catlistings table
# ---------------------------------------------------------------------------

def _parse_book_list(soup):
    """Returns list of dlid strings from the .catlistings table."""
    dlids = []
    table = soup.find("table", class_="catlistings")
    if not table:
        print("Warning: Could not find .catlistings table.")
        return dlids

    for row in table.find_all("tr", class_="overrow"):
        # onclick="mp('77488')"
        onclick = row.get("onclick", "")
        if "mp('" in onclick:
            dlid = onclick.split("mp('")[1].split("')")[0]
            dlids.append(dlid)

    return dlids


# ---------------------------------------------------------------------------
# 4. BOOK: scrape a single issue page (?dlid=XXXX)
# ---------------------------------------------------------------------------

def scrape_book(conn, book_url, series_id, series_name, publisher_id):
    """
    Scrapes one book page. Parses the .indexcard, upserts the book_entry,
    upserts creator text_entries, then scrapes every page image.
    """
    soup = _get_soup(book_url)
    if not soup:
        return

    dlid = book_url.split("dlid=")[-1].split("&")[0]
    book_id = f"cbp_{dlid}"

    cursor = conn.cursor()

    # Check if book already fully scraped
    cursor.execute("SELECT book_id, page_count, pages FROM book_entries WHERE book_id = ?", (book_id,))
    existing = cursor.fetchone()
    if existing:
        existing_pages = json.loads(existing[2]) if existing[2] else []
        if len(existing_pages) >= (existing[1] or 0):
            print(f"☑️  Book {book_id} already complete. Skipping.")
            return
        else:
            print(f"Book {book_id} exists but incomplete ({len(existing_pages)}/{existing[1]} pages). Resuming...")

    # --- Parse .indexcard ---
    meta = _parse_indexcard(soup, book_url)
    if not meta:
        print(f"Error: Could not parse index card for {book_url}")
        return

    title = meta["title"]
    issue_number = meta["issue_number"]
    cover_date = meta["cover_date"]
    page_count = meta["page_count"]
    uploaded_by = meta["uploaded_by"]
    uploaded_date = meta["uploaded_date"]
    image_hash = meta["image_hash"]

    # If called standalone (no series context), resolve series from the index card
    if series_id is None:
        if meta["series_cid"]:
            series_id = f"cbp_series_{meta['series_cid']}"
            series_name = meta["series_name"] or "Unknown Series"
            # Upsert a minimal series text_entry if it doesn't exist yet
            cursor.execute("SELECT entry_id FROM text_entries WHERE entry_id = ?", (series_id,))
            if not cursor.fetchone():
                print(f"  Series '{series_name}' not in DB yet — fetching collection page...")
                cid_url = f"{BASE_URL}/?cid={meta['series_cid']}"
                series_soup = _get_soup(cid_url)
                if series_soup:
                    _parse_and_store_series(conn, series_soup, cid_url)
                else:
                    # Fallback: insert a minimal stub so foreign keys resolve
                    cursor.execute("""
                        INSERT INTO text_entries (
                            entry_id, value, images, isArtist, type,
                            artist_aliases, descriptions, relatedKeywordIds, relatedKeywordStrings
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        series_id, series_name, "[]", 0, "series",
                        "[]", "{}", "[]", "[]"
                    ))
                    conn.commit()
                    print(f"  ⚠️  Inserted stub series entry for {series_name} ({series_id})")
            else:
                print(f"  ☑️  Series '{series_name}' already in DB ({series_id})")
        else:
            series_id = "cbp_series_uncategorized"
            series_name = "Uncategorized"
            cursor.execute("SELECT entry_id FROM text_entries WHERE entry_id = ?", (series_id,))
            if not cursor.fetchone():
                cursor.execute("""
                    INSERT INTO text_entries (
                        entry_id, value, images, isArtist, type,
                        artist_aliases, descriptions, relatedKeywordIds, relatedKeywordStrings
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (series_id, "Uncategorized", "[]", 0, "series", "[]", "{}", "[]", "[]"))
                conn.commit()

    print(f"  Title: {title} | Pages: {page_count} | Hash: {image_hash} | Series: {series_name}")

    if not image_hash:
        print(f"  Error: Could not find image hash for {book_id}. Skipping.")
        return

    # Creator metadata: CBP only exposes uploader username, not actual comic creators.
    # Provenance is stored in descriptions instead. Creator text_entries left for future
    # enrichment (eg: from GCD or manual entry).
    creator_ids, creator_names = [], []

    # --- Build relatedKeywordIds for this book ---
    book_related_ids = [series_id] + creator_ids
    book_related_strings = [series_name] + creator_names

    provenance = (
        f"Uploaded by {uploaded_by} on {uploaded_date} to Comic Book Plus. "
        f"Source URL: {book_url}"
    )
    book_descriptions = {
        "comicbookplus": {
            "cover_date": cover_date,
            "page_count": page_count,
            "provenance": provenance,
            "source_url": book_url
        }
    }

    # --- Upsert book_entry (placeholder pages list, filled as we download) ---
    cursor.execute("SELECT book_id FROM book_entries WHERE book_id = ?", (book_id,))
    if not cursor.fetchone():
        cursor.execute("""
            INSERT INTO book_entries (
                book_id, value, series_id, issue_number, cover_date,
                page_count, pages, cover_image_id,
                descriptions, relatedKeywordIds, relatedKeywordStrings
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            book_id, title, series_id, issue_number, cover_date,
            page_count, json.dumps([]), "",
            json.dumps(book_descriptions),
            json.dumps(book_related_ids),
            json.dumps(book_related_strings)
        ))
        conn.commit()
        print(f"  ✅ Inserted book_entry: {book_id}")

    # --- Scrape pages ---
    page_image_ids = []
    cover_image_id = ""

    for page_num in range(1, page_count + 1):
        image_id = _scrape_and_store_page(
            conn=conn,
            book_id=book_id,
            dlid=dlid,
            image_hash=image_hash,
            page_num=page_num,
            page_count=page_count,
            title=title,
            cover_date=cover_date,
            series_id=series_id,
            series_name=series_name,
            creator_ids=creator_ids,
            creator_names=creator_names,
            uploaded_by=uploaded_by,
            uploaded_date=uploaded_date,
            book_url=book_url,
            book_related_ids=book_related_ids,
            book_related_strings=book_related_strings
        )
        if image_id:
            page_image_ids.append(image_id)
            if page_num == 1:
                cover_image_id = image_id
        time.sleep(0.3)  # polite crawling

    # --- Update book_entry with complete pages list ---
    cursor.execute("""
        UPDATE book_entries
        SET pages = ?, cover_image_id = ?
        WHERE book_id = ?
    """, (json.dumps(page_image_ids), cover_image_id, book_id))
    conn.commit()

    # --- Update series text_entry: add all page image_ids to its images list ---
    _append_images_to_text_entry(conn, series_id, page_image_ids)

    # --- Update creator text_entries: add page image_ids ---
    for creator_id in creator_ids:
        _append_images_to_text_entry(conn, creator_id, page_image_ids)

    print(f"  ✅✅ Book complete: {book_id} ({len(page_image_ids)}/{page_count} pages stored)")


# ---------------------------------------------------------------------------
# 5. INDEXCARD: parse book metadata
# ---------------------------------------------------------------------------

def _parse_indexcard(soup, book_url):
    """
    Parses the .indexcard table on a book page.
    Returns a dict of metadata, including the image hash extracted from #maincomic.
    """
    card = soup.find("table", class_="indexcard")
    if not card:
        return None

    meta = {
        "title": "",
        "issue_number": None,
        "cover_date": "",
        "page_count": 0,
        "uploaded_by": "",
        "uploaded_date": "",
        "image_hash": "",
        "series_cid": "",       # cid extracted from the Title row link
        "series_name": ""       # series name text from the Title row link
    }

    for row in card.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 2:
            continue
        label = cells[0].get_text(strip=True)
        cell = cells[1]

        if "Title" in label or label == "":
            # h1 inside leftfloatbold colspan=2
            h1 = row.find("h1", class_="h1card")
            if h1:
                meta["title"] = h1.get_text(strip=True)
        elif label == "Title":
            # The Title row also contains the series link eg: <a href="/?cid=3653">Charles Dana Gibson</a>
            series_link = cell.find("a", href=lambda h: h and "cid=" in h)
            if series_link:
                meta["series_name"] = series_link.get_text(strip=True)
                cid_href = series_link.get("href", "")
                meta["series_cid"] = cid_href.split("cid=")[-1].split("&")[0]
        elif "Date" in label:
            time_tag = cell.find("time")
            if time_tag:
                meta["cover_date"] = time_tag.get("datetime", "").strip() or time_tag.get_text(strip=True)
            # issue number
            pos = cell.find("span", itemprop="position")
            if pos:
                try:
                    meta["issue_number"] = int(pos.get_text(strip=True))
                except ValueError:
                    pass
        elif "Uploaded" in label:
            time_tag = cell.find("time")
            if time_tag:
                meta["uploaded_date"] = time_tag.get_text(strip=True)
            contributor = cell.find("a")
            if contributor:
                meta["uploaded_by"] = contributor.get_text(strip=True)
        elif "File size" in label:
            pages_span = cell.find("span", itemprop="numberOfPages")
            if pages_span:
                try:
                    meta["page_count"] = int(pages_span.get_text(strip=True))
                except ValueError:
                    pass

    # Also try to get title from h1 directly if we missed it
    if not meta["title"]:
        h1 = soup.find("h1", class_="h1card")
        if h1:
            meta["title"] = h1.get_text(strip=True)

    # Extract image hash from #maincomic src
    # eg: https://box01.comicbookplus.com/viewer/a0/a0665c3a3c91329d3e372a642626260a/6.jpg
    maincomic = soup.find("img", id="maincomic")
    if maincomic:
        src = maincomic.get("src", "")
        # hash is the 32-char md5 segment
        parts = src.split("/viewer/")
        if len(parts) > 1:
            # parts[1] = "a0/a0665c3a3c91329d3e372a642626260a/6.jpg"
            hash_part = parts[1].split("/")
            if len(hash_part) >= 2:
                meta["image_hash"] = hash_part[1]

    return meta


# ---------------------------------------------------------------------------
# 6. PAGE: download one page image, upsert image_entry
# ---------------------------------------------------------------------------

def _scrape_and_store_page(conn, book_id, dlid, image_hash, page_num, page_count,
                            title, cover_date, series_id, series_name,
                            creator_ids, creator_names, uploaded_by, uploaded_date,
                            book_url, book_related_ids, book_related_strings):
    """
    Downloads one comic page image and upserts its image_entry row.
    Returns the image_id on success, None on failure.
    """
    image_id = f"cbp_{dlid}_p{page_num:03d}"

    cursor = conn.cursor()
    cursor.execute("SELECT image_id FROM image_entries WHERE image_id = ?", (image_id,))
    if cursor.fetchone():
        print(f"    ☑️  Page {page_num}/{page_count} already stored ({image_id})")
        return image_id

    # Comic Book Plus page numbering: 0-indexed, so page 1 = 0.jpg
    file_index = page_num - 1
    hash_prefix = image_hash[:2]
    full_url = f"{IMAGE_HOST}/viewer/{hash_prefix}/{image_hash}/{file_index}.jpg"
    thumb_url = f"{IMAGE_HOST}/viewer/{image_hash}/mediumthumb.jpg"

    # Download image
    filename = f"{image_id}.jpg"
    local_path = os.path.join(IMAGES_DIR, filename)
    thumb_filename = f"{image_id}_thumb.jpg"
    thumb_local_path = os.path.join(THUMBS_DIR, thumb_filename)

    if not os.path.exists(local_path):
        print(f"    << Downloading page {page_num}/{page_count}: {full_url} >>")
        try:
            response = requests.get(full_url, headers=HEADERS, stream=True, timeout=15)
            if response.status_code == 200:
                with open(local_path, "wb") as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)
                print(f"    ✅ Downloaded: {filename}")
            else:
                print(f"    ❌ Failed to download page {page_num} (HTTP {response.status_code})")
                return None
        except Exception as e:
            print(f"    ❌ Error downloading page {page_num}: {e}")
            return None
    else:
        print(f"    ☑️  Image already on disk: {filename}")

    # Ensure local thumbnail exists (generated from downloaded page image)
    _ensure_thumbnail(local_path, thumb_local_path)

    # Build image_entry
    image_urls = {
        "full": full_url,
        "full_local": f"/api/comics/image/{filename}",
        "thumb": f"/api/comics/image/thumbs/{thumb_filename}",
        "thumb_remote": thumb_url
    }

    provenance = (
        f"Uploaded by {uploaded_by} on {uploaded_date} to Comic Book Plus. "
        f"Source URL: {book_url}"
    )
    descriptions = {
        "comicbookplus": {
            "book_title": title,
            "cover_date": cover_date,
            "page_number": page_num,
            "page_count": page_count,
            "provenance": provenance,
            "source_url": book_url
        }
    }

    page_value = f"{title} — Page {page_num}"
    rights = "Public Domain / Comic Book Plus"

    # relatedKeywordIds: book + series + creators
    related_ids = [book_id] + book_related_ids
    related_strings = [title] + book_related_strings

    cursor.execute("""
        INSERT INTO image_entries (
            image_id, value, artist_names, image_urls, filename, rights,
            descriptions, relatedKeywordIds, relatedKeywordStrings,
            book_id, page_number
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        image_id,
        page_value,
        json.dumps(creator_names),
        json.dumps(image_urls),
        filename,
        rights,
        json.dumps(descriptions),
        json.dumps(related_ids),
        json.dumps(related_strings),
        book_id,
        page_num
    ))
    conn.commit()

    return image_id


def _ensure_thumbnail(source_path, thumb_path, max_size=(220, 220)):
    """Create/refresh a local thumbnail from the page image."""
    if not os.path.exists(source_path):
        return

    if os.path.exists(thumb_path):
        return

    try:
        os.makedirs(os.path.dirname(thumb_path), exist_ok=True)
        with Image.open(source_path) as image:
            image = image.convert("RGB")
            image.thumbnail(max_size)
            image.save(thumb_path, format="JPEG", quality=82, optimize=True)
    except Exception as e:
        print(f"    ⚠️  Failed to generate thumbnail for {source_path}: {e}")


# ---------------------------------------------------------------------------
# 7. CREATORS: upsert creator text_entries
# ---------------------------------------------------------------------------

def _upsert_creator(conn, uploaded_by, series_id, series_name):
    """
    Upserts a creator text_entry for the uploader/contributor.
    Returns ([creator_id], [creator_name]).

    Note: Comic Book Plus doesn't always expose full creator metadata —
    the contributor field is the uploader username. If you scrape more
    detailed creator info from the book page in future, extend this function.
    """
    if not uploaded_by:
        return [], []

    creator_id = f"cbp_creator_{uploaded_by.lower().replace(' ', '_')}"
    creator_name = uploaded_by

    cursor = conn.cursor()
    cursor.execute("SELECT entry_id FROM text_entries WHERE entry_id = ?", (creator_id,))
    if not cursor.fetchone():
        aliases = [{"name": creator_name, "slug": uploaded_by.lower().replace(" ", "-")}]
        cursor.execute("""
            INSERT INTO text_entries (
                entry_id, value, images, isArtist, type,
                artist_aliases, descriptions, relatedKeywordIds, relatedKeywordStrings
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            creator_id, creator_name, json.dumps([]), 1, "creator",
            json.dumps(aliases),
            json.dumps({"comicbookplus": {"role": "contributor"}}),
            json.dumps([series_id]),
            json.dumps([series_name])
        ))
        conn.commit()
        print(f"  ✅ Inserted creator: {creator_name} ({creator_id})")
    else:
        # Ensure this series is in their relatedKeywordIds
        cursor.execute("SELECT relatedKeywordIds FROM text_entries WHERE entry_id = ?", (creator_id,))
        row = cursor.fetchone()
        existing = json.loads(row[0]) if row and row[0] else []
        if series_id not in existing:
            existing.append(series_id)
            cursor.execute(
                "UPDATE text_entries SET relatedKeywordIds = ? WHERE entry_id = ?",
                (json.dumps(existing), creator_id)
            )
            conn.commit()

    return [creator_id], [creator_name]


# ---------------------------------------------------------------------------
# 8. HELPER: append image_ids to a text_entry's images list
# ---------------------------------------------------------------------------

def _append_images_to_text_entry(conn, entry_id, new_image_ids):
    """Adds new image_ids to the images JSON array of a text_entry."""
    cursor = conn.cursor()
    cursor.execute("SELECT images FROM text_entries WHERE entry_id = ?", (entry_id,))
    row = cursor.fetchone()
    if not row:
        return
    existing = json.loads(row[0]) if row[0] else []
    to_add = [iid for iid in new_image_ids if iid not in existing]
    if to_add:
        existing.extend(to_add)
        cursor.execute(
            "UPDATE text_entries SET images = ? WHERE entry_id = ?",
            (json.dumps(existing), entry_id)
        )
        conn.commit()


# ---------------------------------------------------------------------------
# 9. HELPER: fetch and parse a page with BeautifulSoup
# ---------------------------------------------------------------------------

def _get_soup(url):
    """Fetches a URL and returns a BeautifulSoup object, or None on failure."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        if response.status_code == 200:
            return BeautifulSoup(response.text, "html.parser")
        else:
            print(f"Error fetching {url}: HTTP {response.status_code}")
            return None
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    steps = [
        ("Initialize Comics Database", initialize_comics_db),
        ("Scrape a Collection", _run_scrape_collection),
    ]

    print("Welcome to the Comics Database Builder!")

    for step_name, step_fn in steps:
        while True:
            print(f"\n{step_name}?")
            user_input = input("Enter to skip | '1' to run | 'q' to exit: ").strip().lower()
            if user_input == '':
                print(f"Skipping: {step_name}")
                break
            elif user_input == '1':
                step_fn()
                break
            elif user_input == 'q':
                print("Exiting. Goodbye!")
                return
            else:
                print("Invalid input.")


def _route_url(url):
    """Routes a URL to the appropriate scraper based on whether it is a cid or dlid."""
    if "dlid=" in url:
        dlid = url.split("dlid=")[-1].split("&")[0]
        print(f"Detected book URL (dlid={dlid}). Scraping as standalone book...")
        with sqlite3.connect(DB_PATH) as conn:
            scrape_book(conn, url, series_id=None, series_name="Uncategorized", publisher_id=None)
    elif "cid=" in url:
        scrape_collection(url)
    else:
        print(f"Unrecognized URL format (expected ?cid= or ?dlid=): {url}")


def _run_scrape_collection():
    txt_path = input("Enter path to txt file with URLs (one per line, cid= or dlid=): ").strip()
    if not txt_path or not os.path.exists(txt_path):
        print(f"File not found: {txt_path}")
        return
    with open(txt_path) as f:
        urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    print(f"Found {len(urls)} URLs to scrape.")
    for i, url in enumerate(urls):
        print(f"\n[{i+1}/{len(urls)}] {url}")
        _route_url(url)


if __name__ == "__main__":
    main()