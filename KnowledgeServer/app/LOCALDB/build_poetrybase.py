"""
build_poetrybase.py
===================
Builds poetry.db from the Gutenberg Poetry Corpus by Allison Parrish.
  https://github.com/aparrish/gutenberg-poetry-corpus

Corpus format: gzipped newline-delimited JSON, one line per entry:
  {"s": "The Heav'ns and all the Constellations rung,", "gid": "20"}

Book metadata comes from the Project Gutenberg catalog CSV.
  Download: https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv

Schema
------
  line_entries   — individual lines of poetry (the primary searchable unit)
  book_entries   — individual Gutenberg texts
  text_entries   — authors and subjects (mirrors art history / comics shape)

Usage:
    python3 build_poetrybase.py
"""

import os
import sys
import gzip
import json
import csv
import re
import time
import requests
import sqlean as sqlite3

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "poetry.db")

CORPUS_URL  = "http://static.decontextualize.com/gutenberg-poetry-v001.ndjson.gz"
CATALOG_URL = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv"
CORPUS_PATH  = os.path.join(SCRIPT_DIR, "gutenberg-poetry-v001.ndjson.gz")
CATALOG_PATH = os.path.join(SCRIPT_DIR, "pg_catalog.csv")


# ---------------------------------------------------------------------------
# 0. DB INIT
# ---------------------------------------------------------------------------

def initialize_poetry_db():
    print("Step: Initializing Poetry Database...")

    if os.path.exists(DB_PATH):
        if input(f"'{DB_PATH}' exists. Enter 'd' to delete or Enter to skip: ").strip().lower() == 'd':
            os.remove(DB_PATH)
            print(f"Deleted: {DB_PATH}")
        else:
            print("Skipped Poetry Database initialization.")
            return

    with sqlite3.connect(DB_PATH) as conn:

        # Individual lines of poetry — the primary searchable unit
        conn.execute("""
            CREATE TABLE IF NOT EXISTS line_entries (
                line_id                 TEXT PRIMARY KEY,  -- eg: gbp_20_l00001
                value                   TEXT,              -- the line of poetry itself
                book_id                 TEXT,              -- FK → book_entries.book_id
                line_number             INTEGER,           -- 1-indexed within book
                descriptions            TEXT,              -- JSON dict with "gutenberg" key
                relatedKeywordIds       TEXT,              -- JSON array: [book_id, author_id]
                relatedKeywordStrings   TEXT               -- JSON array: [book_title, author_name]
            )
        """)

        # Individual Gutenberg texts (one entry per gid)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS book_entries (
                book_id                 TEXT PRIMARY KEY,  -- eg: gbp_book_20
                value                   TEXT,              -- book title
                gid                     TEXT,              -- original Gutenberg ID
                author_id               TEXT,              -- FK → text_entries.entry_id (primary author)
                line_count              INTEGER,
                lines                   TEXT,              -- JSON array of line_ids in order
                descriptions            TEXT,              -- JSON dict with "gutenberg" key
                relatedKeywordIds       TEXT,              -- JSON array: [author_id, subject_ids...]
                relatedKeywordStrings   TEXT
            )
        """)

        # Authors and subjects — same shape as art history / comics text_entries
        # type values: "author" | "subject"
        conn.execute("""
            CREATE TABLE IF NOT EXISTS text_entries (
                entry_id                TEXT PRIMARY KEY,  -- eg: gbp_author_tennyson-alfred
                value                   TEXT,
                lines                   TEXT,              -- JSON array of line_ids
                isArtist                INTEGER,           -- 1 for authors, 0 for subjects
                type                    TEXT,              -- "author" | "subject"
                artist_aliases          TEXT,              -- JSON array (for authors)
                descriptions            TEXT,              -- JSON dict with "gutenberg" key
                relatedKeywordIds       TEXT,
                relatedKeywordStrings   TEXT
            )
        """)

    print("Poetry Database initialized successfully.")


# ---------------------------------------------------------------------------
# 1. CATALOG: load Gutenberg metadata CSV
# ---------------------------------------------------------------------------

def load_gutenberg_catalog(csv_path):
    """
    Parse pg_catalog.csv → dict mapping gid (str) → {title, authors, subjects}.
    Download the catalog from: https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv

    CSV columns: Text#, Type, Issued, Title, Language, Authors, Subjects, LoCC, Bookshelves
    Authors format: "Surname, First, YYYY-YYYY" entries separated by "; "
    """
    catalog = {}

    if not os.path.exists(csv_path):
        print(f"Warning: Catalog CSV not found at '{csv_path}'. Proceeding without book metadata.")
        return catalog

    print(f"Loading Gutenberg catalog from {csv_path}...")
    try:
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                gid = str(row.get("Text#", "")).strip()
                if not gid:
                    continue
                title = row.get("Title", "").strip()
                raw_authors = row.get("Authors", "").strip()
                authors = [a.strip() for a in raw_authors.split(";") if a.strip()]
                raw_subjects = row.get("Subjects", "").strip()
                subjects = [s.strip() for s in raw_subjects.split(";") if s.strip()]
                catalog[gid] = {
                    "title": title,
                    "authors": authors,
                    "subjects": subjects,
                }
    except Exception as e:
        print(f"Error reading catalog: {e}")

    print(f"Loaded {len(catalog)} catalog entries.")
    return catalog


# ---------------------------------------------------------------------------
# 2. HELPERS
# ---------------------------------------------------------------------------

def _slugify(text, max_len=64):
    """Lowercase, replace spaces/punctuation with hyphens, cap length."""
    text = text.lower().strip()
    text = re.sub(r'[^a-z0-9\s-]', '', text)
    text = re.sub(r'\s+', '-', text)
    text = re.sub(r'-+', '-', text).strip('-')
    return text[:max_len]


def _author_display_name(gutenberg_str):
    """
    Convert Gutenberg author string to display name.
    "Tennyson, Alfred Tennyson, Baron, 1809-1892" → "Alfred Tennyson Baron Tennyson"
    Simpler heuristic: strip dates, reverse "Surname, First" ordering.
    """
    # Strip trailing date ranges like ", 1809-1892" or ", 1809-"
    name_part = re.sub(r',?\s*\d{4}-?\d{0,4}\s*$', '', gutenberg_str).strip().rstrip(',').strip()
    parts = [p.strip() for p in name_part.split(',', 1)]
    if len(parts) == 2 and parts[1]:
        return f"{parts[1]} {parts[0]}".strip()
    return parts[0]


def _upsert_author(conn, author_str):
    """
    Upsert a text_entry for an author string from the Gutenberg catalog.
    Returns (author_id, display_name).
    """
    if not author_str:
        return None, None

    display_name = _author_display_name(author_str)
    slug = _slugify(display_name)
    author_id = f"gbp_author_{slug}"

    cursor = conn.cursor()
    cursor.execute("SELECT entry_id FROM text_entries WHERE entry_id = ?", (author_id,))
    if not cursor.fetchone():
        aliases = [{"name": display_name, "slug": slug, "gutenberg_string": author_str}]
        cursor.execute("""
            INSERT INTO text_entries (
                entry_id, value, lines, isArtist, type,
                artist_aliases, descriptions, relatedKeywordIds, relatedKeywordStrings
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            author_id, display_name, json.dumps([]), 1, "author",
            json.dumps(aliases),
            json.dumps({"gutenberg": {"original_string": author_str}}),
            json.dumps([]), json.dumps([])
        ))
        conn.commit()
        print(f"    ✅ Inserted author: {display_name} ({author_id})")

    return author_id, display_name


def _upsert_subject(conn, subject_str):
    """Upsert a text_entry for a Gutenberg subject tag. Returns (subject_id, subject_str)."""
    if not subject_str:
        return None, None

    slug = _slugify(subject_str)
    subject_id = f"gbp_subject_{slug}"

    cursor = conn.cursor()
    cursor.execute("SELECT entry_id FROM text_entries WHERE entry_id = ?", (subject_id,))
    if not cursor.fetchone():
        cursor.execute("""
            INSERT INTO text_entries (
                entry_id, value, lines, isArtist, type,
                artist_aliases, descriptions, relatedKeywordIds, relatedKeywordStrings
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            subject_id, subject_str, json.dumps([]), 0, "subject",
            json.dumps([]),
            json.dumps({"gutenberg": {"subject": subject_str}}),
            json.dumps([]), json.dumps([])
        ))
        conn.commit()

    return subject_id, subject_str


def _append_lines_to_text_entry(conn, entry_id, new_line_ids):
    """Append new line_ids to text_entries.lines JSON array."""
    cursor = conn.cursor()
    cursor.execute("SELECT lines FROM text_entries WHERE entry_id = ?", (entry_id,))
    row = cursor.fetchone()
    if not row:
        return
    existing = json.loads(row[0]) if row[0] else []
    to_add = [lid for lid in new_line_ids if lid not in existing]
    if to_add:
        existing.extend(to_add)
        cursor.execute(
            "UPDATE text_entries SET lines = ? WHERE entry_id = ?",
            (json.dumps(existing), entry_id)
        )
        conn.commit()


# ---------------------------------------------------------------------------
# 3. BOOK: store one Gutenberg text and its lines
# ---------------------------------------------------------------------------

def _store_book(conn, gid, raw_lines, catalog):
    """
    Insert book_entry + line_entries for one Gutenberg text.
    raw_lines: list of line strings in order (as read from corpus).
    """
    book_id = f"gbp_book_{gid}"
    cursor = conn.cursor()

    # Skip if already complete
    cursor.execute("SELECT book_id, line_count FROM book_entries WHERE book_id = ?", (book_id,))
    existing = cursor.fetchone()
    if existing:
        stored_count = existing[1] or 0
        if stored_count >= len(raw_lines):
            print(f"  ☑️  {book_id} already complete ({stored_count} lines). Skipping.")
            return
        print(f"  {book_id} incomplete ({stored_count}/{len(raw_lines)} lines). Re-inserting...")

    meta = catalog.get(gid, {})
    title = meta.get("title") or f"Project Gutenberg #{gid}"
    authors = meta.get("authors", [])
    subjects = meta.get("subjects", [])

    # Upsert author(s) — limit to first 3
    author_ids, author_names = [], []
    for author_str in authors[:3]:
        aid, aname = _upsert_author(conn, author_str)
        if aid:
            author_ids.append(aid)
            author_names.append(aname)

    primary_author_id = author_ids[0] if author_ids else None
    primary_author_name = author_names[0] if author_names else "Unknown"

    # Upsert subject text_entries — limit to first 10
    subject_ids, subject_names = [], []
    for subject_str in subjects[:10]:
        sid, sname = _upsert_subject(conn, subject_str)
        if sid:
            subject_ids.append(sid)
            subject_names.append(sname)

    related_ids = author_ids + subject_ids
    related_strings = author_names + subject_names

    book_descriptions = {
        "gutenberg": {
            "gid": gid,
            "title": title,
            "authors": authors,
            "subjects": subjects,
            "source_url": f"https://www.gutenberg.org/ebooks/{gid}"
        }
    }

    # Upsert book_entry
    cursor.execute("SELECT book_id FROM book_entries WHERE book_id = ?", (book_id,))
    if not cursor.fetchone():
        cursor.execute("""
            INSERT INTO book_entries (
                book_id, value, gid, author_id, line_count, lines,
                descriptions, relatedKeywordIds, relatedKeywordStrings
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            book_id, title, gid, primary_author_id, len(raw_lines), json.dumps([]),
            json.dumps(book_descriptions),
            json.dumps(related_ids),
            json.dumps(related_strings)
        ))
        conn.commit()
        print(f"  ✅ Book: {title} ({book_id}) [{len(raw_lines)} lines]")

    # Insert line_entries in batches
    line_ids = []
    batch = []
    BATCH_SIZE = 500

    for i, line_text in enumerate(raw_lines, start=1):
        line_id = f"gbp_{gid}_l{i:05d}"

        cursor.execute("SELECT 1 FROM line_entries WHERE line_id = ?", (line_id,))
        if cursor.fetchone():
            line_ids.append(line_id)
            continue

        descriptions = {
            "gutenberg": {
                "gid": gid,
                "book_title": title,
                "author": primary_author_name,
                "line_number": i
            }
        }
        line_related_ids = [book_id] + (author_ids[:1] if author_ids else [])
        line_related_strings = [title] + (author_names[:1] if author_names else [])

        batch.append((
            line_id,
            line_text,
            book_id,
            i,
            json.dumps(descriptions),
            json.dumps(line_related_ids),
            json.dumps(line_related_strings)
        ))
        line_ids.append(line_id)

        if len(batch) >= BATCH_SIZE:
            cursor.executemany("""
                INSERT INTO line_entries (
                    line_id, value, book_id, line_number,
                    descriptions, relatedKeywordIds, relatedKeywordStrings
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, batch)
            conn.commit()
            batch = []

    if batch:
        cursor.executemany("""
            INSERT INTO line_entries (
                line_id, value, book_id, line_number,
                descriptions, relatedKeywordIds, relatedKeywordStrings
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, batch)
        conn.commit()

    # Update book_entry with final lines list
    cursor.execute(
        "UPDATE book_entries SET lines = ?, line_count = ? WHERE book_id = ?",
        (json.dumps(line_ids), len(line_ids), book_id)
    )
    conn.commit()

    # Update author text_entries with line_ids
    for aid in author_ids:
        _append_lines_to_text_entry(conn, aid, line_ids)

    print(f"  ✅✅ Complete: {book_id} ({len(line_ids)}/{len(raw_lines)} lines stored)")


# ---------------------------------------------------------------------------
# 4. CORPUS: stream and group lines by gid, then store each book
# ---------------------------------------------------------------------------

def build_from_corpus(corpus_path, catalog_path=None):
    """
    Parse the Gutenberg Poetry corpus and populate poetry.db.

    corpus_path:  path to gutenberg-poetry-v001.ndjson.gz (or plain .ndjson)
    catalog_path: optional path to pg_catalog.csv for book/author metadata
    """
    if not os.path.exists(DB_PATH):
        print(f"Error: '{DB_PATH}' not found. Run Initialize first.")
        return

    if not os.path.exists(corpus_path):
        print(f"Error: Corpus file not found: '{corpus_path}'")
        return

    catalog = load_gutenberg_catalog(catalog_path) if catalog_path else {}

    print(f"\nStreaming corpus from {corpus_path}...")

    # Group all lines by gid in one pass
    gid_lines = {}
    total = 0

    # Detect actual file type regardless of extension
    with open(corpus_path, 'rb') as probe:
        magic = probe.read(2)
    is_gzip = (magic == b'\x1f\x8b')

    if corpus_path.endswith('.gz') and not is_gzip:
        print(f"Error: '{corpus_path}' has a .gz extension but is not a gzip file.")
        print("The download may have failed (got an HTML error page instead).")
        print("Check the file contents with: head -c 200 " + corpus_path)
        return

    opener = gzip.open if is_gzip else open
    with opener(corpus_path, 'rt', encoding='utf-8') as f:
        for raw_line in f:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                obj = json.loads(raw_line)
                gid = str(obj.get('gid', '')).strip()
                text = str(obj.get('s', '')).strip()
                if gid and text:
                    if gid not in gid_lines:
                        gid_lines[gid] = []
                    gid_lines[gid].append(text)
                    total += 1
                    if total % 100_000 == 0:
                        print(f"  Read {total:,} lines ({len(gid_lines)} books so far)...")
            except json.JSONDecodeError:
                continue

    print(f"Finished reading: {total:,} lines across {len(gid_lines)} books.")

    # Sort by gid numerically where possible
    def gid_sort_key(g):
        try:
            return int(g)
        except ValueError:
            return float('inf')

    sorted_gids = sorted(gid_lines.items(), key=lambda x: gid_sort_key(x[0]))

    with sqlite3.connect(DB_PATH) as conn:
        for i, (gid, lines) in enumerate(sorted_gids):
            print(f"\n[{i+1}/{len(sorted_gids)}] gid={gid} ({len(lines)} lines)")
            _store_book(conn, gid, lines, catalog)
            time.sleep(0.005)

    print(f"\nDone. {len(gid_lines)} books stored in {DB_PATH}.")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def _download_file(url, dest_path):
    """Download url to dest_path with a progress indicator. Skips if already present and valid."""
    filename = os.path.basename(dest_path)

    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1024:
        # Quick validity check: gzip files start with \x1f\x8b, CSVs start with text
        with open(dest_path, 'rb') as f:
            magic = f.read(2)
        if dest_path.endswith('.gz') and magic == b'\x1f\x8b':
            print(f"  ☑️  {filename} already downloaded. Skipping.")
            return True
        elif not dest_path.endswith('.gz') and magic not in (b'\x1f\x8b',):
            print(f"  ☑️  {filename} already downloaded. Skipping.")
            return True

    print(f"  Downloading {filename} from {url} ...")
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        total = int(response.headers.get('content-length', 0))
        downloaded = 0
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=65536):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  {pct:.1f}% ({downloaded // 1024 // 1024}MB / {total // 1024 // 1024}MB)", end='', flush=True)
        print(f"\r  ✅ Downloaded {filename} ({downloaded // 1024 // 1024}MB)          ")
        return True
    except Exception as e:
        print(f"\n  ❌ Failed to download {filename}: {e}")
        return False


def _run_build():
    print("\nDownloading required files...")
    corpus_ok  = _download_file(CORPUS_URL,  CORPUS_PATH)
    catalog_ok = _download_file(CATALOG_URL, CATALOG_PATH)

    if not corpus_ok:
        print("Cannot build without corpus. Aborting.")
        return

    build_from_corpus(CORPUS_PATH, CATALOG_PATH if catalog_ok else None)


def main():
    steps = [
        ("Initialize Poetry Database", initialize_poetry_db),
        ("Build from Corpus", _run_build),
    ]

    print("Welcome to the Poetry Database Builder!")
    print("Corpus: Gutenberg Poetry Corpus by Allison Parrish")
    print("  https://github.com/aparrish/gutenberg-poetry-corpus\n")

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


if __name__ == "__main__":
    main()
