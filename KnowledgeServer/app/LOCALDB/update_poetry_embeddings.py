"""
update_poetry_embeddings.py
===========================
Adds text embeddings to poetry.db.

  text_entries  (authors, subjects)
    -> vec_value_features   float[384]  MiniLM on type + value

  book_entries  (individual Gutenberg texts)
    -> vec_book_features    float[384]  MiniLM on title + author + subjects

  line_entries  (individual poetry lines)
    -> vec_line_features    float[384]  MiniLM on the line text itself

NOTE: The full Gutenberg Poetry corpus contains ~3 million lines.
  Embedding all lines takes several hours on CPU or 30-60 min on GPU.
  Use --tables text,book first to index metadata quickly, then add
  --tables line when ready for the full embedding pass.

Usage:
    python3 update_poetry_embeddings.py                       # embed everything
    python3 update_poetry_embeddings.py --tables text,book    # metadata only
    python3 update_poetry_embeddings.py --tables line         # lines only
    python3 update_poetry_embeddings.py --remake line         # re-embed all lines

Note: activate the project venv first with `venv_pls`
"""

import os
import json
import logging
import argparse
import sqlean as sqlite3
import sqlite_vec
from sentence_transformers import SentenceTransformer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "poetry.db")


def update_poetry_embeddings(tables=None, remake=None):
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    if not os.path.exists(DB_PATH):
        print(f"Error: '{DB_PATH}' not found. Build the poetry database first.")
        return

    do_text = tables is None or "text" in tables
    do_book = tables is None or "book" in tables
    do_line = tables is None or "line" in tables
    remake_text = remake in ("text", "all")
    remake_line = remake in ("line", "all")

    desc_parts = []
    if do_text:
        desc_parts.append("text_entries → vec_value_features")
    if do_book:
        desc_parts.append("book_entries → vec_book_features")
    if do_line:
        desc_parts.append("line_entries → vec_line_features" +
                          (" (full corpus — may take hours on CPU)" if not remake_line else " (full remake)"))

    print(f"Will embed: {', '.join(desc_parts)}")
    if input("Continue? (y/n): ").strip().lower() not in ('y', 'yes'):
        print("Cancelled.")
        return

    conn = sqlite3.connect(DB_PATH)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    cursor = conn.cursor()

    logging.info("Loading MiniLM model...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    logging.info("Model loaded.")

    # -------------------------------------------------------------------------
    # TEXT ENTRIES  (authors, subjects → vec_value_features)
    # -------------------------------------------------------------------------
    if do_text:
        logging.info("--- TEXT ENTRIES (text_entries → vec_value_features) ---")

        cursor.execute('''
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_value_features USING vec0(
                id TEXT PRIMARY KEY,
                embedding float[384])
        ''')

        cursor.execute("SELECT entry_id, value, type FROM text_entries")
        text_entries = cursor.fetchall()
        logging.info(f"Found {len(text_entries)} text entries.")

        updated = 0
        for entry_id, value, type_ in text_entries:
            cursor.execute("SELECT 1 FROM vec_value_features WHERE id = ?", (entry_id,))
            exists = cursor.fetchone()
            if exists and not remake_text:
                continue
            if exists and remake_text:
                cursor.execute("DELETE FROM vec_value_features WHERE id = ?", (entry_id,))

            if not value:
                continue

            text_to_embed = f"{type_}: {value}" if type_ else value
            embedding = model.encode(text_to_embed)
            cursor.execute(
                "INSERT INTO vec_value_features (id, embedding) VALUES (?, ?)",
                (entry_id, embedding.tobytes())
            )
            updated += 1
            if updated % 500 == 0:
                conn.commit()
                logging.info(f"  Committed {updated} text entries.")

        conn.commit()
        logging.info(f"Done: {updated} text entries embedded.")

    # -------------------------------------------------------------------------
    # BOOK ENTRIES  (titles + metadata → vec_book_features)
    # -------------------------------------------------------------------------
    if do_book:
        logging.info("--- BOOK ENTRIES (book_entries → vec_book_features) ---")

        cursor.execute('''
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_book_features USING vec0(
                book_id TEXT PRIMARY KEY,
                embedding float[384])
        ''')

        cursor.execute(
            "SELECT book_id, value, relatedKeywordStrings, descriptions FROM book_entries"
        )
        book_entries = cursor.fetchall()
        logging.info(f"Found {len(book_entries)} book entries.")

        updated = 0
        for book_id, title, related_strings_json, descriptions_json in book_entries:
            cursor.execute("SELECT 1 FROM vec_book_features WHERE book_id = ?", (book_id,))
            if cursor.fetchone():
                continue

            parts = [title] if title else []

            if related_strings_json:
                try:
                    parts.extend(json.loads(related_strings_json)[:3])
                except json.JSONDecodeError:
                    pass

            if descriptions_json:
                try:
                    desc = json.loads(descriptions_json)
                    subjects = desc.get("gutenberg", {}).get("subjects", [])
                    if subjects:
                        parts.append(', '.join(subjects[:5]))
                except json.JSONDecodeError:
                    pass

            text = ', '.join(p for p in parts if p)
            if not text:
                continue
            if len(text) > 300:
                text = text[:297] + "..."

            embedding = model.encode(text)
            cursor.execute(
                "INSERT INTO vec_book_features (book_id, embedding) VALUES (?, ?)",
                (book_id, embedding.tobytes())
            )
            updated += 1
            if updated % 100 == 0:
                conn.commit()
                logging.info(f"  Committed {updated} book entries.")

        conn.commit()
        logging.info(f"Done: {updated} book entries embedded.")

    # -------------------------------------------------------------------------
    # LINE ENTRIES  (individual poetry lines → vec_line_features)
    # -------------------------------------------------------------------------
    if do_line:
        if remake_line:
            logging.info("--- LINE ENTRIES (remaking all from scratch) ---")
        else:
            logging.info("--- LINE ENTRIES (updating new entries only) ---")

        cursor.execute('''
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_line_features USING vec0(
                line_id TEXT PRIMARY KEY,
                embedding float[384])
        ''')

        cursor.execute("SELECT COUNT(*) FROM line_entries")
        total_lines = cursor.fetchone()[0]
        logging.info(f"Found {total_lines:,} line entries.")

        if remake_line:
            logging.warning("Dropping all existing vec_line_features embeddings...")
            cursor.execute("DELETE FROM vec_line_features")
            conn.commit()

        # Stream in batches to avoid loading all ~3M rows into memory at once
        BATCH_SIZE = 1000
        offset = 0
        updated = 0
        skipped = 0

        while True:
            cursor.execute(
                "SELECT line_id, value FROM line_entries LIMIT ? OFFSET ?",
                (BATCH_SIZE, offset)
            )
            batch = cursor.fetchall()
            if not batch:
                break

            ids_to_embed = []
            texts_to_embed = []

            for line_id, value in batch:
                if not remake_line:
                    cursor.execute("SELECT 1 FROM vec_line_features WHERE line_id = ?", (line_id,))
                    if cursor.fetchone():
                        skipped += 1
                        continue
                if value:
                    ids_to_embed.append(line_id)
                    texts_to_embed.append(value)

            if texts_to_embed:
                # Batch encode — sentence_transformers handles batching internally
                embeddings = model.encode(texts_to_embed, batch_size=64, show_progress_bar=False)
                for line_id, embedding in zip(ids_to_embed, embeddings):
                    cursor.execute(
                        "INSERT OR REPLACE INTO vec_line_features (line_id, embedding) VALUES (?, ?)",
                        (line_id, embedding.tobytes())
                    )
                conn.commit()
                updated += len(ids_to_embed)

            offset += BATCH_SIZE

            processed = updated + skipped
            if processed % 10_000 == 0 and processed > 0:
                pct = processed / total_lines * 100 if total_lines else 0
                logging.info(
                    f"  Progress: {processed:,}/{total_lines:,} ({pct:.1f}%) "
                    f"— {updated:,} embedded, {skipped:,} already existed"
                )

        logging.info(f"Done: {updated:,} lines embedded ({skipped:,} already existed).")

    conn.close()
    logging.info("All done. Database connection closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Add text embeddings to poetry.db')
    parser.add_argument(
        '--tables', type=str, default=None,
        help='Comma-separated list of tables to embed: text,book,line (default: all)'
    )
    parser.add_argument(
        '--remake', type=str, default=None, choices=['text', 'line', 'all'],
        help='Remake embeddings from scratch for the given table(s)'
    )
    args = parser.parse_args()

    tables = [t.strip() for t in args.tables.split(',')] if args.tables else None
    update_poetry_embeddings(tables=tables, remake=args.remake)
