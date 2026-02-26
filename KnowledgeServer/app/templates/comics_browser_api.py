"""
comics_browser_api.py
Blueprint for browsing comics.db — parallel to the existing database_browser routes
in index.py but targeting the comics database.

Register in index.py:
    from templates.comics_browser_api import comics_browser_api_bp
    app.register_blueprint(comics_browser_api_bp)
"""

from flask import Blueprint, jsonify, request, g, current_app, send_from_directory, abort
import sqlite_vec
import sqlean as sqlite3
import os

from config import BASE_DIR

# ---------------------------------------------------------------------------
# Path to the comics database — adjust if yours lives elsewhere
# ---------------------------------------------------------------------------
COMICS_DB_PATH = os.path.join(BASE_DIR, "LOCALDB", "comics.db")
COMICS_IMAGES_DIR = os.path.join(BASE_DIR, "LOCALDB", "comic_images")

comics_browser_api_bp = Blueprint("comics_browser_api", __name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_comics_db():
    """Return a per-request connection to comics.db (stored on Flask's g)."""
    if "comics_db" not in g:
        g.comics_db = sqlite3.connect(COMICS_DB_PATH)
        g.comics_db.row_factory = sqlite3.Row
        # TODO: Uncomment when vector tables are added
        # g.comics_db.enable_load_extension(True)
        # sqlite_vec.load(g.comics_db)
        # g.comics_db.enable_load_extension(False)
    return g.comics_db


# Column whitelists — prevents SQL injection via sort_by param
VALID_SORT_COLUMNS = {
    "image_entries": ["image_id", "value", "filename", "book_id", "page_number"],
    "book_entries":  ["book_id", "value", "series_id", "issue_number", "cover_date", "page_count"],
    "text_entries":  ["entry_id", "value", "type", "isArtist"],
}

DEFAULT_SORT = {
    "image_entries": "page_number",
    "book_entries":  "book_id",
    "text_entries":  "entry_id",
}

VALID_TABLES = set(VALID_SORT_COLUMNS.keys())


# ---------------------------------------------------------------------------
# /api/comics/browse  — paginated table browser
# ---------------------------------------------------------------------------

@comics_browser_api_bp.route("/api/comics/browse")
def api_comics_browse():
    """
    Query params:
        table       image_entries | book_entries | text_entries  (default: book_entries)
        page        1-indexed                                     (default: 1)
        page_size   integer or "all"                             (default: 25)
        sort_by     column name (validated against whitelist)
        sort_dir    asc | desc                                   (default: asc)
        series_id   optional — filter book_entries / image_entries by series
        book_id     optional — filter image_entries by book
    """
    try:
        table    = request.args.get("table", "book_entries")
        page     = int(request.args.get("page", 1))
        raw_size = request.args.get("page_size", "25")
        sort_by  = request.args.get("sort_by", None)
        sort_dir = request.args.get("sort_dir", "asc").lower()
        series_id = request.args.get("series_id", None)
        book_id_filter = request.args.get("book_id", None)

        if table not in VALID_TABLES:
            return jsonify({"success": False, "error": f"Invalid table '{table}'"})

        page_size = 1_000_000 if raw_size == "all" else max(1, int(raw_size))
        offset    = (page - 1) * page_size
        sort_dir  = "DESC" if sort_dir == "desc" else "ASC"

        # Validate / default sort column
        if sort_by not in (VALID_SORT_COLUMNS.get(table) or []):
            sort_by = DEFAULT_SORT[table]

        db = get_comics_db()

        # ----------------------------------------------------------------
        # Build WHERE clause from optional filters
        # ----------------------------------------------------------------
        where_clauses = []
        bind_params   = []

        if table == "book_entries" and series_id:
            where_clauses.append("series_id = ?")
            bind_params.append(series_id)

        if table == "image_entries":
            if book_id_filter:
                where_clauses.append("book_id = ?")
                bind_params.append(book_id_filter)
            elif series_id:
                # image_entries don't have series_id directly — join through book_entries
                where_clauses.append(
                    "book_id IN (SELECT book_id FROM book_entries WHERE series_id = ?)"
                )
                bind_params.append(series_id)

        where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

        # ----------------------------------------------------------------
        # Total count
        # ----------------------------------------------------------------
        count_row = db.execute(
            f"SELECT COUNT(*) AS count FROM {table} {where_sql}", bind_params
        ).fetchone()
        total_rows = count_row["count"]

        # ----------------------------------------------------------------
        # Select columns per table
        # ----------------------------------------------------------------
        if table == "image_entries":
            select_cols = (
                "image_id, value, artist_names, image_urls, filename, "
                "rights, descriptions, relatedKeywordIds, relatedKeywordStrings, "
                "book_id, page_number"
            )
        elif table == "book_entries":
            select_cols = (
                "book_id, value, series_id, issue_number, cover_date, "
                "page_count, cover_image_id, descriptions, "
                "relatedKeywordIds, relatedKeywordStrings"
            )
        else:  # text_entries
            select_cols = (
                "entry_id, value, images, isArtist, type, "
                "artist_aliases, descriptions, relatedKeywordIds, relatedKeywordStrings"
            )

        query = (
            f"SELECT {select_cols} FROM {table} {where_sql} "
            f"ORDER BY {sort_by} {sort_dir} "
            f"LIMIT ? OFFSET ?"
        )
        rows = [dict(r) for r in db.execute(query, bind_params + [page_size, offset]).fetchall()]

        return jsonify({
            "success":    True,
            "table":      table,
            "page":       page,
            "page_size":  page_size,
            "total_rows": total_rows,
            "rows":       rows,
        })

    except Exception as e:
        current_app.logger.exception("Error in api_comics_browse")
        return jsonify({"success": False, "error": str(e)})


# ---------------------------------------------------------------------------
# /api/comics/lookup  — fetch a single entry by id
# ---------------------------------------------------------------------------

@comics_browser_api_bp.route("/api/comics/lookup", methods=["POST"])
def api_comics_lookup():
    """
    Body JSON:
        { "entryId": "cbp_77488",        "type": "book"   }
        { "entryId": "cbp_77488_p001",   "type": "image"  }
        { "entryId": "cbp_series_3503",  "type": "text"   }
    """
    try:
        body     = request.get_json(force=True) or {}
        entry_id = body.get("entryId", "").strip()
        etype    = body.get("type", "").lower()

        if not entry_id:
            return jsonify({"error": "entryId is required"})

        db = get_comics_db()

        if etype == "book":
            row = db.execute(
                "SELECT * FROM book_entries WHERE book_id = ?", (entry_id,)
            ).fetchone()
        elif etype == "image":
            row = db.execute(
                "SELECT * FROM image_entries WHERE image_id = ?", (entry_id,)
            ).fetchone()
        elif etype == "text":
            row = db.execute(
                "SELECT * FROM text_entries WHERE entry_id = ?", (entry_id,)
            ).fetchone()
        else:
            return jsonify({"error": f"Unknown type '{etype}'"})

        if row is None:
            return jsonify({"error": f"No entry found for id '{entry_id}'"})

        return jsonify(dict(row))

    except Exception as e:
        current_app.logger.exception("Error in api_comics_lookup")
        return jsonify({"error": str(e)})


# ---------------------------------------------------------------------------
# /api/comics/book_pages/<book_id>  — convenience: all pages for one book
# ---------------------------------------------------------------------------

@comics_browser_api_bp.route("/api/comics/book_pages/<book_id>")
def api_comics_book_pages(book_id):
    """Return all image_entries for a book, in page order."""
    try:
        db = get_comics_db()
        rows = db.execute(
            "SELECT * FROM image_entries WHERE book_id = ? ORDER BY page_number",
            (book_id,)
        ).fetchall()
        return jsonify({"success": True, "book_id": book_id, "pages": [dict(r) for r in rows]})
    except Exception as e:
        current_app.logger.exception("Error in api_comics_book_pages")
        return jsonify({"success": False, "error": str(e)})


# ---------------------------------------------------------------------------
# /api/comics/image/<path:filename>  — serve local comics images/thumbnails
# ---------------------------------------------------------------------------

@comics_browser_api_bp.route("/api/comics/image/<path:filename>")
def api_comics_image_file(filename):
    """Serve local files from LOCALDB/comic_images (including thumbs/*)."""
    try:
        safe_base = os.path.abspath(COMICS_IMAGES_DIR)
        target = os.path.abspath(os.path.join(safe_base, filename))
        if not target.startswith(safe_base + os.sep):
            abort(400)

        if not os.path.exists(target):
            abort(404)

        rel_dir = os.path.dirname(filename)
        rel_file = os.path.basename(filename)
        directory = os.path.join(COMICS_IMAGES_DIR, rel_dir) if rel_dir else COMICS_IMAGES_DIR
        return send_from_directory(directory, rel_file)
    except Exception:
        current_app.logger.exception("Error serving comics image file")
        abort(404)