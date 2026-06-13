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
import numpy as np
from PIL import Image

from helper_functions import helperfunctions as hf

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
        g.comics_db.enable_load_extension(True)
        sqlite_vec.load(g.comics_db)
        g.comics_db.enable_load_extension(False)
    return g.comics_db


@comics_browser_api_bp.teardown_app_request
def close_comics_db(_exception=None):
    db = g.pop("comics_db", None)
    if db is not None:
        db.close()


def _normalize_id_list(value):
    if value is None:
        return None
    if isinstance(value, list):
        normalized = [str(v).strip() for v in value if str(v).strip()]
        return normalized
    if isinstance(value, str):
        normalized = [v.strip() for v in value.split(",") if v.strip()]
        return normalized
    return None


def _normalize_similarity_rows(raw_rows, id_key):
    if raw_rows is None:
        return []

    if hasattr(raw_rows, "to_dict"):
        records = raw_rows.to_dict(orient="records")
    elif isinstance(raw_rows, list):
        records = raw_rows
    else:
        records = []

    normalized = []
    for record in records:
        if not isinstance(record, dict):
            continue
        result_id = record.get(id_key)
        if not result_id:
            continue
        distance = record.get("distance", 0.0)
        try:
            distance = float(distance)
        except Exception:
            distance = 0.0
        normalized.append({id_key: result_id, "distance": distance})

    return normalized


def _fetch_rows_by_ids(db, table, id_col, ordered_ids):
    if not ordered_ids:
        return {}

    placeholders = ",".join(["?" for _ in ordered_ids])
    query = f"SELECT * FROM {table} WHERE {id_col} IN ({placeholders})"
    rows = db.execute(query, ordered_ids).fetchall()
    return {row[id_col]: dict(row) for row in rows}


def _json_or_error():
    body = request.get_json(silent=True)
    if body is None:
        return None, (jsonify({"success": False, "error": "Request body must be valid JSON"}), 400)
    return body, None


def _get_text_embedding_for_entry(db, entry_id, search_in):
    vectors = []

    if search_in in ("description", "both"):
        row = db.execute(
            "SELECT embedding FROM vec_description_features WHERE id = ?",
            (entry_id,)
        ).fetchone()
        if row and row["embedding"]:
            vectors.append(np.frombuffer(row["embedding"], dtype=np.float32))

    if search_in in ("value", "both"):
        row = db.execute(
            "SELECT embedding FROM vec_value_features WHERE id = ?",
            (entry_id,)
        ).fetchone()
        if row and row["embedding"]:
            vectors.append(np.frombuffer(row["embedding"], dtype=np.float32))

    if vectors:
        if len(vectors) == 1:
            return vectors[0]
        return np.mean(np.vstack(vectors), axis=0).astype(np.float32)

    fallback_row = db.execute(
        "SELECT type, value, artist_aliases, descriptions FROM text_entries WHERE entry_id = ?",
        (entry_id,)
    ).fetchone()
    if not fallback_row:
        return None

    synthetic_text = ", ".join([
        fallback_row["type"] or "",
        fallback_row["value"] or "",
        fallback_row["artist_aliases"] or "",
        fallback_row["descriptions"] or ""
    ]).strip(", ")

    if not synthetic_text:
        return None

    return hf.extract_text_features(synthetic_text)


def _get_image_embedding_for_entry(db, image_id):
    row = db.execute(
        "SELECT embedding FROM vec_image_features WHERE image_id = ?",
        (image_id,)
    ).fetchone()
    if row and row["embedding"]:
        return np.frombuffer(row["embedding"], dtype=np.float32)

    file_row = db.execute(
        "SELECT filename FROM image_entries WHERE image_id = ?",
        (image_id,)
    ).fetchone()
    if not file_row:
        return None, f"No image entry found for image_id '{image_id}'", 404

    filename = file_row["filename"]
    if not filename:
        return None, f"No filename available for image_id '{image_id}'", 400

    image_path = os.path.join(COMICS_IMAGES_DIR, filename)
    if not os.path.exists(image_path):
        return None, f"Image file not found: {filename}", 404

    with Image.open(image_path).convert("RGB") as image:
        return hf.extract_img_features(image), None, None


def _ensure_self_match(db, normalized_rows, table, id_col, query_id):
    if not query_id:
        return normalized_rows

    if any(row[id_col] == query_id for row in normalized_rows):
        return normalized_rows

    exists = db.execute(
        f"SELECT 1 FROM {table} WHERE {id_col} = ? LIMIT 1",
        (query_id,)
    ).fetchone()
    if exists:
        normalized_rows.insert(0, {id_col: query_id, "distance": 0.0})

    return normalized_rows


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
                "book_id, page_number, ocr_text"
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


# ---------------------------------------------------------------------------
# /api/comics/similar/text  — semantic text similarity in comics.db
# ---------------------------------------------------------------------------

@comics_browser_api_bp.route("/api/comics/similar/text", methods=["POST"])
def api_comics_similar_text():
    """
    Body JSON supports either raw query text or entry lookup:
        {
            "query_text": "batman detective",
            "entry_id": "cbp_series_3503",
            "top_k": 10,
            "search_in": "both",
            "entry_ids": ["cbp_series_3503", "cbp_creator_123"]
        }
    """
    try:
        body, error = _json_or_error()
        if error:
            return error

        query_text = str(body.get("query_text", "") or "").strip()
        entry_id = str(body.get("entry_id", "") or "").strip()
        search_in = str(body.get("search_in", "both") or "both").strip().lower()
        top_k_raw = body.get("top_k", 10)
        entry_ids = _normalize_id_list(body.get("entry_ids"))

        if search_in not in {"description", "value", "both"}:
            return jsonify({"success": False, "error": "search_in must be one of: description, value, both"}), 400

        try:
            top_k = max(1, int(top_k_raw))
        except Exception:
            return jsonify({"success": False, "error": "top_k must be an integer"}), 400

        if not query_text and not entry_id:
            return jsonify({"success": False, "error": "Provide either query_text or entry_id"}), 400

        db = get_comics_db()

        if query_text:
            query_vector = hf.extract_text_features(query_text)
            query_type = "query_text"
        else:
            query_vector = _get_text_embedding_for_entry(db, entry_id, search_in)
            query_type = "entry_id"
            if query_vector is None:
                return jsonify({"success": False, "error": f"Unable to generate embedding for entry_id '{entry_id}'"}), 404

        serialized_query = hf.serialize_f32(query_vector)
        raw_rows = hf.find_most_similar_texts(
            serialized_query,
            db,
            top_k=top_k,
            search_in=search_in,
            entry_ids=entry_ids,
        )

        normalized = _normalize_similarity_rows(raw_rows, "entry_id")
        normalized = _ensure_self_match(db, normalized, "text_entries", "entry_id", entry_id)
        normalized = normalized[:top_k]

        ordered_ids = [row["entry_id"] for row in normalized]
        rows_map = _fetch_rows_by_ids(db, "text_entries", "entry_id", ordered_ids)

        rows = []
        for row in normalized:
            full_row = rows_map.get(row["entry_id"])
            if not full_row:
                continue
            full_row["distance"] = row["distance"]
            rows.append(full_row)

        return jsonify({
            "success": True,
            "query_type": query_type,
            "search_in": search_in,
            "top_k": top_k,
            "count": len(rows),
            "rows": rows,
        })

    except Exception as e:
        current_app.logger.exception("Error in api_comics_similar_text")
        return jsonify({"success": False, "error": str(e)}), 500


# ---------------------------------------------------------------------------
# /api/comics/similar/image  — visual similarity in comics.db
# ---------------------------------------------------------------------------

@comics_browser_api_bp.route("/api/comics/similar/image", methods=["POST"])
def api_comics_similar_image():
    """
    Body JSON supports either image lookup by id or query image input:
        {
            "image_id": "cbp_77488_p001",
            "top_k": 10,
            "artwork_ids": ["cbp_77488_p001", "cbp_77488_p002"]
        }

        {
            "image": "https://..." | "data:image/...;base64,...",
            "top_k": 10
        }
    """
    try:
        body, error = _json_or_error()
        if error:
            return error

        image_id = str(body.get("image_id", "") or "").strip()
        image_input = body.get("image")
        image_url = body.get("image_url")
        image_b64 = body.get("image_b64")
        top_k_raw = body.get("top_k", 10)
        artwork_ids = _normalize_id_list(body.get("artwork_ids"))
        if artwork_ids is None:
            artwork_ids = _normalize_id_list(body.get("image_ids"))

        try:
            top_k = max(1, int(top_k_raw))
        except Exception:
            return jsonify({"success": False, "error": "top_k must be an integer"}), 400

        db = get_comics_db()
        query_type = None

        if image_id:
            image_vector_result = _get_image_embedding_for_entry(db, image_id)
            if isinstance(image_vector_result, tuple) and len(image_vector_result) == 3:
                query_vector, error_message, status_code = image_vector_result
                if error_message:
                    return jsonify({"success": False, "error": error_message}), status_code
            else:
                query_vector = image_vector_result
            query_type = "image_id"
        else:
            if image_input:
                if hf.check_image_url(image_input):
                    image_obj = hf.url_to_image(image_input)
                else:
                    image_obj = hf.base64_to_image(image_input)
            elif image_url:
                image_obj = hf.url_to_image(image_url)
            elif image_b64:
                image_obj = hf.base64_to_image(image_b64)
            else:
                return jsonify({"success": False, "error": "Provide image_id or image/image_url/image_b64"}), 400

            if image_obj is None:
                return jsonify({"success": False, "error": "Unable to parse image input"}), 400

            query_vector = hf.extract_img_features(image_obj)
            query_type = "raw_image"

        serialized_query = hf.serialize_f32(query_vector)
        raw_rows = hf.find_most_similar_images(
            serialized_query,
            db,
            top_k=top_k,
            artwork_ids=artwork_ids,
        )

        normalized = _normalize_similarity_rows(raw_rows, "image_id")
        normalized = _ensure_self_match(db, normalized, "image_entries", "image_id", image_id)
        normalized = normalized[:top_k]

        ordered_ids = [row["image_id"] for row in normalized]
        rows_map = _fetch_rows_by_ids(db, "image_entries", "image_id", ordered_ids)

        rows = []
        for row in normalized:
            full_row = rows_map.get(row["image_id"])
            if not full_row:
                continue
            full_row["distance"] = row["distance"]
            rows.append(full_row)

        return jsonify({
            "success": True,
            "query_type": query_type,
            "top_k": top_k,
            "count": len(rows),
            "rows": rows,
        })

    except Exception as e:
        current_app.logger.exception("Error in api_comics_similar_image")
        return jsonify({"success": False, "error": str(e)}), 500