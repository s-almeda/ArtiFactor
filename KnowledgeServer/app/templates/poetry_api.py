"""
poetry_api.py
=============
Blueprint for browsing poetry.db — parallel to comics_browser_api.py.

Register in index.py:
    from templates.poetry_api import poetry_api_bp
    app.register_blueprint(poetry_api_bp)

Endpoints
---------
  GET  /api/poetry/browse                  paginated table browser
  POST /api/poetry/lookup                  fetch one entry by id
  GET  /api/poetry/book_lines/<book_id>    all lines for a book (paginated)
  POST /api/poetry/similar/lines           find semantically similar poetry lines
  POST /api/poetry/similar/text            find similar authors / subjects
  POST /api/poetry/similar/books           find similar books
"""

import json
import os
import numpy as np
from flask import Blueprint, jsonify, request, g, current_app
import sqlite_vec
import sqlean as sqlite3

from helper_functions import helperfunctions as hf
from config import BASE_DIR

POETRY_DB_PATH = os.path.join(BASE_DIR, "LOCALDB", "poetry.db")

poetry_api_bp = Blueprint("poetry_api", __name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_poetry_db():
    """Return a per-request connection to poetry.db (stored on Flask's g)."""
    if "poetry_db" not in g:
        g.poetry_db = sqlite3.connect(POETRY_DB_PATH)
        g.poetry_db.row_factory = sqlite3.Row
        g.poetry_db.enable_load_extension(True)
        sqlite_vec.load(g.poetry_db)
        g.poetry_db.enable_load_extension(False)
    return g.poetry_db


@poetry_api_bp.teardown_app_request
def close_poetry_db(_exception=None):
    db = g.pop("poetry_db", None)
    if db is not None:
        db.close()


def _normalize_id_list(value):
    if value is None:
        return None
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return None


def _json_or_error():
    body = request.get_json(silent=True)
    if body is None:
        return None, (jsonify({"success": False, "error": "Request body must be valid JSON"}), 400)
    return body, None


def _fetch_rows_by_ids(db, table, id_col, ordered_ids):
    if not ordered_ids:
        return {}
    placeholders = ",".join(["?" for _ in ordered_ids])
    rows = db.execute(
        f"SELECT * FROM {table} WHERE {id_col} IN ({placeholders})", ordered_ids
    ).fetchall()
    return {row[id_col]: dict(row) for row in rows}


# Column whitelists — prevents SQL injection via sort_by param
VALID_SORT_COLUMNS = {
    "line_entries": ["line_id", "value", "book_id", "line_number"],
    "book_entries": ["book_id", "value", "gid", "line_count"],
    "text_entries": ["entry_id", "value", "type", "isArtist"],
}
DEFAULT_SORT = {
    "line_entries": "line_number",
    "book_entries": "book_id",
    "text_entries": "entry_id",
}
VALID_TABLES = set(VALID_SORT_COLUMNS.keys())


# ---------------------------------------------------------------------------
# /api/poetry/browse  — paginated table browser
# ---------------------------------------------------------------------------

@poetry_api_bp.route("/api/poetry/browse")
def api_poetry_browse():
    """
    Query params:
        table       line_entries | book_entries | text_entries  (default: book_entries)
        page        1-indexed                                    (default: 1)
        page_size   integer or "all"                            (default: 25)
        sort_by     column name (validated against whitelist)
        sort_dir    asc | desc                                  (default: asc)
        book_id     optional — filter line_entries by book
        author_id   optional — filter book_entries by author
    """
    try:
        table            = request.args.get("table", "book_entries")
        page             = int(request.args.get("page", 1))
        raw_size         = request.args.get("page_size", "25")
        sort_by          = request.args.get("sort_by", None)
        sort_dir         = request.args.get("sort_dir", "asc").lower()
        book_id_filter   = request.args.get("book_id", None)
        author_id_filter = request.args.get("author_id", None)

        if table not in VALID_TABLES:
            return jsonify({"success": False, "error": f"Invalid table '{table}'"})

        page_size = 1_000_000 if raw_size == "all" else max(1, int(raw_size))
        offset    = (page - 1) * page_size
        sort_dir  = "DESC" if sort_dir == "desc" else "ASC"

        if sort_by not in (VALID_SORT_COLUMNS.get(table) or []):
            sort_by = DEFAULT_SORT[table]

        db = get_poetry_db()

        where_clauses = []
        bind_params   = []

        if table == "line_entries" and book_id_filter:
            where_clauses.append("book_id = ?")
            bind_params.append(book_id_filter)

        if table == "book_entries" and author_id_filter:
            where_clauses.append("author_id = ?")
            bind_params.append(author_id_filter)

        where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

        count_row  = db.execute(
            f"SELECT COUNT(*) AS count FROM {table} {where_sql}", bind_params
        ).fetchone()
        total_rows = count_row["count"]

        if table == "line_entries":
            select_cols = (
                "line_id, value, book_id, line_number, "
                "descriptions, relatedKeywordIds, relatedKeywordStrings"
            )
        elif table == "book_entries":
            select_cols = (
                "book_id, value, gid, author_id, line_count, "
                "descriptions, relatedKeywordIds, relatedKeywordStrings"
            )
        else:  # text_entries
            select_cols = (
                "entry_id, value, isArtist, type, artist_aliases, "
                "descriptions, relatedKeywordIds, relatedKeywordStrings"
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
        current_app.logger.exception("Error in api_poetry_browse")
        return jsonify({"success": False, "error": str(e)})


# ---------------------------------------------------------------------------
# /api/poetry/lookup  — fetch a single entry by id
# ---------------------------------------------------------------------------

@poetry_api_bp.route("/api/poetry/lookup", methods=["POST"])
def api_poetry_lookup():
    """
    Body JSON:
        { "entryId": "gbp_book_20",            "type": "book"   }
        { "entryId": "gbp_20_l00001",          "type": "line"   }
        { "entryId": "gbp_author_keats-john",  "type": "text"   }
    """
    try:
        body     = request.get_json(force=True) or {}
        entry_id = body.get("entryId", "").strip()
        etype    = body.get("type", "").lower()

        if not entry_id:
            return jsonify({"error": "entryId is required"})

        db = get_poetry_db()

        if etype == "book":
            row = db.execute(
                "SELECT * FROM book_entries WHERE book_id = ?", (entry_id,)
            ).fetchone()
        elif etype == "line":
            row = db.execute(
                "SELECT * FROM line_entries WHERE line_id = ?", (entry_id,)
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
        current_app.logger.exception("Error in api_poetry_lookup")
        return jsonify({"error": str(e)})


# ---------------------------------------------------------------------------
# /api/poetry/book_lines/<book_id>  — all lines for one book (paginated)
# ---------------------------------------------------------------------------

@poetry_api_bp.route("/api/poetry/book_lines/<book_id>")
def api_poetry_book_lines(book_id):
    """Return line_entries for a book in order, paginated."""
    try:
        page      = int(request.args.get("page", 1))
        page_size = int(request.args.get("page_size", 100))
        offset    = (page - 1) * page_size

        db = get_poetry_db()

        total = db.execute(
            "SELECT COUNT(*) FROM line_entries WHERE book_id = ?", (book_id,)
        ).fetchone()[0]

        rows = db.execute(
            "SELECT * FROM line_entries WHERE book_id = ? ORDER BY line_number LIMIT ? OFFSET ?",
            (book_id, page_size, offset)
        ).fetchall()

        return jsonify({
            "success":     True,
            "book_id":     book_id,
            "total_lines": total,
            "page":        page,
            "page_size":   page_size,
            "lines":       [dict(r) for r in rows],
        })

    except Exception as e:
        current_app.logger.exception("Error in api_poetry_book_lines")
        return jsonify({"success": False, "error": str(e)})


# ---------------------------------------------------------------------------
# /api/poetry/similar/lines  — semantic similarity for poetry lines
# ---------------------------------------------------------------------------

@poetry_api_bp.route("/api/poetry/similar/lines", methods=["POST"])
def api_poetry_similar_lines():
    """
    Find the most semantically similar lines of poetry.

    Body JSON:
        {
            "query_text": "moonlight over still water",
            "line_id": "gbp_20_l00042",        (mutually exclusive with query_text)
            "top_k": 20,
            "book_ids": ["gbp_book_20"]         (optional — restrict to these books)
        }
    """
    try:
        body, error = _json_or_error()
        if error:
            return error

        query_text = str(body.get("query_text", "") or "").strip()
        line_id    = str(body.get("line_id", "") or "").strip()
        top_k_raw  = body.get("top_k", 10)
        book_ids   = _normalize_id_list(body.get("book_ids"))

        try:
            top_k = max(1, int(top_k_raw))
        except Exception:
            return jsonify({"success": False, "error": "top_k must be an integer"}), 400

        if not query_text and not line_id:
            return jsonify({"success": False, "error": "Provide either query_text or line_id"}), 400

        db = get_poetry_db()

        if query_text:
            query_vector = hf.extract_text_features(query_text)
            query_type = "query_text"
        else:
            row = db.execute(
                "SELECT embedding FROM vec_line_features WHERE line_id = ?", (line_id,)
            ).fetchone()
            if row and row["embedding"]:
                query_vector = np.frombuffer(row["embedding"], dtype=np.float32)
            else:
                line_row = db.execute(
                    "SELECT value FROM line_entries WHERE line_id = ?", (line_id,)
                ).fetchone()
                if not line_row:
                    return jsonify({"success": False, "error": f"line_id '{line_id}' not found"}), 404
                query_vector = hf.extract_text_features(line_row["value"])
            query_type = "line_id"

        serialized = hf.serialize_f32(query_vector)

        if book_ids:
            placeholders = ",".join("?" for _ in book_ids)
            sql = f"""
                SELECT line_id, distance
                FROM vec_line_features
                WHERE embedding MATCH ?
                AND line_id IN (
                    SELECT line_id FROM line_entries WHERE book_id IN ({placeholders})
                )
                ORDER BY distance
                LIMIT ?
            """
            params = [serialized] + book_ids + [top_k]
        else:
            sql = """
                SELECT line_id, distance
                FROM vec_line_features
                WHERE embedding MATCH ?
                ORDER BY distance
                LIMIT ?
            """
            params = [serialized, top_k]

        raw_rows = db.execute(sql, params).fetchall()

        # Ensure the query line itself is included in results
        result_ids = [r["line_id"] for r in raw_rows]
        if line_id and line_id not in result_ids:
            exists = db.execute(
                "SELECT 1 FROM line_entries WHERE line_id = ? LIMIT 1", (line_id,)
            ).fetchone()
            if exists:
                raw_rows = [{"line_id": line_id, "distance": 0.0}] + list(raw_rows)

        raw_rows = list(raw_rows)[:top_k]

        if not raw_rows:
            return jsonify({"success": True, "query_type": query_type, "top_k": top_k, "count": 0, "rows": []})

        ordered_ids = [r["line_id"] for r in raw_rows]
        entries_map = _fetch_rows_by_ids(db, "line_entries", "line_id", ordered_ids)

        rows = []
        for raw in raw_rows:
            entry = entries_map.get(raw["line_id"])
            if entry:
                entry["distance"] = raw["distance"]
                rows.append(entry)

        return jsonify({
            "success":    True,
            "query_type": query_type,
            "top_k":      top_k,
            "count":      len(rows),
            "rows":       rows,
        })

    except Exception as e:
        current_app.logger.exception("Error in api_poetry_similar_lines")
        return jsonify({"success": False, "error": str(e)}), 500


# ---------------------------------------------------------------------------
# /api/poetry/similar/text  — semantic similarity for authors / subjects
# ---------------------------------------------------------------------------

@poetry_api_bp.route("/api/poetry/similar/text", methods=["POST"])
def api_poetry_similar_text():
    """
    Find the most semantically similar authors or subjects.

    Body JSON:
        {
            "query_text": "romantic poet nature",
            "entry_id": "gbp_author_keats-john",   (mutually exclusive with query_text)
            "top_k": 5,
            "type_filter": "author"                (optional: "author" | "subject")
        }
    """
    try:
        body, error = _json_or_error()
        if error:
            return error

        query_text  = str(body.get("query_text", "") or "").strip()
        entry_id    = str(body.get("entry_id", "") or "").strip()
        top_k_raw   = body.get("top_k", 10)
        type_filter = str(body.get("type_filter", "") or "").strip().lower()

        try:
            top_k = max(1, int(top_k_raw))
        except Exception:
            return jsonify({"success": False, "error": "top_k must be an integer"}), 400

        if not query_text and not entry_id:
            return jsonify({"success": False, "error": "Provide either query_text or entry_id"}), 400

        db = get_poetry_db()

        if query_text:
            query_vector = hf.extract_text_features(query_text)
            query_type = "query_text"
        else:
            row = db.execute(
                "SELECT embedding FROM vec_value_features WHERE id = ?", (entry_id,)
            ).fetchone()
            if row and row["embedding"]:
                query_vector = np.frombuffer(row["embedding"], dtype=np.float32)
            else:
                te_row = db.execute(
                    "SELECT value, type FROM text_entries WHERE entry_id = ?", (entry_id,)
                ).fetchone()
                if not te_row:
                    return jsonify({"success": False, "error": f"entry_id '{entry_id}' not found"}), 404
                query_vector = hf.extract_text_features(f"{te_row['type']}: {te_row['value']}")
            query_type = "entry_id"

        serialized = hf.serialize_f32(query_vector)

        if type_filter in ("author", "subject"):
            sql = """
                SELECT id AS entry_id, distance
                FROM vec_value_features
                WHERE embedding MATCH ?
                AND id IN (SELECT entry_id FROM text_entries WHERE type = ?)
                ORDER BY distance
                LIMIT ?
            """
            params = [serialized, type_filter, top_k]
        else:
            sql = """
                SELECT id AS entry_id, distance
                FROM vec_value_features
                WHERE embedding MATCH ?
                ORDER BY distance
                LIMIT ?
            """
            params = [serialized, top_k]

        raw_rows = db.execute(sql, params).fetchall()

        if not raw_rows:
            return jsonify({"success": True, "query_type": query_type, "top_k": top_k, "count": 0, "rows": []})

        ordered_ids = [r["entry_id"] for r in raw_rows]
        entries_map = _fetch_rows_by_ids(db, "text_entries", "entry_id", ordered_ids)

        rows = []
        for raw in raw_rows:
            entry = entries_map.get(raw["entry_id"])
            if entry:
                entry["distance"] = raw["distance"]
                rows.append(entry)

        return jsonify({
            "success":    True,
            "query_type": query_type,
            "top_k":      top_k,
            "count":      len(rows),
            "rows":       rows,
        })

    except Exception as e:
        current_app.logger.exception("Error in api_poetry_similar_text")
        return jsonify({"success": False, "error": str(e)}), 500


# ---------------------------------------------------------------------------
# /api/poetry/similar/books  — semantic similarity for book_entries
# ---------------------------------------------------------------------------

@poetry_api_bp.route("/api/poetry/similar/books", methods=["POST"])
def api_poetry_similar_books():
    """
    Find the most semantically similar books.

    Body JSON:
        {
            "query_text": "victorian nature poetry",
            "book_id": "gbp_book_20",   (mutually exclusive with query_text)
            "top_k": 5
        }
    """
    try:
        body, error = _json_or_error()
        if error:
            return error

        query_text = str(body.get("query_text", "") or "").strip()
        book_id    = str(body.get("book_id", "") or "").strip()
        top_k_raw  = body.get("top_k", 10)

        try:
            top_k = max(1, int(top_k_raw))
        except Exception:
            return jsonify({"success": False, "error": "top_k must be an integer"}), 400

        if not query_text and not book_id:
            return jsonify({"success": False, "error": "Provide either query_text or book_id"}), 400

        db = get_poetry_db()

        if query_text:
            query_vector = hf.extract_text_features(query_text)
            query_type = "query_text"
        else:
            row = db.execute(
                "SELECT embedding FROM vec_book_features WHERE book_id = ?", (book_id,)
            ).fetchone()
            if row and row["embedding"]:
                query_vector = np.frombuffer(row["embedding"], dtype=np.float32)
            else:
                be_row = db.execute(
                    "SELECT value, relatedKeywordStrings FROM book_entries WHERE book_id = ?",
                    (book_id,)
                ).fetchone()
                if not be_row:
                    return jsonify({"success": False, "error": f"book_id '{book_id}' not found"}), 404
                text = be_row["value"] or ""
                if be_row["relatedKeywordStrings"]:
                    try:
                        text += ", " + ", ".join(json.loads(be_row["relatedKeywordStrings"])[:3])
                    except Exception:
                        pass
                query_vector = hf.extract_text_features(text)
            query_type = "book_id"

        serialized = hf.serialize_f32(query_vector)

        raw_rows = db.execute("""
            SELECT book_id, distance
            FROM vec_book_features
            WHERE embedding MATCH ?
            ORDER BY distance
            LIMIT ?
        """, [serialized, top_k]).fetchall()

        if not raw_rows:
            return jsonify({"success": True, "query_type": query_type, "top_k": top_k, "count": 0, "rows": []})

        ordered_ids = [r["book_id"] for r in raw_rows]
        placeholders = ",".join("?" for _ in ordered_ids)
        entries_map = {
            row["book_id"]: dict(row)
            for row in db.execute(
                f"SELECT book_id, value, gid, author_id, line_count, "
                f"descriptions, relatedKeywordIds, relatedKeywordStrings "
                f"FROM book_entries WHERE book_id IN ({placeholders})",
                ordered_ids
            ).fetchall()
        }

        rows = []
        for raw in raw_rows:
            entry = entries_map.get(raw["book_id"])
            if entry:
                entry["distance"] = raw["distance"]
                rows.append(entry)

        return jsonify({
            "success":    True,
            "query_type": query_type,
            "top_k":      top_k,
            "count":      len(rows),
            "rows":       rows,
        })

    except Exception as e:
        current_app.logger.exception("Error in api_poetry_similar_books")
        return jsonify({"success": False, "error": str(e)}), 500
