"""
update_comics_ocr.py
====================
Runs OCR on downloaded comic page images and stores the extracted text
in the `ocr_text` column of image_entries in comics.db.

After running this, re-run update_comics_embeddings.py --clip remake
to rebuild vec_artworktext_features using the real OCR text.

Usage:
    python3 update_comics_ocr.py                         # easyocr (default), all unprocessed
    python3 update_comics_ocr.py --method tesseract      # use tesseract instead
    python3 update_comics_ocr.py --remake                # redo all (overwrite existing)
    python3 update_comics_ocr.py --limit 10              # only process first N (for testing)
    python3 update_comics_ocr.py -y --log ~/ocr_run.log  # headless/overnight run

Install:
    EasyOCR:   pip install easyocr
    Tesseract: brew install tesseract && pip install pytesseract   (mac)
               sudo apt-get install tesseract-ocr && pip install pytesseract  (linux)

Note: activate the project venv first with `venv_pls`
"""

import os
import sys
import time
import logging
import argparse
import sqlean as sqlite3
from PIL import Image


DB_PATH = "comics.db"
IMAGES_DIR = "comic_images"
CONFIDENCE_THRESHOLD = 0.4


# ---------------------------------------------------------------------------
# DB migration
# ---------------------------------------------------------------------------

def add_ocr_text_column(conn):
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(image_entries)")
    columns = [row[1] for row in cursor.fetchall()]
    if "ocr_text" not in columns:
        cursor.execute("ALTER TABLE image_entries ADD COLUMN ocr_text TEXT")
        conn.commit()
        logging.info("Added ocr_text column to image_entries.")
    else:
        logging.info("ocr_text column already exists.")


# ---------------------------------------------------------------------------
# OCR backends
# ---------------------------------------------------------------------------

def load_easyocr():
    try:
        import easyocr
    except ImportError:
        print("EasyOCR not installed. Run: pip install easyocr")
        sys.exit(1)
    logging.info("Loading EasyOCR model (first run may download ~500MB)...")
    t0 = time.time()
    reader = easyocr.Reader(['en'], verbose=False)
    logging.info(f"EasyOCR loaded in {time.time()-t0:.1f}s")
    return reader


def ocr_easyocr(reader, image_path, threshold):
    results = reader.readtext(image_path)
    kept = [text for (_, text, conf) in results if conf >= threshold]
    return " ".join(kept)


def load_tesseract():
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
    except ImportError:
        print("pytesseract not installed. Run: pip install pytesseract")
        sys.exit(1)
    except Exception:
        print("Tesseract binary not found. Run: sudo apt-get install tesseract-ocr")
        sys.exit(1)
    logging.info("Tesseract ready.")
    return None  # no stateful object needed


def ocr_tesseract(_, image_path, threshold):
    import pytesseract
    img = Image.open(image_path).convert("RGB")
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    kept = []
    for i, word in enumerate(data["text"]):
        word = word.strip()
        if not word:
            continue
        conf = data["conf"][i]
        conf_norm = conf / 100.0 if conf >= 0 else 0.0
        if conf_norm >= threshold:
            kept.append(word)
    return " ".join(kept)


def load_paddleocr():
    try:
        from paddleocr import PaddleOCR
    except ImportError:
        print("PaddleOCR not installed. Run: pip install 'paddlepaddle==2.6.2' 'paddleocr==2.7.3' 'numpy<2.0'")
        sys.exit(1)
    logging.info("Loading PaddleOCR model...")
    t0 = time.time()
    ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
    logging.info(f"PaddleOCR loaded in {time.time()-t0:.1f}s")
    return ocr


def ocr_paddleocr(ocr, image_path, threshold):
    result = ocr.ocr(image_path, cls=True)
    if not result or not result[0]:
        return ""
    kept = [text for (_, (text, conf)) in result[0] if conf >= threshold]
    return " ".join(kept)


METHODS = {
    "paddleocr": (load_paddleocr, ocr_paddleocr),
    "easyocr":   (load_easyocr,   ocr_easyocr),
    "tesseract": (load_tesseract, ocr_tesseract),
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def update_comics_ocr(method="paddleocr", remake=False, limit=None,
                      conf=CONFIDENCE_THRESHOLD, yes=False, log_file=None, batch=None):
    log_handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        if os.path.dirname(log_file):
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
        log_handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=log_handlers)

    if not os.path.exists(DB_PATH):
        print(f"Error: '{DB_PATH}' not found. Build the comics database first with build_comicsbase.py.")
        return

    remake_str = " (remaking all)" if remake else " (skipping already OCR'd)"
    limit_str = f", limit {limit}" if limit else ""
    if not yes:
        confirm = input(f"Run OCR [{method}] on comics.db{remake_str}{limit_str}. Continue? (y/n): ").strip().lower()
        if confirm not in ("y", "yes"):
            print("Cancelled.")
            return
    else:
        logging.info(f"Run OCR [{method}] on comics.db{remake_str}{limit_str}.")

    conn = sqlite3.connect(DB_PATH)
    add_ocr_text_column(conn)
    cursor = conn.cursor()

    if remake:
        cursor.execute(
            "SELECT image_id, filename FROM image_entries ORDER BY book_id, page_number"
        )
    else:
        cursor.execute(
            "SELECT image_id, filename FROM image_entries "
            "WHERE ocr_text IS NULL OR ocr_text = '' "
            "ORDER BY book_id, page_number"
        )
    rows = cursor.fetchall()
    if limit:
        rows = rows[:limit]
    if batch:
        rows = rows[:batch]

    logging.info(f"Found {len(rows)} image entries to process.")
    if not rows:
        print("Nothing to process.")
        conn.close()
        return

    images_folder = os.path.join(os.getcwd(), IMAGES_DIR)
    loader, ocr_fn = METHODS[method]
    engine = loader()

    updated = 0
    skipped = 0
    errors = 0

    for i, (image_id, filename) in enumerate(rows):
        if not filename:
            logging.warning(f"[{i+1}/{len(rows)}] {image_id}: no filename, skipping.")
            skipped += 1
            continue

        image_path = os.path.join(images_folder, filename)
        if not os.path.exists(image_path):
            logging.warning(f"[{i+1}/{len(rows)}] {image_id}: file not found ({filename}), skipping.")
            skipped += 1
            continue

        try:
            ocr_text = ocr_fn(engine, image_path, conf)

            cursor.execute(
                "UPDATE image_entries SET ocr_text = ? WHERE image_id = ?",
                (ocr_text, image_id)
            )
            updated += 1

            if updated % 25 == 0:
                conn.commit()
                logging.info(f"  Committed {updated} entries...")

            preview = ocr_text[:80].replace("\n", " ") if ocr_text else "(empty)"
            logging.info(f"[{i+1}/{len(rows)}] {image_id}: \"{preview}\"")

        except Exception as e:
            logging.error(f"[{i+1}/{len(rows)}] {image_id}: error — {e}")
            errors += 1
            continue

    conn.commit()
    conn.close()

    print(f"\nDone. Updated: {updated} | Skipped: {skipped} | Errors: {errors}")

    if batch and updated > 0:
        logging.info(f"Batch of {batch} complete. Re-launching to continue...")
        os.execv(sys.executable, [sys.executable] + sys.argv)

    if updated > 0:
        print("Next: re-run update_comics_embeddings.py --clip remake to rebuild text embeddings.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OCR comic page images and store text in comics.db")
    parser.add_argument("--method", choices=list(METHODS.keys()), default="paddleocr",
                        help="OCR engine to use (default: easyocr)")
    parser.add_argument("--remake", action="store_true",
                        help="Reprocess all images, overwriting existing OCR text")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process this many images (useful for testing)")
    parser.add_argument("--conf", type=float, default=CONFIDENCE_THRESHOLD,
                        help=f"Confidence threshold 0–1 (default: {CONFIDENCE_THRESHOLD})")
    parser.add_argument("-y", "--yes", action="store_true",
                        help="Skip confirmation prompt (for server/overnight runs)")
    parser.add_argument("--log", type=str, default=None,
                        help="Optional path to write log output (e.g. ~/ocr_run.log)")
    parser.add_argument("--batch", type=int, default=None,
                        help="Re-launch process every N images to avoid memory buildup (e.g. --batch 200)")
    args = parser.parse_args()

    update_comics_ocr(method=args.method, remake=args.remake, limit=args.limit,
                      conf=args.conf, yes=args.yes, log_file=args.log, batch=args.batch)
