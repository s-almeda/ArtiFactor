"""
test_ocr_options.py
===================
Compare OCR libraries on real comic page images — NO database writes.

Run this first to see which method gives the best results before
committing to update_comics_ocr.py.

Usage:
    python3 test_ocr_options.py                          # auto-picks 5 images
    python3 test_ocr_options.py img1.jpg img2.jpg        # specific images
    python3 test_ocr_options.py --method easyocr         # only one method
    python3 test_ocr_options.py --n 10                   # auto-pick N images
    python3 test_ocr_options.py --conf 0.3               # confidence threshold

Libraries tested:
    easyocr     - pip install easyocr
                  PyTorch-based, good for stylized/hand-lettered comic fonts.
                  No system binary needed.

    tesseract   - brew install tesseract && pip install pytesseract
                  Classic OCR, very fast, accurate for clean printed text.

Note: activate the project venv first with `venv_pls`
"""

import os
import sys
import time
import argparse
import random
from PIL import Image

IMAGES_DIR = os.path.join(os.path.dirname(__file__), "comic_images")
CONFIDENCE_THRESHOLD = 0.4


# ---------------------------------------------------------------------------
# OCR methods
# ---------------------------------------------------------------------------

def run_easyocr(image_path, threshold):
    try:
        import easyocr
    except ImportError:
        return None, "NOT INSTALLED — run: pip install easyocr"

    t0 = time.time()
    reader = easyocr.Reader(['en'], verbose=False)
    results = reader.readtext(image_path)
    elapsed = time.time() - t0

    detections = []
    for (bbox, text, conf) in results:
        detections.append({"text": text, "conf": conf, "kept": conf >= threshold})

    kept = [d["text"] for d in detections if d["kept"]]
    joined = " ".join(kept)
    return {
        "text": joined,
        "detections": detections,
        "elapsed": elapsed,
    }, None


def run_tesseract(image_path, threshold):
    try:
        import pytesseract
    except ImportError:
        return None, "NOT INSTALLED — run: pip install pytesseract  (also: brew install tesseract)"

    try:
        pytesseract.get_tesseract_version()
    except Exception:
        return None, "Tesseract binary not found — run: brew install tesseract"

    t0 = time.time()
    img = Image.open(image_path).convert("RGB")

    # Get word-level data with confidence scores
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    elapsed = time.time() - t0

    detections = []
    for i, word in enumerate(data["text"]):
        word = word.strip()
        if not word:
            continue
        conf = data["conf"][i]
        # tesseract returns -1 for layout elements, 0-100 for text
        conf_normalized = conf / 100.0 if conf >= 0 else 0.0
        detections.append({"text": word, "conf": conf_normalized, "kept": conf_normalized >= threshold})

    kept = [d["text"] for d in detections if d["kept"]]
    joined = " ".join(kept)
    return {
        "text": joined,
        "detections": detections,
        "elapsed": elapsed,
    }, None


def run_tesseract_preprocessed(image_path, threshold):
    """Tesseract with grayscale + Otsu threshold preprocessing — helps with scanned pages."""
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
    except Exception:
        return None, "Tesseract not available (see above)"

    try:
        import cv2
        import numpy as np
    except ImportError:
        return None, "NOT INSTALLED — run: pip install opencv-python-headless numpy"

    t0 = time.time()
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Otsu binarization
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    pil_img = Image.fromarray(thresh)

    data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT)
    elapsed = time.time() - t0

    detections = []
    for i, word in enumerate(data["text"]):
        word = word.strip()
        if not word:
            continue
        conf = data["conf"][i]
        conf_normalized = conf / 100.0 if conf >= 0 else 0.0
        detections.append({"text": word, "conf": conf_normalized, "kept": conf_normalized >= threshold})

    kept = [d["text"] for d in detections if d["kept"]]
    joined = " ".join(kept)
    return {
        "text": joined,
        "detections": detections,
        "elapsed": elapsed,
    }, None


METHODS = {
    "easyocr":               ("EasyOCR",                      run_easyocr),
    "tesseract":             ("Tesseract (plain)",             run_tesseract),
    "tesseract_preprocessed":("Tesseract (binarized)",        run_tesseract_preprocessed),
}


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

SEP = "=" * 72

def print_result(method_label, result, error, image_path, threshold):
    print(f"\n  [{method_label}]")
    if error:
        print(f"  ERROR: {error}")
        return

    detections = result["detections"]
    kept = [d for d in detections if d["kept"]]
    dropped = [d for d in detections if not d["kept"]]

    print(f"  Time:       {result['elapsed']:.2f}s")
    print(f"  Detections: {len(kept)} kept / {len(dropped)} dropped below conf={threshold}")
    if kept:
        print(f"  Extracted:  {result['text'][:200]}{'...' if len(result['text']) > 200 else ''}")
        if len(kept) <= 15:
            print("  Details:")
            for d in kept:
                print(f"    conf={d['conf']:.2f}  \"{d['text']}\"")
    else:
        print("  Extracted:  (nothing above threshold)")
    if dropped:
        dropped_preview = ", ".join(f"\"{d['text']}\"({d['conf']:.2f})" for d in dropped[:5])
        if len(dropped) > 5:
            dropped_preview += f" ... +{len(dropped)-5} more"
        print(f"  Dropped:    {dropped_preview}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compare OCR options on comic images (no DB writes)")
    parser.add_argument("images", nargs="*", help="Image paths to test (default: auto-pick from comic_images/)")
    parser.add_argument("--method", choices=list(METHODS.keys()),
                        help="Only run one method (default: all)")
    parser.add_argument("--n", type=int, default=5,
                        help="Number of images to auto-pick if none specified (default: 5)")
    parser.add_argument("--conf", type=float, default=CONFIDENCE_THRESHOLD,
                        help=f"Confidence threshold 0-1 (default: {CONFIDENCE_THRESHOLD})")
    args = parser.parse_args()

    threshold = args.conf

    # Resolve image paths
    image_paths = []
    if args.images:
        for p in args.images:
            if not os.path.isabs(p):
                # try relative to comic_images/
                candidate = os.path.join(IMAGES_DIR, p)
                p = candidate if os.path.exists(candidate) else p
            if os.path.exists(p):
                image_paths.append(p)
            else:
                print(f"Warning: image not found: {p}")
    else:
        if not os.path.isdir(IMAGES_DIR):
            print(f"Error: comic_images/ directory not found at {IMAGES_DIR}")
            sys.exit(1)
        all_images = sorted(
            [os.path.join(IMAGES_DIR, f) for f in os.listdir(IMAGES_DIR)
             if f.lower().endswith((".jpg", ".jpeg", ".png")) and "thumb" not in f]
        )
        if not all_images:
            print(f"Error: no images found in {IMAGES_DIR}")
            sys.exit(1)
        # Pick spread across the dataset, not just the first N
        step = max(1, len(all_images) // args.n)
        image_paths = all_images[::step][:args.n]
        print(f"Auto-selected {len(image_paths)} images from {len(all_images)} total.")

    methods_to_run = {k: v for k, v in METHODS.items()
                      if args.method is None or k == args.method}

    print(f"\nConfidence threshold: {threshold}")
    print(f"Methods: {', '.join(methods_to_run.keys())}")
    print(f"Images:  {len(image_paths)}")

    # Initialize EasyOCR reader once outside the loop (expensive)
    easyocr_reader = None
    if "easyocr" in methods_to_run:
        try:
            import easyocr
            print("\nLoading EasyOCR model (first run may download ~500MB)...")
            t0 = time.time()
            easyocr_reader = easyocr.Reader(['en'], verbose=False)
            print(f"EasyOCR loaded in {time.time()-t0:.1f}s")
        except ImportError:
            pass

    for image_path in image_paths:
        filename = os.path.basename(image_path)
        print(f"\n{SEP}")
        print(f"IMAGE: {filename}")
        print(SEP)

        for method_key, (method_label, method_fn) in methods_to_run.items():
            # Inject pre-loaded reader for easyocr to avoid reloading per image
            if method_key == "easyocr" and easyocr_reader is not None:
                t0 = time.time()
                try:
                    raw = easyocr_reader.readtext(image_path)
                    elapsed = time.time() - t0
                    detections = [
                        {"text": text, "conf": conf, "kept": conf >= threshold}
                        for (bbox, text, conf) in raw
                    ]
                    kept = [d["text"] for d in detections if d["kept"]]
                    result = {"text": " ".join(kept), "detections": detections, "elapsed": elapsed}
                    error = None
                except Exception as e:
                    result, error = None, str(e)
            else:
                result, error = method_fn(image_path, threshold)

            print_result(method_label, result, error, image_path, threshold)

    print(f"\n{SEP}")
    print("Done. Review the results above and choose the best method.")
    print("Then run update_comics_ocr.py with your chosen method.")


if __name__ == "__main__":
    main()
