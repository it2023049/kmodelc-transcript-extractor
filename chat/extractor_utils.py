"""Shared helper functions used by both chat screenshot extractors."""

import csv
import io
import json
import re
from pathlib import Path
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Sequence, Tuple, Set
from datetime import datetime

import cv2
import numpy as np
import ollama
from PyPDF2 import PdfReader

Box = Tuple[int, int, int, int]
ScreenCrop = Tuple[int, int, int, int, np.ndarray]

def extract_text_from_report(report_path: str) -> str:
    """Reads plain-text or PDF case-report content."""
    path = Path(report_path)

    if path.suffix.lower() == ".txt":
        for enc in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                return path.read_text(encoding=enc)
            except UnicodeDecodeError:
                continue
        return path.read_text(errors="ignore")

    if path.suffix.lower() == ".pdf":
        try:
            reader = PdfReader(str(path))
            pages = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
            return "\n".join(pages)
        except Exception as e:
            print(f"[WARNING] Could not read PDF: {e}")
            return ""

    print("[WARNING] Unsupported report type. Use PDF or TXT.")
    return ""

def extract_year_from_report(
    report_text: str,
    default_year: Optional[int] = None,
) -> int:
    """Finds the first report year, falling back to the current calendar year."""
    years = re.findall(r"\b(20\d{2})\b", report_text)
    if not years:
        return int(default_year) if default_year is not None else datetime.now().year

    # Usually the first report/timeline year is the relevant year.
    return int(years[0])

def parse_grid(grid: Optional[str]) -> Optional[Tuple[int, int]]:
    """Parses a regular collage grid specification."""
    if not grid:
        return None

    m = re.match(r"^(\d+)x(\d+)$", grid.strip().lower())
    if not m:
        raise ValueError("--grid must be like 2x1, 3x2, etc.")

    cols, rows = int(m.group(1)), int(m.group(2))
    if cols <= 0 or rows <= 0:
        raise ValueError("--grid values must be positive.")

    return cols, rows

def parse_layout(layout: Optional[str]) -> Optional[List[int]]:
    """Parses an uneven collage row-layout specification."""
    if not layout:
        return None

    try:
        values = [int(x.strip()) for x in layout.split(",") if x.strip()]
    except ValueError:
        raise ValueError("--layout must be like 2,3 or 1,2,3.")

    if not values or any(v <= 0 for v in values):
        raise ValueError("--layout values must be positive.")

    return values

def trim_white_border(image: np.ndarray, threshold: int = 245, pad: int = 0) -> np.ndarray:
    """Removes near-white outer borders from an image crop."""
    if image.size == 0:
        return image

    white = np.all(image >= threshold, axis=2)
    content = ~white
    ys, xs = np.where(content)

    if len(xs) == 0 or len(ys) == 0:
        return image

    h, w = image.shape[:2]
    x1 = max(0, int(xs.min()) - pad)
    x2 = min(w, int(xs.max()) + 1 + pad)
    y1 = max(0, int(ys.min()) - pad)
    y2 = min(h, int(ys.max()) + 1 + pad)

    return image[y1:y2, x1:x2]

def ranges_from_indices(indices: np.ndarray) -> List[Tuple[int, int]]:
    """Converts consecutive index values into half-open ranges."""
    if len(indices) == 0:
        return []

    values = [int(x) for x in indices]
    ranges = []

    start = prev = values[0]
    for value in values[1:]:
        if value == prev + 1:
            prev = value
        else:
            ranges.append((start, prev + 1))
            start = prev = value

    ranges.append((start, prev + 1))
    return ranges

def find_separator_bands(
    image: np.ndarray,
    axis: str,
    white_threshold: int = 235,
    ratio_threshold: float = 0.72,
    min_band_size: int = 3,
) -> List[Tuple[int, int]]:
    """Finds likely white gutter bands along one image axis."""
    # Notes:
    # axis='x' -> vertical separator columns.
    # axis='y' -> horizontal separator rows.
    if image.size == 0:
        return []

    white = np.all(image >= white_threshold, axis=2)

    if axis == "x":
        ratio = white.mean(axis=0)
    elif axis == "y":
        ratio = white.mean(axis=1)
    else:
        raise ValueError("axis must be 'x' or 'y'")

    candidates = np.where(ratio >= ratio_threshold)[0]
    bands = ranges_from_indices(candidates)

    return [(a, b) for a, b in bands if (b - a) >= min_band_size]

def split_segments_by_bands(
    length: int,
    bands: List[Tuple[int, int]],
    min_size: int
) -> List[Tuple[int, int]]:
    """Splits a dimension into content segments around separator bands."""
    if not bands:
        return [(0, length)]

    segments = []
    cur = 0

    for a, b in bands:
        if a - cur >= min_size:
            segments.append((cur, a))
        cur = b

    if length - cur >= min_size:
        segments.append((cur, length))

    return segments

def manual_grid_split(image: np.ndarray, grid: str) -> List[ScreenCrop]:
    """Splits an image using a user-specified regular grid."""
    cols, rows = parse_grid(grid)
    h, w = image.shape[:2]
    crops = []

    cell_w = w / cols
    cell_h = h / rows

    for r in range(rows):
        for c in range(cols):
            x1 = int(c * cell_w)
            x2 = int((c + 1) * cell_w)
            y1 = int(r * cell_h)
            y2 = int((r + 1) * cell_h)

            crop = trim_white_border(image[y1:y2, x1:x2])
            crops.append((x1, y1, x2 - x1, y2 - y1, crop))

    return crops

def manual_layout_split(image: np.ndarray, layout: str) -> List[ScreenCrop]:
    """Splits an image using a user-specified uneven row layout."""
    # Notes:
    # top row: 2 screenshots
    # bottom row: 3 screenshots
    row_counts = parse_layout(layout)
    h, w = image.shape[:2]
    crops = []

    row_h = h / len(row_counts)

    for r, count in enumerate(row_counts):
        y1 = int(r * row_h)
        y2 = int((r + 1) * row_h)
        col_w = w / count

        for c in range(count):
            x1 = int(c * col_w)
            x2 = int((c + 1) * col_w)

            crop = trim_white_border(image[y1:y2, x1:x2])
            crops.append((x1, y1, x2 - x1, y2 - y1, crop))

    return crops

def auto_split_by_white_gutters(image: np.ndarray) -> List[ScreenCrop]:
    """Automatically splits collages using visible white gutters."""
    # Notes:
    # For important/known uneven layouts, prefer --layout 2,3.
    img = trim_white_border(image)
    h, w = img.shape[:2]

    if h == 0 or w == 0:
        return []

    min_h = max(180, int(h * 0.18))
    min_w = max(140, int(w * 0.13))

    horizontal_bands = find_separator_bands(img, axis="y")
    row_segments = split_segments_by_bands(h, horizontal_bands, min_h)

    crops = []

    for y1, y2 in row_segments:
        row_img = img[y1:y2, :]
        vertical_bands = find_separator_bands(row_img, axis="x")
        col_segments = split_segments_by_bands(w, vertical_bands, min_w)

        for x1, x2 in col_segments:
            crop = trim_white_border(row_img[:, x1:x2])
            ch, cw = crop.shape[:2]

            if cw >= min_w and ch >= min_h:
                crops.append((x1, y1, x2 - x1, y2 - y1, crop))

    if len(crops) <= 1:
        return [(0, 0, w, h, img)]

    return sort_screen_crops(crops)

def contour_fallback_split(image: np.ndarray) -> List[ScreenCrop]:
    """Splits an image using contour boxes when gutters fail."""
    img = trim_white_border(image)
    h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 200)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(
        dilated,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []

    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)

        if bw > w * 0.18 and bh > h * 0.25 and bh > bw * 0.8:
            boxes.append((x, y, bw, bh))

    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)

    kept = []
    for box in boxes:
        x, y, bw, bh = box
        cx, cy = x + bw / 2, y + bh / 2

        contained = False
        for kx, ky, kw, kh in kept:
            if kx <= cx <= kx + kw and ky <= cy <= ky + kh:
                contained = True
                break

        if not contained:
            kept.append(box)

    crops = []

    for x, y, bw, bh in kept:
        pad = 5
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + bw + pad)
        y2 = min(h, y + bh + pad)

        crops.append((x1, y1, x2 - x1, y2 - y1, img[y1:y2, x1:x2]))

    if not crops:
        return [(0, 0, w, h, img)]

    return sort_screen_crops(crops)

def sort_screen_crops(crops: List[ScreenCrop]) -> List[ScreenCrop]:
    """Orders screen crops from top-left to bottom-right."""
    if not crops:
        return []

    boxes = sorted(crops, key=lambda item: item[1])
    heights = [item[3] for item in boxes]
    threshold = max(40, int(np.median(heights) * 0.25))

    rows = []
    current = [boxes[0]]

    for item in boxes[1:]:
        if abs(item[1] - current[-1][1]) <= threshold:
            current.append(item)
        else:
            rows.append(current)
            current = [item]

    rows.append(current)

    ordered = []
    for row in rows:
        ordered.extend(sorted(row, key=lambda item: item[0]))

    return ordered

def minimal_ocr_clean(text: str) -> str:
    """Applies minimal cleanup to OCR text."""
    text = text.replace("\u200b", " ")
    text = text.replace("\ufeff", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

def polygon_to_xywh(poly) -> Box:
    """Converts an OCR polygon into an x/y/width/height box."""
    xs = [int(p[0]) for p in poly]
    ys = [int(p[1]) for p in poly]

    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)

    return x1, y1, x2 - x1, y2 - y1

def looks_like_date(text: str) -> bool:
    """Checks whether text looks like a chat date label."""
    t = text.strip()

    if re.fullmatch(
        r"(?:Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|Sep|Sept|September|Oct|October|Nov|November|Dec|December)\s+\d{1,2}",
        t,
        flags=re.I,
    ):
        return True

    if re.search(
        r"(?:Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|Sep|Sept|September|Oct|October|Nov|November|Dec|December)\s+\d{1,2},?\s+\d{4}",
        t,
        flags=re.I,
    ):
        return True

    return False

def looks_like_time(text: str) -> bool:
    """Checks whether text looks like an HH:MM time token."""
    t = text.strip()
    t = re.sub(r"\s*(vi|v|✓|✔|✔✔)+\s*$", "", t, flags=re.I)
    t = t.replace("*", ":").replace(",", ":").replace(";", ":").replace(".", ":")
    return bool(re.fullmatch(r"\d{1,2}:\d{2}", t))

def looks_like_date_or_time(text: str) -> bool:
    """Checks whether text is a date label or time token."""
    return looks_like_date(text) or looks_like_time(text)

def normalize_visible_time_token(text: str) -> Optional[str]:
    """Normalizes noisy OCR time text into HH:MM format."""
    t = text.strip()
    t = re.sub(r"\s*(vi|v|✓|✔|✔✔)+\s*$", "", t, flags=re.I)
    t = t.replace("*", ":").replace(",", ":").replace(";", ":").replace(".", ":")

    m = re.fullmatch(r"(\d{1,2}):(\d{2})", t)
    if not m:
        return None

    hh = int(m.group(1))
    mm = int(m.group(2))

    if not (0 <= hh <= 23 and 0 <= mm <= 59):
        return None

    return f"{hh:02d}:{mm:02d}"

def parse_ocr_lines(ocr_data: str) -> List[Dict[str, str]]:
    """Parses positioned OCR debug text into row dictionaries."""
    pattern = re.compile(
        r"\[BLOCK\s+(?P<block>\d+)\]\s+"
        r"\[POS=(?P<pos>LEFT|RIGHT|CENTER)\]\s+"
        r"\[X=(?P<x>\d+)\]\s+"
        r"\[Y=(?P<y>\d+)\]\s+"
        r"\[W=(?P<w>\d+)\]\s+"
        r"\[H=(?P<h>\d+)\]"
        r"(?:\s+\[CONF=[^\]]+\])?\s+"
        r"(?P<text>.*)$",
        re.I,
    )

    current_screen = None
    rows = []

    for line in ocr_data.splitlines():
        sm = re.match(r"\[SCREEN\s+(\d+)\]", line.strip(), flags=re.I)
        if sm:
            current_screen = int(sm.group(1))
            continue

        bm = pattern.search(line.strip())
        if not bm:
            continue

        rows.append({
            "screen": str(current_screen or ""),
            "block": bm.group("block"),
            "pos": bm.group("pos").upper(),
            "x": bm.group("x"),
            "y": bm.group("y"),
            "w": bm.group("w"),
            "h": bm.group("h"),
            "text": bm.group("text").strip(),
        })

    return rows

def extract_allowed_times_from_ocr(screen_ocr: str) -> Set[str]:
    """Collects visible bubble times from OCR output."""
    # Notes:
    # Ignores top status bar time by requiring it to appear after the date separator/header area.
    rows = parse_ocr_lines(screen_ocr)
    allowed = set()

    date_y = None
    for row in rows:
        if looks_like_date(row["text"]):
            try:
                date_y = int(row["y"])
            except ValueError:
                pass

    for row in rows:
        try:
            y = int(row["y"])
        except ValueError:
            continue

        # Ignore status/header time near the top.
        if date_y is not None and y <= date_y:
            continue
        if date_y is None and y < 250:
            continue

        t = normalize_visible_time_token(row["text"])
        if t:
            allowed.add(t)

    return allowed

def month_to_number(month: str) -> Optional[int]:
    """Maps an English month name to its numeric month value."""
    m = month.strip().lower()[:3]
    table = {
        "jan": 1,
        "feb": 2,
        "mar": 3,
        "apr": 4,
        "may": 5,
        "jun": 6,
        "jul": 7,
        "aug": 8,
        "sep": 9,
        "oct": 10,
        "nov": 11,
        "dec": 12,
    }
    return table.get(m)

def strip_code_fences(text: str) -> str:
    """Removes markdown code fences around model output."""
    text = text.strip()
    text = re.sub(r"^```(?:csv|json|text)?", "", text, flags=re.I).strip()
    text = re.sub(r"```$", "", text).strip()
    return text

def extract_json_object(text: str) -> str:
    """Extracts the outermost JSON object from model text."""
    text = strip_code_fences(text)
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        return text

    return text[start:end + 1]

def ollama_chat_text(model: str, prompt: str) -> str:
    """Sends a deterministic text-only prompt to Ollama."""
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0},
    )
    return response["message"]["content"].strip()

def ollama_chat_screen(
    model: str,
    prompt: str,
    image_path: str,
    use_vision: bool = True
) -> str:
    """Sends a deterministic vision or OCR-only prompt to Ollama."""
    message = {
        "role": "user",
        "content": prompt,
    }

    if use_vision:
        message["images"] = [image_path]

    try:
        response = ollama.chat(
            model=model,
            messages=[message],
            options={"temperature": 0},
        )
        return response["message"]["content"].strip()

    except Exception as e:
        if use_vision:
            print(f"[WARNING] Vision call failed for {image_path}: {e}")
            print("[WARNING] Retrying with OCR text only. Emojis may not be reliable.")
            return ollama_chat_text(model, prompt)

        raise

def normalize_phone(value: str) -> str:
    """Keeps only digits from a phone/contact string."""
    return re.sub(r"\D+", "", value or "")

def normalize_name(value: str) -> str:
    """Lowercases and normalizes whitespace in a name."""
    return re.sub(r"\s+", " ", value or "").strip().lower()

def clean_name(value: str) -> str:
    """Cleans display names before matching or output.

    Both complete and truncated trailing parenthetical aliases are removed.
    This prevents malformed values such as ``Alice Example (A. Example`` from
    becoming a distinct identity while preserving the canonical visible name.
    """
    value = str(value or "").strip()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"\s*\([^)]*\)\s*$", "", value).strip()
    value = re.sub(r"\s*\([^)]*$", "", value).strip()
    return value

def same_name(a: str, b: str) -> bool:
    """Compares two names after normalization."""
    return normalize_name(clean_name(a)) == normalize_name(clean_name(b))

def name_in_text(name: str, text: str) -> bool:
    """Checks whether a normalized name appears inside normalized text."""
    name_norm = normalize_name(clean_name(name))
    text_norm = normalize_name(text)

    if not name_norm or not text_norm:
        return False

    return name_norm in text_norm

def build_actor_prompt(report_text: str) -> str:
    """Builds the fallback actor-extraction prompt."""
    return f"""
Extract the chat actors from this case report.

CASE REPORT:
{report_text}

Return only JSON with this structure:
{{
  "victim": "victim full name",
  "participants": [
    {{"name":"full name", "role":"victim or suspect", "contact_numbers":["..."]}}
  ]
}}

Rules:
1. Include the victim/complainant.
2. Include suspects/scammers and their contact numbers if present.
3. Do not output explanations.
"""

def infer_report_actors(report_text: str, model: str) -> Dict:
    """Extracts victim and suspect actors deterministically from the report."""
    # Notes:
    # We do NOT use the LLM here because wrong actor JSON breaks side_map.
    # Keeps the same function signature so the rest of the code does not change.
    participants = []

    # -------------------------
    # Victim
    # -------------------------
    victim = ""

    victim_patterns = [
        r"VICTIM\s*/\s*COMPLAINANT:.*?Full Name:\s*([^\n\r•]+)",
        r"Full Name:\s*([^\n\r•]+)",
        r"Target/Victim:\s*([^\n\r•]+)",
    ]

    for pattern in victim_patterns:
        m = re.search(pattern, report_text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            victim = clean_name(m.group(1))
            break

    if victim:
        participants.append({
            "name": victim,
            "role": "victim",
            "contact_numbers": []
        })

    # -------------------------
    # Suspects + nearby contact numbers
    # -------------------------
    suspect_pattern = re.compile(
        r"Suspect\s*\d+:\s*([^\n\r]+)(.*?)(?=(?:•\s*)?Suspect\s*\d+:|2\.\s*BACKGROUND|3\.\s*TECHNICAL|$)",
        flags=re.IGNORECASE | re.DOTALL
    )

    phone_pattern = re.compile(r"\+\d[\d\s().-]{5,}\d")

    for m in suspect_pattern.finditer(report_text):
        suspect_name = clean_name(m.group(1))
        suspect_block = m.group(2)

        phones = phone_pattern.findall(suspect_block)
        phones = [p.strip() for p in phones]

        if suspect_name:
            participants.append({
                "name": suspect_name,
                "role": "suspect",
                "contact_numbers": phones
            })

    # -------------------------
    # Fallback: Scammer Names Used
    # -------------------------
    if not any(p["role"] == "suspect" for p in participants):
        m = re.search(r"Scammer Names Used:\s*([^\n\r]+)", report_text, flags=re.IGNORECASE)
        if m:
            names = [clean_name(x) for x in re.split(r",| and ", m.group(1))]
            for name in names:
                if name and not any(same_name(name, p["name"]) for p in participants):
                    participants.append({
                        "name": name,
                        "role": "suspect",
                        "contact_numbers": []
                    })

    return {
        "victim": victim,
        "participants": participants
    }

def infer_report_actors_fallback(report_text: str) -> Dict:
    """Extracts actors with simpler regex fallbacks."""
    victim = ""
    participants = []

    m = re.search(r"Full Name:\s*([^\n•]+)", report_text, flags=re.I)
    if m:
        victim = clean_name(m.group(1))

    if victim:
        participants.append({
            "name": victim,
            "role": "victim",
            "contact_numbers": [],
        })

    for sm in re.finditer(r"Suspect\s*\d+:\s*([^\n]+)", report_text, flags=re.I):
        name = clean_name(sm.group(1))
        if name:
            participants.append({
                "name": name,
                "role": "suspect",
                "contact_numbers": [],
            })

    # Attach nearby phone numbers as a rough fallback.
    phones = re.findall(r"\+\d[\d\s().-]{5,}\d", report_text)
    suspect_i = 0
    for p in participants:
        if p["role"] == "suspect" and suspect_i < len(phones):
            p["contact_numbers"] = [phones[suspect_i]]
            suspect_i += 1

    return {
        "victim": victim,
        "participants": participants,
    }

def build_side_evidence(ocr_data: str) -> str:
    """Summarizes OCR text and phone-like strings by side."""
    rows = parse_ocr_lines(ocr_data)

    side_texts = {
        "LEFT": [],
        "RIGHT": [],
        "CENTER": [],
    }

    side_phones = {
        "LEFT": [],
        "RIGHT": [],
        "CENTER": [],
    }

    phone_pattern = re.compile(r"\+?\d[\d\s().-]{5,}\d")

    for row in rows:
        pos = row["pos"]
        text = row["text"]

        side_texts[pos].append(text)

        for phone in phone_pattern.findall(text):
            side_phones[pos].append(phone.strip())

    lines = ["SIDE EVIDENCE SUMMARY"]

    for side in ["LEFT", "RIGHT", "CENTER"]:
        lines.append(f"\n{side} phone/contact-like strings:")

        phones = side_phones[side][:12]
        if phones:
            lines.extend(f"- {p}" for p in phones)
        else:
            lines.append("- none")

        lines.append(f"{side} sample OCR texts:")
        for text in side_texts[side][:18]:
            lines.append(f"- {text}")

    return "\n".join(lines)

def force_date_and_year(time_value: str, visible_date: str, default_year: int) -> str:
    """Normalizes timestamps and forces the visible screenshot date."""
    # Notes:
    # Screen extractors first normalize to DD/MM/YYYY HH:MM. A later deterministic
    # post-processing step adds seconds for the final transcript.
    time_value = str(time_value).strip().strip('"')
    time_value = time_value.replace(";", ":")
    time_value = re.sub(r"\s+", " ", time_value)

    # Accept both:
    # DD/MM/YYYY, HH:MM
    # DD/MM/YYYY HH:MM
    # DD/MM/YYYY HH:MM:SS
    m = re.search(
        r"(\d{1,2})/(\d{1,2})/(20\d{2})\s*,?\s*(\d{1,2})[:.;,*](\d{2})(?::(\d{2}))?",
        time_value
    )

    if m:
        hh = int(m.group(4))
        mm = int(m.group(5))

        if visible_date:
            return f"{visible_date} {hh:02d}:{mm:02d}"

        dd = int(m.group(1))
        mo = int(m.group(2))
        return f"{dd:02d}/{mo:02d}/{default_year} {hh:02d}:{mm:02d}"

    # Fallback: if only HH:MM exists, combine with visible date.
    tv = normalize_visible_time_token(time_value)
    if tv and visible_date:
        return f"{visible_date} {tv}"

    return time_value

def extract_hhmm_from_full_time(time_value: str) -> Optional[str]:
    # Accept both:
    # DD/MM/YYYY, HH:MM
    # DD/MM/YYYY HH:MM
    # DD/MM/YYYY HH:MM:SS
    """Extracts the HH:MM part from a full timestamp."""
    m = re.search(r"\s*,?\s*(\d{1,2})[:.;,*](\d{2})(?::(\d{2}))?\s*$", str(time_value or ""))
    if not m:
        return None

    hh = int(m.group(1))
    mm = int(m.group(2))

    if not (0 <= hh <= 23 and 0 <= mm <= 59):
        return None

    return f"{hh:02d}:{mm:02d}"


def _datetime_from_transcript_time(time_value: str):
    """Parse transcript timestamps with or without seconds."""
    from datetime import datetime

    text = str(time_value or "").strip()
    formats = (
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y, %H:%M:%S",
        "%d/%m/%Y,%H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%d/%m/%Y, %H:%M",
        "%d/%m/%Y,%H:%M",
    )
    for fmt in formats:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None

def _write_side_rows(rows: List[List[str]]) -> str:
    """Write Time/Side/Message rows to a side CSV string."""
    out = io.StringIO()
    writer = csv.writer(out, quoting=csv.QUOTE_ALL, lineterminator="\n")
    writer.writerow(["Time", "Side", "Message"])
    for row in rows:
        writer.writerow(row)
    return out.getvalue().strip() + "\n"

def add_sequential_seconds_to_side_csv(side_csv: str, step_seconds: int = 1) -> str:
    """Add deterministic seconds to a Facebook/Messenger side CSV.

    Messenger screenshots often expose only one screen-level HH:MM timestamp.
    The first row keeps that minute as :00. Subsequent rows advance by a small,
    deterministic number of seconds while preserving row order. This avoids
    random/non-reproducible output and does not change message text or sides.
    """
    from datetime import timedelta

    rows = _side_csv_rows(side_csv)
    if not rows:
        return side_csv

    step_seconds = max(1, int(step_seconds or 1))
    out_rows: List[List[str]] = []
    previous_dt = None

    for row in rows:
        dt = _datetime_from_transcript_time(row[0])
        if dt is None:
            out_rows.append(row)
            continue

        dt = dt.replace(second=0, microsecond=0)
        if previous_dt is not None and dt <= previous_dt:
            dt = previous_dt + timedelta(seconds=step_seconds)

        out_rows.append([dt.strftime("%d/%m/%Y %H:%M:%S"), row[1], row[2]])
        previous_dt = dt

    return _write_side_rows(out_rows)

def ensure_zero_seconds_side_csv(side_csv: str) -> str:
    """Add :00 seconds to every Viber side-CSV timestamp."""
    rows = _side_csv_rows(side_csv)
    if not rows:
        return side_csv

    out_rows: List[List[str]] = []
    for row in rows:
        dt = _datetime_from_transcript_time(row[0])
        if dt is None:
            out_rows.append(row)
            continue
        dt = dt.replace(second=0, microsecond=0)
        out_rows.append([dt.strftime("%d/%m/%Y %H:%M:%S"), row[1], row[2]])

    return _write_side_rows(out_rows)

def strip_emojis(text: str) -> str:
    """Removes emoji and symbol ranges from message text."""
    # Notes:
    # This is intentionally generic and not tied to specific emoji characters.
    emoji_re = re.compile(
        "["
        "\U0001F1E6-\U0001F1FF"  # flags
        "\U0001F300-\U0001F5FF"  # symbols/pictographs
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F680-\U0001F6FF"  # transport/map
        "\U0001F700-\U0001F77F"
        "\U0001F780-\U0001F7FF"
        "\U0001F800-\U0001F8FF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FAFF"
        "\u2600-\u26FF"
        "\u2700-\u27BF"
        "]+",
        flags=re.UNICODE,
    )
    text = emoji_re.sub("", str(text or ""))
    # Emoji skin/variation leftovers can remain after pictograph removal.
    text = text.replace("\ufe0f", "").replace("\ufe0e", "").replace("\u200d", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize_text_for_side_overlap(text: str) -> Set[str]:
    """Tokenizes text for fuzzy side-overlap matching."""
    # Notes:
    # This is not used as final transcript text.
    t = str(text or "").lower()
    t = re.sub(r"\b([a-z]+)'\$", r"\1s", t)
    t = t.replace("|", "i").replace("!", "i")
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return {w for w in t.split() if len(w) >= 2}

def count_data_rows(side_csv: str) -> int:
    """Counts non-header data rows in a side CSV."""
    rows = list(csv.reader(io.StringIO(strip_code_fences(side_csv))))
    return sum(1 for r in rows if r and len(r) >= 3 and r[0].strip().lower() != "time")

def remove_leaked_time_tokens_from_message(message: str) -> str:
    """Remove standalone HH:MM-like tokens accidentally copied into message text."""
    text = str(message or "")
    # Remove times such as 08:16, 11,01 or 10.32 when they are standalone OCR leaks.
    text = re.sub(r"(?<![A-Za-z0-9])\d{1,2}[:.,;]\d{2}(?![A-Za-z0-9])", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def looks_like_noisy_ocr_text(message: str) -> bool:
    """Detect OCR-heavy text that should not be trusted as final transcript text."""
    text = str(message or "")
    if not text.strip():
        return True

    # Standalone leaked timestamps are a strong signal that OCR text crossed bubble boundaries.
    if re.search(r"(?<![A-Za-z0-9])\d{1,2}[:.,;]\d{2}(?![A-Za-z0-9])", text):
        return True

    # Too many OCR-only artifacts for one message.
    artifact_patterns = [
        r"\bTII\w*\b",
        r"\bI{2}l\b",
        r"\b1['’]?I[lI]\b",
        r"\bJi00\b",
        r"\$",
        r"\|",
        r"__",
        r"=",
        r"\byoU\b",
        r"\bsO\b",
    ]
    hits = sum(1 for pat in artifact_patterns if re.search(pat, text))
    return hits >= 2

def conservative_clean_message_text(message: str) -> str:
    """Apply generic OCR cleanup without semantic rewriting or case-specific replacements."""
    msg = str(message or "").strip()

    msg = msg.replace('\\"', '"')
    msg = msg.replace("“", '"').replace("”", '"')
    msg = msg.replace("‘", "'").replace("’", "'")
    msg = msg.replace("\u200b", " ").replace("\ufeff", " ")

    msg = remove_leaked_time_tokens_from_message(msg)

    pronoun_verbs = (
        r"am|was|will|can|can't|cannot|need|have|think|feel|want|would|could|"
        r"should|do|don't|dont|love|trust|promise|know|hope|wish|believe|already|"
        r"just|may|must|might|see|ask|tell|try|send|keep|call|go|complete|look|admire"
    )
    msg = re.sub(
        rf"(?i)(^|[\s,.;:!?])\|\s+({pronoun_verbs})\b",
        lambda m: f"{m.group(1)}I {m.group(2)}",
        msg,
    )

    compact_i = {
        "just": "just",
        "will": "will",
        "need": "need",
        "love": "love",
        "may": "may",
        "feel": "feel",
        "already": "already",
        "cant": "can't",
        "can't": "can't",
        "can": "can",
        "have": "have",
        "think": "think",
        "want": "want",
        "would": "would",
        "could": "could",
        "should": "should",
        "promise": "promise",
        "believe": "believe",
        "hope": "hope",
        "see": "see",
    }
    for compact, word in compact_i.items():
        msg = re.sub(rf"\bI{re.escape(compact)}\b", f"I {word}", msg, flags=re.IGNORECASE)

    # Generic missing-prefix and OCR-boundary polish.
    # These are grammar/typography repairs only; they do not mention case data,
    # actor names, locations, amounts, or any known transcript sentence.
    msg = re.sub(r"\b(would|could|should|will)\s+be\s+was\s+", r"\1 be ", msg, flags=re.IGNORECASE)
    msg = re.sub(r"^will\s+([a-z])", r"I will \1", msg)
    msg = re.sub(
        r"^feel\s+(can|could|will|would|should|must|may|might|am|was)\b",
        r"I feel I \1",
        msg,
    )
    msg = re.sub(r"^trust\s+you\b", "I trust you", msg)
    msg = re.sub(
        r"^thinking\s+(it|this|that)\s+(would|will|could|should|is|was)\b",
        r"I think \1 \2",
        msg,
    )
    msg = re.sub(r"\bThey\s+The\b", "The", msg)
    msg = re.sub(r"\bGreatl(?=\s+[A-Z])", "Great!", msg)

    msg = re.sub(r"\bI\s*m\b", "I'm", msg, flags=re.IGNORECASE)
    msg = re.sub(r"\bIm\b", "I'm", msg)
    msg = re.sub(r"\bTm\b", "I'm", msg)
    msg = re.sub(r"\bI\s*ve\b", "I've", msg, flags=re.IGNORECASE)
    msg = re.sub(r"\bFve\b", "I've", msg)
    msg = re.sub(r"\bI\s*ll\b", "I'll", msg, flags=re.IGNORECASE)
    msg = re.sub(r"\bIll\b", "I'll", msg)
    msg = re.sub(r"\bIIl\b", "I'll", msg)
    msg = re.sub(r"\b1['’]?I[lI]\b", "I'll", msg)
    msg = re.sub(r"\b1['’]?ll\b", "I'll", msg, flags=re.IGNORECASE)
    msg = re.sub(r"\bTII\s*keep\b", "I'll keep", msg, flags=re.IGNORECASE)

    msg = re.sub(r"\b([A-Za-z]+)'\$\b", r"\1's", msg)
    msg = re.sub(r"\bit\$\b", "it's", msg, flags=re.IGNORECASE)
    msg = re.sub(r"\bthat\$\b", "that's", msg, flags=re.IGNORECASE)

    msg = re.sub(r"\b([Ii])t\s*[\"]+\s*s\b", "It's", msg)
    msg = re.sub(r"\b([Tt])hat\s*[\"]+\s*s\b", "That's", msg)
    msg = re.sub(r"\b([Dd])on\s*[\"]+\s*t\b", "don't", msg)
    msg = re.sub(r"\b([Cc])an\s*[\"]+\s*t\b", "can't", msg)

    msg = re.sub(r"\bIcant\b", "I can't", msg, flags=re.IGNORECASE)
    msg = re.sub(r"\bIcan't\b", "I can't", msg, flags=re.IGNORECASE)
    msg = re.sub(r"\bdont\b", "don't", msg, flags=re.IGNORECASE)
    msg = re.sub(r"\byoure\b", "you're", msg, flags=re.IGNORECASE)
    msg = re.sub(r"\bthats\b", "that's", msg, flags=re.IGNORECASE)
    msg = re.sub(r"\bits\b", "it's", msg)

    msg = re.sub(r"\bsO\b", "so", msg)
    msg = re.sub(r"\byoU\b", "you", msg)
    msg = re.sub(r"\bJi00\b", "I", msg)
    msg = re.sub(r"\bIı\b", "I", msg)

    msg = msg.replace("=", " ")
    msg = msg.replace("__", "...")
    msg = msg.replace("_", "")

    label_like = (
        r"name|country|city|option|account|iban|reference|mtcn|phone|email|"
        r"amount|details|receiver|sender|beneficiary|bank|address|code|number|date|time"
    )
    has_structured_label = re.search(rf"\b({label_like})\s*:", msg, flags=re.IGNORECASE)
    if not has_structured_label:
        # Low-risk semicolon OCR repairs in conversational text.
        msg = re.sub(
            r";\s+(my|your|our|their)\b",
            lambda m: ", " + m.group(1),
            msg,
            flags=re.IGNORECASE,
        )
        msg = re.sub(r";\s+([A-Z][a-z]{2,})([.!?])", r", \1\2", msg)
        msg = re.sub(
            r";\s+(?=(I|I'm|I'll|you|we|it|that|this|the|they|there|but|and|more|please|thank)\b)",
            ", ",
            msg,
            flags=re.IGNORECASE,
        )
        msg = re.sub(
            r":\s+(?=(I|I'm|I'll|you|we|it|that|this|the|they|there|but|and|because|so|please|thank)\b)",
            ". ",
            msg,
            flags=re.IGNORECASE,
        )

    msg = msg.replace(":_.", "...").replace(":-", "...").replace("_.", "...")
    msg = re.sub(r"\.{4,}", "...", msg)
    msg = re.sub(r"\s+", " ", msg).strip()
    msg = re.sub(r"\s+([,.;:!?])", r"\1", msg)
    msg = re.sub(r"([,.;:!?])(?=[A-Za-z])", r"\1 ", msg)
    msg = re.sub(r"\s+\.\.\.", "...", msg)

    if msg.endswith(":") and not re.search(rf"\b({label_like})\s*:$", msg, flags=re.IGNORECASE):
        msg = msg[:-1] + "."

    return msg.strip()

def _side_csv_rows(side_csv: str) -> List[List[str]]:
    """Read Time/Side/Message rows from a side CSV."""
    rows: List[List[str]] = []
    reader = csv.reader(io.StringIO(strip_code_fences(side_csv)))
    for row in reader:
        if not row:
            continue
        if len(row) >= 3 and row[0].strip().lower() == "time":
            continue
        if len(row) < 3:
            continue
        time_value = row[0].strip()
        side = row[1].strip().upper()
        message = ",".join(row[2:]).strip() if len(row) > 3 else row[2].strip()
        if time_value and side in {"LEFT", "RIGHT"} and message:
            rows.append([time_value, side, message])
    return rows

def normalize_message_for_similarity(message: str) -> str:
    """Normalize message text for duplicate/continuation checks only."""
    text = strip_emojis(str(message or ""))
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def message_similarity_ratio(a: str, b: str) -> float:
    """Return a generic similarity ratio for two message strings."""
    a_norm = normalize_message_for_similarity(a)
    b_norm = normalize_message_for_similarity(b)
    if not a_norm and not b_norm:
        return 1.0
    if not a_norm or not b_norm:
        return 0.0
    return SequenceMatcher(None, a_norm, b_norm).ratio()

def _datetime_from_side_time(time_value: str):
    """Parse transcript timestamp strings when possible."""
    from datetime import datetime
    text = str(time_value or "").strip()
    for fmt in (
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y, %H:%M:%S",
        "%d/%m/%Y,%H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%d/%m/%Y, %H:%M",
        "%d/%m/%Y,%H:%M",
    ):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None

def _minutes_apart(a: str, b: str) -> Optional[float]:
    """Return absolute minute difference between two row timestamps."""
    da = _datetime_from_side_time(a)
    db = _datetime_from_side_time(b)
    if da is None or db is None:
        return None
    return abs((da - db).total_seconds()) / 60.0

def _same_day(a: str, b: str) -> bool:
    """Check if two timestamp strings share the same DD/MM/YYYY prefix."""
    return str(a or "")[:10] == str(b or "")[:10]

def _message_quality_score(message: str) -> Tuple[int, int, int]:
    """Prefer fuller, cleaner messages when removing near duplicates."""
    text = str(message or "").strip()
    words = normalize_message_for_similarity(text).split()
    terminal = 1 if re.search(r"[.!?…]$", text) else 0
    odd = len(re.findall(r"[^\w\s,.;:!?€£$+@%/'\"()\-]", text))
    return (len(words), len(text), terminal - odd)

def are_near_duplicate_messages(a: str, b: str) -> bool:
    """Detect adjacent duplicates with minor OCR/VLM differences."""
    a_norm = normalize_message_for_similarity(a)
    b_norm = normalize_message_for_similarity(b)
    if not a_norm or not b_norm:
        return False

    if a_norm == b_norm:
        return True

    shorter, longer = sorted([a_norm, b_norm], key=len)
    if len(shorter.split()) >= 4 and shorter in longer:
        return True

    ratio = SequenceMatcher(None, a_norm, b_norm).ratio()
    return ratio >= 0.88

def looks_like_orphan_fragment_message(message: str) -> bool:
    """Detect likely OCR/VLM fragments that should trigger repair or cautious merge."""
    raw = str(message or "").strip()
    if not raw:
        return False

    text = strip_emojis(raw).strip()
    words = re.findall(r"[A-Za-z']+", text)
    if not words:
        return False

    first = words[0].lower().strip("'")
    word_count = len(words)
    starts_lower = bool(text[:1].islower())

    if starts_lower and "?" in text and word_count <= 8:
        return True

    fragment_starts = {
        "and", "but", "because", "so", "that", "which", "with", "without",
        "for", "to", "of", "in", "on", "at", "by", "from", "as", "than",
        "my", "your", "our", "their", "the", "a", "an", "there", "then", "right",
    }
    if starts_lower and first in fragment_starts and word_count <= 8:
        return True

    if re.search(r"\b[a-z]{2,}\s+(The|I|I'm|I'll|You|We|They)\b", text) and word_count <= 12:
        return True

    return False

def should_merge_continuation(prev_message: str, message: str) -> bool:
    """Conservatively merge only wrapped-line fragments, not normal adjacent bubbles."""
    prev = str(prev_message or "").strip()
    cur = str(message or "").strip()
    if not prev or not cur:
        return False

    cur_words = re.findall(r"[A-Za-z0-9']+", cur)
    if not cur_words:
        return False

    prev_finished = bool(re.search(r"[.!?…]$", prev))
    first = re.sub(r"[^A-Za-z']+", "", cur_words[0]).lower()
    starts_lower = bool(cur[:1].islower())

    # Never merge a lowercase-start question into a previous completed message;
    # it is suspicious and should be repaired/reordered from the image instead.
    if starts_lower and "?" in cur:
        return False

    continuation_starts = {
        "and", "but", "because", "so", "that", "which", "with", "without",
        "for", "to", "of", "in", "on", "at", "by", "from", "as", "than",
        "my", "your", "our", "their", "the", "a", "an", "better"
    }

    # A current row after an unfinished previous row can be a wrapped line even if
    # it has more than three words. Keep the limit modest to avoid merging bubbles.
    if not prev_finished and len(cur_words) <= 8:
        if starts_lower or first in continuation_starts:
            return True

    # Tiny fragments are safe to absorb when the previous row is unfinished.
    if not prev_finished and len(cur_words) <= 3:
        return True

    # Previous row ending with comma/colon/semicolon can absorb a tiny continuation.
    if len(cur_words) <= 2 and re.search(r"[,;:]$", prev) and first not in {"yes", "no", "ok", "okay"}:
        return True

    return False

def split_side_row_by_known_boundaries(row: List[str]) -> List[List[str]]:
    """Return a cleaned side-CSV row without dataset-specific transcript rewrites."""
    time_value, side, message = row
    msg = conservative_clean_message_text(message)
    if not msg:
        return []
    return [[time_value, side, msg]]

def postprocess_side_csv_rows(side_csv: str) -> str:
    """Generic side-CSV cleanup: near-duplicate removal and safe continuation merges."""
    rows = []
    for _row in _side_csv_rows(side_csv):
        rows.extend(split_side_row_by_known_boundaries(_row))

    # First remove exact and adjacent near duplicates.
    deduped: List[List[str]] = []
    seen_exact = set()
    for row in rows:
        item = tuple(row)
        if item in seen_exact:
            continue
        seen_exact.add(item)

        if deduped:
            prev = deduped[-1]
            close_time = _minutes_apart(prev[0], row[0])
            close_enough = close_time is None or close_time <= 2
            if prev[1] == row[1] and _same_day(prev[0], row[0]) and close_enough and are_near_duplicate_messages(prev[2], row[2]):
                if _message_quality_score(row[2]) > _message_quality_score(prev[2]):
                    # Keep the better text but preserve the earlier timestamp.
                    deduped[-1] = [prev[0], prev[1], row[2]]
                continue
        deduped.append(row)

    # Then merge obvious wrapped/fragments from the same side.
    merged: List[List[str]] = []
    for row in deduped:
        if merged:
            prev = merged[-1]
            close_time = _minutes_apart(prev[0], row[0])
            close_enough = close_time is None or close_time <= 1
            if prev[1] == row[1] and _same_day(prev[0], row[0]) and close_enough and should_merge_continuation(prev[2], row[2]):
                prev[2] = conservative_clean_message_text(prev[2] + " " + row[2])
                continue
        row[2] = conservative_clean_message_text(row[2])
        merged.append(row)

    out = io.StringIO()
    writer = csv.writer(out, quoting=csv.QUOTE_ALL, lineterminator="\n")
    writer.writerow(["Time", "Side", "Message"])

    final_seen = set()
    for row in merged:
        item = tuple(row)
        if item in final_seen:
            continue
        final_seen.add(item)
        writer.writerow(row)

    return out.getvalue().strip() + "\n"

def side_csv_needs_repair(side_csv: str, expected_bubble_count: int = 0) -> bool:
    """Detect whether a screen CSV needs a second VLM repair pass."""
    rows = _side_csv_rows(side_csv)
    row_count = len(rows)

    if row_count == 0:
        return True

    if expected_bubble_count > 0:
        # A one-row difference is common from OCR UI noise, but larger gaps indicate split/merge trouble.
        if abs(row_count - expected_bubble_count) >= 2:
            return True

    suspicious_fragment_count = 0

    for i, row in enumerate(rows):
        message = row[2]
        if re.search(r"\b\d{1,2}[:.,;]\d{2}\b", message):
            return True
        if looks_like_orphan_fragment_message(message):
            suspicious_fragment_count += 1
            return True
        if len(message) >= 220 and re.search(r"[.!?].+\b(I|You|We|They|Please|Thank|The|There)\b", message):
            return True
        # Suspicious OCR word-order: lowercase token before a new capitalized start.
        if re.search(r"\b(they|can|the|and|but)\s+(The|I|I'm|I'll|You|We|They)\b", message):
            return True
        if i > 0:
            prev = rows[i - 1]
            if prev[1] == row[1] and _same_day(prev[0], row[0]) and are_near_duplicate_messages(prev[2], message):
                return True
            close_time = _minutes_apart(prev[0], row[0])
            if prev[1] == row[1] and (close_time is None or close_time <= 1) and should_merge_continuation(prev[2], message):
                return True

    return False

def merge_side_csvs(csv_parts: List[str]) -> str:
    """Merges per-screen side CSV parts while removing exact duplicates."""
    out = io.StringIO()
    writer = csv.writer(out, quoting=csv.QUOTE_ALL, lineterminator="\n")

    writer.writerow(["Time", "Side", "Message"])

    seen = set()

    for part in csv_parts:
        reader = csv.reader(io.StringIO(strip_code_fences(part)))

        for row in reader:
            if not row:
                continue

            if len(row) >= 3 and row[0].strip().lower() == "time":
                continue

            if len(row) != 3:
                continue

            time_value = row[0].strip()
            side = row[1].strip().upper()
            message = row[2].strip()

            if side not in {"LEFT", "RIGHT"} or not time_value or not message:
                continue

            item = (time_value, side, message)
            if item in seen:
                continue

            seen.add(item)
            writer.writerow(item)

    return out.getvalue().strip() + "\n"


def _truncate_prompt_text(text: str, max_chars: int) -> str:
    """Truncates long context while preserving both the beginning and the end."""
    value = str(text or "")
    if max_chars <= 0 or len(value) <= max_chars:
        return value

    head = max_chars // 2
    tail = max_chars - head
    return value[:head] + "\n\n[...TRUNCATED...]\n\n" + value[-tail:]


def _split_mapping_name(value: str) -> List[str]:
    """Splits accidental combined mapping values into individual name candidates."""
    value = clean_name(value)
    if not value:
        return []

    parts = re.split(r"\s*(?:/|&|\band\b|\||;)\s*", value, flags=re.I)
    return [clean_name(part) for part in parts if clean_name(part)]


def _looks_like_human_name(value: str) -> bool:
    """Return ``True`` only for conservative, full human-name candidates.

    The report may contain many title-cased document labels.  A simple
    capitalization test therefore is not sufficient: phrases such as
    ``Matter Reported Suspected`` or ``Incident Summary`` must never become
    chat participants.  The checks below are intentionally domain-neutral and
    are based on document structure rather than on any specific dataset.
    """
    name = clean_name(value).strip(" ,;:.()[]{}\"'")
    if not name or len(name) > 80:
        return False

    # A useful identity normally contains a first and a last name.  Very long
    # title-like phrases are rejected before any model can use them.
    tokens = name.split()
    if len(tokens) < 2 or len(tokens) > 5:
        return False

    # These words are generic report labels, roles, organisations, channels,
    # and evidence descriptors.  Rejecting them prevents headings and field
    # names from being mistaken for people without relying on case-specific
    # names or terminology.
    blocked_tokens = {
        "account", "advisor", "agency", "alleged", "amount", "application",
        "army", "bank", "case", "chat", "chronology", "clinic", "company",
        "complainant", "complaint", "consult", "consultancy", "consulting",
        "contact", "corps", "country", "customs", "date", "department",
        "description", "details", "document", "email", "evidence", "facebook",
        "financial", "findings", "fraud", "full", "hospital", "identity",
        "incident", "information", "investigation", "investment", "legal",
        "location", "marine", "matter", "messenger", "moneygram", "name",
        "narrative", "office", "overview", "package", "payment", "platform",
        "police", "receiver", "report", "reported", "representative", "section",
        "sender", "service", "shipping", "statement", "subject", "summary",
        "support", "suspect", "suspected", "team", "time", "timeline",
        "transaction", "transfer", "viber", "victim", "website", "witness",
    }
    lowered = {token.strip(".'-\"").casefold() for token in tokens}
    if lowered & blocked_tokens:
        return False

    # Reject a small set of common multi-word administrative headings even if
    # punctuation or OCR noise prevents a single-token match above.
    normalized_phrase = " ".join(sorted(lowered))
    blocked_phrases = {
        "case overview", "evidence package", "incident summary",
        "matter reported", "matter reported suspected", "reported suspected",
        "subject details", "suspect details", "victim details",
    }
    if normalized_phrase in {" ".join(sorted(x.split())) for x in blocked_phrases}:
        return False

    # Allow initials, apostrophes, and hyphens, but require name-like casing.
    word_re = re.compile(r"^[A-Z][A-Za-z'’-]*$")
    initial_re = re.compile(r"^[A-Z]\.$")
    return all(word_re.match(token) or initial_re.match(token) for token in tokens)


def _report_name_support(report_text: str, name: str) -> Tuple[int, bool, bool]:
    """Measure whether a broad report match behaves like a person reference.

    Returns ``(occurrence_count, has_person_context, heading_only)``.  Broad
    title-case recall is accepted only when the name repeats or appears in a
    person-bearing sentence.  A one-off short heading is not enough evidence.
    """
    report = str(report_text or "")
    cleaned = clean_name(name)
    if not report or not cleaned:
        return 0, False, False

    # Match the exact sequence with flexible whitespace and word boundaries.
    flexible = r"\s+".join(re.escape(token) for token in cleaned.split())
    exact_re = re.compile(rf"(?<![A-Za-z]){flexible}(?![A-Za-z])", re.I)
    matches = list(exact_re.finditer(report))

    # Generic linguistic frames that normally introduce or describe a person.
    person_context_patterns = [
        rf"(?i)\b(?:mr|mrs|ms|miss|dr|prof)\.?\s+{flexible}\b",
        rf"(?i)\b(?:named|called|known\s+as|using\s+the\s+name|identified\s+as)\s+[\"“']?{flexible}\b",
        rf"(?i)\b(?:victim|complainant|suspect|witness|advisor|officer|agent|contact)\s*[:\-]?\s*{flexible}\b",
        rf"(?i)\b(?:contacted|messaged|called|spoke\s+with|talked\s+to|met\s+with)\s+{flexible}\b",
        rf"(?i)\b{flexible}\s+(?:contacted|messaged|called|said|stated|reported|explained|introduced|requested|sent|received)\b",
    ]
    has_person_context = any(re.search(pattern, report) for pattern in person_context_patterns)

    # A single short line containing only the candidate is likely a heading.
    heading_only = False
    if len(matches) == 1:
        line_start = report.rfind("\n", 0, matches[0].start()) + 1
        line_end = report.find("\n", matches[0].end())
        if line_end < 0:
            line_end = len(report)
        line = report[line_start:line_end].strip(" \t:;-–—")
        heading_only = normalize_name(line) == normalize_name(cleaned) and len(line) <= 80

    return len(matches), has_person_context, heading_only


def _message_explicit_full_names(side_csv: str) -> List[str]:
    """Collect full names that are explicitly introduced inside messages.

    This bootstrap step runs before the allowed-name list exists.  It only
    captures strong forms such as ``This is First Last`` or ``First Last:`` and
    still applies the conservative human-name filter.
    """
    names: List[str] = []
    patterns = [
        r"(?i)\b(?:this\s+is|my\s+name\s+is|i\s+am|i['’]?m)\s+([A-Z][A-Za-z'’.-]+(?:\s+[A-Z][A-Za-z'’.-]+){1,4})",
        r"(?m)^\s*([A-Z][A-Za-z'’.-]+(?:\s+[A-Z][A-Za-z'’.-]+){1,4})\s*[:\-]\s+",
    ]
    messages = "\n".join(row[2] for row in _side_csv_rows(side_csv))
    for pattern in patterns:
        for match in re.finditer(pattern, messages):
            candidate = clean_name(match.group(1)).strip(" ,;:.()[]{}\"'")
            if _looks_like_human_name(candidate):
                names.append(candidate)
    return names


def extract_contextual_name_candidates(
    report_text: str,
    actors: Dict,
    side_map: Dict[str, str],
    side_csv: str,
    max_candidates: int = 30,
) -> List[str]:
    """Collect report-grounded human names with explicit provenance tiers.

    Trusted sources are added first: structured actor fields, the provisional
    side map, explicit report labels, and explicit message self-identification.
    A broad title-case recall pass is retained for coverage, but a candidate
    from that pass must repeat or appear in a person-bearing sentence.  This
    prevents document headings from entering the candidate pool.
    """
    candidates: List[str] = []
    seen = set()

    def add(value: str, *, trusted: bool = False) -> None:
        name = clean_name(value).strip(" ,;:.()[]{}\"'")
        if not _looks_like_human_name(name):
            return

        # Broad report matches need evidence beyond capitalization.  Trusted
        # structured/message sources have already supplied that evidence.
        if not trusted:
            count, has_context, heading_only = _report_name_support(report_text, name)
            if heading_only or (count < 2 and not has_context):
                return

        key = normalize_name(name)
        if key and key not in seen:
            seen.add(key)
            candidates.append(name)

    # Structured report extraction and the current visual/header mapping are
    # treated as trusted, but still pass the administrative-label filter.
    victim = clean_name((actors or {}).get("victim", ""))
    if victim:
        add(victim, trusted=True)

    for participant in (actors or {}).get("participants", []) or []:
        if isinstance(participant, dict):
            add(participant.get("name", ""), trusted=True)

    for mapped in (side_map or {}).values():
        for part in _split_mapping_name(mapped):
            add(part, trusted=True)

    # Names explicitly introduced in a message are among the strongest sources.
    for name in _message_explicit_full_names(side_csv):
        add(name, trusted=True)

    # High-confidence person fields commonly found in structured reports.
    label_patterns = [
        r"(?im)^\s*(?:Full\s+Name|Victim|Complainant|Witness|Target\s*/\s*Victim)\s*[:\-]\s*(.+?)\s*$",
        r"(?im)^\s*Suspect\s*\d*\s*[:\-]\s*(.+?)\s*$",
        r"(?im)^\s*(?:Scammer\s+Names?\s+Used|Scammer\s+Name|Alias|Claimed\s+Identity)\s*[:\-]\s*(.+?)\s*$",
    ]
    for pattern in label_patterns:
        for match in re.finditer(pattern, report_text or ""):
            raw = re.sub(r"\([^)]*\)", " ", match.group(1))
            for part in re.split(r"\s*(?:,|;|/|\band\b|&)\s*", raw, flags=re.I):
                add(part, trusted=True)

    # Narrative identity introductions are trusted because the surrounding
    # grammar explicitly identifies a person rather than a document heading.
    narrative_patterns = [
        r"(?i)\b(?:using\s+the\s+name|known\s+as|called|named)\s+[\"“']?([A-Z][A-Za-z'’.-]+(?:\s+[A-Z][A-Za-z'’.-]+){1,4})",
        r"(?i)\b(?:this\s+is|my\s+name\s+is|i\s+am|i['’]?m)\s+([A-Z][A-Za-z'’.-]+(?:\s+[A-Z][A-Za-z'’.-]+){1,4})",
    ]
    combined_text = (report_text or "") + "\n" + (side_csv or "")
    for pattern in narrative_patterns:
        for match in re.finditer(pattern, combined_text):
            add(match.group(1), trusted=True)

    # Broad recall is deliberately last and evidence-gated.  It recovers names
    # missed by field parsing without allowing arbitrary title-cased phrases.
    for match in re.finditer(
        r"\b[A-Z][A-Za-z'’-]+(?:[ \t]+(?:[A-Z]\.|[A-Z][A-Za-z'’-]+)){1,3}\b",
        report_text or "",
    ):
        add(match.group(0), trusted=False)
        if len(candidates) >= max_candidates:
            break

    return candidates[:max_candidates]

def _unique_names(values: Sequence[str]) -> List[str]:
    """Return cleaned names once, preserving the original order."""
    result: List[str] = []
    seen = set()
    for value in values:
        name = clean_name(value)
        key = normalize_name(name)
        if name and key and key not in seen:
            seen.add(key)
            result.append(name)
    return result


def _identity_variant_map(allowed_names: Sequence[str]) -> Dict[str, str]:
    """Build exact and unique-first-name aliases for message cue matching."""
    cleaned = _unique_names(allowed_names)
    variants: Dict[str, str] = {}
    first_counts: Dict[str, int] = {}

    for name in cleaned:
        tokens = name.split()
        if tokens:
            first = tokens[0].casefold()
            first_counts[first] = first_counts.get(first, 0) + 1

    for name in cleaned:
        variants[normalize_name(name)] = name
        tokens = name.split()
        if tokens and len(tokens[0]) >= 2 and first_counts.get(tokens[0].casefold(), 0) == 1:
            variants[tokens[0].casefold()] = name

    return variants


def _name_regex_variants(name: str, first_name_is_unique: bool) -> List[str]:
    """Return safe regex forms for a full name and, when unambiguous, first name."""
    cleaned = clean_name(name)
    if not cleaned:
        return []
    forms = [r"\s+".join(re.escape(token) for token in cleaned.split())]
    first = cleaned.split()[0]
    if first_name_is_unique and len(first) >= 2:
        forms.append(re.escape(first))
    return forms


def _extract_message_identity_cues(
    side_csv: str,
    allowed_names: Sequence[str],
) -> List[Dict[str, Any]]:
    """Extract deterministic sender/receiver cues from anonymous-side rows.

    The function deliberately distinguishes direct address and self-identification
    from third-party mentions. It never changes message text or row boundaries.
    """
    names = _unique_names(allowed_names)
    first_counts: Dict[str, int] = {}
    for name in names:
        first = name.split()[0].casefold()
        first_counts[first] = first_counts.get(first, 0) + 1

    rows: List[Dict[str, Any]] = []
    for row in _side_csv_rows(side_csv):
        time_value, side, message = row
        item: Dict[str, Any] = {
            "time": time_value,
            "side": side,
            "message": message,
            "speaker_prefix": [],
            "self_identification": [],
            "direct_address": [],
            "third_party_mention": [],
        }

        for name in names:
            first_unique = first_counts.get(name.split()[0].casefold(), 0) == 1
            forms = _name_regex_variants(name, first_unique)
            if not forms:
                continue

            full_form = forms[0]
            all_forms = "(?:" + "|".join(forms) + ")"

            prefix_patterns = [
                rf"(?i)^\s*{full_form}\s*[:\-]\s+",
            ]
            self_patterns = [
                # The introduction may follow a greeting or comma, e.g.
                # "Hello Alice, this is Bob Example".  Matching the exact
                # allowed full name keeps this safe even without sentence start.
                rf"(?i)\b(?:this\s+is|my\s+name\s+is|i\s+am|i['’]?m)\s+{full_form}\b",
            ]
            direct_patterns = [
                rf"(?i)^\s*(?:hello|hi|hey|dear|oh|thanks|thank\s+you|good\s+(?:morning|afternoon|evening))\s*[,!:.\-]?\s*{all_forms}\b",
                rf"(?i)^\s*{all_forms}\s*[,!?;\-]",
                rf"(?i)^\s*[A-Z][A-Za-z'’.-]+(?:\s+[A-Z][A-Za-z'’.-]+){{1,4}}\s*:\s*{all_forms}\s*[,!:;\-]",
                rf"(?i)[.!?]\s*{all_forms}\s*[,!:;!?]",
                rf"(?i)\b(?:please|listen)\s*[,!]?\s*{all_forms}\b",
                # OCR can move a vocative away from sentence start.  A
                # comma-delimited allowed name is still useful receiver
                # evidence, but it is never allowed to reverse an individual
                # row by itself.
                rf"(?i)(?:^|[.!?;:]|\b(?:please|listen|need|want|tell|ask|help)\b[^,.!?;:]{{0,40}})\s*{all_forms}\s*[,!?;:]",
            ]
            third_party_patterns = [
                rf"(?i)\b(?:spoke|talked|communicated|checked|met)\s+(?:with|to|about)\s+{all_forms}\b",
                rf"(?i)\b(?:heard|received|learned)\s+(?:from|about)\s+{all_forms}\b",
                rf"(?i)\baccording\s+to\s+{all_forms}\b",
                rf"(?i)\b{all_forms}\s+(?:told|said|explained|mentioned|reviewed|looked|checked|asked)\b",
                rf"(?i)\b(?:my|the)\s+(?:advisor|friend|contact|associate|doctor|agent)\s+{all_forms}\b",
            ]

            if any(re.search(pattern, message) for pattern in prefix_patterns):
                item["speaker_prefix"].append(name)
            if any(re.search(pattern, message) for pattern in self_patterns):
                item["self_identification"].append(name)
            if any(re.search(pattern, message) for pattern in direct_patterns):
                item["direct_address"].append(name)
            if any(re.search(pattern, message) for pattern in third_party_patterns):
                item["third_party_mention"].append(name)

        for key in ("speaker_prefix", "self_identification", "direct_address", "third_party_mention"):
            item[key] = _unique_names(item[key])
        rows.append(item)

    return rows


def _mapping_receiver(mapping: Dict[str, str], side: str) -> str:
    return mapping.get("RIGHT" if side == "LEFT" else "LEFT", "")


def _side_labels_are_reliable(cue_rows: Sequence[Dict[str, Any]]) -> bool:
    """Return whether both anonymous visual sides occur in the transcript.

    Direct address can orient a fixed LEFT/RIGHT mapping only when side
    extraction produced at least one LEFT and one RIGHT row.  When every row
    has the same side, geometry/grouping has probably lost direction and one
    vocative must not orient the whole screenshot.
    """
    sides = {
        str(row.get("side", "")).upper()
        for row in cue_rows
        if str(row.get("side", "")).upper() in {"LEFT", "RIGHT"}
    }
    return sides == {"LEFT", "RIGHT"}


def _hard_side_constraints(cue_rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, List[str]]]:
    """Build unambiguous sender/receiver requirements for each visual side.

    Speaker prefixes and self-identification constrain the sender on the same
    side.  Direct address constrains the receiver, which is the identity on the
    opposite side.  A requirement is enforced only when all strong cues for a
    side agree; conflicting OCR cues remain soft evidence instead of creating
    an impossible rule.
    """
    raw: Dict[str, Dict[str, List[str]]] = {
        "LEFT": {"sender": [], "receiver": []},
        "RIGHT": {"sender": [], "receiver": []},
    }
    side_labels_reliable = _side_labels_are_reliable(cue_rows)
    for row in cue_rows:
        side = str(row.get("side", "")).upper()
        if side not in raw:
            continue
        raw[side]["sender"].extend(row.get("speaker_prefix", []) or [])
        raw[side]["sender"].extend(row.get("self_identification", []) or [])

        # Direct address becomes a hard receiver constraint only when both
        # visual sides are represented.  Otherwise it remains soft pair
        # evidence and cannot orient an entire same-side extraction.
        if side_labels_reliable:
            raw[side]["receiver"].extend(row.get("direct_address", []) or [])

    constraints: Dict[str, Dict[str, List[str]]] = {
        "LEFT": {"sender": [], "receiver": []},
        "RIGHT": {"sender": [], "receiver": []},
    }
    for side in ("LEFT", "RIGHT"):
        for role in ("sender", "receiver"):
            unique = _unique_names(raw[side][role])
            if len(unique) == 1:
                constraints[side][role] = unique
    return constraints


def _mapping_constraint_violations(
    mapping: Dict[str, str],
    constraints: Dict[str, Dict[str, List[str]]],
) -> List[str]:
    """Return hard cue violations for one fixed mapping hypothesis."""
    violations: List[str] = []
    for side in ("LEFT", "RIGHT"):
        sender = clean_name(mapping.get(side, ""))
        receiver = _mapping_receiver(mapping, side)
        for required in constraints.get(side, {}).get("sender", []):
            if not same_name(sender, required):
                violations.append(f"{side} sender must be {required}")
        for required in constraints.get(side, {}).get("receiver", []):
            if not same_name(receiver, required):
                violations.append(f"{side} receiver must be {required}")
    return violations


def _report_pair_support(mapping: Dict[str, str], report_context: str) -> int:
    """Score whether both candidate identities co-occur in relevant passages.

    This is a small, soft prior.  It cannot overrule message-level hard cues and
    does not assume any case-specific roles.  Co-occurrence is evaluated by
    paragraph so a name mentioned in an unrelated report section contributes
    little or nothing.
    """
    left = clean_name(mapping.get("LEFT", ""))
    right = clean_name(mapping.get("RIGHT", ""))
    if not left or not right or not report_context:
        return 0

    left_key = left.casefold()
    right_key = right.casefold()
    support = 0
    for paragraph in re.split(r"\n\s*\n|(?<=[.!?])\s+(?=[A-Z])", report_context):
        low = paragraph.casefold()
        if left_key in low and right_key in low:
            support += 1
    return min(4, support)


def _same_oriented_mapping(a: Dict[str, str], b: Optional[Dict[str, str]]) -> bool:
    """Check whether two mappings assign the same person to each side."""
    if not b:
        return False
    return same_name(a.get("LEFT", ""), b.get("LEFT", "")) and same_name(
        a.get("RIGHT", ""), b.get("RIGHT", "")
    )


def _same_unordered_pair(a: Dict[str, str], b: Optional[Dict[str, str]]) -> bool:
    """Check whether two mappings contain the same two people in any order."""
    if not b:
        return False
    a_keys = {normalize_name(a.get("LEFT", "")), normalize_name(a.get("RIGHT", ""))}
    b_keys = {normalize_name(b.get("LEFT", "")), normalize_name(b.get("RIGHT", ""))}
    return "" not in a_keys and a_keys == b_keys


def _score_context_mapping(
    mapping: Dict[str, str],
    cue_rows: Sequence[Dict[str, Any]],
    provisional: Dict[str, str],
    prior_side_map: Optional[Dict[str, str]] = None,
    report_context: str = "",
    primary_party: str = "",
) -> Dict[str, Any]:
    """Score one fixed LEFT/RIGHT hypothesis using all available evidence.

    Evidence priority is explicit and generic:
    message-level identity cues dominate, visual/header mapping is a prior,
    continuity is a softer prior, and report co-occurrence is only a tie-breaker.
    """
    left = clean_name(mapping.get("LEFT", ""))
    right = clean_name(mapping.get("RIGHT", ""))
    if not left or not right or same_name(left, right):
        return {
            "mapping": {"LEFT": left, "RIGHT": right},
            "score": -10_000,
            "hard_contradictions": 99,
            "constraint_violations": ["invalid or identical identities"],
            "strong_evidence": 0,
            "continuity_match": False,
            "report_pair_support": 0,
            "primary_party_match": False,
            "support": [],
            "contradictions": ["invalid or identical identities"],
        }

    score = 0
    hard = 0
    strong = 0
    support: List[str] = []
    contradictions: List[str] = []

    # Preserve the current visual/header interpretation unless stronger message
    # evidence justifies an override.
    if same_name(left, provisional.get("LEFT", "")):
        score += 18
    if same_name(right, provisional.get("RIGHT", "")):
        score += 18

    # A previous mapping from the same evidence folder is useful for ambiguous
    # screenshots, but remains weaker than explicit text in the current image.
    continuity_match = _same_oriented_mapping(mapping, prior_side_map)
    if continuity_match:
        score += 42
        support.append("same oriented mapping as previous screenshot in this conversation group")
    elif _same_unordered_pair(mapping, prior_side_map):
        score += 14
        support.append("same participant pair as previous screenshot, reversed orientation")

    # The structured complainant/victim is a soft ambiguity prior.  It
    # remains weaker than explicit sender/receiver evidence and never creates a
    # hard requirement that every screenshot must contain the complainant.
    primary_party = clean_name(primary_party)
    primary_party_match = bool(
        primary_party
        and (same_name(left, primary_party) or same_name(right, primary_party))
    )
    if primary_party_match:
        score += 28
        support.append("mapping includes the structured primary complainant/victim")

    side_labels_reliable = _side_labels_are_reliable(cue_rows)

    strong_by_name: Dict[str, int] = {normalize_name(left): 0, normalize_name(right): 0}
    third_by_name: Dict[str, int] = {normalize_name(left): 0, normalize_name(right): 0}

    for index, row in enumerate(cue_rows, start=1):
        side = str(row.get("side", "")).upper()
        if side not in {"LEFT", "RIGHT"}:
            continue
        sender = mapping.get(side, "")
        receiver = _mapping_receiver(mapping, side)

        # A speaker label at the start of the extracted message is the strongest
        # sender cue because it is explicit rather than inferred from semantics.
        for name in row.get("speaker_prefix", []) or []:
            if same_name(sender, name):
                score += 145
                strong += 1
                strong_by_name[normalize_name(name)] = strong_by_name.get(normalize_name(name), 0) + 1
                support.append(f"row {index}: explicit speaker prefix supports {name} as sender")
            else:
                score -= 185
                hard += 1
                contradictions.append(f"row {index}: speaker prefix says {name}, mapped sender is {sender}")

        # Self-identification is also a hard sender constraint.
        for name in row.get("self_identification", []) or []:
            if same_name(sender, name):
                score += 125
                strong += 1
                strong_by_name[normalize_name(name)] = strong_by_name.get(normalize_name(name), 0) + 1
                support.append(f"row {index}: self-identification supports {name} as sender")
            else:
                score -= 165
                hard += 1
                contradictions.append(f"row {index}: self-identification says {name}, mapped sender is {sender}")

        # Direct address identifies the recipient, not a third party.  It
        # is a hard orientation cue only when both visual sides are present.
        # With a same-side extraction it stays soft, because one greeting or
        # vocative must not reverse every message in the screenshot.
        for name in row.get("direct_address", []) or []:
            if same_name(receiver, name):
                score += 62 if side_labels_reliable else 20
                if side_labels_reliable:
                    strong += 1
                strong_by_name[normalize_name(name)] = strong_by_name.get(normalize_name(name), 0) + 1
                support.append(f"row {index}: direct address supports {name} as receiver")
            elif same_name(sender, name):
                score -= 82 if side_labels_reliable else 22
                if side_labels_reliable:
                    hard += 1
                contradictions.append(f"row {index}: message directly addresses {name}, but {name} is mapped as sender")
            else:
                score -= 95 if side_labels_reliable else 35
                if side_labels_reliable:
                    hard += 1
                contradictions.append(f"row {index}: directly addressed {name} is absent from the pair")

        # Third-party mentions are tracked separately and never become positive
        # participant evidence by themselves.
        for name in row.get("third_party_mention", []) or []:
            key = normalize_name(name)
            if key in third_by_name:
                third_by_name[key] = third_by_name.get(key, 0) + 1

    # Penalize a candidate that appears only in third-party grammatical frames.
    for name in (left, right):
        key = normalize_name(name)
        if third_by_name.get(key, 0) and not strong_by_name.get(key, 0):
            penalty = min(30, 10 * third_by_name[key])
            score -= penalty
            contradictions.append(f"{name} appears only in third-party mention patterns (-{penalty})")

    # Relevant report co-occurrence is deliberately capped so it cannot override
    # explicit sender/receiver evidence from the screenshot itself.
    pair_support = _report_pair_support(mapping, report_context)
    if pair_support:
        score += pair_support * 6
        support.append(f"relevant report passages co-mention the pair ({pair_support})")

    # Apply pre-computed hard side requirements as an additional validation
    # layer.  These violations are used to remove impossible candidates before
    # the LLM sees them.
    constraints = _hard_side_constraints(cue_rows)
    constraint_violations = _mapping_constraint_violations(mapping, constraints)
    hard += len(constraint_violations)
    contradictions.extend(constraint_violations)

    return {
        "mapping": {"LEFT": left, "RIGHT": right},
        "score": score,
        "hard_contradictions": hard,
        "constraint_violations": constraint_violations,
        "strong_evidence": strong,
        "continuity_match": continuity_match,
        "report_pair_support": pair_support,
        "primary_party_match": primary_party_match,
        "support": support[:16],
        "contradictions": contradictions[:16],
    }


def _relevant_report_context(
    report_text: str,
    side_csv: str,
    allowed_names: Sequence[str],
    evidence_hint: str = "",
    max_chars: int = 20000,
) -> str:
    """Retrieve report passages relevant to the current evidence item.

    Terms come from both the extracted transcript and the evidence path/name.
    This reduces distraction from unrelated case stages without introducing
    any dataset-specific keywords or assumptions.
    """
    report = str(report_text or "")
    if not report or len(report) <= max_chars:
        return report

    transcript = " ".join(row[2] for row in _side_csv_rows(side_csv))
    lexical_source = transcript + " " + re.sub(r"[_/\\.-]+", " ", str(evidence_hint or ""))
    stopwords = {
        "about", "after", "again", "because", "before", "could", "every",
        "from", "have", "into", "just", "more", "other", "please", "should",
        "their", "there", "these", "they", "this", "through", "under", "very",
        "what", "when", "where", "which", "with", "would", "your", "youre",
    }
    terms = {
        token.casefold()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9'’.-]{4,}", lexical_source)
        if token.casefold() not in stopwords
    }
    name_terms = [clean_name(name).casefold() for name in allowed_names if clean_name(name)]

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n|(?m)(?=^\s*\d+\.\s+)", report) if part.strip()]
    scored: List[Tuple[int, int, str]] = []
    for index, paragraph in enumerate(paragraphs):
        low = paragraph.casefold()
        score = 0
        score += 12 * sum(1 for name in name_terms if name and name in low)
        score += sum(1 for term in terms if term in low)
        if score > 0:
            scored.append((score, index, paragraph))

    # Keep the opening role definitions and the most relevant later passages.
    selected: Dict[int, str] = {}
    for index, paragraph in enumerate(paragraphs[:3]):
        selected[index] = paragraph
    for _, index, paragraph in sorted(scored, key=lambda item: (-item[0], item[1]))[:12]:
        selected[index] = paragraph

    chunks: List[str] = []
    total = 0
    for index in sorted(selected):
        paragraph = selected[index]
        if total + len(paragraph) + 2 > max_chars:
            remaining = max_chars - total
            if remaining > 200:
                chunks.append(paragraph[:remaining])
            break
        chunks.append(paragraph)
        total += len(paragraph) + 2
    return "\n\n".join(chunks)


def _context_candidate_options(
    allowed_names: Sequence[str],
    cue_rows: Sequence[Dict[str, Any]],
    provisional: Dict[str, str],
    prior_side_map: Optional[Dict[str, str]] = None,
    report_context: str = "",
    primary_party: str = "",
    max_names: int = 16,
    max_options: int = 10,
) -> List[Dict[str, Any]]:
    """Enumerate, hard-filter, and rank fixed two-person side mappings."""
    cue_names: List[str] = []
    for row in cue_rows:
        for key in ("speaker_prefix", "self_identification", "direct_address"):
            cue_names.extend(row.get(key, []) or [])

    # Candidate ordering is evidence-aware: current mapping, previous mapping,
    # explicit message cues, and finally the remaining report-grounded names.
    preferred: List[str] = []
    for mapping in (provisional, prior_side_map or {}):
        for side in ("LEFT", "RIGHT"):
            mapped = clean_name(mapping.get(side, ""))
            if mapped and _looks_like_human_name(mapped):
                preferred.append(mapped)
    pool = _unique_names(preferred + cue_names + list(allowed_names))[:max_names]

    options: List[Dict[str, Any]] = []
    for left in pool:
        for right in pool:
            if same_name(left, right):
                continue
            options.append(
                _score_context_mapping(
                    {"LEFT": left, "RIGHT": right},
                    cue_rows,
                    provisional,
                    prior_side_map=prior_side_map,
                    report_context=report_context,
                    primary_party=primary_party,
                )
            )

    # When at least one mapping satisfies every explicit cue, impossible
    # mappings are removed entirely before the LLM selection step.
    zero_hard = [item for item in options if item["hard_contradictions"] == 0]
    if zero_hard:
        options = zero_hard

    options.sort(
        key=lambda item: (
            item["hard_contradictions"],
            -item["score"],
            -item["strong_evidence"],
            -int(item.get("continuity_match", False)),
            -item.get("report_pair_support", 0),
            -int(item.get("primary_party_match", False)),
            normalize_name(item["mapping"]["LEFT"]),
            normalize_name(item["mapping"]["RIGHT"]),
        )
    )
    return options[:max_options]

def _cue_summary_for_prompt(cue_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summary: List[Dict[str, Any]] = []
    for index, row in enumerate(cue_rows, start=1):
        if not any(row.get(key) for key in ("speaker_prefix", "self_identification", "direct_address", "third_party_mention")):
            continue
        summary.append({
            "row": index,
            "side": row.get("side", ""),
            "message": _truncate_prompt_text(str(row.get("message", "")), 350),
            "speaker_prefix": row.get("speaker_prefix", []),
            "self_identification": row.get("self_identification", []),
            "direct_address": row.get("direct_address", []),
            "third_party_mention": row.get("third_party_mention", []),
        })
    return summary[:40]


def build_contextual_side_map_prompt(
    report_text: str,
    side_csv: str,
    current_side_map: Dict[str, str],
    allowed_names: List[str],
    platform_hint: str,
    cue_rows: Optional[Sequence[Dict[str, Any]]] = None,
    candidate_options: Optional[Sequence[Dict[str, Any]]] = None,
    prior_side_map: Optional[Dict[str, str]] = None,
    evidence_hint: str = "",
    report_context: Optional[str] = None,
    primary_party: str = "",
) -> str:
    """Build a constrained conversation-level identity-attribution prompt.

    The model is never asked to invent a pair.  It can only select one mapping
    that has already passed deterministic candidate generation and filtering.
    """
    cues = list(cue_rows or _extract_message_identity_cues(side_csv, allowed_names))
    selected_report_context = report_context if report_context is not None else _relevant_report_context(
        report_text,
        side_csv,
        allowed_names,
        evidence_hint=evidence_hint,
    )
    options = list(candidate_options or _context_candidate_options(
        allowed_names,
        cues,
        current_side_map,
        prior_side_map=prior_side_map,
        report_context=selected_report_context,
        primary_party=primary_party,
    ))
    transcript_context = _truncate_prompt_text(side_csv, 45000)

    # Expose only compact evidence summaries; the model does not need internal
    # implementation details or unrestricted names.
    prompt_options = []
    for index, option in enumerate(options, start=1):
        prompt_options.append({
            "candidate_id": index,
            "LEFT": option["mapping"]["LEFT"],
            "RIGHT": option["mapping"]["RIGHT"],
            "deterministic_score": option["score"],
            "hard_contradictions": option["hard_contradictions"],
            "strong_message_evidence": option["strong_evidence"],
            "same_as_previous_screenshot": option.get("continuity_match", False),
            "relevant_report_pair_support": option.get("report_pair_support", 0),
            "includes_primary_complainant": option.get("primary_party_match", False),
        })

    return f"""
You are validating ONE fixed LEFT/RIGHT identity mapping for one two-person chat image or collage.
You are not assigning identities independently row by row.

PLATFORM:
{platform_hint}

EVIDENCE PATH / FILENAME HINT:
{evidence_hint or '[not supplied]'}

CURRENT PROVISIONAL MAP FROM HEADER / BUBBLE GEOMETRY:
{json.dumps(current_side_map, ensure_ascii=False, indent=2)}

PREVIOUS ACCEPTED MAP FROM THE SAME EVIDENCE FOLDER:
{json.dumps(prior_side_map or {}, ensure_ascii=False, indent=2)}

ALLOWED HUMAN NAMES:
{json.dumps(allowed_names, ensure_ascii=False, indent=2)}

STRUCTURED PRIMARY COMPLAINANT / VICTIM (SOFT PRIOR ONLY):
{primary_party or "[not available]"}

DETERMINISTIC MESSAGE CUES:
{json.dumps(_cue_summary_for_prompt(cues), ensure_ascii=False, indent=2)}

RANKED, PREVALIDATED MAPPING CANDIDATES:
{json.dumps(prompt_options, ensure_ascii=False, indent=2)}

MOST RELEVANT CASE-REPORT PASSAGES:
---
{selected_report_context}
---

EXTRACTED CHAT WITH FIXED ANONYMOUS SIDES:
---
{transcript_context}
---

Task:
Select the single candidate_id that best identifies LEFT and RIGHT for the entire extracted conversation.
Do not create a new mapping and do not alter any message, side, row, timestamp, order, split, or merge.

Evidence rules, strongest first:
1. A leading speaker label such as "First Last: ..." identifies the sender of that side.
2. Explicit self-identification such as "This is First Last" or "My name is X" identifies the sender.
3. Direct address identifies the receiver. "Hello Alice", "Alice, ...", and "Oh Bob" normally address that person; they do not identify the sender.
4. A person mentioned in "I spoke with X", "X told me", "according to X", or "X reviewed the account" is normally a third party, not automatically a participant in the current exchange.
5. Use previous and following messages to understand replies, but keep one fixed pair and one fixed LEFT/RIGHT mapping for every row.
6. The relevant report passages and evidence path identify the applicable communication stage. Ignore unrelated actors from other stages.
7. Header and bubble geometry are reliable prior evidence. A previous mapping from the same evidence folder is a useful continuity prior, but neither may override explicit message cues.
8. Every supplied candidate has already been checked. Prefer zero-contradiction candidates and do not invent an organisation, role, heading, alias combination, anonymous label, or extra person.
9. The structured primary complainant/victim is a soft prior when the screenshot is otherwise ambiguous. It must not override explicit evidence that the current exchange is between other people.
10. If the evidence does not justify a change, keep the provisional mapping or the consistent previous mapping.
11. Return only one supplied candidate_id.

Return valid JSON only:
{{
  "candidate_id": 1,
  "confidence": "high|medium|low",
  "evidence": ["brief evidence 1", "brief evidence 2"]
}}
"""


def refine_side_mapping_with_context(
    report_text: str,
    actors: Dict,
    side_csv: str,
    current_side_map: Dict[str, str],
    platform_hint: str,
    model: str,
    prior_side_map: Optional[Dict[str, str]] = None,
    evidence_hint: str = "",
) -> Dict[str, str]:
    """Validate one fixed side map using hard cues, continuity, report, and Gemma.

    The acceptance policy is intentionally conservative:
    * explicit message contradictions remove candidates before the LLM call;
    * the report is only a soft tie-breaker;
    * continuity may resolve ambiguous later screenshots but cannot override a
      speaker label, self-identification, or direct address;
    * a model-only narrative guess is never enough to replace a valid map.
    """
    provisional = {
        "LEFT": clean_name((current_side_map or {}).get("LEFT", "")),
        "RIGHT": clean_name((current_side_map or {}).get("RIGHT", "")),
    }
    prior = {
        "LEFT": clean_name((prior_side_map or {}).get("LEFT", "")),
        "RIGHT": clean_name((prior_side_map or {}).get("RIGHT", "")),
    }
    if not provisional["LEFT"] or not provisional["RIGHT"]:
        return current_side_map

    # Build a strictly human, report-grounded candidate list.  Administrative
    # headings and organisations are rejected before candidate enumeration.
    allowed_names = extract_contextual_name_candidates(
        report_text=report_text,
        actors=actors,
        side_map={**provisional, **({} if not prior else {"PRIOR_LEFT": prior["LEFT"], "PRIOR_RIGHT": prior["RIGHT"]})},
        side_csv=side_csv,
    )
    if len(allowed_names) < 2:
        return provisional

    # Canonicalize the structured primary complainant/victim against the
    # filtered name list.  It is used only as a soft ambiguity prior.
    primary_party = clean_name((actors or {}).get("victim", ""))
    if not any(same_name(primary_party, name) for name in allowed_names):
        primary_party = ""

    cue_rows = _extract_message_identity_cues(side_csv, allowed_names)
    report_context = _relevant_report_context(
        report_text,
        side_csv,
        allowed_names,
        evidence_hint=evidence_hint,
    )
    options = _context_candidate_options(
        allowed_names,
        cue_rows,
        provisional,
        prior_side_map=prior if prior["LEFT"] and prior["RIGHT"] else None,
        report_context=report_context,
        primary_party=primary_party,
    )
    if not options:
        return provisional

    provisional_record = _score_context_mapping(
        provisional,
        cue_rows,
        provisional,
        prior_side_map=prior if prior["LEFT"] and prior["RIGHT"] else None,
        report_context=report_context,
        primary_party=primary_party,
    )
    provisional_valid = (
        _looks_like_human_name(provisional["LEFT"])
        and _looks_like_human_name(provisional["RIGHT"])
        and any(_same_oriented_mapping(option["mapping"], provisional) for option in options)
    )
    best = options[0]

    # Deterministic selection can use either strong current-message evidence or
    # a stable previous mapping when the current screenshot is ambiguous.
    deterministic_choice: Optional[Dict[str, Any]] = None
    margin = best["score"] - provisional_record["score"]
    cue_backed = best["strong_evidence"] >= 1
    continuity_backed = bool(best.get("continuity_match"))
    primary_backed = bool(
        best.get("primary_party_match")
        and best.get("report_pair_support", 0) >= 1
        and best["strong_evidence"] == 0
    )
    if best["hard_contradictions"] == 0:
        if cue_backed and (
            not provisional_valid
            or provisional_record["hard_contradictions"] > 0
            or margin >= 20
        ):
            deterministic_choice = best
        elif continuity_backed and (
            not provisional_valid
            or provisional_record["hard_contradictions"] > 0
            or (
                margin >= 8
                and best.get("report_pair_support", 0)
                >= provisional_record.get("report_pair_support", 0)
            )
        ):
            deterministic_choice = best
        elif primary_backed and (
            not provisional_valid
            or margin >= 14
        ):
            # Use the primary-party prior only when no strong message cue exists
            # and the relevant report passage also supports this pair.
            deterministic_choice = best

    prompt = build_contextual_side_map_prompt(
        report_text=report_text,
        side_csv=side_csv,
        current_side_map=provisional,
        allowed_names=allowed_names,
        platform_hint=platform_hint,
        cue_rows=cue_rows,
        candidate_options=options,
        prior_side_map=prior if prior["LEFT"] and prior["RIGHT"] else None,
        evidence_hint=evidence_hint,
        report_context=report_context,
        primary_party=primary_party,
    )

    # Gemma can only select one prevalidated candidate.  Its choice is accepted
    # only when deterministic evidence also makes the candidate competitive.
    llm_choice: Optional[Dict[str, Any]] = None
    confidence = "low"
    try:
        raw = ollama_chat_text(model, prompt)
        data = json.loads(extract_json_object(raw))
        confidence = str(data.get("confidence", "low")).strip().casefold()
        candidate_id = int(data.get("candidate_id", 0))
        if 1 <= candidate_id <= len(options):
            candidate = options[candidate_id - 1]
            evidence_backed = (
                candidate["strong_evidence"] >= 1
                or candidate.get("continuity_match", False)
                or candidate.get("report_pair_support", 0) >= 2
                or (
                    candidate.get("primary_party_match", False)
                    and candidate.get("report_pair_support", 0) >= 1
                )
            )
            if (
                confidence in {"high", "medium"}
                and candidate["hard_contradictions"] == 0
                and evidence_backed
                and candidate["score"] >= best["score"] - 10
                and (
                    not provisional_valid
                    or provisional_record["hard_contradictions"] > 0
                    or candidate["score"] >= provisional_record["score"] + 6
                )
            ):
                llm_choice = candidate
    except Exception as exc:
        print(f"[WARNING] Contextual side-map validation failed; using deterministic/provisional mapping: {exc}")

    chosen = llm_choice or deterministic_choice
    if chosen is None:
        print(
            "-> [SIDE MAP] No supported override accepted "
            f"(provisional score={provisional_record['score']}, "
            f"contradictions={provisional_record['hard_contradictions']})."
        )
        return provisional

    mapping = chosen["mapping"]
    print(
        "-> [SIDE MAP] Accepted constrained mapping "
        f"LEFT={mapping['LEFT']} RIGHT={mapping['RIGHT']} "
        f"score={chosen['score']} strong={chosen['strong_evidence']} "
        f"continuity={chosen.get('continuity_match', False)} "
        f"report_pair={chosen.get('report_pair_support', 0)} "
        f"primary_party={chosen.get('primary_party_match', False)} "
        f"contradictions={chosen['hard_contradictions']} "
        f"llm_confidence={confidence}."
    )
    return {"LEFT": mapping["LEFT"], "RIGHT": mapping["RIGHT"]}


def load_conversation_side_map(
    cache_path: Optional[str],
    conversation_key: str,
) -> Dict[str, str]:
    """Load the last accepted mapping for one evidence-folder conversation.

    The cache is optional and ephemeral in batch mode.  Corrupt or incomplete
    cache data is ignored so it can never stop extraction.
    """
    if not cache_path or not conversation_key:
        return {}
    path = Path(cache_path)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        record = (data.get("conversations", {}) or {}).get(conversation_key, {})
        mapping = {
            "LEFT": clean_name(record.get("LEFT", "")),
            "RIGHT": clean_name(record.get("RIGHT", "")),
        }
        if (
            _looks_like_human_name(mapping["LEFT"])
            and _looks_like_human_name(mapping["RIGHT"])
            and not same_name(mapping["LEFT"], mapping["RIGHT"])
        ):
            return mapping
    except Exception:
        return {}
    return {}


def save_conversation_side_map(
    cache_path: Optional[str],
    conversation_key: str,
    side_map: Dict[str, str],
    source_hint: str = "",
) -> None:
    """Atomically persist one validated mapping for later screenshots.

    Only two distinct human names are stored.  This prevents an invalid model
    output or document heading from contaminating subsequent screenshots.
    """
    if not cache_path or not conversation_key:
        return
    mapping = {
        "LEFT": clean_name((side_map or {}).get("LEFT", "")),
        "RIGHT": clean_name((side_map or {}).get("RIGHT", "")),
    }
    if (
        not _looks_like_human_name(mapping["LEFT"])
        or not _looks_like_human_name(mapping["RIGHT"])
        or same_name(mapping["LEFT"], mapping["RIGHT"])
    ):
        return

    path = Path(cache_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        data: Dict[str, Any] = {"version": 1, "conversations": {}}
        if path.exists():
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data.update(loaded)
                if not isinstance(data.get("conversations"), dict):
                    data["conversations"] = {}
        data["conversations"][conversation_key] = {
            "LEFT": mapping["LEFT"],
            "RIGHT": mapping["RIGHT"],
            "source_hint": str(source_hint or ""),
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)
    except Exception as exc:
        print(f"[WARNING] Could not update conversation continuity cache: {exc}")

def apply_side_mapping(
    side_csv: str,
    side_map: Dict[str, str],
    estimated_timestamp_from_seconds: bool = False,
) -> str:
    """Convert LEFT/RIGHT rows into final Sender/Receiver rows.

    ``Estimated_Timestamp`` is always emitted immediately after ``Timestamp``.
    For Viber and other callers the value is ``False`` by default. Facebook can
    enable ``estimated_timestamp_from_seconds`` because its visible UI times are
    minute-level anchors: ``:00`` remains observed, while deterministic non-zero
    seconds added only to preserve bubble order are marked ``True``.

    The accepted conversation-level mapping is the default for every row.
    Individual direction is changed only by an explicit sender cue on that
    exact row: a leading ``Full Name:`` label or clear self-identification such
    as ``This is Full Name``.  Direct address (``Hello Alice``, ``Oh Bob``)
    identifies the recipient and is used during conversation-level validation,
    but it never flips an individual bubble by itself.
    """
    parsed_rows = _side_csv_rows(side_csv)
    pair_names = _unique_names([
        clean_name((side_map or {}).get("LEFT", "")),
        clean_name((side_map or {}).get("RIGHT", "")),
    ])
    cue_rows = _extract_message_identity_cues(side_csv, pair_names) if len(pair_names) == 2 else []

    out = io.StringIO()
    writer = csv.writer(out, quoting=csv.QUOTE_ALL, lineterminator="\n")
    writer.writerow(["Timestamp", "Estimated_Timestamp", "Sender", "Receiver", "Message"])

    seen = set()
    for index, (time_value, side, message) in enumerate(parsed_rows):
        if side not in {"LEFT", "RIGHT"} or not time_value or not message:
            continue

        sender = clean_name((side_map or {}).get(side, ""))
        receiver = clean_name((side_map or {}).get("RIGHT" if side == "LEFT" else "LEFT", ""))
        cue = cue_rows[index] if index < len(cue_rows) else {}

        # Only an explicit statement of who is speaking can override the
        # visual side for one isolated row.  The identity must already be one
        # of the two accepted participants; this stage cannot add a third name.
        explicit_senders = _unique_names(
            list(cue.get("speaker_prefix", []) or [])
            + list(cue.get("self_identification", []) or [])
        )
        if len(explicit_senders) == 1:
            explicit_sender = explicit_senders[0]
            if same_name(explicit_sender, receiver):
                sender, receiver = receiver, sender

        if not sender or not receiver or same_name(sender, receiver):
            continue

        estimated_timestamp = False
        if estimated_timestamp_from_seconds:
            parsed_time = _datetime_from_transcript_time(time_value)
            estimated_timestamp = bool(parsed_time is not None and parsed_time.second != 0)

        item = (
            time_value,
            estimated_timestamp,
            sender,
            receiver,
            message,
        )
        if item in seen:
            continue
        seen.add(item)
        writer.writerow(item)

    return out.getvalue().strip() + "\n"

def renumber_estimated_facebook_timestamps(final_csv: str) -> str:
    """Renumber surviving Facebook estimated timestamps contiguously.

    ``False`` rows are treated as observed timestamp anchors and are preserved
    exactly. Consecutive ``True`` rows after an observed anchor are rewritten
    to anchor+1s, anchor+2s, ... in their surviving row order. This is intended
    to run after all row filtering/deduplication so removed rows cannot leave
    gaps such as ``:02`` -> ``:05``. The provenance flag itself is never
    changed.
    """
    reader = csv.reader(io.StringIO(final_csv))
    try:
        header = next(reader)
    except StopIteration:
        return final_csv

    normalized = [str(col or "").lstrip("\ufeff").strip() for col in header]
    required = ["Timestamp", "Estimated_Timestamp", "Sender", "Receiver", "Message"]
    if any(col not in normalized for col in required):
        return final_csv

    indexes = {col: normalized.index(col) for col in required}
    rows = [list(row) for row in reader if row]

    anchor_dt = None
    next_offset = 1

    def is_true(value: str) -> bool:
        return str(value or "").strip().casefold() in {"true", "1", "yes", "y"}

    for row in rows:
        ts_idx = indexes["Timestamp"]
        est_idx = indexes["Estimated_Timestamp"]
        if ts_idx >= len(row) or est_idx >= len(row):
            continue

        estimated = is_true(row[est_idx])
        row[est_idx] = "True" if estimated else "False"
        parsed = _datetime_from_transcript_time(row[ts_idx])

        if not estimated:
            if parsed is not None:
                anchor_dt = parsed
                next_offset = 1
            else:
                anchor_dt = None
            continue

        if anchor_dt is None:
            continue

        from datetime import timedelta
        rewritten = anchor_dt + timedelta(seconds=next_offset)
        row[ts_idx] = rewritten.strftime("%d/%m/%Y %H:%M:%S")
        next_offset += 1

    out = io.StringIO()
    writer = csv.writer(out, quoting=csv.QUOTE_ALL, lineterminator="\n")
    writer.writerow(required)
    for row in rows:
        writer.writerow([
            row[indexes["Timestamp"]] if indexes["Timestamp"] < len(row) else "",
            row[indexes["Estimated_Timestamp"]] if indexes["Estimated_Timestamp"] < len(row) else "False",
            row[indexes["Sender"]] if indexes["Sender"] < len(row) else "",
            row[indexes["Receiver"]] if indexes["Receiver"] < len(row) else "",
            row[indexes["Message"]] if indexes["Message"] < len(row) else "",
        ])
    return out.getvalue().strip() + "\n"

def write_crop(path: Path, crop: np.ndarray) -> None:
    """Writes an image crop to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), crop)

def _token_counter(text: str) -> Dict[str, int]:
    """Builds a lowercase token counter for generic coverage scoring."""
    counter: Dict[str, int] = {}
    for token in normalize_message_for_similarity(text).split():
        counter[token] = counter.get(token, 0) + 1
    return counter

def _token_overlap_ratio(a: str, b: str) -> float:
    """Measures how much of the shorter text is covered by the other text."""
    a_counter = _token_counter(a)
    b_counter = _token_counter(b)
    if not a_counter or not b_counter:
        return 0.0

    overlap = 0
    for token, count in a_counter.items():
        overlap += min(count, b_counter.get(token, 0))

    a_total = sum(a_counter.values())
    b_total = sum(b_counter.values())
    return overlap / max(1, min(a_total, b_total))

def _candidate_rows_for_scoring(side_csv: str) -> List[List[str]]:
    """Returns cleaned rows used only for candidate scoring."""
    rows: List[List[str]] = []
    for time_value, side, message in _side_csv_rows(side_csv):
        message = conservative_clean_message_text(message)
        if message:
            rows.append([time_value, side, message])
    return rows

def _bubble_groups_for_scoring(bubble_groups: Optional[List[Dict[str, str]]]) -> List[Dict[str, str]]:
    """Filters OCR/VLM bubble hints down to transcript-like groups."""
    if not bubble_groups:
        return []

    useful: List[Dict[str, str]] = []
    for group in bubble_groups:
        text = str(group.get("text", "")).strip()
        side = str(group.get("side", "")).upper()
        if side not in {"LEFT", "RIGHT"}:
            continue
        if len(normalize_message_for_similarity(text).split()) < 2:
            continue
        useful.append(group)
    return useful

def side_csv_bubble_coverage(
    side_csv: str,
    bubble_groups: Optional[List[Dict[str, str]]] = None,
    min_overlap: float = 0.58,
) -> Tuple[int, int, float]:
    """Scores how many visible OCR bubble hints are represented in a side CSV.

    This is a generic completeness signal. It does not rewrite transcript text and
    it does not contain dataset-specific phrases.
    """
    rows = _candidate_rows_for_scoring(side_csv)
    groups = _bubble_groups_for_scoring(bubble_groups)
    if not groups:
        return 0, 0, 1.0

    covered = 0
    for group in groups:
        group_text = str(group.get("text", "")).strip()
        group_side = str(group.get("side", "")).upper()
        best = 0.0
        for _, row_side, row_message in rows:
            if group_side in {"LEFT", "RIGHT"} and row_side in {"LEFT", "RIGHT"} and group_side != row_side:
                continue
            best = max(best, _token_overlap_ratio(group_text, row_message))
        if best >= min_overlap:
            covered += 1

    return covered, len(groups), covered / max(1, len(groups))

def _candidate_suspicion_penalty(rows: List[List[str]]) -> int:
    """Penalizes structurally suspicious rows without using case-specific content."""
    penalty = 0
    for i, row in enumerate(rows):
        message = row[2]
        if re.search(r"\b\d{1,2}[:.,;]\d{2}\b", message):
            penalty += 4
        if looks_like_orphan_fragment_message(message):
            penalty += 3
        if len(message) >= 240:
            penalty += 2
        if looks_like_noisy_ocr_text(message):
            penalty += 2
        if i > 0:
            prev = rows[i - 1]
            if prev[1] == row[1] and _same_day(prev[0], row[0]) and are_near_duplicate_messages(prev[2], message):
                penalty += 3
            close_time = _minutes_apart(prev[0], row[0])
            if prev[1] == row[1] and (close_time is None or close_time <= 1) and should_merge_continuation(prev[2], message):
                penalty += 2
    return penalty

def score_side_csv_candidate(
    side_csv: str,
    allowed_times: Optional[Set[str]] = None,
    expected_bubble_count: int = 0,
    bubble_groups: Optional[List[Dict[str, str]]] = None,
) -> float:
    """Returns a generic quality score for a draft/repair side CSV candidate."""
    rows = _candidate_rows_for_scoring(side_csv)
    if not rows:
        return -10_000.0

    allowed_times = allowed_times or set()
    row_count = len(rows)
    score = row_count * 2.0

    if expected_bubble_count > 0:
        count_gap = abs(row_count - expected_bubble_count)
        score -= count_gap * 6.0
        if row_count < expected_bubble_count:
            score -= (expected_bubble_count - row_count) * 4.0

    covered, total, coverage_ratio = side_csv_bubble_coverage(
        side_csv,
        bubble_groups=bubble_groups,
    )
    if total:
        score += coverage_ratio * 70.0
        score += covered * 1.5

    invalid_time_count = 0
    if allowed_times:
        for time_value, _, _ in rows:
            hhmm = extract_hhmm_from_full_time(time_value)
            if hhmm and hhmm not in allowed_times:
                invalid_time_count += 1
    score -= invalid_time_count * 5.0
    score -= _candidate_suspicion_penalty(rows) * 4.0

    return score

def _best_bubble_match_for_row(
    row: List[str],
    bubble_groups: Optional[List[Dict[str, str]]] = None,
    min_overlap: float = 0.58,
) -> Tuple[Optional[Tuple[str, int]], float]:
    """Find the best generic OCR-bubble hint match for one side-CSV row."""
    groups = _bubble_groups_for_scoring(bubble_groups)
    if not groups:
        return None, 0.0

    _, row_side, row_message = row
    best_key: Optional[Tuple[str, int]] = None
    best_score = 0.0

    for fallback_order, group in enumerate(groups):
        group_side = str(group.get("side", "")).upper()
        if group_side in {"LEFT", "RIGHT"} and row_side in {"LEFT", "RIGHT"} and group_side != row_side:
            continue

        score = _token_overlap_ratio(row_message, str(group.get("text", "")))
        if score > best_score:
            try:
                order = int(group.get("order", fallback_order))
            except Exception:
                order = fallback_order
            best_score = score
            best_key = (group_side, order)

    if best_key is not None and best_score >= min_overlap:
        return best_key, best_score

    return None, best_score

def _rows_are_generic_duplicates(a: List[str], b: List[str]) -> bool:
    """Detect duplicate candidate rows without transcript-specific phrases."""
    if a[1] != b[1]:
        return False

    a_norm = normalize_message_for_similarity(a[2])
    b_norm = normalize_message_for_similarity(b[2])
    if not a_norm or not b_norm:
        return False

    if a_norm == b_norm:
        return True

    shorter, longer = sorted([a_norm, b_norm], key=len)
    if len(shorter.split()) >= 3 and shorter in longer:
        return True

    return message_similarity_ratio(a[2], b[2]) >= 0.86

def _row_has_valid_visible_time(row: List[str], allowed_times: Optional[Set[str]]) -> bool:
    """Validate row time against visible OCR times when available."""
    if not allowed_times:
        return True
    hhmm = extract_hhmm_from_full_time(row[0])
    return bool(hhmm and hhmm in allowed_times)

def merge_side_csv_candidates_additive(
    draft_norm: str,
    repaired_norm: str,
    allowed_times: Optional[Set[str]] = None,
    expected_bubble_count: int = 0,
    bubble_groups: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Build a conservative additive candidate from draft + repair outputs.

    The draft is used as the baseline. Rows from the repair pass are added only
    when they appear to cover a visible OCR bubble not already represented by
    the draft, or when the row count is clearly below the expected bubble count.
    This is a general recall-improvement step: it uses geometry/OCR bubble
    coverage, not dataset-specific names, locations, amounts, or message text.
    """
    allowed_times = allowed_times or set()
    draft_rows = [r for r in _candidate_rows_for_scoring(draft_norm) if _row_has_valid_visible_time(r, allowed_times)]
    repair_rows = [r for r in _candidate_rows_for_scoring(repaired_norm) if _row_has_valid_visible_time(r, allowed_times)]

    if not draft_rows:
        return repaired_norm
    if not repair_rows:
        return draft_norm

    rows: List[Tuple[int, int, Optional[Tuple[str, int]], List[str]]] = []
    covered_bubbles: Set[Tuple[str, int]] = set()

    for idx, row in enumerate(draft_rows):
        key, _score = _best_bubble_match_for_row(row, bubble_groups=bubble_groups)
        if key is not None:
            covered_bubbles.add(key)
        rows.append((0, idx, key, row))

    for idx, row in enumerate(repair_rows):
        if any(_rows_are_generic_duplicates(row, existing_row) for *_unused, existing_row in rows):
            continue

        key, match_score = _best_bubble_match_for_row(row, bubble_groups=bubble_groups)
        suspicious = looks_like_noisy_ocr_text(row[2]) or looks_like_orphan_fragment_message(row[2])

        should_add = False
        if key is not None and key not in covered_bubbles:
            # Add only rows that map to a previously uncovered visible bubble.
            should_add = True
        elif not _bubble_groups_for_scoring(bubble_groups) and expected_bubble_count > len(rows):
            # Fallback for screenshots with no usable bubble hints: add only if
            # the output is still below the expected bubble count.
            should_add = not suspicious
        elif expected_bubble_count > len(rows) and match_score >= 0.72:
            # Conservative fallback: a strong hint match can fill an under-counted screen.
            should_add = not suspicious

        if not should_add:
            continue

        if key is not None:
            covered_bubbles.add(key)
        rows.append((1, idx, key, row))

    if len(rows) == len(draft_rows):
        return draft_norm

    # Sort by visible bubble order when known; otherwise keep baseline/repair order.
    rows.sort(
        key=lambda item: (
            item[2] is None,
            item[2][1] if item[2] is not None else item[1],
            item[0],
            item[1],
        )
    )

    out = io.StringIO()
    writer = csv.writer(out, quoting=csv.QUOTE_ALL, lineterminator="\n")
    writer.writerow(["Time", "Side", "Message"])

    emitted: List[List[str]] = []
    for _source, _idx, _key, row in rows:
        if any(_rows_are_generic_duplicates(row, prev) for prev in emitted):
            continue
        emitted.append(row)
        writer.writerow(row)

    return out.getvalue().strip() + "\n"

def choose_best_screen_side_csv(
    draft_norm: str,
    repaired_norm: str,
    allowed_times: Set[str],
    expected_bubble_count: int = 0,
    bubble_groups: Optional[List[Dict[str, str]]] = None,
    enable_additive: bool = True,
) -> str:
    """Choose the safest candidate using platform-appropriate generic signals.

    When enable_additive is True, a third candidate is built from draft rows plus
    repair rows that cover previously uncovered OCR bubble hints. This is useful
    for platforms where hidden/shared timestamps make missing-bubble recovery
    more likely. When enable_additive is False, selection is limited to the
    conservative draft-vs-repair comparison.
    """
    draft_count = count_data_rows(draft_norm)
    repair_count = count_data_rows(repaired_norm)

    if repair_count == 0 and draft_count > 0:
        return draft_norm
    if draft_count == 0 and repair_count > 0:
        return repaired_norm
    if draft_count == 0 and repair_count == 0:
        return draft_norm

    visible_limit = expected_bubble_count if expected_bubble_count > 0 else len(allowed_times)
    if visible_limit > 0:
        max_reasonable = max(visible_limit + 3, draft_count + 3)
    else:
        max_reasonable = draft_count + 3

    candidates: List[Tuple[str, str]] = [("draft", draft_norm)]

    if repair_count <= max_reasonable:
        draft_cov = side_csv_bubble_coverage(draft_norm, bubble_groups=bubble_groups)[2]
        repair_cov = side_csv_bubble_coverage(repaired_norm, bubble_groups=bubble_groups)[2]
        if not (repair_count <= max(1, draft_count - 2) and repair_cov <= draft_cov + 0.05):
            candidates.append(("repair", repaired_norm))

    if enable_additive:
        additive_norm = merge_side_csv_candidates_additive(
            draft_norm=draft_norm,
            repaired_norm=repaired_norm,
            allowed_times=allowed_times,
            expected_bubble_count=expected_bubble_count,
            bubble_groups=bubble_groups,
        )
        additive_count = count_data_rows(additive_norm)
        if additive_count <= max_reasonable and additive_norm not in {draft_norm, repaired_norm}:
            candidates.append(("additive", additive_norm))

    scored: List[Tuple[float, int, str, str]] = []
    for name, candidate in candidates:
        score = score_side_csv_candidate(
            candidate,
            allowed_times=allowed_times,
            expected_bubble_count=expected_bubble_count,
            bubble_groups=bubble_groups,
        )
        scored.append((score, count_data_rows(candidate), name, candidate))

    draft_score = next(score for score, _count, name, _candidate in scored if name == "draft")
    best_score, _best_count, best_name, best_candidate = max(scored, key=lambda item: (item[0], item[1]))

    # Hysteresis: do not replace the draft for tiny score changes.
    # Additive candidates need a smaller margin because they preserve the draft
    # and only add uncovered evidence-backed rows.
    margin = 0.35 if best_name == "additive" else 1.0
    if best_score >= draft_score + margin:
        return best_candidate

    return draft_norm

# ============================================================
# GENERIC TEXT-POLISH CANDIDATE GUARD
# ============================================================
def _raw_side_csv_rows_for_polish(side_csv: str) -> List[List[str]]:
    """Parse Time/Side/Message rows for text-polish validation."""
    rows: List[List[str]] = []
    try:
        reader = csv.reader(io.StringIO(strip_code_fences(side_csv)))
        for row in reader:
            if not row:
                continue
            if len(row) >= 3 and row[0].strip().lower() == "time":
                continue
            if len(row) < 3:
                continue
            time_value = str(row[0] or "").strip()
            side = str(row[1] or "").strip().upper()
            message = ",".join(row[2:]).strip() if len(row) > 3 else str(row[2] or "").strip()
            if time_value and side in {"LEFT", "RIGHT"} and message:
                rows.append([time_value, side, message])
    except Exception:
        return []
    return rows

def normalize_polished_side_csv_against_reference(
    polished_csv: str,
    reference_csv: str,
    emoji_mode: str = "omit",
) -> str:
    """Normalize a text-polish CSV while preserving reference row structure.

    The polish pass is allowed to edit only Message text. This helper keeps the
    original Time and Side fields from the reference CSV and rejects candidates
    whose row count does not match exactly. This protects row-level precision,
    recall, and timestamp/side structure while still allowing generic spelling,
    casing, apostrophe, and punctuation improvements.
    """
    ref_rows = _raw_side_csv_rows_for_polish(reference_csv)
    pol_rows = _raw_side_csv_rows_for_polish(polished_csv)

    if not ref_rows or len(ref_rows) != len(pol_rows):
        return ""

    out = io.StringIO()
    writer = csv.writer(out, quoting=csv.QUOTE_ALL, lineterminator="\n")
    writer.writerow(["Time", "Side", "Message"])

    for ref, pol in zip(ref_rows, pol_rows):
        message = conservative_clean_message_text(pol[2])
        if emoji_mode == "omit":
            message = strip_emojis(message)
        if not message:
            return ""
        writer.writerow([ref[0], ref[1], message])

    return out.getvalue().strip() + "\n"

def _token_norm_for_text_guard(text: str) -> str:
    """Normalize text for semantic-stability checks in the polish guard."""
    text = strip_emojis(str(text or ""))
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.lower()
    # Normalize common apostrophe/no-apostrophe forms for the guard only.
    text = re.sub(r"\bim\b", "i am", text)
    text = re.sub(r"\bi'm\b", "i am", text)
    text = re.sub(r"\bcant\b", "can not", text)
    text = re.sub(r"\bcan't\b", "can not", text)
    text = re.sub(r"\bdont\b", "do not", text)
    text = re.sub(r"\bdon't\b", "do not", text)
    text = re.sub(r"\byoure\b", "you are", text)
    text = re.sub(r"\byou're\b", "you are", text)
    text = re.sub(r"\bthats\b", "that is", text)
    text = re.sub(r"\bthat's\b", "that is", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def _text_guard_similarity(a: str, b: str) -> float:
    """Return similarity after guard normalization."""
    a_norm = _token_norm_for_text_guard(a)
    b_norm = _token_norm_for_text_guard(b)
    if not a_norm and not b_norm:
        return 1.0
    if not a_norm or not b_norm:
        return 0.0
    if a_norm == b_norm:
        return 1.0
    return SequenceMatcher(None, a_norm, b_norm).ratio()

def _text_artifact_penalty(message: str) -> float:
    """Generic penalty for OCR/punctuation artifacts, not dataset phrases."""
    text = str(message or "")
    penalty = 0.0

    artifact_patterns = [
        r"\bI{2}l\b",
        r"\b1['’]?I[lI]\b",
        r"\bTII\b",
        r"\bJi00\b",
        r"\bIcant\b",
        r"\bIcan't\b",
        r"\bIjust\b",
        r"\bIwill\b",
        r"\bIlove\b",
        r"^will\s+[a-z]",
        r"^feel\s+(?:can|could|will|would|should|must|may|might|am|was)\b",
        r"^thinking\s+(?:it|this|that)\s+(?:would|will|could|should|is|was)\b",
        r"\bGreatl\b",
        r"\bThey\s+The\b",
        r"\byoU\b",
        r"\bsO\b",
        r"\|",
        r"__",
        r"=",
        r"(?<![A-Za-z0-9])\d{1,2}[:.,;]\d{2}(?![A-Za-z0-9])",
    ]
    for pattern in artifact_patterns:
        penalty += 2.0 * len(re.findall(pattern, text))

    penalty += 1.0 * len(re.findall(r"\s+[,.;:!?]", text))
    penalty += 0.5 * len(re.findall(r"[,.;:!?](?=[A-Za-z])", text))
    penalty += 1.0 * len(re.findall(r"\.{4,}", text))
    penalty += 1.0 * len(re.findall(r"[!?]{3,}", text))

    # Unbalanced plain double quotes are often OCR/polish damage.
    if text.count('"') % 2 == 1:
        penalty += 1.0

    words = re.findall(r"[A-Za-z']+", text)
    if len(words) >= 4 and text[:1].islower():
        penalty += 0.75

    return penalty

def _text_style_score(rows: List[List[str]]) -> float:
    """Score generic transcript text cleanliness for polish selection."""
    score = 0.0
    for _time_value, _side, message in rows:
        msg = str(message or "").strip()
        if not msg:
            score -= 20.0
            continue
        score -= _text_artifact_penalty(msg)
        # Very noisy rows should not be preferred, but do not over-penalize
        # ordinary informal chat without terminal punctuation.
        if looks_like_noisy_ocr_text(msg):
            score -= 5.0
        if looks_like_orphan_fragment_message(msg):
            score -= 2.0
    return score

def choose_text_polished_side_csv(
    reference_csv: str,
    polished_csv: str,
    allowed_times: Optional[Set[str]] = None,
    expected_bubble_count: int = 0,
    bubble_groups: Optional[List[Dict[str, str]]] = None,
    min_similarity: float = 0.82,
) -> str:
    """Choose a text-polished candidate only when it preserves structure.

    This is a general exact-match improvement guard. It can improve absolute
    text match by allowing the vision model to polish spelling/casing/punctuation,
    but it refuses changes that alter the number of rows, Time/Side fields, visible
    time validity, bubble coverage, or message semantics too much.
    """
    if not polished_csv:
        return reference_csv

    allowed_times = allowed_times or set()
    ref_rows = _raw_side_csv_rows_for_polish(reference_csv)
    pol_rows = _raw_side_csv_rows_for_polish(polished_csv)

    if not ref_rows or len(ref_rows) != len(pol_rows):
        return reference_csv

    for ref, pol in zip(ref_rows, pol_rows):
        if ref[0] != pol[0] or ref[1] != pol[1]:
            return reference_csv
        if allowed_times and not _row_has_valid_visible_time(pol, allowed_times):
            return reference_csv

        ref_norm = _token_norm_for_text_guard(ref[2])
        pol_norm = _token_norm_for_text_guard(pol[2])
        ref_words = ref_norm.split()
        pol_words = pol_norm.split()

        # For very short rows, require near-identical token content because one
        # changed word can change the whole message.
        if min(len(ref_words), len(pol_words)) <= 3:
            if ref_norm != pol_norm and _text_guard_similarity(ref[2], pol[2]) < 0.92:
                return reference_csv
        elif _text_guard_similarity(ref[2], pol[2]) < min_similarity:
            return reference_csv

        # Avoid candidates that drastically lengthen/shorten a row.
        ref_len = max(1, len(str(ref[2]).strip()))
        pol_len = len(str(pol[2]).strip())
        if pol_len < ref_len * 0.55 or pol_len > ref_len * 1.65:
            return reference_csv

        if looks_like_noisy_ocr_text(pol[2]) and not looks_like_noisy_ocr_text(ref[2]):
            return reference_csv

    ref_candidate_score = score_side_csv_candidate(
        reference_csv,
        allowed_times=allowed_times,
        expected_bubble_count=expected_bubble_count,
        bubble_groups=bubble_groups,
    )
    pol_candidate_score = score_side_csv_candidate(
        polished_csv,
        allowed_times=allowed_times,
        expected_bubble_count=expected_bubble_count,
        bubble_groups=bubble_groups,
    )

    # Do not sacrifice row/bubble-level quality for text polish.
    if pol_candidate_score < ref_candidate_score - 0.25:
        return reference_csv

    ref_style = _text_style_score(ref_rows)
    pol_style = _text_style_score(pol_rows)

    # Prefer the polish when it is structurally safe and not stylistically worse.
    # This intentionally allows punctuation/case-only improvements that may raise
    # absolute exact match while leaving normalized/F1 metrics stable.
    if pol_style >= ref_style - 0.25:
        return polished_csv

    return reference_csv
