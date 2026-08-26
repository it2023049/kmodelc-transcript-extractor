"""Facebook Messenger screenshot-to-CSV extractor."""

import sys
import re
import csv
import io
import json
import argparse
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import cv2
import numpy as np
import easyocr
import ollama
from PyPDF2 import PdfReader

Box = Tuple[int, int, int, int]
ScreenCrop = Tuple[int, int, int, int, np.ndarray]

from extractor_utils import (
    extract_text_from_report,
    extract_year_from_report,
    parse_grid,
    parse_layout,
    trim_white_border,
    ranges_from_indices,
    find_separator_bands,
    split_segments_by_bands,
    manual_grid_split,
    manual_layout_split,
    auto_split_by_white_gutters,
    contour_fallback_split,
    sort_screen_crops,
    minimal_ocr_clean,
    polygon_to_xywh,
    looks_like_date,
    looks_like_time,
    looks_like_date_or_time,
    normalize_visible_time_token,
    parse_ocr_lines,
    extract_allowed_times_from_ocr,
    month_to_number,
    strip_code_fences,
    extract_json_object,
    ollama_chat_text,
    ollama_chat_screen,
    normalize_phone,
    normalize_name,
    clean_name,
    same_name,
    name_in_text,
    build_actor_prompt,
    infer_report_actors,
    infer_report_actors_fallback,
    build_side_evidence,
    force_date_and_year,
    extract_hhmm_from_full_time,
    strip_emojis,
    normalize_text_for_side_overlap,
    count_data_rows,
    merge_side_csvs,
    apply_side_mapping,
    renumber_estimated_facebook_timestamps,
    refine_side_mapping_with_context,
    load_conversation_side_map,
    save_conversation_side_map,
    write_crop,
    choose_best_screen_side_csv,
    normalize_polished_side_csv_against_reference,
    choose_text_polished_side_csv,
    add_sequential_seconds_to_side_csv,
    conservative_clean_message_text,
    postprocess_side_csv_rows,
    side_csv_needs_repair,
    looks_like_orphan_fragment_message,
)

# ============================================================
# IMAGE / COLLAGE SPLITTING
# ============================================================
def separator_score_y(gray: np.ndarray, y: int, band: int = 3) -> float:
    """Scores a horizontal split candidate for wide phone collages."""
    h, w = gray.shape[:2]
    y = max(band, min(h - band - 1, int(y)))
    strip = gray[y - band:y + band + 1, :]
    white = (strip > 245).mean()
    dark = (strip < 30).mean()
    diff = np.abs(gray[y - 1, :].astype(int) - gray[y + 1, :].astype(int)).mean() / 255.0
    return float(white + dark + diff)

def auto_split_wide_phone_collage(image: np.ndarray) -> List[ScreenCrop]:
    """Splits wide phone screenshot collages into individual screens."""
    img = trim_white_border(image)
    h, w = img.shape[:2]

    if h == 0 or w == 0:
        return []

    aspect = w / h

    # A single portrait phone screenshot is usually much narrower than tall.
    if aspect < 0.85:
        return [(0, 0, w, h, img)]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ------------------------------------------------------------
    # Detect row count.
    #
    # Important: a 2x1 side-by-side collage and a 4x2 collage can have almost
    # the same global aspect ratio. We therefore look for a horizontal row
    # boundary near the middle.
    #
    # For clean multi-row collages, a middle split is often the safest fallback.
    # ------------------------------------------------------------
    row_segments = [(0, h)]

    white_pixels = np.all(img >= 245, axis=2)
    row_white_ratio = white_pixels.mean(axis=1)

    mid1 = int(h * 0.49)
    mid2 = int(h * 0.51)
    middle_white_mean = float(row_white_ratio[mid1:mid2].mean()) if mid2 > mid1 else 0.0

    if middle_white_mean >= 0.74:
        split_y = h // 2
        row_segments = [(0, split_y), (split_y, h)]
    else:
        candidate_rows = np.where(row_white_ratio >= 0.95)[0]
        candidate_bands = [
            (a, b)
            for a, b in ranges_from_indices(candidate_rows)
            if (b - a) >= 6 and int(h * 0.40) <= ((a + b) // 2) <= int(h * 0.60)
        ]

        if candidate_bands:
            a, b = min(candidate_bands, key=lambda band: abs(((band[0] + band[1]) / 2) - (h / 2)))
            split_y = int((a + b) / 2)

            if h * 0.40 <= split_y <= h * 0.60:
                row_segments = [(0, split_y), (split_y, h)]

    # ------------------------------------------------------------
    # Estimate columns per row from phone-crop aspect ratio.
    # Messenger screenshot cells in these generated collages are usually
    # around 0.60-0.70 width/height after cropping.
    # ------------------------------------------------------------
    target_phone_aspect = 0.66
    crops: List[ScreenCrop] = []

    for y1, y2 in row_segments:
        row_h = y2 - y1
        if row_h <= 0:
            continue

        row_aspect = w / row_h
        cols = int(round(row_aspect / target_phone_aspect))
        cols = max(1, min(cols, 6))

        # If the whole image is wide but one inferred row would have one column,
        # keep one. Otherwise split into estimated equal columns.
        cell_w = w / cols
        cell_aspect = cell_w / row_h

        # Reject unlikely phone-cell shapes.
        if cols > 1 and not (0.35 <= cell_aspect <= 0.90):
            return [(0, 0, w, h, img)]

        for c in range(cols):
            x1 = int(c * cell_w)
            x2 = int((c + 1) * cell_w) if c < cols - 1 else w

            crop = trim_white_border(img[y1:y2, x1:x2])
            ch, cw = crop.shape[:2]

            if ch < row_h * 0.70 or cw < 120:
                return [(0, 0, w, h, img)]

            crops.append((x1, y1, x2 - x1, y2 - y1, crop))

    if len(crops) <= 1:
        return [(0, 0, w, h, img)]

    return sort_screen_crops(crops)

# Main crop splitter: manual options first, then automatic fallbacks.
def get_screen_crops(
    image_path: str,
    grid: Optional[str] = None,
    layout: Optional[str] = None,
) -> List[ScreenCrop]:
    """Loads an image and returns the best detected screen crops."""
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    if grid and layout:
        raise ValueError("Use either --grid or --layout, not both.")

    if grid:
        return manual_grid_split(image, grid)

    if layout:
        return manual_layout_split(image, layout)

    gutter_crops = auto_split_by_white_gutters(image)
    if len(gutter_crops) > 1:
        return gutter_crops

    wide_crops = auto_split_wide_phone_collage(image)
    if len(wide_crops) > 1:
        return wide_crops

    contour_crops = contour_fallback_split(image)
    if len(contour_crops) > 1:
        return contour_crops

    # One last wide-image check after contour fallback. This catches very clean
    # Messenger side-by-side collages where contour detection sees the full image
    # as one object.
    if len(contour_crops) == 1:
        wide_crops = auto_split_wide_phone_collage(contour_crops[0][4])
        if len(wide_crops) > 1:
            return wide_crops

    return contour_crops

# ============================================================
# OCR WITH POSITIONED BLOCKS
# ============================================================
def position_tag_from_bbox(bbox: Box, crop_w: int, text: str) -> str:
    """Classifies an OCR box as LEFT, RIGHT, or CENTER by geometry."""
    x, y, bw, bh = bbox
    cx = x + bw / 2

    if looks_like_date_or_time(text) and crop_w * 0.25 <= cx <= crop_w * 0.75:
        return "CENTER"

    # Messenger outgoing blue bubbles usually start farther right than incoming gray bubbles.
    # Use the left edge, not the center, to avoid misclassifying wide incoming bubbles.
    # The threshold is deliberately moderate because long right-side bubbles can start nearer the middle.
    if x >= crop_w * 0.30:
        return "RIGHT"

    return "LEFT"

def refine_ocr_block_positions(blocks: List[Dict], crop_w: int) -> List[Dict]:
    """Corrects OCR side labels for fragments on the same visual line."""
    # Notes:
    # word was near the middle of a wide incoming bubble.
    #
    # Example fixed:
    # LEFT: "A piece of metal from an"
    # LEFT: "explosive hit my"
    # RIGHT wrongly: "leg"
    # RIGHT wrongly: "badly:"
    # -> all same visual line becomes LEFT.
    if not blocks:
        return blocks

    editable = []
    for b in blocks:
        text = b["text"]

        if b["pos"] not in {"LEFT", "RIGHT"}:
            continue
        if looks_like_date_or_time(text):
            continue

        editable.append(b)

    editable.sort(key=lambda b: (b["y"] + b["h"] / 2, b["x"]))

    lines = []
    current = []

    for b in editable:
        cy = b["y"] + b["h"] / 2

        if not current:
            current = [b]
            continue

        prev = current[-1]
        prev_cy = prev["y"] + prev["h"] / 2
        threshold = max(18, min(b["h"], prev["h"]) * 0.75)

        if abs(cy - prev_cy) <= threshold:
            current.append(b)
        else:
            lines.append(current)
            current = [b]

    if current:
        lines.append(current)

    for line in lines:
        min_x = min(b["x"] for b in line)

        # If a visual line starts on the left, the whole line belongs to a LEFT bubble.
        # A real right-side bubble should not have any text starting this far left.
        if min_x < crop_w * 0.35:
            for b in line:
                b["pos"] = "LEFT"
        else:
            for b in line:
                b["pos"] = "RIGHT"

    return blocks

# Runs EasyOCR and keeps each text block with side/position metadata.
def extract_ocr_blocks(reader, crop: np.ndarray, screen_index: int) -> str:
    """Runs EasyOCR and returns positioned OCR blocks as text."""
    upscaled = cv2.resize(
        crop,
        None,
        fx=2.0,
        fy=2.0,
        interpolation=cv2.INTER_CUBIC
    )

    h, w = upscaled.shape[:2]

    data = reader.readtext(
        upscaled,
        detail=1,
        paragraph=False,
        mag_ratio=1.0,
        contrast_ths=0.05,
        adjust_contrast=0.7,
        width_ths=0.8,
        y_ths=0.35,
        decoder="greedy",
    )

    blocks = []

    for item in data:
        poly, text = item[0], item[1]
        conf = item[2] if len(item) > 2 else None

        text = minimal_ocr_clean(text)
        if not text:
            continue

        x, y, bw, bh = polygon_to_xywh(poly)
        pos = position_tag_from_bbox((x, y, bw, bh), w, text)

        blocks.append({
            "pos": pos,
            "x": x,
            "y": y,
            "w": bw,
            "h": bh,
            "conf": conf,
            "text": text,
        })

    blocks.sort(key=lambda b: (b["y"], b["x"]))
    blocks = refine_ocr_block_positions(blocks, w)

    lines = [f"[SCREEN {screen_index}]"]

    for idx, b in enumerate(blocks, start=1):
        conf_text = ""
        if isinstance(b["conf"], float):
            conf_text = f" [CONF={b['conf']:.2f}]"

        lines.append(
            f"[BLOCK {idx}] "
            f"[POS={b['pos']}] "
            f"[X={b['x']}] "
            f"[Y={b['y']}] "
            f"[W={b['w']}] "
            f"[H={b['h']}]"
            f"{conf_text} "
            f"{b['text']}"
        )

    return "\n".join(lines)

# ============================================================
# OCR PARSING / SCREEN TIME LIMITS
# ============================================================
def extract_visible_date_from_ocr(
    screen_ocr: str,
    default_year: int,
    previous_date_hint: str = ""
) -> str:
    """Extracts the visible date separator from OCR text."""
    # Notes:
    # Important forensic rule:
    # Do NOT silently reuse the previous screenshot date when OCR cannot see
    # a date in the current screenshot. A wrong carried-over date is worse than
    # an empty date, because the vision model can still read the separator.
    rows = parse_ocr_lines(screen_ocr)

    def normalize_date_text(text: str) -> str:
        """Normalizes noisy OCR text before Messenger date parsing."""
        t = str(text or "")
        t = t.replace(".", ":")
        t = t.replace(";", ":")
        t = t.replace("|", "I")
        # OCR often reads zero as O only inside numbers.
        t = re.sub(r"(?<=\d)[oO](?=\d)", "0", t)
        t = re.sub(r"(?<=\s)[oO](?=\d)", "0", t)
        t = re.sub(r"(?<=\d)[lI](?=\d)", "1", t)
        t = re.sub(r"\s+", " ", t)
        return t.strip()

    month_pat = (
        r"Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|"
        r"Aug|August|Sep|Sept|September|Oct|October|Nov|November|Dec|December"
    )

    # 1. Prefer OCR blocks in the conversation/date-separator region.
    candidate_texts = []
    for row in rows:
        text = normalize_date_text(row.get("text", ""))
        if not text:
            continue

        try:
            y = int(row.get("y", "0"))
        except Exception:
            y = 0

        # Ignore status bar/header as much as possible, but keep the central
        # Messenger separator around the upper-middle of the crop.
        if y >= 80:
            candidate_texts.append(text)

    # 2. Also join adjacent OCR text because EasyOCR may split the date line.
    candidate_texts.append(" ".join(candidate_texts))
    candidate_texts.append(normalize_date_text(screen_ocr))

    for text in candidate_texts:
        m = re.search(
            rf"\b({month_pat})\s+(\d{{1,2}})(?:,?\s+(20\d{{2}}))?",
            text,
            flags=re.I,
        )
        if not m:
            continue

        month = month_to_number(m.group(1))
        day = int(m.group(2))
        year = int(m.group(3)) if m.group(3) else default_year

        if month and 1 <= day <= 31:
            return f"{day:02d}/{month:02d}/{year}"

    return ""

def infer_visible_date_with_ai(
    model: str,
    image_path: str,
    screen_ocr: str,
    default_year: int,
    use_vision: bool = True,
) -> str:
    """Uses the vision model to recover a missing Messenger date separator."""
    # Notes:
    # Used only when OCR cannot extract the date. It asks the model for the
    # central Messenger separator date, not for chat reconstruction.
    if not use_vision:
        return ""

    prompt = f"""
Read only the central Facebook Messenger date/time separator from this screenshot.
Ignore the phone status bar time and all chat messages.

OCR support:
{screen_ocr}

Return only JSON:
{{"date":"DD/MM/YYYY"}}

Rules:
- Use year {default_year} if the year is not visible.
- If the date is not visible, return {{"date":""}}.
- Do not explain.
"""

    raw = ollama_chat_screen(
        model=model,
        prompt=prompt,
        image_path=image_path,
        use_vision=True,
    )

    try:
        data = json.loads(extract_json_object(raw))
        date = str(data.get("date", "")).strip()
    except Exception:
        m = re.search(r"(\d{2}/\d{2}/20\d{2})", raw)
        date = m.group(1) if m else ""

    if re.fullmatch(r"\d{2}/\d{2}/20\d{2}", date):
        return date

    return ""

# ============================================================
# REPORT ACTORS + SIDE MAP
# ============================================================
def find_header_match(ocr_data: str, actors: Dict) -> str:
    """Matches top header OCR against known report actors."""
    # Notes:
    # This function deterministically checks the top OCR area for participant names
    # and contact numbers from the report.
    rows = parse_ocr_lines(ocr_data)
    participants = [p for p in actors.get("participants", []) if p.get("name")]

    if not rows or not participants:
        return ""

    header_rows = []

    for row in rows:
        try:
            y = int(row["y"])
        except ValueError:
            continue

        # Coordinates are upscaled. Header/contact area is near the top.
        # This includes name and phone, but excludes message bubbles.
        if y <= 350:
            header_rows.append(row)

    if not header_rows:
        return ""

    header_text = " ".join(row["text"] for row in header_rows)
    header_digits = normalize_phone(header_text)

    scores = {}

    for p in participants:
        name = clean_name(p.get("name", ""))
        if not name:
            continue

        score = 0

        # Name match in header.
        if name_in_text(name, header_text):
            score += 10

        # Phone match in header.
        for phone in p.get("contact_numbers", []) or []:
            phone_digits = normalize_phone(phone)
            if phone_digits and phone_digits in header_digits:
                score += 20

        if score > 0:
            scores[name] = score

    if not scores:
        return ""

    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[0][0]

def deterministic_messenger_side_map(
    actors: Dict,
    ocr_data: str
) -> Optional[Dict[str, str]]:
    """Infers Messenger LEFT/RIGHT names from header and contact evidence."""
    # Notes:
    # Main rule:
    # If the Messenger header shows a participant/contact from the report, then:
    # LEFT = header participant/contact
    # RIGHT = victim / screenshot owner
    #
    # This matches the usual Messenger layout:
    # LEFT gray bubbles = incoming messages from the other participant.
    # RIGHT blue bubbles = outgoing messages from the screenshot owner.
    victim = clean_name(actors.get("victim", ""))

    participants = [
        p for p in actors.get("participants", [])
        if p.get("name")
    ]

    # If victim field is empty, recover it from participants.
    if not victim:
        for p in participants:
            if str(p.get("role", "")).lower() == "victim":
                victim = clean_name(p.get("name", ""))
                break

    # -------------------------
    # 1. Header evidence wins
    # -------------------------
    header_name = find_header_match(ocr_data, actors)

    if header_name and victim and not same_name(header_name, victim):
        return {
            "LEFT": clean_name(header_name),
            "RIGHT": clean_name(victim)
        }

    # -------------------------
    # 2. Contact/phone evidence fallback
    # -------------------------
    # Some Messenger captures may include phone/account details in header or profile snippets.
    rows = parse_ocr_lines(ocr_data)

    number_to_name = {}

    for p in participants:
        name = clean_name(p.get("name", ""))

        for num in p.get("contact_numbers", []) or []:
            num_norm = normalize_phone(num)
            if num_norm:
                number_to_name[num_norm] = name

    side_hits = {}

    for row in rows:
        row_digits = normalize_phone(row["text"])
        if not row_digits:
            continue

        for num_norm, owner_name in number_to_name.items():
            if num_norm and num_norm in row_digits:
                pos = row["pos"]

                try:
                    y = int(row["y"])
                except ValueError:
                    y = 9999

                # Header contact evidence maps to LEFT in Messenger.
                if pos == "CENTER" and y <= 350:
                    pos = "LEFT"

                if pos in {"LEFT", "RIGHT"}:
                    side_hits[pos] = clean_name(owner_name)

    if "LEFT" in side_hits and victim and not same_name(side_hits["LEFT"], victim):
        return {
            "LEFT": clean_name(side_hits["LEFT"]),
            "RIGHT": clean_name(victim)
        }

    if "RIGHT" in side_hits and victim and not same_name(side_hits["RIGHT"], victim):
        return {
            "RIGHT": clean_name(side_hits["RIGHT"]),
            "LEFT": clean_name(victim)
        }

    return None

def build_side_map_prompt(
    report_text: str,
    actors: Dict,
    ocr_data: str,
    platform_hint: str
) -> str:
    """Builds the platform-specific LEFT/RIGHT speaker mapping prompt."""
    side_evidence = build_side_evidence(ocr_data)

    return f"""
Determine the fixed LEFT/RIGHT speaker mapping for this Facebook Messenger two-person chat.

Platform: {platform_hint}

ACTORS JSON:
{json.dumps(actors, ensure_ascii=False, indent=2)}

CASE REPORT:
{report_text}

{side_evidence}

Return only JSON:
{{"LEFT":"speaker name","RIGHT":"speaker name"}}

Rules:
1. Decide the mapping once. Do not map row-by-row.
2. In Facebook Messenger, the top header usually shows the other participant/contact, not the screenshot owner.
3. In Facebook Messenger, LEFT gray bubbles are usually incoming from the header contact. RIGHT blue bubbles are outgoing from the screenshot owner.
4. Direct contact evidence wins: if a phone/contact/account detail from the report appears on a side, that side belongs to that detail's owner.
5. Use message context as corroborating evidence. A greeting or direct address such as "Hello Alice" or "Alice, can I ask you something?" usually identifies Alice as the receiver, while "This is Bob Example" identifies the sender.
6. A third-party mention such as "I spoke with Casey Example" does not make Casey Example the sender or receiver of that row.
7. Compare evidence across the whole conversation and the case-report timeline; do not decide from one ambiguous sentence.
8. If the header/contact is a suspect and the victim is the screenshot owner, map LEFT=suspect and RIGHT=victim.
9. Return two distinct human names only. Never return an organisation, role title, or combined identity.
10. Do not output explanations.
"""

# Builds the final LEFT/RIGHT speaker mapping before CSV conversion.
def infer_side_mapping(
    report_text: str,
    actors: Dict,
    ocr_data: str,
    platform_hint: str,
    model: str
) -> Dict[str, str]:
    """Returns the final fixed LEFT/RIGHT speaker mapping."""
    deterministic = deterministic_messenger_side_map(actors, ocr_data)

    if deterministic:
        return deterministic

    prompt = build_side_map_prompt(
        report_text=report_text,
        actors=actors,
        ocr_data=ocr_data,
        platform_hint=platform_hint,
    )

    raw = ollama_chat_text(model, prompt)

    try:
        data = json.loads(extract_json_object(raw))
    except json.JSONDecodeError:
        raise ValueError(f"Could not parse side mapping JSON:\n{raw}")

    left = clean_name(data.get("LEFT", ""))
    right = clean_name(data.get("RIGHT", ""))

    if not left or not right or normalize_name(left) == normalize_name(right):
        raise ValueError(f"Invalid side map:\n{data}")

    return {
        "LEFT": left,
        "RIGHT": right,
    }

# ============================================================
# SCREEN EXTRACTION PROMPTS
# ============================================================
def build_messenger_bubble_groups(screen_ocr: str) -> List[Dict[str, str]]:
    """Groups Messenger OCR blocks into approximate chat bubbles."""
    # Notes:
    # Unlike Viber, Messenger usually does not show a timestamp on every bubble.
    # These groups are used as a conservative side/bubble hint.
    rows = parse_ocr_lines(screen_ocr)

    def to_int(value: str, default: int = 0) -> int:
        """Safely converts a value to int with a fallback."""
        try:
            return int(value)
        except Exception:
            return default

    # Find the last top date/time separator area.
    date_y = None
    for row in rows:
        text = row.get("text", "")
        if looks_like_date(text) or re.search(r"\b\d{1,2}[:.]\d{2}\b", text):
            y = to_int(row.get("y", "0"), -1)
            if y > 100:
                date_y = y if date_y is None else max(date_y, y)

    usable = []
    for row in rows:
        text = row.get("text", "").strip()
        pos = row.get("pos", "").upper()
        y = to_int(row.get("y", "0"))

        if not text or pos not in {"LEFT", "RIGHT"}:
            continue

        if date_y is not None and y <= date_y:
            continue
        if date_y is None and y < 250:
            continue

        if is_ui_message(text) or looks_like_date_or_time(text):
            continue

        usable.append(row)

    usable = sorted(
        usable,
        key=lambda r: (
            to_int(r.get("y", "0")),
            to_int(r.get("x", "0")),
            to_int(r.get("block", "0")),
        )
    )

    if not usable:
        return []

    heights = [max(1, to_int(r.get("h", "1"), 1)) for r in usable]
    median_h = int(np.median(heights)) if heights else 40
    gap_threshold = max(8, int(median_h * 0.25))

    raw_groups: List[List[Dict[str, str]]] = []

    for row in usable:
        if not raw_groups:
            raw_groups.append([row])
            continue

        prev = raw_groups[-1][-1]

        y = to_int(row.get("y", "0"))
        prev_y = to_int(prev.get("y", "0"))
        prev_h = to_int(prev.get("h", "0"))

        y_gap = y - (prev_y + prev_h)
        same_side = row.get("pos", "").upper() == prev.get("pos", "").upper()

        # New bubble if side changes or if there is a clear vertical gap.
        # Wrapped lines inside the same bubble usually have very small/negative gap.
        if (not same_side) or y_gap > gap_threshold:
            raw_groups.append([row])
        else:
            raw_groups[-1].append(row)

    groups: List[Dict[str, str]] = []

    for group in raw_groups:
        group_sorted = sorted(
            group,
            key=lambda r: (
                to_int(r.get("y", "0")),
                to_int(r.get("x", "0")),
                to_int(r.get("block", "0")),
            )
        )

        side_scores = {"LEFT": 0, "RIGHT": 0}
        for r in group_sorted:
            side = r.get("pos", "").upper()
            if side in side_scores:
                side_scores[side] += max(1, len(r.get("text", "")))

        side = "LEFT" if side_scores["LEFT"] >= side_scores["RIGHT"] else "RIGHT"
        text = " ".join(r.get("text", "").strip() for r in group_sorted if r.get("text", "").strip())
        text = re.sub(r"\s+", " ", text).strip()

        if text:
            groups.append({
                "side": side,
                "text": text,
                "order": len(groups),
            })

    return groups

def build_messenger_bubble_hints(screen_ocr: str) -> str:
    """Formats Messenger OCR bubble groups as prompt hints."""
    # Notes:
    # The hints are not treated as final transcript text; they are noisy OCR
    # support only.
    groups = build_messenger_bubble_groups(screen_ocr)

    if not groups:
        return "No reliable bubble hints."

    lines = []
    for i, group in enumerate(groups, start=1):
        lines.append(f"[BUBBLE {i}] SIDE={group.get('side', '')} OCR_TEXT={group.get('text', '')}")

    return "\n".join(lines)

def build_screen_prompt(
    screen_ocr: str,
    screen_index: int,
    visible_date: str,
    default_year: int,
    allowed_times: Set[str],
    emoji_mode: str,
    bubble_hints: str = "",
) -> str:
    """Builds the extraction prompt for one screenshot."""
    allowed_times_text = ", ".join(sorted(allowed_times)) if allowed_times else "unknown"
    if emoji_mode == "vision":
        emoji_rule = "Include only emojis that are clearly and unambiguously visible in the screenshot. Do not infer, guess, or substitute emojis. If unsure, omit the emoji."
    else:
        emoji_rule = "Do not include emojis in the CSV. Omit all emojis, because wrong emojis are worse than missing emojis in bulk forensic extraction."

    return f"""
You are extracting messages from ONE Facebook Messenger screenshot image.
Use the attached image as the primary source. Use OCR blocks only as support for side, visible date/time, and bubble boundaries. OCR text can be noisy; do not copy OCR word-order artifacts when the image is clearer.

SCREEN INDEX: {screen_index}
VISIBLE DATE FOR THIS SCREEN: {visible_date or "unknown - read it from the central Messenger date separator in the screenshot"}
DEFAULT YEAR: {default_year}
ALLOWED BUBBLE TIMES FOR THIS SCREEN: {allowed_times_text}

OCR BLOCKS:
{screen_ocr}

MESSENGER BUBBLE HINTS FROM OCR GEOMETRY:
{bubble_hints or "No reliable bubble hints."}

Return only CSV with this header exactly once:
"Time","Side","Message"

Rules:
1. Output only real human chat bubbles.
2. Ignore UI: status bar clock, battery, Messenger header/contact name, profile/header snippets, call/video/search icons, Active now, Sent, Delivered, Seen, read receipts, date-only separators, text input box, Like button, stickers/GIF labels, and empty OCR noise.
3. Side must be LEFT or RIGHT based only on visible bubble position/color in the image.
4. LEFT = gray incoming bubble. RIGHT = blue outgoing bubble.
5. Do not output participant names. Do not infer side from meaning or from victim/scammer roles.
6. Reconstruct each separate visible rounded chat bubble as one CSV row. Do not output one row per OCR line.
7. Messenger often shows one central date/time separator for many bubbles. Do NOT merge all bubbles after the same timestamp into one message.
8. Never merge two adjacent bubbles just because they have the same sender, same side, or same visible timestamp. If there is a separate rounded bubble boundary, it is a separate CSV row.
9. Use MESSENGER BUBBLE HINTS as a guide for split boundaries and visual order: normally each [BUBBLE n] should become one CSV row unless it is clearly UI noise. Use the image, not noisy OCR text, for final message wording.
10. The output rows must follow the exact visible top-to-bottom order of rounded bubbles in the screenshot. When several rows reuse the same central Messenger timestamp, never reorder them by text meaning or by timestamp; keep visual order.
11. Before the final CSV, compare your rows with MESSENGER BUBBLE HINTS and the screenshot. If a visible human bubble is missing, add it in the correct visual/top-to-bottom order, not at the end.
12. Do not add OCR fragments that duplicate an existing row, even if the OCR wording is noisier.
13. Do not drop the last visible chat bubble near the bottom of the screenshot. A blue/gray rounded bubble above the composer/input bar is still a human message.
14. If one bubble wraps across multiple OCR lines, merge only the lines inside that same rounded bubble; do not output wrapped lines as separate CSV rows.
15. If a draft row starts with a lowercase fragment or has a question in reversed order, re-read that visible bubble from the screenshot and place it where the bubble appears.
11. If a bubble has no individual timestamp, use the visible Messenger screen/date-time separator timestamp for that bubble. Do not fabricate per-message minute offsets.
12. OCR COVERAGE CHECK: Every non-UI LEFT/RIGHT OCR text block must be represented in an output message, but grouped by visible bubble boundaries.
12. Do not shorten messages. Do not keep only the first line of a bubble.
13. Emoji rule: {emoji_rule}
14. Preserve evidence exactly: amounts, names, countries, cities, receiver details, phone numbers, account/payment details, reference numbers, threats, instructions.
15. Time must be the bubble's visible time when present. When individual bubble times are hidden, reuse the visible Messenger screen/date-time separator timestamp combined with VISIBLE DATE FOR THIS SCREEN. Format: "DD/MM/YYYY HH:MM".
16. Only use times from ALLOWED BUBBLE TIMES FOR THIS SCREEN when they are available. Do not use the status bar time.
17. Fix only obvious OCR mistakes when the image clearly supports it: Im -> I'm, Icant -> I can't, IIl/1'Il -> I'll, Ijust -> I just, Iwill -> I will, Ilove -> I love, | -> I, $ -> s, and remove leaked HH:MM tokens from message text.
18. Do not invent, summarize, or add messages from other screenshots.
19. Use exactly three quoted CSV fields per row. No markdown, no explanations.

Final CSV:
"""

def build_screen_repair_prompt(
    draft_csv: str,
    screen_ocr: str,
    screen_index: int,
    visible_date: str,
    default_year: int,
    allowed_times: Set[str],
    emoji_mode: str,
    bubble_hints: str = "",
) -> str:
    """Builds the repair prompt for one screenshot CSV draft."""
    allowed_times_text = ", ".join(sorted(allowed_times)) if allowed_times else "unknown"
    if emoji_mode == "vision":
        emoji_rule = "Include only emojis that are clearly and unambiguously visible in the screenshot. Do not infer, guess, or substitute emojis. If unsure, omit the emoji."
    else:
        emoji_rule = "Do not include emojis in the CSV. Omit all emojis, because wrong emojis are worse than missing emojis in bulk forensic extraction."

    return f"""
Repair this Facebook Messenger screen CSV using the attached screenshot image and OCR blocks.

SCREEN INDEX: {screen_index}
VISIBLE DATE FOR THIS SCREEN: {visible_date or "unknown - read it from the central Messenger date separator in the screenshot"}
DEFAULT YEAR: {default_year}
ALLOWED BUBBLE TIMES FOR THIS SCREEN: {allowed_times_text}

OCR BLOCKS:
{screen_ocr}

MESSENGER BUBBLE HINTS FROM OCR GEOMETRY:
{bubble_hints or "No reliable bubble hints."}

BAD / INCOMPLETE DRAFT CSV:
{draft_csv}

Return only final CSV with this header exactly once:
"Time","Side","Message"

Rules:
1. Rebuild only from this screenshot image and its OCR blocks.
2. Keep every real visible message bubble from this screenshot.
3. Remove all UI rows: status bar clock, battery, Messenger header/contact name, profile/header snippets, call/video/search icons, Active now, Sent, Delivered, Seen, read receipts, date separator as message, text input box, Like button, stickers/GIF labels, and empty OCR noise.
4. Side must be LEFT or RIGHT based on visible bubble position/color. LEFT = gray incoming. RIGHT = blue outgoing. Do not output names.
5. Messenger often shows one central date/time separator for many bubbles. Do NOT merge all bubbles after the same timestamp into one message.
6. Keep each separate visible rounded chat bubble as one CSV row, even if adjacent bubbles have the same sender, same side, or same visible timestamp.
7. Use MESSENGER BUBBLE HINTS as a guide for split boundaries and visual order: normally each [BUBBLE n] should become one CSV row unless it is clearly UI noise. Use the image, not noisy OCR text, for final message wording.
8. The output rows must follow the exact visible top-to-bottom order of rounded bubbles in the screenshot. When several rows reuse the same central Messenger timestamp, never reorder them by text meaning or by timestamp; keep visual order.
9. Before the final CSV, compare your rows with MESSENGER BUBBLE HINTS and the screenshot. If a visible human bubble is missing, add it in the correct visual/top-to-bottom order, not at the end.
9. Do not add OCR fragments that duplicate an existing row, even if the OCR wording is noisier.
10. Do not drop the last visible chat bubble near the bottom of the screenshot. A blue/gray rounded bubble above the composer/input bar is still a human message.
11. If the draft merged two or more visible bubbles into one row, split them back into separate rows using the screenshot.
8. OCR COVERAGE CHECK: Every non-UI LEFT/RIGHT OCR text block must be represented in an output message, but grouped by visible bubble boundaries.
9. If the draft omitted an OCR line from a bubble, add it back.
10. Do not shorten messages. Do not keep only the first line of a bubble.
11. Emoji rule: {emoji_rule}
12. Merge wrapped lines only when they are inside the same visible rounded bubble.
13. Use visible bubble/screen times as anchors. If Messenger hides individual bubble times, reuse the same visible anchor time for those bubbles instead of inventing minute offsets.
14. If VISIBLE DATE FOR THIS SCREEN is unknown, read the central Messenger date separator from the screenshot image. Never reuse a date from another screenshot.
15. Combine each time with VISIBLE DATE FOR THIS SCREEN or the date read from the screenshot. Format: "DD/MM/YYYY HH:MM".
16. Preserve evidence: amounts, names, receiver details, locations, phone numbers, accounts, references, threats, instructions.
16. Do not invent messages and do not include messages from other screenshots.
17. Use exactly three quoted CSV fields per row. No markdown, no explanations.

Final CSV:
"""

def build_text_polish_prompt(
    screen_ocr: str,
    screen_index: int,
    visible_date: str,
    side_csv: str,
    emoji_mode: str,
    bubble_hints: str = "",
) -> str:
    """Builds a guarded text-only polish prompt for exact-match improvement."""
    row_count = count_data_rows(side_csv)
    if emoji_mode == "vision":
        emoji_rule = "Keep only emojis that are clearly visible in the screenshot; do not guess or substitute emojis."
    else:
        emoji_rule = "Do not include emojis. Omit emojis from Message text."

    return f"""
Polish the Message text in this Facebook Messenger side CSV using the attached screenshot as the primary source and OCR only as support.

SCREEN INDEX: {screen_index}
VISIBLE DATE: {visible_date or "unknown"}
EXPECTED ROW COUNT: {row_count}

OCR BLOCKS:
{screen_ocr}

MESSENGER BUBBLE HINTS:
{bubble_hints or "No reliable bubble hints."}

CURRENT SIDE CSV:
{side_csv}

Return only CSV with this header exactly once:
"Time","Side","Message"

Rules:
1. Keep exactly {row_count} data rows. Do not add, remove, split, merge, reorder, or duplicate rows.
2. Keep each row's Time and Side exactly as in CURRENT SIDE CSV.
3. Edit only the Message field.
4. Use the screenshot to fix only obvious OCR/casing/apostrophe/punctuation issues.
5. Fix generic OCR prefix artifacts only when the screenshot clearly supports them: missing initial "I" before first-person verbs, compact I+verb tokens, l/1 read as ! or I at sentence boundaries, and duplicated leading words caused by OCR line overlap.
6. Preserve the same meaning and all evidence values exactly: names, amounts, locations, phone numbers, account/payment details, reference numbers, dates, threats, and instructions.
7. Do not paraphrase, summarize, expand abbreviations, or invent missing text.
8. Do not change actor/person names or evidence values to make punctuation look nicer.
9. Do not change a word unless the screenshot clearly supports the correction.
10. {emoji_rule}
11. Use exactly three quoted CSV fields per row. No markdown, no explanations.

Final polished CSV:
"""

# ============================================================
# CSV NORMALIZATION
# ============================================================
def is_ui_message(message: str) -> bool:
    """Detects platform UI text that should not become a chat row."""
    # Notes:
    # Avoid case-specific names, phone numbers, or conversation content.
    m = str(message or "").strip()
    low = m.lower()

    if not m:
        return True

    # Standalone dates/times are UI, not chat messages.
    if looks_like_date_or_time(m):
        return True

    ui_substrings = [
        "active now",
        "messenger",
        "facebook",
        "you're friends on facebook",
        "you are friends on facebook",
        "wave to",
        "say hi",
        "sent",
        "delivered",
        "seen",
        "typing",
        "search conversation",
        "view profile",
        "voice call",
        "video call",
        "missed call",
        "started a call",
        "changed the chat",
        "end-to-end encrypted",
        "end-to-end encryption",
        "messages and calls are secured",
        "message...",
        "type a message",
        "write a message",
        "aa",
    ]

    if any(part in low for part in ui_substrings):
        return True

    # Messenger reactions/read receipts/icon OCR noise.
    if re.fullmatch(r"(like|thumbs up|gif|sticker|photo|camera|mic|microphone|send)", low, flags=re.IGNORECASE):
        return True

    if re.fullmatch(r"[0o]{2,4}", low):
        return True

    if re.fullmatch(r"[✓✔v]{1,4}", low):
        return True

    return False

def clean_message_text(message: str) -> str:
    """Apply minimal, generic cleanup without semantic rewriting."""
    return conservative_clean_message_text(message)

def build_ocr_bubble_groups(screen_ocr: str) -> List[Dict[str, str]]:
    """Groups OCR blocks into timestamped bubble hints."""
    # Notes:
    # The LLM may sometimes assign LEFT/RIGHT incorrectly. For Facebook Messenger screenshots,
    # the OCR geometry is more reliable for side than the LLM. Each group ends at
    # the visible bubble timestamp that follows the bubble text.
    #
    # Returns groups like:
    # {"time": "11:02", "side": "RIGHT", "text": "..."}
    rows = parse_ocr_lines(screen_ocr)

    def to_int(value: str, default: int = 0) -> int:
        """Safely converts a value to int with a fallback."""
        try:
            return int(value)
        except Exception:
            return default

    rows = sorted(
        rows,
        key=lambda r: (
            to_int(r.get("y", "0")),
            to_int(r.get("x", "0")),
            to_int(r.get("block", "0")),
        )
    )

    date_y = None
    for row in rows:
        if looks_like_date(row.get("text", "")):
            y = to_int(row.get("y", "0"), -1)
            if y >= 0:
                date_y = y if date_y is None else max(date_y, y)

    current: List[Dict[str, str]] = []
    groups: List[Dict[str, str]] = []

    for row in rows:
        text = row.get("text", "").strip()
        pos = row.get("pos", "").upper()
        y = to_int(row.get("y", "0"), 0)

        # Ignore status/header area before the date separator.
        if date_y is not None and y <= date_y:
            continue
        if date_y is None and y < 250:
            continue

        visible_time = normalize_visible_time_token(text)

        if visible_time:
            bubble_rows = [
                r for r in current
                if r.get("pos", "").upper() in {"LEFT", "RIGHT"}
                and not looks_like_date_or_time(r.get("text", ""))
                and not is_ui_message(r.get("text", ""))
            ]

            if bubble_rows:
                side_scores = {"LEFT": 0, "RIGHT": 0}

                for br in bubble_rows:
                    side = br.get("pos", "").upper()
                    if side in side_scores:
                        # Character-weighted voting is more stable than row count
                        # when OCR splits one line into multiple fragments.
                        side_scores[side] += max(1, len(br.get("text", "").strip()))

                total_score = side_scores["LEFT"] + side_scores["RIGHT"]
                if total_score <= 0:
                    side = "UNKNOWN"
                    confidence = 0.0
                else:
                    side = "LEFT" if side_scores["LEFT"] >= side_scores["RIGHT"] else "RIGHT"
                    confidence = max(side_scores["LEFT"], side_scores["RIGHT"]) / total_score

                group_text = " ".join(br.get("text", "").strip() for br in bubble_rows if br.get("text", "").strip())

                groups.append({
                    "time": visible_time,
                    "side": side,
                    "confidence": f"{confidence:.3f}",
                    "text": group_text,
                    "order": len(groups),
                })

            current = []
            continue

        if pos in {"LEFT", "RIGHT"} and text and not is_ui_message(text) and not looks_like_date_or_time(text):
            current.append(row)

    return groups

def infer_side_for_message_from_ocr(
    message: str,
    hhmm: Optional[str],
    ocr_bubble_groups: Optional[List[Dict[str, str]]],
) -> Optional[str]:
    """Corrects message side using matching OCR bubble groups."""
    # Notes:
    # If a timestamp is unique in a screenshot, use its OCR side directly.
    # If the same HH:MM appears in multiple bubbles, choose by fuzzy token overlap.
    if not hhmm or not ocr_bubble_groups:
        return None

    candidates = [
        g for g in ocr_bubble_groups
        if g.get("time") == hhmm
        and g.get("side") in {"LEFT", "RIGHT"}
        and float(g.get("confidence", "1.0") or 0.0) >= 0.62
    ]

    if not candidates:
        return None

    if len(candidates) == 1:
        return candidates[0]["side"]

    message_tokens = normalize_text_for_side_overlap(message)

    best_side = None
    best_score = -1

    for group in candidates:
        group_tokens = normalize_text_for_side_overlap(group.get("text", ""))
        overlap = len(message_tokens & group_tokens)
        # Add a small score for larger group coverage to break ties.
        score = overlap * 100 + min(len(group_tokens), 99)

        if score > best_score:
            best_score = score
            best_side = group.get("side")

    return best_side if best_side in {"LEFT", "RIGHT"} else None

def normalize_for_fragment_check(text: str) -> str:
    """Normalizes Messenger text for fragment and duplicate checks."""
    text = str(text or "").lower()

    # Normalize common OCR spelling variants before duplicate/overlap checks.
    text = text.replace("’", "'")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def token_overlap_score(a: str, b: str) -> float:
    """Scores token overlap while preserving duplicate-token counts."""
    a_norm = normalize_for_fragment_check(a)
    b_norm = normalize_for_fragment_check(b)

    if not a_norm or not b_norm:
        return 0.0

    a_tokens = a_norm.split()
    b_tokens = b_norm.split()

    if not a_tokens or not b_tokens:
        return 0.0

    # Count token overlap with multiplicity.
    b_counts: Dict[str, int] = {}
    for tok in b_tokens:
        b_counts[tok] = b_counts.get(tok, 0) + 1

    overlap = 0
    for tok in a_tokens:
        if b_counts.get(tok, 0) > 0:
            overlap += 1
            b_counts[tok] -= 1

    return overlap / max(1, min(len(a_tokens), len(b_tokens)))

def infer_side_from_messenger_bubbles(
    message: str,
    messenger_bubble_groups: Optional[List[Dict[str, str]]] = None,
) -> Optional[str]:
    """Corrects Messenger side labels using bubble hint overlap."""
    # Notes:
    # The LLM sometimes swaps LEFT/RIGHT based on meaning. OCR geometry is often
    # enough to correct this if the message strongly matches a bubble hint.
    if not messenger_bubble_groups:
        return None

    msg_norm = normalize_for_fragment_check(message)
    if len(msg_norm.split()) < 2:
        return None

    best_side = None
    best_score = 0.0

    for group in messenger_bubble_groups:
        side = str(group.get("side", "")).upper()
        text = group.get("text", "")

        if side not in {"LEFT", "RIGHT"}:
            continue

        score = token_overlap_score(msg_norm, text)

        if score > best_score:
            best_score = score
            best_side = side

    if best_score >= 0.72:
        return best_side

    return None

def split_message_by_messenger_bubbles(
    message: str,
    side: str,
    messenger_bubble_groups: Optional[List[Dict[str, str]]] = None,
) -> List[str]:
    """Splits merged Messenger rows using OCR bubble hints."""
    # Notes:
    # the row back using OCR geometry bubble hints.
    #
    # Conservative: only split when at least two same-side OCR bubble texts are
    # clearly contained in the LLM message in the same order.
    if not messenger_bubble_groups:
        return [message]

    msg_norm = normalize_for_fragment_check(message)
    if len(msg_norm.split()) < 6:
        return [message]

    matches = []
    last_pos = -1

    for group in messenger_bubble_groups:
        g_side = str(group.get("side", "")).upper()
        if g_side != side:
            continue

        g_text = str(group.get("text", "")).strip()
        g_norm = normalize_for_fragment_check(g_text)
        g_words = g_norm.split()

        if len(g_words) < 3:
            continue

        pos = msg_norm.find(g_norm)
        if pos >= 0 and pos > last_pos:
            matches.append(g_text)
            last_pos = pos
            continue

        # Fallback for slightly cleaned LLM text.
        if token_overlap_score(message, g_text) >= 0.92 and len(g_words) >= 4:
            matches.append(g_text)

    if len(matches) >= 2:
        return matches

    return [message]

def is_short_contained_duplicate(
    message: str,
    time_value: str,
    side: str,
    emitted_rows: List[Tuple[str, str, str]]
) -> bool:
    """Detects short duplicate fragments already contained in earlier rows."""

    msg_norm = normalize_for_fragment_check(message)
    word_count = len(msg_norm.split())

    if word_count == 0:
        return True

    if word_count > 3:
        return False

    for prev_time, prev_side, prev_msg in emitted_rows:
        if prev_time != time_value or prev_side != side:
            continue

        prev_norm = normalize_for_fragment_check(prev_msg)

        if msg_norm and msg_norm in prev_norm:
            return True

    return False

def should_prefer_messenger_bubble_text(message: str, bubble_text: str) -> bool:
    """Decide whether an OCR bubble hint is safer than a suspicious draft row.

    This is deliberately narrow: the bubble hint is used as final text only when
    the draft row is structurally suspicious but both strings clearly describe
    the same visible bubble. This avoids general semantic rewriting.
    """
    msg = str(message or "").strip()
    bubble = str(bubble_text or "").strip()
    if not msg or not bubble:
        return False

    msg_norm = normalize_for_fragment_check(msg)
    bubble_norm = normalize_for_fragment_check(bubble)
    msg_words = msg_norm.split()
    bubble_words = bubble_norm.split()
    if len(msg_words) < 3 or len(bubble_words) < 3:
        return False

    overlap = token_overlap_score(msg, bubble)
    if overlap < 0.86:
        return False

    # Use bubble text only for rows that already look broken: lowercase orphan
    # questions/fragments, missing initial-I artifacts, or obvious OCR punctuation.
    suspicious = (
        looks_like_orphan_fragment_message(msg)
        or bool(re.search(r"\b(?:I{2}l|1['’]?ll|Ijust|Iwill|Icant|yoU|sO)\b", msg))
    )
    if not suspicious:
        return False

    # Do not inject visibly noisy OCR.
    if re.search(r"(?<![A-Za-z0-9])\d{1,2}[:.,;]\d{2}(?![A-Za-z0-9])", bubble):
        return False
    if re.search(r"[|_=]", bubble):
        return False

    # Prefer bubble text when it has a more plausible sentence start or more
    # complete wording, but not just because it is longer.
    msg_starts_lower = bool(msg[:1].islower())
    bubble_starts_upper = bool(bubble[:1].isupper())
    if msg_starts_lower and bubble_starts_upper:
        return True

    if len(bubble_words) >= len(msg_words) + 1 and not bubble_starts_upper:
        return False

    # Same token set but different word order: choose the bubble if it has
    # normal question/sentence punctuation and the draft starts as a fragment.
    if sorted(msg_words) == sorted(bubble_words) and msg_starts_lower:
        return True

    return False

def repair_message_from_messenger_bubbles(
    message: str,
    side: str,
    messenger_bubble_groups: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Repair suspicious Facebook rows using the matching visible bubble hint."""
    if not messenger_bubble_groups:
        return message

    best_group_text = ""
    best_score = 0.0
    for group in messenger_bubble_groups:
        group_side = str(group.get("side", "")).upper()
        if side in {"LEFT", "RIGHT"} and group_side in {"LEFT", "RIGHT"} and group_side != side:
            continue
        text = str(group.get("text", "")).strip()
        score = token_overlap_score(message, text)
        if score > best_score:
            best_score = score
            best_group_text = text

    if best_score >= 0.86 and should_prefer_messenger_bubble_text(message, best_group_text):
        return best_group_text

    return message

def messenger_bubble_order_for_message(
    message: str,
    side: str,
    messenger_bubble_groups: Optional[List[Dict[str, str]]] = None,
) -> Optional[int]:
    """Find the best matching visible Messenger bubble order for a row."""
    if not messenger_bubble_groups:
        return None

    best_order: Optional[int] = None
    best_score = 0.0
    for i, group in enumerate(messenger_bubble_groups):
        group_side = str(group.get("side", "")).upper()
        if side in {"LEFT", "RIGHT"} and group_side in {"LEFT", "RIGHT"} and group_side != side:
            # Try same-side matches first; wrong-side duplicate rows are handled by side correction before this.
            continue
        score = token_overlap_score(message, str(group.get("text", "")))
        if score > best_score:
            best_score = score
            try:
                best_order = int(group.get("order", i))
            except Exception:
                best_order = i

    if best_score >= 0.48:
        return best_order
    return None

def reorder_side_csv_by_messenger_order(
    side_csv: str,
    messenger_bubble_groups: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Reorder rows to match visual bubble order when OCR geometry can support it."""
    if not messenger_bubble_groups:
        return side_csv

    rows: List[Tuple[int, Optional[int], List[str]]] = []
    reader = csv.reader(io.StringIO(strip_code_fences(side_csv)))
    original_index = 0
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
        order = messenger_bubble_order_for_message(message, side, messenger_bubble_groups)
        rows.append((original_index, order, [time_value, side, message]))
        original_index += 1

    # When a row can be matched to a visible bubble, sort by bubble order.
    # Unmatched rows stay in their original relative position after matched rows near them.
    rows.sort(key=lambda item: (item[1] is None, item[1] if item[1] is not None else item[0], item[0]))

    out = io.StringIO()
    writer = csv.writer(out, quoting=csv.QUOTE_ALL, lineterminator="\n")
    writer.writerow(["Time", "Side", "Message"])

    seen = set()
    for _, _, row in rows:
        item = tuple(row)
        if item in seen:
            continue
        seen.add(item)
        writer.writerow(row)

    return out.getvalue().strip() + "\n"

# Cleans the LLM CSV, validates times/sides, and removes UI noise.
def normalize_side_csv(
    side_csv: str,
    visible_date: str,
    default_year: int,
    allowed_times: Set[str],
    emoji_mode: str = "omit",
    ocr_bubble_groups: Optional[List[Dict[str, str]]] = None,
    messenger_bubble_groups: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Cleans and validates model side CSV rows."""
    text = strip_code_fences(side_csv)

    reader = csv.reader(io.StringIO(text))
    out = io.StringIO()
    writer = csv.writer(out, quoting=csv.QUOTE_ALL, lineterminator="\n")

    writer.writerow(["Time", "Side", "Message"])

    seen = set()
    emitted_rows: List[Tuple[str, str, str]] = []

    for row in reader:
        if not row:
            continue

        if len(row) >= 3 and row[0].strip().lower() == "time":
            continue

        if len(row) < 3:
            continue

        # If an LLM accidentally leaves commas unquoted in Message, keep them.
        time_value = row[0].strip()
        side = row[1].strip().upper()
        message = ",".join(row[2:]).strip() if len(row) > 3 else row[2].strip()

        side = side.replace("INCOMING", "LEFT").replace("OUTGOING", "RIGHT")

        if side not in {"LEFT", "RIGHT"}:
            continue

        time_value = force_date_and_year(time_value, visible_date, default_year)
        hhmm = extract_hhmm_from_full_time(time_value)

        # Facebook Messenger often shows only a central date/time separator.
        # Accept valid full timestamps, but final postprocessing does not spread
        # repeated screen-level times into fabricated per-message minute offsets.
        if not re.fullmatch(r"\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}", time_value):
            continue

        ocr_side = infer_side_for_message_from_ocr(
            message=message,
            hhmm=hhmm,
            ocr_bubble_groups=ocr_bubble_groups,
        )
        if ocr_side in {"LEFT", "RIGHT"}:
            side = ocr_side

        messenger_side = infer_side_from_messenger_bubbles(
            message=message,
            messenger_bubble_groups=messenger_bubble_groups,
        )
        if messenger_side in {"LEFT", "RIGHT"}:
            side = messenger_side

        message = repair_message_from_messenger_bubbles(
            message=message,
            side=side,
            messenger_bubble_groups=messenger_bubble_groups,
        )

        candidate_messages = split_message_by_messenger_bubbles(
            message=message,
            side=side,
            messenger_bubble_groups=messenger_bubble_groups,
        )

        for candidate_message in candidate_messages:
            candidate_message = repair_message_from_messenger_bubbles(
                message=candidate_message,
                side=side,
                messenger_bubble_groups=messenger_bubble_groups,
            )
            cleaned_message = clean_message_text(candidate_message)
            if emoji_mode == "omit":
                cleaned_message = strip_emojis(cleaned_message)

            if is_ui_message(cleaned_message):
                continue

            if is_short_contained_duplicate(
                message=cleaned_message,
                time_value=time_value,
                side=side,
                emitted_rows=emitted_rows,
            ):
                continue

            item = (time_value, side, cleaned_message)
            if item in seen:
                continue

            seen.add(item)
            emitted_rows.append(item)
            writer.writerow(item)

    add_missing_messenger_bubbles_to_rows(
        emitted_rows=emitted_rows,
        seen=seen,
        writer=writer,
        visible_date=visible_date,
        allowed_times=allowed_times,
        emoji_mode=emoji_mode,
        messenger_bubble_groups=messenger_bubble_groups,
    )

    normalized = out.getvalue().strip() + "\n"
    return reorder_side_csv_by_messenger_order(normalized, messenger_bubble_groups)

def add_missing_messenger_bubbles_to_rows(
    emitted_rows: List[Tuple[str, str, str]],
    seen: Set[Tuple[str, str, str]],
    writer: csv.writer,
    visible_date: str,
    allowed_times: Set[str],
    emoji_mode: str,
    messenger_bubble_groups: Optional[List[Dict[str, str]]] = None,
) -> None:
    """Adds omitted high-confidence Messenger OCR bubble rows."""
    # Notes:
    # This catches bottom bubbles near the composer/input bar that the vision model
    # sometimes drops. It is conservative: if the OCR bubble overlaps any emitted
    # row, it is not added.
    if not messenger_bubble_groups or not emitted_rows:
        return

    fallback_time = emitted_rows[-1][0]
    if visible_date and allowed_times:
        hhmm = sorted(allowed_times)[0]
        fallback_time = f"{visible_date} {hhmm}"

    for group in messenger_bubble_groups:
        side = str(group.get("side", "")).upper()
        raw_text = str(group.get("text", "")).strip()

        if side not in {"LEFT", "RIGHT"} or not raw_text:
            continue

        raw_norm = normalize_for_fragment_check(raw_text)
        if len(raw_norm.split()) < 2:
            continue

        already_present = False
        for _, prev_side, prev_msg in emitted_rows:
            if prev_side != side:
                continue

            if token_overlap_score(prev_msg, raw_text) >= 0.62 or token_overlap_score(raw_text, prev_msg) >= 0.62:
                already_present = True
                break

        if already_present:
            continue

        message = clean_message_text(raw_text)
        if emoji_mode == "omit":
            message = strip_emojis(message)

        if is_ui_message(message):
            continue

        if is_short_contained_duplicate(
            message=message,
            time_value=fallback_time,
            side=side,
            emitted_rows=emitted_rows,
        ):
            continue

        item = (fallback_time, side, message)
        if item in seen:
            continue

        seen.add(item)
        emitted_rows.append(item)
        writer.writerow(item)

def parse_side_csv_datetime(time_value: str):
    """Parses supported side-CSV timestamp formats."""
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
            pass
    return None

def format_side_csv_datetime(dt) -> str:
    """Formats a datetime for the final CSV timestamp style."""
    return dt.strftime("%d/%m/%Y %H:%M:%S")

def estimate_messenger_gap_minutes(prev_side: str, side: str, prev_message: str, message: str) -> int:
    """Estimates a small gap between consecutive Messenger bubbles."""
    # Notes:
    # Same-speaker consecutive bubbles are usually close together.
    # Speaker changes usually imply a slightly larger gap.
    # Longer previous messages may imply a little more elapsed time.
    prev_side = str(prev_side or "").upper()
    side = str(side or "").upper()

    prev_words = len(str(prev_message or "").split())

    if prev_side == side:
        gap = 1
    else:
        gap = 3

    if prev_words >= 10:
        gap += 1

    return max(1, min(gap, 5))

def spread_repeated_messenger_times(side_csv: str) -> str:
    """Spreads repeated Messenger screen-level timestamps forward."""
    # Notes:
    # If many consecutive rows have the same timestamp, spread them forward
    # monotonically so the final CSV does not assign the exact same time to
    # every bubble.
    #
    # This is only a heuristic. It preserves exact visible times as anchors
    # and only changes rows inside consecutive duplicate-time groups.
    rows = []

    reader = csv.reader(io.StringIO(strip_code_fences(side_csv)))
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

        if side in {"LEFT", "RIGHT"} and time_value and message:
            rows.append([time_value, side, message])

    out = io.StringIO()
    writer = csv.writer(out, quoting=csv.QUOTE_ALL, lineterminator="\n")
    writer.writerow(["Time", "Side", "Message"])

    i = 0
    while i < len(rows):
        j = i + 1
        while j < len(rows) and rows[j][0] == rows[i][0]:
            j += 1

        group = rows[i:j]

        if len(group) == 1:
            writer.writerow(group[0])
            i = j
            continue

        base_dt = parse_side_csv_datetime(group[0][0])
        if base_dt is None:
            for row in group:
                writer.writerow(row)
            i = j
            continue

        current_dt = base_dt

        for k, row in enumerate(group):
            original_time, side, message = row

            if k == 0:
                new_time = format_side_csv_datetime(current_dt)
            else:
                prev_side = group[k - 1][1]
                prev_message = group[k - 1][2]
                gap = estimate_messenger_gap_minutes(prev_side, side, prev_message, message)
                from datetime import timedelta
                current_dt = current_dt + timedelta(minutes=gap)
                new_time = format_side_csv_datetime(current_dt)

            writer.writerow([new_time, side, message])

        i = j

    return out.getvalue().strip() + "\n"

def looks_like_fragment_message(message: str) -> bool:
    """Detects small OCR fragments that should merge into the previous row."""
    # Notes:
    # Examples: "ahead", "with you:", short continuation pieces.
    msg = str(message or "").strip()
    if not msg:
        return False

    words = msg.split()
    lower = msg.lower().strip(" .,:;!?")

    if len(words) <= 2 and msg[:1].islower():
        return True

    return False

def postprocess_facebook_side_rows(side_csv: str) -> str:
    """Deprecated compatibility wrapper for older call sites.

    Final cleanup is intentionally generic and lives in extractor_utils.py.
    It should not contain dataset-specific phrase rewrites.
    """
    return postprocess_side_csv_rows(side_csv)

def postprocess_facebook_side_rows_patch4(side_csv: str) -> str:
    """Deprecated compatibility wrapper for older call sites."""
    return postprocess_side_csv_rows(side_csv)

def extract_last_date_from_side_csv(side_csv: str) -> str:
    """Returns the last DD/MM/YYYY date seen in a side CSV."""
    reader = csv.reader(io.StringIO(strip_code_fences(side_csv)))
    last_date = ""

    for row in reader:
        if len(row) >= 1:
            m = re.match(r"^(\d{2}/\d{2}/\d{4})(?:,|\s)", row[0].strip())
            if m:
                last_date = m.group(1)
    return last_date

# ============================================================
# PIPELINE
# ============================================================

def process_facebook_image(
    image_path: str,
    report_text: str,
    model: str,
    langs: List[str],
    use_gpu: bool,
    grid: Optional[str],
    layout: Optional[str],
    use_vision: bool,
    emoji_mode: str,
    output_debug_dir: Path,
    dump_ocr: bool,
    dump_draft: bool,
    dump_side_map: bool,
    conversation_state_cache: Optional[str] = None,
    conversation_key: str = "",
) -> str:
    # Use the report year when screenshots show dates without a year.
    """Runs the full Facebook Messenger extraction pipeline for one image."""
    default_year = extract_year_from_report(report_text)

    # Split the input image/collage into individual phone screens.
    crops = get_screen_crops(
        image_path,
        grid=grid,
        layout=layout,
    )

    if not crops:
        raise ValueError("No screen crops were found.")

    print(f"-> [CV] Found {len(crops)} screenshot(s).")

    output_debug_dir.mkdir(parents=True, exist_ok=True)

    # Load EasyOCR once; it will be reused for all screen crops.
    reader = easyocr.Reader(langs, gpu=use_gpu)

    # Extract participant context from the report for side mapping.
    # Keep this internal: empty/partial actor JSON is not printed to the run log.
    actors = infer_report_actors(report_text, model)

    # Reuse the last validated mapping from the same evidence folder as a soft
    # continuity prior. Explicit cues in the current screenshot always win.
    prior_side_map = load_conversation_side_map(
        conversation_state_cache,
        conversation_key,
    )
    if prior_side_map:
        print(
            "-> [SIDE MAP] Previous folder mapping: "
            f"LEFT = {prior_side_map['LEFT']} | RIGHT = {prior_side_map['RIGHT']}"
        )

    # First pass: save crops and collect positioned OCR for each screen.
    all_screen_ocr = []
    screen_paths = []

    for idx, (_, _, _, _, crop) in enumerate(crops, start=1):
        crop_path = output_debug_dir / f"screen_{idx:02d}.png"
        write_crop(crop_path, crop)
        screen_paths.append(crop_path)

        print(f"   - OCR screen #{idx}...")
        screen_ocr = extract_ocr_blocks(reader, crop, idx)
        all_screen_ocr.append(screen_ocr)

        if dump_ocr:
            (output_debug_dir / f"screen_{idx:02d}_ocr.txt").write_text(
                screen_ocr,
                encoding="utf-8",
            )

    full_ocr = "\n\n".join(all_screen_ocr)

    if dump_ocr:
        (output_debug_dir / "all_ocr.txt").write_text(
            full_ocr,
            encoding="utf-8",
        )

    # Infer one fixed LEFT/RIGHT mapping for the whole conversation.
    print("-> [SIDE MAP] Inferring fixed LEFT/RIGHT mapping...")
    side_map = infer_side_mapping(
        report_text,
        actors,
        full_ocr,
        "facebook_messenger",
        model,
    )

    print(f"-> [SIDE MAP] LEFT = {side_map['LEFT']} | RIGHT = {side_map['RIGHT']}")

    if dump_side_map:
        (output_debug_dir / "side_map.json").write_text(
            json.dumps(side_map, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # Second pass: ask the VLM to extract each screen as Side CSV.
    side_csv_parts = []
    previous_date_hint = ""

    for idx, screen_ocr in enumerate(all_screen_ocr, start=1):
        print(f"-> [LLM] Extracting screen #{idx} as Side CSV...")

        crop_path = str(screen_paths[idx - 1])

        visible_date = extract_visible_date_from_ocr(
            screen_ocr,
            default_year=default_year,
            previous_date_hint="",
        )

        if not visible_date and use_vision:
            print(f"   - [DATE] OCR date not found for screen #{idx}; asking vision model...")
            visible_date = infer_visible_date_with_ai(
                model=model,
                image_path=crop_path,
                screen_ocr=screen_ocr,
                default_year=default_year,
                use_vision=use_vision,
            )

        if dump_draft:
            (output_debug_dir / f"screen_{idx:02d}_visible_date.txt").write_text(
                visible_date or "",
                encoding="utf-8",
            )

        # Limit accepted times to times actually visible in this screen.
        allowed_times = extract_allowed_times_from_ocr(screen_ocr)
        ocr_bubble_groups = build_ocr_bubble_groups(screen_ocr)
        messenger_bubble_groups = build_messenger_bubble_groups(screen_ocr)
        bubble_hints = build_messenger_bubble_hints(screen_ocr)

        if dump_draft:
            (output_debug_dir / f"screen_{idx:02d}_ocr_bubbles.json").write_text(
                json.dumps(ocr_bubble_groups, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (output_debug_dir / f"screen_{idx:02d}_bubble_hints.txt").write_text(
                bubble_hints,
                encoding="utf-8",
            )
            (output_debug_dir / f"screen_{idx:02d}_messenger_bubbles.json").write_text(
                json.dumps(messenger_bubble_groups, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        prompt = build_screen_prompt(
            screen_ocr=screen_ocr,
            screen_index=idx,
            visible_date=visible_date,
            default_year=default_year,
            allowed_times=allowed_times,
            emoji_mode=emoji_mode,
            bubble_hints=bubble_hints,
        )

        # Initial VLM extraction from screenshot + OCR hints.
        draft = ollama_chat_screen(
            model,
            prompt,
            crop_path,
            use_vision=use_vision,
        )

        if dump_draft:
            (output_debug_dir / f"screen_{idx:02d}_draft.csv").write_text(
                draft,
                encoding="utf-8",
            )

        draft_norm = normalize_side_csv(
            draft,
            visible_date=visible_date,
            default_year=default_year,
            allowed_times=allowed_times,
            emoji_mode=emoji_mode,
            ocr_bubble_groups=ocr_bubble_groups,
            messenger_bubble_groups=messenger_bubble_groups,
        )

        expected_bubbles = len(messenger_bubble_groups) if messenger_bubble_groups else 0
        repair_needed = side_csv_needs_repair(
            draft_norm,
            expected_bubble_count=expected_bubbles,
        )

        if repair_needed:
            print(f"-> [LLM] Repairing suspicious screen #{idx} Side CSV...")

            repair_prompt = build_screen_repair_prompt(
                draft_csv=draft_norm,
                screen_ocr=screen_ocr,
                screen_index=idx,
                visible_date=visible_date,
                default_year=default_year,
                allowed_times=allowed_times,
                emoji_mode=emoji_mode,
                bubble_hints=bubble_hints,
            )

            # Repair pass is now run only for suspicious screens/rows.
            repaired = ollama_chat_screen(
                model,
                repair_prompt,
                crop_path,
                use_vision=use_vision,
            )

            if dump_draft:
                (output_debug_dir / f"screen_{idx:02d}_repair_raw.csv").write_text(
                    repaired,
                    encoding="utf-8",
                )

            repaired_norm = normalize_side_csv(
                repaired,
                visible_date=visible_date,
                default_year=default_year,
                allowed_times=allowed_times,
                emoji_mode=emoji_mode,
                ocr_bubble_groups=ocr_bubble_groups,
                messenger_bubble_groups=messenger_bubble_groups,
            )
        else:
            print(f"-> [LLM] Screen #{idx} looks stable; skipping repair pass.")
            repaired_norm = ""

        # Keep the repaired CSV only if it stays within a sane row count.
        if repaired_norm:
            chosen = choose_best_screen_side_csv(
                draft_norm=draft_norm,
                repaired_norm=repaired_norm,
                allowed_times=allowed_times,
                expected_bubble_count=expected_bubbles,
                bubble_groups=messenger_bubble_groups,
                enable_additive=True,
            )
        else:
            chosen = draft_norm

        # Optional exact-text polish: same rows/times/sides, Message field only.
        # The guard in extractor_utils.py rejects candidates that change structure,
        # visible-time validity, bubble coverage, or message meaning too much.
        if use_vision and count_data_rows(chosen) > 0:
            print(f"-> [LLM] Text-polishing screen #{idx} without changing rows...")
            polish_prompt = build_text_polish_prompt(
                screen_ocr=screen_ocr,
                screen_index=idx,
                visible_date=visible_date,
                side_csv=chosen,
                emoji_mode=emoji_mode,
                bubble_hints=bubble_hints,
            )

            if dump_draft:
                (output_debug_dir / f"screen_{idx:02d}_polish_prompt.txt").write_text(
                    polish_prompt,
                    encoding="utf-8",
                )

            polished_raw = ollama_chat_screen(
                model,
                polish_prompt,
                crop_path,
                use_vision=use_vision,
            )

            if dump_draft:
                (output_debug_dir / f"screen_{idx:02d}_polish_raw.csv").write_text(
                    polished_raw,
                    encoding="utf-8",
                )

            polished_norm = normalize_polished_side_csv_against_reference(
                polished_raw,
                chosen,
                emoji_mode=emoji_mode,
            )

            if polished_norm:
                chosen = choose_text_polished_side_csv(
                    reference_csv=chosen,
                    polished_csv=polished_norm,
                    allowed_times=allowed_times,
                    expected_bubble_count=expected_bubbles,
                    bubble_groups=messenger_bubble_groups,
                )

        # Final timestamp policy for Facebook/Messenger: first row uses :00,
        # later rows in the same screen advance deterministically by one second.
        chosen = add_sequential_seconds_to_side_csv(chosen, step_seconds=1)

        if dump_draft:
            (output_debug_dir / f"screen_{idx:02d}_side.csv").write_text(
                chosen,
                encoding="utf-8",
            )

        last_date = extract_last_date_from_side_csv(chosen)
        if last_date:
            previous_date_hint = last_date

        side_csv_parts.append(chosen)

    # Merge all screen-level CSV parts before applying real names.
    raw_merged_side_csv = merge_side_csvs(side_csv_parts)

    # Keep Facebook Messenger timestamps as visible/screen-level anchors.
    # Do not spread/fabricate per-message minute offsets.
    merged_side_csv = raw_merged_side_csv

    # Generic cleanup only: adjacent near-duplicates and safe continuation fragments.
    merged_side_csv = postprocess_side_csv_rows(merged_side_csv)

    if dump_draft:
        (output_debug_dir / "merged_side_raw.csv").write_text(
            raw_merged_side_csv,
            encoding="utf-8",
        )
        (output_debug_dir / "merged_side.csv").write_text(
            merged_side_csv,
            encoding="utf-8",
        )

    # Final Gemma validation uses both the report and the already extracted
    # messages. It may correct the fixed pair/mapping, but it cannot alter rows
    # or introduce unsupported/combined names.
    print("-> [SIDE MAP] Validating mapping against report + extracted messages...")
    provisional_side_map = dict(side_map)
    side_map = refine_side_mapping_with_context(
        report_text=report_text,
        actors=actors,
        side_csv=merged_side_csv,
        current_side_map=side_map,
        platform_hint="facebook_messenger",
        model=model,
        prior_side_map=prior_side_map,
        evidence_hint=image_path,
    )
    if side_map != provisional_side_map:
        print(
            f"-> [SIDE MAP] Context correction: "
            f"LEFT = {side_map['LEFT']} | RIGHT = {side_map['RIGHT']}"
        )
    else:
        print(
            f"-> [SIDE MAP] Context validation kept: "
            f"LEFT = {side_map['LEFT']} | RIGHT = {side_map['RIGHT']}"
        )

    # Persist the accepted map for the next screenshot in the same folder.
    save_conversation_side_map(
        conversation_state_cache,
        conversation_key,
        side_map,
        source_hint=image_path,
    )

    if dump_side_map:
        (output_debug_dir / "side_map_initial.json").write_text(
            json.dumps(provisional_side_map, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output_debug_dir / "side_map.json").write_text(
            json.dumps(side_map, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # Replace LEFT/RIGHT with actual Sender/Receiver names.
    # The final writer normally trusts the accepted visual side mapping.
    # A second constrained Gemma pass is enabled only when deterministic
    # checks detect collapsed or contradictory row directions.
    final_csv = apply_side_mapping(
        merged_side_csv,
        side_map,
        estimated_timestamp_from_seconds=True,
        # model=model,
        # report_text=report_text,
        # platform_hint="facebook_messenger",
        # evidence_hint=image_path,
    )

    # apply_side_mapping may remove invalid/duplicate rows. Re-number only the
    # surviving synthetic Facebook seconds afterwards so the public CSV has
    # contiguous +1 second ordering without changing any observed :00 anchor.
    final_csv = renumber_estimated_facebook_timestamps(final_csv)

    return final_csv

# ============================================================
# MAIN
# ============================================================

# CLI entry point: parse arguments, run extraction, and write outputs.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract Facebook Messenger chat CSV from screenshot or collage screenshot."
    )

    parser.add_argument(
        "image",
        help="Path to Facebook Messenger screenshot/collage image",
    )

    parser.add_argument(
        "report",
        help="Path to case report PDF/TXT",
    )

    parser.add_argument(
        "--model",
        default="gemma3:12b",
        help="Ollama model, default: gemma3:12b",
    )

    parser.add_argument(
        "--langs",
        default="en",
        help="EasyOCR languages, e.g. en or en,el",
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU for EasyOCR",
    )

    parser.add_argument(
        "--grid",
        default=None,
        help="Manual regular collage split, e.g. 2x1 or 3x2",
    )

    parser.add_argument(
        "--layout",
        default=None,
        help="Manual uneven collage split by rows, e.g. 2,3",
    )

    parser.add_argument(
        "--no-vision",
        action="store_true",
        help="Do not send screen images to Ollama; OCR only. Emojis will be less reliable.",
    )

    parser.add_argument(
        "-emoji-mode",
        "--emoji-mode",
        choices=["omit", "vision"],
        default="omit",
        help=(
            "Emoji handling mode. "
            "'omit' removes emojis/symbols from final messages to avoid wrong guessed emojis. "
            "'vision' asks the vision model to preserve only clearly visible emojis."
        ),
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Keep debug crops/OCR/intermediate files. Default: off.",
    )

    parser.add_argument(
        "--debug-dir",
        default=None,
        help="Directory for crops/OCR/debug files. Implies --debug.",
    )

    # Standard double-dash flags.
    # Also accepts your accidental single-dash versions for convenience.
    parser.add_argument(
        "-dump-ocr",
        "--dump-ocr",
        action="store_true",
        help="Save positioned OCR debug files",
    )

    parser.add_argument(
        "-dump-draft",
        "--dump-draft",
        action="store_true",
        help="Save draft/repaired Side CSV files",
    )

    parser.add_argument(
        "-dump-side-map",
        "--dump-side-map",
        action="store_true",
        help="Save inferred LEFT/RIGHT side map",
    )

    parser.add_argument(
        "--case-context-cache",
        default=None,
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--conversation-state-cache",
        default=None,
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--conversation-key",
        default="",
        help=argparse.SUPPRESS,
    )

    args = parser.parse_args()

    if args.grid and args.layout:
        print("[ERROR] Use either --grid or --layout, not both.")
        sys.exit(1)

    image_path = Path(args.image)
    report_path = Path(args.report)

    if not image_path.exists():
        print(f"[ERROR] Image not found: {image_path}")
        sys.exit(1)

    if not report_path.exists():
        print(f"[ERROR] Report not found: {report_path}")
        sys.exit(1)

    print("\n[START]")
    print("-> Reading case report...")

    report_text = extract_text_from_report(str(report_path))

    if not report_text.strip():
        print("[WARNING] Report appears empty or could not be read correctly.")

    script_dir = Path(__file__).resolve().parent
    image_base = image_path.stem

    debug_requested = bool(args.debug or args.debug_dir or args.dump_ocr or args.dump_draft or args.dump_side_map)
    debug_temp = None
    if args.debug_dir:
        debug_dir = Path(args.debug_dir)
    elif debug_requested:
        debug_dir = script_dir / f"{image_base}_debug"
    else:
        debug_temp = tempfile.TemporaryDirectory(prefix=f"{image_base}_extract_")
        debug_dir = Path(debug_temp.name)

    output_path = Path(args.output) if args.output else script_dir / f"{image_base}_extracted.csv"

    langs = [x.strip() for x in args.langs.split(",") if x.strip()]

    try:
        final_csv = process_facebook_image(
        image_path=str(image_path),
        report_text=report_text,
        model=args.model,
        langs=langs,
        use_gpu=not args.cpu,
        grid=args.grid,
        layout=args.layout,
        use_vision=not args.no_vision,
        emoji_mode=args.emoji_mode,
        output_debug_dir=debug_dir,
        dump_ocr=args.dump_ocr,
        dump_draft=args.dump_draft,
        dump_side_map=args.dump_side_map,
        conversation_state_cache=args.conversation_state_cache,
        conversation_key=args.conversation_key,
    )
    finally:
        if debug_temp is not None:
            debug_temp.cleanup()

    output_path.write_text(final_csv, encoding="utf-8-sig")

    print("\n--- FINAL CSV ---\n")
    print(final_csv)

    print(f"\n[SUCCESS] CSV saved to: {output_path}")
    if debug_requested:
        print(f"[DEBUG] Debug folder: {debug_dir}")
