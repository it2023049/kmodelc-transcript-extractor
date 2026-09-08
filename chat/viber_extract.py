"""Viber screenshot-to-CSV extractor."""

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
VIBER_RIGHT_MIN_X_RATIO = 0.245

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
    split_trailing_visible_time_token,
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
    refine_side_mapping_with_context,
    load_conversation_side_map,
    load_conversation_date_hint,
    save_conversation_side_map,
    save_conversation_date_hint,
    write_crop,
    choose_best_screen_side_csv,
    normalize_polished_side_csv_against_reference,
    choose_text_polished_side_csv,
    ensure_zero_seconds_side_csv,
    conservative_clean_message_text,
    postprocess_side_csv_rows,
    side_csv_needs_repair,
    looks_like_noisy_ocr_text,
)

# ============================================================
# IMAGE / COLLAGE SPLITTING
# ============================================================
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

    return contour_fallback_split(image)

# ============================================================
# OCR WITH POSITIONED BLOCKS
# ============================================================
def position_tag_from_bbox(bbox: Box, crop_w: int, text: str) -> str:
    """Classifies an OCR box as LEFT, RIGHT, or CENTER by geometry."""
    x, y, bw, bh = bbox
    cx = x + bw / 2

    if looks_like_date_or_time(text) and crop_w * 0.25 <= cx <= crop_w * 0.75:
        return "CENTER"

    # Viber outgoing bubbles start far enough to the right.
    # Use the left edge, not the center, to avoid misclassifying wide incoming bubbles.
    if x >= crop_w * VIBER_RIGHT_MIN_X_RATIO:
        return "RIGHT"

    return "LEFT"

def estimate_side_from_bubble_color(image: np.ndarray, bbox: Box) -> Optional[str]:
    """Detects Viber outgoing bubbles using local purple color evidence."""
    # Notes:
    # This is used only as a side hint. It is deliberately conservative:
    # - Purple/high-saturation background -> RIGHT
    # - Otherwise return None and fall back to geometry.
    if image is None or image.size == 0:
        return None

    h, w = image.shape[:2]
    x, y, bw, bh = bbox

    # Sample only the immediate text background. A large crop can include a
    # neighbouring purple bubble, while a single purple emoji inside a dark
    # bubble must not determine the side of the entire message.
    pad_x = max(10, int(bw * 0.08))
    pad_y = max(8, int(bh * 0.20))

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w, x + bw + pad_x)
    y2 = min(h, y + bh + pad_y)

    region = image[y1:y2, x1:x2]
    if region.size == 0:
        return None

    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    # Viber purple bubble in OpenCV HSV is usually around hue 120-165.
    # White text has low saturation/high value and is excluded by the mask.
    purple_mask = (
        (sat >= 55) &
        (val >= 45) &
        (val <= 245) &
        (hue >= 118) &
        (hue <= 170)
    )

    purple_ratio = float(purple_mask.mean())
    purple_row_coverage = float((purple_mask.mean(axis=1) >= 0.10).mean())

    # A real bubble provides broad, continuous purple background. Isolated
    # coloured glyphs and emoji have low vertical coverage and are rejected.
    if purple_ratio >= 0.12 and purple_row_coverage >= 0.35:
        return "RIGHT"

    return None

def position_tag_from_color_or_bbox(
    image: np.ndarray,
    bbox: Box,
    crop_w: int,
    text: str
) -> str:
    """Classifies a Viber OCR box using color evidence before geometry."""
    if looks_like_date_or_time(text):
        return position_tag_from_bbox(bbox, crop_w, text)

    color_side = estimate_side_from_bubble_color(image, bbox)
    if color_side in {"LEFT", "RIGHT"}:
        return color_side

    return position_tag_from_bbox(bbox, crop_w, text)

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
        # Purple bubble colour is stronger evidence than left-edge geometry.
        # Preserve it when OCR splits one visual line into several fragments.
        color_sides = {
            b.get("color_side") for b in line
            if b.get("color_side") in {"LEFT", "RIGHT"}
        }
        if len(color_sides) == 1:
            fixed_side = next(iter(color_sides))
            for b in line:
                b["pos"] = fixed_side
            continue

        min_x = min(b["x"] for b in line)

        # If a visual line starts on the left, the whole line belongs to a LEFT bubble.
        # A real right-side bubble should not have any text starting this far left.
        # Keep this threshold consistent with position_tag_from_bbox(). The
        # previous inconsistent thresholds made wrapped lines near the boundary flip
        # sides inside one wide bubble, especially in dark Viber themes.
        if min_x < crop_w * VIBER_RIGHT_MIN_X_RATIO:
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
        color_side = None if looks_like_date_or_time(text) else estimate_side_from_bubble_color(
            upscaled, (x, y, bw, bh)
        )
        pos = color_side or position_tag_from_bbox((x, y, bw, bh), w, text)

        blocks.append({
            "pos": pos,
            "x": x,
            "y": y,
            "w": bw,
            "h": bh,
            "conf": conf,
            "color_side": color_side,
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
    # If no date is visible, returns previous_date_hint if available.
    rows = parse_ocr_lines(screen_ocr)

    for row in rows:
        text = row["text"].strip()

        # Month-first layouts: "April 07" / "April 7, 2026".
        m = re.search(
            r"\b(Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|Sep|Sept|September|Oct|October|Nov|November|Dec|December)\s+(\d{1,2})(?:,?\s+(20\d{2}))?\b",
            text,
            flags=re.I,
        )
        if m:
            month = month_to_number(m.group(1))
            day = int(m.group(2))
            year = int(m.group(3)) if m.group(3) else default_year
            if month and 1 <= day <= 31:
                return f"{day:02d}/{month:02d}/{year}"

        # Day-first layouts are also common in mobile Viber screenshots:
        # "07 April" / "7 Apr 2026".
        m = re.search(
            r"\b(\d{1,2})\s+(Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|Sep|Sept|September|Oct|October|Nov|November|Dec|December)(?:,?\s+(20\d{2}))?\b",
            text,
            flags=re.I,
        )
        if m:
            day = int(m.group(1))
            month = month_to_number(m.group(2))
            year = int(m.group(3)) if m.group(3) else default_year
            if month and 1 <= day <= 31:
                return f"{day:02d}/{month:02d}/{year}"

    if previous_date_hint:
        return previous_date_hint

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

def find_report_grounded_header_match(ocr_data: str, report_text: str) -> str:
    """Find an exact human name in both the Viber header and case report.

    This fallback is intentionally conservative. It is used when structured
    actor parsing missed a participant and accepts only a complete two-to-five
    token title-cased header row that also occurs verbatim in the report.
    """
    if not report_text:
        return ""

    blocked_words = {
        "active", "april", "august", "december", "february", "friday",
        "january", "july", "june", "march", "monday", "november",
        "october", "saturday", "september", "sunday", "thursday",
        "tuesday", "viber", "wednesday",
    }
    candidates: List[str] = []

    for row in parse_ocr_lines(ocr_data):
        try:
            y = int(row.get("y", "9999"))
        except (TypeError, ValueError):
            continue
        if y > 350:
            continue

        candidate = clean_name(row.get("text", "")).strip(" ,;:()[]{}\"'")
        tokens = candidate.split()
        if not 2 <= len(tokens) <= 5:
            continue
        if any(token.casefold().strip(".,") in blocked_words for token in tokens):
            continue
        if not re.fullmatch(
            r"[A-Z][A-Za-z'’.-]+(?:\s+[A-Z][A-Za-z'’().-]+){1,4}",
            candidate,
        ):
            continue

        pattern = r"(?<![A-Za-z])" + r"\s+".join(
            re.escape(token) for token in tokens
        ) + r"(?![A-Za-z])"
        match = re.search(pattern, report_text, flags=re.I)
        if match:
            candidates.append(clean_name(match.group(0)))

    if not candidates:
        return ""
    return sorted(candidates, key=lambda value: (-len(value.split()), -len(value)))[0]

def participant_initials(name: str) -> str:
    """Return initials for a canonical participant name."""
    words = re.findall(r"[A-Za-z]+", clean_name(name))
    return "".join(word[0].upper() for word in words if word)

def refine_side_map_with_avatar_initials(
    side_map: Dict[str, str],
    actors: Dict,
    ocr_data: str,
) -> Dict[str, str]:
    """Use a uniquely report-grounded Viber avatar initial as side evidence.

    Only a standalone 2--4 letter uppercase token near the far-left avatar
    gutter is considered. If it uniquely identifies the participant currently
    assigned to the opposite side, the two-person mapping is swapped. No new
    identity is invented and ordinary message acronyms are ignored.
    """
    mapping = {
        "LEFT": clean_name(side_map.get("LEFT", "")),
        "RIGHT": clean_name(side_map.get("RIGHT", "")),
    }
    if not mapping["LEFT"] or not mapping["RIGHT"]:
        return side_map

    known_names: List[str] = []
    for participant in (actors or {}).get("participants", []) or []:
        if isinstance(participant, dict):
            name = clean_name(participant.get("name", ""))
            if name and not any(same_name(name, seen) for seen in known_names):
                known_names.append(name)
    for name in mapping.values():
        if name and not any(same_name(name, seen) for seen in known_names):
            known_names.append(name)

    initials_to_names: Dict[str, List[str]] = {}
    for name in known_names:
        initials = participant_initials(name)
        if 2 <= len(initials) <= 4:
            initials_to_names.setdefault(initials, []).append(name)

    for row in parse_ocr_lines(ocr_data):
        token = str(row.get("text", "")).strip()
        try:
            x = int(row.get("x", "9999"))
            y = int(row.get("y", "0"))
        except (TypeError, ValueError):
            continue
        if not re.fullmatch(r"[A-Z]{2,4}", token) or x > 160 or y < 350:
            continue
        matches = initials_to_names.get(token, [])
        if len(matches) != 1:
            continue
        avatar_name = matches[0]
        avatar_side = str(row.get("pos", "")).upper()
        if avatar_side not in {"LEFT", "RIGHT"}:
            continue
        other_side = "RIGHT" if avatar_side == "LEFT" else "LEFT"
        if same_name(mapping[other_side], avatar_name) and not same_name(
            mapping[avatar_side], avatar_name
        ):
            return {
                avatar_side: mapping[other_side],
                other_side: mapping[avatar_side],
            }

    return mapping

def deterministic_viber_side_map(
    actors: Dict,
    ocr_data: str,
    report_text: str = "",
) -> Optional[Dict[str, str]]:
    """Infers Viber LEFT/RIGHT names from header and contact evidence."""
    # Notes:
    # Main rule:
    # If Viber header shows a suspect/contact from the report, then:
    # LEFT = header contact
    # RIGHT = victim / screenshot owner
    victim = clean_name(actors.get("victim", ""))

    participants = [
        p for p in actors.get("participants", [])
        if p.get("name")
    ]

    # If victim field is empty, try to recover it from participants.
    if not victim:
        for p in participants:
            if str(p.get("role", "")).lower() == "victim":
                victim = clean_name(p.get("name", ""))
                break

    # -------------------------
    # 1. Header evidence wins
    # -------------------------
    # A complete person name that is visibly present in the Viber header and
    # also occurs in the case report is stronger identity evidence than a
    # phone number inherited from a broad/flattened report section.  In
    # particular, one number can be repeated in a report near several actors,
    # while the current screenshot header directly identifies the contact.
    # This rule remains report-grounded: an OCR-only name is never accepted.
    header_name = find_report_grounded_header_match(ocr_data, report_text)
    if not header_name:
        header_name = find_header_match(ocr_data, actors)

    if header_name and victim and not same_name(header_name, victim):
        return {
            "LEFT": clean_name(header_name),
            "RIGHT": clean_name(victim)
        }

    # -------------------------
    # 2. Phone/contact evidence fallback
    # -------------------------
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

                # Header phone can be CENTER/LEFT depending OCR.
                # In Viber header contact still maps to LEFT.
                try:
                    y = int(row["y"])
                except ValueError:
                    y = 9999

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
Determine the fixed LEFT/RIGHT speaker mapping for this Viber two-person chat.

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
2. In Viber, the chat header usually shows the other participant/contact, not the screenshot owner.
3. In Viber, LEFT dark/gray bubbles are usually incoming from the header contact. RIGHT purple bubbles are outgoing from the screenshot owner.
4. Direct contact evidence wins: if a phone/contact number from the report appears on a side, that side belongs to that number's owner.
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
    deterministic = deterministic_viber_side_map(actors, ocr_data, report_text)

    if deterministic:
        return refine_side_map_with_avatar_initials(
            deterministic,
            actors,
            ocr_data,
        )

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

    return refine_side_map_with_avatar_initials(
        {
            "LEFT": left,
            "RIGHT": right,
        },
        actors,
        ocr_data,
    )

# ============================================================
# SCREEN EXTRACTION PROMPTS
# ============================================================
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
You are extracting messages from ONE Viber screenshot image.
Use the attached image as the primary source. Use OCR blocks only as support for time, side, and bubble boundaries. OCR text can be noisy; do not copy OCR word-order artifacts when the image is clearer.

SCREEN INDEX: {screen_index}
VISIBLE DATE FOR THIS SCREEN: {visible_date or "unknown"}
DEFAULT YEAR: {default_year}
ALLOWED BUBBLE TIMES FOR THIS SCREEN: {allowed_times_text}

OCR BLOCKS:
{screen_ocr}

VIBER BUBBLE HINTS FROM OCR GEOMETRY:
{bubble_hints or "No reliable Viber bubble hints."}

Return only CSV with this header exactly once:
"Time","Side","Message"

Rules:
1. Output only real human chat bubbles.
2. Ignore UI: status bar clock, battery, header/contact name, contact phone, encryption banner, date-only bubble, Type a message, icons, GIF, calls, read ticks.
3. Side must be LEFT or RIGHT based only on visible bubble position/color in the image.
4. LEFT = dark/gray incoming bubble. RIGHT = purple outgoing bubble.
5. Do not output participant names. Do not infer side from meaning.
6. Reconstruct each visible rounded bubble as one row. Do not output one row per OCR line. Do not merge different rounded bubbles.
7. Use VIBER BUBBLE HINTS as a guide for TIME, SIDE, split boundaries, and visual order. Normally each [BUBBLE n] should become one CSV row with that TIME and SIDE, unless it is clearly UI noise. Use the image, not noisy OCR text, for final message wording.
8. The output rows must follow the exact visible top-to-bottom order of rounded bubbles in the screenshot.
9. If two separate rounded bubbles show the same timestamp, still output two separate rows.
9. OCR COVERAGE CHECK: Every non-UI LEFT/RIGHT OCR text block must be represented in one output message. Do not drop lines from a bubble.
10. If a bubble has multiple OCR lines before its timestamp, merge all those lines into the same message.
11. Do not shorten messages. Do not keep only the first line of a bubble.
12. If the draft has a row with obvious word-order noise, lowercase orphan fragments, or leaked neighboring bubble text, re-read the rounded bubble from the image and repair only that row/split.
10. Emoji rule: {emoji_rule}
12. Preserve evidence exactly: amounts, names, countries, cities, receiver details, phone numbers, account/payment details, reference numbers.
13. Time must be the bubble's visible time, combined with VISIBLE DATE FOR THIS SCREEN. Format: "DD/MM/YYYY HH:MM".
14. Only use times from ALLOWED BUBBLE TIMES FOR THIS SCREEN. Do not use the status bar time.
15. Fix only obvious OCR mistakes when the image clearly supports it: Im -> I'm, Icant -> I can't, IIl/1'Il -> I'll, Ijust -> I just, Iwill -> I will, Ilove -> I love, | -> I, $ -> s, and remove leaked HH:MM tokens from message text.
16. Do not invent, summarize, or add messages from other screenshots.
17. Use exactly three quoted CSV fields per row. No markdown, no explanations.

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
Repair this Viber screen CSV using the attached screenshot image and OCR blocks.

SCREEN INDEX: {screen_index}
VISIBLE DATE FOR THIS SCREEN: {visible_date or "unknown"}
DEFAULT YEAR: {default_year}
ALLOWED BUBBLE TIMES FOR THIS SCREEN: {allowed_times_text}

OCR BLOCKS:
{screen_ocr}

VIBER BUBBLE HINTS FROM OCR GEOMETRY:
{bubble_hints or "No reliable Viber bubble hints."}

BAD / INCOMPLETE DRAFT CSV:
{draft_csv}

Return only final CSV with this header exactly once:
"Time","Side","Message"

Rules:
1. Rebuild only from this screenshot image and its OCR blocks.
2. Keep every real visible message bubble from this screenshot.
3. Remove all UI rows: status bar clock, battery, header/contact, contact phone, encryption banner, date separator as message, Type a message, GIF, icons, read ticks.
4. Side must be LEFT or RIGHT based on visible bubble position/color. Do not output names.
5. Use VIBER BUBBLE HINTS as a guide for TIME, SIDE, and split boundaries only. Normally each [BUBBLE n] should become one CSV row with that TIME and SIDE, unless it is clearly UI noise. Use the image, not noisy OCR text, for final message wording.
6. If two separate rounded bubbles show the same timestamp, still output two separate rows.
7. OCR COVERAGE CHECK: Every non-UI LEFT/RIGHT OCR text block must be represented in one output message.
8. If the draft omitted an OCR line from a bubble, add it back.
9. Do not shorten messages. Do not keep only the first line of a bubble.
10. Emoji rule: {emoji_rule}
11. Merge wrapped lines of the same visible bubble into one message.
12. Use only visible bubble times listed in ALLOWED BUBBLE TIMES FOR THIS SCREEN. Remove leaked HH:MM tokens from message text; they are timestamps, not message content.
11. Combine each time with VISIBLE DATE FOR THIS SCREEN. Format: "DD/MM/YYYY HH:MM".
13. Preserve evidence: amounts, names, receiver details, locations, phone numbers, accounts, references.
14. Do not invent messages and do not include messages from other screenshots.
15. Use exactly three quoted CSV fields per row. No markdown, no explanations.

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
Polish the Message text in this Viber side CSV using the attached screenshot as the primary source and OCR only as support.

SCREEN INDEX: {screen_index}
VISIBLE DATE: {visible_date or "unknown"}
EXPECTED ROW COUNT: {row_count}

OCR BLOCKS:
{screen_ocr}

VIBER BUBBLE HINTS:
{bubble_hints or "No reliable Viber bubble hints."}

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

    # Generic Viber/chat UI text.
    ui_substrings = [
        "type a message",
        "messages in this chat are private",
        "end-to-end encryption",
        "learn more",
        "active now",
        "delivered",
        "seen",
        "typing",
    ]

    if any(part in low for part in ui_substrings):
        return True

    # OCR often splits or damages the standard Viber privacy banner. Match its
    # stable concepts rather than requiring one exact sentence.
    ui_norm = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", low)).strip()
    ui_words = set(ui_norm.split())
    if {"messages", "chat", "private"}.issubset(ui_words):
        return True
    if {"viber", "end", "to", "encryption"}.issubset(ui_words):
        return True
    if {"protected", "end", "to"}.issubset(ui_words):
        return True
    if "encryption" in ui_words and ({"viber", "protected"} & ui_words):
        return True
    if re.fullmatch(r"(?:protec?ted|cted)\s+by(?:\s+(?:viber|more))?", ui_norm):
        return True

    # Bottom icon OCR noise.
    if re.fullmatch(r"(gif|gıf|g1f)", low, flags=re.IGNORECASE):
        return True

    if re.fullmatch(r"[0o]{2,4}", low):
        return True

    # Very short icon/call artifacts.
    if re.fullmatch(r"[✓✔v]{1,4}", low):
        return True

    return False

def clean_message_text(message: str) -> str:
    """Apply minimal, generic cleanup without semantic rewriting."""
    return conservative_clean_message_text(message)

def build_ocr_bubble_groups(
    screen_ocr: str,
    allow_top_content: bool = False,
) -> List[Dict[str, str]]:
    """Group positioned OCR into one hint per visible Viber bubble.

    The implementation handles three common OCR failures generically:
    tiny time glyphs (``Ji00`` -> ``11:00``), a timestamp attached to the last
    message line, and fragments from one visual line returned in x/y order that
    would otherwise read backwards.
    """
    rows = parse_ocr_lines(screen_ocr)

    def to_int(value: str, default: int = 0) -> int:
        """Safely converts a value to int with a fallback."""
        try:
            return int(value)
        except Exception:
            return default

    # Cluster blocks that overlap vertically, then read each line left-to-right.
    # This repairs cases such as x=733/y=849 "ready:" being emitted before
    # x=256/y=859 "Your account is now".
    raw_order = sorted(rows, key=lambda r: (
        to_int(r.get("y", "0")),
        to_int(r.get("x", "0")),
        to_int(r.get("block", "0")),
    ))
    visual_lines: List[List[Dict[str, str]]] = []
    for row in raw_order:
        top = to_int(row.get("y", "0"))
        bottom = top + max(1, to_int(row.get("h", "1"), 1))
        best_line = None
        best_overlap = 0
        for line in visual_lines[-3:]:
            line_top = min(to_int(item.get("y", "0")) for item in line)
            line_bottom = max(
                to_int(item.get("y", "0")) + max(1, to_int(item.get("h", "1"), 1))
                for item in line
            )
            overlap = min(bottom, line_bottom) - max(top, line_top)
            if overlap > best_overlap:
                best_overlap = overlap
                best_line = line
        row_h = max(1, bottom - top)
        if best_line is not None and best_overlap >= max(8, int(row_h * 0.25)):
            best_line.append(row)
        else:
            visual_lines.append([row])
    visual_lines.sort(key=lambda line: min(to_int(item.get("y", "0")) for item in line))
    rows = [
        item
        for line in visual_lines
        for item in sorted(line, key=lambda r: (
            to_int(r.get("x", "0")),
            to_int(r.get("block", "0")),
        ))
    ]

    # Mark all vertically contiguous fragments of the standard privacy banner.
    # OCR may return its last word (for example "more") as a separate block
    # that is not identifiable as UI when considered in isolation.
    privacy_banner_row_ids: Set[int] = set()
    vertical_rows = sorted(rows, key=lambda r: (
        to_int(r.get("y", "0")),
        to_int(r.get("x", "0")),
    ))
    for index, row in enumerate(vertical_rows):
        anchor_norm = re.sub(
            r"\s+",
            " ",
            re.sub(r"[^a-z0-9]+", " ", row.get("text", "").casefold()),
        ).strip()
        anchor_words = set(anchor_norm.split())
        if not {"messages", "chat", "private"}.issubset(anchor_words):
            continue

        privacy_banner_row_ids.add(id(row))
        previous_bottom = to_int(row.get("y", "0")) + max(
            1, to_int(row.get("h", "1"), 1)
        )
        previous_height = max(1, to_int(row.get("h", "1"), 1))
        for following in vertical_rows[index + 1:]:
            if looks_like_date_or_time(following.get("text", "")):
                break
            following_y = to_int(following.get("y", "0"))
            following_height = max(1, to_int(following.get("h", "1"), 1))
            gap = following_y - previous_bottom
            if gap > max(100, int(max(previous_height, following_height) * 0.80)):
                break
            privacy_banner_row_ids.add(id(following))
            previous_bottom = max(previous_bottom, following_y + following_height)
            previous_height = following_height

    date_y = None
    for row in rows:
        if looks_like_date(row.get("text", "")):
            y = to_int(row.get("y", "0"), -1)
            if y >= 0:
                date_y = y if date_y is None else max(date_y, y)

    current: List[Dict[str, str]] = []
    pending_bubbles: List[List[Dict[str, str]]] = []
    groups: List[Dict[str, str]] = []

    continuation_endings = {
        "a", "an", "and", "as", "at", "because", "but", "by", "for",
        "from", "in", "into", "is", "my", "of", "on", "or", "our",
        "that", "the", "their", "to", "was", "were", "which", "with",
        "would", "your",
    }

    def should_split_before(row: Dict[str, str]) -> bool:
        if not current:
            return False
        previous = current[-1]
        prev_side = previous.get("pos", "").upper()
        next_side = row.get("pos", "").upper()
        prev_bottom = to_int(previous.get("y", "0")) + max(1, to_int(previous.get("h", "1"), 1))
        gap = to_int(row.get("y", "0")) - prev_bottom
        height = max(
            to_int(previous.get("h", "1"), 1),
            to_int(row.get("h", "1"), 1),
        )
        # Wrapped lines in one bubble can receive different side labels when a
        # wide dark bubble crosses the geometric boundary. Their OCR boxes
        # normally overlap vertically or have only a very small gap. Do not,
        # however, merge two distinct compact rounded bubbles merely because
        # they have the same side and timestamp.
        if gap <= max(34, int(height * 0.42)):
            return False
        if prev_side in {"LEFT", "RIGHT"} and next_side in {"LEFT", "RIGHT"} and prev_side != next_side:
            return True
        previous_text = " ".join(item.get("text", "").strip() for item in current).strip()
        words = re.findall(r"[A-Za-z']+", previous_text.casefold())
        last_word = words[-1] if words else ""
        if last_word in continuation_endings or re.search(r"[-,/;(]$", previous_text):
            return False
        # A very large gap is a bubble boundary. A shorter gap is accepted for
        # compact complete messages such as "You need to act quickly".
        return gap >= max(58, int(height * 0.52)) or len(words) <= 8

    def emit(timestamp: str) -> None:
        bubble_sets = list(pending_bubbles)
        if current:
            bubble_sets.append(list(current))
        for bubble_rows in bubble_sets:
            raw_group_text = " ".join(
                item.get("text", "").strip()
                for item in bubble_rows
                if item.get("text", "").strip()
            )
            # A privacy-banner sentence may be fragmented into individually
            # meaningless OCR tokens. Evaluate the reconstructed group as well
            # as each individual block before accepting it as a chat bubble.
            if is_ui_message(raw_group_text):
                continue
            usable = [
                r for r in bubble_rows
                if r.get("pos", "").upper() in {"LEFT", "RIGHT"}
                and not is_ui_message(r.get("text", ""))
            ]
            if not usable:
                continue
            side_scores = {"LEFT": 0, "RIGHT": 0}
            for bubble_row in usable:
                side = bubble_row.get("pos", "").upper()
                side_scores[side] += max(1, len(bubble_row.get("text", "").strip()))
            side = "LEFT" if side_scores["LEFT"] >= side_scores["RIGHT"] else "RIGHT"
            group_text = " ".join(
                bubble_row.get("text", "").strip()
                for bubble_row in usable
                if bubble_row.get("text", "").strip()
            )
            if group_text:
                groups.append({
                    "time": timestamp,
                    "side": side,
                    "text": group_text,
                    "order": len(groups),
                })

    for row in rows:
        text = row.get("text", "").strip()
        pos = row.get("pos", "").upper()
        y = to_int(row.get("y", "0"), 0)

        if id(row) in privacy_banner_row_ids:
            continue

        # Ignore status/header area before the date separator.
        if date_y is not None and y <= date_y:
            continue
        # A continuation screenshot may start in the middle of a bubble and
        # legitimately contain message text above y=250. Its date is inherited
        # only from the same evidence-folder conversation.
        if date_y is None and y < 250 and not allow_top_content:
            continue

        message_prefix, visible_time = split_trailing_visible_time_token(text)

        if visible_time:
            if message_prefix and pos in {"LEFT", "RIGHT"} and not is_ui_message(message_prefix):
                prefix_row = dict(row)
                prefix_row["text"] = message_prefix
                if should_split_before(prefix_row):
                    pending_bubbles.append(list(current))
                    current = []
                current.append(prefix_row)
            emit(visible_time)
            current = []
            pending_bubbles = []
            continue

        if pos in {"LEFT", "RIGHT"} and text and not is_ui_message(text) and not looks_like_date_or_time(text):
            if should_split_before(row):
                pending_bubbles.append(list(current))
                current = []
            current.append(row)

    return repair_isolated_viber_hour_errors(groups)


def repair_isolated_viber_hour_errors(
    groups: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """Repair a strongly constrained OCR error in one bubble's hour.

    A timestamp is corrected only when both adjacent visible bubbles use the
    same hour, the current minute falls between their minutes, and the
    suspicious hour differs from the shared hour by exactly one character.
    This uses only local chronological evidence and never inserts a timestamp
    that is not supported by its two neighbours.
    """
    repaired = [dict(group) for group in groups]
    if len(repaired) < 3:
        return repaired

    def parse_hhmm(value: str) -> Optional[Tuple[str, int]]:
        match = re.fullmatch(r"(\d{2}):(\d{2})", str(value or "").strip())
        if not match:
            return None
        hour, minute = match.group(1), int(match.group(2))
        if int(hour) > 23 or minute > 59:
            return None
        return hour, minute

    for index in range(1, len(repaired) - 1):
        previous = parse_hhmm(repaired[index - 1].get("time", ""))
        current = parse_hhmm(repaired[index].get("time", ""))
        following = parse_hhmm(repaired[index + 1].get("time", ""))
        if not previous or not current or not following:
            continue

        previous_hour, previous_minute = previous
        current_hour, current_minute = current
        following_hour, following_minute = following

        if previous_hour != following_hour or current_hour == previous_hour:
            continue
        if sum(a != b for a, b in zip(current_hour, previous_hour)) != 1:
            continue
        lower = min(previous_minute, following_minute)
        upper = max(previous_minute, following_minute)
        if not lower <= current_minute <= upper:
            continue

        repaired[index]["time"] = f"{previous_hour}:{current_minute:02d}"

    return repaired

def build_viber_bubble_hints(ocr_bubble_groups: Optional[List[Dict[str, str]]]) -> str:
    """Formats Viber OCR bubble groups as prompt hints."""
    # Notes:
    # These are noisy support, but they help the VLM keep one row per rounded bubble.
    if not ocr_bubble_groups:
        return "No reliable Viber bubble hints."

    lines = []
    for i, group in enumerate(ocr_bubble_groups, start=1):
        lines.append(
            f"[BUBBLE {i}] "
            f"TIME={group.get('time', '')} "
            f"SIDE={group.get('side', '')} "
            f"OCR_TEXT={group.get('text', '')}"
        )

    return "\n".join(lines)

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
        if g.get("time") == hhmm and g.get("side") in {"LEFT", "RIGHT"}
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

def normalize_for_viber_fragment(text: str) -> str:
    """Normalizes Viber text for fragment and overlap checks."""
    text = str(text or "").lower()
    text = text.replace("’", "'")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def viber_token_overlap_score(a: str, b: str) -> float:
    """Scores Viber token overlap while preserving duplicate-token counts."""
    a_norm = normalize_for_viber_fragment(a)
    b_norm = normalize_for_viber_fragment(b)

    a_tokens = a_norm.split()
    b_tokens = b_norm.split()

    if not a_tokens or not b_tokens:
        return 0.0

    b_counts: Dict[str, int] = {}
    for tok in b_tokens:
        b_counts[tok] = b_counts.get(tok, 0) + 1

    overlap = 0
    for tok in a_tokens:
        if b_counts.get(tok, 0) > 0:
            overlap += 1
            b_counts[tok] -= 1

    return overlap / max(1, min(len(a_tokens), len(b_tokens)))

def split_message_by_ocr_bubbles(
    message: str,
    hhmm: Optional[str],
    side: str,
    ocr_bubble_groups: Optional[List[Dict[str, str]]] = None,
) -> List[Tuple[Optional[Dict[str, str]], str]]:
    """Split a clearly merged Viber row using OCR bubble groups as boundaries only."""
    # Important: OCR bubble text is noisy. We use it to detect that a split is
    # needed, but we do not blindly replace good VLM text with raw OCR text.
    if not hhmm or not ocr_bubble_groups:
        return [(None, message)]

    # Prefer same-time candidates, but also allow adjacent same-side bubbles when
    # the draft row visibly merged text from neighboring timestamped bubbles.
    candidates = [
        g for g in ocr_bubble_groups
        if g.get("time") == hhmm and g.get("side") == side
    ]

    msg_norm = normalize_for_viber_fragment(message)
    if len(msg_norm.split()) < 9:
        return [(None, message)]

    if len(candidates) < 2:
        # Include all same-side groups whose text is strongly represented in the row.
        candidates = [
            g for g in ocr_bubble_groups
            if g.get("side") == side
            and (
                normalize_for_viber_fragment(str(g.get("text", ""))) in msg_norm
                or viber_token_overlap_score(message, str(g.get("text", ""))) >= 0.86
            )
        ]

    if len(candidates) < 2:
        return [(None, message)]

    matches: List[Dict[str, str]] = []
    for group in candidates:
        group_text = str(group.get("text", "")).strip()
        group_norm = normalize_for_viber_fragment(group_text)
        if len(group_norm.split()) < 3:
            continue
        # High threshold: only split when the OCR group is clearly represented.
        if group_norm in msg_norm or viber_token_overlap_score(message, group_text) >= 0.90:
            matches.append(group)

    if len(matches) < 2:
        return [(None, message)]

    ordered = sorted(matches, key=lambda g: int(g.get("order", 0)))

    # Prefer a clean OCR group text only if it is not visibly noisy. Otherwise
    # keep the original VLM row unsplit rather than injecting noisy OCR text.
    split_rows: List[Tuple[Optional[Dict[str, str]], str]] = []
    for group in ordered:
        group_text = str(group.get("text", "")).strip()
        if looks_like_noisy_ocr_text(group_text):
            return [(None, message)]
        split_rows.append((group, group_text))

    return split_rows

def best_ocr_group_for_message(
    message: str,
    hhmm: Optional[str],
    side: str,
    ocr_bubble_groups: Optional[List[Dict[str, str]]] = None,
) -> Optional[Dict[str, str]]:
    """Finds the best matching OCR bubble group for a Viber message."""
    if not hhmm or not ocr_bubble_groups:
        return None

    candidates = [
        g for g in ocr_bubble_groups
        if g.get("time") == hhmm and g.get("side") == side
    ]

    if not candidates:
        return None

    best = None
    best_score = 0.0

    for group in candidates:
        score = viber_token_overlap_score(message, group.get("text", ""))
        if score > best_score:
            best_score = score
            best = group

    if best_score >= 0.66:
        return best

    return None

def repair_message_from_ocr_group_if_suspicious(
    message: str,
    group: Optional[Dict[str, str]],
) -> str:
    """Use OCR bubble text only for structurally suspicious Viber rows.

    The OCR group is not a general text source. It is used only when the VLM
    output has clear orphan/word-order artifacts and the OCR bubble appears
    cleaner for the same visible bubble.
    """
    if group is None:
        return message

    msg = str(message or "").strip()
    raw = str(group.get("text", "")).strip()
    if not msg or not raw:
        return message

    if looks_like_noisy_ocr_text(raw):
        return message

    score = viber_token_overlap_score(msg, raw)
    if score < 0.86:
        return message

    msg_norm = normalize_for_viber_fragment(msg)
    raw_norm = normalize_for_viber_fragment(raw)
    msg_words = msg_norm.split()
    raw_words = raw_norm.split()
    if len(msg_words) < 3 or len(raw_words) < 3:
        return message

    suspicious = (
        bool(msg[:1].islower())
        or bool(re.search(r"\b(they|can|the|and|but)\s+(The|I|I'm|I'll|You|We|They)\b", msg))
        or bool(re.search(r"\b(?:I{2}l|1['’]?ll|Ijust|Iwill|Icant|yoU|sO|Ji00)\b", msg))
    )
    if not suspicious:
        return message

    if bool(msg[:1].islower()) and bool(raw[:1].isupper()):
        return raw

    if sorted(msg_words) == sorted(raw_words) and bool(msg[:1].islower()):
        return raw

    return message

def add_missing_ocr_bubbles_to_rows(
    rows: List[Dict[str, object]],
    visible_date: str,
    emoji_mode: str,
    ocr_bubble_groups: Optional[List[Dict[str, str]]] = None,
) -> None:
    """Adds omitted high-confidence Viber OCR bubble rows."""
    # Notes:
    # Safer than Messenger auto-add because Viber has per-bubble timestamps. Still
    # conservative: no add if a same-time/same-side row already overlaps strongly.
    if not visible_date or not ocr_bubble_groups:
        return

    for group in ocr_bubble_groups:
        side = str(group.get("side", "")).upper()
        hhmm = str(group.get("time", "")).strip()
        raw_text = str(group.get("text", "")).strip()

        if side not in {"LEFT", "RIGHT"} or not re.fullmatch(r"\d{2}:\d{2}", hhmm):
            continue

        if len(normalize_for_viber_fragment(raw_text).split()) < 4:
            continue

        # Do not add raw OCR as a final row if it looks noisy; this keeps OCR
        # hints as evidence for split/time/side rather than transcript text.
        if looks_like_noisy_ocr_text(raw_text):
            continue

        time_value = f"{visible_date} {hhmm}"

        already_present = False
        for row in rows:
            if row["time"] != time_value or row["side"] != side:
                continue

            if (
                viber_token_overlap_score(str(row["message"]), raw_text) >= 0.62
                or viber_token_overlap_score(raw_text, str(row["message"])) >= 0.62
            ):
                already_present = True
                break

        if already_present:
            continue

        message = clean_message_text(raw_text)
        if emoji_mode == "omit":
            message = strip_emojis(message)

        if is_ui_message(message):
            continue

        rows.append({
            "time": time_value,
            "side": side,
            "message": message,
            "order": int(group.get("order", 9999)),
        })

# Cleans the LLM CSV, validates times/sides, and removes UI noise.
def normalize_side_csv(
    side_csv: str,
    visible_date: str,
    default_year: int,
    allowed_times: Set[str],
    emoji_mode: str = "omit",
    ocr_bubble_groups: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Cleans and validates model side CSV rows."""
    text = strip_code_fences(side_csv)

    reader = csv.reader(io.StringIO(text))
    out = io.StringIO()
    writer = csv.writer(out, quoting=csv.QUOTE_ALL, lineterminator="\n")

    writer.writerow(["Time", "Side", "Message"])

    rows_to_write: List[Dict[str, object]] = []
    emitted_index = 0

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

        if allowed_times and hhmm not in allowed_times:
            continue

        if not re.fullmatch(r"\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}", time_value):
            continue

        ocr_side = infer_side_for_message_from_ocr(
            message=message,
            hhmm=hhmm,
            ocr_bubble_groups=ocr_bubble_groups,
        )
        if ocr_side in {"LEFT", "RIGHT"}:
            side = ocr_side

        candidate_messages = split_message_by_ocr_bubbles(
            message=message,
            hhmm=hhmm,
            side=side,
            ocr_bubble_groups=ocr_bubble_groups,
        )

        for group, candidate_message in candidate_messages:
            chosen_group = group or best_ocr_group_for_message(
                message=candidate_message,
                hhmm=hhmm,
                side=side,
                ocr_bubble_groups=ocr_bubble_groups,
            )

            candidate_message = repair_message_from_ocr_group_if_suspicious(
                message=candidate_message,
                group=chosen_group,
            )

            message_clean = clean_message_text(candidate_message)
            if emoji_mode == "omit":
                message_clean = strip_emojis(message_clean)

            if is_ui_message(message_clean):
                continue

            order = emitted_index
            row_time_value = time_value
            if chosen_group is not None:
                try:
                    order = int(chosen_group.get("order", emitted_index))
                except Exception:
                    order = emitted_index
                group_hhmm = str(chosen_group.get("time", "")).strip()
                if visible_date and re.fullmatch(r"\d{2}:\d{2}", group_hhmm):
                    row_time_value = f"{visible_date} {group_hhmm}"

            rows_to_write.append({
                "time": row_time_value,
                "side": side,
                "message": message_clean,
                "order": order,
            })

            emitted_index += 1

    add_missing_ocr_bubbles_to_rows(
        rows=rows_to_write,
        visible_date=visible_date,
        emoji_mode=emoji_mode,
        ocr_bubble_groups=ocr_bubble_groups,
    )

    rows_to_write.sort(
        key=lambda item: (
            extract_hhmm_from_full_time(str(item["time"])) or "99:99",
            int(item.get("order", 9999)),
        )
    )

    seen = set()

    for item in rows_to_write:
        row_tuple = (
            str(item["time"]),
            str(item["side"]),
            str(item["message"]),
        )

        if row_tuple in seen:
            continue

        seen.add(row_tuple)
        writer.writerow(row_tuple)

    return out.getvalue().strip() + "\n"

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
def write_debug_text(path: Path, content: str) -> None:
    """Writes debug text to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(content or ""), encoding="utf-8")

def process_viber_image(
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
    """Runs the full Viber extraction pipeline for one image."""
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

    if dump_ocr or dump_draft or dump_side_map:
        write_debug_text(
            output_debug_dir / "run_config.json",
            json.dumps(
                {
                    "image_path": image_path,
                    "model": model,
                    "langs": langs,
                    "use_gpu": use_gpu,
                    "grid": grid,
                    "layout": layout,
                    "use_vision": use_vision,
                    "emoji_mode": emoji_mode,
                    "dump_ocr": dump_ocr,
                    "dump_draft": dump_draft,
                    "dump_side_map": dump_side_map,
                },
                ensure_ascii=False,
                indent=2,
            ),
        )

    # Load EasyOCR once; it will be reused for all screen crops.
    reader = easyocr.Reader(langs, gpu=use_gpu)

    # Extract participant context from the report for side mapping.
    # Keep the full structure out of the normal run log, but preserve it in an
    # explicitly requested debug folder so actor-parser failures are auditable.
    actors = infer_report_actors(report_text, model)
    if dump_side_map:
        write_debug_text(
            output_debug_dir / "report_actors.json",
            json.dumps(actors, ensure_ascii=False, indent=2),
        )

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
            write_debug_text(output_debug_dir / f"screen_{idx:02d}_ocr.txt", screen_ocr)

    full_ocr = "\n\n".join(all_screen_ocr)

    if dump_ocr:
        write_debug_text(output_debug_dir / "all_ocr.txt", full_ocr)

    # Infer one fixed LEFT/RIGHT mapping for the whole conversation.
    print("-> [SIDE MAP] Inferring fixed LEFT/RIGHT mapping...")
    side_map = infer_side_mapping(
        report_text,
        actors,
        full_ocr,
        "viber",
        model,
    )

    print(f"-> [SIDE MAP] LEFT = {side_map['LEFT']} | RIGHT = {side_map['RIGHT']}")

    if dump_side_map:
        write_debug_text(
            output_debug_dir / "side_map.json",
            json.dumps(side_map, ensure_ascii=False, indent=2),
        )

    # Second pass: ask the VLM to extract each screen as Side CSV. If this file
    # is a continuation screenshot with no visible date separator, reuse only
    # the last validated date from the same platform/folder conversation.
    side_csv_parts = []
    previous_date_hint = load_conversation_date_hint(
        conversation_state_cache,
        conversation_key,
    )
    if previous_date_hint:
        print(f"-> [DATE] Previous folder date: {previous_date_hint}")

    for idx, screen_ocr in enumerate(all_screen_ocr, start=1):
        print(f"-> [LLM] Extracting screen #{idx} as Side CSV...")

        # Detect an explicit separator independently from the inherited hint.
        # A collage splitter may place the date separator in its own crop and
        # the actual message bubbles in the following crops. Persist the date
        # immediately, even when the date-only crop produces no chat row.
        explicit_visible_date = extract_visible_date_from_ocr(
            screen_ocr,
            default_year=default_year,
            previous_date_hint="",
        )
        if explicit_visible_date:
            previous_date_hint = explicit_visible_date
            save_conversation_date_hint(
                conversation_state_cache,
                conversation_key,
                explicit_visible_date,
            )
        visible_date = explicit_visible_date or previous_date_hint

        # Limit accepted times to message-bubble times. The bubble grouping also
        # applies a tightly constrained local correction for an isolated hour
        # OCR error between two chronologically consistent neighbouring rows.
        ocr_bubble_groups = build_ocr_bubble_groups(
            screen_ocr,
            allow_top_content=bool(previous_date_hint),
        )
        grouped_times = {
            str(group.get("time", "")).strip()
            for group in ocr_bubble_groups
            if re.fullmatch(r"\d{2}:\d{2}", str(group.get("time", "")).strip())
        }
        allowed_times = grouped_times or extract_allowed_times_from_ocr(screen_ocr)
        bubble_hints = build_viber_bubble_hints(ocr_bubble_groups)

        if dump_draft:
            write_debug_text(output_debug_dir / f"screen_{idx:02d}_visible_date.txt", visible_date)
            write_debug_text(
                output_debug_dir / f"screen_{idx:02d}_allowed_times.json",
                json.dumps(sorted(allowed_times), ensure_ascii=False, indent=2),
            )
            write_debug_text(
                output_debug_dir / f"screen_{idx:02d}_ocr_bubbles.json",
                json.dumps(ocr_bubble_groups, ensure_ascii=False, indent=2),
            )
            write_debug_text(output_debug_dir / f"screen_{idx:02d}_bubble_hints.txt", bubble_hints)

        prompt = build_screen_prompt(
            screen_ocr=screen_ocr,
            screen_index=idx,
            visible_date=visible_date,
            default_year=default_year,
            allowed_times=allowed_times,
            emoji_mode=emoji_mode,
            bubble_hints=bubble_hints,
        )

        crop_path = str(screen_paths[idx - 1])

        if dump_draft:
            write_debug_text(output_debug_dir / f"screen_{idx:02d}_prompt.txt", prompt)

        # Initial VLM extraction from screenshot + OCR hints.
        draft = ollama_chat_screen(
            model,
            prompt,
            crop_path,
            use_vision=use_vision,
        )

        if dump_draft:
            write_debug_text(output_debug_dir / f"screen_{idx:02d}_draft_raw.csv", draft)
            # Keep the old Facebook-like/debug filename too.
            write_debug_text(output_debug_dir / f"screen_{idx:02d}_draft.csv", draft)

        draft_norm = normalize_side_csv(
            draft,
            visible_date=visible_date,
            default_year=default_year,
            allowed_times=allowed_times,
            emoji_mode=emoji_mode,
            ocr_bubble_groups=ocr_bubble_groups,
        )

        if dump_draft:
            write_debug_text(output_debug_dir / f"screen_{idx:02d}_draft_norm.csv", draft_norm)

        expected_bubbles = len(ocr_bubble_groups) if ocr_bubble_groups else 0
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

            if dump_draft:
                write_debug_text(output_debug_dir / f"screen_{idx:02d}_repair_prompt.txt", repair_prompt)

            # Repair pass is now run only for suspicious screens/rows.
            repaired = ollama_chat_screen(
                model,
                repair_prompt,
                crop_path,
                use_vision=use_vision,
            )

            if dump_draft:
                write_debug_text(output_debug_dir / f"screen_{idx:02d}_repair_raw.csv", repaired)

            repaired_norm = normalize_side_csv(
                repaired,
                visible_date=visible_date,
                default_year=default_year,
                allowed_times=allowed_times,
                emoji_mode=emoji_mode,
                ocr_bubble_groups=ocr_bubble_groups,
            )

            if dump_draft:
                write_debug_text(output_debug_dir / f"screen_{idx:02d}_repair_norm.csv", repaired_norm)
        else:
            print(f"-> [LLM] Screen #{idx} looks stable; skipping repair pass.")
            repaired_norm = ""

        # Keep the repaired CSV only if it stays within a sane row count.
        chosen_source = "draft_norm"
        if repaired_norm:
            chosen = choose_best_screen_side_csv(
                draft_norm=draft_norm,
                repaired_norm=repaired_norm,
                allowed_times=allowed_times,
                expected_bubble_count=expected_bubbles,
                bubble_groups=ocr_bubble_groups,
                enable_additive=False,
            )
            if chosen == repaired_norm:
                chosen_source = "repaired_norm"
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
                write_debug_text(output_debug_dir / f"screen_{idx:02d}_polish_prompt.txt", polish_prompt)

            polished_raw = ollama_chat_screen(
                model,
                polish_prompt,
                crop_path,
                use_vision=use_vision,
            )

            if dump_draft:
                write_debug_text(output_debug_dir / f"screen_{idx:02d}_polish_raw.csv", polished_raw)

            polished_norm = normalize_polished_side_csv_against_reference(
                polished_raw,
                chosen,
                emoji_mode=emoji_mode,
            )

            if polished_norm:
                polished_choice = choose_text_polished_side_csv(
                    reference_csv=chosen,
                    polished_csv=polished_norm,
                    allowed_times=allowed_times,
                    expected_bubble_count=expected_bubbles,
                    bubble_groups=ocr_bubble_groups,
                )
                if polished_choice != chosen:
                    chosen_source += "+row_polish"
                chosen = polished_choice

        # Final timestamp policy for Viber: keep visible per-bubble minutes and
        # add zero seconds to every row.
        chosen = ensure_zero_seconds_side_csv(chosen)

        if dump_draft:
            write_debug_text(output_debug_dir / f"screen_{idx:02d}_chosen_source.txt", chosen_source)
            write_debug_text(output_debug_dir / f"screen_{idx:02d}_side.csv", chosen)

        last_date = extract_last_date_from_side_csv(chosen)
        if last_date:
            previous_date_hint = last_date
            save_conversation_date_hint(
                conversation_state_cache,
                conversation_key,
                previous_date_hint,
            )

        side_csv_parts.append(chosen)

    # Merge all screen-level CSV parts before applying real names.
    merged_side_csv = merge_side_csvs(side_csv_parts)

    # Bubble grouping has already decided row boundaries. Do not merge two
    # distinct Viber bubbles merely because their sender and minute match.
    merged_side_csv = postprocess_side_csv_rows(
        merged_side_csv,
        merge_continuations=False,
    )

    if dump_draft:
        write_debug_text(output_debug_dir / "merged_side.csv", merged_side_csv)

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
        platform_hint="viber",
        model=model,
        prior_side_map=prior_side_map,
        evidence_hint=image_path,
    )
    # Preserve uniquely matched avatar-initial evidence even if the broader
    # context pass proposed the opposite visual orientation.
    side_map = refine_side_map_with_avatar_initials(
        side_map,
        actors,
        full_ocr,
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
        write_debug_text(
            output_debug_dir / "side_map_initial.json",
            json.dumps(provisional_side_map, ensure_ascii=False, indent=2),
        )
        write_debug_text(
            output_debug_dir / "side_map.json",
            json.dumps(side_map, ensure_ascii=False, indent=2),
        )

    # Replace LEFT/RIGHT with actual Sender/Receiver names.
    # The final writer normally trusts the accepted visual side mapping.
    # A second constrained Gemma pass is enabled only when deterministic
    # checks detect collapsed or contradictory row directions.
    final_csv = apply_side_mapping(
        merged_side_csv,
        side_map,
        # model=model,
        # report_text=report_text,
        # platform_hint="viber",
        # evidence_hint=image_path,
    )

    if dump_draft:
        write_debug_text(output_debug_dir / "final.csv", final_csv)

    return final_csv

# ============================================================
# MAIN
# ============================================================
# CLI entry point: parse arguments, run extraction, and write outputs.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract Viber chat CSV from screenshot or collage screenshot."
    )

    parser.add_argument(
        "image",
        help="Path to Viber screenshot/collage image",
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
        "--dump_ocr",
        action="store_true",
        help="Save positioned OCR debug files",
    )

    parser.add_argument(
        "-dump-draft",
        "--dump-draft",
        "--dump_draft",
        action="store_true",
        help="Save draft/repaired Side CSV files",
    )

    parser.add_argument(
        "-dump-side-map",
        "--dump-side-map",
        "--dump_side_map",
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
        final_csv = process_viber_image(
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