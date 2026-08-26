"""Extract one email screenshot into a single CSV transcript row.

Output schema:
    Timestamp, Estimated_Timestamp, Sender, Receiver, Message

The extractor uses a vision-capable Ollama model by default.  The case report is
used only to resolve identities and missing date context; it must never be used
to invent message text that is not visible in the screenshot.
"""


import argparse
import csv
import json
import re
import sys
import unicodedata
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

try:
    from PyPDF2 import PdfReader
except ImportError:  # Newer package name.
    from pypdf import PdfReader


MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}


def read_report(path_value: str) -> str:
    """Read a TXT or PDF case report."""
    path = Path(path_value)
    suffix = path.suffix.lower()

    if suffix == ".txt":
        for encoding in ("utf-8-sig", "utf-8", "latin-1"):
            try:
                return path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
        return path.read_text(errors="ignore")

    if suffix == ".pdf":
        reader = PdfReader(str(path))
        pages: List[str] = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if text.strip():
                pages.append(text)
        return "\n".join(pages)

    raise ValueError("Case report must be a PDF or TXT file.")


def extract_json_object(text: str) -> Dict[str, Any]:
    """Parse a JSON object even when the model wrapped it in markdown."""
    value = str(text or "").strip()
    value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.I)
    value = re.sub(r"\s*```$", "", value)

    start = value.find("{")
    end = value.rfind("}")
    if start >= 0 and end > start:
        value = value[start : end + 1]

    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def ollama_chat(
    model: str,
    messages: List[Dict[str, Any]],
    *,
    num_predict: int = 2048,
) -> str:
    """Call Ollama without importing it until execution time."""
    try:
        import ollama
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(f"Could not import ollama: {exc}") from exc

    response = ollama.chat(
        model=model,
        messages=messages,
        options={
            "temperature": 0,
            # Reserving fewer output tokens for short identity-resolution calls
            # leaves more of the model context available for report evidence.
            "num_predict": int(num_predict),
        },
    )
    return str(response["message"]["content"] or "").strip()


def normalize_space(text: Any) -> str:
    value = str(text or "")
    value = value.replace("\u00a0", " ")
    value = value.replace("“", '"').replace("”", '"')
    value = value.replace("‘", "'").replace("’", "'")
    value = re.sub(r"\s+", " ", value).strip()
    value = re.sub(r"\s+([,.;:!?])", r"\1", value)
    return value


def extract_salutation_identity(value: Any) -> str:
    """Extract a visible addressee name from an email salutation."""
    text = normalize_space(value)
    if not text:
        return ""

    match = re.match(
        r"(?i)^(?:dear|hello|hi|good\s+(?:morning|afternoon|evening))\s+"
        r"(?P<name>.+?)(?:[,!:;]|$)",
        text,
    )
    if not match:
        return ""

    name = normalize_space(match.group("name"))
    name = re.sub(
        r"(?i)^(?:mr|mrs|ms|miss|dr|prof|professor)\.?\s+",
        "",
        name,
    )
    name = name.strip(" ,.;:!?")

    if name.lower() in {
        "sir", "madam", "sir or madam", "customer", "client",
        "valued customer", "valued client", "recipient",
    }:
        return ""
    return name


def _fold_identity_text(value: Any) -> str:
    """Normalize identity text for robust, accent-insensitive comparison."""
    text = normalize_space(value)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.casefold()
    text = re.sub(r"[^a-z0-9' -]+", " ", text)
    return normalize_space(text)


def _name_tokens(value: Any) -> List[str]:
    text = normalize_space(value)
    text = re.sub(
        r"(?i)^(?:mr|mrs|ms|miss|dr|prof|professor)\.?\s+",
        "",
        text,
    )
    return [
        _fold_identity_text(token).strip(" .'-")
        for token in re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ'’.-]+", text)
        if _fold_identity_text(token).strip(" .'-")
    ]


def _report_person_candidates(report_text: str) -> List[str]:
    """Collect plausible person names while filtering report headings."""
    patterns = (
        r"\b(?:Mr|Mrs|Ms|Miss|Dr|Prof|Professor)\.?\s+"
        r"[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’-]{1,30}"
        r"(?:\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’-]{1,30}){0,3}\b",
        r"\b[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’-]{1,30}"
        r"(?:\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’-]{1,30}){1,3}\b",
        r"\b[A-Z]{2,}(?:\s+[A-Z]{2,}){1,3}\b",
    )
    blocked_phrases = {
        "case report", "executive summary", "possible offences",
        "supporting evidence", "timeline overview", "email screenshot",
        "claims department", "clearance department", "payment details",
        "action required", "customer support", "evidence package",
        "financial authorities", "investment platform",
    }
    blocked_tokens = {
        "report", "summary", "timeline", "evidence", "package", "offences",
        "offenses", "department", "authorities", "screenshot", "payment",
        "details", "subject", "sender", "receiver", "message",
    }

    candidates: List[str] = []
    seen = set()
    for pattern in patterns:
        for match in re.finditer(pattern, str(report_text or "")):
            candidate = normalize_space(match.group(0)).strip(" ,.;:!?")
            candidate = re.sub(
                r"(?i)^(?:mr|mrs|ms|miss|dr|prof|professor)\.?\s+",
                "",
                candidate,
            )
            tokens = _name_tokens(candidate)
            key = " ".join(tokens)
            if len(tokens) < 2 or key in blocked_phrases or key in seen:
                continue
            if any(token in blocked_tokens for token in tokens):
                continue
            # Person names should not be dominated by long all-uppercase labels.
            if candidate.isupper() and len(tokens) > 3:
                continue
            seen.add(key)
            candidates.append(candidate)
    return candidates


def _person_role_bonus(report_text: str, candidate: str) -> float:
    """Softly favor people described as the affected/receiving party."""
    report = str(report_text or "")
    tokens = _name_tokens(candidate)
    if not tokens:
        return 0.0
    pattern = r"\b" + r"\s+".join(re.escape(token) for token in tokens) + r"\b"
    bonus = 0.0
    folded_report = _fold_identity_text(report)
    for match in re.finditer(pattern, folded_report, flags=re.I):
        window = folded_report[max(0, match.start() - 180): match.end() + 180]
        if re.search(
            r"\b(?:victim|complainant|recipient|addressee|account holder|"
            r"affected investor|customer|client|reporting person)\b",
            window,
        ):
            bonus += 8.0
        if re.search(r"\b(?:suspect|offender|fraudster|sender)\b", window):
            bonus -= 2.0
    return max(-4.0, min(16.0, bonus))


def canonicalize_person_from_report(visible_name: Any, report_text: str) -> str:
    """Map OCR/VLM name variants to a report-supported canonical spelling."""
    visible = normalize_space(visible_name).strip(" ,.;:!?")
    visible = re.sub(
        r"(?i)^(?:mr|mrs|ms|miss|dr|prof|professor)\.?\s+",
        "",
        visible,
    )
    if not visible or "@" in visible:
        return ""

    visible_tokens = _name_tokens(visible)
    if not visible_tokens:
        return ""

    candidates = _report_person_candidates(report_text)
    ranked: List[Tuple[float, int, str]] = []

    for index, candidate in enumerate(candidates):
        candidate_tokens = _name_tokens(candidate)
        if not candidate_tokens:
            continue

        exact = int(visible_tokens == candidate_tokens)
        visible_last = visible_tokens[-1]
        candidate_last = candidate_tokens[-1]
        last_similarity = SequenceMatcher(None, visible_last, candidate_last).ratio()

        if len(visible_tokens) == 1:
            # A title/surname salutation can safely use an exact or very close
            # report surname, but should not jump to a different surname.
            if last_similarity < 0.84:
                continue
            score = 72.0 * last_similarity
        else:
            first_similarity = SequenceMatcher(
                None, visible_tokens[0], candidate_tokens[0]
            ).ratio()
            if last_similarity < 0.76 or first_similarity < 0.60:
                continue
            length_penalty = abs(len(visible_tokens) - len(candidate_tokens)) * 3.0
            score = (
                125.0 * exact
                + 48.0 * first_similarity
                + 62.0 * last_similarity
                - length_penalty
            )
            overlap = len(set(visible_tokens) & set(candidate_tokens))
            score += overlap * 6.0

        score += _person_role_bonus(report_text, candidate)
        # Earlier report mentions are a very small tie-breaker only.
        ranked.append((score, -index, candidate))

    if ranked:
        best_score, _, best = max(ranked)
        threshold = 58.0 if len(visible_tokens) == 1 else 82.0
        if best_score >= threshold:
            return best

    # Keep a visibly complete name when the report offers no safe match.
    return visible


def _exact_report_casing(value: str, report_text: str) -> str:
    """Return the report's original casing for an exact phrase match."""
    tokens = re.findall(r"[A-Za-z0-9À-ÖØ-öø-ÿ&'’.-]+", normalize_space(value))
    if not tokens:
        return ""
    pattern = r"(?i)(?<!\w)" + r"\s+".join(re.escape(token) for token in tokens) + r"(?!\w)"
    match = re.search(pattern, str(report_text or ""))
    return normalize_space(match.group(0)) if match else ""


def _smart_entity_case(value: Any) -> str:
    """Normalize an all-uppercase organization without damaging mixed-case brands."""
    text = normalize_space(value)
    if not text or not text.isupper() or len(text.split()) < 2:
        return text

    preserve = {
        "EU", "UK", "US", "USA", "UN", "UAE", "IBAN", "VAT", "LLC",
        "PLC", "LTD", "GMBH", "SA", "AG", "BV", "NV",
    }
    words: List[str] = []
    for word in text.split():
        core = word.strip(".,;:()[]{}")
        prefix = word[: len(word) - len(word.lstrip(".,;:()[]{}"))]
        suffix = word[len(word.rstrip(".,;:()[]{}")):]
        if core in preserve or any(char.isdigit() for char in core):
            transformed = core
        else:
            transformed = core[:1].upper() + core[1:].lower()
        words.append(prefix + transformed + suffix)
    return " ".join(words)


def canonicalize_sender_entity(value: Any, report_text: str) -> str:
    """Use report-supported casing, otherwise conservatively normalize all caps."""
    sender = normalize_space(value)
    if not sender:
        return ""
    report_case = _exact_report_casing(sender, report_text)
    return report_case or _smart_entity_case(sender)


def extract_report_year(report_text: str, fallback: int = 2026) -> int:
    """Choose a likely evidence year from the report."""
    years = [int(value) for value in re.findall(r"\b(20\d{2})\b", report_text)]
    if not years:
        return fallback

    # Timeline reports often repeat their operative year.  Frequency is more
    # robust than selecting one arbitrary first occurrence such as a DOB year.
    counts: Dict[int, int] = {}
    for year in years:
        counts[year] = counts.get(year, 0) + 1
    return max(counts, key=lambda year: (counts[year], year))


def normalize_date(date_text: Any, report_year: int) -> str:
    """Normalize visible email dates to DD/MM/YYYY, leaving time absent."""
    raw = normalize_space(date_text)
    if not raw:
        return ""

    # Prefer explicit ISO or numeric dates.
    for fmt in (
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%m/%d/%Y",
        "%B %d, %Y",
        "%b %d, %Y",
        "%d %B %Y",
        "%d %b %Y",
    ):
        try:
            return datetime.strptime(raw, fmt).strftime("%d/%m/%Y")
        except ValueError:
            pass

    # Remove weekday and ordinal suffixes before a more permissive parse.
    cleaned = re.sub(
        r"(?i)\b(?:mon(?:day)?|tue(?:sday)?|wed(?:nesday)?|thu(?:rsday)?|fri(?:day)?|sat(?:urday)?|sun(?:day)?)\b,?",
        " ",
        raw,
    )
    cleaned = re.sub(r"(?i)(\d)(st|nd|rd|th)\b", r"\1", cleaned)
    cleaned = normalize_space(cleaned)

    match = re.search(
        r"(?i)\b(?P<day>\d{1,2})\s+(?P<month>[A-Za-z]{3,9})(?:\s*,?\s*(?P<year>20\d{2}))?\b",
        cleaned,
    )
    if match:
        month = MONTHS.get(match.group("month").lower())
        if month:
            year = int(match.group("year") or report_year)
            try:
                return datetime(year, month, int(match.group("day"))).strftime("%d/%m/%Y")
            except ValueError:
                return ""

    match = re.search(
        r"(?i)\b(?P<month>[A-Za-z]{3,9})\s+(?P<day>\d{1,2})(?:\s*,?\s*(?P<year>20\d{2}))?\b",
        cleaned,
    )
    if match:
        month = MONTHS.get(match.group("month").lower())
        if month:
            year = int(match.group("year") or report_year)
            try:
                return datetime(year, month, int(match.group("day"))).strftime("%d/%m/%Y")
            except ValueError:
                return ""

    # Numeric date without year, interpreted day-first for this evidence set.
    match = re.search(r"\b(?P<day>\d{1,2})[/-](?P<month>\d{1,2})\b", cleaned)
    if match:
        try:
            return datetime(
                report_year,
                int(match.group("month")),
                int(match.group("day")),
            ).strftime("%d/%m/%Y")
        except ValueError:
            return ""

    return ""


def report_excerpt(report_text: str, draft: Dict[str, Any], max_chars: int = 4500) -> str:
    """Retrieve compact report blocks relevant to draft identities and date."""
    report = str(report_text or "").strip()
    if len(report) <= max_chars:
        return report

    query_text = " ".join(
        normalize_space(draft.get(key, ""))
        for key in (
            "sender_display",
            "sender_address",
            "sender_brand",
            "recipient_text",
            "salutation",
            "date_text",
            "subject",
        )
    ).lower()
    query_terms = {
        term
        for term in re.findall(r"[a-z0-9@.-]{3,}", query_text)
        if term not in {"the", "and", "from", "with", "email", "inbox", "dear"}
    }

    blocks = [
        normalize_space(block)
        for block in re.split(r"\n\s*\n|(?<=\.)\s+(?=[A-Z])", report)
        if normalize_space(block)
    ]
    scored: List[Tuple[int, int, str]] = []
    for index, block in enumerate(blocks):
        lower = block.lower()
        overlap = sum(1 for term in query_terms if term in lower)
        identity_bonus = 2 if re.search(
            r"(?i)\b(victim|complainant|suspect|sender|receiver|email|organisation|organization)\b",
            block,
        ) else 0
        date_bonus = 1 if re.search(r"\b20\d{2}\b", block) else 0
        scored.append((overlap * 5 + identity_bonus + date_bonus, index, block))

    # Put the highest-scoring evidence first. The previous implementation put
    # blocks back into report order and then sliced by characters, which could
    # discard the relevant recipient block at the end of a long report.
    ordered_indexes: List[int] = []
    for _, index, _ in sorted(scored, key=lambda item: (item[0], -item[1]), reverse=True)[:18]:
        if index not in ordered_indexes:
            ordered_indexes.append(index)
    for index in range(min(4, len(blocks))):
        if index not in ordered_indexes:
            ordered_indexes.append(index)

    selected: List[str] = []
    used_chars = 0
    for index in ordered_indexes:
        block = blocks[index]
        addition = len(block) + (1 if selected else 0)
        if selected and used_chars + addition > max_chars:
            continue
        if not selected and len(block) > max_chars:
            block = block[:max_chars]
            addition = len(block)
        selected.append(block)
        used_chars += addition
        if used_chars >= max_chars:
            break

    return "\n".join(selected)[:max_chars]


def build_vision_prompt() -> str:
    return r"""
You are extracting one open email from an evidence screenshot.
Return ONLY one JSON object using this exact schema:
{
  "is_email": true,
  "date_text": "date exactly as visibly shown",
  "sender_display": "visible From display name",
  "sender_address": "visible sender email address, if shown",
  "sender_brand": "clearly visible organization/brand name, if shown",
  "recipient_text": "visible To recipient, or 'me' when that is all the UI shows",
  "salutation": "opening salutation only, if present",
  "subject": "visible subject",
  "layout_type": "plain_text, designed_html, or unknown",
  "signature_contact": "email/phone/web address visibly located inside the signature block, not in the From header",
  "content_blocks": [
    {
      "order": 1,
      "kind": "body, closing, signature, contact, callout, table, footer, button, or other",
      "flow": "primary or separate",
      "text": "visible text for this block"
    }
  ],
  "full_body_text": "all visible email-message text after the salutation, in reading order",
  "primary_body": "continuous primary-flow message text, excluding the salutation"
}

Rules:
1. Extract only the open email. Exclude navigation, toolbar/status text, buttons, reply controls, inbox counts, labels, and unrelated UI.
2. Put the opening greeting only in salutation. Do not silently discard it.
3. content_blocks must follow visual reading order. Each visually distinct region is a separate block.
4. flow="primary" means the block belongs to the continuous message column/paragraph flow. flow="separate" means a visually detached card, panel, table, badge, button, support box, or footer.
5. For a plain-text email, body paragraphs, closing, signature and signature contact normally have flow="primary".
6. For a designed/HTML email, introductory prose in the main article area has flow="primary". Visually separate action/payment/parcel/support panels, cards, tables, buttons and automated notices have flow="separate", even when they contain readable text.
7. full_body_text is an exact transcription of all visible email-message content after the salutation, in natural reading order. It may include detached panels because it is a transcription field, not a selection decision.
8. primary_body is only the continuous primary-flow message, excluding the salutation. For designed HTML mail, stop before the first detached panel/card. For plain text, include inline payment/bank details, closing and signature.
9. signature_contact must be populated only when the contact is visibly part of the signature/body. Do not copy the From-header address into this field merely because it is visible in the header.
10. Do not include subject or From/To headers in content_blocks, full_body_text, or primary_body.
11. Preserve visible wording exactly. Do not correct, summarize, complete, paraphrase, or invent text. Do not omit the first words of a sentence.
12. sender_brand is the full human-readable parent organization, not a logo acronym alone and not merely a department suffix.
13. If the screenshot is not an email, set is_email=false and leave remaining strings/lists empty.
14. Identity fields may contain only a visible person name, organization name, email address, or the literal UI value 'me'. Never put a document heading, section title, offence label, role, or sentence in an identity field.
15. Treat an email address or website as one indivisible identifier. Join line-wrap/OCR spaces inside it; never split around '.', '/', ':', '@', or the top-level domain.
""".strip()


def extract_draft_with_vision(image_path: Path, model: str) -> Dict[str, Any]:
    text = ollama_chat(
        model,
        [
            {
                "role": "user",
                "content": build_vision_prompt(),
                "images": [str(image_path)],
            }
        ],
        num_predict=2048,
    )
    data = extract_json_object(text)
    if not data:
        raise RuntimeError("Vision model did not return a valid JSON object.")
    return data



def _needs_layout_refinement(draft: Dict[str, Any]) -> bool:
    """Run a cheap classification audit whenever layout evidence is incomplete."""
    layout = normalize_space(draft.get("layout_type")).lower()
    blocks = _ordered_content_blocks(draft) if "_ordered_content_blocks" in globals() else []
    primary_body = str(draft.get("primary_body") or "")

    if layout not in {"plain_text", "designed_html"}:
        return True
    if not blocks:
        return True

    has_separate = any(
        normalize_space(block.get("flow")).lower() == "separate"
        or normalize_space(block.get("kind")).lower()
        in {"callout", "table", "footer", "button"}
        for block in blocks
    )
    has_signature = any(
        normalize_space(block.get("kind")).lower() in {"closing", "signature", "contact"}
        for block in blocks
    ) or _contains_closing_or_signature(primary_body)

    # Designed mail without a detached region and plain mail without a closing
    # are the two cases in which a second, label-only pass is most useful.
    if layout == "designed_html" and not has_separate:
        return True
    if layout == "plain_text" and not has_signature:
        return True
    return False


def refine_layout_with_vision(
    image_path: Path,
    model: str,
    draft: Dict[str, Any],
) -> Dict[str, Any]:
    """Audit only block labels; never ask the second pass to rewrite prose."""
    original_blocks = _ordered_content_blocks(draft)
    compact_blocks = []
    for index, block in enumerate(original_blocks, start=1):
        compact_blocks.append({
            "block_id": index,
            "existing_kind": normalize_space(block.get("kind")),
            "existing_flow": normalize_space(block.get("flow")),
            "text_preview": normalize_space(block.get("text"))[:240],
        })

    prompt = f"""
Audit the visual structure of this email screenshot. Do NOT retranscribe or rewrite the email text.
Return ONLY JSON with this schema:
{{
  "layout_type": "plain_text, designed_html, or unknown",
  "signature_contact": "contact visibly inside the signature, not the From header",
  "block_labels": [
    {{
      "block_id": 1,
      "kind": "body, closing, signature, contact, callout, table, footer, button, or other",
      "flow": "primary or separate"
    }}
  ]
}}

Rules:
- Classify each supplied block_id using the screenshot. Preserve the supplied block ids.
- plain_text means one continuous message column where paragraphs, inline payment details, closing, signature and signature contact remain in the same flow.
- designed_html means a rich template with detached boxes, cards, columns, tables, support panels, badges, buttons, or decorative footers.
- In designed_html, only introductory prose in the main article column is primary. Detached action/payment/parcel/support regions are separate.
- Do not return primary_body, full_body_text, identities, subject, date, or rewritten block text.

BLOCKS TO LABEL:
{json.dumps(compact_blocks, ensure_ascii=False, indent=2)}
""".strip()
    return extract_json_object(
        ollama_chat(
            model,
            [{"role": "user", "content": prompt, "images": [str(image_path)]}],
            num_predict=512,
        )
    )


def merge_layout_refinement(
    draft: Dict[str, Any],
    refined: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge label-only layout evidence while preserving first-pass wording."""
    if not refined:
        return draft

    merged = dict(draft)
    layout = normalize_space(refined.get("layout_type")).lower()
    if layout in {"plain_text", "designed_html"}:
        merged["layout_type"] = layout

    refined_contact = normalize_space(refined.get("signature_contact"))
    if refined_contact and not normalize_space(merged.get("signature_contact")):
        merged["signature_contact"] = refined_contact

    blocks = merged.get("content_blocks")
    labels = refined.get("block_labels")
    if isinstance(blocks, list) and isinstance(labels, list):
        by_id: Dict[int, Dict[str, Any]] = {}
        for label in labels:
            if not isinstance(label, dict):
                continue
            try:
                block_id = int(label.get("block_id"))
            except (TypeError, ValueError):
                continue
            by_id[block_id] = label

        relabelled: List[Any] = []
        for index, block in enumerate(blocks, start=1):
            if not isinstance(block, dict):
                relabelled.append(block)
                continue
            updated = dict(block)
            label = by_id.get(index)
            if label:
                kind = normalize_space(label.get("kind")).lower()
                flow = normalize_space(label.get("flow")).lower()
                if kind in {"body", "closing", "signature", "contact", "callout", "table", "footer", "button", "other"}:
                    updated["kind"] = kind
                if flow in {"primary", "separate"}:
                    updated["flow"] = flow
            relabelled.append(updated)
        merged["content_blocks"] = relabelled

    # Keep the audit itself for deterministic layout scoring and debugging.
    merged["_layout_audit"] = refined
    return merged


def easyocr_lines(image_path: Path, langs: Sequence[str], use_cpu: bool) -> List[str]:
    """OCR fallback used only when vision is disabled or fails."""
    try:
        import easyocr
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(f"Could not import easyocr for fallback extraction: {exc}") from exc

    reader = easyocr.Reader(list(langs), gpu=not use_cpu)
    items = reader.readtext(str(image_path), detail=1, paragraph=False)

    sortable: List[Tuple[float, float, str]] = []
    for box, text, confidence in items:
        if float(confidence) < 0.15:
            continue
        xs = [float(point[0]) for point in box]
        ys = [float(point[1]) for point in box]
        sortable.append((min(ys), min(xs), normalize_space(text)))

    sortable.sort(key=lambda item: (item[0], item[1]))
    return [text for _, _, text in sortable if text]


def extract_draft_from_ocr(ocr_lines: Sequence[str], model: str) -> Dict[str, Any]:
    ocr_text = "\n".join(ocr_lines)
    prompt = f"""
The following text was produced by OCR from one email screenshot. Reconstruct the open email only.
Return ONLY the same JSON schema described below:
{{
  "is_email": true,
  "date_text": "",
  "sender_display": "",
  "sender_address": "",
  "sender_brand": "",
  "recipient_text": "",
  "salutation": "",
  "subject": "",
  "layout_type": "plain_text, designed_html, or unknown",
  "signature_contact": "",
  "content_blocks": [],
  "full_body_text": "",
  "primary_body": ""
}}

Apply these rules:
- Exclude app UI, toolbar text, status-bar text, buttons, labels, counts, and reply controls.
- For plain-text email, classify body, closing, signature, and signature contact as primary-flow content blocks.
- For richly designed email, classify detached cards, boxes, tables, side panels, payment panels, support panels, badges, buttons, and automated footers as separate blocks.
- Put the greeting only in salutation and exclude it from full_body_text and primary_body.
- full_body_text should contain all OCR-supported email-message content after the salutation in reading order.
- Preserve OCR-supported wording; do not invent missing sentences.
- Identity fields may contain only a person name, organization name, email address, or the literal UI value 'me'; never a heading, role, label, or sentence.
- Keep each email address and website as one uninterrupted identifier. Repair only whitespace that OCR inserted inside the identifier.

OCR TEXT:
---
{ocr_text[:50000]}
---
""".strip()
    data = extract_json_object(
        ollama_chat(model, [{"role": "user", "content": prompt}], num_predict=2048)
    )
    if not data:
        raise RuntimeError("OCR reconstruction model did not return valid JSON.")
    return data


def resolve_draft_with_report(
    draft: Dict[str, Any],
    report_text: str,
    model: str,
) -> Dict[str, Any]:
    """Canonicalize sender/receiver/date while keeping body screenshot-grounded."""
    excerpt = report_excerpt(report_text, draft)
    identity_draft = {
        key: draft.get(key, "")
        for key in (
            "date_text",
            "sender_display",
            "sender_address",
            "sender_brand",
            "recipient_text",
            "salutation",
            "subject",
        )
    }
    # Never send the full message body to the report-resolution call. Apart
    # from being unnecessary, long bodies can crowd the identity evidence out
    # of a 4K model context.
    draft_json = json.dumps(identity_draft, ensure_ascii=False, indent=2)
    prompt = f"""
Resolve identities and date for one screenshot-derived email record using a case report.
Return ONLY JSON with this exact schema:
{{
  "date_text": "",
  "sender": "",
  "receiver": ""
}}

Rules:
1. Resolve a UI recipient such as 'me', a surname-only salutation, or a partial name to the most strongly supported full person name in the report.
2. Prefer a visible human-readable sender organization/brand over its raw email address. Remove a department suffix only when the parent organization is unambiguous. Do not replace a visible organization with a person unless the screenshot clearly identifies that person as the sender.
3. date_text must represent the visible date. Do not invent a day or month. When the screenshot omits only the year, use the report timeline year supported by the case report.
4. When the report clearly supports the same person or organization, return the exact report spelling and capitalization, including minor OCR variants such as a missing/repeated letter.
5. If the report does not support a more specific identity, keep the visible screenshot value rather than guessing.
6. Do not return or reconstruct the email body.
7. sender and receiver must each be one exact person or organization name only. Never return a document heading, section title, offence label, role, department-only label, sentence, or combined identity. If a more specific supported name cannot be established, retain the visible person/organization name rather than inserting descriptive text.

SCREENSHOT DRAFT:
{draft_json}

RELEVANT CASE REPORT EXCERPT:
---
{excerpt}
---
""".strip()
    resolved = extract_json_object(
        ollama_chat(model, [{"role": "user", "content": prompt}], num_predict=256)
    )
    return resolved


def choose_sender(
    draft: Dict[str, Any],
    resolved: Dict[str, Any],
    report_text: str,
) -> str:
    visible = normalize_space(
        resolved.get("sender")
        or draft.get("sender_brand")
        or draft.get("sender_display")
        or draft.get("sender_address")
    )
    return canonicalize_sender_entity(visible, report_text)


def choose_receiver(
    draft: Dict[str, Any],
    resolved: Dict[str, Any],
    report_text: str,
) -> str:
    """Choose receiver from visible evidence, using the report only to canonicalize."""
    salutation_name = extract_salutation_identity(draft.get("salutation"))
    if salutation_name:
        canonical = canonicalize_person_from_report(salutation_name, report_text)
        if canonical:
            return canonical

    mailbox_fallback = ""
    for candidate in (resolved.get("receiver"), draft.get("recipient_text")):
        value = normalize_space(candidate)
        if not value or value.lower() in {"me", "to me", "myself"}:
            continue
        if "@" not in value:
            return canonicalize_person_from_report(value, report_text) or value
        if not mailbox_fallback:
            mailbox_fallback = value
    return mailbox_fallback


def _ordered_content_blocks(draft: Dict[str, Any]) -> List[Dict[str, Any]]:
    blocks = draft.get("content_blocks")
    if not isinstance(blocks, list):
        return []

    cleaned: List[Tuple[int, int, Dict[str, Any]]] = []
    for index, block in enumerate(blocks):
        if not isinstance(block, dict):
            continue
        text = str(block.get("text") or "").strip()
        if not text:
            continue
        try:
            order = int(block.get("order", index + 1))
        except (TypeError, ValueError):
            order = index + 1
        cleaned.append((order, index, block))
    cleaned.sort(key=lambda item: (item[0], item[1]))
    return [block for _, _, block in cleaned]


def _looks_like_detached_heading(line: str) -> bool:
    """Detect a short panel/card heading, usually rendered in capitals."""
    text = normalize_space(line).strip(" :-")
    words = text.split()
    if not (1 <= len(words) <= 8 and 3 <= len(text) <= 80):
        return False
    letters = [char for char in text if char.isalpha()]
    return bool(letters) and all(char.isupper() for char in letters)


def _block_is_detached(block: Dict[str, Any]) -> bool:
    kind = normalize_space(block.get("kind")).lower()
    flow = normalize_space(block.get("flow")).lower()
    return flow == "separate" or kind in {"callout", "table", "footer", "button"}


def _effective_layout(draft: Dict[str, Any]) -> str:
    """Resolve noisy model layout labels from all available structural evidence."""
    labelled = normalize_space(draft.get("layout_type")).lower()
    blocks = _ordered_content_blocks(draft)
    primary_body = str(draft.get("primary_body") or "")
    full_body = str(draft.get("full_body_text") or "")
    combined = "\n".join(value for value in (primary_body, full_body) if value)

    designed_score = 0
    plain_score = 0
    if labelled == "designed_html":
        designed_score += 2
    elif labelled == "plain_text":
        plain_score += 2

    separate_blocks = [block for block in blocks if _block_is_detached(block)]
    primary_signature_blocks = [
        block for block in blocks
        if not _block_is_detached(block)
        and normalize_space(block.get("kind")).lower()
        in {"closing", "signature", "contact"}
    ]
    designed_score += min(12, len(separate_blocks) * 4)
    plain_score += min(8, len(primary_signature_blocks) * 2)

    lines = [normalize_space(line) for line in combined.splitlines() if normalize_space(line)]
    detached_headings = sum(
        1 for index, line in enumerate(lines)
        if index > 0 and _looks_like_detached_heading(line)
    )
    designed_score += min(9, detached_headings * 3)

    if _contains_closing_or_signature(combined):
        plain_score += 4
    if re.search(r"(?i)\b(?:sincerely|regards)\b.*\b(?:department|consultants?|company|team)\b", normalize_space(combined)):
        plain_score += 3

    if len(separate_blocks) >= 2:
        designed_score += 5
    if not separate_blocks and _contains_closing_or_signature(combined):
        plain_score += 4

    if designed_score >= plain_score + 2:
        return "designed_html"
    if plain_score >= designed_score + 2:
        return "plain_text"
    if labelled in {"plain_text", "designed_html"}:
        return labelled
    return "unknown"


def _normalize_for_match(value: Any) -> str:
    text = _fold_identity_text(value)
    return re.sub(r"\s+", " ", text).strip()


def _text_similarity(left: Any, right: Any) -> float:
    a = _normalize_for_match(left)
    b = _normalize_for_match(right)
    if not a or not b:
        return 0.0
    if a in b or b in a:
        return min(len(a), len(b)) / max(len(a), len(b))
    return SequenceMatcher(None, a, b).ratio()


def _looks_like_structured_detached_paragraph(text: str) -> bool:
    """Generic fallback for a card/table start when block labels are noisy."""
    value = normalize_space(text).strip()
    if not value:
        return False
    if _looks_like_detached_heading(value):
        return True
    if re.match(
        r"(?i)^(?:action required|payment (?:instructions|details)|parcel information|"
        r"customer support|order details|transaction details|account details|"
        r"contact us|support|tracking number|beneficiary(?: name)?|iban|bank|"
        r"revolut tag|wire instructions|transfer instructions)\b",
        value,
    ):
        return True
    if re.match(r"(?i)^please\s+(?:transfer|pay|send|remit|wire)\b", value):
        return True
    if re.match(
        r"(?i)^(?:this is an automated (?:message|email)|"
        r"please do not reply(?: directly)? to this email)\b",
        value,
    ):
        return True
    label_count = len(re.findall(r"\b[A-Za-z][A-Za-z /&-]{1,28}:\s*", value))
    if label_count >= 2:
        return True
    if len(re.findall(r"(?:https?://|www\.|\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b)", value, flags=re.I)) >= 2:
        return True
    return False


def _split_paragraphs(raw: str) -> List[str]:
    value = str(raw or "").strip()
    if not value:
        return []
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", value) if part.strip()]
    if len(paragraphs) == 1 and "\n" in value:
        paragraphs = [line.strip() for line in value.splitlines() if line.strip()]
    return paragraphs


def _trim_designed_body_text(raw: str, draft: Dict[str, Any]) -> str:
    """Keep only the first continuous prose region of a rich HTML email."""
    paragraphs = _split_paragraphs(raw)
    if not paragraphs:
        return ""

    separate_texts = [
        str(block.get("text") or "")
        for block in _ordered_content_blocks(draft)
        if _block_is_detached(block) and normalize_space(block.get("text"))
    ]

    kept: List[str] = []
    for paragraph in paragraphs:
        normalized = normalize_space(paragraph)
        if kept:
            if any(_text_similarity(normalized, separate) >= 0.48 for separate in separate_texts):
                break
            if _looks_like_structured_detached_paragraph(normalized):
                break
        if re.match(
            r"(?i)^(?:this is an automated (?:message|email)|"
            r"please do not reply(?: directly)? to this email)\b",
            normalized,
        ):
            break
        kept.append(paragraph)

    return "\n\n".join(kept).strip()


def _compose_block_message(draft: Dict[str, Any], layout: str) -> str:
    """Compose every visible evidence block in reading order.

    ``content_blocks`` is already instructed to omit application chrome and
    mail headers.  Detached cards, tables, buttons, and footers can contain
    material evidence (amounts, destinations, contact details, or requested
    actions), so a forensic transcript must not discard them merely because
    the email uses a rich HTML layout.
    """
    del layout
    blocks = _ordered_content_blocks(draft)
    if not blocks:
        return ""

    included: List[str] = []
    for block in blocks:
        text = str(block.get("text") or "").strip()
        if not text:
            continue
        # Rich-email VLM passes sometimes repeat the same sentence in a body
        # block and again in a panel block.  Remove only near-exact adjacent
        # repetition; preserve distinct visible evidence.
        if included and _text_similarity(included[-1], text) >= 0.97:
            continue
        included.append(text)

    return "\n\n".join(included).strip()


def _contains_closing_or_signature(text: Any) -> bool:
    """Detect a visible email closing/signature without relying on one dataset."""
    value = _fold_identity_text(text)
    if not value:
        return False
    return bool(re.search(
        r"(?:^|\b)(?:sincerely|kind regards|best regards|warm regards|regards|"
        r"yours sincerely|yours faithfully|respectfully|thank you|thanks|"
        r"claims department|customer service|support team|clearance department)"
        r"(?:\b|$)",
        value,
    ))


def _body_candidate_score(text: str, layout: str) -> float:
    value = normalize_space(text)
    if not value:
        return float("-inf")

    score = min(len(value), 2400) / 20.0
    first = next((char for char in value if char.isalpha()), "")
    if first and first.isupper():
        score += 35
    elif first and first.islower():
        score -= 70

    if re.search(r"[:;,]\s*$", value):
        score -= 45
    if value.endswith((".", "!", "?")):
        score += 8

    if layout == "plain_text":
        if _contains_closing_or_signature(value):
            score += 110
        if re.search(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", value):
            score += 15
    elif layout == "designed_html":
        panel_hits = sum(
            1 for phrase in (
                "action required", "payment instructions", "parcel information",
                "customer support", "tracking number", "beneficiary name",
                "revolut tag", "this is an automated message",
            )
            if phrase in _fold_identity_text(value)
        )
        score -= panel_hits * 65
        score -= len(re.findall(r"(?:https?://|www\.|\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b)", value, flags=re.I)) * 25
        if len(value) > 1100:
            score -= (len(value) - 1100) / 8.0
    return score


def _select_message_body(draft: Dict[str, Any]) -> Tuple[str, str, Dict[str, str]]:
    """Select the best transcript source and return body, effective layout and candidates."""
    layout = _effective_layout(draft)
    primary = str(draft.get("primary_body") or "").strip()
    full = str(draft.get("full_body_text") or "").strip()
    blocks = _compose_block_message(draft, layout)

    candidates: Dict[str, str] = {
        "primary_body": primary,
        "full_body_text": full,
        "content_blocks": blocks,
    }
    if layout == "designed_html":
        # Prefer the ordered structural transcript because it retains visible
        # detached cards/tables/actions.  Fall back to the fullest direct body
        # candidate only when the structural pass produced nothing.
        candidates = {key: value for key, value in candidates.items() if value}
        if blocks:
            return blocks, layout, candidates
        if full:
            return full, layout, candidates
    else:
        candidates = {key: value for key, value in candidates.items() if value}

    if not candidates:
        return "", layout, {}

    _, best_value = max(
        candidates.items(),
        key=lambda item: (_body_candidate_score(item[1], layout), len(normalize_space(item[1]))),
    )
    return best_value, layout, candidates


def _canonicalize_salutation_text(salutation: Any, report_text: str) -> str:
    """Correct a visibly full addressee name without expanding surname-only greetings."""
    text = normalize_space(salutation)
    if not text:
        return ""
    match = re.match(
        r"(?i)^(?P<prefix>(?:dear|hello|hi|good\s+(?:morning|afternoon|evening))\s+)"
        r"(?P<name>.+?)(?P<punct>[,!:;]?)$",
        text,
    )
    if not match:
        return text

    visible_name = normalize_space(match.group("name"))
    untitled = re.sub(
        r"(?i)^(?:mr|mrs|ms|miss|dr|prof|professor)\.?\s+",
        "",
        visible_name,
    )
    if len(_name_tokens(untitled)) < 2:
        return text

    canonical = canonicalize_person_from_report(untitled, report_text)
    if not canonical:
        return text
    return f"{match.group('prefix')}{canonical}{match.group('punct')}"


def _split_leading_salutation(body: str) -> Tuple[str, str]:
    """Separate a greeting accidentally duplicated inside a body block."""
    normalized = normalize_space(body)
    match = re.match(
        r"(?i)^(?P<sal>(?:dear|hello|hi|good\s+(?:morning|afternoon|evening))\s+"
        r".+?[,!:;])\s*(?P<rest>.*)$",
        normalized,
    )
    if not match:
        return "", normalized
    return normalize_space(match.group("sal")), normalize_space(match.group("rest"))


def _opening_needs_repair(body: str) -> bool:
    value = normalize_space(body)
    if not value:
        return False
    first_alpha = next((char for char in value if char.isalpha()), "")
    if first_alpha and first_alpha.islower():
        return True
    first_word = (_fold_identity_text(value).split() or [""])[0]
    return first_word in {"to", "and", "but", "because", "which", "that", "of", "for"}


def refine_opening_with_vision(
    image_path: Path,
    model: str,
    salutation: str,
    body_hint: str,
) -> Dict[str, Any]:
    """Recover only a damaged opening; never rewrite the rest of the message."""
    prompt = f"""
Transcribe only the first one or two continuous body paragraphs immediately after the email salutation.
Return ONLY JSON: {{"opening_body": "exact visible text"}}.

Rules:
- Copy exact visible wording from the screenshot.
- Do not include the salutation, subject, sender/recipient headers, or app UI.
- Stop before any detached card, callout, table, payment panel, support panel, button, or footer.
- Do not summarize, paraphrase, correct, or complete text that is not visible.

VISIBLE SALUTATION HINT: {normalize_space(salutation)}
CURRENT DAMAGED BODY PREFIX: {normalize_space(body_hint)[:500]}
""".strip()
    return extract_json_object(
        ollama_chat(
            model,
            [{"role": "user", "content": prompt, "images": [str(image_path)]}],
            num_predict=420,
        )
    )


def _merge_opening_repair(body: str, opening: str) -> str:
    current = normalize_space(body)
    repaired = normalize_space(opening)
    if not current or not repaired:
        return current
    if not _opening_needs_repair(current):
        return current

    current_folded = current.casefold()
    repaired_folded = repaired.casefold()
    max_probe = min(len(current), 220)
    for length in range(max_probe, 19, -1):
        probe = current_folded[:length]
        position = repaired_folded.find(probe)
        if 0 < position <= 80:
            return normalize_space(repaired[:position] + current)

    if SequenceMatcher(None, repaired_folded[:300], current_folded[:300]).ratio() >= 0.72:
        return repaired
    return current


def _should_append_signature_contact(
    draft: Dict[str, Any],
    body: str,
    signature_contact: str,
    layout: str,
) -> bool:
    if not signature_contact:
        return False
    if _fold_identity_text(signature_contact) in _fold_identity_text(body):
        return False

    blocks = _ordered_content_blocks(draft)
    has_primary_signature_block = any(
        not _block_is_detached(block)
        and normalize_space(block.get("kind")).lower() in {"closing", "signature", "contact"}
        for block in blocks
    )
    if layout == "designed_html":
        return _contains_closing_or_signature(body)
    return _contains_closing_or_signature(body) or has_primary_signature_block


def _repair_split_evidence_identifiers(text: str) -> str:
    """Join OCR whitespace only inside unmistakable email/URL identifiers."""
    value = str(text or "")
    tlds = (
        "com|net|org|edu|gov|mil|int|io|ai|co|uk|gr|eu|de|fr|it|es|nl|"
        "be|ch|at|us|ca|au|info|biz|online|site|app|dev|tech|me|tv"
    )
    value = re.sub(
        rf"\b(?i:https)\s+(?:[:/\\|]+|I[lI]|l[I|])\s*"
        rf"(?=[A-Za-z0-9][A-Za-z0-9.-]*[-A-Za-z0-9]\s+(?:{tlds})\b)",
        "https://",
        value,
    )
    value = re.sub(
        rf"([\w.+-]+@[A-Za-z0-9.-]*[A-Za-z0-9-])\s*\.\s*({tlds})\b",
        r"\1.\2",
        value,
        flags=re.I,
    )
    scheme_host = r"((?:https?://|www\.)[A-Za-z0-9][A-Za-z0-9.-]*[A-Za-z0-9-])"
    value = re.sub(
        rf"{scheme_host}\s*\.\s*({tlds})\b",
        r"\1.\2",
        value,
        flags=re.I,
    )
    value = re.sub(
        rf"{scheme_host}\s+({tlds})\b",
        r"\1.\2",
        value,
        flags=re.I,
    )
    value = re.sub(
        rf"\b((?:website|web\s+site|site|url|visit|go\s+to)\s+)"
        rf"([A-Za-z0-9-]{{2,}}?)({tlds})\b",
        r"\1\2.\3",
        value,
        flags=re.I,
    )
    return value


def choose_message(
    draft: Dict[str, Any],
    resolved: Dict[str, Any],
    report_text: str,
    *,
    selected_body: str = "",
    effective_layout: str = "",
    opening_refined: Dict[str, Any] | None = None,
) -> str:
    """Compose screenshot-grounded message, preserving greeting and signature."""
    del resolved

    if not selected_body:
        selected_body, effective_layout, _ = _select_message_body(draft)
    layout = effective_layout or _effective_layout(draft)
    body = selected_body
    if opening_refined:
        body = _merge_opening_repair(body, opening_refined.get("opening_body", ""))

    salutation = _canonicalize_salutation_text(draft.get("salutation"), report_text)
    signature_contact = normalize_space(draft.get("signature_contact"))

    embedded_salutation, body = _split_leading_salutation(body)
    if not salutation:
        salutation = embedded_salutation

    parts: List[str] = []
    if salutation:
        parts.append(salutation)
    if body:
        parts.append(body)
    if _should_append_signature_contact(draft, body, signature_contact, layout):
        parts.append(signature_contact)

    return _repair_split_evidence_identifiers(normalize_space("\n\n".join(parts)))


def validate_record(record: Dict[str, str]) -> None:
    missing = [key for key in ("Time", "Sender", "Receiver", "Message") if not record.get(key)]
    if missing:
        raise RuntimeError(f"Could not reliably extract required field(s): {', '.join(missing)}")


def write_csv(output_path: Path, record: Dict[str, str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, quoting=csv.QUOTE_ALL, lineterminator="\n")
        writer.writerow(["Timestamp", "Estimated_Timestamp", "Sender", "Receiver", "Message"])
        writer.writerow([
            record["Time"],
            "True",
            record["Sender"],
            record["Receiver"],
            record["Message"],
        ])


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract one email screenshot to the common forensic CSV schema."
    )
    parser.add_argument("image", help="Email screenshot path.")
    parser.add_argument("case_report", help="Case report PDF/TXT path.")
    parser.add_argument("--model", default="gemma3:12b", help="Vision-capable Ollama model.")
    parser.add_argument("--langs", default="en", help="Comma-separated EasyOCR fallback languages.")
    parser.add_argument("--cpu", action="store_true", help="Use CPU for EasyOCR fallback.")
    parser.add_argument(
        "--no-vision",
        action="store_true",
        help="Disable direct image input and use EasyOCR plus the text model.",
    )
    parser.add_argument("--output", default=None, help="Output CSV path.")
    parser.add_argument("--debug-dir", default=None, help="Optional debug directory.")
    parser.add_argument("--dump-ocr", action="store_true")
    parser.add_argument("--dump-draft", action="store_true")

    # Compatibility with mass_extract.py's common extractor command.
    parser.add_argument("--emoji-mode", default="omit", choices=["omit", "vision"])
    parser.add_argument("--dump-side-map", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--conversation-state-cache", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--conversation-key", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--case-context-cache", default=None, help=argparse.SUPPRESS)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    image_path = Path(args.image).expanduser()
    report_path = Path(args.case_report).expanduser()

    if not image_path.is_file():
        print(f"[ERROR] Image not found: {image_path}", file=sys.stderr)
        return 2
    if not report_path.is_file():
        print(f"[ERROR] Case report not found: {report_path}", file=sys.stderr)
        return 2

    output_path = (
        Path(args.output).expanduser()
        if args.output
        else image_path.with_name(f"{image_path.stem}_email_extracted.csv")
    )
    debug_dir = Path(args.debug_dir).expanduser() if args.debug_dir else None

    try:
        report_text = read_report(str(report_path))
        report_year = extract_report_year(report_text)

        draft: Dict[str, Any]
        layout_refined: Dict[str, Any] = {}
        opening_refined: Dict[str, Any] = {}
        message_debug: Dict[str, Any] = {}
        ocr_lines: List[str] = []
        if args.no_vision:
            ocr_lines = easyocr_lines(
                image_path,
                [value.strip() for value in args.langs.split(",") if value.strip()] or ["en"],
                use_cpu=args.cpu,
            )
            draft = extract_draft_from_ocr(ocr_lines, args.model)
        else:
            try:
                draft = extract_draft_with_vision(image_path, args.model)
            except Exception as vision_exc:
                print(f"[WARN] Vision extraction failed; trying OCR fallback: {vision_exc}")
                ocr_lines = easyocr_lines(
                    image_path,
                    [value.strip() for value in args.langs.split(",") if value.strip()] or ["en"],
                    use_cpu=args.cpu,
                )
                draft = extract_draft_from_ocr(ocr_lines, args.model)

        if not bool(draft.get("is_email", True)):
            raise RuntimeError("The image was not recognized as an email screenshot.")

        if not args.no_vision and _needs_layout_refinement(draft):
            try:
                layout_refined = refine_layout_with_vision(image_path, args.model, draft)
                draft = merge_layout_refinement(draft, layout_refined)
            except Exception as layout_exc:
                print(f"[WARN] Layout refinement failed; using first-pass draft: {layout_exc}")

        resolved = resolve_draft_with_report(draft, report_text, args.model)
        selected_body, effective_layout, body_candidates = _select_message_body(draft)
        if not args.no_vision and _opening_needs_repair(selected_body):
            try:
                opening_refined = refine_opening_with_vision(
                    image_path,
                    args.model,
                    normalize_space(draft.get("salutation")),
                    selected_body,
                )
            except Exception as opening_exc:
                print(f"[WARN] Opening verification failed; keeping first-pass text: {opening_exc}")

        message_debug = {
            "effective_layout": effective_layout,
            "selected_body_before_opening_repair": selected_body,
            "body_candidates": body_candidates,
            "opening_refined": opening_refined,
        }
        normalized_email_date = normalize_date(
            resolved.get("date_text") or draft.get("date_text"), report_year,
        )
        record = {
            "Time": f"{normalized_email_date} 00:00:00" if normalized_email_date else "",
            "Sender": choose_sender(draft, resolved, report_text),
            "Receiver": choose_receiver(draft, resolved, report_text),
            "Message": choose_message(
                draft,
                resolved,
                report_text,
                selected_body=selected_body,
                effective_layout=effective_layout,
                opening_refined=opening_refined,
            ),
        }
        # Persist diagnostic data before validation so failed records remain
        # inspectable instead of leaving an empty debug directory.
        if debug_dir and (args.dump_draft or args.dump_ocr):
            debug_dir.mkdir(parents=True, exist_ok=True)
            if args.dump_draft:
                dump_json(debug_dir / "email_draft.json", draft)
                if layout_refined:
                    dump_json(debug_dir / "email_layout_refined.json", layout_refined)
                if opening_refined:
                    dump_json(debug_dir / "email_opening_refined.json", opening_refined)
                dump_json(debug_dir / "email_message_debug.json", message_debug)
                dump_json(debug_dir / "email_resolved.json", resolved)
                dump_json(debug_dir / "email_final.json", record)
            if args.dump_ocr and ocr_lines:
                (debug_dir / "email_ocr.txt").write_text("\n".join(ocr_lines), encoding="utf-8")

        validate_record(record)
        write_csv(output_path, record)

        print(f"[SUCCESS] CSV saved to: {output_path}")
        return 0
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
