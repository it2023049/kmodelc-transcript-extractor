"""Shared helper functions for forensic audio parsing, participant inference, and output generation."""

import csv
import json
import re
import shutil
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

AUDIO_EXTENSIONS = {
    ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma", ".mp4", ".mkv", ".mov"
}


@dataclass

# =============================================================================
# CORE AUDIO DATA MODELS
# =============================================================================
class SpeakerSegment:
    """One diarized transcript segment with optional timing, an anonymous speaker label, and

    verbatim text.
    """
    start: Optional[float]
    end: Optional[float]
    speaker: str
    text: str


@dataclass
class ConversationTurn:
    """One normalized communication turn containing sender, receiver, message, speaker label, and

    optional timing.
    """
    sender: str
    receiver: str
    message: str
    speaker: str
    start: Optional[float] = None
    end: Optional[float] = None



# =============================================================================
# TEXT AND IDENTITY NORMALIZATION
# =============================================================================
def normalize_space(text: str) -> str:
    """Normalize non-breaking spaces and repeated whitespace while preserving the original wording."""
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def strip_parenthetical_alias(value: str) -> str:
    """Remove complete or truncated trailing parenthetical aliases.

    Examples:
      ``Alex Example (A. Example)`` -> ``Alex Example``
      ``Alex Example (A. Example``  -> ``Alex Example``
    """
    value = normalize_space(str(value or ""))
    value = re.sub(r"\s*\([^)]*\)\s*$", "", value).strip()
    value = re.sub(r"\s*\([^)]*$", "", value).strip()
    return normalize_space(value)


def clean_message(text: str) -> str:
    """Normalize message spacing and punctuation without rewriting or semantically correcting the

    transcript.
    """
    text = normalize_space(text)
    text = re.sub(r"\s+([,.?!:;])", r"\1", text)
    text = re.sub(r"([([{])\s+", r"\1", text)
    text = re.sub(r"\s+([])}])", r"\1", text)
    return text.strip()


def safe_stem(path: Path) -> str:
    """Return a stable output stem after removing one recognized media extension."""
    name = path.name
    for ext in [".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma", ".mp4", ".mkv", ".mov"]:
        if name.lower().endswith(ext):
            return name[: -len(ext)]
    return path.stem



# =============================================================================
# INPUT DISCOVERY AND SAFE ARCHIVE HANDLING
# =============================================================================
def discover_audio_files(inputs: Sequence[str], work_dir: Path) -> List[Path]:
    """Resolve files, folders, globs, and zip archives into a de-duplicated ordered media list."""
    found: List[Path] = []
    temp_dir = work_dir / "_expanded_audio_inputs"

    for item in inputs:
        p = Path(item)
        matches: List[Path]
        if any(ch in item for ch in "*?["):
            matches = [Path(x) for x in sorted(map(str, Path().glob(item)))]
        else:
            matches = [p]

        for match in matches:
            if not match.exists():
                continue
            if match.is_dir():
                for child in sorted(match.rglob("*")):
                    if child.is_file() and child.suffix.lower() in AUDIO_EXTENSIONS:
                        found.append(child)
                continue
            if match.is_file() and match.suffix.lower() == ".zip":
                target = temp_dir / match.stem
                safe_extract_zip(match, target)
                for child in sorted(target.rglob("*")):
                    if child.is_file() and child.suffix.lower() in AUDIO_EXTENSIONS:
                        found.append(child)
                continue
            if match.is_file() and match.suffix.lower() in AUDIO_EXTENSIONS:
                found.append(match)

    deduped: List[Path] = []
    seen = set()
    for p in found:
        key = str(p.resolve()) if p.exists() else str(p)
        if key not in seen:
            seen.add(key)
            deduped.append(p)
    return deduped


def safe_extract_zip(zip_path: Path, target_dir: Path) -> None:
    """Extract an archive only after rejecting members that would escape the target directory."""
    target_dir.mkdir(parents=True, exist_ok=True)
    root = target_dir.resolve()
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            member_path = root / member.filename
            resolved = member_path.resolve()
            if root not in resolved.parents and resolved != root:
                raise ValueError(f"Unsafe zip path: {member.filename}")
        zf.extractall(root)



# =============================================================================
# CASE-REPORT AND PARTICIPANT PARSING
# =============================================================================
def read_case_report(path: Optional[str]) -> str:
    """Read a TXT or PDF case report into plain text for deterministic and LLM-assisted

    attribution.
    """
    if not path:
        return ""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Case report not found: {p}")
    if p.suffix.lower() == ".txt":
        return p.read_text(encoding="utf-8", errors="replace")
    if p.suffix.lower() == ".pdf":
        try:
            import PyPDF2
        except Exception as exc:
            raise RuntimeError("Reading PDF case reports requires PyPDF2. Install with: python -m pip install PyPDF2") from exc
        chunks: List[str] = []
        with p.open("rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                chunks.append(page.extract_text() or "")
        return "\n".join(chunks)
    return p.read_text(encoding="utf-8", errors="replace")


def parse_participants(value: Optional[str]) -> List[str]:
    """Parse a user-supplied participant string into clean, unique human names."""
    if not value:
        return []
    raw_parts = re.split(r"[,;\n]+", value)
    participants = []
    seen = set()
    for part in raw_parts:
        name = normalize_person_name(part)
        if name and name.casefold() not in seen:
            seen.add(name.casefold())
            participants.append(name)
    return participants


def normalize_person_name(name: str) -> str:
    """Remove structural noise from a person name without applying fuzzy or dataset-specific

    rewriting.
    """
    name = strip_parenthetical_alias(name)
    name = re.sub(r"^[\-•*\d.)\s]+", "", name)
    name = re.sub(r"\s+", " ", name)
    name = name.strip(" ,;:.()[]{}\"'")
    # Keep natural user-supplied casing when it already looks reasonable.
    if not name:
        return ""
    tokens = name.split()
    if len(tokens) > 6:
        return ""
    if any(len(t) == 1 and not t.endswith(".") for t in tokens):
        pass
    return name


def compact_identity(value: str) -> str:
    """Alphanumeric identity key used for conservative name/provenance matching."""
    return "".join(ch for ch in normalize_space(str(value or "")).casefold() if ch.isalnum())


def canonical_participant_name(value: str, participants: Sequence[str]) -> str:
    """Map a generated/name-like value to one exact allowed participant spelling."""
    key = compact_identity(value)
    if not key:
        return ""
    exact = {compact_identity(name): normalize_person_name(name) for name in participants if normalize_person_name(name)}
    if key in exact:
        return exact[key]
    candidates = [name for item_key, name in exact.items() if len(key) >= 5 and (key in item_key or item_key in key)]
    return candidates[0] if len(candidates) == 1 else ""


def match_participants_in_filename(audio_path: Optional[Path], participants: Sequence[str]) -> List[str]:
    """Return exact participant names explicitly encoded in an audio filename.

    This is provenance, not semantic inference. It supports spacing/underscore
    differences such as a concatenated full name versus its spaced form and ignores
    numeric staging prefixes added by batch processing.
    """
    if audio_path is None:
        return []
    name = audio_path.name
    # Strip repeated media extensions and batch/index boilerplate.
    for _ in range(3):
        stem, suffix = Path(name).stem, Path(name).suffix.lower()
        if suffix in AUDIO_EXTENSIONS:
            name = stem
        else:
            break
    normalized = compact_identity(re.sub(r"^\d+[_.\- ]+", "", name))
    if not normalized:
        return []
    matches: List[str] = []
    for participant in participants:
        key = compact_identity(participant)
        if key and key in normalized and participant.casefold() not in {x.casefold() for x in matches}:
            matches.append(normalize_person_name(participant))
    return matches


def self_identified_participant(text: str, participants: Sequence[str]) -> Optional[str]:
    """Return the exact participant explicitly self-identified by the speaker."""
    for participant in participants:
        if speaker_self_identifies_as(clean_message(text), participant):
            return normalize_person_name(participant)
    return None


def directly_addressed_participants(text: str, participants: Sequence[str]) -> List[str]:
    """Return one unambiguous directly addressed participant, if present.

    The full participant set is evaluated together so a first name is accepted
    only when unique.  This avoids the ambiguity introduced by testing each
    participant in isolation.
    """
    addressed = directly_addresses_participant(clean_message(text), participants)
    return [normalize_person_name(addressed)] if addressed else []


# =============================================================================
# REPORT-GROUNDED HUMAN ACTOR EXTRACTION
# =============================================================================
def extract_actor_candidates(case_report_text: str, max_candidates: int = 50) -> List[str]:
    """Extract likely human case participants from an operational report.

    This is intentionally role/label based. It avoids using every capitalized
    phrase in the report as a participant, because reports contain banks,
    services, countries, subject lines, companies, and evidence package names.
    The returned names are candidates only; per-audio participant selection is
    performed later from MOSS speaker text and generic cues.
    """
    text = case_report_text or ""
    candidates: List[str] = []
    seen = set()

    def add(value: str) -> None:
        for part in split_possible_names(value):
            name = normalize_person_name(part)
            if not looks_like_person_name(name):
                continue
            key = name.casefold()
            if key not in seen:
                seen.add(key)
                candidates.append(name)

    # High-confidence labels used in complaint reports and operational use cases.
    label_patterns = [
        r"(?im)^\s*(?:Full\s+Name|Target\s*/\s*Victim|Victim|Complainant|Witness)\s*[:\-]\s*(.+?)\s*$",
        r"(?im)^\s*Suspect\s*\d*\s*[:\-]\s*(.+?)\s*$",
        r"(?im)^\s*(?:Scammer\s+Names\s+Used|Scammer\s+Name|Suspect\s+Names\s+Used)\s*[:\-]\s*(.+?)\s*$",
        r"(?im)^\s*(?:Alias|Claimed\s+Identity|Name)\s*[:\-]\s*(.+?)\s*$",
    ]
    for pattern in label_patterns:
        for m in re.finditer(pattern, text):
            add(m.group(1))

    # PDF table extraction often puts a label on one line and the value on the next.
    lines = [normalize_space(line) for line in text.splitlines()]
    table_labels = {
        "full name", "complainant", "victim", "target/victim", "suspect", "suspect / alias",
        "reported identity", "alias", "scammer names used", "name"
    }
    for i, line in enumerate(lines[:-1]):
        key = line.strip(" :.-").casefold()
        if key in table_labels:
            for nxt in lines[i + 1 : min(i + 5, len(lines))]:
                if not nxt or nxt.casefold() in {"field", "value", "attribute", "details", "channels and contact", "details"}:
                    continue
                add(nxt)
                break

    # Common narrative forms: "using the name X", "persona X", "identifying himself as X".
    narrative_patterns = [
        r"(?is)\busing\s+the\s+name\s+[\"“']?([A-Z][A-Za-z'\-]+(?:\s+[A-Z][A-Za-z'\-]+){1,4})[\"”']?",
        r"(?is)\bpersona\s+[\"“']?([A-Z][A-Za-z'\-]+(?:\s+[A-Z][A-Za-z'\-]+){1,4})[\"”']?",
        r"(?is)\bidentif(?:ied|ying)\s+(?:himself|herself|themselves)?\s*as\s*[:\-]?\s*[\"“']?([A-Z][A-Za-z'\-]+(?:\s+[A-Z][A-Za-z'\-]+){1,4})[\"”']?",
        r"(?is)\bassociate\s+[\"“']?([A-Z][A-Za-z'\-]+(?:\s+[A-Z][A-Za-z'\-]+){1,4})[\"”']?",
        r"(?is)\bfrom\s+(?:a\s+male\s+individual\s+)?(?:named|called)\s+[\"“']?([A-Z][A-Za-z'\-]+(?:\s+[A-Z][A-Za-z'\-]+){1,4})[\"”']?",
    ]
    for pattern in narrative_patterns:
        for m in re.finditer(pattern, text):
            add(m.group(1))

    # Additional generic report forms seen in statements, timeline tables, and
    # communication summaries. These remain candidate-only and are validated by
    # later attribution rather than forced into a conversation.
    extra_patterns = [
        r"(?im)^\s*Statement\s+of\s+([A-Z][A-Za-z'\-]+(?:\s+[A-Z][A-Za-z'\-]+){1,4})",
        r"(?im)^\s*(?:Financial\s+Advisor|Advisor|Associate|Accomplice|Courier|Agent)\s*[:\-]\s*([A-Z][A-Za-z'\-]+(?:\s+[A-Z][A-Za-z'\-]+){1,4})",
        r"(?is)\bthis\s+is\s+([A-Z][A-Za-z'\-]+(?:\s+[A-Z][A-Za-z'\-]+){1,4})",
        r"(?is)\b(?:contacted|messaged|called|communicated\s+with)\s+(?:by\s+)?([A-Z][A-Za-z'\-]+(?:\s+[A-Z][A-Za-z'\-]+){1,4})",
    ]
    for pattern in extra_patterns:
        for m in re.finditer(pattern, text):
            add(m.group(1))

    return candidates[:max_candidates]



def extract_primary_report_party(case_report_text: str) -> Optional[str]:
    """Extract the primary reporting party from a victim/witness report.

    This is report-structure based, not dataset-name based. It helps two-party
    calls in victim statements where one external speaker self-identifies and
    the other speaker is the complainant/victim but does not say their full name.
    """
    text = case_report_text or ""
    candidates: List[str] = []

    patterns = [
        r"(?im)^\s*Statement\s+of\s+([A-Z][A-Za-z'\-]+(?:\s+[A-Z][A-Za-z'\-]+){1,4})(?:\s*\([^)]*\))?\s*$",
        r"(?im)^\s*(?:Full\s+Name|Victim\s*/\s*Complainant|Complainant|Victim)\s*[:\-]\s*([A-Z][A-Za-z'\-]+(?:\s+[A-Z][A-Za-z'\-]+){1,4})(?:\s*\([^)]*\))?\s*$",
        r"(?im)^\s*(?:Full\s+Name|Victim\s*/\s*Complainant|Complainant|Victim)\s*$\s*^\s*([A-Z][A-Za-z'\-]+(?:\s+[A-Z][A-Za-z'\-]+){1,4})(?:\s*\([^)]*\))?\s*$",
    ]
    for pattern in patterns:
        for m in re.finditer(pattern, text):
            for possible in split_possible_names(m.group(1)):
                name = normalize_person_name(possible)
                if looks_like_person_name(name):
                    candidates.append(name)

    # PDF table fallback: label line followed by value line.
    lines = [normalize_space(line) for line in text.splitlines()]
    primary_labels = {"full name", "complainant", "victim", "victim / complainant"}
    for i, line in enumerate(lines[:-1]):
        if line.strip(" :.-").casefold() not in primary_labels:
            continue
        for nxt in lines[i + 1 : min(i + 5, len(lines))]:
            if not nxt or nxt.casefold() in {"field", "value", "attribute", "details"}:
                continue
            for possible in split_possible_names(nxt):
                name = normalize_person_name(possible)
                if looks_like_person_name(name):
                    candidates.append(name)
                    break
            if candidates:
                break

    unique = unique_names(candidates)
    return unique[0] if unique else None

def split_possible_names(value: str) -> List[str]:
    """Split a report field that may contain several human names while removing trailing contact

    metadata.
    """
    value = normalize_space(value)
    value = re.sub(r"\([^)]*\)", " ", value)
    value = re.sub(r"\b(?:o|•|\-|Contact\s+Number|Viber|Email|Address)\b.*$", " ", value, flags=re.I)
    parts = re.split(r"\s*(?:,|;|/|\band\b|\&|\n)\s*", value)
    return [p for p in parts if p.strip()]


def looks_like_person_name(name: str) -> bool:
    """Reject organization, field-label, and evidence terms that should not become human

    participants.
    """
    if not name:
        return False
    blocked_words = {
        "case", "report", "email", "address", "phone", "number", "viber", "facebook",
        "payment", "transfer", "application", "software", "cryptocurrency", "investment",
        "platform", "account", "legal", "representative", "victims", "organisation",
        "organization", "country", "city", "customs", "required", "urgent", "subject",
        "sender", "receiver",
        "beneficiary", "service", "country", "method", "purpose", "reference",
        "claimed", "identity", "role", "details", "attribute", "field", "value",
    }
    tokens = [t.strip(".'-\"") for t in name.split()]
    if len(tokens) < 2 or len(tokens) > 5:
        return False
    if any(t.casefold() in blocked_words for t in tokens):
        return False
    # Require alphabetic-looking words, allowing initials such as "K." in the middle.
    alpha_tokens = [t for t in tokens if re.search(r"[A-Za-z]", t)]
    if len(alpha_tokens) < 2:
        return False
    return all(re.match(r"^[A-Za-z][A-Za-z'.\-]*$", t) for t in tokens)


# =============================================================================
# GENERIC TRANSCRIPTION HOTWORDS
# =============================================================================
def build_hotwords(case_report_text: str, participants: Sequence[str], max_items: int = 80) -> List[str]:
    """Build generic transcription hotwords from validated participants and report entities."""
    hotwords: List[str] = []
    seen = set()

    def add(value: str) -> None:
        value = normalize_space(value).strip(" ,;:.()[]{}\"'")
        if not value:
            return
        key = value.casefold()
        if key not in seen:
            seen.add(key)
            hotwords.append(value)

    for p in participants:
        add(p)

    # Keep this generic: named entities, reference-like codes, amounts, phone-like strings, email/domain tokens.
    text = case_report_text or ""
    for candidate in extract_actor_candidates(text, max_candidates=40):
        add(candidate)

    for m in re.finditer(r"\b[A-Z][A-Za-z0-9&.'\-]+(?:\s+[A-Z][A-Za-z0-9&.'\-]+){1,4}\b", text):
        add(m.group(0))
        if len(hotwords) >= max_items:
            break

    for m in re.finditer(r"\b(?:[A-Z]{2,}\d{2,}|\d{3,}[A-Z]{2,}|[A-Z]{2,}[-_/]\d{2,})\b", text):
        add(m.group(0))
        if len(hotwords) >= max_items:
            break

    for m in re.finditer(r"(?:\+?\d[\d\s().-]{6,}\d|[$€£]\s?\d[\d,]*(?:\.\d+)?)", text):
        add(m.group(0))
        if len(hotwords) >= max_items:
            break

    return hotwords[:max_items]


def name_to_regex(name: str) -> str:
    """Convert a normalized participant name into a whitespace-tolerant regular expression."""
    parts = [re.escape(p) for p in normalize_person_name(name).split()]
    return r"\s+".join(parts)



# =============================================================================
# DETERMINISTIC SPEAKER-IDENTITY CUES
# =============================================================================
def speaker_self_identifies_as(text: str, participant: str) -> bool:
    """Return whether an utterance explicitly identifies its own speaker as the supplied

    participant.
    """
    name_pat = name_to_regex(participant)
    patterns = [
        rf"(?i)\bthis\s+is\s+{name_pat}\b",
        rf"(?i)\bmy\s+name\s+is\s+{name_pat}\b",
        rf"(?i)\bi\s+am\s+{name_pat}\b",
        rf"(?i)\bi'm\s+{name_pat}\b",
        rf"(?i)\bim\s+{name_pat}\b",
    ]
    return any(re.search(p, text) for p in patterns)


def directly_addresses_participant(text: str, participants: Sequence[str]) -> Optional[str]:
    """Return one participant who is explicitly addressed in the utterance.

    Full names are always eligible.  A first name is used only when it is unique
    among the supplied participants, preventing two people with the same first
    name from being conflated.  The patterns are grammatical/vocative cues and
    contain no case-specific names.
    """
    cleaned = clean_message(text)
    canonical = unique_names(participants)
    first_counts: Dict[str, int] = {}
    for participant in canonical:
        first = participant.split()[0] if participant.split() else ""
        if first:
            first_counts[first.casefold()] = first_counts.get(first.casefold(), 0) + 1

    matches: List[str] = []
    for participant in canonical:
        first = participant.split()[0] if participant.split() else participant
        forms = [participant]
        if first and first_counts.get(first.casefold(), 0) == 1:
            forms.append(first)

        for name in forms:
            name_pat = name_to_regex(name)
            patterns = [
                rf"(?i)^\s*(?:hello|hi|hey|dear|good\s+(?:morning|afternoon|evening)),?\s+{name_pat}\b",
                rf"(?i)[,.!?]\s*{name_pat}\s*[,!?]",
                rf"(?i)\b{name_pat}\s*,\s+(?:please|can\s+you|could\s+you|you\s+need|you\s+will|do\s+you|are\s+you)\b",
                rf"(?i)\b(?:thank\s+you|thanks|i\s+love\s+you|love\s+you|miss\s+you|good\s+night|good\s+morning)\s*,?\s+{name_pat}\s*[.!?…]*$",
                rf"(?i)[,!]\s*{name_pat}\s*[.!?…]*$",
            ]
            if any(re.search(pattern, cleaned) for pattern in patterns):
                matches.append(participant)
                break

    unique = unique_names(matches)
    return unique[0] if len(unique) == 1 else None


# =============================================================================
# RECORDING-LEVEL PARTICIPANT SELECTION
# =============================================================================
def infer_call_participants(
    segments: Sequence[SpeakerSegment],
    actor_candidates: Sequence[str],
    audio_path: Optional[Path] = None,
    max_participants: int = 2,
    primary_participant: Optional[str] = None,
) -> List[str]:
    """Select likely participants for this audio from a neutral case report.

    The selection is based on generic evidence, in this order of reliability:
      1. speaker self-identification ("this is X", "my name is X"),
      2. direct address to another candidate ("hello X", ", X,"),
      3. filename hints,
      4. primary reporting party fallback for victim/witness statements,
      5. weak transcript mentions.

    Names that are merely mentioned as third parties (for example, "I spoke
    with X", "according to X", "X told me") are deliberately not allowed to
    outrank a directly addressed or primary reporting participant.
    """
    candidates = unique_names(actor_candidates)
    primary = normalize_person_name(primary_participant or "")
    if primary and looks_like_person_name(primary) and primary.casefold() not in {c.casefold() for c in candidates}:
        candidates.insert(0, primary)
    if not candidates or max_participants <= 0:
        return []

    scores: Dict[str, float] = {c: 0.0 for c in candidates}
    strong_scores: Dict[str, float] = {c: 0.0 for c in candidates}
    weak_scores: Dict[str, float] = {c: 0.0 for c in candidates}
    first_seen: Dict[str, int] = {}
    speaker_self_map: Dict[str, str] = {}
    speaker_address_map: Dict[str, str] = {}
    transcript = clean_message(" ".join(seg.text for seg in segments))
    file_text = normalize_space(str(audio_path.name if audio_path else "")).replace("_", " ").replace("-", " ")

    def add_score(candidate: str, amount: float, order: int, strong: bool) -> None:
        scores[candidate] = scores.get(candidate, 0.0) + amount
        if strong:
            strong_scores[candidate] = strong_scores.get(candidate, 0.0) + amount
        else:
            weak_scores[candidate] = weak_scores.get(candidate, 0.0) + amount
        first_seen.setdefault(candidate, order)

    for order, seg in enumerate(segments):
        text = clean_message(seg.text)
        if not text:
            continue
        for candidate in candidates:
            if speaker_self_identifies_as(text, candidate):
                add_score(candidate, 120.0, order, strong=True)
                speaker_self_map.setdefault(seg.speaker, candidate)
            addressed = directly_addresses_participant(text, [candidate])
            if addressed:
                add_score(candidate, 90.0, order, strong=True)
                speaker_address_map.setdefault(seg.speaker, candidate)
            if name_mentioned(text, candidate):
                # Mentions are weak because a current speaker may mention a third party.
                # Explicit third-party contexts receive no participant score.
                if not third_party_mention_context(text, candidate):
                    add_score(candidate, 2.0, order, strong=False)

    for candidate in candidates:
        if name_mentioned(file_text, candidate, allow_first_name=True):
            add_score(candidate, 50.0, -1, strong=True)
        elif name_mentioned(transcript, candidate) and not third_party_mention_context(transcript, candidate):
            add_score(candidate, 1.0, 9999, strong=False)

    speakers = ordered_speakers(segments)

    # Strong two-party pattern: one speaker self-identifies and directly addresses the other participant.
    if max_participants == 2 and len(speakers) == 2:
        for seg in segments:
            self_name = speaker_self_map.get(seg.speaker)
            if not self_name:
                continue
            addressed = directly_addresses_participant(seg.text, candidates)
            if addressed and addressed.casefold() != self_name.casefold():
                return unique_names([self_name, addressed])[:2]

    # Victim/witness statement fallback: if exactly one external speaker self-identifies and
    # there is another anonymous speaker, the other likely participant is the primary reporting party,
    # unless another candidate has strong direct evidence.
    self_identified = unique_names(list(speaker_self_map.values()))
    if max_participants == 2 and len(speakers) == 2 and len(self_identified) == 1 and primary:
        self_name = self_identified[0]
        if primary.casefold() != self_name.casefold():
            # Use this fallback only when the primary party has at least some evidence
            # (direct address/mention) or no other non-self candidate has strong evidence.
            other_strong = [
                c for c in candidates
                if c.casefold() != self_name.casefold()
                and c.casefold() != primary.casefold()
                and strong_scores.get(c, 0.0) >= 50.0
            ]
            primary_evidence = strong_scores.get(primary, 0.0) > 0 or name_mentioned(transcript, primary, allow_first_name=True)
            if primary_evidence or not other_strong:
                return unique_names([self_name, primary])[:2]

    # Prefer candidates with strong evidence; only use weak mention-only candidates when there
    # are not enough strong candidates and no primary-party fallback is available.
    strong_ranked = sorted(
        [c for c in candidates if strong_scores.get(c, 0.0) > 0],
        key=lambda c: (-strong_scores[c], first_seen.get(c, 99999), c.casefold()),
    )
    selected = unique_names(strong_ranked)[:max_participants]

    if len(selected) < max_participants and primary and primary.casefold() not in {s.casefold() for s in selected}:
        # In a victim statement, the complainant is a safer second participant than
        # a third party that is only mentioned in the transcript.
        selected = unique_names(selected + [primary])[:max_participants]

    if len(selected) < max_participants:
        weak_ranked = sorted(
            [c for c in candidates if scores.get(c, 0.0) > 0 and c.casefold() not in {s.casefold() for s in selected}],
            key=lambda c: (-scores[c], first_seen.get(c, 99999), c.casefold()),
        )
        selected = unique_names(selected + weak_ranked)[:max_participants]

    return selected



# =============================================================================
# SENDER AND RECEIVER EVIDENCE HELPERS
# =============================================================================
def third_party_mention_context(text: str, participant: str) -> bool:
    """Return True when a name is mentioned as someone outside the current exchange."""
    name_pat = name_to_regex(participant)
    patterns = [
        rf"(?i)\b(?:spoke|talked|communicated|checked|consulted)\s+with\s+{name_pat}\b",
        rf"(?i)\b(?:according\s+to|as\s+per|based\s+on)\s+{name_pat}\b",
        rf"(?i)\b{ name_pat }\s+(?:told|said|explained|informed|introduced|reported)\b",
        rf"(?i)\b(?:introduced|assigned|referred)\s+(?:me|you|us|her|him|them)\s+to\s+{name_pat}\b",
    ]
    return any(re.search(p, text) for p in patterns)


def participant_receiver_evidence(
    text: str,
    participant: str,
    participants: Sequence[str],
) -> Dict[str, Any]:
    """Describe neutral textual evidence that one person is the recipient.

    The helper is intentionally small and deterministic.  Direct address is
    positive evidence; a grammatical third-party mention is negative evidence;
    a plain name mention is weak and cannot establish a receiver by itself.
    """
    cleaned = clean_message(text)
    addressed = directly_addresses_participant(cleaned, participants)
    direct = bool(addressed and addressed.casefold() == normalize_person_name(participant).casefold())
    third_party = third_party_mention_context(cleaned, participant)
    mentioned = name_mentioned(cleaned, participant, allow_first_name=True)
    score = 0.0
    reasons: List[str] = []
    if direct:
        score += 100.0
        reasons.append("explicit direct address")
    if third_party and not direct:
        score -= 40.0
        reasons.append("third-party grammatical mention")
    elif mentioned and not direct:
        score += 2.0
        reasons.append("plain name mention")
    return {
        "participant": normalize_person_name(participant),
        "score": score,
        "direct_address": direct,
        "third_party_mention": third_party,
        "mentioned": mentioned,
        "reasons": reasons,
    }


def participant_sender_evidence(text: str, participant: str) -> Dict[str, Any]:
    """Describe deterministic self-identification evidence for one sender."""
    cleaned = clean_message(text)
    self_identified = speaker_self_identifies_as(cleaned, participant)
    return {
        "participant": normalize_person_name(participant),
        "score": 120.0 if self_identified else 0.0,
        "self_identified": self_identified,
        "reasons": ["explicit self-identification"] if self_identified else [],
    }


def unique_names(names: Sequence[str]) -> List[str]:
    """Normalize and de-duplicate names while preserving first-seen order."""
    out: List[str] = []
    seen = set()
    for name in names:
        norm = normalize_person_name(name)
        if not norm:
            continue
        key = norm.casefold()
        if key not in seen:
            seen.add(key)
            out.append(norm)
    return out


def name_mentioned(text: str, participant: str, allow_first_name: bool = False) -> bool:
    """Check for a full-name mention and optionally an explicit first-name mention."""
    text = text or ""
    full_pat = name_to_regex(participant)
    if re.search(rf"(?i)\b{full_pat}\b", text):
        return True
    if allow_first_name:
        first = participant.split()[0] if participant.split() else participant
        if first and re.search(rf"(?i)\b{re.escape(first)}\b", text):
            return True
    return False

def ordered_speakers(segments: Sequence[SpeakerSegment]) -> List[str]:
    """Return anonymous speaker labels in their first appearance order."""
    result: List[str] = []
    seen = set()
    for seg in segments:
        speaker = normalize_space(seg.speaker)
        if speaker and speaker not in seen:
            seen.add(speaker)
            result.append(speaker)
    return result



# =============================================================================
# TURN CONSTRUCTION AND RECEIVER FALLBACKS
# =============================================================================
def infer_receiver(sender: str, participants: Sequence[str], speaker_map: Dict[str, str], raw_speaker: str) -> str:
    """Infer the opposite participant only when a stable two-person mapping is available."""
    if len(participants) == 2:
        sender_cf = sender.casefold()
        for p in participants:
            if p.casefold() != sender_cf:
                return p
    # If participants are unknown but two speakers are mapped to people, infer the other mapped person.
    mapped = list(dict.fromkeys(speaker_map.values()))
    if len(mapped) == 2 and sender in mapped:
        return mapped[1] if mapped[0] == sender else mapped[0]
    return "Unknown"




# =============================================================================
# ANONYMOUS FIRST-STAGE TURN MAP
# =============================================================================
def build_anonymous_speaker_turns(
    segments: Sequence[SpeakerSegment],
    merge_same_speaker: bool = True,
) -> List[ConversationTurn]:
    """Collapse diarized speaker segments into anonymous speaker turns.

    This is the first-stage forensic artifact: it answers "when did the
    speaker change?" without trying to guess names. Later LLM attribution reads
    this artifact together with the case report.
    """
    turns: List[ConversationTurn] = []
    for seg in segments:
        message = clean_message(seg.text)
        if not message:
            continue
        sender = normalize_space(seg.speaker) or "UnknownSpeaker"
        if merge_same_speaker and turns and turns[-1].speaker == sender:
            turns[-1].message = clean_message(turns[-1].message + " " + message)
            if seg.end is not None:
                turns[-1].end = seg.end
            continue
        turns.append(
            ConversationTurn(
                sender=sender,
                receiver="Unknown",
                message=message,
                speaker=sender,
                start=seg.start,
                end=seg.end,
            )
        )
    return turns



# =============================================================================
# OUTPUT SERIALIZATION AND DEBUG ARTIFACTS
# =============================================================================
def write_speaker_turns_txt(path: Path, turns: Sequence[ConversationTurn]) -> None:
    """Write anonymous speaker-change map used as LLM input."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    for index, turn in enumerate(turns, start=1):
        stamp = ""
        if turn.start is not None and turn.end is not None:
            stamp = f"[{turn.start:.2f}-{turn.end:.2f}] "
        lines.append(f"TURN_{index:03d} | {stamp}{turn.speaker}: {clean_message(turn.message)}")
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def turns_to_llm_items(turns: Sequence[ConversationTurn]) -> List[Dict[str, Any]]:
    """Serialize anonymous turns into the minimal numbered structure used by LLM prompts and debug

    output.
    """
    items: List[Dict[str, Any]] = []
    for index, turn in enumerate(turns, start=1):
        items.append(
            {
                "turn_id": index,
                "speaker": turn.speaker,
                "message": clean_message(turn.message),
            }
        )
    return items

def parse_speaker_map_file(path: Optional[str]) -> Dict[str, str]:
    """Load and normalize a manual speaker-map JSON file."""
    if not path:
        return {}
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "speaker_map" in data and isinstance(data["speaker_map"], dict):
        data = data["speaker_map"]
    if not isinstance(data, dict):
        raise ValueError("speaker-map JSON must be an object, or contain a speaker_map object")
    return {normalize_space(str(k)): normalize_person_name(str(v)) for k, v in data.items() if str(k).strip() and str(v).strip()}


def write_diarized_txt(path: Path, segments: Sequence[SpeakerSegment], speaker_map: Dict[str, str]) -> None:
    """Write a readable diarized transcript using resolved names where available."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    for seg in segments:
        speaker = speaker_map.get(seg.speaker, seg.speaker)
        stamp = ""
        if seg.start is not None and seg.end is not None:
            stamp = f"[{seg.start:.2f}-{seg.end:.2f}] "
        lines.append(f"{stamp}{speaker}: {clean_message(seg.text)}")
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def csv_field(value: str, always_quote: bool = False) -> str:
    """Escape one CSV field using stable RFC-style quoting rules."""
    value = "" if value is None else str(value)
    must_quote = always_quote or any(ch in value for ch in [",", "\"", "\n", "\r"])
    value = value.replace('"', '""')
    return f'"{value}"' if must_quote else value


def write_conversation_csv(path: Path, turns: Sequence[ConversationTurn]) -> None:
    """Write stable forensic CSV while preserving real MOSS turn offsets."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["Offset_Seconds,Sender,Receiver,Message"]
    first_start = float(turns[0].start or 0.0) if turns else 0.0
    for turn in turns:
        sender = strip_parenthetical_alias(turn.sender) or "Unknown"
        receiver = strip_parenthetical_alias(turn.receiver) or "Unknown"
        # The first retained speaker turn is the 00:00:00 anchor. Every later
        # value is the real MOSS speaker-change offset relative to that anchor.
        offset = max(0.0, float(turn.start or 0.0) - first_start)
        lines.append(
            ",".join(
                [
                    csv_field(f"{offset:.3f}".rstrip("0").rstrip(".") or "0"),
                    csv_field(sender),
                    csv_field(receiver),
                    csv_field(turn.message, always_quote=True),
                ]
            )
        )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

def write_json(path: Path, data: Any) -> None:
    """Write UTF-8 JSON with readable indentation and preserved non-ASCII text."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def segments_to_dicts(segments: Sequence[SpeakerSegment]) -> List[Dict[str, Any]]:
    """Convert speaker-segment dataclasses into JSON-serializable dictionaries."""
    return [asdict(seg) for seg in segments]


def turns_to_dicts(turns: Sequence[ConversationTurn]) -> List[Dict[str, Any]]:
    """Convert conversation-turn dataclasses into JSON-serializable dictionaries."""
    return [asdict(turn) for turn in turns]


def speaker_map_template(segments: Sequence[SpeakerSegment], speaker_map: Dict[str, str], participants: Sequence[str]) -> Dict[str, Any]:
    """Build an editable speaker-map template covering every anonymous speaker in the recording."""
    speakers = ordered_speakers(segments)
    return {
        "speaker_map": {speaker: speaker_map.get(speaker, "Unknown") for speaker in speakers},
        "participants": list(participants),
        "note": "Edit speaker_map values if automatic mapping is uncertain, then rerun with --speaker-map this_file.json.",
    }


def copy_source_files_to_output(output_dir: Path, source_files: Sequence[Path]) -> None:
    """Copy selected source files into an output folder without overwriting a file with itself."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for src in source_files:
        if not src.exists():
            continue
        dst = output_dir / src.name
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
