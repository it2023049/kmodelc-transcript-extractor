"""Batch orchestrator for chat, email, and diarized audio evidence.

The script accepts a case report plus a ZIP/folder (or a self-contained package),
extracts the ZIP under the results directory, runs the existing chat screenshot
extractors, runs ``email_extract.py`` for email screenshots, runs
``audio_diarize.py`` for audio/video evidence, and writes one stable,
chronologically sorted CSV using the public schema:
Timestamp, Estimated_Timestamp, Sender, Receiver, Message.
"""

import argparse
from collections import Counter
import csv
from difflib import SequenceMatcher
import glob
import hashlib
import io
import json
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"
}
AUDIO_EXTENSIONS = {
    ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma",
    ".mp4", ".mkv", ".mov",
}
REPORT_EXTENSIONS = {".pdf", ".txt"}
IGNORED_DIR_NAMES = {
    "__macosx", ".git", ".svn", ".hg", "node_modules", "__pycache__",
}

# Output layout created by this script:
# results/
#   extracted/       -> extracted ZIP contents used during this run
#   <input>_merged.csv
#   per_image/       -> optional, created only with --keep-per-image or debug/dump flags
#   per_audio/       -> optional, created only with --keep-audio-output
PER_IMAGE_DIR_NAME = "per_image"
PER_AUDIO_DIR_NAME = "per_audio"
EXTRACTED_ZIPS_DIR_NAME = "extracted"
SHARED_UTILS_FILENAME = "extractor_utils.py"
DEFAULT_AUDIO_MODEL_ID = "OpenMOSS-Team/MOSS-Transcribe-Diarize"
AUDIO_DUPLICATE_ROW_SEQUENCE_THRESHOLD = 0.88
AUDIO_DUPLICATE_TOKEN_CONTAINMENT_THRESHOLD = 0.85
AUDIO_DUPLICATE_MIN_TOKENS = 8
# Inferred audio dates are contextual estimates, not intrinsic recording
# identifiers.  When two candidates received different inferred dates, require
# substantially stronger transcript agreement before treating one as covered
# by the other.
AUDIO_DUPLICATE_CROSS_DATE_ROW_SEQUENCE_THRESHOLD = 0.96
AUDIO_DUPLICATE_CROSS_DATE_TOKEN_CONTAINMENT_THRESHOLD = 0.95

@dataclass
class EvidenceSource:
    """Resolved evidence source after optional ZIP extraction."""
    original_path: Path
    root_path: Path
    source_type: str  # file, folder, zip
    extracted: bool = False

@dataclass
class InputPlan:
    """Resolved input plan for either package or explicit-report mode."""
    mode: str
    report_path: Path
    evidence_sources: List[EvidenceSource]
    images: List[Path]
    audio_files: List[Path]
    file_inventory: List[Dict[str, str]]
    run_stem: str

@dataclass
class AudioCandidate:
    """One diarized audio file and the rows proposed for the merged CSV."""
    rows: List[Dict[str, str]]
    record: Dict[str, str]
    source_audio: Path
    source_csv: Path
    order: int
    sha256: str = ""

# ============================================================
# PATH / INPUT HELPERS
# ============================================================
def is_image_file(path: Path) -> bool:
    """Checks whether a path is a supported image file."""
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS

def is_audio_file(path: Path) -> bool:
    """Checks whether a path is supported by audio_diarize.py."""
    return path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS

def is_report_file(path: Path) -> bool:
    """Checks whether a path can be used as a case report/overview."""
    return path.is_file() and path.suffix.lower() in REPORT_EXTENSIONS

def is_zip_file(path: Path) -> bool:
    """Checks whether a path is a ZIP archive."""
    return path.is_file() and path.suffix.lower() == ".zip"

def default_script_path(script_name: str) -> Path:
    """Returns the default path for an extractor script beside this file."""
    return Path(__file__).resolve().parent / script_name

def default_email_script_path() -> Path:
    """Returns ../email/email_extract.py for the packaged layout."""
    return Path(__file__).resolve().parents[1] / "email" / "email_extract.py"

def safe_output_stem(path: Path) -> str:
    """Creates a filesystem-safe filename stem for generated outputs."""
    stem = path.stem if path.suffix else path.name
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)
    stem = re.sub(r"_+", "_", stem).strip("_")
    return stem or "input"

def unique_output_stem(path: Path, used_stems: Set[str]) -> str:
    """Creates a collision-free output stem for one input image."""
    base = safe_output_stem(path)
    stem = base
    counter = 2

    while stem.lower() in used_stems:
        stem = f"{base}_{counter:03d}"
        counter += 1

    used_stems.add(stem.lower())
    return stem

def ensure_exists(path: Path, label: str) -> None:
    """Raises a clear error when a required file or folder is missing."""
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")

def ensure_extractor_utils_available(script_path: Path) -> None:
    """Checks that extractor_utils.py is beside a refactored extractor script."""
    utils_path = script_path.resolve().parent / SHARED_UTILS_FILENAME

    if not utils_path.exists():
        raise FileNotFoundError(
            f"{SHARED_UTILS_FILENAME} not found next to {script_path}. "
            "Keep extractor_utils.py in the same folder as facebook_extract.py and viber_extract.py."
        )

def resolve_output_path(path_value: Optional[str], base_dir: Path, default_name: str) -> Path:
    """Resolves an output path under a base directory unless it is absolute."""
    if path_value is None:
        return base_dir / default_name

    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return base_dir / path


def should_keep_chat_csvs(args: argparse.Namespace) -> bool:
    """Retain both chat checkpoints only after an explicit CLI request."""
    return bool(
        args.keep_chat_csvs
        or args.raw_chat_output is not None
        or args.polished_chat_output is not None
    )

def list_files_recursively(root: Path) -> List[Path]:
    """Returns all files under a folder, skipping common metadata/cache folders."""
    files: List[Path] = []

    if root.is_file():
        return [root]

    for path in sorted(root.rglob("*"), key=lambda p: str(p).lower()):
        if not path.is_file():
            continue
        if any(part.lower() in IGNORED_DIR_NAMES for part in path.parts):
            continue
        files.append(path)

    return files

def short_path_for_manifest(path: Path, root: Optional[Path] = None) -> str:
    """Returns a readable relative path when possible."""
    if root:
        try:
            return str(path.relative_to(root))
        except ValueError:
            pass
    return str(path)

# ============================================================
# ZIP HANDLING
# ============================================================
def stable_zip_extract_dir(zip_path: Path, extract_root: Path) -> Path:
    """Builds a stable extraction folder for one ZIP input."""
    try:
        stat = zip_path.stat()
        fingerprint_source = f"{zip_path.resolve()}:{stat.st_mtime_ns}:{stat.st_size}"
    except OSError:
        fingerprint_source = str(zip_path.resolve())

    digest = hashlib.sha1(fingerprint_source.encode("utf-8", errors="ignore")).hexdigest()[:10]
    return extract_root / f"{safe_output_stem(zip_path)}_{digest}"

def safe_extract_zip(zip_path: Path, extract_root: Path) -> Path:
    """Extracts a ZIP archive safely under extract_root and returns the folder."""
    ensure_exists(zip_path, "ZIP input")
    output_dir = stable_zip_extract_dir(zip_path, extract_root)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            member_name = member.filename
            if not member_name or member_name.endswith("/"):
                continue

            target_path = output_dir / member_name
            resolved_target = target_path.resolve()
            resolved_root = output_dir.resolve()

            # Prevent ZIP Slip path traversal.
            if resolved_root not in resolved_target.parents and resolved_target != resolved_root:
                print(f"[WARN] Skipping unsafe ZIP member: {member_name}")
                continue

            target_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, target_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)

    return output_dir

def resolve_evidence_source(raw_path: Path, extract_root: Path) -> EvidenceSource:
    """Resolves a file/folder/ZIP evidence source."""
    raw_path = raw_path.expanduser()
    ensure_exists(raw_path, "input")

    if is_zip_file(raw_path):
        extracted_dir = safe_extract_zip(raw_path, extract_root)
        return EvidenceSource(
            original_path=raw_path,
            root_path=extracted_dir,
            source_type="zip",
            extracted=True,
        )

    if raw_path.is_dir():
        return EvidenceSource(
            original_path=raw_path,
            root_path=raw_path,
            source_type="folder",
            extracted=False,
        )

    return EvidenceSource(
        original_path=raw_path,
        root_path=raw_path,
        source_type="file",
        extracted=False,
    )

def resolve_evidence_sources(raw_inputs: Sequence[str], extract_root: Path) -> List[EvidenceSource]:
    """Expands globs and resolves all evidence inputs."""
    sources: List[EvidenceSource] = []

    for raw in raw_inputs:
        expanded = glob.glob(raw)
        values = expanded if expanded else [raw]

        for value in values:
            path = Path(value)
            try:
                sources.append(resolve_evidence_source(path, extract_root))
            except FileNotFoundError:
                print(f"[WARN] Skipping missing input: {value}")

    return sources

# ============================================================
# PACKAGE SCANNING / REPORT DISCOVERY
# ============================================================
def report_candidate_score(path: Path, source_root: Path) -> int:
    """Scores likely case overview/report files inside a package."""
    if not is_report_file(path):
        return -10_000

    name = path.name.lower()
    full = str(path.relative_to(source_root)).lower() if source_root in path.parents or path == source_root else str(path).lower()

    score = 0

    # Strongly prefer the parent-level Cases Overview / initial allegation file.
    if re.search(r"cases?[_\s-]*overview", name):
        score += 100
    if "overview" in name:
        score += 60
    if "allegation" in name or "initial" in name:
        score += 45
    if "case" in name:
        score += 35
    if "report" in name:
        score += 30
    if "complaint" in name or "complainant" in name or "victim" in name:
        score += 25
    if "statement" in name:
        score += 15

    # De-prioritize obvious generated/output/debug files.
    noisy_terms = [
        "log", "ground_truth", "groundtruth", "accuracy", "result", "output",
        "debug", "extracted", "merged", "transcript",
    ]
    if any(term in name for term in noisy_terms):
        score -= 40

    # Prefer files closer to the package root.
    try:
        depth = len(path.relative_to(source_root).parts)
    except ValueError:
        depth = len(path.parts)
    score -= max(0, depth - 1) * 3

    if "evidence" in full:
        score -= 8
    if "structured" in full:
        score -= 10
    if "unstructured" in full:
        score -= 3

    # Prefer PDFs over TXT when names are otherwise similar.
    if path.suffix.lower() == ".pdf":
        score += 5

    return score

def discover_case_report(files: List[Path], source_root: Path) -> Optional[Path]:
    """Finds the most likely case report/overview inside a use-case package."""
    candidates = [p for p in files if is_report_file(p)]
    if not candidates:
        return None

    scored = sorted(
        ((report_candidate_score(path, source_root), path) for path in candidates),
        key=lambda item: (item[0], -len(item[1].parts), str(item[1]).lower()),
        reverse=True,
    )

    best_score, best_path = scored[0]
    if best_score <= -1000:
        return None
    return best_path

def scan_source_files(source: EvidenceSource) -> List[Path]:
    """Lists files contained in one evidence source."""
    if source.root_path.is_file():
        return [source.root_path]
    return list_files_recursively(source.root_path)

def build_file_inventory(
    sources: List[EvidenceSource],
    report_path: Optional[Path],
    images: List[Path],
    audio_files: List[Path],
) -> List[Dict[str, str]]:
    """Builds manifest records for all files found in inputs."""
    image_set = {str(p.resolve()) for p in images if p.exists()}
    audio_set = {str(p.resolve()) for p in audio_files if p.exists()}
    report_resolved = str(report_path.resolve()) if report_path and report_path.exists() else ""
    records: List[Dict[str, str]] = []

    for source in sources:
        for path in scan_source_files(source):
            resolved = str(path.resolve()) if path.exists() else str(path)
            if resolved == report_resolved:
                status = "case_report"
            elif resolved in image_set:
                status = "candidate_image"
            elif resolved in audio_set:
                status = "candidate_audio"
            elif is_report_file(path):
                status = "ignored_report_candidate"
            elif is_image_file(path):
                status = "candidate_image"
            elif is_audio_file(path):
                status = "candidate_audio"
            else:
                status = "ignored_unsupported"

            records.append({
                "source": str(source.original_path),
                "path": short_path_for_manifest(path, source.root_path if source.root_path.is_dir() else None),
                "absolute_path": str(path),
                "extension": path.suffix.lower(),
                "status": status,
            })

    return records

def collect_media_from_sources(
    sources: List[EvidenceSource],
) -> Tuple[List[Path], List[Path]]:
    """Collects supported image and audio/video files recursively."""
    images: List[Path] = []
    audio_files: List[Path] = []

    for source in sources:
        for path in scan_source_files(source):
            if is_image_file(path):
                images.append(path)
            elif is_audio_file(path):
                audio_files.append(path)

    def dedupe(paths: List[Path]) -> List[Path]:
        unique: Dict[str, Path] = {}
        for path in sorted(paths, key=lambda p: str(p).lower()):
            unique[str(path.resolve())] = path
        return list(unique.values())

    return dedupe(images), dedupe(audio_files)

def resolve_input_plan(args: argparse.Namespace, extract_root: Path) -> InputPlan:
    """Resolves CLI input into report path, image list, and manifest inventory."""
    first_path = Path(args.package_or_report).expanduser()

    # Mode 1: explicit --case-report, all positional paths are evidence inputs.
    if args.case_report:
        report_path = Path(args.case_report).expanduser()
        ensure_exists(report_path, "case report")
        raw_inputs = [args.package_or_report] + list(args.inputs)
        sources = resolve_evidence_sources(raw_inputs, extract_root)
        images, audio_files = collect_media_from_sources(sources)
        run_stem = safe_output_stem(sources[0].original_path if sources else report_path)
        inventory = build_file_inventory(sources, report_path, images, audio_files)
        return InputPlan(
            "explicit_report", report_path, sources, images, audio_files,
            inventory, run_stem,
        )

    # Mode 2: legacy command: first positional is report, remaining are evidence inputs.
    if args.inputs:
        report_path = first_path
        ensure_exists(report_path, "case report")
        if not is_report_file(report_path):
            raise ValueError(
                "When multiple positional arguments are used, the first one must be a PDF/TXT case report. "
                "For package auto-discovery, pass only the ZIP/folder or use --case-report."
            )
        sources = resolve_evidence_sources(args.inputs, extract_root)
        images, audio_files = collect_media_from_sources(sources)
        run_stem = safe_output_stem(sources[0].original_path if sources else report_path)
        inventory = build_file_inventory(sources, report_path, images, audio_files)
        return InputPlan(
            "explicit_report", report_path, sources, images, audio_files,
            inventory, run_stem,
        )

    # Mode 3: package auto-discovery: one ZIP/folder contains overview/report + evidence.
    package_source = resolve_evidence_source(first_path, extract_root)
    files = scan_source_files(package_source)
    report_path = discover_case_report(files, package_source.root_path if package_source.root_path.is_dir() else package_source.root_path.parent)

    if report_path is None:
        raise FileNotFoundError(
            "Could not auto-discover a case report/overview PDF or TXT in the input package. "
            "Use --case-report /path/to/report.pdf or the legacy form: "
            "python3 mass_extract.py case_report.pdf evidence.zip"
        )

    images, audio_files = collect_media_from_sources([package_source])

    inventory = build_file_inventory(
        [package_source], report_path, images, audio_files,
    )
    return InputPlan(
        mode="package_auto_report",
        report_path=report_path,
        evidence_sources=[package_source],
        images=images,
        audio_files=audio_files,
        file_inventory=inventory,
        run_stem=safe_output_stem(package_source.original_path),
    )

# ============================================================
# PLATFORM CLASSIFICATION
# ============================================================
def normalize_platform(value: str) -> str:
    """Normalizes platform aliases to facebook, viber, email, non_chat, or unknown."""
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")

    if text in {"facebook", "messenger", "facebook_messenger", "fb"}:
        return "facebook"
    if text == "viber":
        return "viber"
    if text in {"email", "e_mail", "mail", "webmail"}:
        return "email"
    if text in {
        "non_chat", "not_chat", "no_chat", "other", "irrelevant",
        "not_a_chat", "profile", "contact_profile", "account_profile",
        "non_conversation",
    }:
        return "non_chat"
    return "unknown"

def classify_by_filename(image_path: Path) -> str:
    """Classifies the platform using deterministic filename and folder hints."""
    text = str(image_path).lower().replace("\\", "/")

    facebook_keywords = [
        "facebook", "messenger", "fb_", "/fb/", "/facebook/", "/messenger/",
    ]
    viber_keywords = ["viber", "/viber/"]
    email_keywords = ["email", "e-mail", "mail_", "/mail/", "/email/", "/emails/"]

    if any(keyword in text for keyword in facebook_keywords):
        return "facebook"
    if any(keyword in text for keyword in viber_keywords):
        return "viber"
    if any(keyword in text for keyword in email_keywords):
        return "email"
    return "unknown"

def extract_json_object(text: str) -> Dict[str, str]:
    """Extracts and parses a JSON object from model text output."""
    text = str(text or "").strip()
    text = re.sub(r"^```(?:json)?", "", text, flags=re.I).strip()
    text = re.sub(r"```$", "", text).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start:end + 1]

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}

def classify_platform_with_vlm(image_path: Path, model: str) -> str:
    """Uses a vision-capable Ollama model to classify the image or skip non-chat images."""
    try:
        import ollama
    except Exception as exc:
        print(f"[WARN] Could not import ollama for VLM classification: {exc}")
        return "unknown"

    prompt = """
You are classifying an evidence image.

Look only at the image UI/content and return one of:
- facebook: a Facebook Messenger chat screenshot or Messenger chat collage
- viber: a Viber chat screenshot or Viber chat collage
- email: an email message screenshot, webmail view, or rendered email
- non_chat: not chat or email evidence
- unknown: could be communication evidence, but its type cannot be determined

Return only JSON with this exact schema:
{"platform":"facebook"}
or
{"platform":"viber"}
or
{"platform":"email"}
or
{"platform":"non_chat"}
or
{"platform":"unknown"}

Rules:
1. Use facebook only for Facebook Messenger / Messenger UI containing at least one actual human message bubble.
2. Use viber only for Viber UI containing at least one actual human message bubble.
3. Use email only when sender/recipient/subject/body or unmistakable email UI is visible.
4. A standalone contact/profile/account-information screen without visible message bubbles is non_chat, even when it uses Facebook or Viber branding.
5. Use non_chat for unrelated documents, tables, photos, scans, maps, logos, payment confirmations, and other non-conversation evidence.
6. If unsure whether it is communication evidence, return unknown.
7. Do not explain.
"""

    try:
        response = ollama.chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [str(image_path)],
                }
            ],
            options={"temperature": 0},
        )

        text = response["message"]["content"].strip()
        data = extract_json_object(text)
        platform = normalize_platform(data.get("platform", ""))

        if platform in {"viber", "facebook", "email", "non_chat"}:
            return platform

        # Small fallback if model returned text instead of JSON.
        low = text.lower()
        if "non_chat" in low or "not chat" in low or "not a chat" in low:
            return "non_chat"
        if "viber" in low:
            return "viber"
        if "email" in low or "e-mail" in low or "webmail" in low:
            return "email"
        if "facebook" in low or "messenger" in low:
            return "facebook"
        return "unknown"

    except Exception as exc:
        print(f"[WARN] VLM classification failed for {image_path.name}: {exc}")
        return "unknown"

def classify_platform(
    image_path: Path,
    model: str,
    mode: str,
    force_platform: str = "auto",
) -> str:
    """Applies the selected platform classification strategy."""
    forced = normalize_platform(force_platform)
    if forced in {"facebook", "viber", "email"}:
        return forced

    if mode == "filename":
        return classify_by_filename(image_path)

    if mode == "vision":
        platform = classify_platform_with_vlm(image_path, model)
        if platform == "unknown":
            platform = classify_by_filename(image_path)
        return platform

    # Default: auto. Prefer a deterministic platform hint from the filename or
    # directory whenever one is available. This prevents a valid chat image
    # from being dropped when the VLM incorrectly labels it as ``non_chat``.
    # Profile/contact screenshots are still handled safely by the zero-row
    # verification performed after the selected chat extractor runs.
    filename_platform = classify_by_filename(image_path)
    if filename_platform in {"facebook", "viber", "email"}:
        print(
            "-> [CLASSIFY] Using deterministic filename/path hint: "
            f"{filename_platform}"
        )
        return filename_platform

    # With no reliable path hint, inspect the actual image. In auto mode the
    # VLM result is authoritative because there is no deterministic fallback.
    return classify_platform_with_vlm(image_path, model)

def classify_zero_row_image(image_path: Path, model: str) -> str:
    """Distinguish a real empty/failed chat from a non-conversation profile UI."""
    try:
        import ollama
    except Exception as exc:
        print(f"[WARN] Could not import ollama for zero-row verification: {exc}")
        return "unknown"

    prompt = """
Inspect this evidence image after a chat extractor produced zero message rows.
Return JSON only: {"kind":"conversation|profile|email|other|unknown"}.

Definitions:
- conversation: at least one actual human message bubble/message is visibly present.
- profile: standalone contact, user profile, account details, avatar, or contact-information screen with no visible message bubbles.
- email: an open email message with sender/recipient/body evidence.
- other: non-conversation evidence such as a payment confirmation, document, photo, table, or settings screen.

Do not infer from filenames. Judge only the visible image. Do not explain.
"""
    try:
        response = ollama.chat(
            model=model,
            messages=[{
                "role": "user",
                "content": prompt,
                "images": [str(image_path)],
            }],
            options={"temperature": 0},
        )
        data = extract_json_object(response["message"]["content"])
        kind = str(data.get("kind", "unknown")).strip().casefold()
        return kind if kind in {"conversation", "profile", "email", "other"} else "unknown"
    except Exception as exc:
        print(f"[WARN] Zero-row image verification failed for {image_path.name}: {exc}")
        return "unknown"

# ============================================================
# EXTRACTOR RUNNING
# ============================================================

def build_extractor_command(
    script_path: Path,
    image_path: Path,
    report_path: Path,
    model: str,
    langs: str,
    use_cpu: bool,
    no_vision: bool,
    emoji_mode: str,
    dump_ocr: bool,
    dump_draft: bool,
    dump_side_map: bool,
    output_csv_path: Path,
    debug_dir_path: Path,
    conversation_state_cache: Path,
    conversation_key: str,
    extra_args: List[str],
) -> List[str]:
    """Builds the subprocess command for one platform extractor run."""
    cmd = [
        sys.executable,
        "-u",
        str(script_path),
        str(image_path),
        str(report_path),
        "--model", model,
        "--langs", langs,
        "--emoji-mode", emoji_mode,
        "--output", str(output_csv_path),
        "--debug-dir", str(debug_dir_path),
        # Batch-only continuity state.  The child extractor uses this as a soft
        # prior for later screenshots in the same evidence folder.
        "--conversation-state-cache", str(conversation_state_cache),
        "--conversation-key", conversation_key,
    ]

    if use_cpu:
        cmd.append("--cpu")
    if no_vision:
        cmd.append("--no-vision")
    if dump_ocr:
        cmd.append("--dump-ocr")
    if dump_draft:
        cmd.append("--dump-draft")
    if dump_side_map:
        cmd.append("--dump-side-map")

    cmd.extend(extra_args)
    return cmd

def run_command_streaming(cmd: Sequence[str]) -> Tuple[int, str]:
    """Runs a command while teeing combined output to the current job log."""
    collected: List[str] = []
    process = subprocess.Popen(
        list(cmd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        collected.append(line)
    return process.wait(), "".join(collected)

def run_extractor(
    platform: str,
    image_path: Path,
    report_path: Path,
    viber_script: Path,
    facebook_script: Path,
    email_script: Path,
    model: str,
    langs: str,
    use_cpu: bool,
    no_vision: bool,
    emoji_mode: str,
    dump_ocr: bool,
    dump_draft: bool,
    dump_side_map: bool,
    output_csv_path: Path,
    debug_dir_path: Path,
    conversation_state_cache: Path,
    conversation_key: str,
    extra_args: List[str],
    keep_debug: bool,
) -> Optional[Path]:
    """Runs the selected extractor and returns the produced CSV path."""
    if platform == "viber":
        script_path = viber_script
    elif platform == "facebook":
        script_path = facebook_script
    elif platform == "email":
        script_path = email_script
    else:
        print(f"[WARN] Unknown/non-chat platform for {image_path.name}; skipping.")
        return None

    ensure_exists(script_path, f"{platform} extractor script")

    if platform == "email":
        cmd = [
            sys.executable, "-u", str(script_path), str(image_path),
            str(report_path), "--model", model, "--langs", langs,
            "--output", str(output_csv_path), "--debug-dir", str(debug_dir_path),
        ]
        if use_cpu:
            cmd.append("--cpu")
        if no_vision:
            cmd.append("--no-vision")
        if dump_ocr:
            cmd.append("--dump-ocr")
        if dump_draft:
            cmd.append("--dump-draft")
        cmd.extend(extra_args)
    else:
        cmd = build_extractor_command(
            script_path=script_path,
            image_path=image_path,
            report_path=report_path,
            model=model,
            langs=langs,
            use_cpu=use_cpu,
            no_vision=no_vision,
            emoji_mode=emoji_mode,
            dump_ocr=dump_ocr,
            dump_draft=dump_draft,
            dump_side_map=dump_side_map,
            output_csv_path=output_csv_path,
            debug_dir_path=debug_dir_path,
            conversation_state_cache=conversation_state_cache,
            conversation_key=conversation_key,
            extra_args=extra_args,
        )

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    debug_dir_path.mkdir(parents=True, exist_ok=True)

    print(f"\n[RUN] {platform.upper()} extractor for: {image_path}")
    print("[CMD]", " ".join(quote_for_log(x) for x in cmd))

    returncode, output = run_command_streaming(cmd)

    if not keep_debug and debug_dir_path.exists():
        shutil.rmtree(debug_dir_path, ignore_errors=True)

    if returncode != 0:
        print(f"[ERROR] Extractor failed for {image_path.name} with exit code {returncode}")
        return None

    if output_csv_path.exists():
        return output_csv_path

    csv_path = parse_success_csv_path(output)
    if csv_path and csv_path.exists():
        return csv_path

    print(f"[WARN] Could not locate CSV output for {image_path.name}")
    print(f"[WARN] Expected CSV at: {output_csv_path}")
    return None

def quote_for_log(value: str) -> str:
    """Quotes command arguments only when logging needs it for readability."""
    if re.search(r"\s", value):
        return f'"{value}"'
    return value

def parse_success_csv_path(output: str) -> Optional[Path]:
    """Finds a CSV output path mentioned in an extractor success log."""
    patterns = [
        r"\[SUCCESS\]\s*CSV saved to:\s*(.+)",
        r"CSV saved to:\s*(.+)",
        r"Saved to:\s*(.+\.csv)",
    ]

    for pattern in patterns:
        m = re.search(pattern, output)
        if not m:
            continue
        raw = m.group(1).strip().strip('"').strip("'")
        path = Path(raw)
        if path.suffix.lower() == ".csv":
            return path
    return None

# ============================================================
# AUDIO DIARIZATION
# ============================================================
def default_audio_script_path() -> Path:
    """Returns scripts/speech/audio_diarize.py for the packaged layout."""
    return Path(__file__).resolve().parents[1] / "speech" / "audio_diarize.py"

def ensure_audio_script_available(script_path: Path) -> None:
    """Checks the diarizer and its colocated audio_utils.py dependency."""
    ensure_exists(script_path, "audio diarization script")
    utils_path = script_path.resolve().parent / "audio_utils.py"
    ensure_exists(utils_path, "audio_utils.py")

def stage_audio_inputs(
    audio_files: Sequence[Path],
    stage_dir: Path,
) -> Dict[str, Path]:
    """Stages uniquely named links so one diarizer process can load the model once."""
    stage_dir.mkdir(parents=True, exist_ok=True)
    source_by_stem: Dict[str, Path] = {}

    for index, source in enumerate(audio_files, start=1):
        stem = f"{index:04d}_{safe_output_stem(source)}"
        staged = stage_dir / f"{stem}{source.suffix.lower()}"
        try:
            staged.symlink_to(source.resolve())
        except OSError:
            # Symlinks may be disabled on some filesystems. Copying is a safe
            # fallback, although it consumes additional temporary space.
            shutil.copy2(source, staged)
        source_by_stem[stem] = source

    return source_by_stem

def build_audio_command(
    script_path: Path,
    staged_inputs_dir: Path,
    report_path: Path,
    output_dir: Path,
    case_context_cache: Path,
    args: argparse.Namespace,
) -> List[str]:
    """Builds the audio_diarize.py command using mass_extract CLI settings."""
    cmd = [
        sys.executable,
        "-u",
        str(script_path),
        str(staged_inputs_dir),
        "--case-report", str(report_path),
        "--output-dir", str(output_dir),
        "--case-context-cache", str(case_context_cache),
        "--backend", args.audio_backend,
        "--model-id", args.audio_model_id,
        "--language", args.audio_language,
        "--device", args.audio_device,
        "--dtype", args.audio_dtype,
        "--max-new-tokens", str(args.audio_max_new_tokens),
        "--max-hotwords", str(args.audio_max_hotwords),
        "--api-url", args.audio_api_url,
        "--api-timeout", str(args.audio_api_timeout),
        "--llm-backend", args.audio_llm_backend,
        "--ollama-model", args.audio_ollama_model or args.model,
        "--llm-timeout", str(args.audio_llm_timeout),
        "--llm-temperature", str(args.audio_llm_temperature),
        "--max-case-report-chars", str(args.audio_max_case_report_chars),
        "--max-transcript-chars", str(args.audio_max_transcript_chars),
    ]

    if args.audio_ollama_host:
        cmd.extend(["--ollama-host", args.audio_ollama_host])
    if args.audio_hotwords:
        cmd.extend(["--hotwords", args.audio_hotwords])
    if args.audio_prompt:
        cmd.extend(["--prompt", args.audio_prompt])
    if args.audio_speaker_map:
        cmd.extend(["--speaker-map", args.audio_speaker_map])
    if args.audio_participants:
        cmd.extend(["--participants", args.audio_participants])
    if args.audio_map_speakers_by_order:
        cmd.append("--map-speakers-by-order")
    if args.audio_no_merge_turns:
        cmd.append("--no-merge-turns")
    if args.audio_debug_json:
        cmd.append("--debug-json")

    cmd.extend(args.extra_audio_arg)
    return cmd

def run_audio_diarizer(
    audio_files: Sequence[Path],
    report_path: Path,
    audio_script: Path,
    output_dir: Path,
    staging_dir: Path,
    case_context_cache: Path,
    args: argparse.Namespace,
) -> Tuple[List[Path], List[Dict[str, str]], int]:
    """Runs one diarizer process and returns generated CSVs and item records."""
    source_by_stem = stage_audio_inputs(audio_files, staging_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_audio_command(
        script_path=audio_script,
        staged_inputs_dir=staging_dir,
        report_path=report_path,
        output_dir=output_dir,
        case_context_cache=case_context_cache,
        args=args,
    )

    print(f"\n[RUN] AUDIO diarizer for {len(audio_files)} file(s)")
    print("[CMD]", " ".join(quote_for_log(x) for x in cmd))
    returncode, _ = run_command_streaming(cmd)

    csv_paths = [
        output_dir / f"{staged_stem}.diarized.csv"
        for staged_stem in source_by_stem
        if (output_dir / f"{staged_stem}.diarized.csv").exists()
    ]
    csv_by_stem = {
        path.name[: -len(".diarized.csv")]: path
        for path in csv_paths
    }
    records: List[Dict[str, str]] = []

    for staged_stem, source in source_by_stem.items():
        csv_path = csv_by_stem.get(staged_stem)
        records.append({
            "audio": str(source),
            "csv": str(csv_path) if csv_path and args.keep_audio_output else "",
            "status": "ok" if csv_path else "failed",
            "reason": "processed" if csv_path else "diarized CSV missing",
            "rows": "0",
        })

    if returncode != 0:
        print(
            f"[WARN] audio_diarize.py exited with code {returncode}; "
            "successfully produced CSV files will still be merged."
        )

    return csv_paths, records, returncode

# ============================================================
# CSV MERGING / SORTING
# ============================================================
def normalize_estimated_timestamp(value: str) -> str:
    """Normalize truthy/falsy CSV values to canonical ``True``/``False`` text."""
    return "True" if str(value or "").strip().casefold() in {"true", "1", "yes", "y"} else "False"

def read_chat_csv(csv_path: Path, source_image: Optional[Path] = None) -> List[Dict[str, str]]:
    """Read chat or email CSV rows into the common public schema."""
    rows: List[Dict[str, str]] = []
    text = read_text_flexible(csv_path)
    reader = csv.reader(io.StringIO(text))

    try:
        header = next(reader)
    except StopIteration:
        print(f"[WARN] Empty CSV, skipping: {csv_path}")
        return rows

    normalized_header = [str(col or "").lstrip("\ufeff").strip() for col in header]
    timestamp_column = (
        "Timestamp" if "Timestamp" in normalized_header
        else "Time" if "Time" in normalized_header
        else ""
    )
    estimated_column = "Estimated_Timestamp" if "Estimated_Timestamp" in normalized_header else ""
    required_columns = ["Sender", "Receiver", "Message"]
    missing = [col for col in required_columns if col not in normalized_header]

    if not timestamp_column:
        missing.insert(0, "Timestamp")
    if missing:
        print(f"[WARN] CSV does not have expected columns, skipping: {csv_path}")
        print(f"[WARN] Header found: {normalized_header}")
        return rows

    indexes = {col: normalized_header.index(col) for col in required_columns}
    timestamp_index = normalized_header.index(timestamp_column)

    for index, row in enumerate(reader, start=2):
        if not row:
            continue

        def get_cell(col: str) -> str:
            i = indexes[col]
            return str(row[i]).strip() if i < len(row) else ""

        item = {
            "Timestamp": str(row[timestamp_index]).strip() if timestamp_index < len(row) else "",
            "Estimated_Timestamp": (
                normalize_estimated_timestamp(row[normalized_header.index(estimated_column)])
                if estimated_column and normalized_header.index(estimated_column) < len(row)
                else "False"
            ),
            "Sender": get_cell("Sender"),
            "Receiver": get_cell("Receiver"),
            "Message": get_cell("Message"),
            "_source_csv": str(csv_path),
            "_source_image": str(source_image or ""),
            "_source_kind": "chat",
            "_source_row": str(index),
        }

        if item["Timestamp"] and item["Sender"] and item["Receiver"] and item["Message"]:
            rows.append(item)

    return rows

def read_audio_csv(
    csv_path: Path,
    source_audio: Optional[Path] = None,
    base_date: Optional[datetime] = None,
    include_undated: bool = False,
) -> List[Dict[str, str]]:
    """Read diarizer rows and combine their real MOSS offsets with a case date."""
    rows: List[Dict[str, str]] = []
    text = read_text_flexible(csv_path)
    reader = csv.reader(io.StringIO(text))

    try:
        header = next(reader)
    except StopIteration:
        print(f"[WARN] Empty audio CSV, skipping: {csv_path}")
        return rows

    normalized_header = [
        str(column or "").lstrip("\ufeff").strip()
        for column in header
    ]
    required_columns = ["Offset_Seconds", "Sender", "Receiver", "Message"]
    missing = [column for column in required_columns if column not in normalized_header]
    if missing:
        print(f"[WARN] Audio CSV does not have expected columns, skipping: {csv_path}")
        print(f"[WARN] Header found: {normalized_header}")
        return rows

    indexes = {column: normalized_header.index(column) for column in required_columns}
    for index, row in enumerate(reader, start=2):
        if not row:
            continue

        def get_cell(column: str) -> str:
            position = indexes[column]
            return str(row[position]).strip() if position < len(row) else ""

        try:
            offset_seconds = max(0.0, float(get_cell("Offset_Seconds") or 0))
        except ValueError:
            print(f"[WARN] Invalid audio offset at {csv_path}:{index}; row skipped")
            continue

        item = {
            "Timestamp": (
                (base_date + timedelta(seconds=offset_seconds)).strftime("%d/%m/%Y %H:%M:%S")
                if base_date else ""
            ),
            "Estimated_Timestamp": "True",
            "Sender": get_cell("Sender"),
            "Receiver": get_cell("Receiver"),
            "Message": get_cell("Message"),
            "_source_csv": str(csv_path),
            "_source_audio": str(source_audio or ""),
            "_source_kind": "audio",
            "_source_row": str(index),
            "_offset_seconds": str(offset_seconds),
        }
        if (
            (item["Timestamp"] or include_undated)
            and item["Sender"] and item["Receiver"] and item["Message"]
        ):
            rows.append(item)

    return rows

def read_text_flexible(path: Path) -> str:
    """Reads text using common encodings, including UTF-8 with BOM."""
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(errors="ignore")

def read_case_report_text(path: Path) -> str:
    """Read TXT/PDF report text without using it to invent evidence content."""
    if path.suffix.lower() == ".txt":
        return read_text_flexible(path)
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        from pypdf import PdfReader
    return "\n".join((page.extract_text() or "") for page in PdfReader(str(path)).pages)

def resolve_audio_base_date(
    rows: Sequence[Dict[str, str]],
    report_path: Path,
    explicit_date: Optional[str] = None,
) -> Optional[datetime]:
    """Resolve a date for audio offsets, preferring observed chat evidence."""
    if explicit_date:
        try:
            return datetime.strptime(explicit_date.strip(), "%d/%m/%Y")
        except ValueError as exc:
            raise ValueError("--audio-date must use DD/MM/YYYY") from exc

    counts: Dict[str, int] = {}
    order: List[str] = []
    for row in rows:
        rank, parsed = parse_chat_datetime(row.get("Timestamp", ""))
        if rank == 0 and row.get("_source_kind") == "chat":
            key = parsed.strftime("%d/%m/%Y")
            if key not in counts:
                order.append(key)
            counts[key] = counts.get(key, 0) + 1
    if counts:
        selected = max(order, key=lambda key: counts[key])
        return datetime.strptime(selected, "%d/%m/%Y")

    report = read_case_report_text(report_path)
    candidates: List[datetime] = []
    for pattern, formats in (
        (r"\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b", ("%d/%m/%Y", "%d-%m-%Y")),
        (r"\b\d{4}-\d{1,2}-\d{1,2}\b", ("%Y-%m-%d",)),
        (r"\b\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}\b", ("%d %B %Y", "%d %b %Y")),
    ):
        for value in re.findall(pattern, report):
            for fmt in formats:
                try:
                    candidates.append(datetime.strptime(value, fmt))
                    break
                except ValueError:
                    continue
    return candidates[0].replace(hour=0, minute=0, second=0, microsecond=0) if candidates else None

def collect_evidence_date_candidates(
    chat_rows: Sequence[Dict[str, str]], report_text: str,
) -> List[str]:
    """Collect only dates actually present in chat evidence or the report."""
    dates: Set[str] = set()
    for row in chat_rows:
        rank, parsed = parse_chat_datetime(row.get("Timestamp", ""))
        if rank == 0 and row.get("_source_kind") == "chat":
            dates.add(parsed.strftime("%d/%m/%Y"))

    for pattern, formats in (
        (r"\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b", ("%d/%m/%Y", "%d-%m-%Y")),
        (r"\b\d{4}-\d{1,2}-\d{1,2}\b", ("%Y-%m-%d",)),
        (r"\b\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}\b", ("%d %B %Y", "%d %b %Y")),
    ):
        for value in re.findall(pattern, report_text):
            for fmt in formats:
                try:
                    dates.add(datetime.strptime(value, fmt).strftime("%d/%m/%Y"))
                    break
                except ValueError:
                    continue
    return sorted(dates, key=lambda value: datetime.strptime(value, "%d/%m/%Y"))

def _audio_date_context_tokens(value: str) -> Set[str]:
    stopwords = {
        "about", "after", "again", "also", "been", "before", "being",
        "could", "from", "have", "here", "into", "just", "more", "much",
        "only", "should", "that", "their", "there", "these", "they", "this",
        "through", "very", "what", "when", "where", "which", "with", "would",
        "your", "you're", "will", "shall", "hello", "thanks", "thank",
    }
    return {
        token for token in re.findall(r"[^\W_]+", str(value or "").casefold(), flags=re.UNICODE)
        if len(token) >= 4 and token not in stopwords
    }

def build_audio_date_chat_context(
    audio_rows: Sequence[Dict[str, str]], chat_rows: Sequence[Dict[str, str]],
    max_dates: int = 14, max_rows_per_date: int = 8,
) -> str:
    """Retrieve compact date-grouped chat excerpts relevant to one audio transcript."""
    audio_tokens = _audio_date_context_tokens(
        " ".join(row.get("Message", "") for row in audio_rows)
    )
    grouped: Dict[str, List[Tuple[int, Dict[str, str]]]] = {}
    for row in chat_rows:
        rank, parsed = parse_chat_datetime(row.get("Timestamp", ""))
        if rank != 0 or row.get("_source_kind") != "chat":
            continue
        date = parsed.strftime("%d/%m/%Y")
        overlap = len(audio_tokens & _audio_date_context_tokens(row.get("Message", "")))
        grouped.setdefault(date, []).append((overlap, row))

    ranked_dates = sorted(
        grouped,
        key=lambda date: (
            -sum(score for score, _ in grouped[date]),
            -max((score for score, _ in grouped[date]), default=0),
            datetime.strptime(date, "%d/%m/%Y"),
        ),
    )[:max_dates]
    blocks: List[str] = []
    for date in ranked_dates:
        selected = sorted(
            enumerate(grouped[date]),
            key=lambda item: (-item[1][0], item[0]),
        )[:max_rows_per_date]
        selected.sort(key=lambda item: item[0])
        lines = [f"DATE {date}"]
        for _, (_, row) in selected:
            lines.append(
                f"- {row.get('Sender', '')} -> {row.get('Receiver', '')}: "
                f"{row.get('Message', '')}"
            )
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)

def infer_audio_date_with_llm(
    *,
    source_audio: Path,
    audio_rows: Sequence[Dict[str, str]],
    chat_rows: Sequence[Dict[str, str]],
    report_text: str,
    model: str,
    host: str,
) -> Dict[str, str]:
    """Infer one evidence-bounded date independently for one audio file."""
    candidates = collect_evidence_date_candidates(chat_rows, report_text)
    if not candidates:
        return {
            "ok": "False", "date": "", "confidence": "low",
            "evidence": "", "error": "no report/chat date candidates",
        }
    try:
        import ollama
    except Exception as exc:
        return {
            "ok": "False", "date": "", "confidence": "low",
            "evidence": "", "error": str(exc),
        }

    transcript = "\n".join(
        f"{row.get('Sender', '')} -> {row.get('Receiver', '')}: {row.get('Message', '')}"
        for row in audio_rows
    )
    chat_context = build_audio_date_chat_context(audio_rows, chat_rows)
    prompt = f"""Choose the date of ONE forensic audio recording from evidence context.

Return JSON only:
{{"date":"DD/MM/YYYY or Unknown","confidence":"high|medium|low","evidence":"brief evidence-grounded reason"}}

Hard rules:
1. Choose at most one date, independently for this audio file.
2. The date must be exactly one value from ALLOWED DATES, or Unknown.
3. Match the transcript's events, participants, sequence, filename provenance, report timeline, and nearby chat content.
4. Do not use the most frequent/global chat date merely because it has more rows.
5. Do not invent a date. If the evidence is ambiguous, return Unknown.
6. The chosen date anchors the entire recording; every speaker turn in this file keeps that same calendar date.

AUDIO FILENAME:
{source_audio.name}

AUDIO TRANSCRIPT:
---
{transcript[:16000]}
---

ALLOWED DATES:
{json.dumps(candidates, ensure_ascii=False)}

RELEVANT DATE-GROUPED CHAT EXCERPTS:
---
{chat_context[:26000]}
---

CASE REPORT:
---
{report_text[:32000]}
---
"""
    try:
        client = ollama.Client(host=normalize_http_base_url(host))
        response = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={"temperature": 0},
        )
        data = _parse_json_object(response["message"]["content"])
        date = str(data.get("date", "")).strip()
        confidence = str(data.get("confidence", "low")).strip().casefold()
        evidence = str(data.get("evidence", "")).strip()[:800]
        if date not in candidates:
            raise ValueError("LLM date is not in the evidence-bounded candidate set")
        if confidence not in {"high", "medium"}:
            raise ValueError("LLM date confidence is low")
        return {
            "ok": "True", "date": date, "confidence": confidence,
            "evidence": evidence, "error": "",
        }
    except Exception as exc:
        return {
            "ok": "False", "date": "", "confidence": "low",
            "evidence": "", "error": str(exc),
        }

def parse_chat_datetime(time_value: str) -> Tuple[int, datetime]:
    """Parses transcript timestamps for chronological sorting."""
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
            return 0, datetime.strptime(text, fmt)
        except ValueError:
            pass
    return 1, datetime.max

def normalize_message_for_dedupe(message: str) -> str:
    """Normalizes message text only for duplicate-row detection."""
    text = str(message or "").lower()
    text = text.replace("’", "'")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest without loading large media into memory."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError:
        return ""
    return digest.hexdigest()

def normalize_audio_transcript_text(message: str) -> str:
    """Normalize speech text for comparison only; never alter evidence text."""
    text = str(message or "").casefold().replace("’", "'")
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()

def audio_text_tokens(message: str) -> Set[str]:
    """Return normalized unique tokens used by conservative overlap checks."""
    return set(normalize_audio_transcript_text(message).split())

def is_unresolved_audio_participant(value: str) -> bool:
    """Recognize unresolved diarizer labels without treating real names as unknown."""
    label = re.sub(r"[_\s-]+", " ", str(value or "").strip().casefold())
    return (
        not label
        or label in {"unknown", "unknown sender", "unknown receiver", "unresolved"}
        or bool(re.fullmatch(r"speaker\s*\d+", label))
    )

def audio_row_match_details(
    duplicate_row: Dict[str, str],
    canonical_row: Dict[str, str],
) -> Optional[Dict[str, float]]:
    """Return similarity details when one audio row is safely covered by another."""
    duplicate_text = normalize_audio_transcript_text(duplicate_row.get("Message", ""))
    canonical_text = normalize_audio_transcript_text(canonical_row.get("Message", ""))
    if not duplicate_text or not canonical_text:
        return None

    duplicate_tokens = audio_text_tokens(duplicate_text)
    canonical_tokens = audio_text_tokens(canonical_text)
    minimum_tokens = min(len(duplicate_tokens), len(canonical_tokens))
    if minimum_tokens < AUDIO_DUPLICATE_MIN_TOKENS:
        return None

    # Conflicting resolved identities are evidence against duplication. When
    # one side is unresolved, compare the known participant set without
    # assuming that the partial attribution assigned the correct role.
    duplicate_people = {
        str(duplicate_row.get(field, "")).strip().casefold()
        for field in ("Sender", "Receiver")
        if not is_unresolved_audio_participant(duplicate_row.get(field, ""))
    }
    canonical_people = {
        str(canonical_row.get(field, "")).strip().casefold()
        for field in ("Sender", "Receiver")
        if not is_unresolved_audio_participant(canonical_row.get(field, ""))
    }
    duplicate_fully_resolved = len(duplicate_people) == 2
    canonical_fully_resolved = len(canonical_people) == 2
    if duplicate_fully_resolved and canonical_fully_resolved:
        for field in ("Sender", "Receiver"):
            if (
                str(duplicate_row.get(field, "")).strip().casefold()
                != str(canonical_row.get(field, "")).strip().casefold()
            ):
                return None
    elif duplicate_people and not duplicate_people.issubset(canonical_people):
        return None

    sequence_ratio = SequenceMatcher(None, duplicate_text, canonical_text).ratio()
    shared_tokens = duplicate_tokens & canonical_tokens
    token_containment = len(shared_tokens) / max(1, minimum_tokens)

    if (
        sequence_ratio < AUDIO_DUPLICATE_ROW_SEQUENCE_THRESHOLD
        or token_containment < AUDIO_DUPLICATE_TOKEN_CONTAINMENT_THRESHOLD
    ):
        return None
    return {
        "sequence_ratio": sequence_ratio,
        "token_containment": token_containment,
    }

def audio_candidate_coverage(
    duplicate: AudioCandidate,
    canonical: AudioCandidate,
) -> Optional[Dict[str, Any]]:
    """Check whether every row of ``duplicate`` is represented by ``canonical``."""
    if duplicate.sha256 and duplicate.sha256 == canonical.sha256:
        return {
            "method": "identical_audio_sha256",
            "matched_rows": len(duplicate.rows),
            "minimum_sequence_ratio": 1.0,
            "minimum_token_containment": 1.0,
        }

    duplicate_date = str(duplicate.record.get("inferred_date", "")).strip()
    canonical_date = str(canonical.record.get("inferred_date", "")).strip()
    dates_match = bool(duplicate_date and duplicate_date == canonical_date)
    if not duplicate.rows or len(duplicate.rows) > len(canonical.rows):
        return None

    available = set(range(len(canonical.rows)))
    matches: List[Dict[str, float]] = []
    for duplicate_row in duplicate.rows:
        best_index: Optional[int] = None
        best_details: Optional[Dict[str, float]] = None
        for index in available:
            details = audio_row_match_details(duplicate_row, canonical.rows[index])
            if details is None:
                continue
            # Different or missing dates must not block duplicate detection:
            # these dates were inferred after transcription and may vary
            # between otherwise identical runs.  Compensate by requiring
            # near-exact transcript agreement for cross-date matches.
            if not dates_match and (
                details["sequence_ratio"]
                < AUDIO_DUPLICATE_CROSS_DATE_ROW_SEQUENCE_THRESHOLD
                or details["token_containment"]
                < AUDIO_DUPLICATE_CROSS_DATE_TOKEN_CONTAINMENT_THRESHOLD
            ):
                continue
            if (
                best_details is None
                or (details["sequence_ratio"], details["token_containment"])
                > (best_details["sequence_ratio"], best_details["token_containment"])
            ):
                best_index = index
                best_details = details
        if best_index is None or best_details is None:
            return None
        available.remove(best_index)
        matches.append(best_details)

    return {
        "method": (
            "covered_transcript_rows"
            if dates_match
            else "covered_transcript_rows_cross_date"
        ),
        "matched_rows": len(matches),
        "minimum_sequence_ratio": min(item["sequence_ratio"] for item in matches),
        "minimum_token_containment": min(item["token_containment"] for item in matches),
        "duplicate_inferred_date": duplicate_date,
        "canonical_inferred_date": canonical_date,
    }

def audio_candidate_quality(candidate: AudioCandidate) -> Tuple[Any, ...]:
    """Rank canonical recordings using evidence quality, never transcript meaning."""
    rows = candidate.rows
    unresolved_fields = sum(
        is_unresolved_audio_participant(row.get(field, ""))
        for row in rows
        for field in ("Sender", "Receiver")
    )
    resolved_fields = (2 * len(rows)) - unresolved_fields
    known_people = {
        str(row.get(field, "")).strip().casefold()
        for row in rows
        for field in ("Sender", "Receiver")
        if not is_unresolved_audio_participant(row.get(field, ""))
    }
    nonzero_offsets = sum(
        1 for row in rows if float(row.get("_offset_seconds", "0") or 0) > 0
    )
    token_count = sum(len(normalize_audio_transcript_text(row.get("Message", "")).split()) for row in rows)
    return (
        unresolved_fields == 0,
        resolved_fields,
        len(known_people),
        len(rows),
        nonzero_offsets,
        token_count,
        -candidate.order,
    )

def remove_exact_rows_within_audio_candidate(candidate: AudioCandidate) -> int:
    """Remove only byte-equivalent CSV artifacts at the same audio offset."""
    kept: List[Dict[str, str]] = []
    seen: Set[Tuple[str, ...]] = set()
    for row in candidate.rows:
        key = (
            str(row.get("Timestamp", "")),
            str(row.get("_offset_seconds", "")),
            str(row.get("Sender", "")).strip().casefold(),
            str(row.get("Receiver", "")).strip().casefold(),
            normalize_message_for_dedupe(row.get("Message", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        kept.append(row)
    removed = len(candidate.rows) - len(kept)
    candidate.rows = kept
    return removed

def deduplicate_audio_candidates(
    candidates: List[AudioCandidate],
    enabled: bool = True,
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    """Select canonical audio evidence while retaining auditable source records."""
    summary: Dict[str, Any] = {
        "enabled": bool(enabled),
        "candidate_recordings": len(candidates),
        "canonical_recordings": len(candidates),
        "duplicate_recordings_removed": 0,
        "exact_rows_removed": 0,
        "thresholds": {
            "row_sequence_ratio": AUDIO_DUPLICATE_ROW_SEQUENCE_THRESHOLD,
            "token_containment": AUDIO_DUPLICATE_TOKEN_CONTAINMENT_THRESHOLD,
            "minimum_unique_tokens": AUDIO_DUPLICATE_MIN_TOKENS,
            "cross_date_row_sequence_ratio": (
                AUDIO_DUPLICATE_CROSS_DATE_ROW_SEQUENCE_THRESHOLD
            ),
            "cross_date_token_containment": (
                AUDIO_DUPLICATE_CROSS_DATE_TOKEN_CONTAINMENT_THRESHOLD
            ),
        },
        "decisions": [],
    }
    if not enabled:
        return [row for candidate in candidates for row in candidate.rows], summary

    for candidate in candidates:
        removed = remove_exact_rows_within_audio_candidate(candidate)
        if removed:
            candidate.record["exact_rows_removed"] = str(removed)
            candidate.record["rows"] = str(len(candidate.rows))
            summary["exact_rows_removed"] += removed

    ranked = sorted(candidates, key=audio_candidate_quality, reverse=True)
    canonical_candidates: List[AudioCandidate] = []
    for candidate in ranked:
        duplicate_of: Optional[AudioCandidate] = None
        duplicate_details: Optional[Dict[str, Any]] = None
        for canonical in canonical_candidates:
            details = audio_candidate_coverage(candidate, canonical)
            if details is not None:
                duplicate_of = canonical
                duplicate_details = details
                break

        if duplicate_of is None:
            canonical_candidates.append(candidate)
            continue

        candidate.record["status"] = "duplicate"
        candidate.record["rows_before_dedupe"] = str(len(candidate.rows))
        candidate.record["rows"] = "0"
        candidate.record["duplicate_of"] = str(duplicate_of.source_audio)
        candidate.record["duplicate_method"] = str(duplicate_details["method"])
        candidate.record["duplicate_min_sequence_ratio"] = (
            f"{float(duplicate_details['minimum_sequence_ratio']):.6f}"
        )
        candidate.record["duplicate_min_token_containment"] = (
            f"{float(duplicate_details['minimum_token_containment']):.6f}"
        )
        candidate.record["reason"] = (
            "excluded from normalized CSV because its transcript is covered by "
            f"the canonical audio source {duplicate_of.source_audio.name}"
        )
        summary["decisions"].append({
            "excluded_audio": str(candidate.source_audio),
            "canonical_audio": str(duplicate_of.source_audio),
            **duplicate_details,
        })

    canonical_candidates.sort(key=lambda item: item.order)
    summary["canonical_recordings"] = len(canonical_candidates)
    summary["duplicate_recordings_removed"] = len(candidates) - len(canonical_candidates)
    rows = [row for candidate in canonical_candidates for row in candidate.rows]
    return rows, summary

COMMON_IDENTIFIER_TLDS = (
    "com|net|org|edu|gov|mil|int|io|ai|co|uk|gr|eu|de|fr|it|es|nl|"
    "be|ch|at|us|ca|au|info|biz|online|site|app|dev|tech|me|tv"
)

def repair_broken_identifiers(message: str) -> str:
    """Repair OCR whitespace/noise only inside unmistakable web/email identifiers."""
    value = str(message or "")

    # OCR frequently reads the colon/slashes in ``https://`` as ``Il`` or
    # vertical bars. Restrict this repair to a following host plus known TLD.
    value = re.sub(
        rf"\b(?i:https)\s+(?:[:/\\|]+|I[lI]|l[I|])\s*"
        rf"(?=[A-Za-z0-9][A-Za-z0-9.-]*[-A-Za-z0-9]\s+(?:{COMMON_IDENTIFIER_TLDS})\b)",
        "https://",
        value,
    )

    # ``www`` is itself an unambiguous web cue, so OCR whitespace around its
    # dot can be removed without touching ordinary sentence punctuation.
    value = re.sub(
        r"\bwww\s*\.\s*(?=[A-Za-z0-9])",
        "www.",
        value,
        flags=re.I,
    )

    # Email addresses with OCR whitespace around ``@`` or the final dot/TLD:
    # ``name @ example. com`` -> ``name@example.com``.
    value = re.sub(
        rf"([\w.+-]+)\s*@\s*"
        rf"([A-Za-z0-9][A-Za-z0-9.-]*[A-Za-z0-9-])\s*\.\s*"
        rf"({COMMON_IDENTIFIER_TLDS})\b",
        r"\1@\2.\3",
        value,
        flags=re.I,
    )

    # Web hosts with either ``host. tld`` or ``host tld``.
    scheme_host = r"((?:https?://|www\.)[A-Za-z0-9][A-Za-z0-9.-]*[A-Za-z0-9-])"
    value = re.sub(
        rf"{scheme_host}\s*\.\s*({COMMON_IDENTIFIER_TLDS})\b",
        r"\1.\2",
        value,
        flags=re.I,
    )
    value = re.sub(
        rf"{scheme_host}\s+({COMMON_IDENTIFIER_TLDS})\b",
        r"\1.\2",
        value,
        flags=re.I,
    )

    # A bare host has no scheme or ``www`` prefix. Repair a separated final
    # dot/TLD only after an explicit web-context cue; this covers OCR such as
    # ``website anydesk. com`` without treating normal prose as a domain.
    web_context_cue = r"((?:website|web\s+site|site|url|visit|go\s+to)\s+)"
    bare_host = r"([A-Za-z0-9][A-Za-z0-9.-]*[A-Za-z0-9-])"
    value = re.sub(
        rf"\b{web_context_cue}{bare_host}\s*\.\s*"
        rf"({COMMON_IDENTIFIER_TLDS})\b",
        r"\1\2.\3",
        value,
        flags=re.I,
    )

    # A domain may instead lose its only dot (``website anydeskcom``).
    # Keep the same explicit-context restriction to avoid changing words.
    value = re.sub(
        rf"\b{web_context_cue}"
        rf"([A-Za-z0-9-]{{2,}}?)({COMMON_IDENTIFIER_TLDS})\b",
        r"\1\2.\3",
        value,
        flags=re.I,
    )
    return value


def repair_chat_message_identifiers(rows: Sequence[Dict[str, str]]) -> int:
    """Apply low-risk URL/email repair deterministically to chat rows.

    This pass deliberately runs outside the LLM. It changes only ``Message``
    and only when ``repair_broken_identifiers`` recognizes an unmistakable
    URL, website, domain, or email pattern. The returned count is the number
    of chat rows whose message changed.
    """
    repaired = 0
    for row in rows:
        if row.get("_source_kind") != "chat":
            continue
        original = str(row.get("Message", ""))
        corrected = repair_broken_identifiers(original)
        if corrected != original:
            row["Message"] = corrected
            repaired += 1
    return repaired

def conservative_final_message_cleanup(message: str) -> str:
    """Final low-risk OCR cleanup for merged transcript rows."""
    msg = str(message or "").strip()
    msg = msg.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")

    msg = re.sub(r"\s+", " ", msg).strip()
    msg = re.sub(r"\s+([,.;:!?])", r"\1", msg)
    msg = re.sub(r"([,.;:!?])(?=[A-Za-z])", r"\1 ", msg)

    # Run identifier repair last. The generic punctuation rule above inserts
    # sentence spacing after a period; if identifier repair ran first, that
    # rule would turn ``example.com`` back into ``example. com``.
    return repair_broken_identifiers(msg)

def split_final_chat_row(row: Dict[str, str]) -> List[Dict[str, str]]:
    """Apply final message cleanup without dataset-specific row splitting."""
    fixed = dict(row)
    fixed["Message"] = conservative_final_message_cleanup(fixed.get("Message", ""))
    return [fixed] if fixed["Message"] else []

def merge_final_adjacent_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Return final rows without dataset-specific adjacent-row merges."""
    return rows

def postprocess_final_chat_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Apply final merged-transcript cleanup, guarded bubble splits, and tiny row merges."""
    out: List[Dict[str, str]] = []
    for row in rows:
        out.extend(split_final_chat_row(row))
    return merge_final_adjacent_rows(out)

def _message_tokens(value: str) -> List[str]:
    """Tokenize only for a conservative, language-agnostic edit guard."""
    return re.findall(r"[^\W_]+", str(value or "").casefold(), flags=re.UNICODE)

def _protected_message_values(value: str) -> List[str]:
    """Extract evidence tokens that a polish pass must preserve byte-for-byte."""
    pattern = re.compile(
        r"https?://\S+|www\.\S+|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|"
        r"(?<!\w)[€$£]?\d[\d.,:/#-]*",
        flags=re.I,
    )
    return pattern.findall(str(value or ""))

def _is_obvious_wrapped_clause_rotation(original: str, proposed: str) -> bool:
    """Allow only the common OCR case where a trailing clause was read first."""
    before_matches = list(re.finditer(r"[^\W_]+", original, flags=re.UNICODE))
    before_tokens = [match.group(0).casefold() for match in before_matches]
    after_tokens = _message_tokens(proposed)
    if len(before_tokens) < 3 or len(before_tokens) != len(after_tokens):
        return False

    before_first = next((char for char in original if char.isalpha()), "")
    after_first = next((char for char in proposed if char.isalpha()), "")
    if not (before_first.islower() and after_first.isupper()):
        return False

    for cut in range(1, len(before_tokens)):
        if before_tokens[cut:] + before_tokens[:cut] != after_tokens:
            continue
        boundary = original[
            before_matches[cut - 1].end():before_matches[cut].start()
        ]
        suffix_first = before_matches[cut].group(0)
        # Colons/semicolons can also mark the clause boundary that OCR wrapped.
        # The exact cyclic-token check above still prevents arbitrary reorderings.
        if re.search(r"[.!?…:;]", boundary) and suffix_first[:1].isupper():
            return True
    return False

def safe_message_only_edit(original: str, proposed: str) -> Tuple[bool, str]:
    """Accept only corrections that cannot add or remove lexical content.

    Allowed edits are casing, punctuation, whitespace, and reordering of the
    exact same tokens.  This accepts corrections such as
    ``doing today? How are you`` -> ``How are you doing today?`` while rejecting
    paraphrases, new facts, deleted words, changed numbers, and changed URLs.
    """
    before = str(original or "").strip()
    after = str(proposed or "").strip()
    if not before or not after:
        return False, "empty message"
    if before == after:
        return True, "unchanged"
    if _protected_message_values(before) != _protected_message_values(after):
        return False, "protected number/URL/email changed"
    before_tokens = _message_tokens(before)
    after_tokens = _message_tokens(after)
    if not before_tokens or not after_tokens:
        return False, "no lexical tokens"
    if before_tokens == after_tokens:
        return True, "same token sequence"
    if (
        Counter(before_tokens) == Counter(after_tokens)
        and _is_obvious_wrapped_clause_rotation(before, after)
    ):
        return True, "obvious wrapped-clause rotation"
    if "".join(before_tokens) == "".join(after_tokens):
        return True, "whitespace/punctuation repair"
    return False, "lexical content changed"

def _parse_json_object(value: str) -> Dict[str, Any]:
    """Parse a JSON object, tolerating a surrounding Markdown fence."""
    text = str(value or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except Exception:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("LLM did not return a JSON object")
        data = json.loads(text[start:end + 1])
    if not isinstance(data, dict):
        raise ValueError("LLM response must be a JSON object")
    return data

def normalize_http_base_url(value: str, default: str = "http://127.0.0.1:11434") -> str:
    """Normalize an Ollama host exported with or without an HTTP scheme."""
    text = str(value or default).strip().rstrip("/")
    if not re.match(r"^https?://", text, flags=re.I):
        text = "http://" + text
    return text

def strict_polish_chat_messages(
    rows: List[Dict[str, str]],
    *,
    model: str,
    host: str,
    batch_size: int,
) -> Dict[str, Any]:
    """Run a post-extraction LLM pass that can mutate only ``Message``.

    Row count/order and every non-message field are immutable.  Proposed text
    is additionally rejected unless it passes ``safe_message_only_edit``.
    """
    try:
        import ollama
    except Exception as exc:
        return {"ok": False, "accepted": 0, "rejected": 0, "error": str(exc)}

    client = ollama.Client(host=normalize_http_base_url(host))
    accepted = 0
    rejected = 0
    batch_size = max(1, min(int(batch_size or 20), 50))

    prompt_rules = """You are performing a STRICT forensic transcript cleanup.
Correct only obvious OCR word-order, spacing, capitalization, or punctuation errors inside each Message.

Hard rules:
1. Return one item for every supplied id, in the same order.
2. Change only the message string. Never add/delete/split/merge rows.
3. Do not paraphrase, summarize, translate, complete, or improve style.
4. Preserve every word/token, name, number, amount, phone number, URL, email, and factual claim.
5. Reorder words only when the current OCR order is plainly broken, e.g. "doing today? How are you" -> "How are you doing today?".
6. If uncertain, return the original message exactly.
7. Treat every website, URL, domain, and email address as one indivisible identifier. Never insert spaces inside it; remove only obvious OCR whitespace around '.', '/', ':', '@', or its top-level domain.

Return JSON only: {"items":[{"id":1,"message":"..."}]}.
"""

    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        payload = [
            {"id": start + offset + 1, "message": row.get("Message", "")}
            for offset, row in enumerate(batch)
        ]
        try:
            response = client.chat(
                model=model,
                messages=[{
                    "role": "user",
                    "content": prompt_rules + "\nINPUT:\n" + json.dumps(payload, ensure_ascii=False),
                }],
                format="json",
                options={"temperature": 0},
            )
            parsed = _parse_json_object(response["message"]["content"])
            items = parsed.get("items")
            if not isinstance(items, list) or len(items) != len(batch):
                raise ValueError("LLM returned a different number of rows")
            by_id = {
                int(item.get("id")): item
                for item in items
                if isinstance(item, dict) and str(item.get("id", "")).isdigit()
            }
            expected_ids = list(range(start + 1, start + len(batch) + 1))
            if sorted(by_id) != expected_ids:
                raise ValueError("LLM changed or omitted row ids")

            for offset, row in enumerate(batch):
                row_id = start + offset + 1
                proposed = str(by_id[row_id].get("message", ""))
                original = row.get("Message", "")
                allowed, _ = safe_message_only_edit(original, proposed)
                if allowed:
                    if proposed.strip() != str(original).strip():
                        row["Message"] = proposed.strip()
                        accepted += 1
                else:
                    rejected += 1
        except Exception as exc:
            return {
                "ok": False,
                "accepted": accepted,
                "rejected": rejected,
                "error": f"batch {start // batch_size + 1}: {exc}",
            }

    return {"ok": True, "accepted": accepted, "rejected": rejected, "error": ""}

def invalid_chat_participant_labels(rows: Sequence[Dict[str, str]]) -> List[str]:
    """Return generic report/UI headings that leaked into chat identities."""
    heading_terms = {
        "possible offences", "possible offenses", "executive summary",
        "incident summary", "case overview", "evidence package",
        "matter reported", "timeline", "victim details",
        "complainant details", "suspect details", "report summary",
    }
    invalid: Set[str] = set()
    blocked_identity_tokens = {
        "offence", "offences", "offense", "offenses", "summary", "overview",
        "details", "evidence", "report", "timeline", "section", "subject",
        "victim", "complainant", "suspect", "sender", "receiver", "profile",
        "account", "contact", "information", "matter", "incident",
    }
    for row in rows:
        if row.get("_source_kind") != "chat":
            continue
        for field in ("Sender", "Receiver"):
            value = re.sub(r"\s+", " ", str(row.get(field, "")).strip())
            tokens = {
                token.casefold()
                for token in re.findall(r"[^\W_]+", value, flags=re.UNICODE)
            }
            if value.casefold() in heading_terms or tokens & blocked_identity_tokens:
                invalid.add(value)
    return sorted(invalid, key=str.casefold)

def report_human_names(report_text: str) -> Tuple[List[str], str]:
    """Reuse the audio module's report-grounded, dataset-neutral name parser."""
    speech_dir = Path(__file__).resolve().parents[1] / "speech"
    speech_dir_text = str(speech_dir)
    if speech_dir_text not in sys.path:
        sys.path.insert(0, speech_dir_text)
    try:
        from audio_utils import extract_actor_candidates, extract_primary_report_party

        names = list(extract_actor_candidates(report_text))
        primary = str(extract_primary_report_party(report_text) or "").strip()
        if primary and primary.casefold() not in {name.casefold() for name in names}:
            names.insert(0, primary)
        return names, primary
    except Exception as exc:
        print(f"[WARN] Could not load report-grounded participant parser: {exc}")
        return [], ""

def repair_invalid_chat_participants(
    rows: Sequence[Dict[str, str]], report_path: Path,
) -> Tuple[int, List[str]]:
    """Replace leaked report headings only from a structured victim field.

    This is a deterministic post-extraction guard, not an LLM attribution
    pass. It activates only when a known document heading already occupies a
    Sender/Receiver field and the report exposes one unambiguous structured
    victim/complainant full name.
    """
    report = read_case_report_text(report_path)
    allowed_names, primary_party = report_human_names(report)
    invalid_values = {
        value.casefold() for value in invalid_chat_participant_labels(rows)
    }

    changes = 0
    unresolved: Set[str] = set()
    for row in rows:
        for field, other_field in (("Sender", "Receiver"), ("Receiver", "Sender")):
            current = str(row.get(field, "")).strip()
            if current.casefold() not in invalid_values:
                continue
            other = str(row.get(other_field, "")).strip()
            replacement = ""
            if primary_party and primary_party.casefold() != other.casefold():
                replacement = primary_party
            else:
                alternatives = [
                    name for name in allowed_names
                    if name.casefold() != other.casefold()
                ]
                if len(alternatives) == 1:
                    replacement = alternatives[0]
            if replacement:
                row[field] = replacement
                changes += 1
            else:
                unresolved.add(current)
    return changes, sorted(unresolved, key=str.casefold)

def renumber_surviving_facebook_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Close gaps in synthetic Facebook seconds after final de-duplication."""
    groups: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        if row.get("_source_kind") == "chat":
            groups.setdefault(str(row.get("_source_csv", "")), []).append(row)
    for group_rows in groups.values():
        group_rows.sort(key=lambda item: int(item.get("_source_row", "0") or 0))
        anchor: Optional[datetime] = None
        offset = 1
        for row in group_rows:
            parsed_rank, parsed = parse_chat_datetime(row.get("Timestamp", ""))
            if normalize_estimated_timestamp(row.get("Estimated_Timestamp")) == "False":
                anchor = parsed if parsed_rank == 0 else None
                offset = 1
            elif anchor is not None:
                row["Timestamp"] = (anchor + timedelta(seconds=offset)).strftime("%d/%m/%Y %H:%M:%S")
                row["Estimated_Timestamp"] = "True"
                offset += 1
    return rows

def write_merged_csv(rows: List[Dict[str, str]], output_path: Path, dedupe: bool = True) -> int:
    """Write every evidence row in stable chronological ascending order."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = postprocess_final_chat_rows(rows)
    indexed_rows = list(enumerate(rows))
    indexed_rows.sort(
        key=lambda pair: (
            parse_chat_datetime(pair[1]["Timestamp"])[0],
            parse_chat_datetime(pair[1]["Timestamp"])[1],
            pair[0],
        )
    )

    out_rows: List[Dict[str, str]] = []
    seen: Set[Tuple[str, ...]] = set()

    for _, row in indexed_rows:
        if row.get("_source_kind") == "audio":
            # Repeated speech can be genuine evidence. Preserve every audio
            # turn while still deduplicating exact repeated chat OCR rows.
            key = (
                "audio",
                row.get("_source_csv", ""),
                row.get("_source_row", ""),
            )
        else:
            key = (
                "chat",
                row["Timestamp"],
                row["Sender"],
                row["Receiver"],
                normalize_message_for_dedupe(row["Message"]),
            )
        if dedupe and key in seen:
            continue
        seen.add(key)
        out_rows.append(row)

    out_rows = renumber_surviving_facebook_rows(out_rows)
    out_rows = [row for _, row in sorted(
        enumerate(out_rows),
        key=lambda pair: (
            parse_chat_datetime(pair[1]["Timestamp"])[0],
            parse_chat_datetime(pair[1]["Timestamp"])[1],
            pair[0],
        ),
    )]

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL, lineterminator="\n")
        writer.writerow(["Timestamp", "Estimated_Timestamp", "Sender", "Receiver", "Message"])
        for row in out_rows:
            writer.writerow([
                row["Timestamp"],
                normalize_estimated_timestamp(row.get("Estimated_Timestamp")),
                row["Sender"], row["Receiver"], row["Message"],
            ])

    return len(out_rows)

# ============================================================
# MANIFEST
# ============================================================
def utc_now_iso() -> str:
    """Returns current UTC timestamp for manifest metadata."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def write_run_manifest(
    manifest_path: Path,
    input_plan: InputPlan,
    image_records: List[Dict[str, str]],
    audio_records: List[Dict[str, str]],
    output_csv: Path,
    rows_merged: int,
    args: argparse.Namespace,
    *,
    polish_result: Optional[Dict[str, Any]] = None,
    failure_reasons: Optional[Sequence[str]] = None,
    raw_chat_output: Optional[Path] = None,
    polished_chat_output: Optional[Path] = None,
    audio_deduplication: Optional[Dict[str, Any]] = None,
) -> None:
    """Writes a JSON manifest describing extraction inputs/results."""
    unique_failures = list(dict.fromkeys(failure_reasons or []))
    manifest = {
        "created_at": utc_now_iso(),
        "status": "partial_failure" if unique_failures else "success",
        "failure_reasons": unique_failures,
        "mode": input_plan.mode,
        "case_report": str(input_plan.report_path),
        "output_csv": str(output_csv),
        "raw_chat_output": str(raw_chat_output) if raw_chat_output else None,
        "polished_chat_output": str(polished_chat_output) if polished_chat_output else None,
        "rows_merged": rows_merged,
        "chat_polish": polish_result,
        "audio_deduplication": audio_deduplication,
        "settings": {
            "model": args.model,
            "langs": args.langs,
            "emoji_mode": args.emoji_mode,
            "classify_mode": args.classify_mode,
            "force_platform": args.force_platform,
            "debug": bool(args.debug),
            "no_vision": bool(args.no_vision),
            "cpu": bool(args.cpu),
            "audio_backend": args.audio_backend,
            "audio_model_id": args.audio_model_id,
            "audio_device": args.audio_device,
            "audio_dtype": args.audio_dtype,
            "audio_llm_backend": args.audio_llm_backend,
            "audio_ollama_model": args.audio_ollama_model or args.model,
            "audio_deduplication_enabled": not args.keep_duplicates,
            "chat_polish_enabled": not args.no_chat_polish,
            "chat_polish_model": args.chat_polish_model or args.model,
            "chat_csvs_retained": should_keep_chat_csvs(args),
        },
        "evidence_sources": [
            {
                "source": str(source.original_path),
                "resolved_root": str(source.root_path),
                "type": source.source_type,
                "extracted": source.extracted,
            }
            for source in input_plan.evidence_sources
        ],
        "counts": {
            "files_seen": len(input_plan.file_inventory),
            "candidate_images": len(input_plan.images),
            "candidate_audio": len(input_plan.audio_files),
            "processed_images": sum(1 for item in image_records if item.get("status") == "ok"),
            "skipped_images": sum(1 for item in image_records if item.get("status", "").startswith("skipped")),
            "failed_images": sum(1 for item in image_records if item.get("status") == "failed"),
            "processed_audio": sum(1 for item in audio_records if item.get("status") == "ok"),
            "duplicate_audio": sum(1 for item in audio_records if item.get("status") == "duplicate"),
            "failed_audio": sum(
                1 for item in audio_records
                if item.get("status") not in {"ok", "duplicate"}
            ),
        },
        "images": image_records,
        "audio": audio_records,
        "files": input_plan.file_inventory,
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

# ============================================================
# MAIN
# ============================================================
def build_parser() -> argparse.ArgumentParser:
    """Builds the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Extract chat screenshots, email screenshots, and diarized audio from a use-case ZIP/folder, "
            "then merge them into one CSV."
        )
    )

    parser.add_argument(
        "package_or_report",
        help=(
            "Either: (1) a use-case ZIP/folder containing the case overview/report and evidence, "
            "or (2) a PDF/TXT case report when additional input paths are supplied."
        ),
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Legacy mode: evidence files, folders, ZIPs, or globs used with the explicit case report.",
    )
    parser.add_argument(
        "--case-report",
        default=None,
        help=(
            "Override auto-discovery and use this PDF/TXT report. With this option, "
            "all positional paths are treated as evidence inputs."
        ),
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help=(
            "Output merged CSV path. Default: ./results/<input_stem>_merged.csv. "
            "Relative paths are placed directly under --results-dir."
        ),
    )
    parser.add_argument(
        "--results-dir",
        default="./results",
        help="Root folder for generated files. Default: ./results.",
    )
    parser.add_argument(
        "--viber-script",
        default=None,
        help="Path to viber_extract.py. Default: next to this script.",
    )
    parser.add_argument(
        "--facebook-script",
        default=None,
        help="Path to facebook_extract.py. Default: next to this script.",
    )
    parser.add_argument(
        "--audio-script",
        default=None,
        help="Path to audio_diarize.py. Default: ../speech/audio_diarize.py.",
    )
    parser.add_argument(
        "--email-script",
        default=None,
        help="Path to email_extract.py. Default: ../email/email_extract.py.",
    )
    parser.add_argument(
        "--classify-mode",
        choices=["auto", "vision", "filename"],
        default="auto",
        help="Default: auto. auto = VLM content first, then filename/path if unknown.",
    )
    parser.add_argument(
        "--force-platform",
        choices=["auto", "facebook", "viber", "email"],
        default="auto",
        help="Force all candidate images to one extractor. Default: auto.",
    )
    parser.add_argument(
        "--model",
        default="gemma3:12b",
        help="Ollama model for classification and extractors. Default: gemma3:12b.",
    )
    polish_group = parser.add_argument_group("strict post-chat Message polish")
    polish_group.add_argument(
        "--chat-polish-model",
        default=None,
        help="Ollama model for strict Message-only cleanup. Default: value of --model.",
    )
    polish_group.add_argument(
        "--chat-polish-host",
        default=None,
        help="Ollama host. Default: OLLAMA_HOST or http://127.0.0.1:11434.",
    )
    polish_group.add_argument(
        "--chat-polish-batch-size",
        type=int,
        default=20,
        help="Rows per strict polish request. Default: 20.",
    )
    polish_group.add_argument(
        "--no-chat-polish",
        action="store_true",
        help="Disable the post-extraction Message-only LLM pass.",
    )
    polish_group.add_argument(
        "--keep-chat-csvs",
        action="store_true",
        help=(
            "Keep <stem>_chat_raw.csv and <stem>_chat_polished.csv. "
            "Default: off; only the final merged CSV is retained."
        ),
    )
    polish_group.add_argument(
        "--raw-chat-output",
        default=None,
        help=(
            "Custom raw chat-only CSV path. Supplying this option enables "
            "chat-CSV retention even without --keep-chat-csvs."
        ),
    )
    polish_group.add_argument(
        "--polished-chat-output",
        default=None,
        help=(
            "Custom polished chat-only CSV path. Supplying this option enables "
            "chat-CSV retention even without --keep-chat-csvs."
        ),
    )
    parser.add_argument(
        "--langs",
        default="en",
        help="EasyOCR languages passed to extractors. Default: en.",
    )
    parser.add_argument("--cpu", action="store_true", help="Pass --cpu to extractors.")
    parser.add_argument(
        "--no-vision",
        action="store_true",
        help=(
            "Pass --no-vision to extractors. Classification still uses VLM unless "
            "--classify-mode=filename or --force-platform is used."
        ),
    )
    parser.add_argument(
        "--emoji-mode",
        choices=["omit", "vision"],
        default="omit",
        help="Passed to extractors. Default: omit.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Keep per-image debug folders under results/per_image. "
            "Default: off; temporary per-image files are removed."
        ),
    )
    parser.add_argument(
        "--keep-per-image",
        action="store_true",
        help=(
            "Keep one intermediate CSV per processed image under results/per_image. "
            "Default: off; per-image CSVs are temporary."
        ),
    )
    parser.add_argument("--dump-ocr", action="store_true", help="Pass --dump-ocr to extractors and keep debug output.")
    parser.add_argument("--dump-draft", action="store_true", help="Pass --dump-draft to extractors and keep debug output.")
    parser.add_argument("--dump-side-map", action="store_true", help="Pass --dump-side-map to extractors and keep debug output.")
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help=(
            "Disable final chat-row deduplication and audio-recording deduplication. "
            "Default: remove conservative, auditable duplicates."
        ),
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional JSON manifest path. No manifest is written by default.",
    )
    parser.add_argument(
        "--extra-extractor-arg",
        action="append",
        default=[],
        help="Extra argument passed to both extractors. Use multiple times if needed.",
    )
    parser.add_argument(
        "--extra-email-arg",
        action="append",
        default=[],
        help="Extra argument passed only to email_extract.py. Use multiple times if needed.",
    )

    audio_group = parser.add_argument_group("audio diarization")
    audio_group.add_argument(
        "--audio-backend",
        choices=["moss-local", "moss-api"],
        default="moss-local",
        help="audio_diarize.py MOSS backend. Default: moss-local.",
    )
    audio_group.add_argument(
        "--audio-model-id",
        default=DEFAULT_AUDIO_MODEL_ID,
        help="MOSS model id or local model path.",
    )
    audio_group.add_argument("--audio-language", default="en", help="MOSS language hint. Default: en.")
    audio_group.add_argument(
        "--audio-date",
        default=None,
        help="Optional DD/MM/YYYY override. Otherwise date is inferred from chats, then report.",
    )
    audio_group.add_argument(
        "--audio-device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device passed to audio_diarize.py. Use auto, cuda, or cpu as appropriate.",
    )
    audio_group.add_argument(
        "--audio-dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="auto",
        help="MOSS dtype. Default: auto.",
    )
    audio_group.add_argument("--audio-max-new-tokens", type=int, default=65536)
    audio_group.add_argument("--audio-hotwords", default=None)
    audio_group.add_argument("--audio-max-hotwords", type=int, default=100)
    audio_group.add_argument("--audio-prompt", default=None)
    audio_group.add_argument(
        "--audio-api-url",
        default="http://127.0.0.1:8000/v1/audio/transcriptions",
    )
    audio_group.add_argument("--audio-api-timeout", type=int, default=3600)
    audio_group.add_argument(
        "--audio-llm-backend",
        choices=["ollama", "none"],
        default="ollama",
    )
    audio_group.add_argument(
        "--audio-ollama-model",
        default=None,
        help="Ollama attribution model. Default: value of --model.",
    )
    audio_group.add_argument(
        "--audio-ollama-host",
        default=None,
        help="Optional Ollama host override. Otherwise audio_diarize.py uses OLLAMA_HOST.",
    )
    audio_group.add_argument("--audio-llm-timeout", type=int, default=900)
    audio_group.add_argument("--audio-llm-temperature", type=float, default=0.0)
    audio_group.add_argument("--audio-max-case-report-chars", type=int, default=45000)
    audio_group.add_argument("--audio-max-transcript-chars", type=int, default=45000)
    audio_group.add_argument("--audio-speaker-map", default=None)
    audio_group.add_argument("--audio-participants", default=None)
    audio_group.add_argument("--audio-map-speakers-by-order", action="store_true")
    audio_group.add_argument("--audio-no-merge-turns", action="store_true")
    audio_group.add_argument("--audio-debug-json", action="store_true")
    audio_group.add_argument(
        "--keep-audio-output",
        action="store_true",
        help="Keep diarizer intermediate files under results/per_audio.",
    )
    audio_group.add_argument(
        "--extra-audio-arg",
        action="append",
        default=[],
        help="Extra argument passed to audio_diarize.py. Use multiple times if needed.",
    )
    return parser

def main() -> int:
    """Parses CLI arguments and runs the batch extraction pipeline."""
    parser = build_parser()
    args = parser.parse_args()

    results_dir = Path(args.results_dir).expanduser()
    extract_root = results_dir / EXTRACTED_ZIPS_DIR_NAME

    results_dir.mkdir(parents=True, exist_ok=True)
    extract_root.mkdir(parents=True, exist_ok=True)

    try:
        input_plan = resolve_input_plan(args, extract_root)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    viber_script = Path(args.viber_script).expanduser() if args.viber_script else default_script_path("viber_extract.py")
    facebook_script = Path(args.facebook_script).expanduser() if args.facebook_script else default_script_path("facebook_extract.py")
    email_script = Path(args.email_script).expanduser() if args.email_script else default_email_script_path()
    audio_script = Path(args.audio_script).expanduser() if args.audio_script else default_audio_script_path()

    try:
        if input_plan.images:
            ensure_exists(viber_script, "Viber extractor script")
            ensure_exists(facebook_script, "Facebook extractor script")
            ensure_exists(email_script, "Email extractor script")
            ensure_extractor_utils_available(viber_script)
            ensure_extractor_utils_available(facebook_script)
        if input_plan.audio_files:
            ensure_audio_script_available(audio_script)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    output_path = resolve_output_path(
        args.output,
        results_dir,
        f"{input_plan.run_stem}_merged.csv",
    )
    keep_chat_csvs = should_keep_chat_csvs(args)
    raw_chat_path = resolve_output_path(
        args.raw_chat_output,
        results_dir,
        f"{input_plan.run_stem}_chat_raw.csv",
    )
    polished_chat_path = resolve_output_path(
        args.polished_chat_output,
        results_dir,
        f"{input_plan.run_stem}_chat_polished.csv",
    )
    manifest_path = (
        resolve_output_path(args.manifest, results_dir, output_path.name + ".manifest.json")
        if args.manifest
        else None
    )

    if not input_plan.images and not input_plan.audio_files:
        print(
            "[ERROR] No supported chat images or audio/video files found in the input.",
            file=sys.stderr,
        )
        if manifest_path:
            write_run_manifest(
                manifest_path, input_plan, [], [], output_path, 0, args,
                failure_reasons=["no supported chat images or audio/video files found"],
            )
            print(f"[INFO] Manifest saved to: {manifest_path}")
        return 3

    keep_debug = bool(args.debug or args.dump_ocr or args.dump_draft or args.dump_side_map)
    keep_per_image = bool(args.keep_per_image or keep_debug)
    temporary_per_image_workspace: Optional[tempfile.TemporaryDirectory[str]] = None
    temporary_audio_workspace: Optional[tempfile.TemporaryDirectory[str]] = None
    temporary_case_context_workspace: Optional[tempfile.TemporaryDirectory[str]] = tempfile.TemporaryDirectory(prefix="kmodelc_case_context_")
    case_context_cache = Path(temporary_case_context_workspace.name) / "case_context.json"

    # Chat continuity is deliberately separate from the audio case graph.  It
    # stores only the last validated LEFT/RIGHT mapping per evidence folder.
    conversation_state_cache = Path(temporary_case_context_workspace.name) / "chat_conversation_state.json"

    if input_plan.images and keep_per_image:
        per_image_dir = results_dir / PER_IMAGE_DIR_NAME
        per_image_dir.mkdir(parents=True, exist_ok=True)
        per_image_label = str(per_image_dir)
    elif input_plan.images:
        temporary_per_image_workspace = tempfile.TemporaryDirectory(prefix="chat_extract_per_image_")
        per_image_dir = Path(temporary_per_image_workspace.name)
        per_image_dir.mkdir(parents=True, exist_ok=True)
        per_image_label = "temporary workspace, removed after merge"
    else:
        per_image_dir = results_dir / PER_IMAGE_DIR_NAME
        per_image_label = "not used"

    if input_plan.audio_files:
        temporary_audio_workspace = tempfile.TemporaryDirectory(prefix="mass_extract_audio_")
        temporary_audio_root = Path(temporary_audio_workspace.name)
        audio_staging_dir = temporary_audio_root / "staged_inputs"
        if args.keep_audio_output:
            audio_output_parent = results_dir / PER_AUDIO_DIR_NAME
            audio_output_parent.mkdir(parents=True, exist_ok=True)
            audio_output_dir = Path(
                tempfile.mkdtemp(
                    prefix=f"{input_plan.run_stem}_",
                    dir=audio_output_parent,
                )
            )
            audio_output_label = str(audio_output_dir)
        else:
            audio_output_dir = temporary_audio_root / "diarizer_output"
            audio_output_label = "temporary workspace, removed after merge"
    else:
        audio_staging_dir = results_dir / PER_AUDIO_DIR_NAME / "_unused"
        audio_output_dir = results_dir / PER_AUDIO_DIR_NAME
        audio_output_label = "not used"

    print("[START]")
    print(f"-> Mode: {input_plan.mode}")
    print(f"-> Case report: {input_plan.report_path}")
    print(f"-> Candidate images found: {len(input_plan.images)}")
    print(f"-> Candidate audio/video files found: {len(input_plan.audio_files)}")
    if input_plan.images:
        print(f"-> Viber script: {viber_script}")
        print(f"-> Facebook script: {facebook_script}")
        print(f"-> Email script: {email_script}")
    if input_plan.audio_files:
        print(f"-> Audio script: {audio_script}")
    print(f"-> Results root: {results_dir}")
    print(f"-> Extracted ZIP contents: {extract_root}")
    print(f"-> Per-image outputs: {per_image_label}")
    print(f"-> Per-audio outputs: {audio_output_label}")
    print(f"-> Output CSV: {output_path}")
    if keep_chat_csvs:
        print(f"-> Raw chat-only CSV: {raw_chat_path}")
        print(f"-> Polished chat-only CSV: {polished_chat_path}")
    else:
        print("-> Chat-only CSV retention: disabled (use --keep-chat-csvs to enable)")
    print(f"-> Manifest: {manifest_path or 'disabled'}")
    print(f"-> Classify mode: {args.classify_mode}")
    print(f"-> Force platform: {args.force_platform}")
    print(f"-> Emoji mode: {args.emoji_mode}")
    print(f"-> Keep debug: {keep_debug}")
    print(f"-> Keep per-image outputs: {keep_per_image}")
    print(f"-> Strict post-chat Message polish: {not args.no_chat_polish}")
    print("-> Chat mapping continuity: enabled per evidence folder; explicit current-image cues remain authoritative")
    print("-> Shared validated case context: retained for audio attribution")

    all_rows: List[Dict[str, str]] = []
    image_records: List[Dict[str, str]] = []
    audio_records: List[Dict[str, str]] = []
    audio_candidates: List[AudioCandidate] = []
    audio_dedupe_summary: Dict[str, Any] = {
        "enabled": not args.keep_duplicates,
        "candidate_recordings": 0,
        "canonical_recordings": 0,
        "duplicate_recordings_removed": 0,
        "exact_rows_removed": 0,
        "decisions": [],
    }
    deferred_emails: List[Tuple[Path, Path, Path, Dict[str, str]]] = []
    failure_reasons: List[str] = []
    audio_returncode = 0
    polish_result: Dict[str, Any] = {
        "ok": True, "accepted": 0, "rejected": 0, "error": "disabled",
    }
    used_output_stems: Set[str] = set()

    def conversation_key_for_image(image_path: Path, platform: str) -> str:
        """Build a run-local continuity key from platform and parent folder.

        Evidence packages usually keep consecutive screenshots of one chat in
        the same directory.  The prior remains soft, so a new explicit contact
        or self-identification in a later screenshot can still replace it.
        """
        return f"{platform}:{image_path.parent.resolve()}"

    for image_path in input_plan.images:
        print(f"\n[CLASSIFY] {image_path}")
        platform = classify_platform(
            image_path=image_path,
            model=args.model,
            mode=args.classify_mode,
            force_platform=args.force_platform,
        )
        print(f"-> Platform: {platform}")

        output_stem = unique_output_stem(image_path, used_output_stems)
        extractor_csv_path = per_image_dir / f"{output_stem}_extracted.csv"
        extractor_debug_dir = per_image_dir / f"{output_stem}_debug"

        record: Dict[str, str] = {
            "image": str(image_path),
            "platform": platform,
            "csv": "",
            "debug_dir": str(extractor_debug_dir) if keep_debug else "",
            "status": "skipped_unknown_or_non_chat",
            "reason": "",
            "rows": "0",
        }

        if platform not in {"facebook", "viber", "email"}:
            record["reason"] = "classifier returned non_chat/unknown"
            print("-> Skipping image because it is not recognized chat/email evidence.")
            image_records.append(record)
            continue

        # Classify all evidence in one pass, but execute email only after the
        # completed chat -> strict Message polish -> audio sequence.
        if platform == "email":
            record["status"] = "deferred_email"
            record["reason"] = "classified; scheduled after audio"
            deferred_emails.append(
                (image_path, extractor_csv_path, extractor_debug_dir, record)
            )
            image_records.append(record)
            print("-> Email classified and deferred until after chat polish and audio.")
            continue

        csv_path = run_extractor(
            platform=platform,
            image_path=image_path,
            report_path=input_plan.report_path,
            viber_script=viber_script,
            facebook_script=facebook_script,
            email_script=email_script,
            model=args.model,
            langs=args.langs,
            use_cpu=args.cpu,
            no_vision=args.no_vision,
            emoji_mode=args.emoji_mode,
            dump_ocr=args.dump_ocr,
            dump_draft=args.dump_draft,
            dump_side_map=args.dump_side_map,
            output_csv_path=extractor_csv_path,
            debug_dir_path=extractor_debug_dir,
            conversation_state_cache=conversation_state_cache,
            conversation_key=conversation_key_for_image(image_path, platform),
            extra_args=args.extra_extractor_arg,
            keep_debug=keep_debug,
        )

        if csv_path:
            rows = read_chat_csv(csv_path, source_image=image_path)
            print(f"-> Rows read: {len(rows)}")
            if rows:
                all_rows.extend(rows)
                record["csv"] = str(csv_path) if keep_per_image else ""
                record["status"] = "ok"
                record["reason"] = "processed"
                record["rows"] = str(len(rows))
            else:
                zero_row_kind = classify_zero_row_image(image_path, args.model)
                if zero_row_kind in {"profile", "other"}:
                    record["status"] = "skipped_non_conversation"
                    record["reason"] = (
                        f"zero-row verification classified image as {zero_row_kind}"
                    )
                    print(
                        f"-> No message bubbles found; classified as {zero_row_kind} "
                        "and skipped without extraction failure."
                    )
                else:
                    record["status"] = "empty"
                    record["reason"] = (
                        "extractor CSV contained no valid rows; "
                        f"zero-row verification={zero_row_kind}"
                    )
                    failure_reasons.append(
                        f"{image_path.name}: chat extractor returned zero rows"
                    )
        else:
            record["status"] = "failed"
            record["reason"] = "extractor failed or CSV output missing"
            failure_reasons.append(f"{image_path.name}: chat extractor failed")

        image_records.append(record)

    # ------------------------------------------------------------
    # STRICT ORDER: completed chats -> optional raw chat checkpoint ->
    # Message-only LLM polish -> optional polished checkpoint -> audio -> email.
    # ------------------------------------------------------------
    chat_rows = [row for row in all_rows if row.get("_source_kind") == "chat"]
    if chat_rows:
        if keep_chat_csvs:
            raw_count = write_merged_csv(
                rows=[dict(row) for row in chat_rows],
                output_path=raw_chat_path,
                dedupe=not args.keep_duplicates,
            )
            print(f"\n[INFO] Raw chat-only CSV written: {raw_chat_path} ({raw_count} rows)")

        if not args.no_chat_polish:
            polish_host = args.chat_polish_host or os.environ.get(
                "OLLAMA_HOST", "http://127.0.0.1:11434"
            )
            print(
                "[INFO] Running strict post-chat LLM polish: Message-only; "
                "row count/order and metadata are immutable."
            )
            polish_result = strict_polish_chat_messages(
                chat_rows,
                model=args.chat_polish_model or args.model,
                host=polish_host,
                batch_size=args.chat_polish_batch_size,
            )
            if not polish_result.get("ok"):
                reason = "strict chat Message polish failed: " + str(polish_result.get("error", "unknown"))
                failure_reasons.append(reason)
                print(f"[ERROR] {reason}", file=sys.stderr)
            print(
                "[INFO] Strict polish summary: "
                f"accepted={polish_result.get('accepted', 0)} "
                f"rejected={polish_result.get('rejected', 0)}"
            )

        repaired_identifiers = repair_chat_message_identifiers(chat_rows)
        print(
            "[INFO] Deterministic chat identifier repair: "
            f"repaired_rows={repaired_identifiers}"
        )

        repaired_participants, unresolved_labels = repair_invalid_chat_participants(
            chat_rows, input_plan.report_path,
        )
        if repaired_participants:
            print(
                f"[INFO] Deterministic participant guard repaired "
                f"{repaired_participants} leaked report-heading field(s)."
            )
        invalid_labels = invalid_chat_participant_labels(chat_rows)
        invalid_labels = sorted(set(invalid_labels) | set(unresolved_labels), key=str.casefold)
        if invalid_labels:
            reason = "invalid report/UI participant label(s): " + ", ".join(invalid_labels)
            failure_reasons.append(reason)
            print(f"[ERROR] {reason}", file=sys.stderr)

        if keep_chat_csvs:
            polished_count = write_merged_csv(
                rows=[dict(row) for row in chat_rows],
                output_path=polished_chat_path,
                dedupe=not args.keep_duplicates,
            )
            print(
                f"[INFO] Polished chat-only CSV written: {polished_chat_path} "
                f"({polished_count} rows)"
            )

    if input_plan.audio_files:
        csv_paths, audio_records, audio_returncode = run_audio_diarizer(
            audio_files=input_plan.audio_files,
            report_path=input_plan.report_path,
            audio_script=audio_script,
            output_dir=audio_output_dir,
            staging_dir=audio_staging_dir,
            case_context_cache=case_context_cache,
            args=args,
        )
        successful_audio_records = [
            record for record in audio_records if record.get("status") == "ok"
        ]
        report_text_for_audio_dates = read_case_report_text(input_plan.report_path)
        explicit_audio_date = (
            resolve_audio_base_date([], input_plan.report_path, explicit_date=args.audio_date)
            if args.audio_date else None
        )
        for audio_order, (csv_path, record) in enumerate(
            zip(csv_paths, successful_audio_records)
        ):
            source_audio = Path(record["audio"])
            undated_rows = read_audio_csv(
                csv_path,
                source_audio=source_audio,
                base_date=None,
                include_undated=True,
            )
            if explicit_audio_date is not None:
                audio_base_date = explicit_audio_date
                date_result = {
                    "ok": "True",
                    "date": audio_base_date.strftime("%d/%m/%Y"),
                    "confidence": "override",
                    "evidence": "--audio-date explicit override",
                    "error": "",
                }
            else:
                date_result = infer_audio_date_with_llm(
                    source_audio=source_audio,
                    audio_rows=undated_rows,
                    chat_rows=chat_rows,
                    report_text=report_text_for_audio_dates,
                    model=args.audio_ollama_model or args.model,
                    host=(
                        args.audio_ollama_host
                        or args.chat_polish_host
                        or os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
                    ),
                )
                audio_base_date = (
                    datetime.strptime(date_result["date"], "%d/%m/%Y")
                    if date_result.get("ok") == "True" else None
                )

            record["inferred_date"] = date_result.get("date", "")
            record["date_confidence"] = date_result.get("confidence", "")
            record["date_evidence"] = date_result.get("evidence", "")
            if audio_base_date is None:
                record["status"] = "date_failed"
                record["reason"] = (
                    "no defensible per-file audio date: "
                    + date_result.get("error", "unknown")
                )
                print(
                    f"[ERROR] Audio date inference failed for {source_audio.name}: "
                    f"{date_result.get('error', 'unknown')}",
                    file=sys.stderr,
                )
                continue

            print(
                f"-> Audio date for {source_audio.name}: "
                f"{audio_base_date.strftime('%d/%m/%Y')} "
                f"({date_result.get('confidence', '')})"
            )
            rows = read_audio_csv(
                csv_path, source_audio=source_audio, base_date=audio_base_date,
            )
            print(f"-> Audio rows read from {source_audio.name}: {len(rows)}")
            if rows:
                record["rows"] = str(len(rows))
                audio_hash = sha256_file(source_audio)
                record["sha256"] = audio_hash
                audio_candidates.append(AudioCandidate(
                    rows=rows,
                    record=record,
                    source_audio=source_audio,
                    source_csv=csv_path,
                    order=audio_order,
                    sha256=audio_hash,
                ))
            else:
                record["status"] = "empty"
                record["reason"] = "audio CSV contained no mergeable dated rows"

        selected_audio_rows, audio_dedupe_summary = deduplicate_audio_candidates(
            audio_candidates,
            enabled=not args.keep_duplicates,
        )
        all_rows.extend(selected_audio_rows)
        print(
            "[INFO] Audio deduplication: "
            f"candidates={audio_dedupe_summary['candidate_recordings']} "
            f"canonical={audio_dedupe_summary['canonical_recordings']} "
            f"recordings_removed={audio_dedupe_summary['duplicate_recordings_removed']} "
            f"exact_rows_removed={audio_dedupe_summary['exact_rows_removed']}"
        )

        failed_audio_count = sum(
            1 for record in audio_records
            if record.get("status") not in {"ok", "duplicate"}
        )
        if audio_returncode != 0 or failed_audio_count:
            reason = (
                f"audio diarization/attribution incomplete: "
                f"{failed_audio_count}/{len(input_plan.audio_files)} file(s) failed"
            )
            failure_reasons.append(reason)
            print(f"[ERROR] {reason}", file=sys.stderr)

    # Email extraction is deliberately last. It cannot influence chat polish
    # or audio date/context inference.
    for image_path, extractor_csv_path, extractor_debug_dir, record in deferred_emails:
        csv_path = run_extractor(
            platform="email",
            image_path=image_path,
            report_path=input_plan.report_path,
            viber_script=viber_script,
            facebook_script=facebook_script,
            email_script=email_script,
            model=args.model,
            langs=args.langs,
            use_cpu=args.cpu,
            no_vision=args.no_vision,
            emoji_mode=args.emoji_mode,
            dump_ocr=args.dump_ocr,
            dump_draft=args.dump_draft,
            dump_side_map=args.dump_side_map,
            output_csv_path=extractor_csv_path,
            debug_dir_path=extractor_debug_dir,
            conversation_state_cache=conversation_state_cache,
            conversation_key=conversation_key_for_image(image_path, "email"),
            extra_args=args.extra_email_arg,
            keep_debug=keep_debug,
        )
        if csv_path:
            rows = read_chat_csv(csv_path, source_image=image_path)
            for row in rows:
                row["Estimated_Timestamp"] = "True"
                row["_source_kind"] = "email"
            print(f"-> Email rows read: {len(rows)}")
            if rows:
                all_rows.extend(rows)
                record["csv"] = str(csv_path) if keep_per_image else ""
                record["status"] = "ok"
                record["reason"] = "processed after audio"
                record["rows"] = str(len(rows))
            else:
                record["status"] = "empty"
                record["reason"] = "email CSV contained no valid rows"
                failure_reasons.append(f"{image_path.name}: email extractor returned zero rows")
        else:
            record["status"] = "failed"
            record["reason"] = "email extractor failed or CSV output missing"
            failure_reasons.append(f"{image_path.name}: email extractor failed")


    if not all_rows:
        print("[ERROR] No rows extracted. Merged CSV was not created.", file=sys.stderr)
        if manifest_path:
            write_run_manifest(
                manifest_path,
                input_plan,
                image_records,
                audio_records,
                output_path,
                0,
                args,
                polish_result=polish_result,
                failure_reasons=failure_reasons or ["no rows extracted"],
                raw_chat_output=raw_chat_path if chat_rows and keep_chat_csvs else None,
                polished_chat_output=polished_chat_path if chat_rows and keep_chat_csvs else None,
                audio_deduplication=audio_dedupe_summary,
            )
            print(f"[INFO] Manifest saved to: {manifest_path}")
        if temporary_per_image_workspace is not None:
            temporary_per_image_workspace.cleanup()
        if temporary_audio_workspace is not None:
            temporary_audio_workspace.cleanup()
        if temporary_case_context_workspace is not None:
            temporary_case_context_workspace.cleanup()
        return 4

    rows_written = write_merged_csv(
        rows=all_rows,
        output_path=output_path,
        dedupe=not args.keep_duplicates,
    )

    partial_failure = bool(failure_reasons)

    if manifest_path:
        write_run_manifest(
            manifest_path,
            input_plan,
            image_records,
            audio_records,
            output_path,
            rows_written,
            args,
            polish_result=polish_result,
            failure_reasons=failure_reasons,
            raw_chat_output=raw_chat_path if chat_rows and keep_chat_csvs else None,
            polished_chat_output=polished_chat_path if chat_rows and keep_chat_csvs else None,
            audio_deduplication=audio_dedupe_summary,
        )

    print("\n[PARTIAL FAILURE]" if partial_failure else "\n[SUCCESS]")
    print(f"CSV saved to: {output_path}")
    print(f"Rows collected before final dedupe: {len(all_rows)}")
    print(f"Rows written: {rows_written}")
    if partial_failure:
        print("One or more required stages were incomplete:", file=sys.stderr)
        for reason in dict.fromkeys(failure_reasons):
            print(f"  - {reason}", file=sys.stderr)
    if manifest_path:
        print(f"Manifest saved to: {manifest_path}")
    if keep_per_image:
        print(f"Per-image files saved under: {per_image_dir}")
    elif input_plan.images:
        print("Per-image files were temporary and have been removed.")
    if args.keep_audio_output:
        print(f"Per-audio files saved under: {audio_output_dir}")
    elif input_plan.audio_files:
        print("Per-audio files were temporary and have been removed.")
    if chat_rows and keep_chat_csvs:
        print(f"Raw chat-only CSV saved to: {raw_chat_path}")
        print(f"Polished chat-only CSV saved to: {polished_chat_path}")
    for source in input_plan.evidence_sources:
        if source.extracted:
            print(f"Extracted ZIP saved under: {source.root_path}")

    if temporary_per_image_workspace is not None:
        temporary_per_image_workspace.cleanup()
    if temporary_audio_workspace is not None:
        temporary_audio_workspace.cleanup()
    if temporary_case_context_workspace is not None:
        temporary_case_context_workspace.cleanup()

    return 5 if partial_failure else 0

if __name__ == "__main__":
    raise SystemExit(main())