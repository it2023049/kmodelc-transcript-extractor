"""Two-stage MOSS and LLM pipeline for forensic audio diarization and sender/receiver attribution."""

import argparse
import inspect
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from audio_utils import (
    ConversationTurn,
    SpeakerSegment,
    build_anonymous_speaker_turns,
    build_hotwords,
    clean_message,
    discover_audio_files,
    extract_actor_candidates,
    extract_primary_report_party,
    canonical_participant_name,
    match_participants_in_filename,
    self_identified_participant,
    directly_addressed_participants,
    participant_receiver_evidence,
    third_party_mention_context,
    unique_names,
    infer_receiver,
    ordered_speakers,
    parse_participants,
    parse_speaker_map_file,
    read_case_report,
    safe_stem,
    strip_parenthetical_alias,
    segments_to_dicts,
    speaker_map_template,
    turns_to_dicts,
    turns_to_llm_items,
    write_conversation_csv,
    write_diarized_txt,
    write_json,
    write_speaker_turns_txt,
)

DEFAULT_MODEL_ID = "OpenMOSS-Team/MOSS-Transcribe-Diarize"
DEFAULT_MOSS_PROMPT = (
    "Transcribe the audio. For each segment, start with the timestamp and speaker ID "
    "([S01], [S02], [S03], ...), then the spoken text, and end with the segment timestamp."
)


def normalize_http_base_url(value: str, default: str = "http://127.0.0.1:11434") -> str:
    """Return a requests-compatible HTTP(S) base URL.

    HPC job scripts commonly export ``OLLAMA_HOST=127.0.0.1:<port>``.  The
    Ollama Python client accepts that form, but ``requests`` does not, so add
    the scheme deterministically before Stage-2 attribution starts.
    """
    text = str(value or default).strip().rstrip("/")
    if not re.match(r"^https?://", text, flags=re.I):
        text = "http://" + text
    return text



# =============================================================================
# COMMAND-LINE CONFIGURATION
# =============================================================================
def parse_args() -> argparse.Namespace:
    """Define the command-line interface for transcription, diarization, attribution, debugging,

    and backend selection.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Two-stage MOSS + LLM pipeline: first create an anonymous speaker-turn map, "
            "then let an LLM read the case report and assign Sender/Receiver."
        )
    )
    parser.add_argument("inputs", nargs="+", help="Audio/video file(s), folders, globs, or zip files.")
    parser.add_argument("--case-report", required=True, help="Victim/case report PDF/TXT used by the LLM for attribution.")
    parser.add_argument("--case-context-cache", default=None, help="Optional shared validated case-context JSON cache.")
    parser.add_argument("--output-dir", default="audio_diarized_moss_llm", help="Output directory.")

    parser.add_argument("--backend", choices=["moss-local", "moss-api"], default="moss-local", help="MOSS inference backend.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="MOSS model id or local model path.")
    parser.add_argument("--language", default="en", help="Language hint for the MOSS prompt.")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="Device for local MOSS inference.")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto", help="Torch dtype for local MOSS inference.")
    parser.add_argument("--max-new-tokens", type=int, default=65536, help="Maximum MOSS generation tokens. Raise for long calls.")
    parser.add_argument("--hotwords", default=None, help="Extra comma-separated hotwords for MOSS transcription only.")
    parser.add_argument("--max-hotwords", type=int, default=100, help="Maximum hotwords included in the MOSS prompt.")
    parser.add_argument("--prompt", default=None, help="Optional custom MOSS transcription prompt. Hotwords are appended automatically.")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000/v1/audio/transcriptions", help="OpenAI-compatible audio transcription endpoint for --backend moss-api.")
    parser.add_argument("--api-timeout", type=int, default=3600, help="HTTP timeout seconds for --backend moss-api.")

    parser.add_argument("--llm-backend", choices=["ollama", "none"], default="ollama", help="LLM attribution backend. Use 'none' to output only anonymous speaker turns.")
    parser.add_argument("--ollama-model", default="gemma3:12b", help="Ollama model used to infer Sender/Receiver from case report + speaker-turn TXT.")
    parser.add_argument("--ollama-host", default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"), help="Ollama host URL.")
    parser.add_argument("--llm-timeout", type=int, default=900, help="HTTP timeout seconds for Ollama attribution.")
    parser.add_argument("--llm-temperature", type=float, default=0.0, help="LLM attribution temperature.")
    parser.add_argument("--max-case-report-chars", type=int, default=45000, help="Maximum case-report characters sent to the LLM.")
    parser.add_argument("--max-transcript-chars", type=int, default=45000, help="Maximum speaker-turn transcript characters sent to the LLM.")

    parser.add_argument("--speaker-map", default=None, help="Optional manual JSON mapping like {'S01':'Name A','S02':'Name B'}. Overrides LLM attribution.")
    parser.add_argument("--participants", default=None, help="Optional debug override limiting participant names. Not required in normal runs.")
    parser.add_argument("--map-speakers-by-order", action="store_true", help="Manual fallback: map first appearing speaker to first participant, second to second, etc.")
    parser.add_argument("--no-merge-turns", action="store_true", help="Do not merge consecutive MOSS segments from the same anonymous speaker before LLM attribution.")
    parser.add_argument("--debug-json", action="store_true", help="Write raw/debug JSON files, including the LLM response.")
    parser.add_argument("--check-deps", action="store_true", help="Check required backend dependencies and exit.")
    args = parser.parse_args()
    args.ollama_host = normalize_http_base_url(args.ollama_host)
    return args



# =============================================================================
# MOSS TRANSCRIPTION BACKENDS
# =============================================================================
class MossLocalRunner:
    """Lazy-loading wrapper around the local MOSS model and processor. Heavy model initialization

    occurs only on the allocated compute node when transcription starts.
    """
    def __init__(self, model_id: str, device_arg: str = "auto", dtype_arg: str = "auto") -> None:
        """Store model, device, and dtype preferences without loading GPU resources."""
        self.model_id = model_id
        self.device_arg = device_arg
        self.dtype_arg = dtype_arg
        self.model = None
        self.processor = None
        self.device = None
        self.dtype = None
        self._build_transcription_messages = None
        self._generate_transcription = None
        self._resolve_device = None

    def load(self) -> None:
        """Import MOSS dependencies, resolve the allocated device, and load the local model once."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor
            from moss_transcribe_diarize.inference_utils import (
                build_transcription_messages,
                generate_transcription,
                resolve_device,
            )
        except Exception as exc:
            raise RuntimeError(
                "MOSS local backend dependencies are missing. Install them with:\n"
                "  git clone https://github.com/OpenMOSS/MOSS-Transcribe-Diarize.git\n"
                "  cd MOSS-Transcribe-Diarize\n"
                "  python -m pip install -e .\n"
                "  python -m pip install transformers accelerate soundfile\n"
                "For CUDA, install a Torch build compatible with your HPC CUDA libraries."
            ) from exc

        self._build_transcription_messages = build_transcription_messages
        self._generate_transcription = generate_transcription
        self._resolve_device = resolve_device

        if self.device_arg == "auto":
            self.device = resolve_device("auto")
        elif self.device_arg == "cuda":
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        if self.dtype_arg == "auto":
            self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        elif self.dtype_arg == "bfloat16":
            self.dtype = torch.bfloat16
        elif self.dtype_arg == "float16":
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

        print(f"[INFO] Loading MOSS local model: {self.model_id}")
        print(f"[INFO] Local MOSS device: {self.device}; dtype: {self.dtype}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            dtype="auto",
        ).to(dtype=self.dtype).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )

    def transcribe(self, audio_path: Path, prompt: str, max_new_tokens: int) -> Dict[str, Any]:
        """Run deterministic local MOSS transcription for one audio file."""
        if self.model is None or self.processor is None:
            self.load()
        assert self._build_transcription_messages is not None
        assert self._generate_transcription is not None

        messages = call_with_supported_kwargs(
            self._build_transcription_messages,
            str(audio_path),
            prompt=prompt,
            instruction=prompt,
            language=None,
        )
        result = call_with_supported_kwargs(
            self._generate_transcription,
            self.model,
            self.processor,
            messages,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            device=self.device,
            dtype=self.dtype,
        )
        if isinstance(result, dict):
            text = str(result.get("text", ""))
            return {"text": text, "raw": result}
        return {"text": str(result), "raw": {"result": str(result)}}


class MossApiRunner:
    """HTTP client for an OpenAI-compatible MOSS transcription service."""
    def __init__(self, api_url: str, model_id: str, timeout: int = 3600) -> None:
        """Store the API endpoint, model identifier, and request timeout."""
        self.api_url = api_url
        self.model_id = model_id
        self.timeout = timeout

    def transcribe(self, audio_path: Path, prompt: str, max_new_tokens: int) -> Dict[str, Any]:
        """Upload one media file to the remote MOSS endpoint and normalize its response."""
        try:
            import requests
        except Exception as exc:
            raise RuntimeError("MOSS API backend requires requests. Install with: python -m pip install requests") from exc
        data = {
            "model": self.model_id,
            "response_format": "verbose_json",
            "temperature": "0",
            "max_new_tokens": str(max_new_tokens),
            "prompt": prompt,
        }
        with audio_path.open("rb") as f:
            files = {"file": (audio_path.name, f, "application/octet-stream")}
            response = requests.post(self.api_url, data=data, files=files, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        text = extract_text_from_api_payload(payload)
        return {"text": text, "raw": payload}



# =============================================================================
# BACKEND-COMPATIBILITY HELPERS
# =============================================================================
def call_with_supported_kwargs(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Call a helper whose signature may change between MOSS releases."""
    try:
        signature = inspect.signature(fn)
    except Exception:
        return fn(*args, **{k: v for k, v in kwargs.items() if v is not None})
    supported = {}
    for key, value in kwargs.items():
        if value is not None and key in signature.parameters:
            supported[key] = value
    return fn(*args, **supported)


def extract_text_from_api_payload(payload: Any) -> str:
    """Normalize common OpenAI-compatible API response shapes into one transcript string."""
    if isinstance(payload, dict):
        if isinstance(payload.get("text"), str):
            return payload["text"]
        if isinstance(payload.get("transcript"), str):
            return payload["transcript"]
        if isinstance(payload.get("segments"), list):
            return segments_payload_to_moss_text(payload["segments"])
        if isinstance(payload.get("choices"), list) and payload["choices"]:
            first = payload["choices"][0]
            if isinstance(first, dict):
                message = first.get("message") or {}
                if isinstance(message, dict) and isinstance(message.get("content"), str):
                    return message["content"]
                if isinstance(first.get("text"), str):
                    return first["text"]
    return str(payload)


def segments_payload_to_moss_text(segments: Sequence[Any]) -> str:
    """Convert structured API segments into the bracketed MOSS text format consumed by the shared

    parser.
    """
    chunks: List[str] = []
    for index, seg in enumerate(segments, start=1):
        if not isinstance(seg, dict):
            continue
        start = seg.get("start") or seg.get("start_time") or 0.0
        end = seg.get("end") or seg.get("end_time") or start
        speaker = seg.get("speaker") or seg.get("speaker_label") or seg.get("label") or f"S{index:02d}"
        speaker = normalize_speaker_label(str(speaker))
        text = clean_message(str(seg.get("text") or seg.get("transcript") or ""))
        chunks.append(f"[{float_or_zero(start):.2f}][{speaker}]{text}[{float_or_zero(end):.2f}]")
    return "".join(chunks)



# =============================================================================
# MOSS TRANSCRIPT PARSING AND NORMALIZATION
# =============================================================================
def parse_moss_transcript(text: str) -> List[SpeakerSegment]:
    """Parse MOSS output into speaker segments.

    MOSS releases/backends may emit different but valid-looking formats:
      - [0.48][S01]Hello[1.66][2.00][S02]Hi[2.40]
      - [S01] Hello. [S01] More text. [S02] Reply.
      - S01: Hello

    The important forensic invariant is speaker-turn splitting. Therefore this
    parser treats any embedded [Sxx] or [SPEAKER_xx] label as a potential speaker
    boundary, strips speaker/timestamp tokens from the message text, and merges
    consecutive chunks from the same speaker.
    """
    text = text or ""

    # Prefer the parser shipped by the official OpenMOSS package.  It tracks
    # the canonical [start][Sxx]text[end] grammar and should remain compatible
    # when the upstream implementation evolves.
    try:
        from moss_transcribe_diarize import parse_transcript as official_parse_transcript

        official_segments: List[SpeakerSegment] = []
        for item in official_parse_transcript(text):
            if isinstance(item, dict):
                start = item.get("start", item.get("start_time"))
                end = item.get("end", item.get("end_time"))
                speaker = item.get("speaker", item.get("speaker_label", ""))
                body = item.get("text", item.get("transcript", ""))
            else:
                start = getattr(item, "start", getattr(item, "start_time", None))
                end = getattr(item, "end", getattr(item, "end_time", None))
                speaker = getattr(item, "speaker", getattr(item, "speaker_label", ""))
                body = getattr(item, "text", getattr(item, "transcript", ""))
            cleaned = clean_message(str(body or ""))
            if cleaned:
                append_segment(official_segments, SpeakerSegment(
                    start=float_or_none(start),
                    end=float_or_none(end),
                    speaker=normalize_speaker_label(str(speaker or "S00")),
                    text=cleaned,
                ))
        if official_segments:
            return official_segments
    except Exception:
        # The API backend may run without the local OpenMOSS Python package.
        # The strict canonical fallback below remains available in that case.
        pass

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"```(?:\w+)?", " ", text)
    text = text.replace("```", " ")

    embedded = parse_embedded_speaker_labels(text)
    if embedded:
        return embedded

    # Fallback parser for line-oriented variants: [S01] text, SPEAKER_00: text, etc.
    segments: List[SpeakerSegment] = []
    line_pattern = re.compile(
        r"^\s*(?:\[(?P<speaker1>S\d{1,3}|SPEAKER_\d{1,3})\]|(?P<speaker2>S\d{1,3}|SPEAKER_\d{1,3})\s*:|(?P<speaker3>SPEAKER[_\s-]?\d{1,3})\s*[:\-])\s*(?P<text>.+?)\s*$",
        re.I,
    )
    for line in text.splitlines():
        m = line_pattern.match(line)
        if not m:
            continue
        speaker = normalize_speaker_label(m.group("speaker1") or m.group("speaker2") or m.group("speaker3") or "S00")
        body = clean_moss_body(m.group("text") or "")
        if body:
            append_segment(segments, SpeakerSegment(start=None, end=None, speaker=speaker, text=body))
    return segments


def moss_timestamps_are_complete(segments: Sequence[SpeakerSegment]) -> bool:
    """Require real, monotonic MOSS timings for every retained segment."""
    if not segments:
        return False
    previous_start = -1.0
    for segment in segments:
        if segment.start is None or segment.end is None:
            return False
        if segment.start < 0 or segment.end < segment.start:
            return False
        if segment.start < previous_start:
            return False
        previous_start = segment.start
    return True


def parse_embedded_speaker_labels(text: str) -> List[SpeakerSegment]:
    """Parse transcripts that contain speaker labels embedded inside a continuous text stream."""
    label_pattern = re.compile(r"\[(?P<speaker>S\d{1,3}|SPEAKER_\d{1,3})\]", re.I)
    labels = list(label_pattern.finditer(text))
    if not labels:
        return []

    segments: List[SpeakerSegment] = []
    preamble = clean_moss_body(text[: labels[0].start()])

    for index, label in enumerate(labels):
        speaker = normalize_speaker_label(label.group("speaker"))
        body_start = label.end()
        body_end = labels[index + 1].start() if index + 1 < len(labels) else len(text)
        raw_body = text[body_start:body_end]

        # When MOSS emits a leading greeting before the first explicit label
        # (e.g. "Hello John. [S01] This is ..."), attach it to the first speaker.
        if index == 0 and preamble:
            raw_body = preamble + " " + raw_body

        body = clean_moss_body(raw_body)
        if not body:
            continue

        start = timestamp_immediately_before(text, label.start())
        end = trailing_timestamp(raw_body)
        append_segment(segments, SpeakerSegment(start=start, end=end, speaker=speaker, text=body))

    return segments


def append_segment(segments: List[SpeakerSegment], segment: SpeakerSegment) -> None:
    """Append a speaker segment while merging immediately adjacent chunks from the same anonymous

    speaker.
    """
    if segments and segments[-1].speaker == segment.speaker:
        segments[-1].text = clean_message(segments[-1].text + " " + segment.text)
        if segment.end is not None:
            segments[-1].end = segment.end
        return
    segments.append(segment)


def clean_moss_body(text: str) -> str:
    """Remove speaker labels, timestamps, code fences, and wrapper text from one MOSS transcript

    body.
    """
    text = text or ""
    text = re.sub(r"```(?:\w+)?", " ", text)
    text = text.replace("```", " ")
    text = re.sub(r"\[(?:S\d{1,3}|SPEAKER_\d{1,3})\]", " ", text, flags=re.I)
    text = re.sub(r"\[\d+(?:\.\d+)?\]", " ", text)
    text = re.sub(r"^\s*(?:transcript|output|result)\s*[:\-]\s*", " ", text, flags=re.I)
    return clean_message(text)


def timestamp_immediately_before(text: str, label_start: int) -> Optional[float]:
    """Read a bracketed timestamp placed directly before a speaker label, when present."""
    prefix = text[max(0, label_start - 48): label_start]
    m = re.search(r"\[(\d+(?:\.\d+)?)\]\s*$", prefix)
    if not m:
        return None
    return float_or_none(m.group(1))


def trailing_timestamp(text: str) -> Optional[float]:
    """Return the segment end timestamp from a canonical transcript fragment.

    Between two speaker labels the canonical stream contains ``[current_end]``
    followed by ``[next_start]``.  The first numeric marker is therefore the
    current segment's end; taking the last one incorrectly extends the segment
    to the next speaker's start.
    """
    matches = list(re.finditer(r"\[(\d+(?:\.\d+)?)\]", text or ""))
    if not matches:
        return None
    return float_or_none(matches[0].group(1))


def normalize_speaker_label(value: str) -> str:
    """Convert backend-specific speaker labels into the stable S01, S02, and related form."""
    value = value.strip().upper()
    m = re.match(r"SPEAKER[_\s-]?(\d+)", value)
    if m:
        return f"S{int(m.group(1)) + 1:02d}" if int(m.group(1)) < 10 else f"S{int(m.group(1)):02d}"
    m = re.match(r"S(\d+)", value)
    if m:
        return f"S{int(m.group(1)):02d}"
    return value or "S00"


def float_or_none(value: Any) -> Optional[float]:
    """Convert a value to float and return None when conversion is not possible."""
    try:
        return float(value)
    except Exception:
        return None


def float_or_zero(value: Any) -> float:
    """Convert a value to float and return zero when conversion is not possible."""
    try:
        return float(value)
    except Exception:
        return 0.0



# =============================================================================
# PROMPT, DEPENDENCY, AND RUNNER CONFIGURATION
# =============================================================================
def build_prompt(base_prompt: Optional[str], hotwords: Sequence[str], language: Optional[str]) -> str:
    """Build the final MOSS transcription prompt from the base instructions, language hint, and

    hotwords.
    """
    prompt = clean_message(base_prompt or DEFAULT_MOSS_PROMPT)
    if language:
        prompt += f" Language hint: {language}."
    if hotwords:
        prompt += " Hotword hints: " + ", ".join(hotwords) + "."
    return prompt


def check_deps(backend: str) -> None:
    """Import the dependencies required by the selected backend and fail early when any are

    unavailable.
    """
    if backend == "moss-api":
        import requests  # noqa: F401
        print("moss-api deps ok")
        return
    import torch  # noqa: F401
    import transformers  # noqa: F401
    import moss_transcribe_diarize  # noqa: F401
    print("moss-local deps ok")


def make_runner(args: argparse.Namespace) -> Any:
    """Instantiate the local-model or HTTP-API MOSS runner selected by the command-line arguments."""
    if args.backend == "moss-api":
        return MossApiRunner(args.api_url, args.model_id, args.api_timeout)
    return MossLocalRunner(args.model_id, args.device, args.dtype)




# =============================================================================
# CASE-GRAPH CONSTRUCTION AND CONTEXT VALIDATION
# =============================================================================
def truncate_middle(text: str, max_chars: int) -> str:
    """Limit long prompt material while preserving both the beginning and the end of the source

    text.
    """
    text = text or ""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    keep_head = max_chars // 2
    keep_tail = max_chars - keep_head
    return text[:keep_head] + "\n\n[...TRUNCATED FOR PROMPT LENGTH...]\n\n" + text[-keep_tail:]


def _compact_identity(value: str) -> str:
    """Build a punctuation-free identity key used only for exact and conservative report

    validation.
    """
    return "".join(ch for ch in clean_message(str(value or "")).casefold() if ch.isalnum())


def _case_context_allowed_names(case_context: Dict[str, Any], actor_candidates: Sequence[str]) -> List[str]:
    """Collect the validated human names that downstream attribution is allowed to emit."""
    names: List[str] = []
    for item in case_context.get("participants", []) if isinstance(case_context, dict) else []:
        if isinstance(item, dict):
            name = clean_message(str(item.get("name", ""))).strip(" ,;:.()[]{}\"'")
            if name:
                names.append(name)
    names.extend(actor_candidates)
    return unique_names(names)


def _validate_case_context(
    data: Dict[str, Any],
    case_report_text: str,
    actor_candidates: Sequence[str],
    primary_participant: Optional[str],
) -> Dict[str, Any]:
    """Validate a cached/generated case graph against names present in the report."""
    report_key = _compact_identity(case_report_text)
    names = list(unique_names(actor_candidates))
    raw_participants = data.get("participants", []) if isinstance(data, dict) else []
    if isinstance(raw_participants, list):
        for item in raw_participants:
            if not isinstance(item, dict):
                continue
            raw = clean_message(str(item.get("name", ""))).strip(" ,;:.()[]{}\"'")
            # Never accept combined identities as one participant.
            for piece in re.split(r"\s*(?:/|;|,|\band\b|\&)\s*", raw, flags=re.I):
                piece = clean_message(piece).strip(" ,;:.()[]{}\"'")
                key = _compact_identity(piece)
                if key and len(piece.split()) >= 2 and key in report_key:
                    names.append(piece)
    if primary_participant:
        names.insert(0, primary_participant)
    names = unique_names(names)

    participants: List[Dict[str, Any]] = []
    by_key: Dict[str, Dict[str, Any]] = {}
    for name in names:
        key = _compact_identity(name)
        by_key[key] = {
            "name": name,
            "role": "other",
            "aliases": [],
            "contact_numbers": [],
            "channels": [],
        }
        participants.append(by_key[key])
    if isinstance(raw_participants, list):
        for item in raw_participants:
            if not isinstance(item, dict):
                continue
            canonical = canonical_participant_name(str(item.get("name", "")), names)
            if not canonical:
                continue
            rec = by_key[_compact_identity(canonical)]
            role = str(item.get("role", "other") or "other").strip().lower()
            if role in {"victim", "complainant", "suspect", "advisor", "associate", "witness", "other"}:
                rec["role"] = role
            for field in ("aliases", "contact_numbers", "channels"):
                values = item.get(field, [])
                if isinstance(values, list):
                    rec[field] = [clean_message(str(x)) for x in values if clean_message(str(x))][:20]

    primary = canonical_participant_name(str(data.get("primary_complainant", "")), names) if isinstance(data, dict) else ""
    if not primary and primary_participant:
        primary = canonical_participant_name(primary_participant, names)
    if primary:
        rec = by_key.get(_compact_identity(primary))
        if rec and rec["role"] == "other":
            rec["role"] = "victim"

    interactions: List[Dict[str, str]] = []
    seen_pairs = set()
    for item in data.get("interaction_pairs", []) if isinstance(data, dict) and isinstance(data.get("interaction_pairs"), list) else []:
        if not isinstance(item, dict):
            continue
        a = canonical_participant_name(str(item.get("party_a", "")), names)
        b = canonical_participant_name(str(item.get("party_b", "")), names)
        if not a or not b or a.casefold() == b.casefold():
            continue
        channel = clean_message(str(item.get("channel", "")))
        key = tuple(sorted((_compact_identity(a), _compact_identity(b)))) + (channel.casefold(),)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        interactions.append({
            "party_a": a,
            "party_b": b,
            "channel": channel,
            "relationship": clean_message(str(item.get("relationship", "")))[:300],
            "evidence": clean_message(str(item.get("evidence", "")))[:500],
        })

    timeline: List[Dict[str, Any]] = []
    for item in data.get("timeline", []) if isinstance(data, dict) and isinstance(data.get("timeline"), list) else []:
        if not isinstance(item, dict):
            continue
        actors = []
        for raw in item.get("actors", []) if isinstance(item.get("actors"), list) else []:
            name = canonical_participant_name(str(raw), names)
            if name and name not in actors:
                actors.append(name)
        event = clean_message(str(item.get("event", "")))[:800]
        if event:
            timeline.append({
                "date_or_range": clean_message(str(item.get("date_or_range", "")))[:120],
                "actors": actors,
                "event": event,
            })

    return {
        "primary_complainant": primary,
        "participants": participants,
        "interaction_pairs": interactions,
        "timeline": timeline[:100],
        "key_entities": list(data.get("key_entities", []) or [])[:160] if isinstance(data, dict) else [],
        "case_summary": clean_message(str(data.get("case_summary", "")))[:4000] if isinstance(data, dict) else "",
    }


def build_case_analysis_prompt(case_report_text: str, actor_candidates: Sequence[str]) -> str:
    """Build the JSON-only prompt that converts the report into a validated forensic case graph."""
    report = truncate_middle(case_report_text, 70000)
    return f"""You are building a structured forensic case graph for later speaker attribution.

HIGH-CONFIDENCE NAME CANDIDATES (may be incomplete):
{json.dumps(list(actor_candidates), ensure_ascii=False)}

CASE REPORT:
---
{report}
---

Return JSON only:
{{
  "primary_complainant":"exact name or empty",
  "participants":[{{"name":"one exact human name", "role":"victim|complainant|suspect|advisor|associate|witness|other", "aliases":[], "contact_numbers":[], "channels":[]}}],
  "interaction_pairs":[{{"party_a":"exact name", "party_b":"exact name", "channel":"facebook|viber|audio|email|other", "relationship":"brief", "evidence":"brief report-grounded reason"}}],
  "timeline":[{{"date_or_range":"date/range or empty", "actors":["exact name"], "event":"brief factual event"}}],
  "key_entities":["exact evidence term"],
  "case_summary":"brief chronological plot"
}}

Rules:
1. Read the complete plot, including later actors and later fraud stages.
2. Include every human who actually sends or receives communications.
3. Never combine two names into one participant with '/', '&', commas, or 'and'.
4. Distinguish a communicator from a third party merely mentioned in a message.
5. Use exact names from the report and do not invent facts.
6. Interaction pairs must reflect who actually communicated with whom and by which channel.
7. Every participant, primary_complainant, party_a, party_b, and actors value must be one person's name only. Never put a document heading, section title, offence label, role, channel, organization, or explanatory phrase in a person-name field.
"""


def load_or_build_case_context(
    case_report_text: str,
    actor_candidates: Sequence[str],
    primary_participant: Optional[str],
    model: str,
    host: str,
    timeout: int,
    temperature: float,
    cache_path: Optional[str],
) -> Dict[str, Any]:
    """Load a cached case graph when available or generate, validate, and atomically cache a new

    graph.
    """
    path = Path(cache_path) if cache_path else None
    if path and path.exists():
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(cached, dict):
                return _validate_case_context(cached, case_report_text, actor_candidates, primary_participant)
        except Exception:
            pass

    prompt = build_case_analysis_prompt(case_report_text, actor_candidates)
    try:
        generated = run_ollama_attribution(prompt, model, host, timeout, temperature)
        generated.pop("_raw_response", None)
    except Exception:
        generated = {}
    context = _validate_case_context(generated, case_report_text, actor_candidates, primary_participant)
    if path:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(path)
        except Exception:
            pass
    return context



# =============================================================================
# CASE-STAGE AND RECEIVER EVIDENCE SCORING
# =============================================================================
def _token_set(text: str) -> set:
    """Return informative lowercase tokens for case-stage matching.

    Very common conversational words are removed so phrases such as ``my love``
    do not dominate receiver inference.  The remaining lexical overlap is used
    only as supporting evidence against the validated interaction/timeline graph.
    """
    stopwords = {
        "about", "after", "again", "always", "because", "before", "could",
        "from", "good", "have", "hello", "here", "just", "love", "more",
        "please", "really", "should", "thank", "thanks", "that", "their",
        "there", "these", "they", "this", "through", "today", "very",
        "want", "what", "when", "where", "which", "with", "would", "your",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9]+", str(text or "").casefold())
        if len(token) >= 3 and token not in stopwords
    }


def rank_context_counterparts(
    sender: str,
    message: str,
    case_context: Dict[str, Any],
    stage_hint: str = "",
    channel_hint: str = "",
) -> List[Dict[str, Any]]:
    """Rank possible recipients using validated graph and stage evidence.

    Interaction membership alone receives only a small score.  Timeline/stage
    overlap, channel compatibility, and direct address are stronger.  A person
    mentioned as a third party is penalised unless the same utterance directly
    addresses them.  The result is diagnostic; callers decide how much margin
    is needed before using the top candidate.
    """
    sender = clean_message(sender)
    if not sender:
        return []

    msg_tokens = _token_set(message)
    stage_tokens = _token_set(stage_hint)
    channel_tokens = _token_set(channel_hint)
    candidates: Dict[str, Dict[str, Any]] = {}

    allowed_names = _case_context_allowed_names(case_context, [])

    def record_for(name: str) -> Dict[str, Any]:
        rec = candidates.setdefault(name, {
            "name": name,
            "score": 0.0,
            "overlap": 0,
            "timeline_hits": 0,
            "reasons": [],
        })
        return rec

    def add(name: str, score: float, reason: str, overlap: int = 0, timeline: bool = False) -> None:
        if not name or name.casefold() == sender.casefold():
            return
        rec = record_for(name)
        rec["score"] += score
        rec["overlap"] += overlap
        rec["timeline_hits"] += int(timeline)
        rec["reasons"].append(reason)

    # Direct recipient wording is independent of the report and is strongest.
    for candidate in allowed_names:
        evidence = participant_receiver_evidence(message, candidate, allowed_names)
        if evidence.get("direct_address"):
            add(candidate, 100.0, "explicit direct address")
        elif evidence.get("third_party_mention"):
            add(candidate, -20.0, "third-party mention penalty")

    for pair in case_context.get("interaction_pairs", []) if isinstance(case_context, dict) else []:
        if not isinstance(pair, dict):
            continue
        a, b = str(pair.get("party_a", "")), str(pair.get("party_b", ""))
        if a.casefold() == sender.casefold():
            other = b
        elif b.casefold() == sender.casefold():
            other = a
        else:
            continue

        pair_text = " ".join([
            str(pair.get("relationship", "")),
            str(pair.get("evidence", "")),
            str(pair.get("channel", "")),
        ])
        pair_tokens = _token_set(pair_text)
        overlap = len(msg_tokens & pair_tokens)
        stage_overlap = len(stage_tokens & pair_tokens)
        pair_channel_tokens = _token_set(str(pair.get("channel", "")))
        channel_overlap = len(channel_tokens & pair_channel_tokens)

        score = 0.5 + 1.4 * overlap + 1.2 * stage_overlap + 0.8 * channel_overlap
        add(other, score, "validated interaction edge", overlap + stage_overlap)

    for event in case_context.get("timeline", []) if isinstance(case_context, dict) else []:
        if not isinstance(event, dict):
            continue
        actors = [str(value) for value in event.get("actors", []) if str(value)] if isinstance(event.get("actors"), list) else []
        if not any(value.casefold() == sender.casefold() for value in actors):
            continue

        event_tokens = _token_set(str(event.get("event", "")))
        overlap = len(msg_tokens & event_tokens)
        stage_overlap = len(stage_tokens & event_tokens)
        if overlap + stage_overlap <= 0:
            continue
        for other in actors:
            if other.casefold() != sender.casefold():
                add(
                    other,
                    2.5 + 2.0 * overlap + 1.5 * stage_overlap,
                    "matching timeline stage",
                    overlap + stage_overlap,
                    timeline=True,
                )

    # Apply a final grammatical third-party penalty even when the candidate was
    # introduced through a report edge.  A direct address already dominates it.
    for name, rec in candidates.items():
        if third_party_mention_context(message, name):
            direct = participant_receiver_evidence(message, name, allowed_names).get("direct_address")
            if not direct:
                rec["score"] -= 15.0
                rec["reasons"].append("utterance refers to candidate as a third party")

    ranked = sorted(
        candidates.values(),
        key=lambda item: (
            -float(item["score"]),
            -int(item["timeline_hits"]),
            -int(item["overlap"]),
            str(item["name"]).casefold(),
        ),
    )
    for item in ranked:
        item["score"] = round(float(item["score"]), 3)
    return ranked


def infer_context_counterpart(
    sender: str,
    message: str,
    case_context: Dict[str, Any],
    stage_hint: str = "",
    channel_hint: str = "",
) -> str:
    """Return a context recipient only when score and margin are decisive."""
    ranked = rank_context_counterparts(
        sender,
        message,
        case_context,
        stage_hint=stage_hint,
        channel_hint=channel_hint,
    )
    if not ranked:
        return ""

    top = ranked[0]
    second_score = float(ranked[1]["score"]) if len(ranked) > 1 else 0.0
    direct = "explicit direct address" in top.get("reasons", [])
    timeline = int(top.get("timeline_hits", 0)) > 0
    minimum_score = 50.0 if direct else (4.0 if timeline else 4.5)
    minimum_margin = 0.0 if direct else (1.25 if timeline else 2.0)
    if float(top["score"]) >= minimum_score and float(top["score"]) >= second_score + minimum_margin:
        return str(top["name"])
    return ""


# =============================================================================
# CONSTRAINED LLM ATTRIBUTION
# =============================================================================
def build_llm_attribution_prompt(
    case_report_text: str,
    case_context: Dict[str, Any],
    anonymous_turns: Sequence[ConversationTurn],
    explicit_participants: Sequence[str],
    actor_candidates: Sequence[str],
    audio_path: Path,
    primary_participant: Optional[str],
    max_case_chars: int,
    max_transcript_chars: int,
) -> str:
    """Build the constrained attribution prompt for matching anonymous audio turns to exact case

    participants.
    """
    case_text = truncate_middle(case_report_text, max_case_chars)
    turn_lines = [
        f"TURN_{index:03d} | {turn.speaker}: {clean_message(turn.message)}"
        for index, turn in enumerate(anonymous_turns, start=1)
    ]
    transcript_text = truncate_middle("\n".join(turn_lines), max_transcript_chars)
    allowed_names = _case_context_allowed_names(case_context, explicit_participants or actor_candidates)
    filename_matches = match_participants_in_filename(audio_path, allowed_names)
    unique_speakers = ordered_speakers([
        SpeakerSegment(turn.start, turn.end, turn.speaker, turn.message) for turn in anonymous_turns
    ])

    return f"""You are assigning Sender and Receiver identities to a forensic audio transcript.
Read the complete case plot and match this recording to the correct communication stage.
Do not rewrite any transcript message.

AUDIO SOURCE PROVENANCE:
- filename: {audio_path.name}
- detected anonymous speakers: {json.dumps(unique_speakers)}
- number of detected speakers: {len(unique_speakers)}
- exact participant names explicitly present in filename: {json.dumps(filename_matches, ensure_ascii=False)}
- primary complainant from report: {primary_participant or "unknown"}

VALIDATED CASE GRAPH:
{json.dumps(case_context, ensure_ascii=False, indent=2)}

ALLOWED EXACT HUMAN NAMES:
{json.dumps(allowed_names, ensure_ascii=False)}

FULL CASE REPORT:
---
{case_text}
---

ANONYMOUS SPEAKER-TURN TRANSCRIPT:
---
{transcript_text}
---

Return valid JSON only:
{{
  "recording_type":"voice_note|two_party_call|multi_party_call|unknown",
  "matched_case_stage":"brief report-grounded stage",
  "participants":["exact sender/receiver names involved in this recording"],
  "speaker_map":{{"S01":"exact audible speaker name or Unknown"}},
  "turns":[
    {{"turn_id":1, "speaker":"S01", "sender":"exact name or Unknown", "receiver":"exact name or Unknown", "confidence":"high|medium|low", "evidence":"specific filename/transcript/case-plot evidence"}}
  ],
  "notes":"brief"
}}

Strict attribution rules:
1. The sender and receiver fields may contain only one exact human name from ALLOWED EXACT HUMAN NAMES, or the literal Unknown. Never put a document heading, section title, offence label, role, organization, channel, sentence, or combined identity in sender/receiver.
2. For a one-speaker voice note, the receiver is usually not audible. Infer the intended recipient independently from direct address, filename provenance, interaction_pairs, timeline, and the matched case stage.
3. The primary complainant/victim label does not imply that person is the sender or receiver of every recording. Never choose a recipient from report role alone.
4. A filename containing one exact participant name is strong provenance for the sender of a single-speaker clip, unless explicit self-identification contradicts it. If two exact names occur, they are strong pair provenance.
5. Self-identification is strongest sender evidence: "This is X", "My name is X", "I am X".
6. Vocative/direct address is strongest receiver evidence: a greeting plus a name, a name followed by a comma, or a direct thank-you addressed to a name.
7. A person merely mentioned as a third party is not automatically sender or receiver. "I spoke with X" normally means X is not the current receiver.
8. Attribute first-person facts to their owner: who performed an action, has a condition, made a payment, asks for help, gives instructions, refuses, answers, or reacts. Use adjacent turns and chronology.
9. Keep one stable identity per anonymous speaker label inside this file, unless the transcript clearly proves diarization label reuse.
10. Do not force the two most prominent report actors into every recording. Later-stage actors must be considered.
11. If the evidence genuinely cannot distinguish a person, use Unknown and explain why.
12. Do not change, split, merge, correct, summarize, or translate Message text.
"""


def run_ollama_attribution(
    prompt: str,
    model: str,
    host: str,
    timeout: int,
    temperature: float,
) -> Dict[str, Any]:
    """Submit one JSON-constrained attribution request to the configured Ollama server."""
    try:
        import requests
    except Exception as exc:
        raise RuntimeError("LLM attribution with Ollama requires requests. Install with: python -m pip install requests") from exc

    host = normalize_http_base_url(host)
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": temperature, "num_ctx": 65536},
    }
    response = requests.post(f"{host}/api/generate", json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    raw = data.get("response", "") if isinstance(data, dict) else str(data)
    parsed = parse_llm_json(str(raw))
    parsed["_raw_response"] = raw
    return parsed


def parse_llm_json(text: str) -> Dict[str, Any]:
    """Parse an LLM response as a JSON object, tolerating a surrounding Markdown code fence."""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("LLM did not return a JSON object.")
        data = json.loads(text[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("LLM JSON response must be an object.")
    return data


def normalize_llm_name(value: Any) -> str:
    """Normalize an LLM-produced name while preserving only explicit Unknown values and clean text."""
    value = clean_message(str(value or ""))
    if not value or value.casefold() in {"unknown", "none", "null", "n/a", "na"}:
        return "Unknown"
    # Keep exact LLM/case spelling but strip obvious structural punctuation.
    value = value.strip(" ,;:.()[]{}\"'")
    if not value:
        return "Unknown"
    return value


def llm_speaker_map(attribution: Dict[str, Any]) -> Dict[str, str]:
    """Extract and normalize the anonymous-speaker mapping returned by the LLM."""
    raw = attribution.get("speaker_map")
    out: Dict[str, str] = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            speaker = normalize_speaker_label(str(key))
            name = normalize_llm_name(value)
            if speaker:
                out[speaker] = name
    return out


def llm_participants(attribution: Dict[str, Any]) -> List[str]:
    """Extract a de-duplicated participant list from the LLM attribution response."""
    raw = attribution.get("participants")
    result: List[str] = []
    seen = set()
    if isinstance(raw, list):
        for item in raw:
            name = normalize_llm_name(item)
            if name == "Unknown":
                continue
            key = name.casefold()
            if key not in seen:
                seen.add(key)
                result.append(name)
    return result


def llm_turn_records(attribution: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """Index LLM turn-attribution records by their numeric turn identifier."""
    records: Dict[int, Dict[str, Any]] = {}
    raw = attribution.get("turns")
    if not isinstance(raw, list):
        return records
    for item in raw:
        if not isinstance(item, dict):
            continue
        turn_id_raw = item.get("turn_id") or item.get("id") or item.get("turn")
        if isinstance(turn_id_raw, str):
            m = re.search(r"\d+", turn_id_raw)
            if not m:
                continue
            turn_id = int(m.group(0))
        else:
            try:
                turn_id = int(turn_id_raw)
            except Exception:
                continue
        records[turn_id] = item
    return records



# =============================================================================
# DETERMINISTIC ATTRIBUTION VALIDATION
# =============================================================================
def _record_confidence(record: Dict[str, Any]) -> str:
    """Normalise one LLM turn confidence value."""
    value = str(record.get("confidence", "low")).strip().casefold()
    return value if value in {"high", "medium", "low"} else "low"


def _interaction_pair_supported(case_context: Dict[str, Any], a: str, b: str) -> bool:
    """Check whether the validated graph contains the unordered pair."""
    if not a or not b or a.casefold() == b.casefold():
        return False
    for pair in case_context.get("interaction_pairs", []) if isinstance(case_context, dict) else []:
        if not isinstance(pair, dict):
            continue
        left, right = str(pair.get("party_a", "")), str(pair.get("party_b", ""))
        if {left.casefold(), right.casefold()} == {a.casefold(), b.casefold()}:
            return True
    return False


def _select_recording_counterpart(
    sender: str,
    anonymous_turns: Sequence[ConversationTurn],
    turn_records: Dict[int, Dict[str, Any]],
    allowed_names: Sequence[str],
    audio_path: Path,
    case_context: Dict[str, Any],
    matched_stage: str,
) -> Tuple[str, Dict[str, Any]]:
    """Choose one stable non-audible recipient for a single-speaker recording.

    Direct address and two-name filename provenance are deterministic.  LLM
    receiver votes and case-stage rankings are combined conservatively.  The
    primary reporting party receives no automatic bonus.
    """
    scores: Dict[str, float] = {
        name: 0.0 for name in allowed_names if name.casefold() != sender.casefold()
    }
    reasons: Dict[str, List[str]] = {name: [] for name in scores}
    full_text = " ".join(clean_message(turn.message) for turn in anonymous_turns)

    # Direct address anywhere in the clip is strongest recipient evidence.
    direct = directly_addressed_participants(full_text, allowed_names)
    for name in direct:
        if name in scores:
            scores[name] += 100.0
            reasons[name].append("direct address in recording")

    # A filename carrying both sender and one other exact participant is strong
    # pair provenance and does not depend on the report's primary role.
    filename_matches = match_participants_in_filename(audio_path, allowed_names)
    other_filename = [name for name in filename_matches if name.casefold() != sender.casefold()]
    if len(other_filename) == 1 and other_filename[0] in scores:
        scores[other_filename[0]] += 30.0
        reasons[other_filename[0]].append("two-party filename provenance")

    # Aggregate only medium/high turn-level receiver votes.  A third-party
    # grammatical mention cannot support the same candidate on that turn.
    for index, turn in enumerate(anonymous_turns, start=1):
        record = turn_records.get(index, {})
        candidate = canonical_participant_name(str(record.get("receiver", "")), allowed_names)
        confidence = _record_confidence(record)
        if not candidate or candidate not in scores or confidence == "low":
            continue
        if third_party_mention_context(turn.message, candidate):
            continue
        weight = 5.0 if confidence == "high" else 2.5
        scores[candidate] += weight
        reasons[candidate].append(f"{confidence} LLM receiver vote on turn {index}")

    # Rank the entire recording against the matched case stage.  This avoids a
    # fragile per-turn choice based on one short phrase.
    ranked = rank_context_counterparts(
        sender,
        full_text,
        case_context,
        stage_hint=matched_stage,
        channel_hint=audio_path.name,
    )
    for item in ranked:
        name = str(item.get("name", ""))
        if name not in scores:
            continue
        context_score = min(12.0, max(0.0, float(item.get("score", 0.0))))
        scores[name] += context_score
        reasons[name].append(f"case-stage counterpart score {context_score:.2f}")

    ranked_scores = sorted(scores.items(), key=lambda item: (-item[1], item[0].casefold()))
    if not ranked_scores:
        return "", {"scores": {}, "reasons": {}}
    top_name, top_score = ranked_scores[0]
    second_score = ranked_scores[1][1] if len(ranked_scores) > 1 else 0.0

    direct_top = "direct address in recording" in reasons.get(top_name, [])
    filename_top = "two-party filename provenance" in reasons.get(top_name, [])
    minimum = 50.0 if direct_top else (20.0 if filename_top else 7.0)
    margin = 0.0 if direct_top else (4.0 if filename_top else 2.5)
    selected = top_name if top_score >= minimum and top_score >= second_score + margin else ""
    return selected, {
        "selected": selected or None,
        "scores": {name: round(score, 3) for name, score in ranked_scores},
        "reasons": reasons,
        "matched_stage": matched_stage,
        "filename_matches": filename_matches,
    }


def build_final_turns_from_attribution(
    anonymous_turns: Sequence[ConversationTurn],
    attribution: Dict[str, Any],
    allowed_names: Sequence[str],
    audio_path: Path,
    case_context: Dict[str, Any],
    primary_participant: Optional[str],
) -> Tuple[List[ConversationTurn], Dict[str, str], List[str]]:
    """Build final turns with stable speaker identity and conservative receivers."""
    raw_map = llm_speaker_map(attribution)
    turn_records = llm_turn_records(attribution)
    speakers: List[str] = []
    for turn in anonymous_turns:
        if turn.speaker not in speakers:
            speakers.append(turn.speaker)
    single_speaker = len(speakers) == 1
    filename_matches = match_participants_in_filename(audio_path, allowed_names)
    filename_sender = filename_matches[0] if single_speaker and len(filename_matches) == 1 else ""

    speaker_map: Dict[str, str] = {}
    for speaker, value in raw_map.items():
        canonical = canonical_participant_name(value, allowed_names)
        speaker_map[speaker] = canonical or "Unknown"

    # Establish one stable sender identity for each anonymous speaker label.
    for speaker in speakers:
        speaker_turns = [turn for turn in anonymous_turns if turn.speaker == speaker]
        explicit_ids = unique_names([
            self_identified_participant(turn.message, allowed_names) or "" for turn in speaker_turns
        ])
        explicit_ids = [value for value in explicit_ids if value]
        if len(explicit_ids) == 1:
            speaker_map[speaker] = explicit_ids[0]
        elif single_speaker and filename_sender:
            speaker_map[speaker] = filename_sender
        else:
            record_senders: List[str] = []
            for index, turn in enumerate(anonymous_turns, start=1):
                if turn.speaker != speaker:
                    continue
                record = turn_records.get(index, {})
                if _record_confidence(record) == "low":
                    continue
                canonical = canonical_participant_name(str(record.get("sender", "")), allowed_names)
                if canonical:
                    record_senders.append(canonical)
            unique_record_senders = unique_names(record_senders)
            if speaker_map.get(speaker) in {None, "", "Unknown"}:
                speaker_map[speaker] = unique_record_senders[0] if len(unique_record_senders) == 1 else "Unknown"

    mapped_people = unique_names([value for value in speaker_map.values() if value != "Unknown"])
    participants = unique_names(
        [canonical_participant_name(value, allowed_names) for value in llm_participants(attribution)]
        + mapped_people
    )
    participants = [value for value in participants if value]

    matched_stage = clean_message(str(attribution.get("matched_case_stage", "")))
    recording_counterpart = ""
    recording_counterpart_debug: Dict[str, Any] = {}
    if single_speaker and speakers:
        stable_sender = speaker_map.get(speakers[0], "Unknown")
        if stable_sender != "Unknown":
            recording_counterpart, recording_counterpart_debug = _select_recording_counterpart(
                stable_sender,
                anonymous_turns,
                turn_records,
                allowed_names,
                audio_path,
                case_context,
                matched_stage,
            )

    final: List[ConversationTurn] = []
    validation_notes: List[Dict[str, Any]] = []
    for index, turn in enumerate(anonymous_turns, start=1):
        record = turn_records.get(index, {})
        record_confidence = _record_confidence(record)
        sender = speaker_map.get(turn.speaker, "Unknown") or "Unknown"

        explicit_sender = self_identified_participant(turn.message, allowed_names)
        if explicit_sender:
            sender = explicit_sender
            speaker_map[turn.speaker] = sender

        # If the stable sender is still unknown, a medium/high per-turn record
        # may identify it, but low-confidence guesses are ignored.
        if sender == "Unknown" and record_confidence in {"high", "medium"}:
            record_sender = canonical_participant_name(str(record.get("sender", "")), allowed_names)
            sender = record_sender or "Unknown"

        direct_receivers = [
            value for value in directly_addressed_participants(turn.message, allowed_names)
            if sender == "Unknown" or value.casefold() != sender.casefold()
        ]
        record_receiver = canonical_participant_name(str(record.get("receiver", "")), allowed_names)
        if record_receiver and third_party_mention_context(turn.message, record_receiver) and record_receiver not in direct_receivers:
            record_receiver = ""

        context_rank = rank_context_counterparts(
            sender,
            turn.message,
            case_context,
            stage_hint=matched_stage,
            channel_hint=audio_path.name,
        ) if sender != "Unknown" else []
        context_receiver = infer_context_counterpart(
            sender,
            turn.message,
            case_context,
            stage_hint=matched_stage,
            channel_hint=audio_path.name,
        ) if sender != "Unknown" else ""

        receiver = ""
        receiver_basis = ""
        if len(direct_receivers) == 1:
            receiver = direct_receivers[0]
            receiver_basis = "direct_address"
        elif single_speaker and recording_counterpart:
            receiver = recording_counterpart
            receiver_basis = "recording_level_consensus"
        elif record_receiver and context_receiver and record_receiver.casefold() == context_receiver.casefold():
            receiver = record_receiver
            receiver_basis = "llm_context_agreement"
        elif record_receiver and record_confidence == "high" and _interaction_pair_supported(case_context, sender, record_receiver):
            receiver = record_receiver
            receiver_basis = "high_confidence_llm_supported_pair"
        elif context_receiver:
            receiver = context_receiver
            receiver_basis = "decisive_context_margin"
        elif record_receiver and record_confidence in {"high", "medium"}:
            # Medium/high record output is a fallback only when it is not a
            # third-party mention and does not conflict with deterministic data.
            receiver = record_receiver
            receiver_basis = "validated_llm_fallback"

        if receiver and sender != "Unknown" and receiver.casefold() == sender.casefold():
            receiver = ""
            receiver_basis = ""

        if not receiver and len(mapped_people) == 2 and sender in mapped_people:
            receiver = mapped_people[1] if mapped_people[0] == sender else mapped_people[0]
            receiver_basis = "other_audible_mapped_speaker"

        if not receiver and primary_participant and sender != "Unknown" and sender.casefold() != primary_participant.casefold():
            # Primary-party fallback is allowed only for a validated pair and
            # never outranks a competing receiver candidate.
            if _interaction_pair_supported(case_context, sender, primary_participant):
                receiver = primary_participant
                receiver_basis = "primary_party_supported_pair_fallback"

        if not receiver or (sender != "Unknown" and receiver.casefold() == sender.casefold()):
            receiver = "Unknown"
            receiver_basis = receiver_basis or "insufficient_evidence"

        final.append(ConversationTurn(
            sender=sender,
            receiver=receiver,
            message=clean_message(turn.message),
            speaker=turn.speaker,
            start=turn.start,
            end=turn.end,
        ))
        validation_notes.append({
            "turn_id": index,
            "speaker": turn.speaker,
            "sender": sender,
            "receiver": receiver,
            "receiver_basis": receiver_basis,
            "record_confidence": record_confidence,
            "filename_sender_hint": filename_sender or None,
            "direct_receivers": direct_receivers,
            "record_receiver": record_receiver or None,
            "context_receiver": context_receiver or None,
            "context_ranking": context_rank[:4],
            "self_identified_sender": explicit_sender,
        })

    attribution["_deterministic_validation"] = {
        "audio_filename": audio_path.name,
        "allowed_names": list(allowed_names),
        "filename_matches": filename_matches,
        "single_speaker": single_speaker,
        "matched_case_stage": matched_stage,
        "recording_counterpart": recording_counterpart_debug,
        "turns": validation_notes,
    }
    return final, speaker_map, participants


# =============================================================================
# MANUAL AND ANONYMOUS FALLBACK OUTPUT
# =============================================================================
def build_manual_or_anonymous_turns(
    anonymous_turns: Sequence[ConversationTurn],
    speaker_map: Dict[str, str],
    participants: Sequence[str],
) -> List[ConversationTurn]:
    """Create final turns for manual mappings or anonymous-output mode without invoking the LLM."""
    final: List[ConversationTurn] = []
    for turn in anonymous_turns:
        sender = speaker_map.get(turn.speaker, turn.speaker)
        receiver = infer_receiver(sender, participants, speaker_map, turn.speaker)
        final.append(
            ConversationTurn(
                sender=sender,
                receiver=receiver,
                message=clean_message(turn.message),
                speaker=turn.speaker,
                start=turn.start,
                end=turn.end,
            )
        )
    return final



# =============================================================================
# PIPELINE ORCHESTRATION
# =============================================================================
def main() -> int:
    """Run the complete two-stage audio pipeline and write per-file artifacts plus the final

    manifest.
    """
    args = parse_args()
    if args.check_deps:
        check_deps(args.backend)
        if args.llm_backend == "ollama":
            import requests  # noqa: F401
            print("ollama attribution deps ok")
        return 0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    case_report_text = read_case_report(args.case_report)
    actor_candidates = extract_actor_candidates(case_report_text)
    primary_participant = extract_primary_report_party(case_report_text)
    explicit_participants = parse_participants(args.participants)
    manual_map = parse_speaker_map_file(args.speaker_map)
    extra_hotwords = parse_participants(args.hotwords)
    prompt_participants = explicit_participants if explicit_participants else actor_candidates
    hotwords = build_hotwords(case_report_text, prompt_participants, args.max_hotwords)
    for word in extra_hotwords:
        if word.casefold() not in {h.casefold() for h in hotwords}:
            hotwords.append(word)
    hotwords = hotwords[: args.max_hotwords]
    prompt = build_prompt(args.prompt, hotwords, args.language)

    if args.llm_backend == "ollama":
        case_context = load_or_build_case_context(
            case_report_text=case_report_text,
            actor_candidates=explicit_participants or actor_candidates,
            primary_participant=primary_participant,
            model=args.ollama_model,
            host=args.ollama_host,
            timeout=args.llm_timeout,
            temperature=args.llm_temperature,
            cache_path=args.case_context_cache,
        )
    else:
        case_context = _validate_case_context({}, case_report_text, actor_candidates, primary_participant)
    allowed_case_names = _case_context_allowed_names(case_context, explicit_participants or actor_candidates)

    audio_files = discover_audio_files(args.inputs, output_dir)
    print(f"[INFO] Audio files: {len(audio_files)}")
    print("[INFO] Stage 1: MOSS transcription + anonymous speaker-change mapping")
    print("[INFO] Stage 2: LLM attribution from case report + speaker-turn TXT")
    print(f"[INFO] MOSS backend: {args.backend}")
    print(f"[INFO] MOSS model: {args.model_id}")
    if args.backend == "moss-local":
        print(f"[INFO] MOSS device: {args.device}; dtype: {args.dtype}")
    if args.llm_backend == "ollama":
        print(f"[INFO] LLM backend: ollama; model: {args.ollama_model}; host: {args.ollama_host}")
    else:
        print("[INFO] LLM backend disabled; final CSV will keep anonymous speaker labels.")
    print(f"[INFO] Case report loaded. Actor candidates for hotwords: {len(actor_candidates)}")
    print(f"[INFO] Validated case graph participants: {len(allowed_case_names)}")
    if explicit_participants:
        print(f"[INFO] Optional participant constraint supplied: {len(explicit_participants)}")
    if manual_map:
        print(f"[INFO] Manual speaker map supplied: {len(manual_map)}; LLM attribution will be skipped for mapped speakers.")
    if hotwords:
        print(f"[INFO] Hotwords supplied to MOSS prompt: {len(hotwords)}")

    if not audio_files:
        print("[ERROR] No audio files found.", file=sys.stderr)
        return 2

    runner = make_runner(args)
    manifest: Dict[str, Any] = {
        "backend": args.backend,
        "model_id": args.model_id,
        "llm_backend": args.llm_backend,
        "ollama_model": args.ollama_model if args.llm_backend == "ollama" else None,
        "case_report": args.case_report,
        "actor_candidates_count": len(actor_candidates),
        "case_context_participants": allowed_case_names,
        "primary_participant": primary_participant,
        "files": [],
    }
    failures = 0

    for index, audio_path in enumerate(audio_files, start=1):
        print(f"\n[RUN] ({index}/{len(audio_files)}) {audio_path}")
        stem = safe_stem(audio_path)
        raw_txt_path = output_dir / f"{stem}.moss.raw.txt"
        speaker_turns_path = output_dir / f"{stem}.speaker_turns.txt"
        final_txt_path = output_dir / f"{stem}.diarized.txt"
        csv_path = output_dir / f"{stem}.diarized.csv"
        speaker_template_path = output_dir / f"{stem}.speaker_map_template.json"
        llm_prompt_path = output_dir / f"{stem}.llm_prompt.txt"
        llm_attr_path = output_dir / f"{stem}.llm_attribution.json"
        debug_path = output_dir / f"{stem}.debug.json"

        file_record: Dict[str, Any] = {"input": str(audio_path), "status": "started"}
        try:
            result = runner.transcribe(audio_path, prompt=prompt, max_new_tokens=args.max_new_tokens)
            raw_text = str(result.get("text", ""))
            segments = parse_moss_transcript(raw_text)
            timestamp_retry_used = False
            if not moss_timestamps_are_complete(segments):
                timestamp_retry_used = True
                print(
                    "[WARN] MOSS output omitted or malformed canonical timestamps; "
                    "retrying once with the official timestamped-diarization prompt."
                )
                retry_prompt = build_prompt(DEFAULT_MOSS_PROMPT, [], args.language)
                retry_result = runner.transcribe(
                    audio_path,
                    prompt=retry_prompt,
                    max_new_tokens=args.max_new_tokens,
                )
                retry_text = str(retry_result.get("text", ""))
                retry_segments = parse_moss_transcript(retry_text)
                if moss_timestamps_are_complete(retry_segments):
                    result = retry_result
                    raw_text = retry_text
                    segments = retry_segments

            raw_txt_path.write_text(raw_text.strip() + "\n", encoding="utf-8")
            if not segments:
                raise RuntimeError(
                    "MOSS returned no parseable speaker segments. Check .moss.raw.txt "
                    "and try larger --max-new-tokens."
                )
            if not moss_timestamps_are_complete(segments):
                raise RuntimeError(
                    "MOSS did not return complete [start][Sxx]text[end] timestamps "
                    "after the strict retry; refusing to invent zero/+1-second offsets."
                )

            anonymous_turns = build_anonymous_speaker_turns(segments, merge_same_speaker=not args.no_merge_turns)
            if not anonymous_turns:
                raise RuntimeError("No anonymous speaker turns could be built from MOSS output.")
            write_speaker_turns_txt(speaker_turns_path, anonymous_turns)
            print(f"[OK] Stage 1 speaker-turn TXT: {speaker_turns_path}")

            if manual_map:
                participants = [v for v in manual_map.values() if v and v.casefold() != "unknown"]
                if explicit_participants:
                    participants = explicit_participants
                final_turns = build_manual_or_anonymous_turns(anonymous_turns, manual_map, participants)
                speaker_map = dict(manual_map)
                attribution: Dict[str, Any] = {"mode": "manual speaker-map", "speaker_map": speaker_map, "participants": participants, "turns": []}
            elif args.llm_backend == "ollama":
                llm_prompt = build_llm_attribution_prompt(
                    case_report_text=case_report_text,
                    case_context=case_context,
                    anonymous_turns=anonymous_turns,
                    explicit_participants=explicit_participants,
                    actor_candidates=actor_candidates,
                    audio_path=audio_path,
                    primary_participant=primary_participant,
                    max_case_chars=args.max_case_report_chars,
                    max_transcript_chars=args.max_transcript_chars,
                )
                if args.debug_json:
                    llm_prompt_path.write_text(llm_prompt, encoding="utf-8")
                attribution = run_ollama_attribution(
                    prompt=llm_prompt,
                    model=args.ollama_model,
                    host=args.ollama_host,
                    timeout=args.llm_timeout,
                    temperature=args.llm_temperature,
                )
                final_turns, speaker_map, participants = build_final_turns_from_attribution(
                    anonymous_turns=anonymous_turns,
                    attribution=attribution,
                    allowed_names=allowed_case_names,
                    audio_path=audio_path,
                    case_context=case_context,
                    primary_participant=primary_participant,
                )
                write_json(llm_attr_path, attribution)
                print(f"[OK] Stage 2 LLM attribution JSON: {llm_attr_path}")
            else:
                speaker_map = {}
                participants = []
                final_turns = build_manual_or_anonymous_turns(anonymous_turns, speaker_map, participants)
                attribution = {"mode": "llm disabled", "speaker_map": {}, "participants": [], "turns": []}

            # Final defense against malformed aliases returned by case parsing or the LLM.
            for final_turn in final_turns:
                final_turn.sender = strip_parenthetical_alias(final_turn.sender) or "Unknown"
                final_turn.receiver = strip_parenthetical_alias(final_turn.receiver) or "Unknown"

            write_conversation_csv(csv_path, final_turns)
            write_diarized_txt(final_txt_path, segments, speaker_map)
            write_json(speaker_template_path, speaker_map_template(segments, speaker_map, participants))

            if args.debug_json:
                write_json(
                    debug_path,
                    {
                        "input": str(audio_path),
                        "raw_text": raw_text,
                        "segments": segments_to_dicts(segments),
                        "anonymous_turns": turns_to_dicts(anonymous_turns),
                        "anonymous_turn_items_for_llm": turns_to_llm_items(anonymous_turns),
                        "speakers": ordered_speakers(segments),
                        "speaker_map": speaker_map,
                        "participants": participants,
                        "actor_candidates": actor_candidates,
                        "primary_participant": primary_participant,
                        "case_context": case_context,
                        "allowed_case_names": allowed_case_names,
                        "hotwords": hotwords,
                        "llm_attribution": attribution,
                        "final_turns": turns_to_dicts(final_turns),
                        "backend_raw": result.get("raw"),
                    },
                )

            print(f"[OK] Final CSV: {csv_path}")
            print(f"[OK] Final TXT: {final_txt_path}")
            print(f"[OK] Speaker map template: {speaker_template_path}")
            file_record.update(
                {
                    "status": "ok",
                    "raw_txt": str(raw_txt_path),
                    "speaker_turns_txt": str(speaker_turns_path),
                    "final_txt": str(final_txt_path),
                    "csv": str(csv_path),
                    "speaker_map_template": str(speaker_template_path),
                    "llm_attribution": str(llm_attr_path) if args.llm_backend == "ollama" or manual_map else None,
                    "segments": len(segments),
                    "anonymous_turns": len(anonymous_turns),
                    "final_turns": len(final_turns),
                    "timestamp_retry_used": timestamp_retry_used,
                    "speaker_map": speaker_map,
                    "participants": participants,
                }
            )
        except Exception as exc:
            failures += 1
            print(f"[ERROR] Failed for {audio_path}: {exc}", file=sys.stderr)
            file_record.update({"status": "failed", "error": str(exc)})
        manifest["files"].append(file_record)

    manifest_path = output_dir / "diarization_manifest.json"
    write_json(manifest_path, manifest)
    print(f"\n[DONE] Manifest: {manifest_path}")
    if failures:
        print(f"[DONE] Completed with {failures} failure(s).")
        return 1
    print("[DONE] Completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
