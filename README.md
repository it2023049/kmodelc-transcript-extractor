# KModelC Transcript Extractor

A Python toolkit for extracting communication evidence from chat screenshots,
email screenshots, voice messages, and recorded calls into one normalized,
chronologically ordered CSV transcript.

The pipeline combines OCR, vision-language models, MOSS transcription and
speaker diarization, and report-grounded participant attribution. It is designed
for research, evaluation, and evidence-review workflows in which the original
source material remains available for human verification.

## Features

- Facebook Messenger screenshot and collage extraction
- Viber screenshot and collage extraction
- email screenshot extraction
- voice-message and recorded-call transcription
- speaker-change timestamps from MOSS
- report-grounded sender and receiver attribution
- ZIP, folder, file, and glob input
- recursive evidence discovery
- automatic case-report discovery
- vision-assisted evidence classification
- skipping profiles, receipts, dashboards, unrelated documents, and other
  non-conversation images
- strict post-chat LLM cleanup limited to the `Message` field
- contextual date selection for each audio file
- chronological merge into one forensic CSV
- optional intermediate outputs, debug artifacts, and JSON manifests

The public output schema is:

```csv
"Timestamp","Estimated_Timestamp","Sender","Receiver","Message"
"DD/MM/YYYY HH:MM:SS","False","Sender Name","Receiver Name","Message text"
```

## Processing Order

The batch pipeline follows a fixed order:

1. classify and extract chat screenshots
2. write the raw chat-only CSV
3. run strict `Message`-only cleanup
4. write the polished chat-only CSV
5. transcribe and attribute audio evidence
6. extract deferred email screenshots
7. merge and sort all retained rows chronologically

The post-chat cleanup is not allowed to change timestamps, sender/receiver
fields, row order, or row count. It may only correct obvious OCR word-order,
spacing, capitalization, punctuation, or identifier-spacing errors inside a
message.

## Supported Evidence

### Chat screenshots

- Facebook Messenger
- Viber
- individual screenshots
- multi-screen collages

### Email screenshots

- open email messages
- webmail views
- rendered email content with visible sender, recipient, date, and body evidence

### Audio and video

Audio and video formats accepted by the installed MOSS and media dependencies
can be processed as:

- single-speaker voice messages
- two-party calls
- multi-party calls
- reconstructed conversation recordings

### Automatically skipped images

Examples include:

- standalone contact or profile screens without message bubbles
- payment receipts
- account dashboards
- transaction confirmations
- identity documents
- unrelated photographs
- settings screens
- unsupported or unrecognized evidence

Skipped non-conversation images are not treated as extraction failures.

## Project Structure

```text
kmodelc-transcript-extractor/
├── README.md
├── chat/
│   ├── mass_extract.py
│   ├── facebook_extract.py
│   ├── viber_extract.py
│   └── extractor_utils.py
├── speech/
│   ├── audio_diarize.py
│   └── audio_utils.py
└── email/
    └── email_extract.py
```

Treat these seven Python files as one matching release set. Do not combine files
from different releases of the pipeline.

## Components

### `chat/mass_extract.py`

The integrated batch orchestrator. It:

- discovers or accepts a case report
- extracts ZIP packages safely
- discovers supported evidence recursively
- classifies screenshots
- runs the appropriate chat or email extractor
- preserves conversation-level chat mapping continuity
- writes raw and polished chat-only CSV files
- runs the audio pipeline once for all discovered recordings
- infers an evidence-bounded date independently for each audio file
- merges retained rows into one chronological transcript
- reports partial failure when a required stage does not complete

### `chat/facebook_extract.py`

Processes one Facebook Messenger screenshot or collage. It extracts anonymous
left/right message rows, infers a participant mapping, validates that mapping
against the report and conversation evidence, and writes the common CSV schema.

### `chat/viber_extract.py`

Processes one Viber screenshot or collage using the same evidence-constrained
attribution strategy while accounting for Viber-specific layout and timestamp
behavior.

### `chat/extractor_utils.py`

Shared chat functionality, including:

- report parsing and participant discovery
- screenshot and collage splitting
- OCR block parsing
- timestamp normalization
- bubble-side analysis
- message cleanup
- duplicate detection
- participant-pair scoring
- conversation-level side-map validation
- CSV generation

### `speech/audio_diarize.py`

Runs a two-stage audio pipeline:

1. MOSS transcription and anonymous speaker-change mapping
2. constrained sender/receiver attribution using the case report, transcript,
   filename provenance, interaction graph, and conversation context

The canonical MOSS transcript format is parsed into real start/end offsets.
Outputs without complete, monotonic timestamps are retried with the official
timestamped-diarization prompt and are not silently replaced with invented
offsets.

### `speech/audio_utils.py`

Shared audio helpers for:

- media discovery
- speaker and conversation data structures
- case-report participant parsing
- receiver inference
- turn merging
- CSV, TXT, and JSON output

### `email/email_extract.py`

Processes one email screenshot and writes one row in the common transcript
schema. It uses visible screenshot evidence for the message body and uses the
case report only to constrain and canonicalize supported identities and dates.

## Sender and Receiver Attribution

### Chat attribution

Chat attribution is performed at conversation level rather than by freely
guessing an identity for every row.

The pipeline derives a provisional `LEFT`/`RIGHT` mapping using evidence such as:

- platform layout
- bubble geometry and color
- visible header or contact text
- phone and account evidence
- case-report participants
- extracted conversation content
- continuity from earlier screenshots in the same evidence folder

Strong identity cues include:

- an explicit leading speaker label
- self-identification such as `This is Alex Example`
- direct address such as `Hello Jordan`
- exact visible contact/header evidence
- a consistent two-person conversation structure

A third-party mention is not treated as automatic speaker evidence. For example:

```text
I spoke with Alex.
Jordan told me about the payment.
```

does not imply that Alex or Jordan sent the current message.

Participant fields are restricted to individual report-grounded names. Document
headings, offence labels, section titles, roles, channels, and explanatory
phrases are not valid sender or receiver values.

### Audio attribution

Audio attribution combines:

- anonymous MOSS speaker labels
- explicit self-identification
- direct address
- exact report-grounded participant names
- filename provenance
- report timeline and interaction evidence
- constrained LLM attribution
- deterministic post-validation

For a single-speaker voice message, the audible speaker may be identified from
self-identification or filename provenance. The non-audible receiver is inferred
conservatively from direct address and interaction evidence. If the evidence is
insufficient, the attribution remains unresolved rather than inventing a name.

## Timestamp Semantics

### `Estimated_Timestamp`

- `False`: the timestamp is directly supported by the source screenshot.
- `True`: some or all timestamp components were inferred or synthesized from
  bounded evidence.

### Facebook Messenger

Facebook screenshots may expose a screen-level time rather than one complete
timestamp per bubble. The extractor preserves the visible anchor and may add
deterministic seconds to retain bubble order.

### Viber

Visible Viber times are normalized to:

```text
DD/MM/YYYY HH:MM:SS
```

When seconds are not visible, `:00` is added.

### Email

Email rows use the supported email date at `00:00:00` and are marked as
estimated.

### Audio

Each audio file is evaluated independently against dates observed in the report
and extracted chats. The selected date anchors the complete recording, so every
turn in one conversation keeps the same calendar date.

Real MOSS speaker-change offsets are then added to that date. Separate audio
files do not automatically inherit the date chosen for an earlier file.

## Prerequisites

- Python 3.10 or newer
- Ollama installed and available on `PATH`
- a vision-capable Ollama model such as `gemma3:12b`
- EasyOCR
- OpenCV
- NumPy
- PyPDF2 or pypdf
- PyTorch appropriate for the selected device
- MOSS Transcribe-Diarize
- Transformers, Accelerate, SoundFile, and Requests for audio processing

A compatible GPU is recommended for practical local vision and audio inference.

## Installation

Clone the repository:

```bash
git clone https://github.com/it2023049/kmodelc-transcript-extractor.git
cd kmodelc-transcript-extractor
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
```

Install the core dependencies:

```bash
python3 -m pip install \
  opencv-python-headless \
  easyocr \
  numpy \
  ollama \
  PyPDF2 \
  torch \
  transformers \
  accelerate \
  soundfile \
  requests
```

Install MOSS Transcribe-Diarize:

```bash
python3 -m pip install \
  "moss-transcribe-diarize @ git+https://github.com/OpenMOSS/MOSS-Transcribe-Diarize.git"
```

Install a PyTorch build compatible with your device before installing the audio
dependencies when the default PyTorch package is not appropriate.

## Ollama Setup

Pull the default model:

```bash
ollama pull gemma3:12b
```

Confirm that it is available:

```bash
ollama list
```

Start Ollama:

```bash
ollama serve
```

An alternate Ollama endpoint may be configured with:

```bash
export OLLAMA_HOST=http://127.0.0.1:11434
```

## Quick Start

Run all commands from the repository root.

### ZIP package with an explicit case report

```bash
python3 chat/mass_extract.py \
  evidence_package.zip \
  --case-report case_reports/case_report.pdf \
  --results-dir results \
  --model gemma3:12b
```

### Package with an auto-discovered case report

```bash
python3 chat/mass_extract.py evidence_package.zip
```

### Folder input

```bash
python3 chat/mass_extract.py evidence_folder/
```

### Explicit report followed by multiple evidence paths

```bash
python3 chat/mass_extract.py \
  case_reports/case_report.pdf \
  evidence/facebook/ \
  evidence/viber/ \
  evidence/audio/ \
  evidence/email/
```

## Standalone Extraction

### Facebook Messenger

```bash
python3 chat/facebook_extract.py \
  images/example_facebook.png \
  case_reports/case_report.pdf \
  --model gemma3:12b \
  --langs en \
  --emoji-mode omit \
  --output results/example_facebook.csv \
  --debug-dir results/example_facebook_debug
```

### Viber

```bash
python3 chat/viber_extract.py \
  images/example_viber.png \
  case_reports/case_report.pdf \
  --model gemma3:12b \
  --langs en \
  --emoji-mode omit \
  --output results/example_viber.csv \
  --debug-dir results/example_viber_debug
```

### Email

```bash
python3 email/email_extract.py \
  images/example_email.png \
  case_reports/case_report.pdf \
  --model gemma3:12b \
  --langs en \
  --output results/example_email.csv \
  --debug-dir results/example_email_debug
```

### Audio

```bash
python3 speech/audio_diarize.py \
  audio/example_call.mp3 \
  --case-report case_reports/case_report.pdf \
  --output-dir results/audio \
  --backend moss-local \
  --ollama-model gemma3:12b
```

## Platform Classification

`chat/mass_extract.py` provides three modes:

| Mode | Behavior |
| --- | --- |
| `auto` | Inspects image content first and uses filename/path evidence when the visual classifier is uncertain. |
| `vision` | Uses visual classification and falls back to filename/path evidence when needed. |
| `filename` | Uses filename and folder hints only. |

The platform can also be forced when every candidate image belongs to one known
type:

```bash
--force-platform facebook
--force-platform viber
--force-platform email
```

## Useful Batch Options

| Option | Purpose |
| --- | --- |
| `--case-report PATH` | Override automatic case-report discovery. |
| `--output PATH` | Set the final merged CSV path. |
| `--results-dir PATH` | Set the output root directory. |
| `--manifest PATH` | Write a JSON run manifest. |
| `--classify-mode MODE` | Select `auto`, `vision`, or `filename` classification. |
| `--force-platform TYPE` | Force `facebook`, `viber`, or `email`. |
| `--model MODEL` | Set the Ollama classification/extraction model. |
| `--langs LANGS` | Set EasyOCR languages. |
| `--cpu` | Use CPU mode for OCR. |
| `--no-vision` | Disable direct image input inside the extractors. |
| `--no-chat-polish` | Disable strict post-chat `Message` cleanup. |
| `--chat-polish-model MODEL` | Set the message-cleanup model. |
| `--keep-per-image` | Keep intermediate image CSV files. |
| `--keep-audio-output` | Keep audio intermediate files. |
| `--debug` | Keep per-image debug directories. |
| `--dump-ocr` | Save OCR artifacts. |
| `--dump-draft` | Save extraction drafts. |
| `--dump-side-map` | Save provisional and final side mappings. |
| `--audio-date DD/MM/YYYY` | Override contextual audio-date selection. |
| `--audio-backend BACKEND` | Select local or API-based MOSS processing. |
| `--audio-debug-json` | Save detailed audio JSON artifacts. |
| `--keep-duplicates` | Disable exact chat/email row deduplication. |

Use the built-in help for the complete option set:

```bash
python3 chat/mass_extract.py --help
python3 speech/audio_diarize.py --help
python3 email/email_extract.py --help
```

## Output Files

A typical retained output tree is:

```text
results/
├── extracted/
├── per_image/
├── per_audio/
├── <package>_chat_raw.csv
├── <package>_chat_polished.csv
├── <package>_merged.csv
└── <run_manifest>.json
```

Only retained outputs are present. Intermediate directories may be temporary
unless their corresponding retention flags are enabled.

## Status and Failure Semantics

- `[SUCCESS]` means every required stage completed and the final CSV was
  written.
- `[PARTIAL FAILURE]` means a final CSV was written, but at least one required
  image, audio, email, attribution, date-selection, or cleanup stage was
  incomplete.
- a non-zero exit status is returned for partial or complete failure.
- recognized non-conversation evidence is skipped and is not counted as a
  failure.

## Troubleshooting

### Case report cannot be discovered

Supply it explicitly:

```bash
python3 chat/mass_extract.py \
  evidence_package.zip \
  --case-report case_reports/case_report.pdf
```

### Unknown or incorrect screenshot platform

Use clear `facebook`, `messenger`, `viber`, or `email` filename/folder hints, or
compare:

```bash
--classify-mode auto
--classify-mode filename
--classify-mode vision
```

Use a forced platform only when all candidate screenshots use the same
application.

### Incorrect sender or receiver

Retain the attribution evidence:

```bash
--dump-side-map --dump-ocr --dump-draft --keep-per-image
```

Review:

- provisional and final participant mappings
- visible header/contact evidence
- bubble geometry
- direct-address and self-identification cues
- third-party mentions
- conversation-continuity influence

### Audio timestamps are unavailable

MOSS output must contain complete and monotonic start/end timestamps. Inspect
the retained audio artifacts with:

```bash
--keep-audio-output --audio-debug-json
```

The pipeline does not generate synthetic replacement offsets when the MOSS
output remains malformed after retry.

### Audio date cannot be established

The date-selection model may only choose from dates observed in the case report
or extracted chats. Use `--audio-date DD/MM/YYYY` only when an explicit,
externally verified date is available.

### Ollama connection error

Verify the model and server:

```bash
ollama list
ollama serve
```

### Excessive OCR errors

Try:

- a higher-resolution source
- one screenshot instead of a collage
- the correct OCR languages
- retained OCR and draft artifacts
- visual extraction instead of OCR-only mode

Avoid adding evidence-specific replacement rules to shared extraction code.

## Reproducibility

For reproducible evaluation:

- use fixed model and dependency versions
- use deterministic model temperature where supported
- retain a run manifest
- preserve the original evidence separately
- do not modify source screenshots or recordings before extraction
- compare the final CSV with every original source
- document any manual corrections separately

## Privacy and Repository Safety

Do not commit real evidence or private case material to a public repository.

Do not publish:

- real case reports
- evidence packages
- private screenshots or recordings
- generated transcripts from real cases
- OCR dumps and extraction drafts
- phone numbers or email addresses
- account credentials
- financial details
- access tokens
- model caches
- execution logs containing evidence

Publish only source code, documentation, synthetic examples, anonymized fixtures,
and explicitly authorized test data.

Recommended `.gitignore` entries include:

```gitignore
.venv/
__pycache__/
*.py[cod]
results/
evidence/
case_reports/
*.zip
*.log
*.tmp
.DS_Store
```

## Limitations

- This is a research prototype, not an evidence-acquisition system.
- OCR may miss, reorder, merge, split, or incorrectly recognize visible text.
- Vision-language models may misclassify evidence or alter message segmentation.
- Sender/receiver attribution is evidence-constrained but not guaranteed.
- A screenshot containing multiple conversations may violate the fixed-pair
  assumption.
- Conversation continuity may become invalid when the visible account owner or
  participant pair changes.
- Email body extraction may omit content when the screenshot is cropped or the
  layout is highly structured.
- Speaker diarization and transcription quality depend on audio quality,
  overlap, noise, and model behavior.
- Separate audio files may contain overlapping speech and therefore require
  review for duplicate content.
- Contextual audio dates are estimates and must remain marked accordingly.
- The final CSV must be reviewed against the original evidence before use.

For faster iteration, test one representative screenshot, one recording, one
email, and the associated report before processing a large package.

## Related Projects

- [forensic-chat-screenshot-extractor](https://github.com/it2023049/forensic-chat-screenshot-extractor)
- [chat-screenshot-and-audio-to-csv-extractor](https://github.com/it2023049/chat-screenshot-and-audio-to-csv-extractor)
- [MOSS Transcribe-Diarize](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize)

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for
the full license text.
