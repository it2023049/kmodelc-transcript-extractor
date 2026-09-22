# KModelC Transcript Extractor

A Python toolkit for extracting communication evidence from Facebook Messenger and Viber screenshots, email screenshots, and audio recordings.

The pipeline combines OCR, vision-language models, audio transcription and diarization, and case-report context to produce a unified CSV transcript.

## Features

- Facebook Messenger and Viber screenshot extraction
- Email screenshot extraction
- Audio transcription and speaker diarization using MOSS
- Local MOSS inference or an external MOSS API
- ZIP, folder, and individual evidence-file processing
- Screenshot-collage handling
- Case-report-assisted participant attribution
- Conversation-level sender/receiver continuity
- Source-image verification of selected OCR corrections
- Recovery of one or more omitted standalone `I` tokens
- Bounded correction of OCR word-order and character errors
- Separate visual verification of sentence punctuation
- Guarded post-extraction cleanup of chat messages
- Chronological merging into one CSV
- Optional intermediate outputs and diagnostic artifacts

This is a research tool. Extracted transcripts and participant assignments require review against the original evidence.

## Repository Structure

```text
kmodelc-transcript-extractor/
├── .github/
│   └── workflows/
│       └── python-check.yml
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── chat/
│   ├── extractor_utils.py
│   ├── facebook_extract.py
│   ├── viber_extract.py
│   └── mass_extract.py
├── email/
│   └── email_extract.py
└── speech/
    ├── audio_diarize.py
    └── audio_utils.py
```

Keep the repository layout intact. The batch orchestrator discovers the email and audio scripts through their sibling directories.

Use matching versions of the shared utilities and extractors.

## Components

### `chat/mass_extract.py`

The batch orchestrator:

- discovers evidence files;
- extracts ZIP packages;
- loads or discovers a case report;
- routes supported screenshots to the appropriate extractor;
- runs audio transcription and attribution;
- maintains chat conversation continuity;
- applies guarded post-extraction chat cleanup;
- merges results into a unified transcript;
- optionally retains intermediate outputs and a run manifest.

### `chat/extractor_utils.py`

Shared chat functionality, including:

- case-report parsing;
- participant and contact processing;
- screenshot and collage utilities;
- OCR processing;
- timestamp normalization;
- message cleanup;
- sender/receiver validation;
- source-image literal repairs;
- output formatting.

The literal-repair functionality is integrated into this file. A separate `literal_repairs.py` module is not required.

### `chat/facebook_extract.py`

Processes Facebook Messenger screenshots and collages.

It uses OCR geometry, visual evidence, and case context to extract messages and resolve their senders and receivers. It also handles screen-level timestamps and deterministic ordering of messages without individual visible times.

### `chat/viber_extract.py`

Processes Viber screenshots and collages, accounting for Viber-specific layouts and visible message timestamps.

### `email/email_extract.py`

Extracts email evidence from supported screenshots for inclusion in the merged transcript.

### `speech/audio_diarize.py`

Runs MOSS transcription and speaker diarization, with optional Ollama-assisted participant attribution using the case report.

### `speech/audio_utils.py`

Shared utilities for audio processing, participant attribution, and audio output generation.

## Requirements

- Python 3.10 or newer
- Ollama installed and available on `PATH`
- A vision-capable Ollama model, such as `gemma3:12b`
- The Python dependencies listed in `requirements.txt`
- Sufficient memory for the selected models
- A compatible GPU and PyTorch installation for CUDA execution

The dependency set includes:

- NumPy
- OpenCV
- EasyOCR
- Ollama Python client
- PyPDF2
- Requests
- PyTorch and Torchvision
- Transformers
- Accelerate
- SoundFile
- MOSS-Transcribe-Diarize

Model downloads may require internet access during initial setup.

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
```

Install dependencies:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## Ollama Setup

Download the model:

```bash
ollama pull gemma3:12b
```

Start the server if it is not already running:

```bash
ollama serve
```

Check installed models:

```bash
ollama list
```

## Input Evidence

The batch pipeline accepts evidence packages and directories containing supported communication evidence.

Supported processing includes:

- Facebook Messenger screenshots
- Viber screenshots
- Email screenshots
- Audio recordings supported by the selected backend

Other images may be classified as unsupported and skipped.

A PDF or TXT case report supplies contextual information such as:

- participant names
- aliases and contact details
- relationships
- relevant dates
- communication context

Case context helps constrain attribution, but it does not guarantee that every sender or receiver will be identified correctly.

## Quick Start

Run the following commands from the repository root.

### Process a ZIP package

```bash
python3 chat/mass_extract.py \
  evidence_package.zip \
  --case-report case_reports/report.pdf \
  --results-dir results \
  --model gemma3:12b
```

### Process an evidence directory

```bash
python3 chat/mass_extract.py \
  evidence/ \
  --case-report case_reports/report.pdf \
  --results-dir results \
  --model gemma3:12b
```

### Use automatic case-report discovery

If the evidence package contains a discoverable case report:

```bash
python3 chat/mass_extract.py evidence_package.zip
```

Pass `--case-report` explicitly when automatic discovery cannot identify the intended report.

### Full pipeline with local GPU audio processing

```bash
python3 chat/mass_extract.py \
  evidence_package.zip \
  --case-report case_reports/report.pdf \
  --results-dir results \
  --model gemma3:12b \
  --audio-backend moss-local \
  --audio-device cuda \
  --audio-dtype bfloat16 \
  --audio-llm-backend ollama \
  --audio-ollama-model gemma3:12b
```

Select an audio dtype supported by your hardware.

### Diagnostic run

```bash
python3 chat/mass_extract.py \
  evidence_package.zip \
  --case-report case_reports/report.pdf \
  --results-dir results/debug \
  --model gemma3:12b \
  --keep-chat-csvs \
  --keep-per-image \
  --keep-audio-output \
  --debug \
  --dump-ocr \
  --dump-draft \
  --dump-side-map \
  --manifest results/debug/run_manifest.json
```

Diagnostic outputs may contain the full evidence text. Store them with the same care as the original evidence.

## Single-Image Extraction

### Facebook Messenger

```bash
python3 chat/facebook_extract.py \
  images/facebook_example.png \
  case_reports/report.pdf \
  --model gemma3:12b \
  --langs en \
  --emoji-mode omit \
  --output results/facebook_example.csv \
  --debug-dir results/facebook_example_debug \
  --dump-ocr \
  --dump-draft \
  --dump-side-map
```

### Viber

```bash
python3 chat/viber_extract.py \
  images/viber_example.png \
  case_reports/report.pdf \
  --model gemma3:12b \
  --langs en \
  --emoji-mode omit \
  --output results/viber_example.csv \
  --debug-dir results/viber_example_debug \
  --dump-ocr \
  --dump-draft \
  --dump-side-map
```

### Collage layout overrides

For a fixed grid:

```bash
--grid 2x1
```

For an uneven row layout:

```bash
--layout 2,3
```

Use one layout override at a time.

## Chat Extraction and Correction

Chat processing includes several distinct stages.

### 1. OCR and visual extraction

OCR supplies text and geometry. The vision model uses the screenshot and extraction context to produce message rows with anonymous `LEFT` or `RIGHT` sides.

The pipeline then resolves participant identities using available evidence.

### 2. Per-screen text polish

A per-screen polishing stage may propose corrections to extracted text.

Validation restricts accepted changes, including checks against credential changes and selected contraction or tense rewrites.

### 3. Source-image literal repairs

The Facebook and Viber extractors call `repair_screen_literals()` from `extractor_utils.py`.

This stage:

1. Locates a candidate message region using OCR geometry.
2. Reads the image region without supplying the proposed replacement text.
3. Constructs a bounded repair candidate.
4. Reads an enlarged version of the region.
5. Accepts the repair only when both reads produce the same validated candidate.

Supported repair categories include:

- omitted standalone `I` tokens;
- multiple omitted standalone `I` tokens;
- selected adjacent or wrapped word-order errors;
- `!` versus lowercase `l` confusion;
- selected apostrophe-adjacent `$` versus `s` confusion;
- selected contraction or wording differences;
- sentence-punctuation differences.

These checks do not permit unrestricted rewriting.

Ambiguous regions, uncertain model reads, and conflicting confirmations leave the original message unchanged. Agreement between model reads is a safeguard, not proof of transcription accuracy.

Facebook processing may expand the OCR crop toward a detected bubble boundary. If a reliable boundary cannot be established, it retains the original crop.

### Emoji handling

With `--emoji-mode omit`, the existing emoji filter is applied to image-read proposals before repair validation.

This prevents omitted emojis from unnecessarily invalidating otherwise eligible corrections.

### Visual punctuation verification

The punctuation-only validator permits changes to:

```text
. , ; : ! ? …
```

It requires matching lexical content and protects against unrelated changes.

Messages containing detected credentials, identifiers, numbers, or other sensitive patterns are excluded from this punctuation-only correction path.

Apostrophes, quotation marks, and hyphens are outside this pass's allowed punctuation-edit set.

### Feature switches

The following features are enabled by default:

| Environment variable           | Function                                                       |
| ------------------------------ | -------------------------------------------------------------- |
| `KMODELC_MULTI_I=1`            | Permit recovery of multiple omitted standalone `I` tokens.     |
| `KMODELC_VISUAL_PUNCTUATION=1` | Permit source-image-verified sentence-punctuation corrections. |

Disable a feature by setting its value to `0`:

```bash
KMODELC_VISUAL_PUNCTUATION=0 \
python3 chat/mass_extract.py \
  evidence_package.zip \
  --case-report case_reports/report.pdf
```

With `--dump-draft`, per-screen `*_literal_repairs.json` files record repair attempts, source reads, accepted changes, and rejection reasons.

### 4. Post-extraction Message-only LLM pass

After chat extraction, `mass_extract.py` runs a separate text-only cleanup pass.

This pass has no source-image evidence. Its validator permits a limited set of changes, including:

- selected spacing and capitalization corrections;
- equivalent typographic forms;
- one hypothesized missing standalone `I`;
- bounded word-order corrections.

A missing `I` accepted here is a text-based hypothesis, not an image-confirmed correction.

The pass rejects changes to protected values and rejects punctuation additions, deletions, or replacements that require source verification. Selected contraction and tense rewrites are also rejected.

The following remain unchanged by this pass:

- `Timestamp`
- `Estimated_Timestamp`
- `Sender`
- `Receiver`
- row count
- row order

Disable this final text-only pass with:

```bash
--no-chat-polish
```

This flag does **not** disable the source-image literal-repair stage inside the screenshot extractors.

To retain the merged chat transcript before the final text-only pass:

```bash
--raw-chat-output results/before_llm.csv
```

This is not a raw OCR dump: earlier screenshot-level processing has already taken place.

## Sender and Receiver Attribution

Chat attribution combines evidence such as:

- bubble position and geometry;
- visible contact or header text;
- case-report participants;
- explicit self-identification;
- direct address;
- contact details;
- conversation continuity.

Third-party mentions are not automatically treated as evidence of the current speaker.

The batch pipeline may retain a temporary conversation-state cache for screenshots from the same evidence folder. This supplies a continuity prior; current-image evidence may override it.

Audio attribution uses diarized speaker labels, transcription content, and case context. Ambiguous or incorrect assignments remain possible and require review.

## Output Format

The final merged transcript uses:

```csv
"Timestamp","Estimated_Timestamp","Sender","Receiver","Message"
```

Example:

```csv
"Timestamp","Estimated_Timestamp","Sender","Receiver","Message"
"12/03/2026 10:15:00","False","Alice Example","Bob Example","Hello Bob."
"12/03/2026 10:15:01","True","Bob Example","Alice Example","Hi Alice."
```

### Timestamps

Timestamps use:

```text
DD/MM/YYYY HH:MM:SS
```

`Estimated_Timestamp` indicates the pipeline's timestamp classification. It is not a general confidence score for the entire row.

### Facebook Messenger

Visible timestamps supply anchors. Deterministic second increments may be used to preserve visual message order when individual message times are unavailable.

Rows with generated seconds are marked as estimated.

### Viber

Visible message times are normalized, with `:00` added when seconds are not shown.

The current Viber pipeline marks these timestamps as non-estimated.

### Audio

Standalone audio CSVs use:

```csv
Offset_Seconds,Sender,Receiver,Message
```

During batch merging, audio offsets are combined with an inferred or supplied date. The resulting absolute timestamps are marked as estimated.

### Email

Email rows use the unified schema. Review whether the date and time were visible in the source or inferred by the extraction and merge process.

## Useful Batch Options

| Option                                          | Purpose                                       |
| ----------------------------------------------- | --------------------------------------------- |
| `--case-report PATH`                            | Supply the case report explicitly.            |
| `--results-dir PATH`                            | Set the output directory.                     |
| `--output PATH`                                 | Set the merged CSV path.                      |
| `--manifest PATH`                               | Retain a run manifest.                        |
| `--model MODEL`                                 | Select the Ollama model.                      |
| `--classify-mode {auto,filename,vision}`        | Select screenshot classification mode.        |
| `--force-platform {auto,facebook,viber,email}`  | Override screenshot routing.                  |
| `--facebook-script PATH`                        | Override the Facebook extractor path.         |
| `--viber-script PATH`                           | Override the Viber extractor path.            |
| `--email-script PATH`                           | Override the email extractor path.            |
| `--audio-script PATH`                           | Override the audio diarizer path.             |
| `--audio-backend {moss-local,moss-api}`         | Select the audio backend.                     |
| `--audio-device {auto,cuda,cpu}`                | Select the audio inference device.            |
| `--audio-dtype {auto,bfloat16,float16,float32}` | Select the audio inference dtype.             |
| `--chat-polish-model MODEL`                     | Override the final chat-cleanup model.        |
| `--chat-polish-host URL`                        | Override the final chat-cleanup endpoint.     |
| `--chat-polish-batch-size N`                    | Set the final chat-cleanup batch size.        |
| `--no-chat-polish`                              | Disable the final text-only chat cleanup.     |
| `--raw-chat-output PATH`                        | Retain chat output before that final cleanup. |
| `--keep-chat-csvs`                              | Retain intermediate chat CSVs.                |
| `--keep-per-image`                              | Retain per-image outputs.                     |
| `--keep-audio-output`                           | Retain audio intermediate outputs.            |
| `--debug`                                       | Retain diagnostic outputs.                    |
| `--dump-ocr`                                    | Save OCR artifacts.                           |
| `--dump-draft`                                  | Save draft and literal-repair diagnostics.    |
| `--dump-side-map`                               | Save side-mapping diagnostics.                |

For the complete argument list:

```bash
python3 chat/mass_extract.py --help
python3 chat/facebook_extract.py --help
python3 chat/viber_extract.py --help
python3 email/email_extract.py --help
python3 speech/audio_diarize.py --help
```

## Limitations

- OCR can omit, reorder, merge, or misrecognize text.
- Vision models can alter message boundaries or produce incorrect readings.
- Two agreeing image reads can still be wrong.
- Text-only cleanup may accept an incorrect missing-subject hypothesis.
- Conservative validation can leave real errors unchanged.
- Sender/receiver attribution may be incorrect or unresolved.
- Audio transcription may omit short utterances, hesitations, or interjections.
- Estimated timestamps are not recovered recording times.
- Conversation continuity can be misleading when participants or screenshot ownership change.
- Runtime increases with the number of images and verification calls.

Review the final transcript against the original evidence before relying on it.

## License

This project is licensed under the Apache License 2.0.

See [LICENSE](LICENSE) for details.
