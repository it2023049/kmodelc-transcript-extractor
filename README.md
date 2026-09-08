# KModelC-Griphia

A Python toolkit for extracting communication evidence from Facebook Messenger and Viber screenshots and converting the results into one normalized CSV transcript.

KModelC-Griphia is **chat-only**. It processes supported chat screenshots and deliberately ignores audio/video evidence. Email screenshots and other unsupported images are classified as non-chat/unsupported and skipped.

## Features

- Facebook Messenger screenshot extraction
- Viber screenshot extraction
- screenshot-collage handling
- ZIP package input
- folder input
- single-image extraction
- recursive evidence discovery
- automatic case-report discovery
- automatic Facebook/Viber routing
- skipping unsupported, email, non-chat, and unknown images
- case-aware sender/receiver attribution
- conversation-level `LEFT`/`RIGHT` continuity
- chronological merge into one CSV
- strict post-extraction LLM cleanup limited to the `Message` field
- deterministic validation of every proposed LLM edit
- optional debug and intermediate outputs

The final transcript schema is:

```csv
"Timestamp","Estimated_Timestamp","Sender","Receiver","Message"
"DD/MM/YYYY HH:MM:SS","False","Sender Name","Receiver Name","Message text"
```

## Scope

KModelC-Griphia processes **chat screenshots only**.

Supported communication platforms:

- Facebook Messenger
- Viber

Not processed:

- audio recordings
- video recordings
- email screenshots
- payment receipts
- banking screenshots
- trading dashboards
- identity documents
- unrelated images

Audio/video files may exist inside an evidence ZIP, but the pipeline intentionally ignores them.

## Project Structure

KModelC-Griphia uses a flat script layout:

```text
KModelC-Griphia/
├── README.md
├── extractor_utils.py
├── facebook_extract.py
├── viber_extract.py
└── mass_extract.py
```

These four Python files form one matching code package:

```text
extractor_utils.py
facebook_extract.py
viber_extract.py
mass_extract.py
```

Keep the four matching Python files together; mixing files from different code packages can break internal arguments and shared behavior.

## Components

### `extractor_utils.py`

Shared functionality for:

- case-report parsing
- participant extraction
- conservative participant validation
- screenshot/collage splitting
- OCR parsing
- timestamp normalization
- bubble-side handling
- message cleanup
- duplicate detection
- sender/receiver evidence scoring
- conversation-level side-map validation
- final CSV generation

### `facebook_extract.py`

Processes one Facebook Messenger screenshot or collage.

The extractor:

1. reads the case report
2. splits collages when necessary
3. extracts OCR and visual evidence
4. produces anonymous `LEFT`/`RIGHT` message rows
5. infers a provisional participant mapping
6. validates the mapping against report and conversation evidence
7. writes the final sender/receiver CSV

### `viber_extract.py`

Processes one Viber screenshot or collage using the same general attribution strategy while accounting for Viber-specific layout and timestamp behavior.

### `mass_extract.py`

Batch orchestrator for evidence ZIPs, folders, and groups of images.

It:

- extracts ZIP packages safely
- discovers candidate screenshots recursively
- discovers or accepts a case report
- classifies screenshots as Facebook, Viber, or unsupported
- runs the appropriate extractor
- preserves conversation-level mapping continuity within an evidence folder
- merges extracted rows chronologically
- runs a guarded LLM cleanup after all chat screenshots have been extracted
- changes only validated `Message` values
- writes one final CSV

## Attribution Strategy

Sender and receiver attribution is performed at conversation level rather than by independently guessing a participant for every row.

The pipeline first derives a provisional `LEFT`/`RIGHT` mapping using evidence such as:

- platform layout
- bubble geometry
- bubble color where available
- visible header/contact text
- phone/contact evidence
- case-report participants
- extracted conversation content
- continuity from previous screenshots in the same evidence folder

The mapping is then validated against deterministic evidence.

Strong cues include:

- an explicit leading speaker label
- self-identification such as `This is Alex Example`
- direct address such as `Hello Jordan`
- exact contact/header evidence
- consistent two-person conversation structure

Third-party mentions are not treated as automatic speaker evidence.

For example:

```text
I spoke with Alex.
Jordan told me about the payment.
```

does not imply that Alex or Jordan is the sender of the current message.

When evidence is insufficient to justify a correction, the provisional geometry/header mapping is preserved.

## Prerequisites

- Python 3.10 or newer
- Ollama installed and available on `PATH`
- a vision-capable Ollama model such as `gemma3:12b`
- EasyOCR
- OpenCV
- NumPy
- PyPDF2
- the Python `ollama` package

A GPU is recommended for practical vision-model processing, but the pipeline can also use CPU-backed OCR where supported.

## Installation

Clone the repository:

```bash
git clone https://github.com/it2023049/KModelC-Griphia.git
cd KModelC-Griphia
```

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Upgrade pip:

```bash
python3 -m pip install --upgrade pip
```

Install the chat dependencies:

```bash
python3 -m pip install \
  opencv-python-headless \
  easyocr \
  numpy \
  ollama \
  PyPDF2
```

If the repository includes a matching `requirements.txt`, you can use:

```bash
python3 -m pip install -r requirements.txt
```

## Ollama Setup

Pull the model:

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

If needed, configure a custom Ollama host:

```bash
export OLLAMA_HOST=http://127.0.0.1:11434
```

## Dependency Check

A lightweight import check:

```bash
python3 -c "import cv2, easyocr, ollama, PyPDF2, numpy; print('chat dependencies ok')"
```

## Input Data

The pipeline accepts:

- a ZIP package
- a folder
- one or more screenshot files
- a PDF/TXT case report supplied explicitly

Supported image extensions:

```text
.png
.jpg
.jpeg
.webp
.bmp
.tif
.tiff
```

Supported case-report extensions:

```text
.pdf
.txt
```

The case report is used to:

- identify valid human participants
- recover aliases and contact details
- constrain sender/receiver attribution
- understand communication relationships
- provide timeline/context information
- distinguish active participants from third parties

## Output Format

All final merged transcripts use:

```csv
"Timestamp","Estimated_Timestamp","Sender","Receiver","Message"
```

`Estimated_Timestamp` records whether the timestamp was directly derived from
visible source evidence (`False`) or whether part of it was deterministically
estimated by the pipeline (`True`). Only Facebook rows whose
seconds were generated to preserve visual message order are marked `True`.

### Strict Message-only LLM pass

After all supported screenshots have been extracted, `mass_extract.py` runs a
strict Ollama cleanup pass over the `Message` field before writing the final
merged CSV.

The pass is limited to obvious OCR corrections such as:

- broken word order, for example `doing today? How are you` → `How are you doing today?`
- spacing errors
- capitalization and punctuation errors
- obvious whitespace inside a website, URL, domain, or email identifier

The following values are immutable during this pass:

- `Timestamp`
- `Estimated_Timestamp`
- `Sender`
- `Receiver`
- row count and row order

The model is not allowed to paraphrase, summarize, translate, add or remove
words, change names or numbers, or split and merge rows. Every proposed edit is
checked deterministically before it is accepted. Changes that alter lexical
content or protected identifiers are rejected.

If an LLM batch fails, all edits from the pass are rolled back. The final CSV is
still written with the original extracted messages, the run is reported as
`PARTIAL FAILURE`, and the process returns a non-zero exit code.

The pass is enabled by default and uses `--model` unless a separate
`--chat-polish-model` is supplied. It can be disabled explicitly with:

```bash
python3 mass_extract.py evidence_package.zip --no-chat-polish
```

By default, no additional pre-LLM CSV is created. To retain one for audit or
comparison, provide its path explicitly:

```bash
python3 mass_extract.py evidence_package.zip \
  --raw-chat-output results/before_llm.csv
```

### Viber timestamps

Visible Viber message times are normalized to:

```text
DD/MM/YYYY HH:MM:SS
```

When the source UI does not show seconds, `:00` is added. Because the
Viber timestamp itself is extracted from the visible chat UI, the pipeline writes
`Estimated_Timestamp=False` for Viber rows.

### Facebook Messenger timestamps

Facebook screenshots may expose a screen-level or group-level visible time rather than a timestamp for every individual bubble.

The extractor keeps each visible minute-level timestamp as an observed anchor.
The anchor row keeps `:00` and is written with `Estimated_Timestamp=False`.
If later Facebook rows would otherwise have the same or an earlier minute-level
timestamp, the extractor adds deterministic one-second increments only to
preserve bubble order. Those generated-second rows are written with
`Estimated_Timestamp=True`.

Example:

```csv
"Timestamp","Estimated_Timestamp","Sender","Receiver","Message"
"12/03/2026 10:15:00","False","Alice Example","Bob Example","Hello Bob."
"12/03/2026 10:15:01","True","Bob Example","Alice Example","Hi Alice."
"12/03/2026 10:15:02","True","Alice Example","Bob Example","How are you?"
```

## Quick Start

All commands below assume the current directory contains:

```text
mass_extract.py
facebook_extract.py
viber_extract.py
extractor_utils.py
```

### ZIP package with explicit case report

```bash
python3 mass_extract.py \
  evidence_package.zip \
  --case-report case_reports/case_report.pdf \
  --results-dir results \
  --model gemma3:12b
```

### Package with auto-discovered case report

If the package contains a recognizable PDF/TXT case report:

```bash
python3 mass_extract.py evidence_package.zip
```

### Folder input

```bash
python3 mass_extract.py evidence_folder/
```

### Legacy explicit-report mode

```bash
python3 mass_extract.py \
  case_reports/case_report.pdf \
  evidence_package.zip
```

Multiple evidence paths may also be supplied:

```bash
python3 mass_extract.py \
  case_reports/case_report.pdf \
  evidence/facebook/ \
  evidence/viber/
```

## Single-Image Extraction

### Facebook Messenger

```bash
python3 facebook_extract.py \
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
python3 viber_extract.py \
  images/example_viber.png \
  case_reports/case_report.pdf \
  --model gemma3:12b \
  --langs en \
  --emoji-mode omit \
  --output results/example_viber.csv \
  --debug-dir results/example_viber_debug
```

### OCR-only mode

Disable direct vision-model image input while retaining OCR processing:

```bash
python3 facebook_extract.py \
  images/example_facebook.png \
  case_reports/case_report.pdf \
  --no-vision
```

OCR-only mode may reduce GPU use but can reduce text, layout, and bubble-boundary accuracy.

## Platform Classification

`mass_extract.py` supports three routing modes:

| Mode       | Behavior                                                                       |
| ---------- | ------------------------------------------------------------------------------ |
| `auto`     | Uses filename/path hints first and falls back to the vision model when needed. |
| `filename` | Uses filename/path hints only.                                                 |
| `vision`   | Uses the vision model first, then falls back to filename/path hints.           |

Recognized chat platforms are:

```text
facebook
viber
```

Email and other unsupported images are skipped.

### Filename-only classification

```bash
python3 mass_extract.py \
  evidence_package.zip \
  --classify-mode filename
```

### Force Facebook

```bash
python3 mass_extract.py \
  evidence_folder/ \
  --case-report case_reports/case_report.pdf \
  --force-platform facebook
```

### Force Viber

```bash
python3 mass_extract.py \
  evidence_folder/ \
  --case-report case_reports/case_report.pdf \
  --force-platform viber
```

Forced platform mode should only be used when all candidate screenshots belong to the selected application.

## Useful Batch Flags

| Flag                                     | Purpose                                                    |
| ---------------------------------------- | ---------------------------------------------------------- |
| `--case-report PATH`                     | Override automatic case-report discovery.                  |
| `--output PATH`                          | Set the merged CSV path.                                   |
| `--results-dir PATH`                     | Set the output root directory.                             |
| `--manifest PATH`                        | Optionally write a JSON run manifest.                      |
| `--classify-mode {auto,filename,vision}` | Select screenshot classification strategy.                 |
| `--force-platform {auto,facebook,viber}` | Force all candidate screenshots to one supported platform. |
| `--model MODEL`                          | Set the Ollama model.                                      |
| `--chat-polish-model MODEL`              | Override the model used by the strict Message-only pass.   |
| `--chat-polish-host URL`                 | Override the Ollama host used by the strict cleanup pass.  |
| `--chat-polish-batch-size N`             | Set rows per cleanup request (default `20`, maximum `50`). |
| `--no-chat-polish`                       | Disable the strict post-extraction LLM pass.               |
| `--raw-chat-output PATH`                 | Optionally retain the merged CSV before the LLM pass.      |
| `--langs LANGS`                          | Set EasyOCR languages.                                     |
| `--cpu`                                  | Force OCR CPU mode.                                        |
| `--no-vision`                            | Disable direct image input to the extractor vision model.  |
| `--emoji-mode omit`                      | Omit emojis from final chat text.                          |
| `--emoji-mode vision`                    | Keep emojis judged clearly visible by the vision model.    |
| `--debug`                                | Retain per-image debug directories.                        |
| `--keep-per-image`                       | Retain intermediate per-image CSVs.                        |
| `--dump-ocr`                             | Save OCR artifacts.                                        |
| `--dump-draft`                           | Save intermediate extraction drafts.                       |
| `--dump-side-map`                        | Save provisional/final side mappings.                      |
| `--keep-duplicates`                      | Disable exact final-row deduplication.                     |
| `--facebook-script PATH`                 | Override `facebook_extract.py`.                            |
| `--viber-script PATH`                    | Override `viber_extract.py`.                               |
| `--extra-extractor-arg VALUE`            | Forward an extra argument to chat extractors.              |

## Collage Handling

Automatic collage splitting is attempted first.

For standalone extraction, manual layout flags can be used when automatic splitting is unreliable.

### Fixed grid

```bash
--grid 2x1
```

Example:

```bash
python3 facebook_extract.py \
  images/collage.png \
  case_reports/case_report.pdf \
  --grid 2x1 \
  --output results/collage.csv \
  --debug-dir results/collage_debug
```

### Uneven row layout

```bash
--layout 2,3
```

Use either `--grid` or `--layout`, not both.

## Internal Conversation-State Cache

During a batch run, the pipeline can maintain a temporary continuity cache containing the last accepted `LEFT`/`RIGHT` mapping for each platform and evidence-folder conversation key.

This is only a **soft prior**.

Current-image evidence remains authoritative, including:

- a different visible contact
- explicit self-identification
- an exact speaker label
- strong direct-address evidence
- a participant pair incompatible with the cached mapping

The orchestrator passes internal arguments such as:

```text
--conversation-state-cache
--conversation-key
```

Users normally do not need to provide these manually.

The cache is temporary and should not be committed.

## Output Retention

By default, the batch pipeline writes the final merged CSV directly. It does
not create a separate pre-LLM CSV or run manifest unless the corresponding
arguments are supplied.

Use debug/retention flags when detailed review is needed:

```bash
--keep-per-image
--debug
--dump-ocr
--dump-draft
--dump-side-map
```

To retain the transcript immediately before the strict LLM pass, use:

```bash
--raw-chat-output results/before_llm.csv
```

A retained output tree may look like:

```text
results/
├── extracted/
├── per_image/
│   ├── <image>_extracted.csv
│   └── <image>_debug/
├── <optional_pre_llm>.csv
├── <package_stem>_merged.csv
└── <run_manifest>.json
```

## Troubleshooting

### Case report cannot be auto-discovered

If you see:

```text
Could not auto-discover a case report/overview PDF or TXT in the input package.
```

supply the report explicitly:

```bash
python3 mass_extract.py \
  evidence_package.zip \
  --case-report case_reports/case_report.pdf
```

### Conversation-state argument mismatch

If an extractor reports:

```text
unrecognized arguments: --conversation-state-cache --conversation-key
```

use all four files from the same code package:

```text
mass_extract.py
facebook_extract.py
viber_extract.py
extractor_utils.py
```

Do not mix files from different code packages.

### Missing `extractor_utils.py`

Verify:

```text
mass_extract.py
facebook_extract.py
viber_extract.py
extractor_utils.py
```

are kept together in the same directory.

### Unknown screenshot platform

Use recognizable filename/folder terms such as:

```text
facebook
messenger
viber
```

or use:

```bash
--classify-mode vision
```

### Ollama connection error

Verify:

```bash
ollama list
```

and make sure the server is running:

```bash
ollama serve
```

Check the configured host:

```bash
echo "$OLLAMA_HOST"
```

If only the strict Message cleanup should use another endpoint, provide:

```bash
--chat-polish-host http://127.0.0.1:11434
```

### Strict Message-only pass reports a partial failure

The final CSV is written with the original extracted messages because all LLM
changes are rolled back when a batch fails. Verify that the selected model is
installed, Ollama is reachable, and the requested batch fits the available
resources. A smaller batch may help:

```bash
--chat-polish-batch-size 10
```

For audit or comparison during a retry, request the optional pre-LLM CSV:

```bash
--raw-chat-output results/before_llm.csv
```

### Incorrect sender or receiver

Run the affected screenshot with:

```bash
--dump-side-map --dump-ocr --dump-draft
```

Review:

- provisional/final `LEFT`/`RIGHT` mapping
- OCR header/contact evidence
- bubble geometry
- direct-address cues
- self-identification
- speaker prefixes
- third-party mentions
- continuity-cache influence

### Missing or incorrect chat date

Use:

```bash
--dump-ocr --dump-draft
```

Check whether:

- the visible date separator was detected
- status-bar clocks were excluded
- the year was recovered correctly
- the current screenshot rather than an earlier screenshot supplied the date

### Excessive OCR errors

Try:

- a higher-resolution source
- one screenshot instead of a collage
- a manual collage layout
- the correct EasyOCR languages
- vision mode instead of OCR-only mode
- retained OCR/draft artifacts for review

Avoid adding case-specific text-replacement rules to shared extraction code.

## Reproducibility

For reproducible runs:

- use a fixed Ollama model build
- keep model temperature deterministic where configured
- record Python and package build identifiers
- retain a run manifest where needed
- preserve the original evidence package separately
- retain the optional pre-LLM CSV when an audit trail is required
- do not modify source screenshots before extraction
- review the merged CSV against the source evidence

The extractor output is intended to support review and should not replace human verification.

## Privacy and Repository Safety

Do not commit real evidence or private case material to a public repository.

Do not publish:

- real case reports
- evidence ZIPs
- private screenshots
- generated transcripts from real cases
- OCR dumps
- debug artifacts containing evidence
- phone numbers
- email addresses
- account credentials
- payment details
- IP addresses
- access tokens
- model caches
- execution logs containing evidence

Only publish:

- code
- documentation
- synthetic examples
- properly anonymized examples
- explicitly authorized test data

Generated output, cache, temporary, and evidence directories should be excluded using `.gitignore`.

## Limitations

- This is a research/prototype pipeline.
- OCR may miss, reorder, merge, split, or incorrectly recognize visible text.
- Vision-model extraction may alter message segmentation.
- The strict Message-only pass is conservative and may leave complex OCR errors unchanged.
- Sender/receiver attribution is evidence-constrained but not guaranteed.
- A screenshot containing multiple unrelated conversations may violate the fixed-pair assumption.
- A changing screenshot owner may invalidate conversation continuity.
- Collage boundaries may be detected incorrectly.
- Messenger timestamps may represent screen-level rather than message-level time.
- Filename/path classification depends on sensible evidence organization.
- The final CSV should be manually reviewed before being treated as final evidence.

For faster iteration, test:

1. one representative screenshot
2. the associated case report
3. one small evidence package

before processing a complete dataset.

## License

This project is licensed under the Apache License 2.0.

See the [LICENSE](LICENSE) file for details.
