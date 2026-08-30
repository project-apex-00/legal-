# Legal Metrology Compliance Scanner

Upload the front and back label images of a packaged commodity. A single
Gemini multimodal API call reads both images, extracts every mandatory
declaration required under the Legal Metrology (Packaged Commodities)
Rules, 2011, and flags anything missing or non-compliant. Results are
shown as tap-to-inspect field chips, and a formatted PDF compliance
report can be generated on demand.

## How it works

```
Upload front + back images
        ↓
Single Gemini multimodal call (images + rules-aware prompt)
        ↓
Structured JSON: every declaration -> value, source image,
compliant/violation, reason
        ↓
UI: color-coded field chips (tap any field for detail)
        ↓
"Generate Report" -> Jinja2 + xhtml2pdf renders a downloadable
compliance report PDF
```

No word-cloud/topic module, no chunking, no multi-call merge step —
labels are short enough that one call handles the full extraction.

## 1. Prerequisites

- Python 3.9+
- A Google Gemini API key: https://aistudio.google.com/apikey

## 2. Install

```bash
cd legal-metrology-app
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# then edit .env and paste in your GEMINI_API_KEY
```

## 3. Run

```bash
python app.py
```

Open **http://127.0.0.1:5000**.

## 4. Use it

1. Upload the front label photo (required) and back label photo (optional
   but recommended — most declarations live on the back panel).
2. Click **Analyze Compliance**. One Gemini call extracts and checks every
   declaration.
3. Tap any field chip (MRP, Net Quantity, Expiry Date, etc.) to see the
   exact detected value, which image it came from, and why it passed or
   failed.
4. Click **Generate PDF Report** for a formatted compliance report with
   a full declaration table, violations summary, and additional
   observations.

## 5. Config knobs (`.env`)

| Variable | What it does |
|---|---|
| `GEMINI_MODEL` | Which Gemini model to call |
| `GEMINI_MAX_CONCURRENT` | Concurrent Gemini calls allowed at once |
| `GEMINI_MAX_RPM` | Rolling requests-per-minute cap (free-tier safety) |
| `GEMINI_MAX_RETRIES` / `GEMINI_RETRY_BASE_DELAY` | Backoff behaviour on 429s |

## 6. Known limitations (honest scope, Phase 2 candidates)

- **Font-size / physical readability compliance is not measured.**
  Gemini can read the text, but converting pixel height to real-world mm
  requires a reference object or known package dimensions in the photo —
  not implemented here.
- **English/Hindi labels are best supported**; other regional-language
  labels may extract less reliably.
- **Curved or reflective packaging** (bottles, tins) will have lower OCR
  accuracy than flat labels — always double-check flagged violations
  against the physical product before enforcement action.
- The tool produces a **draft compliance report**, not a final legal
  determination — a human reviewer stays in the loop, matching how the
  UI surfaces every extracted field for verification rather than an
  opaque pass/fail.

## Folder structure

```
legal-metrology-app/
├── app.py                       # Flask backend (upload, Gemini call, rendering)
├── requirements.txt
├── Procfile                     # gunicorn entrypoint for deployment
├── .env.example
├── templates/
│   ├── index.html               # Upload UI + field-chip results
│   └── report_template.html     # Compliance report template (Jinja2)
├── uploads/                     # Uploaded label images land here (gitignored)
├── jobs/                        # Stored scan results as JSON (gitignored)
└── output/                      # Generated report PDFs land here (gitignored)
```
