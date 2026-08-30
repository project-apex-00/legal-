import os
import json
import time
import uuid
import threading
import traceback
from datetime import datetime
from collections import deque

from dotenv import load_dotenv
from flask import Flask, request, render_template, send_from_directory, jsonify
from xhtml2pdf import pisa
from jinja2 import Environment, FileSystemLoader
from google import genai
from google.genai import types
from PIL import Image

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

ai_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

GEMINI_MAX_CONCURRENT = int(os.environ.get("GEMINI_MAX_CONCURRENT", "2"))
GEMINI_MAX_RPM = int(os.environ.get("GEMINI_MAX_RPM", "10"))
GEMINI_MAX_RETRIES = int(os.environ.get("GEMINI_MAX_RETRIES", "3"))
GEMINI_RETRY_BASE_DELAY = float(os.environ.get("GEMINI_RETRY_BASE_DELAY", "3"))

MAX_IMAGE_DIMENSION = 1600  # downscale large photos before sending to Gemini

BASE_DIR = os.path.dirname(__file__)
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
JOBS_DIR = os.path.join(BASE_DIR, "jobs")

for d in (UPLOAD_DIR, OUTPUT_DIR, JOBS_DIR):
    os.makedirs(d, exist_ok=True)

app = Flask(__name__)
jinja_env = Environment(loader=FileSystemLoader("templates"))

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# ---------------------------------------------------------------------------
# Mandatory declarations under Legal Metrology (Packaged Commodities)
# Rules, 2011 — this list drives both the Gemini prompt and the UI field
# chips, so it lives in one place.
# ---------------------------------------------------------------------------

DECLARATION_FIELDS = [
    {
        "field": "manufacturer_address",
        "label": "Manufacturer / Packer / Importer",
        "rule": "Full name and complete address must be present.",
    },
    {
        "field": "common_name",
        "label": "Common / Generic Name",
        "rule": "The common or generic name of the commodity must be clearly stated.",
    },
    {
        "field": "net_quantity",
        "label": "Net Quantity",
        "rule": "Must be declared in standard units (g/kg, ml/l, or count) and clearly printed.",
    },
    {
        "field": "mfg_date",
        "label": "Month & Year of Manufacture/Packing/Import",
        "rule": "Must be present, in MM/YYYY or an equivalent unambiguous format.",
    },
    {
        "field": "mrp",
        "label": "Maximum Retail Price (MRP)",
        "rule": 'Must be in ₹ and include "inclusive of all taxes" or equivalent wording.',
    },
    {
        "field": "consumer_care",
        "label": "Consumer Care Details",
        "rule": "A phone number, email, or address for consumer complaints must be present.",
    },
    {
        "field": "country_of_origin",
        "label": "Country of Origin",
        "rule": "Required if the product is imported.",
    },
    {
        "field": "expiry_or_best_before",
        "label": "Best Before / Use By / Expiry Date",
        "rule": "Required for perishables, food, cosmetics, and pharmaceuticals.",
    },
]

# ---------------------------------------------------------------------------
# Gemini call wrapper — rolling rate limiter + concurrency gate + backoff,
# same resilience pattern used in the CMPDI report generator.
# ---------------------------------------------------------------------------

class GeminiRateLimitError(RuntimeError):
    pass

class _RollingRateLimiter:
    def __init__(self, max_per_minute):
        self.max_per_minute = max_per_minute
        self._calls = deque()
        self._lock = threading.Lock()

    def wait_for_slot(self):
        while True:
            with self._lock:
                now = time.monotonic()
                while self._calls and now - self._calls[0] > 60:
                    self._calls.popleft()
                if len(self._calls) < self.max_per_minute:
                    self._calls.append(now)
                    return
                sleep_for = 60 - (now - self._calls[0]) + 0.05
            time.sleep(max(sleep_for, 0.05))

_rate_limiter = _RollingRateLimiter(GEMINI_MAX_RPM)
_concurrency_gate = threading.Semaphore(GEMINI_MAX_CONCURRENT)

def _is_quota_error(err) -> bool:
    msg = str(err).upper()
    return any(token in msg for token in ("429", "RESOURCE_EXHAUSTED", "QUOTA", "RATE LIMIT"))

def get_gemini_client():
    global ai_client
    if not ai_client:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")
        ai_client = genai.Client(api_key=api_key)
    return ai_client

def call_gemini_vision(prompt: str, image_paths: list, temperature: float = 0.1) -> str:
    """Send a text prompt plus one or more images to Gemini and return the
    raw text response. Wrapped with the same rate-limit / retry / backoff
    behaviour as the rest of the pipeline."""
    client = get_gemini_client()

    contents = [prompt]
    for path in image_paths:
        img = Image.open(path)
        contents.append(img)

    config = types.GenerateContentConfig(
        temperature=temperature,
        response_mime_type="application/json",
    )

    candidate_models = [GEMINI_MODEL, "gemini-2.5-flash", "gemini-2.0-flash"]
    last_err = None
    hit_quota = False

    with _concurrency_gate:
        for model in candidate_models:
            for attempt in range(GEMINI_MAX_RETRIES):
                _rate_limiter.wait_for_slot()
                try:
                    response = client.models.generate_content(
                        model=model,
                        contents=contents,
                        config=config,
                    )
                    return response.text or ""
                except Exception as e:
                    last_err = e
                    if _is_quota_error(e):
                        hit_quota = True
                        delay = GEMINI_RETRY_BASE_DELAY * (2 ** attempt)
                        print(f"[gemini] rate limited on {model} "
                              f"(attempt {attempt + 1}/{GEMINI_MAX_RETRIES}), backing off {delay:.1f}s")
                        time.sleep(delay)
                        continue
                    break

    if hit_quota:
        raise GeminiRateLimitError(
            "Gemini free-tier rate limit reached. Wait a minute and try again, "
            "or lower GEMINI_MAX_CONCURRENT / GEMINI_MAX_RPM."
        )
    raise RuntimeError(f"Gemini API request failed across candidate models: {str(last_err)}")

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def build_extraction_prompt() -> str:
    field_lines = "\n".join(
        f'{i + 1}. {f["label"]} (field key: "{f["field"]}") — Rule: {f["rule"]}'
        for i, f in enumerate(DECLARATION_FIELDS)
    )
    return f"""You are a Legal Metrology compliance inspector analysing packaged
commodity label images under the Legal Metrology (Packaged Commodities)
Rules, 2011.

You are given images of a product's FRONT label and BACK label (there may
be one or two images; use whichever are provided). Read all visible text
on the label(s) carefully, including small print.

For EACH of the following mandatory declarations, determine:
- whether it is present on the label(s)
- the exact value/text found (verbatim, as printed)
- which image it was found on: "front", "back", or "unknown"
- whether it appears compliant based on the rule given
- a short, specific reason if it is missing or non-compliant (else null)

Declarations to check:
{field_lines}

Also note, in "additional_observations", any other visibly inconsistent,
misleading, or non-standard declaration you notice on the label, even if
not explicitly listed above (e.g. multiple conflicting MRPs, unit
mismatches, illegible or extremely small mandatory text).

Respond with STRICT JSON ONLY — no markdown, no commentary — matching
exactly this schema:

{{
  "product_name_guess": "string or null",
  "declarations": [
    {{
      "field": "one of the field keys above",
      "label": "the human-readable label",
      "found": true or false,
      "value": "extracted text or null",
      "source_image": "front" | "back" | "unknown" | null,
      "compliant": true or false,
      "reason": "short explanation if missing/non-compliant, else null"
    }}
  ],
  "additional_observations": ["short strings, or empty list"],
  "overall_verdict": "COMPLIANT" | "NON_COMPLIANT" | "NEEDS_REVIEW"
}}

Include exactly one object in "declarations" for every field key listed
above, in the same order. Do not invent extra top-level keys.
"""

# ---------------------------------------------------------------------------
# Job storage helpers
# ---------------------------------------------------------------------------

def job_path_for(job_id):
    return os.path.join(JOBS_DIR, f"{job_id}.json")

def save_job(job_id, payload):
    with open(job_path_for(job_id), "w") as f:
        json.dump(payload, f)

def load_job(job_id):
    path = job_path_for(job_id)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

def resize_if_needed(path):
    """Downscale very large photos so Gemini calls stay fast/cheap."""
    try:
        img = Image.open(path)
        w, h = img.size
        longest = max(w, h)
        if longest > MAX_IMAGE_DIMENSION:
            scale = MAX_IMAGE_DIMENSION / float(longest)
            new_size = (int(w * scale), int(h * scale))
            img = img.convert("RGB")
            img = img.resize(new_size)
            img.save(path, quality=88)
    except Exception:
        # Non-fatal — worst case Gemini gets a larger image than ideal.
        traceback.print_exc()

def parse_gemini_json(raw_text: str) -> dict:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    return json.loads(cleaned)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html", fields=DECLARATION_FIELDS)

@app.route("/scan", methods=["POST"])
def scan():
    """Accepts front (required) and back (optional) images, runs a single
    Gemini multimodal call, stores the structured result as a job."""
    front = request.files.get("front")
    back = request.files.get("back")

    if not front:
        return jsonify({"error": "Front label image is required."}), 400

    job_id = uuid.uuid4().hex[:10]
    image_paths = []

    for label, f in (("front", front), ("back", back)):
        if not f:
            continue
        ext = os.path.splitext(f.filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            return jsonify({"error": f"Unsupported file type: {ext}"}), 400
        path = os.path.join(UPLOAD_DIR, f"{job_id}_{label}{ext}")
        f.save(path)
        resize_if_needed(path)
        image_paths.append(path)

    start_time = time.time()
    try:
        prompt = build_extraction_prompt()
        raw = call_gemini_vision(prompt, image_paths)
        result = parse_gemini_json(raw)

        elapsed_seconds = round(time.time() - start_time, 1)

        job_data = {
            "job_id": job_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "generation_seconds": elapsed_seconds,
            "image_paths": image_paths,
            "result": result,
        }
        save_job(job_id, job_data)

        return jsonify({
            "success": True,
            "job_id": job_id,
            "elapsed_seconds": elapsed_seconds,
            "result": result,
        })

    except GeminiRateLimitError as e:
        elapsed_seconds = round(time.time() - start_time, 1)
        return jsonify({"error": str(e), "rate_limited": True,
                         "elapsed_seconds": elapsed_seconds}), 429
    except json.JSONDecodeError:
        return jsonify({"error": "Could not parse the model's response. Please try again."}), 500
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/field/<job_id>/<field_key>")
def field_lookup(job_id, field_key):
    """Field-chip lookup — no new Gemini call, just filters the already
    extracted JSON for the tapped field (e.g. 'mrp', 'expiry_or_best_before')."""
    job_data = load_job(job_id)
    if job_data is None:
        return jsonify({"error": "Unknown job — please scan again."}), 404

    declarations = job_data["result"].get("declarations", [])
    match = next((d for d in declarations if d["field"] == field_key), None)
    if match is None:
        return jsonify({"error": f"Unknown field: {field_key}"}), 404

    return jsonify({"success": True, "field": match})

@app.route("/report/<job_id>", methods=["POST"])
def report(job_id):
    job_data = load_job(job_id)
    if job_data is None:
        return jsonify({"error": "Unknown job — please scan again."}), 404

    try:
        result = job_data["result"]
        declarations = result.get("declarations", [])
        violations = [d for d in declarations if not d.get("compliant", False)]

        report_data = {
            "product_name_guess": result.get("product_name_guess") or "Unidentified product",
            "generated_on": datetime.now().strftime("%d %b %Y, %I:%M %p"),
            "job_id": job_id,
            "generation_seconds": job_data.get("generation_seconds"),
            "declarations": declarations,
            "violations": violations,
            "additional_observations": result.get("additional_observations", []),
            "overall_verdict": result.get("overall_verdict", "NEEDS_REVIEW"),
            "violation_count": len(violations),
        }

        out_pdf_name = f"{job_id}_compliance_report.pdf"
        out_pdf_path = os.path.join(OUTPUT_DIR, out_pdf_name)

        template = jinja_env.get_template("report_template.html")
        html_str = template.render(report=report_data)
        with open(out_pdf_path, "wb") as f:
            pisa_result = pisa.CreatePDF(html_str, dest=f)
        if pisa_result.err:
            raise RuntimeError("xhtml2pdf failed to render the compliance report.")

        return jsonify({
            "success": True,
            "report": report_data,
            "pdf_url": f"/download/{out_pdf_name}",
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/download/<path:filename>")
def download(filename):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=False)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
