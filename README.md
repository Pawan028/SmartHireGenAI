# SmartHire ATS Pro

Production-grade ATS screening app built with Streamlit and Gemini Flash model fallbacks.

## What is new

- Multi-resume screening and ranking (up to 15 resumes in one run)
- Deterministic ATS scoring engine:
  - Keywords match
  - Skills match
  - Experience fit
  - Formatting readiness
- Resume gap analysis with concrete, actionable suggestions
- Structured AI deep analysis with JSON output validation
- Gemini fallback chain optimized for free-tier friendly Flash models:
  - `gemini-3-flash-preview`
  - `gemini-2.5-flash`
  - `gemini-2.5-flash-lite`
  - `gemini-2.0-flash`
  - `gemini-2.0-flash-lite`
- Scanned PDF fallback using Gemini vision extraction when text parsing is weak
- Export candidate report as Markdown

## Architecture highlights

- ATS scoring is deterministic and always available
- AI generation is optional and fault tolerant
- Automatic retry and model fallback for reliability
- Session caching for PDF text extraction
- Strict JSON parsing/validation for deep-analysis responses

## Setup

1. Create and activate your environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set Gemini key in `.env`:

```env
GOOGLE_API_KEY=your_key_here
```

4. Run the app:

```bash
streamlit run app.py
```

## Usage flow

1. Paste a job description.
2. Upload one or more PDF resumes.
3. Click **Run ATS Screening**.
4. Review ranking and candidate-level gaps.
5. Click **Generate AI Deep Analysis** for rewrite-ready insights.
6. Download a report for the selected candidate.

## Notes

- If no API key is configured, the app still works in deterministic heuristic mode.
- For best ATS reliability, use text-based PDFs instead of image-only scans.
