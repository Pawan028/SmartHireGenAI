# SmartHire ATS Pro

Production-grade ATS screening with deterministic scoring, Gemini Flash fallback models, explicit Dark/Light themes, and two deployment paths:

- Streamlit app (`app.py`) for full recruiter workflow
- Vercel app (`public/` + `api/index.py`) for serverless deployment

## Core features

- Multi-resume ranking and ATS scorecards
- Skills/keywords/experience/formatting scoring
- Resume improvement suggestions and rewrite guidance
- AI deep analysis with fallback model order:
  - `gemini-3-flash-preview`
  - `gemini-2.5-flash`
  - `gemini-2.5-flash-lite`
  - `gemini-2.0-flash`
  - `gemini-2.0-flash-lite`
- Scanned-PDF fallback using Gemini vision extraction
- Dark and Light theme support in UI

## Local setup (Streamlit)

```bash
pip install -r requirements.txt
```

Create `.env`:

```env
GOOGLE_API_KEY=your_key_here
```

Run:

```bash
streamlit run app.py
```

## Vercel deployment

This repo now includes:

- `vercel.json`
- `api/index.py` (FastAPI serverless endpoint)
- `public/index.html`, `public/styles.css`, `public/app.js`

### Steps

1. Import this repo in Vercel.
2. Set environment variable in Vercel Project Settings:
   - `GOOGLE_API_KEY` = your Gemini key
3. Deploy.

### Endpoints

- `GET /api/health`
- `POST /api/analyze` (multipart form: `resume`, `job_description`, optional `enable_ai`, `temperature`)

## Security notes

- `.env` is ignored and not tracked.
- Use `.env.example` as template.
- Never commit API keys.
