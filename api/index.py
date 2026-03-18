import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import fitz
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app import (
    FREE_TIER_MODEL_CANDIDATES,
    build_quick_suggestions,
    build_resume_profile,
    configure_gemini,
    deep_analysis_with_fallback,
    heuristic_deep_analysis,
    score_resume_against_job,
)

MAX_FILE_SIZE_MB = 8
BASE_DIR = Path(__file__).resolve().parent.parent
PUBLIC_DIR = BASE_DIR / "public"

app = FastAPI(title="SmartHire ATS API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    if not pdf_bytes:
        return ""
    pages: List[str] = []
    doc = None
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page in doc:
            page_text = page.get_text("text", sort=True)
            if page_text:
                pages.append(page_text)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse PDF: {exc}") from exc
    finally:
        if doc:
            doc.close()
    return " ".join(pages).strip()


@app.get("/")
def root() -> Any:
    index_file = PUBLIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file), media_type="text/html")
    return {"ok": True, "message": "SmartHire ATS API is running. Frontend file not found."}


@app.get("/styles.css")
def styles() -> Any:
    css_file = PUBLIC_DIR / "styles.css"
    if css_file.exists():
        return FileResponse(str(css_file), media_type="text/css")
    raise HTTPException(status_code=404, detail="styles.css not found")


@app.get("/app.js")
def app_js() -> Any:
    js_file = PUBLIC_DIR / "app.js"
    if js_file.exists():
        return FileResponse(str(js_file), media_type="application/javascript")
    raise HTTPException(status_code=404, detail="app.js not found")


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "service": "smarthire-ats-api", "time_utc": datetime.utcnow().isoformat()}


@app.post("/api/analyze")
async def analyze(
    job_description: str = Form(...),
    resume: UploadFile = File(...),
    enable_ai: bool = Form(True),
    temperature: float = Form(0.2),
    model_order: str = Form("gemini-3-flash-preview,gemini-2.5-flash,gemini-2.5-flash-lite"),
) -> Dict[str, Any]:
    if not job_description.strip():
        raise HTTPException(status_code=400, detail="Job description is required.")

    file_bytes = await resume.read()
    file_size_mb = len(file_bytes) / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"Resume is too large. Max allowed size is {MAX_FILE_SIZE_MB} MB.",
        )

    resume_text = extract_text_from_pdf_bytes(file_bytes)
    if len(resume_text) < 80:
        raise HTTPException(
            status_code=400,
            detail="Could not extract enough text from resume. Please upload a text-based PDF.",
        )

    profile = build_resume_profile(resume.filename or "resume.pdf", resume_text)
    score = score_resume_against_job(profile, job_description)
    quick_suggestions = build_quick_suggestions(profile, score)

    models = [m.strip() for m in model_order.split(",") if m.strip()]
    if not models:
        models = FREE_TIER_MODEL_CANDIDATES[:3]

    ai_warning = None
    deep_analysis = heuristic_deep_analysis(profile, score, job_description)
    if enable_ai:
        api_key = os.getenv("GOOGLE_API_KEY", "").strip()
        if api_key:
            configure_gemini(api_key)
            try:
                deep_analysis = deep_analysis_with_fallback(
                    profile=profile,
                    score=score,
                    job_description=job_description,
                    model_candidates=models,
                    temperature=max(0.0, min(1.0, float(temperature))),
                )
            except Exception as exc:
                ai_warning = str(exc)
        else:
            ai_warning = "GOOGLE_API_KEY is not configured on the deployment."

    return {
        "candidate": {
            "file_name": profile.file_name,
            "word_count": profile.word_count,
            "years_experience": profile.years_experience,
            "email": profile.email,
            "phone": profile.phone,
            "linkedin": profile.linkedin,
            "github": profile.github,
        },
        "score_card": asdict(score),
        "quick_suggestions": quick_suggestions,
        "deep_analysis": deep_analysis,
        "ai_warning": ai_warning,
    }
