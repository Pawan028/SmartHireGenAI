import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import fitz
import google.generativeai as genai
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

FREE_TIER_MODEL_CANDIDATES: List[str] = [
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
]

MAX_FILE_SIZE_MB = 8
MAX_RESUME_COUNT = 15
MAX_RESUME_CHARS_FOR_AI = 18000
MAX_JOB_DESC_CHARS_FOR_AI = 8000

STOPWORDS = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "me",
    "more",
    "most",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "now",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "with",
    "would",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "will",
    "must",
    "role",
    "candidate",
    "team",
    "work",
    "working",
    "job",
    "responsibilities",
    "requirements",
    "preferred",
    "experience",
    "years",
    "year",
    "strong",
    "ability",
    "skills",
    "knowledge",
    "using",
    "including",
}

SKILL_TAXONOMY = {
    "python",
    "java",
    "javascript",
    "typescript",
    "go",
    "rust",
    "c++",
    "c#",
    "sql",
    "nosql",
    "postgresql",
    "mysql",
    "mongodb",
    "redis",
    "snowflake",
    "bigquery",
    "oracle",
    "sqlite",
    "pandas",
    "numpy",
    "scikit-learn",
    "tensorflow",
    "pytorch",
    "mlops",
    "machine learning",
    "deep learning",
    "nlp",
    "computer vision",
    "llm",
    "prompt engineering",
    "data analysis",
    "data science",
    "data engineering",
    "etl",
    "airflow",
    "dbt",
    "spark",
    "hadoop",
    "kafka",
    "power bi",
    "tableau",
    "excel",
    "statistics",
    "ab testing",
    "forecasting",
    "aws",
    "azure",
    "gcp",
    "docker",
    "kubernetes",
    "terraform",
    "ansible",
    "linux",
    "bash",
    "git",
    "github",
    "gitlab",
    "ci/cd",
    "jenkins",
    "devops",
    "sre",
    "microservices",
    "rest",
    "graphql",
    "fastapi",
    "flask",
    "django",
    "streamlit",
    "react",
    "next.js",
    "agile",
    "scrum",
    "jira",
    "project management",
    "communication",
    "leadership",
}

SECTION_PATTERNS = {
    "summary": ["summary", "professional summary", "profile"],
    "experience": ["experience", "work history", "employment"],
    "skills": ["skills", "technical skills", "core competencies"],
    "education": ["education", "academic", "qualification"],
    "projects": ["projects", "project experience"],
    "certifications": ["certifications", "certificates"],
}


@dataclass
class ResumeProfile:
    candidate_id: str
    file_name: str
    text: str
    word_count: int
    email: Optional[str]
    phone: Optional[str]
    linkedin: Optional[str]
    github: Optional[str]
    sections: Dict[str, bool]
    years_experience: float


@dataclass
class ScoreCard:
    overall_score: int
    skills_match: int
    experience_match: int
    keywords_match: int
    formatting_score: int
    matched_skills: List[str]
    missing_skills: List[str]
    missing_keywords: List[str]
    summary: str


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z0-9+#\./\-]{1,}", text.lower())


def keyword_present(text: str, keyword: str) -> bool:
    if not text or not keyword:
        return False
    pattern = rf"(?<!\w){re.escape(keyword.lower())}(?!\w)"
    return re.search(pattern, text.lower()) is not None


def parse_json_object(raw: str) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    cleaned = raw.strip().replace("```json", "").replace("```", "").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    snippet = cleaned[start : end + 1]
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        return None


def clamp(value: float, low: int = 0, high: int = 100) -> int:
    return int(max(low, min(high, round(value))))


def short_text(text: str, max_chars: int) -> str:
    return normalize_whitespace(text)[:max_chars]


def ensure_list(value: Any, limit: int = 8) -> List[str]:
    if not isinstance(value, list):
        return []
    return [normalize_whitespace(str(item)) for item in value if normalize_whitespace(str(item))][:limit]


def ensure_dict_list(value: Any, limit: int = 8) -> List[Dict[str, str]]:
    if not isinstance(value, list):
        return []
    output: List[Dict[str, str]] = []
    for item in value:
        if isinstance(item, dict):
            cleaned = {str(k): normalize_whitespace(str(v)) for k, v in item.items()}
            if cleaned:
                output.append(cleaned)
        if len(output) >= limit:
            break
    return output


@st.cache_data(show_spinner=False)
def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    if not pdf_bytes:
        return ""
    doc = None
    pages: List[str] = []
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page_idx in range(len(doc)):
            page_text = doc.load_page(page_idx).get_text("text", sort=True)
            if page_text:
                pages.append(page_text)
    except Exception:
        return ""
    finally:
        if doc:
            doc.close()
    return normalize_whitespace("\n".join(pages))


def extract_contact_info(text: str) -> Dict[str, Optional[str]]:
    email_match = re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", text)
    phone_match = re.search(r"(?:\+\d{1,3}[\s-]?)?(?:\(?\d{3}\)?[\s-]?)?\d{3}[\s-]?\d{4}", text)
    linkedin_match = re.search(
        r"(?:https?://)?(?:www\.)?linkedin\.com/[A-Za-z0-9_\-/]+", text, re.IGNORECASE
    )
    github_match = re.search(r"(?:https?://)?(?:www\.)?github\.com/[A-Za-z0-9_\-]+", text, re.IGNORECASE)
    return {
        "email": email_match.group(0) if email_match else None,
        "phone": phone_match.group(0) if phone_match else None,
        "linkedin": linkedin_match.group(0) if linkedin_match else None,
        "github": github_match.group(0) if github_match else None,
    }


def detect_sections(text: str) -> Dict[str, bool]:
    lowered = text.lower()
    sections: Dict[str, bool] = {}
    for section_name, variants in SECTION_PATTERNS.items():
        sections[section_name] = any(keyword_present(lowered, variant) for variant in variants)
    return sections


def extract_years_experience(text: str) -> float:
    lowered = text.lower()
    years_matches = [int(x) for x in re.findall(r"\b(\d{1,2})\+?\s*(?:years|yrs)\b", lowered)]
    if years_matches:
        return float(max(years_matches))
    found_years = [int(x) for x in re.findall(r"\b(?:19|20)\d{2}\b", lowered)]
    if len(found_years) >= 2:
        span = max(found_years) - min(found_years)
        if 0 < span <= 45:
            return float(span)
    return 0.0


def build_resume_profile(file_name: str, text: str) -> ResumeProfile:
    normalized = normalize_whitespace(text)
    digest = hashlib.sha256(f"{file_name}:{normalized[:5000]}".encode("utf-8")).hexdigest()[:16]
    contacts = extract_contact_info(normalized)
    return ResumeProfile(
        candidate_id=digest,
        file_name=file_name,
        text=normalized,
        word_count=len(normalized.split()),
        email=contacts["email"],
        phone=contacts["phone"],
        linkedin=contacts["linkedin"],
        github=contacts["github"],
        sections=detect_sections(normalized),
        years_experience=extract_years_experience(normalized),
    )


def extract_required_years(job_description: str) -> Optional[float]:
    lowered = job_description.lower()
    years = [int(x) for x in re.findall(r"\b(\d{1,2})\+?\s*(?:years|yrs)\b", lowered)]
    if years:
        return float(max(years))
    return None


def extract_job_keywords(job_description: str, top_k: int = 35) -> List[str]:
    lowered = job_description.lower()
    tokens = tokenize(job_description)
    freq: Dict[str, int] = {}
    for token in tokens:
        if len(token) < 3 or token in STOPWORDS:
            continue
        freq[token] = freq.get(token, 0) + 1
    ranked_tokens = sorted(freq.items(), key=lambda item: (-item[1], -len(item[0]), item[0]))
    keywords = [token for token, _ in ranked_tokens[:top_k]]
    jd_skills = [skill for skill in SKILL_TAXONOMY if keyword_present(lowered, skill)]
    for skill in sorted(jd_skills, key=len, reverse=True):
        if skill not in keywords:
            keywords.insert(0, skill)
    unique_keywords: List[str] = []
    seen = set()
    for keyword in keywords:
        if keyword not in seen:
            unique_keywords.append(keyword)
            seen.add(keyword)
        if len(unique_keywords) >= top_k:
            break
    return unique_keywords


def detect_skills(text: str) -> List[str]:
    lowered = text.lower()
    return sorted([skill for skill in SKILL_TAXONOMY if keyword_present(lowered, skill)])


def formatting_score(profile: ResumeProfile) -> int:
    contact_points = 0
    if profile.email:
        contact_points += 14
    if profile.phone:
        contact_points += 14
    if profile.linkedin or profile.github:
        contact_points += 12
    section_points = 0
    for section in ["summary", "experience", "skills", "education"]:
        if profile.sections.get(section):
            section_points += 12
    if profile.sections.get("projects"):
        section_points += 6
    if profile.sections.get("certifications"):
        section_points += 6
    word_count_points = 0
    if 350 <= profile.word_count <= 1200:
        word_count_points = 16
    elif 250 <= profile.word_count < 350 or 1200 < profile.word_count <= 1600:
        word_count_points = 10
    elif 120 <= profile.word_count < 250:
        word_count_points = 6
    return clamp(contact_points + section_points + word_count_points)


def build_score_summary(overall: int) -> str:
    if overall >= 85:
        return "Strong shortlist candidate with high ATS alignment."
    if overall >= 70:
        return "Good alignment, shortlist after minor resume optimization."
    if overall >= 55:
        return "Moderate fit. Needs targeted keyword and skills alignment."
    return "Low match today. Significant resume targeting needed before applying."


def score_resume_against_job(profile: ResumeProfile, job_description: str) -> ScoreCard:
    resume_lower = profile.text.lower()
    job_keywords = extract_job_keywords(job_description)
    missing_keywords = [kw for kw in job_keywords if not keyword_present(resume_lower, kw)]
    keyword_match_ratio = 1.0 - (len(missing_keywords) / max(1, len(job_keywords)))
    keywords_match = clamp(keyword_match_ratio * 100)
    resume_skills = set(detect_skills(profile.text))
    required_skills = {skill for skill in SKILL_TAXONOMY if keyword_present(job_description.lower(), skill)}
    if required_skills:
        missing_skills = sorted(required_skills - resume_skills)
        matched_skills = sorted(required_skills & resume_skills)
        skills_match = clamp((len(matched_skills) / len(required_skills)) * 100)
    else:
        missing_skills = []
        matched_skills = sorted(resume_skills)[:20]
        skills_match = keywords_match
    required_years = extract_required_years(job_description)
    if required_years is None:
        experience_match = 75 if profile.years_experience >= 1 else 55
    else:
        delta = profile.years_experience - required_years
        if delta >= 0:
            experience_match = clamp(82 + min(delta * 4, 18))
        else:
            experience_match = clamp(82 + delta * 12, low=25)
    fmt_score = formatting_score(profile)
    overall = clamp((0.35 * keywords_match) + (0.30 * skills_match) + (0.20 * experience_match) + (0.15 * fmt_score))
    prioritized_missing = missing_skills + [kw for kw in missing_keywords if kw not in missing_skills]
    return ScoreCard(
        overall_score=overall,
        skills_match=skills_match,
        experience_match=experience_match,
        keywords_match=keywords_match,
        formatting_score=fmt_score,
        matched_skills=matched_skills[:20],
        missing_skills=missing_skills[:15],
        missing_keywords=prioritized_missing[:15],
        summary=build_score_summary(overall),
    )


def build_quick_suggestions(profile: ResumeProfile, score: ScoreCard) -> List[str]:
    suggestions: List[str] = []
    if score.keywords_match < 70:
        missing = ", ".join(score.missing_keywords[:6]) if score.missing_keywords else "role-specific keywords"
        suggestions.append(f"Add missing JD keywords naturally in experience/projects: {missing}.")
    if score.skills_match < 70 and score.missing_skills:
        missing_skills = ", ".join(score.missing_skills[:5])
        suggestions.append(f"Create a dedicated technical skills block including: {missing_skills}.")
    if score.experience_match < 65:
        suggestions.append(
            "Strengthen achievement bullets with measurable impact (%, $, time saved, volume handled) for each role."
        )
    if score.formatting_score < 70:
        suggestions.append(
            "Use ATS-safe layout: single-column PDF, clear section headers, and standard fonts without graphics-heavy elements."
        )
    if not profile.sections.get("summary"):
        suggestions.append("Add a 3-4 line professional summary tailored to the exact target role.")
    if not profile.sections.get("projects"):
        suggestions.append("Add a projects section with stack, scope, and outcomes to improve recruiter confidence.")
    if not profile.linkedin and not profile.github:
        suggestions.append("Add LinkedIn/GitHub profile links near contact details to improve profile credibility.")
    if not suggestions:
        suggestions.append("Resume is already strong. Focus on tailoring language to each job posting before applying.")
    return suggestions[:6]


def configure_gemini(api_key: str) -> bool:
    if not api_key:
        return False
    genai.configure(api_key=api_key)
    return True


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=12), reraise=True)
def generate_text_once(
    model_name: str,
    prompt: str,
    parts: Optional[List[Dict[str, Any]]] = None,
    temperature: float = 0.2,
    max_tokens: int = 2200,
    expect_json: bool = True,
) -> str:
    model = genai.GenerativeModel(model_name=model_name)
    content: List[Any] = [prompt]
    if parts:
        content.extend(parts)
    config_kwargs: Dict[str, Any] = {"temperature": temperature, "max_output_tokens": max_tokens}
    if expect_json:
        try:
            config = genai.types.GenerationConfig(response_mime_type="application/json", **config_kwargs)
            response = model.generate_content(content, generation_config=config)
        except Exception:
            config = genai.types.GenerationConfig(**config_kwargs)
            response = model.generate_content(content, generation_config=config)
    else:
        config = genai.types.GenerationConfig(**config_kwargs)
        response = model.generate_content(content, generation_config=config)
    text = getattr(response, "text", "")
    if not text:
        raise RuntimeError("Model returned an empty response.")
    return text


def generate_with_fallback(
    prompt: str,
    model_candidates: List[str],
    parts: Optional[List[Dict[str, Any]]] = None,
    temperature: float = 0.2,
    max_tokens: int = 2200,
    expect_json: bool = True,
) -> Tuple[str, str]:
    if not model_candidates:
        raise RuntimeError("No model candidates were provided.")
    errors: List[str] = []
    for model_name in model_candidates:
        try:
            output = generate_text_once(
                model_name=model_name,
                prompt=prompt,
                parts=parts,
                temperature=temperature,
                max_tokens=max_tokens,
                expect_json=expect_json,
            )
            return output, model_name
        except Exception as exc:
            errors.append(f"{model_name}: {exc}")
    raise RuntimeError("All configured Gemini models failed. " + " | ".join(errors))


def pdf_to_image_parts(pdf_bytes: bytes, max_pages: int = 3) -> List[Dict[str, Any]]:
    parts: List[Dict[str, Any]] = []
    doc = None
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for idx in range(min(max_pages, len(doc))):
            page = doc.load_page(idx)
            pix = page.get_pixmap(matrix=fitz.Matrix(1.6, 1.6), alpha=False)
            png_data = pix.tobytes("png")
            parts.append({"mime_type": "image/png", "data": png_data})
    except Exception:
        return []
    finally:
        if doc:
            doc.close()
    return parts


def try_vision_text_extraction(pdf_bytes: bytes, model_candidates: List[str]) -> Tuple[str, Optional[str]]:
    parts = pdf_to_image_parts(pdf_bytes)
    if not parts:
        return "", None
    prompt = (
        "Extract all readable text from these resume pages. "
        "Return plain text only. Keep original ordering and include bullet points where possible."
    )
    try:
        text, model_used = generate_with_fallback(
            prompt=prompt,
            model_candidates=model_candidates,
            parts=parts,
            temperature=0.0,
            max_tokens=3000,
            expect_json=False,
        )
        return normalize_whitespace(text), model_used
    except Exception:
        return "", None


def build_deep_analysis_prompt(profile: ResumeProfile, score: ScoreCard, job_description: str) -> str:
    score_dict = asdict(score)
    schema = {
        "executive_summary": "string",
        "strengths": ["string"],
        "gaps": ["string"],
        "ats_risks": ["string"],
        "targeted_improvements": [
            {"section": "string", "issue": "string", "rewrite": "string", "impact": "string"}
        ],
        "optimized_professional_summary": "string",
        "interview_focus": ["string"],
        "first_30_day_plan": ["string"],
    }
    return f"""
You are a senior ATS strategist and resume writer.

Return ONLY a valid JSON object following this schema:
{json.dumps(schema, indent=2)}

Rules:
- No markdown.
- Keep all items concise and practical.
- Use concrete language from the provided job description.
- `targeted_improvements` must include at least 4 items.
- Each rewrite should be ATS-friendly and metric-focused.

Job Description:
{short_text(job_description, MAX_JOB_DESC_CHARS_FOR_AI)}

Resume Text:
{short_text(profile.text, MAX_RESUME_CHARS_FOR_AI)}

Deterministic ATS Scores:
{json.dumps(score_dict, indent=2)}
""".strip()


def deep_analysis_with_fallback(
    profile: ResumeProfile,
    score: ScoreCard,
    job_description: str,
    model_candidates: List[str],
    temperature: float,
) -> Dict[str, Any]:
    prompt = build_deep_analysis_prompt(profile, score, job_description)
    response_text, model_used = generate_with_fallback(
        prompt=prompt,
        model_candidates=model_candidates,
        temperature=temperature,
        max_tokens=2600,
        expect_json=True,
    )
    parsed = parse_json_object(response_text)
    if not parsed:
        raise RuntimeError("Model response was not valid JSON.")
    return {
        "model_used": model_used,
        "executive_summary": normalize_whitespace(str(parsed.get("executive_summary", ""))),
        "strengths": ensure_list(parsed.get("strengths"), limit=8),
        "gaps": ensure_list(parsed.get("gaps"), limit=8),
        "ats_risks": ensure_list(parsed.get("ats_risks"), limit=8),
        "targeted_improvements": ensure_dict_list(parsed.get("targeted_improvements"), limit=10),
        "optimized_professional_summary": normalize_whitespace(str(parsed.get("optimized_professional_summary", ""))),
        "interview_focus": ensure_list(parsed.get("interview_focus"), limit=8),
        "first_30_day_plan": ensure_list(parsed.get("first_30_day_plan"), limit=8),
    }


def heuristic_deep_analysis(profile: ResumeProfile, score: ScoreCard, job_description: str) -> Dict[str, Any]:
    _ = job_description
    suggestions = build_quick_suggestions(profile, score)
    targeted_improvements = [
        {
            "section": "Professional Summary",
            "issue": "Summary is not fully aligned with role requirements.",
            "rewrite": "Results-driven professional with hands-on experience in key role responsibilities and a track record of measurable impact across cross-functional teams.",
            "impact": "Improves recruiter relevance scan in the first 10 seconds.",
        },
        {
            "section": "Experience Bullets",
            "issue": "Bullets are likely task-focused instead of outcome-focused.",
            "rewrite": "Reframe bullets as action + tool + metric, e.g., 'Built X using Y, improving Z by 25%.'",
            "impact": "Boosts credibility and interview conversion.",
        },
        {
            "section": "Skills",
            "issue": "Important JD skills may be missing or buried.",
            "rewrite": "Use grouped skills subsections: Languages, Data, Cloud, Frameworks, Tools.",
            "impact": "Improves ATS extraction and keyword match rates.",
        },
        {
            "section": "Projects",
            "issue": "Projects may not reflect target role depth.",
            "rewrite": "Add 2 role-aligned projects with stack, scope, and quantified outcomes.",
            "impact": "Adds practical signal for hiring managers.",
        },
    ]
    return {
        "model_used": "heuristic-engine",
        "executive_summary": score.summary,
        "strengths": [
            "Resume includes core section structure used by ATS systems.",
            "Deterministic scoring indicates role alignment potential.",
        ],
        "gaps": suggestions,
        "ats_risks": [
            "Possible mismatch between resume wording and job description terminology.",
            "Insufficient quantified impact in bullets can reduce recruiter confidence.",
        ],
        "targeted_improvements": targeted_improvements,
        "optimized_professional_summary": "Target-role focused candidate with proven delivery across relevant tools and workflows, known for measurable execution, strong collaboration, and consistent outcome ownership.",
        "interview_focus": [
            "Explain role-relevant projects with measurable business results.",
            "Prepare examples of ownership, tradeoffs, and cross-team collaboration.",
            "Be ready to map each key JD requirement to specific past experience.",
        ],
        "first_30_day_plan": [
            "Week 1: map team goals and delivery expectations.",
            "Week 2: deep dive into core systems, tools, and metrics.",
            "Week 3: deliver one scoped improvement with measurable value.",
            "Week 4: propose a quarter roadmap aligned to business priorities.",
        ],
    }


def load_custom_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Source+Sans+3:wght@400;500;600;700&display=swap');
        :root {
            --bg: #f4f7f3;
            --text: #182218;
            --muted: #5b6a5d;
            --brand: #0f766e;
            --brand-soft: #d8f3ef;
            --line: #d4ded1;
        }
        .stApp {
            background: radial-gradient(circle at top left, #eef5ec 0%, #f7faf6 45%, #eef3f8 100%);
            color: var(--text);
            font-family: 'Source Sans 3', sans-serif;
        }
        h1, h2, h3 {
            font-family: 'Space Grotesk', sans-serif;
            color: var(--text);
        }
        .app-hero {
            border: 1px solid var(--line);
            background: linear-gradient(130deg, #ffffff 0%, #eef6f4 100%);
            border-radius: 18px;
            padding: 1.2rem 1.4rem;
            margin-bottom: 1rem;
        }
        .meta-pill {
            display: inline-block;
            padding: 0.2rem 0.55rem;
            border-radius: 999px;
            background: var(--brand-soft);
            color: var(--brand);
            border: 1px solid #b8e4dd;
            font-size: 0.78rem;
            margin-right: 0.4rem;
        }
        [data-testid="stMetricValue"] { color: var(--brand); }
        [data-testid="stSidebar"] {
            border-right: 1px solid var(--line);
            background: #fbfdfb;
        }
        .mini-note { color: var(--muted); font-size: 0.86rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_session_state() -> None:
    defaults = {"screening_results": [], "deep_analysis_cache": {}, "last_job_description": ""}
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def build_candidate_report_markdown(
    profile: ResumeProfile,
    score: ScoreCard,
    quick_suggestions: List[str],
    deep_analysis: Dict[str, Any],
) -> str:
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# ATS Candidate Report - {profile.file_name}",
        "",
        f"Generated: {date_str}",
        f"Model: {deep_analysis.get('model_used', 'N/A')}",
        "",
        "## Scorecard",
        f"- Overall: {score.overall_score}/100",
        f"- Skills: {score.skills_match}/100",
        f"- Keywords: {score.keywords_match}/100",
        f"- Experience: {score.experience_match}/100",
        f"- Formatting: {score.formatting_score}/100",
        "",
        "## Missing Keywords",
        f"- {', '.join(score.missing_keywords[:15]) if score.missing_keywords else 'None'}",
        "",
        "## Quick Suggestions",
    ]
    for item in quick_suggestions:
        lines.append(f"- {item}")
    lines.extend(["", "## Executive Summary", deep_analysis.get("executive_summary", ""), "", "## Strengths"])
    for item in deep_analysis.get("strengths", []):
        lines.append(f"- {item}")
    lines.extend(["", "## Gaps"])
    for item in deep_analysis.get("gaps", []):
        lines.append(f"- {item}")
    lines.extend(["", "## ATS Risks"])
    for item in deep_analysis.get("ats_risks", []):
        lines.append(f"- {item}")
    lines.extend(["", "## Targeted Improvements"])
    for item in deep_analysis.get("targeted_improvements", []):
        lines.append(
            f"- [{item.get('section', 'Section')}] {item.get('issue', '')} | Rewrite: {item.get('rewrite', '')} | Impact: {item.get('impact', '')}"
        )
    lines.extend(["", "## Optimized Summary", deep_analysis.get("optimized_professional_summary", ""), "", "## Interview Focus"])
    for item in deep_analysis.get("interview_focus", []):
        lines.append(f"- {item}")
    lines.extend(["", "## 30-Day Plan"])
    for item in deep_analysis.get("first_30_day_plan", []):
        lines.append(f"- {item}")
    return "\n".join(lines)


def main() -> None:
    st.set_page_config(
        page_title="SmartHire ATS Pro",
        page_icon="📌",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    load_custom_css()
    init_session_state()
    env_api_key = os.getenv("GOOGLE_API_KEY", "").strip()

    with st.sidebar:
        st.header("Runtime Settings")
        override_key = st.text_input("Gemini API key override", type="password")
        effective_api_key = override_key.strip() or env_api_key
        if effective_api_key:
            configure_gemini(effective_api_key)
            st.success("Gemini client configured")
        else:
            st.warning("No API key found. AI deep analysis will use heuristic mode.")
        selected_models = st.multiselect(
            "Gemini fallback order",
            options=FREE_TIER_MODEL_CANDIDATES,
            default=FREE_TIER_MODEL_CANDIDATES[:3],
            help="The app will try these models in order until one succeeds.",
        )
        if not selected_models:
            selected_models = FREE_TIER_MODEL_CANDIDATES[:3]
        creativity = st.slider(
            "AI creativity",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.1,
            help="Lower values keep suggestions more deterministic.",
        )
        use_vision_fallback = st.toggle(
            "Scanned PDF vision fallback",
            value=True,
            help="Use Gemini vision to read scanned PDFs when direct text extraction is weak.",
            disabled=not bool(effective_api_key),
        )
        st.markdown("---")
        st.markdown(
            "<p class='mini-note'>Free-tier optimized model set includes Gemini 3 Flash Preview and Gemini 2.x Flash variants.</p>",
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="app-hero">
            <h1>SmartHire ATS Pro</h1>
            <p class="mini-note">Production-grade ATS screening with deterministic scoring, multi-resume ranking, and Gemini-powered rewrite suggestions.</p>
            <span class="meta-pill">Resume Screening</span>
            <span class="meta-pill">Keyword Gap Analysis</span>
            <span class="meta-pill">AI Rewrite Suggestions</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns([1.3, 1.0])
    with col_a:
        job_description = st.text_area(
            "Job Description",
            height=230,
            placeholder="Paste full role description (responsibilities + required skills + experience)...",
        )
    with col_b:
        uploaded_files = st.file_uploader(
            "Upload resumes (PDF)",
            type=["pdf"],
            accept_multiple_files=True,
            help=f"Upload up to {MAX_RESUME_COUNT} resumes. Each file max {MAX_FILE_SIZE_MB} MB.",
        )
        run_screening = st.button("Run ATS Screening", type="primary", use_container_width=True)

    if run_screening:
        if not job_description.strip():
            st.error("Please provide a job description before screening.")
            return
        if not uploaded_files:
            st.error("Please upload at least one resume PDF.")
            return
        if len(uploaded_files) > MAX_RESUME_COUNT:
            st.error(f"Please upload at most {MAX_RESUME_COUNT} resumes at once.")
            return

        results: List[Dict[str, Any]] = []
        warnings: List[str] = []
        progress = st.progress(0, text="Processing resumes...")

        for idx, uploaded in enumerate(uploaded_files):
            file_bytes = uploaded.getvalue()
            file_size_mb = len(file_bytes) / (1024 * 1024)
            if file_size_mb > MAX_FILE_SIZE_MB:
                warnings.append(f"{uploaded.name}: skipped (>{MAX_FILE_SIZE_MB} MB).")
                progress.progress((idx + 1) / len(uploaded_files), text=f"Skipping {uploaded.name}")
                continue

            extracted_text = extract_text_from_pdf_bytes(file_bytes)
            vision_model_used = None
            if len(extracted_text) < 200 and use_vision_fallback and effective_api_key:
                vision_text, vision_model_used = try_vision_text_extraction(file_bytes, selected_models)
                if len(vision_text) > len(extracted_text):
                    extracted_text = vision_text
            if len(extracted_text) < 80:
                warnings.append(f"{uploaded.name}: could not extract enough text for reliable scoring.")

            profile = build_resume_profile(uploaded.name, extracted_text)
            score = score_resume_against_job(profile, job_description)
            quick_suggestions = build_quick_suggestions(profile, score)
            results.append(
                {
                    "profile": profile,
                    "score": score,
                    "quick_suggestions": quick_suggestions,
                    "vision_model_used": vision_model_used,
                }
            )
            progress.progress((idx + 1) / len(uploaded_files), text=f"Processed {idx + 1}/{len(uploaded_files)} resumes")

        results.sort(key=lambda item: item["score"].overall_score, reverse=True)
        st.session_state.screening_results = results
        st.session_state.deep_analysis_cache = {}
        st.session_state.last_job_description = job_description
        progress.empty()
        for warning in warnings:
            st.warning(warning)

    results = st.session_state.screening_results
    if not results:
        st.info("Run screening to see ATS rankings and improvement suggestions.")
        return

    st.subheader("Candidate Ranking")
    table_rows = []
    for rank, item in enumerate(results, start=1):
        score: ScoreCard = item["score"]
        profile: ResumeProfile = item["profile"]
        table_rows.append(
            {
                "Rank": rank,
                "Resume": profile.file_name,
                "Overall": score.overall_score,
                "Skills": score.skills_match,
                "Keywords": score.keywords_match,
                "Experience": score.experience_match,
                "Formatting": score.formatting_score,
                "Years Exp": round(profile.years_experience, 1),
            }
        )
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    overall_scores = [item["score"].overall_score for item in results]
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Average ATS Score", round(sum(overall_scores) / len(overall_scores), 1))
    metric_col2.metric("Median ATS Score", round(float(pd.Series(overall_scores).median()), 1))
    metric_col3.metric("Top Candidate Score", max(overall_scores))

    candidate_options = [item["profile"].file_name for item in results]
    selected_resume_name = st.selectbox("Detailed candidate review", candidate_options)
    selected_item = next(item for item in results if item["profile"].file_name == selected_resume_name)
    selected_profile: ResumeProfile = selected_item["profile"]
    selected_score: ScoreCard = selected_item["score"]

    st.markdown("---")
    st.subheader(f"Detailed ATS Review: {selected_profile.file_name}")
    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric("Overall", f"{selected_score.overall_score}/100")
    d2.metric("Skills", f"{selected_score.skills_match}/100")
    d3.metric("Keywords", f"{selected_score.keywords_match}/100")
    d4.metric("Experience", f"{selected_score.experience_match}/100")
    d5.metric("Formatting", f"{selected_score.formatting_score}/100")
    st.write(selected_score.summary)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Missing Skills**")
        st.write(", ".join(selected_score.missing_skills) if selected_score.missing_skills else "No high-priority skill gaps detected.")
    with c2:
        st.markdown("**Missing Keywords**")
        st.write(", ".join(selected_score.missing_keywords) if selected_score.missing_keywords else "No major keyword gaps detected.")

    st.markdown("**Fast Improvement Suggestions**")
    for suggestion in selected_item["quick_suggestions"]:
        st.markdown(f"- {suggestion}")
    if selected_item.get("vision_model_used"):
        st.caption(f"Scanned PDF text extraction used model: {selected_item['vision_model_used']}")

    st.markdown("---")
    st.subheader("AI Deep Analysis & Rewrite")
    candidate_cache_key = (
        f"{selected_profile.candidate_id}:"
        f"{hashlib.sha256(st.session_state.last_job_description.encode('utf-8')).hexdigest()[:12]}"
    )
    trigger_deep_analysis = st.button("Generate AI Deep Analysis", use_container_width=True)

    if trigger_deep_analysis:
        if not effective_api_key:
            st.warning("No API key found. Generating heuristic deep analysis.")
            st.session_state.deep_analysis_cache[candidate_cache_key] = heuristic_deep_analysis(
                selected_profile, selected_score, st.session_state.last_job_description
            )
        else:
            with st.spinner("Generating structured deep analysis..."):
                try:
                    analysis = deep_analysis_with_fallback(
                        selected_profile,
                        selected_score,
                        st.session_state.last_job_description,
                        selected_models,
                        temperature=creativity,
                    )
                    st.session_state.deep_analysis_cache[candidate_cache_key] = analysis
                except Exception as exc:
                    st.error(f"AI deep analysis failed ({exc}). Falling back to deterministic insights.")
                    st.session_state.deep_analysis_cache[candidate_cache_key] = heuristic_deep_analysis(
                        selected_profile, selected_score, st.session_state.last_job_description
                    )

    deep_analysis = st.session_state.deep_analysis_cache.get(candidate_cache_key)
    if deep_analysis:
        st.caption(f"Analysis source: {deep_analysis.get('model_used', 'unknown')}")
        st.markdown("**Executive Summary**")
        st.write(deep_analysis.get("executive_summary", ""))
        x1, x2 = st.columns(2)
        with x1:
            st.markdown("**Strengths**")
            for item in deep_analysis.get("strengths", []):
                st.markdown(f"- {item}")
            st.markdown("**ATS Risks**")
            for item in deep_analysis.get("ats_risks", []):
                st.markdown(f"- {item}")
        with x2:
            st.markdown("**Gaps**")
            for item in deep_analysis.get("gaps", []):
                st.markdown(f"- {item}")
            st.markdown("**Interview Focus**")
            for item in deep_analysis.get("interview_focus", []):
                st.markdown(f"- {item}")

        st.markdown("**Targeted Rewrite Suggestions**")
        for rewrite in deep_analysis.get("targeted_improvements", []):
            st.markdown(f"- **{rewrite.get('section', 'Section')}**")
            st.markdown(f"  Issue: {rewrite.get('issue', '')}")
            st.markdown(f"  Rewrite: {rewrite.get('rewrite', '')}")
            st.markdown(f"  Impact: {rewrite.get('impact', '')}")

        st.markdown("**Optimized Professional Summary**")
        st.code(deep_analysis.get("optimized_professional_summary", ""), language="text")
        st.markdown("**First 30-Day Plan**")
        for item in deep_analysis.get("first_30_day_plan", []):
            st.markdown(f"- {item}")

        report_markdown = build_candidate_report_markdown(
            selected_profile,
            selected_score,
            selected_item["quick_suggestions"],
            deep_analysis,
        )
        safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", selected_profile.file_name)
        st.download_button(
            label="Download Candidate Report (.md)",
            data=report_markdown,
            file_name=f"{safe_name}_ats_report.md",
            mime="text/markdown",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
