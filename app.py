import base64
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import fitz
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

GEMINI_API_KEY = ""

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
    "responsibility",
    "requirements",
    "required",
    "preferred",
    "build",
    "develop",
    "developing",
    "improve",
    "improved",
    "optimize",
    "optimized",
    "reduce",
    "reduced",
    "increase",
    "increased",
    "engineering",
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

IMPACT_VERBS = {
    "achieved",
    "automated",
    "built",
    "created",
    "delivered",
    "designed",
    "developed",
    "enhanced",
    "improved",
    "increased",
    "launched",
    "led",
    "optimized",
    "reduced",
    "scaled",
    "streamlined",
}

OUTCOME_VERBS = {
    "accelerated",
    "boosted",
    "cut",
    "decreased",
    "grew",
    "improved",
    "increased",
    "reduced",
    "saved",
}

QUANTIFIED_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:%|percent\b|x\b|k\b|m\b|b\b|ms\b|s\b|sec\b|seconds?\b|minutes?\b|hours?\b|days?\b|weeks?\b|months?\b|years?\b|users?\b|customers?\b|clients?\b|projects?\b|pipelines?\b|models?\b|tickets?\b|requests?\b|records?\b)",
    re.IGNORECASE,
)
GENERIC_LANGUAGE_PATTERN = re.compile(
    r"\b(responsible for|worked on|involved in|participated in|helped with|assisted with)\b",
    re.IGNORECASE,
)


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
    impact_match: int
    formatting_score: int
    confidence: int
    matched_skills: List[str]
    missing_skills: List[str]
    missing_keywords: List[str]
    evidence_gaps: List[str]
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
        cleaned = token.strip(".,:;()[]{}")
        if len(cleaned) < 3 or cleaned in STOPWORDS:
            continue
        freq[cleaned] = freq.get(cleaned, 0) + 1
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


def count_keyword_occurrences(text: str, keywords: List[str]) -> Dict[str, int]:
    lowered = text.lower()
    counts: Dict[str, int] = {}
    for keyword in keywords:
        pattern = rf"(?<!\w){re.escape(keyword.lower())}(?!\w)"
        occurrences = len(re.findall(pattern, lowered))
        if occurrences:
            counts[keyword] = occurrences
    return counts


def analyze_resume_evidence(profile: ResumeProfile) -> Dict[str, int]:
    lowered = profile.text.lower()
    tokens = tokenize(lowered)
    impact_hits = sum(1 for token in tokens if token in IMPACT_VERBS)
    outcome_hits = sum(1 for token in tokens if token in OUTCOME_VERBS)
    quantified_hits = len(QUANTIFIED_PATTERN.findall(lowered))
    quantified_outcome_hits = len(
        re.findall(
            r"\b(improved|increased|reduced|saved|boosted|cut|grew|optimized)\b[^.]{0,48}\b\d",
            lowered,
            re.IGNORECASE,
        )
    )
    generic_hits = len(GENERIC_LANGUAGE_PATTERN.findall(lowered))
    return {
        "impact_hits": impact_hits,
        "outcome_hits": outcome_hits,
        "quantified_hits": quantified_hits,
        "quantified_outcome_hits": quantified_outcome_hits,
        "generic_hits": generic_hits,
    }


def compute_impact_score(evidence: Dict[str, int], years_experience: float) -> int:
    quantified_component = min(evidence["quantified_hits"], 10) / 10 * 52
    outcome_component = min(evidence["quantified_outcome_hits"], 6) / 6 * 26
    action_component = min(evidence["impact_hits"], 20) / 20 * 16
    outcome_verb_component = min(evidence["outcome_hits"], 10) / 10 * 6
    penalty = min(evidence["generic_hits"] * 2, 14)
    if years_experience >= 3 and evidence["quantified_hits"] < 2:
        penalty += 8
    return clamp(quantified_component + outcome_component + action_component + outcome_verb_component - penalty, low=20)


def compute_confidence_score(
    profile: ResumeProfile,
    keyword_coverage_ratio: float,
    skill_coverage_ratio: float,
    evidence: Dict[str, int],
) -> int:
    major_sections = sum(1 for section in ["summary", "experience", "skills", "education"] if profile.sections.get(section))
    confidence = 30.0
    if profile.word_count >= 300:
        confidence += 16
    elif profile.word_count >= 180:
        confidence += 10
    elif profile.word_count >= 120:
        confidence += 6
    else:
        confidence -= 10
    confidence += major_sections * 7
    if profile.email and profile.phone:
        confidence += 8
    if profile.linkedin or profile.github:
        confidence += 4
    confidence += min(keyword_coverage_ratio, 1.0) * 14
    confidence += min(skill_coverage_ratio, 1.0) * 10
    confidence += min(evidence["quantified_hits"], 6) * 2
    confidence -= min(evidence["generic_hits"], 6) * 2
    if not profile.sections.get("experience"):
        confidence -= 12
    if len(profile.text) < 550:
        confidence -= 8
    return clamp(confidence, low=25, high=98)


def build_evidence_gaps(
    profile: ResumeProfile,
    missing_skills: List[str],
    missing_keywords: List[str],
    evidence: Dict[str, int],
    required_years: Optional[float],
) -> List[str]:
    gaps: List[str] = []
    if evidence["quantified_hits"] < 2:
        gaps.append("Add 3-5 quantified outcomes in experience bullets (% growth, latency reduction, volume handled, or cost saved).")
    if evidence["quantified_outcome_hits"] < 2:
        gaps.append("Rewrite bullets to action + scope + measurable result format for stronger recruiter signal.")
    if evidence["generic_hits"] >= 4:
        gaps.append("Replace generic language like 'responsible for' with direct ownership and results statements.")
    if not profile.sections.get("experience"):
        gaps.append("Create a dedicated Experience section with role, company, dates, and achievement bullets.")
    if required_years and profile.years_experience < required_years:
        gaps.append(
            f"Role asks for ~{int(required_years)}+ years; emphasize equivalent depth through complex projects and leadership outcomes."
        )
    if missing_skills:
        gaps.append(f"Highlight missing core skills where you have equivalent experience: {', '.join(missing_skills[:4])}.")
    if missing_keywords:
        gaps.append(f"Use exact role language naturally in bullets: {', '.join(missing_keywords[:4])}.")
    if not profile.linkedin and not profile.github:
        gaps.append("Add LinkedIn or GitHub URL to improve recruiter validation confidence.")
    return gaps[:6]


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


def build_score_summary(overall: int, confidence: int, impact_match: int, gap_count: int) -> str:
    if overall >= 85 and confidence >= 78 and impact_match >= 65 and gap_count <= 2:
        return "Strong shortlist candidate with proven ATS and outcome alignment."
    if overall >= 72:
        return "Good fit with interview potential after targeted evidence and keyword tuning."
    if overall >= 58:
        return "Moderate fit. Needs stronger impact evidence and role-specific alignment."
    if confidence < 55:
        return "Low-confidence match from weak resume signal quality. Improve structure and measurable outcomes."
    return "Low match today. Significant resume targeting needed before applying."


def score_resume_against_job(profile: ResumeProfile, job_description: str) -> ScoreCard:
    resume_lower = profile.text.lower()
    job_lower = job_description.lower()
    job_keywords = extract_job_keywords(job_description)
    keyword_occurrences = count_keyword_occurrences(resume_lower, job_keywords)
    matched_keywords = sorted(keyword_occurrences.keys())
    missing_keywords = [kw for kw in job_keywords if kw not in keyword_occurrences]
    keyword_coverage_ratio = len(matched_keywords) / max(1, len(job_keywords))
    keyword_breadth_factor = min(len(job_keywords) / 25, 1.0)
    repetitive_keyword_count = sum(1 for count in keyword_occurrences.values() if count >= 4)
    keyword_repeat_penalty = min(repetitive_keyword_count * 2, 12)
    keywords_match = clamp((keyword_coverage_ratio * 88) + (keyword_breadth_factor * 12) - keyword_repeat_penalty)

    resume_skills = set(detect_skills(profile.text))
    required_skills = {skill for skill in SKILL_TAXONOMY if keyword_present(job_lower, skill)}
    if required_skills:
        missing_skills = sorted(required_skills - resume_skills)
        matched_skills = sorted(required_skills & resume_skills)
        skill_coverage_ratio = len(matched_skills) / max(1, len(required_skills))
        skill_breadth_factor = min(len(required_skills) / 10, 1.0)
        skills_match = clamp((skill_coverage_ratio * 90) + (skill_breadth_factor * 10))
    else:
        missing_skills = []
        matched_skills = sorted(resume_skills)[:20]
        skill_coverage_ratio = min(len(matched_skills) / 12, 1.0)
        skills_match = clamp((keywords_match * 0.85) + (skill_coverage_ratio * 15))

    required_years = extract_required_years(job_description)
    if required_years is None:
        if profile.years_experience >= 8:
            experience_match = 82
        elif profile.years_experience >= 4:
            experience_match = 75
        elif profile.years_experience >= 2:
            experience_match = 68
        elif profile.years_experience >= 1:
            experience_match = 60
        else:
            experience_match = 48
    else:
        delta = profile.years_experience - required_years
        if delta >= 0:
            experience_match = clamp(74 + min(delta * 5, 18))
        else:
            experience_match = clamp(74 + delta * 11, low=28)

    evidence = analyze_resume_evidence(profile)
    impact_match = compute_impact_score(evidence, profile.years_experience)
    confidence = compute_confidence_score(profile, keyword_coverage_ratio, skill_coverage_ratio, evidence)
    fmt_score = formatting_score(profile)
    base_overall = (
        (0.26 * keywords_match)
        + (0.24 * skills_match)
        + (0.18 * experience_match)
        + (0.20 * impact_match)
        + (0.12 * fmt_score)
    )
    reliability_factor = 0.72 + (confidence / 100) * 0.28
    penalties = 0
    if required_skills and not matched_skills:
        penalties += 15
    if required_skills and len(missing_skills) >= max(3, int(len(required_skills) * 0.45)):
        penalties += 6
    if impact_match < 45:
        penalties += 8
    if keywords_match > 90 and impact_match < 45:
        penalties += 8
    if profile.word_count < 160 or profile.word_count > 1800:
        penalties += 6
    if confidence < 55:
        penalties += 6
    overall = clamp((base_overall * reliability_factor) - penalties)
    prioritized_missing = missing_skills + [kw for kw in missing_keywords if kw not in missing_skills]
    evidence_gaps = build_evidence_gaps(profile, missing_skills, prioritized_missing, evidence, required_years)
    return ScoreCard(
        overall_score=overall,
        skills_match=skills_match,
        experience_match=experience_match,
        keywords_match=keywords_match,
        impact_match=impact_match,
        formatting_score=fmt_score,
        confidence=confidence,
        matched_skills=matched_skills[:20],
        missing_skills=missing_skills[:15],
        missing_keywords=prioritized_missing[:15],
        evidence_gaps=evidence_gaps,
        summary=build_score_summary(overall, confidence, impact_match, len(evidence_gaps)),
    )


def build_quick_suggestions(profile: ResumeProfile, score: ScoreCard) -> List[str]:
    suggestions: List[str] = []
    suggestions.extend(score.evidence_gaps[:2])
    if score.keywords_match < 75:
        missing = ", ".join(score.missing_keywords[:6]) if score.missing_keywords else "role-specific keywords"
        suggestions.append(f"Add missing JD keywords naturally in experience/projects: {missing}.")
    if score.skills_match < 75 and score.missing_skills:
        missing_skills = ", ".join(score.missing_skills[:5])
        suggestions.append(f"Create a dedicated technical skills block and show project evidence for: {missing_skills}.")
    if score.impact_match < 60:
        suggestions.append(
            "Rewrite at least 4 experience bullets to include measurable outcomes (%, time saved, latency, revenue, or volume impact)."
        )
    if score.experience_match < 65:
        suggestions.append("Strengthen role chronology and date ranges so ATS can infer experience depth more accurately.")
    if score.formatting_score < 70:
        suggestions.append(
            "Use ATS-safe layout: single-column PDF, clear section headers, and standard fonts without graphics-heavy elements."
        )
    if score.confidence < 65:
        suggestions.append("Resume signal confidence is moderate; add clearer section labels, contact links, and stronger result evidence.")
    if not profile.sections.get("summary"):
        suggestions.append("Add a 3-4 line professional summary tailored to the exact target role.")
    if not profile.sections.get("projects"):
        suggestions.append("Add a projects section with stack, scope, and outcomes to improve recruiter confidence.")
    if not profile.linkedin and not profile.github:
        suggestions.append("Add LinkedIn/GitHub profile links near contact details to improve profile credibility.")
    if not suggestions:
        suggestions.append("Strong baseline. Tailor 2-3 bullets per application to mirror the exact JD outcomes and terminology.")
    deduped: List[str] = []
    seen = set()
    for item in suggestions:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped[:6]


def configure_gemini(api_key: str) -> bool:
    global GEMINI_API_KEY
    if not api_key:
        return False
    GEMINI_API_KEY = api_key.strip()
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
    api_key = GEMINI_API_KEY or os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Gemini API key is not configured.")

    request_parts: List[Dict[str, Any]] = [{"text": prompt}]
    for part in parts or []:
        mime_type = str(part.get("mime_type", "")).strip()
        blob = part.get("data")
        if mime_type and isinstance(blob, (bytes, bytearray)):
            request_parts.append(
                {
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": base64.b64encode(bytes(blob)).decode("ascii"),
                    }
                }
            )

    payload: Dict[str, Any] = {
        "contents": [{"role": "user", "parts": request_parts}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        },
    }
    if expect_json:
        payload["generationConfig"]["responseMimeType"] = "application/json"

    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent",
        params={"key": api_key},
        json=payload,
        timeout=60,
    )
    if response.status_code >= 400:
        details = response.text[:600]
        raise RuntimeError(f"Gemini API error ({response.status_code}): {details}")

    response_payload = response.json()
    candidates = response_payload.get("candidates") or []
    if not candidates:
        feedback = response_payload.get("promptFeedback")
        raise RuntimeError(f"Gemini returned no candidates. promptFeedback={feedback}")
    candidate = candidates[0]
    content = candidate.get("content", {})
    text = "".join(str(part.get("text", "")) for part in content.get("parts", []) if isinstance(part, dict))
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
- Be critical: avoid generic praise unless supported by deterministic score evidence.
- If `confidence` < 70 or `impact_match` < 60, prioritize evidence-building actions in `gaps` and `ats_risks`.

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
    suggestions = build_quick_suggestions(profile, score)
    role_keywords = extract_job_keywords(job_description, top_k=12)
    keyword_focus = ", ".join(score.missing_keywords[:4]) if score.missing_keywords else ", ".join(role_keywords[:4])
    skill_focus = ", ".join(score.missing_skills[:4]) if score.missing_skills else "core stack from the job description"
    targeted_improvements = [
        {
            "section": "Professional Summary",
            "issue": "Summary is broad and can be more role-specific.",
            "rewrite": f"Engineer with {max(1, int(profile.years_experience))}+ years building production systems across {skill_focus}; delivered measurable improvements in reliability, velocity, and user impact.",
            "impact": "Improves recruiter relevance scan in the first 8-10 seconds.",
        },
        {
            "section": "Experience Bullets",
            "issue": "Current bullets may describe tasks more than outcomes.",
            "rewrite": "Use this format: action + stack + scope + metric (e.g., 'Optimized ETL pipeline on Spark, reducing runtime by 37% and improving SLA compliance to 99.9%').",
            "impact": "Raises interview conversion by making achievements verifiable.",
        },
        {
            "section": "Keyword Alignment",
            "issue": "Critical role terms are missing or underrepresented.",
            "rewrite": f"Add these terms where true in experience/project bullets: {keyword_focus}.",
            "impact": "Improves ATS retrieval for targeted role searches.",
        },
        {
            "section": "Projects",
            "issue": "Projects can better demonstrate complexity and ownership.",
            "rewrite": "Add 2 role-aligned projects with baseline/problem, architecture choices, and measured outcomes.",
            "impact": "Adds concrete execution signal for hiring manager review.",
        },
    ]
    strengths: List[str] = []
    if score.matched_skills:
        strengths.append(f"Matches core role skills: {', '.join(score.matched_skills[:6])}.")
    if score.formatting_score >= 80:
        strengths.append("ATS-safe structure quality is strong (sections/contact/readability).")
    if score.impact_match >= 65:
        strengths.append("Resume includes meaningful evidence of delivered outcomes.")
    if not strengths:
        strengths = ["Resume has baseline structure for ATS parsing and downstream optimization."]

    gaps = suggestions if suggestions else score.evidence_gaps
    ats_risks = [
        "Keyword overuse without measurable outcomes can look like ATS optimization without delivery proof.",
        "Missing quantified impact reduces recruiter trust during shortlist decisions.",
    ]
    if score.confidence < 65:
        ats_risks.insert(0, "Signal confidence is moderate; extracted resume evidence may be insufficient for high-certainty ranking.")
    return {
        "model_used": "heuristic-engine",
        "executive_summary": f"{score.summary} Confidence: {score.confidence}/100. Impact evidence: {score.impact_match}/100.",
        "strengths": strengths,
        "gaps": gaps[:8],
        "ats_risks": ats_risks[:8],
        "targeted_improvements": targeted_improvements,
        "optimized_professional_summary": (
            "Target-role aligned engineer with proven ownership of production delivery, measurable performance gains, "
            "and hands-on execution across cross-functional teams."
        ),
        "interview_focus": [
            "Prepare 3 STAR stories with hard metrics and tradeoff decisions.",
            "Map each top JD requirement to one concrete project/result example.",
            "Be ready to map each key JD requirement to specific past experience.",
        ],
        "first_30_day_plan": [
            "Week 1: map team goals and delivery expectations.",
            "Week 2: baseline core system metrics and identify one quick-win bottleneck.",
            "Week 3: deliver one scoped improvement with measurable before/after impact.",
            "Week 4: propose a role-aligned 90-day execution roadmap.",
        ],
    }


def load_custom_css(theme_mode: str) -> None:
    is_dark = theme_mode.lower() == "dark"
    palette = {
        "bg_gradient": "radial-gradient(circle at top left, #0e1726 0%, #0b1220 45%, #1b2035 100%)"
        if is_dark
        else "radial-gradient(circle at top left, #eef5ec 0%, #f7faf6 45%, #eef3f8 100%)",
        "text": "#e4ecff" if is_dark else "#182218",
        "muted": "#9fb2d8" if is_dark else "#5b6a5d",
        "brand": "#2dd4bf" if is_dark else "#0f766e",
        "brand_soft": "#153b38" if is_dark else "#d8f3ef",
        "line": "#2a3855" if is_dark else "#d4ded1",
        "hero_bg": "linear-gradient(130deg, #111a2e 0%, #12243d 100%)"
        if is_dark
        else "linear-gradient(130deg, #ffffff 0%, #eef6f4 100%)",
        "sidebar_bg": "#0f1728" if is_dark else "#fbfdfb",
        "card_bg": "rgba(18, 30, 52, 0.82)" if is_dark else "rgba(255, 255, 255, 0.88)",
    }

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Source+Sans+3:wght@400;500;600;700&display=swap');
        :root {{
            --text: {palette["text"]};
            --muted: {palette["muted"]};
            --brand: {palette["brand"]};
            --brand-soft: {palette["brand_soft"]};
            --line: {palette["line"]};
            --card-bg: {palette["card_bg"]};
        }}
        .stApp {{
            background: {palette["bg_gradient"]};
            color: var(--text);
            font-family: 'Source Sans 3', sans-serif;
        }}
        h1, h2, h3 {{
            font-family: 'Space Grotesk', sans-serif;
            color: var(--text) !important;
            letter-spacing: 0.01em;
        }}
        p, div, span, label {{
            color: var(--text);
        }}
        .app-hero {{
            border: 1px solid var(--line);
            background: {palette["hero_bg"]};
            border-radius: 18px;
            padding: 1.2rem 1.4rem;
            margin-bottom: 1rem;
            box-shadow: 0 8px 34px rgba(0, 0, 0, 0.16);
        }}
        .panel {{
            border: 1px solid var(--line);
            background: var(--card-bg);
            border-radius: 14px;
            padding: 0.9rem 1rem;
            margin-bottom: 0.7rem;
        }}
        .meta-pill {{
            display: inline-block;
            padding: 0.2rem 0.55rem;
            border-radius: 999px;
            background: var(--brand-soft);
            color: var(--brand);
            border: 1px solid var(--line);
            font-size: 0.78rem;
            margin-right: 0.4rem;
            margin-bottom: 0.35rem;
        }}
        [data-testid="stMetricValue"] {{
            color: var(--brand);
        }}
        [data-testid="stSidebar"] {{
            border-right: 1px solid var(--line);
            background: {palette["sidebar_bg"]};
        }}
        [data-testid="stDataFrame"] {{
            border: 1px solid var(--line);
            border-radius: 12px;
            overflow: hidden;
        }}
        .mini-note {{
            color: var(--muted);
            font-size: 0.88rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_session_state() -> None:
    defaults = {
        "screening_results": [],
        "deep_analysis_cache": {},
        "last_job_description": "",
        "theme_mode": "Dark",
        "auto_deep_analysis": True,
    }
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
        f"- Impact Evidence: {score.impact_match}/100",
        f"- Formatting: {score.formatting_score}/100",
        f"- Signal Confidence: {score.confidence}/100",
        "",
        "## Missing Keywords",
        f"- {', '.join(score.missing_keywords[:15]) if score.missing_keywords else 'None'}",
        "",
        "## Evidence Gaps",
        f"- {'; '.join(score.evidence_gaps) if score.evidence_gaps else 'No major evidence gaps detected.'}",
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
    init_session_state()
    load_custom_css(st.session_state.theme_mode)
    env_api_key = os.getenv("GOOGLE_API_KEY", "").strip()

    with st.sidebar:
        st.header("Runtime Settings")
        theme_mode = st.radio(
            "Theme",
            options=["Dark", "Light"],
            index=0 if st.session_state.theme_mode == "Dark" else 1,
            horizontal=True,
        )
        if theme_mode != st.session_state.theme_mode:
            st.session_state.theme_mode = theme_mode
            st.rerun()

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
        auto_deep_analysis = st.toggle(
            "Auto-run deep analysis",
            value=bool(st.session_state.auto_deep_analysis),
            help="Automatically generate AI/heuristic deep analysis for selected candidate once scoring is complete.",
        )
        st.session_state.auto_deep_analysis = auto_deep_analysis
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

    total_candidates = len(results)
    strong_fit = len([item for item in results if item["score"].overall_score >= 80])
    moderate_fit = len([item for item in results if 60 <= item["score"].overall_score < 80])
    low_fit = len([item for item in results if item["score"].overall_score < 60])

    st.markdown(
        f"""
        <div class="panel">
            <strong>Screening Snapshot</strong><br/>
            <span class="mini-note">
                Total: {total_candidates} &nbsp;|&nbsp; Strong Fit: {strong_fit} &nbsp;|&nbsp; Moderate Fit: {moderate_fit} &nbsp;|&nbsp; Low Fit: {low_fit}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

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
                "Impact": score.impact_match,
                "Formatting": score.formatting_score,
                "Confidence": score.confidence,
                "Evidence Gaps": len(score.evidence_gaps),
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
    d1, d2, d3, d4, d5, d6 = st.columns(6)
    d1.metric("Overall", f"{selected_score.overall_score}/100")
    d2.metric("Skills", f"{selected_score.skills_match}/100")
    d3.metric("Keywords", f"{selected_score.keywords_match}/100")
    d4.metric("Experience", f"{selected_score.experience_match}/100")
    d5.metric("Impact", f"{selected_score.impact_match}/100")
    d6.metric("Confidence", f"{selected_score.confidence}/100")
    st.caption(f"Formatting score: {selected_score.formatting_score}/100")
    st.write(selected_score.summary)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Missing Skills**")
        st.write(", ".join(selected_score.missing_skills) if selected_score.missing_skills else "No high-priority skill gaps detected.")
    with c2:
        st.markdown("**Missing Keywords**")
        st.write(", ".join(selected_score.missing_keywords) if selected_score.missing_keywords else "No major keyword gaps detected.")

    st.markdown("**Evidence Gaps**")
    if selected_score.evidence_gaps:
        for gap in selected_score.evidence_gaps:
            st.markdown(f"- {gap}")
    else:
        st.write("No major evidence gaps detected.")

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
    button_label = (
        "Regenerate AI Deep Analysis"
        if candidate_cache_key in st.session_state.deep_analysis_cache
        else "Generate AI Deep Analysis"
    )
    trigger_deep_analysis = st.button(button_label, use_container_width=True)
    should_generate_deep = trigger_deep_analysis or (
        bool(st.session_state.auto_deep_analysis) and candidate_cache_key not in st.session_state.deep_analysis_cache
    )

    if should_generate_deep:
        if not effective_api_key:
            if trigger_deep_analysis:
                st.warning("No API key found. Generating heuristic deep analysis.")
            st.session_state.deep_analysis_cache[candidate_cache_key] = heuristic_deep_analysis(
                selected_profile, selected_score, st.session_state.last_job_description
            )
        else:
            spinner_msg = "Generating structured deep analysis..." if trigger_deep_analysis else "Auto-generating deep analysis..."
            with st.spinner(spinner_msg):
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
    elif not st.session_state.auto_deep_analysis:
        st.info("Click 'Generate AI Deep Analysis' to create detailed rewrite guidance for this candidate.")


if __name__ == "__main__":
    main()
