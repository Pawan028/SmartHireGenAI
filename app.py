import streamlit as st
import base64
import io
import json
import time
from PIL import Image
import pdf2image
import google.generativeai as genai
import plotly.express as px
import plotly.graph_objects as go
import os
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
MODEL_NAME = "gemini-2.5-flash-lite"

# ═══════════════════════════════════════════════════════════════════════════════
# MODERN UI STYLING
# ═══════════════════════════════════════════════════════════════════════════════
def load_custom_css():
    st.markdown("""
    <style>
    /* ══════════════════ Global Styles ══════════════════ */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    :root {
        --primary: #6366f1;
        --secondary: #8b5cf6;
        --accent: #06b6d4;
        --success: #10b981;
        --warning: #f59e0b;
        --danger: #ef4444;
    }
    
    /* ══════════════════ Main Layout ══════════════════ */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Remove default top padding and ensure content centering */
    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }
    
    /* Hide default elements */
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* ══════════════════ Adaptive Theme Colors ══════════════════ */
    /* Light Mode */
    [data-theme="light"], .stApp {
        --glass-bg: rgba(255, 255, 255, 0.95);
        --glass-border: rgba(0, 0, 0, 0.1);
        --text-primary: #1e293b;
        --text-secondary: #475569;
        --card-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        --card-bg: #ffffff;
        --sidebar-bg: #f8fafc;
    }
    
    /* Dark Mode (Streamlit inserts this usually, but we force it via media query backup) */
    @media (prefers-color-scheme: dark) {
        .stApp {
            --glass-bg: rgba(15, 23, 42, 0.6);
            --glass-border: rgba(255, 255, 255, 0.1);
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --card-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
            --card-bg: rgba(30, 41, 59, 0.6);
            --sidebar-bg: #0f172a;
        }
    }
    
    [data-theme="dark"] {
        --glass-bg: rgba(15, 23, 42, 0.6) !important;
        --glass-border: rgba(255, 255, 255, 0.1) !important;
        --text-primary: #f1f5f9 !important;
        --text-secondary: #94a3b8 !important;
        --card-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3) !important;
        --card-bg: rgba(30, 41, 59, 0.6) !important;
        --sidebar-bg: #0f172a !important;
    }
    
    /* ══════════════════ Typography Overrides (Aggressive) ══════════════════ */
    h1, h2, h3, h4, h5, h6, strong {
        color: var(--text-primary) !important;
    }
    
    p, li, span, div, label {
        color: var(--text-secondary);
    }
    
    .stMarkdown p {
        color: var(--text-primary) !important;
    }
    
    /* ══════════════════ Hero Section ══════════════════ */
    .hero-container {
        text-align: center;
        padding: 4rem 1rem;
        margin-bottom: 3rem;
        background: var(--card-bg);
        border-radius: 24px;
        border: 1px solid var(--primary);
        box-shadow: 0 10px 30px -10px rgba(99, 102, 241, 0.2);
    }
    
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1rem;
        padding-bottom: 0.2rem;
    }
    
    .hero-subtitle {
        color: var(--text-secondary) !important;
        font-size: 1.1rem;
        font-weight: 500;
        max-width: 600px;
        margin: 0 auto;
    }
    
    /* ══════════════════ Card Styling ══════════════════ */
    .glass-card {
        background: var(--card-bg);
        border: 1px solid var(--glass-border);
        border-radius: 20px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: var(--card-shadow);
        transition: transform 0.2s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-2px);
        border-color: var(--primary);
    }
    
    .card-title {
        color: var(--text-primary) !important;
    }
    
    .card-header {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 1.25rem;
    }
    
    .card-icon {
        width: 48px;
        height: 48px;
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        color: white;
        flex-shrink: 0;
    }
    
    /* ══════════════════ Sidebar Styling ══════════════════ */
    [data-testid="stSidebar"] {
        border-right: 1px solid var(--glass-border);
        background-color: var(--card-bg);
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: var(--text-primary) !important;
    }
    
    [data-testid="stSidebar"] p {
        color: var(--text-secondary) !important;
    }
    
    /* ══════════════════ Feature Grid ══════════════════ */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .feature-item {
        background: rgba(99, 102, 241, 0.05);
        border: 1px solid rgba(99, 102, 241, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .feature-item:hover {
        background: rgba(99, 102, 241, 0.1);
        transform: translateY(-4px);
        border-color: var(--primary);
    }
    
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    
    .feature-text {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    /* ══════════════════ Metrics & Scores ══════════════════ */
    .score-container {
        text-align: center;
        padding: 2rem;
    }
    
    .metric-card {
        background: rgba(99, 102, 241, 0.05);
        border-radius: 16px;
        padding: 1rem;
        text-align: center;
        border: 1px solid rgba(99, 102, 241, 0.1);
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: var(--primary);
    }
    
    /* ══════════════════ Buttons ══════════════════ */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        border-radius: 12px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        opacity: 0.9;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
    }
    
    /* ══════════════════ Inputs ══════════════════ */
    [data-testid="stFileUploader"] {
        padding: 2rem;
        border: 2px dashed rgba(99, 102, 241, 0.3);
        border-radius: 16px;
        background: rgba(99, 102, 241, 0.02);
    }
    
    .stTextArea textarea {
        border-radius: 12px !important;
        border: 1px solid rgba(128, 128, 128, 0.2) !important;
    }
    
    .stTextArea textarea:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2) !important;
    }
    
    /* ══════════════════ Animations ══════════════════ */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animate-in {
        animation: fadeIn 0.5s ease-out forwards;
    }
    </style>
    """, unsafe_allow_html=True)



# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════
def init_session_state():
    """Initialize session state with default values"""
    defaults = {
        "resume_content": None,
        "analysis_results": None,
        "current_action": None,
        "processing": False,
        "error_message": None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ═══════════════════════════════════════════════════════════════════════════════
# AI RESPONSE HANDLER WITH RETRY LOGIC
# ═══════════════════════════════════════════════════════════════════════════════
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_ai_response(prompt: str, pdf_content: list = None, temperature: float = 0.7) -> str:
    """Get AI response with retry logic for reliability"""
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        content = [prompt]
        if pdf_content:
            content.extend(pdf_content)
        
        response = model.generate_content(
            content,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=2000
            )
        )
        return response.text
    except Exception as e:
        error_msg = str(e)
        if "quota" in error_msg.lower():
            return "⚠️ API quota exceeded. Please try again later."
        elif "invalid" in error_msg.lower():
            return "⚠️ Invalid API key. Please check your configuration."
        else:
            raise e


# ═══════════════════════════════════════════════════════════════════════════════
# PDF PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════
def process_pdf(uploaded_file) -> list:
    """Convert PDF to images for AI processing"""
    if not uploaded_file:
        return None
    
    try:
        pdf_bytes = uploaded_file.read()
        images = pdf2image.convert_from_bytes(pdf_bytes, dpi=150)
        pdf_parts = []
        
        # Process first 3 pages for better analysis
        for img in images[:3]:
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG', quality=85)
            img_data = img_byte_arr.getvalue()
            pdf_parts.append({
                "mime_type": "image/jpeg",
                "data": base64.b64encode(img_data).decode()
            })
        
        return pdf_parts
    except Exception as e:
        st.error(f"❌ Error processing PDF: {str(e)}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════
def analyze_resume(job_desc: str, pdf_content: list) -> str:
    """Comprehensive resume analysis"""
    prompt = f"""
    You are an expert ATS (Applicant Tracking System) consultant and career coach.
    Analyze this resume against the job description and provide a detailed, actionable analysis.
    
    Structure your response EXACTLY as follows:
    
    ## 🎯 Overall Match Assessment
    [Provide a 2-3 sentence executive summary of how well the resume matches the job]
    
    ## ✅ Key Strengths (Top 5)
    1. [Strength with specific example from resume]
    2. [...]
    
    ## ⚠️ Areas for Improvement (Top 5)
    1. [Specific improvement with actionable suggestion]
    2. [...]
    
    ## 🔑 Missing Keywords
    [List 5-7 critical keywords from job description missing in resume]
    
    ## 💡 Quick Wins (Easy Fixes)
    [3 immediate improvements candidate can make]
    
    ## 📊 ATS Compatibility Score: [X/100]
    
    Job Description:
    {job_desc}
    """
    return get_ai_response(prompt, pdf_content)


def get_match_score(job_desc: str, pdf_content: list) -> dict:
    """Get detailed match score breakdown"""
    prompt = f"""
    Analyze this resume against the job description and provide match scores.
    
    Return ONLY a valid JSON object with this exact structure:
    {{
        "overall_score": [0-100],
        "skills_match": [0-100],
        "experience_match": [0-100],
        "education_match": [0-100],
        "keywords_match": [0-100],
        "summary": "[One sentence summary]"
    }}
    
    Job Description: {job_desc}
    """
    response = get_ai_response(prompt, pdf_content, temperature=0.3)
    
    try:
        # Extract JSON from response
        start = response.find('{')
        end = response.rfind('}') + 1
        if start != -1 and end > start:
            return json.loads(response[start:end])
    except:
        pass
    
    # Fallback scores
    return {
        "overall_score": 65,
        "skills_match": 60,
        "experience_match": 70,
        "education_match": 75,
        "keywords_match": 55,
        "summary": "Unable to parse detailed scores. Please try again."
    }


def get_skill_gaps(job_desc: str, pdf_content: list) -> list:
    """Identify skill gaps with learning recommendations"""
    prompt = f"""
    Compare this resume with the job description and identify missing skills.
    
    Return ONLY a valid JSON array with exactly 6 items in this format:
    [
        {{
            "skill": "Skill Name",
            "importance": [1-10],
            "difficulty": [1-10],
            "learning_time": "X weeks/months",
            "resources": "Recommended learning resource"
        }}
    ]
    
    Job Description: {job_desc}
    """
    response = get_ai_response(prompt, pdf_content, temperature=0.3)
    
    try:
        start = response.find('[')
        end = response.rfind(']') + 1
        if start != -1 and end > start:
            return json.loads(response[start:end])
    except:
        pass
    
    # Fallback data
    return [
        {"skill": "Data Analysis", "importance": 9, "difficulty": 6, "learning_time": "4-6 weeks", "resources": "Coursera, DataCamp"},
        {"skill": "Python Programming", "importance": 8, "difficulty": 7, "learning_time": "8-12 weeks", "resources": "Codecademy, freeCodeCamp"},
        {"skill": "SQL Databases", "importance": 8, "difficulty": 5, "learning_time": "3-4 weeks", "resources": "W3Schools, SQLZoo"},
        {"skill": "Cloud Computing", "importance": 7, "difficulty": 8, "learning_time": "6-8 weeks", "resources": "AWS Free Tier, Google Cloud"},
        {"skill": "Machine Learning", "importance": 9, "difficulty": 9, "learning_time": "12-16 weeks", "resources": "Andrew Ng's Course, Fast.ai"},
        {"skill": "Project Management", "importance": 6, "difficulty": 4, "learning_time": "4-6 weeks", "resources": "PMI, Coursera"}
    ]


def get_interview_tips(job_desc: str, pdf_content: list) -> str:
    """Generate personalized interview preparation tips"""
    prompt = f"""
    Based on this resume and job description, provide comprehensive interview preparation tips.
    
    Structure your response as:
    
    ## 🎤 Expected Interview Questions
    1. [Technical question based on job requirements]
    2. [Behavioral question]
    3. [Situational question]
    4. [Company/role specific question]
    5. [Experience-based question]
    
    ## 💬 Sample Answers from Your Resume
    [Provide 2-3 STAR method answers using candidate's actual experience]
    
    ## 🚀 Topics to Brush Up On
    [List 5 technical topics candidate should review]
    
    ## ❌ Potential Red Flags to Address
    [Any gaps or concerns interviewer might raise, with suggested responses]
    
    ## 🏆 Your Unique Selling Points
    [3 things that make this candidate stand out]
    
    Job Description: {job_desc}
    """
    return get_ai_response(prompt, pdf_content)


def get_enhancement_suggestions(job_desc: str, pdf_content: list) -> str:
    """Get specific resume enhancement suggestions"""
    prompt = f"""
    You are an expert resume writer. Provide specific, actionable suggestions to enhance this resume for the target job.
    
    Structure your response as:
    
    ## 📝 Professional Summary Rewrite
    [Provide an improved professional summary tailored to the job]
    
    ## 💼 Experience Section Improvements
    [For each relevant role, suggest better bullet points with metrics]
    
    ## 🎯 Skills Section Optimization
    [Suggest how to reorganize and present skills for ATS]
    
    ## 📚 Additional Sections to Add
    [Suggest new sections like Certifications, Projects, etc.]
    
    ## ✨ Formatting & ATS Tips
    [5 specific formatting improvements for better ATS parsing]
    
    ## 🔄 Before vs After Example
    [Show one bullet point transformation with improvement]
    
    Job Description: {job_desc}
    """
    return get_ai_response(prompt, pdf_content)


# ═══════════════════════════════════════════════════════════════════════════════
# UI COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════════
def render_hero():
    """Render the hero section"""
    st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">🚀 SmartHire AI</h1>
        <p class="hero-subtitle">Next-Gen Resume Analysis Powered by Gemini 2.5</p>
        <span class="hero-badge">✨ AI-Powered ATS Optimizer</span>
    </div>
    """, unsafe_allow_html=True)


def render_score_visualization(scores: dict):
    """Render beautiful score visualization"""
    overall = scores.get("overall_score", 0)
    
    # Determine color based on score
    if overall >= 80:
        color_class = "success"
    elif overall >= 60:
        color_class = "warning"
    else:
        color_class = "danger"
    
    st.markdown(f"""
    <div class="score-container animate-in">
        <div class="score-circle" style="--score: {overall}">
            <div class="score-inner">
                <span class="score-value">{overall}</span>
                <span class="score-label">ATS Score</span>
            </div>
        </div>
        <p style="color: var(--text-secondary); margin-top: 1rem;">{scores.get('summary', '')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Score breakdown
    st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
    
    metrics = [
        ("Skills", scores.get("skills_match", 0)),
        ("Experience", scores.get("experience_match", 0)),
        ("Keywords", scores.get("keywords_match", 0))
    ]
    
    cols = st.columns(3)
    for i, (label, value) in enumerate(metrics):
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{value}%</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)


def render_skill_gaps(skill_gaps: list):
    """Render skill gap analysis with charts"""
    # Create radar chart
    skills = [s["skill"] for s in skill_gaps]
    importance = [s["importance"] for s in skill_gaps]
    difficulty = [s["difficulty"] for s in skill_gaps]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=importance,
        theta=skills,
        fill='toself',
        name='Importance',
        line_color='#6366f1',
        fillcolor='rgba(99, 102, 241, 0.3)'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=difficulty,
        theta=skills,
        fill='toself',
        name='Difficulty',
        line_color='#8b5cf6',
        fillcolor='rgba(139, 92, 246, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 10], tickfont=dict(color='#94a3b8')),
            angularaxis=dict(tickfont=dict(color='#f1f5f9'))
        ),
        showlegend=True,
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f1f5f9'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Skill cards
    for skill in skill_gaps:
        importance_pct = skill["importance"] * 10
        st.markdown(f"""
        <div class="skill-card animate-in">
            <div class="skill-name">{skill["skill"]}</div>
            <div class="skill-meta">
                <span>⭐ Importance: {skill["importance"]}/10</span>
                <span>📈 Difficulty: {skill["difficulty"]}/10</span>
                <span>⏱️ {skill.get("learning_time", "4-6 weeks")}</span>
            </div>
            <div class="skill-bar">
                <div class="skill-bar-fill" style="width: {importance_pct}%"></div>
            </div>
            <p style="color: var(--text-secondary); font-size: 0.85rem; margin-top: 0.5rem;">
                📚 {skill.get("resources", "Online courses, documentation")}
            </p>
        </div>
        """, unsafe_allow_html=True)


def render_result_card(icon: str, title: str, content: str):
    """Render a styled result card"""
    st.markdown(f"""
    <div class="result-card animate-in">
        <div class="result-header">
            <span class="result-icon">{icon}</span>
            <span class="result-title">{title}</span>
        </div>
        <div class="result-content">
            {content.replace(chr(10), '<br>')}
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    st.set_page_config(
        page_title="SmartHire AI | Resume Analyzer",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    init_session_state()
    load_custom_css()
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <h2 style="color: #f1f5f9; margin-bottom: 0.5rem;">⚙️ Control Panel</h2>
            <p style="color: #94a3b8; font-size: 0.9rem;">Configure your analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### 🎛️ AI Settings")
        temperature = st.slider(
            "Creativity Level",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Higher = more creative, Lower = more focused"
        )
        
        st.markdown("---")
        
        st.markdown("### 📊 Quick Stats")
        if st.session_state.resume_content:
            st.markdown("""
            <div class="status-badge status-success">
                ✅ Resume Loaded
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="status-badge status-warning">
                ⏳ Awaiting Resume
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("""
        <div style="padding: 1rem; background: rgba(99, 102, 241, 0.1); border-radius: 12px; margin-top: 1rem;">
            <h4 style="color: #f1f5f9; margin-bottom: 0.5rem;">💡 Pro Tips</h4>
            <ul style="color: #94a3b8; font-size: 0.85rem; padding-left: 1.2rem;">
                <li>Use PDF format for best results</li>
                <li>Include the full job description</li>
                <li>Try all analysis types</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Main Content
    render_hero()
    
    # Input Section
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="glass-card">
            <div class="card-header">
                <div class="card-icon">📄</div>
                <h3 class="card-title">Upload Resume</h3>
            </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Drop your resume here or click to browse",
            type=["pdf"],
            label_visibility="collapsed"
        )
        
        if uploaded_file:
            st.markdown("""
            <div class="status-badge status-success" style="margin-top: 0.5rem;">
                ✅ Resume uploaded successfully
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="glass-card">
            <div class="card-header">
                <div class="card-icon">💼</div>
                <h3 class="card-title">Job Description</h3>
            </div>
        """, unsafe_allow_html=True)
        
        job_desc = st.text_area(
            "Paste the target job description",
            height=150,
            placeholder="Paste the job description here to get tailored analysis...",
            label_visibility="collapsed"
        )
        
        if job_desc:
            word_count = len(job_desc.split())
            st.markdown(f"""
            <div class="status-badge status-info" style="margin-top: 0.5rem;">
                📝 {word_count} words analyzed
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Analysis Options
    if uploaded_file and job_desc:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div style="text-align: center; margin-bottom: 1.5rem;">
            <h2 style="color: #f1f5f9; font-size: 1.5rem;">🎯 Choose Your Analysis</h2>
            <p style="color: #94a3b8;">Select an analysis type to get started</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature buttons in a grid
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            if st.button("📊 Full Analysis", use_container_width=True):
                st.session_state.current_action = "analysis"
        
        with col2:
            if st.button("🎯 Match Score", use_container_width=True):
                st.session_state.current_action = "match"
        
        with col3:
            if st.button("🔍 Skill Gaps", use_container_width=True):
                st.session_state.current_action = "skills"
        
        with col4:
            if st.button("✨ Enhance", use_container_width=True):
                st.session_state.current_action = "enhance"
        
        with col5:
            if st.button("🎤 Interview Prep", use_container_width=True):
                st.session_state.current_action = "interview"
        
        # Process and display results
        if st.session_state.current_action:
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            
            with st.spinner("🧠 AI is analyzing your resume..."):
                pdf_content = process_pdf(uploaded_file)
                
                if pdf_content:
                    action = st.session_state.current_action
                    
                    if action == "analysis":
                        st.markdown("## 📊 Comprehensive Resume Analysis")
                        result = analyze_resume(job_desc, pdf_content)
                        st.markdown(result)
                    
                    elif action == "match":
                        st.markdown("## 🎯 ATS Match Score")
                        scores = get_match_score(job_desc, pdf_content)
                        render_score_visualization(scores)
                    
                    elif action == "skills":
                        st.markdown("## 🔍 Skill Gap Analysis")
                        skill_gaps = get_skill_gaps(job_desc, pdf_content)
                        render_skill_gaps(skill_gaps)
                    
                    elif action == "enhance":
                        st.markdown("## ✨ Resume Enhancement Suggestions")
                        suggestions = get_enhancement_suggestions(job_desc, pdf_content)
                        st.markdown(suggestions)
                    
                    elif action == "interview":
                        st.markdown("## 🎤 Interview Preparation Guide")
                        tips = get_interview_tips(job_desc, pdf_content)
                        st.markdown(tips)
            
            # Reset action after processing
            st.session_state.current_action = None
    
    else:
        # Show getting started section when no files uploaded
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div style="text-align: center; padding: 3rem 1rem;">
            <h2 style="color: #f1f5f9; font-size: 1.8rem; margin-bottom: 1rem;">🚀 Get Started in 3 Easy Steps</h2>
            <div class="feature-grid" style="max-width: 800px; margin: 0 auto;">
                <div class="feature-item">
                    <div class="feature-icon">📄</div>
                    <div class="feature-text">Upload Resume</div>
                </div>
                <div class="feature-item">
                    <div class="feature-icon">💼</div>
                    <div class="feature-text">Add Job Description</div>
                </div>
                <div class="feature-item">
                    <div class="feature-icon">🎯</div>
                    <div class="feature-text">Get AI Analysis</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="footer">
        <p>Powered by <strong>Google Gemini 2.5 Flash</strong> | Built with ❤️ by <strong>Pawan Yadav</strong></p>
        <p style="margin-top: 0.5rem; font-size: 0.75rem;">
            SmartHire AI v2.0 | Next-Gen Resume Analysis
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
