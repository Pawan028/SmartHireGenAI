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

# Configuration
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
MODEL_NAME = "gemini-1.5-pro"

# Initialize session state
def init_session_state():
    defaults = {
        "chat_history": [],
        "resume_content": None,
        "current_action": None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# API Management
def get_ai_response(prompt, pdf_content=None, temperature=0.7):
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        content = [prompt]
        if pdf_content:
            content.extend(pdf_content)
        
        response = model.generate_content(
            content,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=1500
            )
        )
        return response.text
    except Exception as e:
        return f"❌ Error: {str(e)}"

# PDF Processing
def process_pdf(uploaded_file):
    if not uploaded_file:
        return None
    
    try:
        pdf_bytes = uploaded_file.read()
        images = pdf2image.convert_from_bytes(pdf_bytes)
        pdf_parts = []
        
        for img in images[:2]:  # Process first 2 pages only
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG')
            img_data = img_byte_arr.getvalue()
            pdf_parts.append({
                "mime_type": "image/jpeg",
                "data": base64.b64encode(img_data).decode()
            })
        
        return pdf_parts
    except Exception as e:
        st.error(f"❌ Error processing PDF: {str(e)}")
        return None

# Analysis Functions
def analyze_resume(job_desc, pdf_content):
    prompt = f"""
    Analyze this resume against the job description and provide:
    1. Key strengths (3 points)
    2. Areas for improvement (3 points)
    3. Match score (0-100)
    
    Job Description: {job_desc}
    """
    return get_ai_response(prompt, pdf_content)

def get_match_score(job_desc, pdf_content):
    prompt = f"Rate how well this resume matches the job description. Return only a number 0-100. Job: {job_desc}"
    response = get_ai_response(prompt, pdf_content, temperature=0.3)
    try:
        return int(''.join(filter(str.isdigit, response)) or 65)
    except:
        return 65

def get_skill_gaps(job_desc, pdf_content):
    prompt = f"""
    Compare the resume with this job description and identify 5 key missing skills.
    Return ONLY a JSON array: [{"skill": "name", "importance": 1-10, "difficulty": 1-10}]
    Job: {job_desc}
    """
    response = get_ai_response(prompt, pdf_content, temperature=0.3)
    try:
        # Extract JSON from response
        start = response.find('[')
        end = response.rfind(']') + 1
        if start != -1 and end != 0:
            return json.loads(response[start:end])
    except:
        pass
    
    # Fallback data
    return [
        {"skill": "Python Programming", "importance": 9, "difficulty": 7},
        {"skill": "Data Analysis", "importance": 8, "difficulty": 6},
        {"skill": "Project Management", "importance": 7, "difficulty": 5},
        {"skill": "Cloud Computing", "importance": 8, "difficulty": 8},
        {"skill": "Machine Learning", "importance": 9, "difficulty": 9}
    ]

# UI Styling
def load_css():
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .upload-area {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: rgba(102, 126, 234, 0.1);
        margin: 1rem 0;
    }
    
    .action-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        cursor: pointer;
        width: 100%;
        margin: 0.25rem 0;
    }
    
    .skill-card {
        background: rgba(102, 126, 234, 0.1);
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    
    .progress-container {
        background: rgba(102, 126, 234, 0.1);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Main App
def main():
    st.set_page_config(
        page_title="SmartHire AI",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    init_session_state()
    load_css()
    
    # Header
    st.markdown('<h1 class="main-header">🤖 SmartHire AI</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">Intelligent Resume Analysis & Career Enhancement</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Control Panel")
        
        # Settings
        st.subheader("Settings")
        temperature = st.slider("AI Creativity", 0.0, 1.0, 0.7)
        
        st.markdown("---")
        st.info("💡 Tip: Higher creativity values generate more creative responses, while lower values are more focused and factual.")
    
    # Main Content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # File Upload
        st.subheader("📄 Upload Resume")
        uploaded_file = st.file_uploader("Choose PDF file", type="pdf")
        
        # Job Description
        st.subheader("💼 Job Description")
        job_desc = st.text_area("Paste job description here...", height=150)
        
        # Action Buttons
        if uploaded_file and job_desc:
            st.subheader("🚀 Analysis Options")
            
            col1a, col1b, col1c, col1d = st.columns(4)
            
            with col1a:
                if st.button("📊 Full Analysis", use_container_width=True):
                    st.session_state.current_action = "analysis"
            
            with col1b:
                if st.button("🎯 Match Score", use_container_width=True):
                    st.session_state.current_action = "match"
            
            with col1c:
                if st.button("🔍 Skill Gaps", use_container_width=True):
                    st.session_state.current_action = "skills"
            
            with col1d:
                if st.button("✨ Enhance", use_container_width=True):
                    st.session_state.current_action = "enhance"
    
    with col2:
        # Quick Stats
        if uploaded_file:
            st.subheader("📈 Quick Stats")
            st.markdown('<div class="metric-card">PDF Processed ✅</div>', unsafe_allow_html=True)
            
            if job_desc:
                words = len(job_desc.split())
                st.markdown(f'<div class="metric-card">Job Description<br>{words} words</div>', unsafe_allow_html=True)
    
    # Results Section
    if st.session_state.current_action and uploaded_file and job_desc:
        st.markdown("---")
        
        with st.spinner("🧠 Analyzing with AI..."):
            pdf_content = process_pdf(uploaded_file)
            
            if pdf_content:
                if st.session_state.current_action == "analysis":
                    st.subheader("📊 Complete Analysis")
                    result = analyze_resume(job_desc, pdf_content)
                    st.markdown(f'<div style="background: rgba(102, 126, 234, 0.1); padding: 1rem; border-radius: 10px;">{result}</div>', unsafe_allow_html=True)
                
                elif st.session_state.current_action == "match":
                    st.subheader("🎯 Resume Match Score")
                    score = get_match_score(job_desc, pdf_content)
                    
                    # Create pie chart
                    fig = px.pie(
                        values=[score, 100-score], 
                        names=["Match", "Gap"],
                        hole=0.6,
                        color_discrete_sequence=["#667eea", "#e0e0e0"]
                    )
                    fig.update_layout(
                        showlegend=True,
                        height=400,
                        font=dict(size=16)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown(f'<div class="metric-card">Match Score: {score}%</div>', unsafe_allow_html=True)
                
                elif st.session_state.current_action == "skills":
                    st.subheader("🔍 Skill Gap Analysis")
                    skill_gaps = get_skill_gaps(job_desc, pdf_content)
                    
                    # Create radar chart
                    skills = [item["skill"] for item in skill_gaps]
                    importance = [item["importance"] for item in skill_gaps]
                    difficulty = [item["difficulty"] for item in skill_gaps]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatterpolar(
                        r=importance,
                        theta=skills,
                        fill='toself',
                        name='Importance',
                        line_color='#667eea'
                    ))
                    fig.add_trace(go.Scatterpolar(
                        r=difficulty,
                        theta=skills,
                        fill='toself',
                        name='Difficulty',
                        line_color='#764ba2'
                    ))
                    fig.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
                        showlegend=True,
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show skill details
                    for skill in skill_gaps:
                        st.markdown(f"""
                        <div class="skill-card">
                            <h4>{skill['skill']}</h4>
                            <p>Importance: {skill['importance']}/10 | Difficulty: {skill['difficulty']}/10</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                elif st.session_state.current_action == "enhance":
                    st.subheader("✨ Enhancement Suggestions")
                    prompt = f"Provide 5 specific actionable suggestions to improve this resume for the job: {job_desc}"
                    suggestions = get_ai_response(prompt, pdf_content)
                    st.markdown(f'<div style="background: rgba(102, 126, 234, 0.1); padding: 1rem; border-radius: 10px;">{suggestions}</div>', unsafe_allow_html=True)
        
        # Reset action
        st.session_state.current_action = None
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<div style="text-align: center; color: #666; padding: 2rem;">Powered by Gemini AI | SmartHire v2.0</div>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
