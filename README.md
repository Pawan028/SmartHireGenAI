 # 🧠 SmartHire AI  
### *Intelligent Resume Analysis & Career Enhancement Tool*  

<img width="1916" height="873" alt="image" src="https://github.com/user-attachments/assets/275f8fe6-77c7-43f7-92d5-5eebe2ce59a0" />


---

## 🚀 Overview  
**SmartHire AI** is an intelligent web application that analyzes resumes using **Google Gemini** and **prompt engineering** to provide actionable insights for improving job compatibility and **ATS (Applicant Tracking System)** performance.  

This tool allows users to **upload their resumes** (PDF format) and optionally **paste a job description** to receive instant feedback on skill gaps, keyword matching, and content optimization.  

🔗 **Live Demo:** [SmartHire AI Web App](https://smarthiregenai.streamlit.app)

---

## 🎯 Key Features  
✅ Upload and analyze resumes (PDF format, up to 200MB)  
✅ Compare resume content with a specific job description  
✅ Get ATS-friendly feedback and optimization suggestions  
✅ Skill gap detection and recommendations  
✅ Real-time insights powered by **Google Gemini AI**  
✅ Clean and interactive UI built with **Streamlit**

---

## 🧩 Tech Stack  

| Layer | Technologies Used |
|-------|--------------------|
| **Frontend** | Streamlit (Python Framework) |
| **Backend** | Python |
| **AI Model** | Google Gemini (Generative AI) |
| **Core Concepts** | NLP, Prompt Engineering, Resume Parsing |
| **Libraries** | PyPDF2, Pandas, NumPy, Requests, Regex |
| **Deployment** | Streamlit Cloud |

---

## ⚙️ How It Works  
1. User uploads a **resume (PDF)**.  
2. Optionally, pastes a **job description** for targeted analysis.  
3. The app extracts and cleans text data using **NLP**.  
4. **Google Gemini API** processes the content through a crafted prompt.  
5. The system generates **ATS match scores**, highlights missing skills, and provides feedback for improvement.  

---

## 🧠 Example Use Case  
- A job seeker uploads their resume and pastes a job posting for “Data Analyst.”  
- SmartHire AI analyzes the resume, identifies missing keywords (e.g., Power BI, SQL, or Tableau), and suggests how to improve alignment.  
- The user receives actionable recommendations to enhance resume quality and job match probability.  

---

## 🛠️ Installation and Setup  

### Prerequisites  
- Python 3.8 or higher  
- Google Gemini API key  

### Steps  
```bash
# Clone the repository
git clone https://github.com/pawan028/smarthire-ai.git
cd smarthire-ai

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
🔑 Environment Variables

Create a .env file in your project directory and add the following line:

GOOGLE_API_KEY=your_gemini_api_key_here

**📊 Sample Output

ATS Match Score: 85%

Missing Keywords: SQL, Data Visualization, Excel

Strengths: Relevant experience, strong analytical skills

Feedback: Add quantifiable achievements and recent tools**

**🧩 Future Enhancements**

AI-powered resume builder with pre-designed templates

Integration with LinkedIn for automatic profile analysis

Multi-language resume evaluation

Recruiter dashboard for bulk resume screening

**👨‍💻 Author**

Pawan Yadav
📍 Delhi, India
📧 pawanya28@gmail.com


 

This project is open-source and available under the MIT License.
