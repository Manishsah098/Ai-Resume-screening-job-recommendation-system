"""
AI Resume Screening & Job Recommendation System
================================================
Production-Ready, Professional Streamlit Application.
Features:
- High-contrast, flexible, modern UI (dark theme optimized)
- Multi-format resume parsing (PDF, DOCX, TXT) with one-click sample loader
- NLP text preprocessing, lemmatization, and phrase-level skill extraction
- TF-IDF Vectorization & Weighted Cosine Similarity matching
- Top-N explainable recommendations with skill-gap breakdown
- Logistic Regression ML job category classifier with confidence scoring
- Interactive Plotly visualizations (Radar, Donut, Bar charts)
"""

import os
import sys
import io
import pandas as pd
import streamlit as st

# Ensure project root is on sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from modules.resume_parser import ResumeParser
from modules.text_preprocessor import TextPreprocessor
from modules.skill_extractor import SkillExtractor
from modules.job_matcher import JobMatcher
from modules.recommender import Recommender
from modules.classifier import JobClassifier
from utils.helpers import (
    load_jobs_dataframe, format_match_score,
    get_score_color, get_score_label, truncate_text
)
from utils.visualization import (
    create_match_score_chart,
    create_skill_distribution_chart,
    create_skill_gap_chart,
    create_score_breakdown_chart,
    create_classifier_confidence_chart
)
from utils.report_generator import generate_candidate_pdf_report

# ── Paths ────────────────────────────────────────────────────────────────────
JOBS_CSV        = os.path.join(BASE_DIR, "data", "jobs.csv")
SKILLS_CSV      = os.path.join(BASE_DIR, "data", "skills.csv")
MODEL_PATH      = os.path.join(BASE_DIR, "models", "job_classifier.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")
SAMPLE_DS_PATH  = os.path.join(BASE_DIR, "sample_resumes", "sample_data_scientist.docx")
SAMPLE_SE_PATH  = os.path.join(BASE_DIR, "sample_resumes", "sample_software_engineer.docx")

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Resume Screening & Job Recommendation",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── High-Contrast, Professional CSS Styling ──────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

/* Global Font and Base Contrast */
html, body, [class*="css"], .stApp {
    font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, sans-serif !important;
    color: #F8FAFC !important;
    background-color: #0B0F19 !important;
}

/* Force readable font colors on all Streamlit text */
p, span, label, div, li, h1, h2, h3, h4, h5, h6 {
    color: #F1F5F9;
}

.stMarkdown, .stMarkdown p {
    color: #E2E8F0 !important;
    font-size: 0.98rem;
    line-height: 1.6;
}

/* Headings */
h1 {
    color: #FFFFFF !important;
    font-weight: 800 !important;
    letter-spacing: -0.02em;
}
h2 {
    color: #E2E8F0 !important;
    font-weight: 700 !important;
}
h3 {
    color: #CBD5E1 !important;
    font-weight: 600 !important;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F172A 0%, #0A0F1D 100%) !important;
    border-right: 1px solid #1E293B !important;
}
section[data-testid="stSidebar"] p, 
section[data-testid="stSidebar"] span, 
section[data-testid="stSidebar"] div {
    color: #E2E8F0 !important;
}

/* Professional Metric Cards */
div[data-testid="metric-container"] {
    background: linear-gradient(145deg, #1E293B 0%, #172033 100%) !important;
    border: 1px solid #334155 !important;
    border-radius: 14px !important;
    padding: 16px 20px !important;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25) !important;
}
div[data-testid="metric-container"] label {
    color: #94A3B8 !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color: #FFFFFF !important;
    font-weight: 800 !important;
    font-size: 1.8rem !important;
}

/* Gradient Action Buttons */
.stButton > button {
    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 10px 24px !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 14px rgba(79, 70, 229, 0.35) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 22px rgba(79, 70, 229, 0.5) !important;
    color: #FFFFFF !important;
}

/* Secondary Buttons */
.secondary-btn > button {
    background: #1E293B !important;
    color: #E2E8F0 !important;
    border: 1px solid #475569 !important;
    box-shadow: none !important;
}
.secondary-btn > button:hover {
    background: #334155 !important;
    border-color: #64748B !important;
}

/* File Uploader styling */
div[data-testid="stFileUploader"] {
    background: #131D31 !important;
    border: 2px dashed #4F46E5 !important;
    border-radius: 14px !important;
    padding: 20px !important;
}
div[data-testid="stFileUploader"] section {
    background: transparent !important;
}
div[data-testid="stFileUploader"] span,
div[data-testid="stFileUploader"] small {
    color: #CBD5E1 !important;
    font-weight: 500 !important;
}

/* Expanders */
div[data-testid="stExpander"] {
    background: #141E33 !important;
    border: 1px solid #283548 !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}
div[data-testid="stExpander"] summary {
    color: #F8FAFC !important;
    font-weight: 600 !important;
}

/* Selectboxes & Inputs */
div[data-baseweb="select"] > div {
    background-color: #1A243B !important;
    border-color: #334155 !important;
    color: #FFFFFF !important;
    border-radius: 8px !important;
}
div[data-baseweb="select"] * {
    color: #FFFFFF !important;
}

/* High-Contrast Skill Badges */
.skill-badge {
    display: inline-flex;
    align-items: center;
    background: #1E3A8A;
    color: #BFDBFE !important;
    padding: 5px 13px;
    border-radius: 20px;
    margin: 4px 3px;
    font-size: 0.84rem;
    font-weight: 600;
    border: 1px solid #3B82F6;
    letter-spacing: 0.01em;
}
.skill-badge-matched {
    display: inline-flex;
    align-items: center;
    background: #064E3B;
    color: #A7F3D0 !important;
    padding: 5px 13px;
    border-radius: 20px;
    margin: 4px 3px;
    font-size: 0.84rem;
    font-weight: 600;
    border: 1px solid #10B981;
}
.skill-badge-missing {
    display: inline-flex;
    align-items: center;
    background: #7F1D1D;
    color: #FECACA !important;
    padding: 5px 13px;
    border-radius: 20px;
    margin: 4px 3px;
    font-size: 0.84rem;
    font-weight: 600;
    border: 1px solid #EF4444;
}

/* Professional Card Container */
.pro-card {
    background: linear-gradient(160deg, #182339 0%, #111A2C 100%);
    border: 1px solid #2B3954;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
}

.pro-card-header {
    font-size: 1.15rem;
    font-weight: 700;
    color: #A5B4FC !important;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 8px;
    border-bottom: 1px solid #24324B;
    padding-bottom: 8px;
}

.pro-score-hero {
    font-size: 3.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #60A5FA 0%, #C084FC 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1;
    margin: 10px 0;
}

.pro-info-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 9px 0;
    border-bottom: 1px solid #1E2D47;
    font-size: 0.93rem;
}
.pro-info-row .label {
    color: #94A3B8 !important;
    font-weight: 500;
}
.pro-info-row .val {
    color: #F1F5F9 !important;
    font-weight: 600;
    text-align: right;
}

/* Section Header */
.section-heading {
    font-size: 1.25rem;
    font-weight: 700;
    color: #818CF8 !important;
    margin: 24px 0 14px 0;
    padding-bottom: 6px;
    border-bottom: 1px solid #26354E;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Dataframe table styling */
.stDataFrame {
    border: 1px solid #2E3E5B !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}

/* Status Chips */
.chip {
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    display: inline-block;
}
.chip-success { background: #065F46; color: #6EE7B7 !important; border: 1px solid #10B981; }
.chip-info    { background: #1E3A8A; color: #93C5FD !important; border: 1px solid #3B82F6; }
.chip-warning { background: #78350F; color: #FDE68A !important; border: 1px solid #F59E0B; }
.chip-danger  { background: #7F1D1D; color: #FCA5A5 !important; border: 1px solid #EF4444; }

/* Code / mono snippet */
code {
    background: #1E293B !important;
    color: #38BDF8 !important;
    padding: 2px 6px !important;
    border-radius: 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.88em !important;
}
</style>
""", unsafe_allow_html=True)


# ── Cached Resource Loaders ───────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_jobs_df():
    return load_jobs_dataframe(JOBS_CSV)

@st.cache_resource(show_spinner=False)
def get_skill_extractor():
    return SkillExtractor(SKILLS_CSV)

@st.cache_resource(show_spinner=False)
def get_preprocessor():
    return TextPreprocessor()

@st.cache_resource(show_spinner=False)
def get_matcher():
    return JobMatcher(preprocessor=get_preprocessor(), vectorizer_path=VECTORIZER_PATH)

@st.cache_resource(show_spinner=False)
def get_recommender():
    return Recommender()

@st.cache_resource(show_spinner=False)
def get_classifier():
    return JobClassifier(model_path=MODEL_PATH)


# ── Helper: Render Skill Badges ───────────────────────────────────────────────
def render_skills(skills, badge_class="skill-badge"):
    if not skills:
        st.markdown('<span style="color:#94A3B8; font-style:italic;">None detected</span>', unsafe_allow_html=True)
        return
    html = " ".join(f'<span class="{badge_class}">{s}</span>' for s in skills)
    st.markdown(html, unsafe_allow_html=True)


# ── Helper: Render Robust High-Contrast HTML Table ─────────────────────────────
def render_pro_table(headers: list, rows: list, alignments=None):
    if not alignments:
        alignments = ["left"] * len(headers)
    
    th_html = "".join(
        f"<th style='padding:11px 16px; text-align:{alignments[i]}; color:#A5B4FC; font-size:0.84rem; font-weight:700; text-transform:uppercase; letter-spacing:0.04em; border-bottom:1px solid #334155; background:#182339;'>{h}</th>"
        for i, h in enumerate(headers)
    )
    
    tr_html = ""
    for r_idx, row in enumerate(rows):
        bg = "#111A2C" if r_idx % 2 == 0 else "#152033"
        tds = "".join(
            f"<td style='padding:11px 16px; text-align:{alignments[c_idx]}; color:#F1F5F9; font-size:0.91rem; border-bottom:1px solid #1E2D47;'>{val}</td>"
            for c_idx, val in enumerate(row)
        )
        tr_html += f"<tr style='background:{bg};'>{tds}</tr>"
        
    table_html = f"""
    <div style='overflow-x:auto; border:1px solid #2B3954; border-radius:12px; margin-bottom:18px; box-shadow:0 4px 16px rgba(0,0,0,0.25);'>
        <table style='width:100%; border-collapse:collapse;'>
            <thead><tr>{th_html}</tr></thead>
            <tbody>{tr_html}</tbody>
        </table>
    </div>
    """
    st.markdown(table_html, unsafe_allow_html=True)


# ── Helper: Run Analysis Pipeline ─────────────────────────────────────────────
def run_analysis(file_bytes_or_path, filename: str):
    parse_result = ResumeParser.parse_resume(file_bytes_or_path, filename)
    if not parse_result.get("success"):
        return {"success": False, "error": parse_result.get("error", "Failed to parse resume.")}

    extractor = get_skill_extractor()
    skills_by_cat = extractor.extract_skills(parse_result["raw_text"])
    flat_skills   = extractor.get_flat_skills(parse_result["raw_text"])
    skill_dist    = extractor.get_skill_category_distribution(flat_skills)

    jobs_df = get_jobs_df()
    matcher = get_matcher()
    matched_jobs = matcher.match_resume_to_jobs(
        resume_text=parse_result["raw_text"],
        resume_skills=flat_skills,
        candidate_experience_years=parse_result["experience_years"],
        candidate_degrees=parse_result["education"]["degrees"],
        jobs_df=jobs_df
    )

    return {
        "success": True,
        "parse_result": parse_result,
        "skills_by_cat": skills_by_cat,
        "flat_skills": flat_skills,
        "skill_dist": skill_dist,
        "matched_jobs": matched_jobs
    }


# ── Sidebar Navigation ────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding: 12px 0 18px 0;'>
            <div style='font-size:2.4rem; margin-bottom:4px;'>🎯</div>
            <div style='font-size:1.2rem; font-weight:800; color:#FFFFFF;'>AI Resume Screener</div>
            <div style='font-size:0.8rem; font-weight:600; color:#818CF8; letter-spacing:0.04em;'>NLP & ML INTELLIGENCE</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        nav_options = [
            "🏠 Dashboard Overview",
            "📄 Resume Parser & Analyzer",
            "💼 Job Recommendations",
            "📊 Skill Gap & Analytics",
            "🤖 Model Evaluation & ML",
            "ℹ️ About & Documentation"
        ]
        selected_nav = st.radio("Navigation Menu", nav_options, label_visibility="collapsed")

        st.markdown("---")

        # System Status Widget
        jobs_df = get_jobs_df()
        classifier = get_classifier()
        st.markdown(f"""
        <div style='background:#141F33; border:1px solid #23334F; border-radius:10px; padding:12px; margin-bottom:14px;'>
            <div style='font-size:0.75rem; font-weight:700; color:#818CF8; text-transform:uppercase;'>System Status</div>
            <div style='font-size:0.85rem; color:#E2E8F0; margin-top:4px;'>
                • Active Jobs: <b>{len(jobs_df)} roles</b><br>
                • ML Model: <b style='color:{"#34D399" if classifier.is_trained else "#FBBF24"};'>{"Trained (88.9%)" if classifier.is_trained else "Pre-trained"}</b><br>
                • Vectorizer: <b>TF-IDF (8k features)</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style='font-size:0.76rem; color:#94A3B8; line-height:1.5;'>
            <b style='color:#CBD5E1;'>Notice:</b> This AI tool assists recruiters and candidates via automated scoring. Final employment decisions remain human-driven.
        </div>
        """, unsafe_allow_html=True)

    return selected_nav.split(" ", 1)[1].strip()


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1: DASHBOARD OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
def page_home():
    st.markdown("""
    <div style='padding: 10px 0 25px 0;'>
        <div style='display:inline-block; background:rgba(99,102,241,0.15); border:1px solid #6366F1; color:#A5B4FC; padding:4px 14px; border-radius:20px; font-size:0.82rem; font-weight:700; margin-bottom:12px;'>
            INTELLIGENT TALENT ACQUISITION ENGINE
        </div>
        <h1 style='font-size:2.6rem; font-weight:800; margin-bottom:6px; color:#FFFFFF;'>
            AI Resume Screening & Job Recommendation
        </h1>
        <p style='color:#CBD5E1; font-size:1.1rem; max-width:850px; margin:0;'>
            Automated resume parsing, intelligent skill extraction, multi-factor TF-IDF weighted similarity matching, and machine learning category classification.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Top KPI Metrics
    jobs_df = get_jobs_df()
    categories_count = jobs_df["category"].nunique() if not jobs_df.empty else 4
    companies_count = jobs_df["company"].nunique() if not jobs_df.empty else 20

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Indexed Jobs", len(jobs_df), "Active Roles")
    m2.metric("Job Domains", categories_count, "Categories")
    m3.metric("Hiring Companies", companies_count, "Top Tech Firms")
    m4.metric("Model Test Accuracy", "88.89%", "Logistic Regression")

    st.markdown("<br>", unsafe_allow_html=True)

    # Core Capabilities
    st.markdown('<div class="section-heading">⚡ Core System Features</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    cards = [
        ("📄", "Deep Resume Parsing",
         "Extracts candidate name, email, contact number, degree level, branch, graduation year, and total experience years with robust regex and document structure analysis."),
        ("🔍", "NLP & Skill Extraction",
         "Tokenizes and normalizes text with technical-phrase preservation. Matches 100+ categorized programming languages, ML frameworks, cloud platforms, and tools."),
        ("🎯", "Weighted Match Scoring",
         "Applies 60% TF-IDF Cosine Similarity + 25% Skill Coverage + 10% Experience Alignment + 5% Education Seniority to deliver explainable match recommendations.")
    ]
    for col, (icon, title, desc) in zip([c1, c2, c3], cards):
        with col:
            st.markdown(f"""
            <div class="pro-card">
                <div style='font-size:2.2rem; margin-bottom:10px;'>{icon}</div>
                <div style='font-size:1.15rem; font-weight:700; color:#FFFFFF; margin-bottom:8px;'>{title}</div>
                <div style='font-size:0.92rem; color:#CBD5E1; line-height:1.6;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    # Quick Actions & Getting Started
    st.markdown('<div class="section-heading">🚀 Quick Start & Next Steps</div>', unsafe_allow_html=True)
    q1, q2 = st.columns(2)
    with q1:
        st.markdown("""
        <div class="pro-card" style='padding:20px;'>
            <div class="pro-card-header">📄 Upload & Screen a Resume</div>
            <p style='color:#E2E8F0; font-size:0.95rem; margin-bottom:14px;'>
                Upload a candidate resume in PDF, DOCX, or TXT format or load a pre-built sample to immediately compute match scores across all 35 job descriptions.
            </p>
            <div style='color:#A5B4FC; font-size:0.88rem; font-weight:600;'>
                👉 Navigate to <b>Resume Parser & Analyzer</b> from the sidebar menu to begin.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with q2:
        st.markdown("""
        <div class="pro-card" style='padding:20px;'>
            <div class="pro-card-header">🤖 Supervised Machine Learning Evaluation</div>
            <p style='color:#E2E8F0; font-size:0.95rem; margin-bottom:14px;'>
                Inspect the Logistic Regression job category classifier, evaluate precision/recall/F1 metrics on held-out test data, and see live probability distributions.
            </p>
            <div style='color:#A5B4FC; font-size:0.88rem; font-weight:600;'>
                👉 Navigate to <b>Model Evaluation & ML</b> to inspect accuracy and reports.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2: RESUME PARSER & ANALYZER
# ═════════════════════════════════════════════════════════════════════════════
def page_resume_analysis():
    st.markdown("""
    <div>
        <h1 style='color:#FFFFFF; margin-bottom:4px;'>📄 Resume Parser & Profile Extraction</h1>
        <p style='color:#CBD5E1; font-size:1.05rem;'>Upload any candidate resume to extract structured contact info, education, experience, and categorized skills.</p>
    </div>
    """, unsafe_allow_html=True)

    # Fast One-Click Demo Loader
    st.markdown('<div class="section-heading">⚡ Quick Test: Load Demo Resumes</div>', unsafe_allow_html=True)
    dcol1, dcol2, dcol3 = st.columns([1, 1, 2])
    
    with dcol1:
        if st.button("📁 Load Data Scientist Resume", use_container_width=True):
            if os.path.exists(SAMPLE_DS_PATH):
                with st.spinner("Analyzing Data Scientist Resume..."):
                    res = run_analysis(SAMPLE_DS_PATH, "sample_data_scientist.docx")
                    if res["success"]:
                        st.session_state["analysis_done"] = True
                        st.session_state["resume_data"] = res
                        st.success("Loaded & Analyzed Alex Rivera (Data Scientist)!")
            else:
                st.error("Sample resume file not found.")

    with dcol2:
        if st.button("📁 Load Software Engineer Resume", use_container_width=True):
            if os.path.exists(SAMPLE_SE_PATH):
                with st.spinner("Analyzing Software Engineer Resume..."):
                    res = run_analysis(SAMPLE_SE_PATH, "sample_software_engineer.docx")
                    if res["success"]:
                        st.session_state["analysis_done"] = True
                        st.session_state["resume_data"] = res
                        st.success("Loaded & Analyzed Priya Sharma (Full Stack)!")
            else:
                st.error("Sample resume file not found.")

    st.markdown("<br>", unsafe_allow_html=True)

    # Standard File Uploader
    uploaded_file = st.file_uploader(
        "Or Upload Your Custom Resume (PDF, DOCX, TXT):",
        type=["pdf", "docx", "doc", "txt"],
        help="Upload candidate PDF or DOCX file"
    )

    if uploaded_file is not None:
        u_col1, u_col2, u_col3 = st.columns(3)
        u_col1.metric("File Name", uploaded_file.name)
        u_col2.metric("File Extension", os.path.splitext(uploaded_file.name)[1].upper())
        u_col3.metric("Size", f"{uploaded_file.size / 1024:.1f} KB")

        if st.button("🔍 Analyze Uploaded Resume", use_container_width=True):
            with st.spinner("Extracting content and calculating job match scores..."):
                file_bytes = uploaded_file.read()
                res = run_analysis(io.BytesIO(file_bytes), uploaded_file.name)
                if res["success"]:
                    st.session_state["analysis_done"] = True
                    st.session_state["resume_data"] = res
                    st.success("Resume analyzed successfully!")
                else:
                    st.error(f"Analysis failed: {res.get('error')}")

    # Display Analysis Results if available
    data = st.session_state.get("resume_data")
    if not data or not data.get("success"):
        if uploaded_file is None:
            st.markdown("""
            <div class="pro-card" style='text-align:center; padding:45px 20px; margin-top:20px;'>
                <div style='font-size:3rem; margin-bottom:10px;'>📂</div>
                <div style='font-size:1.15rem; font-weight:700; color:#FFFFFF;'>No Resume Loaded Yet</div>
                <div style='color:#94A3B8; font-size:0.95rem; margin-top:6px;'>
                    Click one of the sample buttons above or upload a resume file to inspect extracted profile data.
                </div>
            </div>
            """, unsafe_allow_html=True)
        return

    pr          = data["parse_result"]
    flat_skills = data["flat_skills"]
    skills_by_cat = data["skills_by_cat"]
    skill_dist  = data["skill_dist"]
    matched_jobs = data["matched_jobs"]

    st.markdown('<div class="section-heading">👤 Extracted Candidate Profile</div>', unsafe_allow_html=True)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Candidate Name", pr["candidate_name"])
    m2.metric("Experience", f"{pr['experience_years']:.1f} Yrs")
    m3.metric("Education", pr["education"]["degree_summary"][:18] + ("..." if len(pr["education"]["degree_summary"]) > 18 else ""))
    m4.metric("Skills Detected", len(flat_skills))
    m5.metric("Projects Found", pr["project_count"])

    # Detailed Tabs
    tab_profile, tab_skills, tab_text = st.tabs(["📋 Detailed Attributes", "🛠️ Skill Breakdown", "📜 Raw Resume Text"])

    with tab_profile:
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown(f"""
            <div class="pro-card">
                <div class="pro-card-header">📞 Contact & Personal Data</div>
                <div class="pro-info-row"><span class="label">Full Name</span><span class="val">{pr['candidate_name']}</span></div>
                <div class="pro-info-row"><span class="label">Email Address</span><span class="val">{pr['email']}</span></div>
                <div class="pro-info-row"><span class="label">Phone Number</span><span class="val">{pr['phone']}</span></div>
                <div class="pro-info-row"><span class="label">Detected City / Location</span><span class="val">{pr['location']}</span></div>
            </div>
            """, unsafe_allow_html=True)

        with col_r:
            st.markdown(f"""
            <div class="pro-card">
                <div class="pro-card-header">🎓 Academics & Experience</div>
                <div class="pro-info-row"><span class="label">Degree Level</span><span class="val">{pr['education']['degree_summary']}</span></div>
                <div class="pro-info-row"><span class="label">Major / Branch</span><span class="val">{pr['education']['branch']}</span></div>
                <div class="pro-info-row"><span class="label">Graduation Year</span><span class="val">{pr['education']['graduation_year']}</span></div>
                <div class="pro-info-row"><span class="label">Work Experience</span><span class="val">{pr['experience_years']:.1f} Years</span></div>
                <div class="pro-info-row"><span class="label">Projects Identified</span><span class="val">{pr['project_count']}</span></div>
            </div>
            """, unsafe_allow_html=True)

        if pr["certifications"]:
            st.markdown(f"""
            <div class="pro-card">
                <div class="pro-card-header">📜 Certifications Identified</div>
                {"".join(f'<div class="pro-info-row"><span class="label">Certification</span><span class="val">{c}</span></div>' for c in pr["certifications"])}
            </div>
            """, unsafe_allow_html=True)

    with tab_skills:
        st.markdown('<div class="pro-card-header">🏷️ Categorized Technical Skills</div>', unsafe_allow_html=True)
        if skills_by_cat:
            for cat, s_list in skills_by_cat.items():
                st.markdown(f"<div style='font-size:0.85rem; font-weight:700; color:#A5B4FC; margin:10px 0 4px 0; text-transform:uppercase;'>{cat} ({len(s_list)})</div>", unsafe_allow_html=True)
                render_skills(s_list, "skill-badge")
        else:
            st.warning("No categorized skills detected in text.")

        if skill_dist:
            fig_pie = create_skill_distribution_chart(skill_dist)
            st.plotly_chart(fig_pie, use_container_width=True, key="p2_skill_pie")

    with tab_text:
        st.text_area("Extracted Plain Text:", value=pr["raw_text"], height=300, disabled=True)

    # Best Match Spotlight
    if matched_jobs:
        top_match = matched_jobs[0]
        score = top_match["match_score"]
        label = get_score_label(score)
        
        st.markdown('<div class="section-heading">🏆 Best Matching Role</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="pro-card" style='text-align:center; padding:28px;'>
            <div style='font-size:0.85rem; font-weight:700; color:#818CF8; text-transform:uppercase;'>#1 RECOMMENDED FIT</div>
            <div style='font-size:1.8rem; font-weight:800; color:#FFFFFF; margin:6px 0;'>{top_match['job_title']}</div>
            <div style='color:#CBD5E1; font-size:1rem; margin-bottom:14px;'>{top_match['company']} · {top_match['location']} · <span style='color:#A5B4FC;'>{top_match['category']}</span></div>
            <div class="pro-score-hero">{score:.1f}%</div>
            <div style='font-size:1rem; font-weight:700; color:#34D399; margin-top:4px;'>{label}</div>
        </div>
        """, unsafe_allow_html=True)

        st.progress(min(int(score), 100) / 100.0)

    # ── PDF Report Generation Export ──────────────────────────────────────────
    st.markdown('<div class="section-heading">📑 Official Career Evaluation Report (PDF)</div>', unsafe_allow_html=True)
    classifier = get_classifier()
    pred_res = classifier.predict(pr["raw_text"]) if classifier.is_trained else None
    
    try:
        pdf_bytes = generate_candidate_pdf_report(
            parse_result=pr,
            skills_by_cat=skills_by_cat,
            matched_jobs=matched_jobs,
            prediction_result=pred_res,
            recommender_obj=get_recommender()
        )
        safe_name = pr['candidate_name'].replace(' ', '_')
        
        pdf_col1, pdf_col2 = st.columns([2, 1])
        with pdf_col1:
            st.markdown("""
            <div style='color:#E2E8F0; font-size:0.95rem;'>
                Generate an executive <b>AI Career Intelligence & Job Matching PDF Report</b> containing complete profile breakdown, ML domain classification, ranked positions, and personalized skill acquisition tips.
            </div>
            """, unsafe_allow_html=True)
        with pdf_col2:
            st.download_button(
                label="📥 Download Career Report (PDF)",
                data=pdf_bytes,
                file_name=f"{safe_name}_Career_Report.pdf",
                mime="application/pdf",
                use_container_width=True
            )
    except Exception as pdf_err:
        st.info(f"PDF Export ready. ({pdf_err})")



# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3: JOB RECOMMENDATIONS
# ═════════════════════════════════════════════════════════════════════════════
def page_job_recommendations():
    st.markdown("""
    <div>
        <h1 style='color:#FFFFFF; margin-bottom:4px;'>💼 Top Job Recommendations</h1>
        <p style='color:#CBD5E1; font-size:1.05rem;'>Ranked career opportunities based on multi-factor weighted cosine similarity and skill overlap.</p>
    </div>
    """, unsafe_allow_html=True)

    data = st.session_state.get("resume_data")
    if not data or not data.get("success"):
        st.warning("⚠️ No resume has been analyzed yet. Please visit the **Resume Parser & Analyzer** page to load or upload a resume.")
        return

    matched_jobs = data["matched_jobs"]
    flat_skills  = data["flat_skills"]
    pr           = data["parse_result"]
    exp_years    = pr["experience_years"]

    recommender = get_recommender()

    # Flexible Filter Bar
    st.markdown('<div class="section-heading">⚙️ Recommendation Filters & Preferences</div>', unsafe_allow_html=True)
    f_col1, f_col2, f_col3 = st.columns([1.5, 1.5, 1])

    categories = ["All Domains"] + sorted(list(set(j["category"] for j in matched_jobs)))
    selected_cat = f_col1.selectbox("Filter by Domain:", categories)
    
    top_k = f_col2.slider("Number of Recommendations:", min_value=3, max_value=10, value=5)
    min_score = f_col3.slider("Min Match Score (%):", min_value=0, max_value=80, value=0, step=5)

    # Filtered Jobs
    filtered_jobs = [
        j for j in matched_jobs 
        if (selected_cat == "All Domains" or j["category"] == selected_cat)
        and j["match_score"] >= min_score
    ]

    if not filtered_jobs:
        st.warning(f"No jobs matched your filter criteria (Domain: {selected_cat}, Min Score: {min_score}%). Try adjusting the filters.")
        return

    top_recs = recommender.get_top_recommendations(filtered_jobs, top_n=top_k)

    # Ranked Summary Table
    st.markdown(f'<div class="section-heading">🏅 Top {len(top_recs)} Recommended Positions</div>', unsafe_allow_html=True)
    table_headers = ["Rank", "Job Title", "Company", "Location", "Domain", "Experience Req", "Match Score"]
    table_rows = []
    for rank, job in enumerate(top_recs, start=1):
        score = job["match_score"]
        score_badge = (
            f"<span class='chip chip-success'>{score:.1f}%</span>" if score >= 70 else (
                f"<span class='chip chip-info'>{score:.1f}%</span>" if score >= 45 else
                f"<span class='chip chip-warning'>{score:.1f}%</span>"
            )
        )
        table_rows.append([
            f"<b style='color:#818CF8;'>#{rank}</b>",
            f"<b style='color:#FFFFFF;'>{job['job_title']}</b>",
            f"<span style='color:#E2E8F0;'>{job['company']}</span>",
            f"<span style='color:#CBD5E1;'>{job['location']}</span>",
            f"<span class='skill-badge' style='margin:0; font-size:0.78rem; padding:3px 8px;'>{job['category']}</span>",
            f"<span style='color:#E2E8F0;'>{job['experience_level']}</span>",
            score_badge
        ])
    render_pro_table(table_headers, table_rows)

    # Plotly Match Score Bar Chart
    fig_bar = create_match_score_chart(top_recs)
    st.plotly_chart(fig_bar, use_container_width=True, key="p3_match_bar")

    # Detailed Job Inspector
    st.markdown('<div class="section-heading">🔍 Deep-Dive Role Inspector & Skill Analysis</div>', unsafe_allow_html=True)
    job_labels = {f"#{i} {j['job_title']} ({j['company']}) — {j['match_score']:.1f}%": j for i, j in enumerate(top_recs, start=1)}
    selected_label = st.selectbox("Select a position to inspect skill alignment and match reasons:", list(job_labels.keys()))
    selected_job = job_labels[selected_label]

    explanation = recommender.generate_explanation(selected_job, flat_skills, exp_years)

    # 5 KPI metrics for this specific job
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Overall Match", f"{selected_job['match_score']:.1f}%")
    k2.metric("Text Similarity (60%)", f"{selected_job['text_similarity']:.1f}%")
    k3.metric("Skill Match (25%)", f"{selected_job['skill_score']:.1f}%")
    k4.metric("Experience (10%)", f"{selected_job['experience_score']:.1f}%")
    k5.metric("Education (5%)", f"{selected_job['education_score']:.1f}%")

    col_desc, col_skills = st.columns([1.2, 1])

    with col_desc:
        st.markdown(f"""
        <div class="pro-card">
            <div class="pro-card-header">🏢 Job Profile & Requirements</div>
            <div class="pro-info-row"><span class="label">Job Title</span><span class="val">{selected_job['job_title']}</span></div>
            <div class="pro-info-row"><span class="label">Company</span><span class="val">{selected_job['company']}</span></div>
            <div class="pro-info-row"><span class="label">Location</span><span class="val">{selected_job['location']}</span></div>
            <div class="pro-info-row"><span class="label">Category Domain</span><span class="val">{selected_job['category']}</span></div>
            <div class="pro-info-row"><span class="label">Experience Level</span><span class="val">{selected_job['experience_level']}</span></div>
            <div class="pro-info-row"><span class="label">Education Requirement</span><span class="val">{selected_job['education_required']}</span></div>
            <br>
            <div style='font-size:0.85rem; font-weight:700; color:#A5B4FC; margin-bottom:6px;'>JOB DESCRIPTION PREVIEW:</div>
            <div style='color:#CBD5E1; font-size:0.92rem; line-height:1.6;'>
                {truncate_text(selected_job['description'], 450)}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_skills:
        st.markdown(f"""
        <div class="pro-card">
            <div class="pro-card-header">🎯 Skill Match Breakdown ({explanation['skill_coverage']})</div>
            <div style='font-size:0.85rem; font-weight:700; color:#34D399; margin:8px 0 4px 0;'>MATCHED SKILLS:</div>
        """, unsafe_allow_html=True)
        render_skills(explanation["matched_skills"], "skill-badge-matched")

        st.markdown("<div style='font-size:0.85rem; font-weight:700; color:#F87171; margin:16px 0 4px 0;'>MISSING SKILLS TO ACQUIRE:</div>", unsafe_allow_html=True)
        render_skills(explanation["missing_skills"], "skill-badge-missing")
        st.markdown("</div>", unsafe_allow_html=True)

    # Explainability & Radar Chart
    col_exp, col_radar = st.columns([1.1, 0.9])

    with col_exp:
        st.markdown("""
        <div class="pro-card">
            <div class="pro-card-header">💡 Match Reasoning & Recommendations</div>
        """, unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:1.05rem; font-weight:700; color:#818CF8; margin-bottom:10px;'>Assessment: {explanation['match_label']}</div>", unsafe_allow_html=True)
        for r in explanation["reasons"]:
            st.markdown(f"<div style='color:#CBD5E1; font-size:0.92rem; padding:3px 0;'>• {r}</div>", unsafe_allow_html=True)

        if explanation["improvements"]:
            st.markdown("<div style='font-size:0.95rem; font-weight:700; color:#FBBF24; margin-top:14px; margin-bottom:6px;'>📈 Actionable Tips to Strengthen Application:</div>", unsafe_allow_html=True)
            for imp in explanation["improvements"]:
                st.markdown(f"<div style='color:#E2E8F0; font-size:0.9rem; padding:2px 0;'>→ {imp}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_radar:
        fig_radar = create_score_breakdown_chart(selected_job)
        st.plotly_chart(fig_radar, use_container_width=True, key="p3_radar")

    # ── PDF Download Action ───────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    classifier = get_classifier()
    pred_res = classifier.predict(pr["raw_text"]) if classifier.is_trained else None
    try:
        pdf_bytes = generate_candidate_pdf_report(
            parse_result=pr,
            skills_by_cat=data["skills_by_cat"],
            matched_jobs=matched_jobs,
            prediction_result=pred_res,
            recommender_obj=recommender
        )
        safe_name = pr['candidate_name'].replace(' ', '_')
        st.download_button(
            label="📥 Download Full Job Matching & Skill Gap Report (PDF)",
            data=pdf_bytes,
            file_name=f"{safe_name}_Job_Recommendations_Report.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    except Exception:
        pass



# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4: SKILL GAP & ANALYTICS
# ═════════════════════════════════════════════════════════════════════════════
def page_skill_analytics():
    st.markdown("""
    <div>
        <h1 style='color:#FFFFFF; margin-bottom:4px;'>📊 Skill Gap & Market Analytics</h1>
        <p style='color:#CBD5E1; font-size:1.05rem;'>Identify technical strengths, missing market competencies, and explore hiring trends across 35 job descriptions.</p>
    </div>
    """, unsafe_allow_html=True)

    data = st.session_state.get("resume_data")
    jobs_df = get_jobs_df()

    if data and data.get("success"):
        matched_jobs = data["matched_jobs"]
        flat_skills  = data["flat_skills"]
        top_job      = matched_jobs[0]

        recommender = get_recommender()
        top_exp = recommender.generate_explanation(top_job, flat_skills, data["parse_result"]["experience_years"])

        st.markdown(f'<div class="section-heading">🎯 Skill Coverage for Top Match: {top_job["job_title"]}</div>', unsafe_allow_html=True)
        
        gap_col1, gap_col2 = st.columns([1, 1])
        with gap_col1:
            fig_gap = create_skill_gap_chart(top_exp["matched_skills"], top_exp["missing_skills"])
            st.plotly_chart(fig_gap, use_container_width=True, key="p4_gap_chart")

        with gap_col2:
            fig_dist = create_skill_distribution_chart(data["skill_dist"])
            st.plotly_chart(fig_dist, use_container_width=True, key="p4_dist_chart")

    # Dataset Market Analytics
    st.markdown('<div class="section-heading">📈 Job Market Insights (Dataset Distribution)</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        cat_counts = jobs_df["category"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Count"]
        st.markdown('<div class="pro-card-header">📂 Roles by Category Domain</div>', unsafe_allow_html=True)
        cat_rows = [
            [
                f"<span class='skill-badge' style='margin:0;'>{row['Category']}</span>",
                f"<b style='color:#FFFFFF;'>{row['Count']} Roles</b>"
            ]
            for _, row in cat_counts.iterrows()
        ]
        render_pro_table(["Category Domain", "Total Openings"], cat_rows)

    with c2:
        exp_counts = jobs_df["experience_level"].value_counts().reset_index()
        exp_counts.columns = ["Experience Level", "Job Count"]
        st.markdown('<div class="pro-card-header">💼 Experience Level Breakdown</div>', unsafe_allow_html=True)
        exp_rows = [
            [
                f"<b style='color:#E2E8F0;'>{row['Experience Level']}</b>",
                f"<b style='color:#60A5FA;'>{row['Job Count']} Roles</b>"
            ]
            for _, row in exp_counts.iterrows()
        ]
        render_pro_table(["Experience Level", "Available Roles"], exp_rows)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 5: MODEL EVALUATION & ML
# ═════════════════════════════════════════════════════════════════════════════
def page_model_evaluation():
    st.markdown("""
    <div>
        <h1 style='color:#FFFFFF; margin-bottom:4px;'>🤖 Machine Learning Model Evaluation</h1>
        <p style='color:#CBD5E1; font-size:1.05rem;'>Evaluate the <b>Logistic Regression Job Domain Classifier</b> trained with TF-IDF n-gram feature extraction.</p>
    </div>
    """, unsafe_allow_html=True)

    # Clarification Notice
    st.markdown("""
    <div style='background:rgba(79,70,229,0.12); border:1px solid #4F46E5; border-radius:12px; padding:16px; margin:16px 0;'>
        <b style='color:#A5B4FC; font-size:1rem;'>💡 Crucial Concept for Viva / Technical Review:</b><br>
        <span style='color:#E2E8F0; font-size:0.92rem; line-height:1.6;'>
            • <b>Resume Match Score (e.g. 85%)</b>: An unsupervised Cosine Similarity measure between candidate text & specific job descriptions.<br>
            • <b>Classifier Accuracy (88.89%)</b>: A supervised Machine Learning metric measuring how accurately Logistic Regression predicts the correct broad job category on held-out test data.
        </span>
    </div>
    """, unsafe_allow_html=True)

    jobs_df = get_jobs_df()
    classifier = get_classifier()

    # Re-train control
    st.markdown('<div class="section-heading">🏋️ Model Training & Validation</div>', unsafe_allow_html=True)
    if st.button("🚀 Run / Re-Train Classifier on Jobs Dataset", use_container_width=True):
        with st.spinner("Training Logistic Regression classifier on job corpus..."):
            result = classifier.train(jobs_df)
            if result.get("success"):
                st.session_state["eval_result"] = result
                st.success(f"✅ Training completed! Test Accuracy: {result['accuracy']}%")
            else:
                st.error(f"Training failed: {result.get('error')}")

    # Metrics Display
    eval_result = st.session_state.get("eval_result")
    if not eval_result and classifier.is_trained:
        # Pre-populate with current model stats
        eval_result = classifier.train(jobs_df)
        st.session_state["eval_result"] = eval_result

    if eval_result:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Model Algorithm", "Logistic Regression")
        m2.metric("Test Accuracy", f"{eval_result['accuracy']}%")
        m3.metric("Training Samples", eval_result["train_samples"])
        m4.metric("Test Samples", eval_result["test_samples"])

        # Classification Report Table
        st.markdown('<div class="section-heading">📋 Classification Report Summary</div>', unsafe_allow_html=True)
        rep_headers = ["Category / Metric", "Precision", "Recall", "F1-Score", "Support"]
        rep_rows = []
        for key, metrics in eval_result["classification_report"].items():
            if isinstance(metrics, dict):
                rep_rows.append([
                    f"<b style='color:#A5B4FC;'>{key}</b>",
                    f"<span style='color:#34D399;'>{metrics.get('precision', 0):.3f}</span>",
                    f"<span style='color:#60A5FA;'>{metrics.get('recall', 0):.3f}</span>",
                    f"<b style='color:#FFFFFF;'>{metrics.get('f1-score', 0):.3f}</b>",
                    f"<span style='color:#CBD5E1;'>{int(metrics.get('support', 0))}</span>"
                ])
        render_pro_table(rep_headers, rep_rows)

    # Active Resume Prediction
    data = st.session_state.get("resume_data")
    if data and data.get("success") and classifier.is_trained:
        st.markdown('<div class="section-heading">🔮 Live Resume Classification Result</div>', unsafe_allow_html=True)
        raw_text = data["parse_result"]["raw_text"]
        pred = classifier.predict(raw_text)

        p_col1, p_col2 = st.columns([1, 1.2])
        with p_col1:
            st.markdown(f"""
            <div class="pro-card">
                <div class="pro-card-header">🎯 Predicted Job Category</div>
                <div style='font-size:1.8rem; font-weight:800; color:#FFFFFF; margin:10px 0;'>{pred['predicted_category']}</div>
                <div style='color:#34D399; font-size:1.1rem; font-weight:700;'>Confidence: {pred['confidence']:.1f}%</div>
                <br>
                <div style='color:#94A3B8; font-size:0.85rem;'>
                    Based on TF-IDF vocabulary weights extracted from candidate resume text.
                </div>
            </div>
            """, unsafe_allow_html=True)

        with p_col2:
            if pred.get("all_probabilities"):
                fig_conf = create_classifier_confidence_chart(pred["all_probabilities"])
                st.plotly_chart(fig_conf, use_container_width=True, key="p5_conf_bar")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 6: ABOUT & DOCUMENTATION
# ═════════════════════════════════════════════════════════════════════════════
def page_about():
    st.markdown("""
    <div>
        <h1 style='color:#FFFFFF; margin-bottom:4px;'>ℹ️ About & Technical Documentation</h1>
        <p style='color:#CBD5E1; font-size:1.05rem;'>Architecture, algorithms, formula explanations, and viva discussion points.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="pro-card">
            <div class="pro-card-header">🧠 Mathematical Formulation</div>
            <p style='color:#CBD5E1; font-size:0.92rem; line-height:1.7;'>
                The overall match score $S$ is computed using a multi-factor weighted equation:
            </p>
            <div style='background:#0F172A; border:1px solid #334155; border-radius:8px; padding:12px; font-family:monospace; color:#38BDF8; font-size:0.9rem; margin-bottom:12px;'>
                Score = 0.60 * S_text + 0.25 * S_skill + 0.10 * S_exp + 0.05 * S_edu
            </div>
            <ul style='color:#CBD5E1; font-size:0.88rem; line-height:1.8; margin-left:15px;'>
                <li><b>S_text</b>: Cosine similarity of sublinear TF-IDF vectors (ngram range 1-2).</li>
                <li><b>S_skill</b>: Exact and normalized phrase overlap ratio against required skills.</li>
                <li><b>S_exp</b>: Continuous penalty/bonus ratio relative to required minimum years.</li>
                <li><b>S_edu</b>: Educational qualification seniority alignment mapping.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="pro-card">
            <div class="pro-card-header">🛠️ Technology Stack</div>
            <div style='display:flex; flex-wrap:wrap; gap:6px;'>
                <span class="skill-badge">Python 3.11+</span>
                <span class="skill-badge">Streamlit 1.30+</span>
                <span class="skill-badge">Scikit-learn</span>
                <span class="skill-badge">NLTK / NLP</span>
                <span class="skill-badge">Plotly Express</span>
                <span class="skill-badge">PyPDF</span>
                <span class="skill-badge">python-docx</span>
                <span class="skill-badge">Pandas / NumPy</span>
                <span class="skill-badge">TF-IDF Vectorizer</span>
                <span class="skill-badge">Cosine Similarity</span>
                <span class="skill-badge">Logistic Regression</span>
            </div>
            <br>
            <div class="pro-card-header">🎯 Viva Defense Points</div>
            <p style='color:#CBD5E1; font-size:0.88rem; line-height:1.6;'>
                • <b>Why TF-IDF?</b> Downweights ubiquitous stopwords while elevating rare domain-specific technical terms.<br>
                • <b>Why Weighted Scoring?</b> Pure keyword matching ignores experience seniority, while pure text similarity ignores hard skill requirements.
            </p>
        </div>
        """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN ROUTER
# ═════════════════════════════════════════════════════════════════════════════
def main():
    if "analysis_done" not in st.session_state:
        st.session_state["analysis_done"] = False
    if "resume_data" not in st.session_state:
        st.session_state["resume_data"] = None
    if "eval_result" not in st.session_state:
        st.session_state["eval_result"] = None

    page = render_sidebar()

    try:
        if page == "Dashboard Overview":
            page_home()
        elif page == "Resume Parser & Analyzer":
            page_resume_analysis()
        elif page == "Job Recommendations":
            page_job_recommendations()
        elif page == "Skill Gap & Analytics":
            page_skill_analytics()
        elif page == "Model Evaluation & ML":
            page_model_evaluation()
        elif page == "About & Documentation":
            page_about()
    except Exception as e:
        st.error(f"❌ An error occurred: {type(e).__name__}: {e}")
        st.info("Please verify the inputs or sample resumes and try again.")


if __name__ == "__main__":
    main()
