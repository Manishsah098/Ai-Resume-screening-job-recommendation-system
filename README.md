# AI Resume Screening & Job Recommendation System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue?logo=python" />
  <img src="https://img.shields.io/badge/Streamlit-1.30+-red?logo=streamlit" />
  <img src="https://img.shields.io/badge/Scikit--learn-1.3+-orange?logo=scikit-learn" />
  <img src="https://img.shields.io/badge/NLP-TF--IDF%20%2B%20Cosine%20Similarity-purple" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
</p>

---

## 📋 Description

An intelligent, intermediate-level AI/ML project that:
- Parses **PDF and DOCX** resumes
- Extracts **skills, education, experience** using NLP
- Compares resumes against **35+ job descriptions** using **TF-IDF + Cosine Similarity**
- Ranks and recommends the **top 5 best-matching jobs** with explainable reasoning
- Highlights **matched and missing skills** for each recommendation
- Classifies resumes into **job categories** using Logistic Regression

Built for college AI/ML projects, internship demos, and technical vivas.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📄 **Resume Upload** | PDF and DOCX format support |
| 🔍 **NLP Extraction** | Skills, education, experience, name, email, phone |
| 🧠 **TF-IDF Matching** | Bi-gram TF-IDF vectorization with 8000 features |
| 📐 **Cosine Similarity** | Vector similarity for resume–job matching |
| ⚖️ **Weighted Scoring** | 60% text + 25% skills + 10% experience + 5% education |
| 🏅 **Job Ranking** | Top 5 jobs ranked by weighted match score |
| 💡 **Explainability** | Human-readable reasons for each recommendation |
| 🔴 **Skill Gap Analysis** | Matched vs. missing skills per job |
| 📊 **Charts** | Bar chart, donut chart, radar chart, skill gap chart |
| 🤖 **Classifier** | Logistic Regression for job category prediction |
| 📈 **Model Evaluation** | Accuracy, Precision, Recall, F1-score on test data |
| 🌑 **Dark UI** | Modern Streamlit dark dashboard |

---

## 🛠️ Technology Stack

```
Python 3.11+        → Core language
Scikit-learn        → TF-IDF, Logistic Regression, Cosine Similarity
NLTK                → Stopword removal, tokenization
Pandas / NumPy      → Data processing
PyPDF               → PDF text extraction
python-docx         → DOCX text extraction
Streamlit           → Web UI dashboard
Plotly              → Interactive charts
Joblib              → Model persistence
```

---

## 🗂️ Project Structure

```
AIResumeScreening/
│
├── app.py                      ← Main Streamlit application
├── requirements.txt            ← Python dependencies
├── README.md
├── .gitignore
│
├── data/
│   ├── jobs.csv                ← 35 sample job records
│   └── skills.csv              ← Categorized skill dictionary
│
├── models/                     ← Auto-created on first run
│   ├── tfidf_vectorizer.pkl
│   └── job_classifier.pkl
│
├── modules/
│   ├── __init__.py
│   ├── resume_parser.py        ← PDF/DOCX extraction + entity parsing
│   ├── text_preprocessor.py   ← NLP cleaning pipeline
│   ├── skill_extractor.py     ← Skill matching from dictionary
│   ├── job_matcher.py         ← TF-IDF + Cosine Similarity + weighted score
│   ├── recommender.py         ← Ranking + explainability
│   └── classifier.py          ← Logistic Regression job category predictor
│
├── utils/
│   ├── __init__.py
│   ├── helpers.py              ← Shared utility functions
│   └── visualization.py       ← Plotly chart builders
│
└── training/
    └── train_model.py          ← Standalone classifier training script
```

---

## ⚙️ System Architecture

```
                    USER
                      |
                      ↓
              Resume Upload (PDF / DOCX)
                      |
                      ↓
             PDF / DOCX Parser (pypdf / python-docx)
                      |
                      ↓
              Text Extraction
                      |
                      ↓
             NLP Preprocessing
             (lowercase → clean → tokenize → remove stopwords)
                      |
          ┌───────────┴───────────┐
          ↓                       ↓
    Skill Extraction        Feature Extraction
    (phrase matching)       (education, experience)
          |                       |
          └───────────┬───────────┘
                      ↓
                TF-IDF Vectorizer (1-2 ngrams, 8000 features)
                      |
          ┌───────────┴───────────┐
          ↓                       ↓
     Resume Vector           Job Vectors (35+ jobs)
          |                       |
          └───────────┬───────────┘
                      ↓
             Cosine Similarity
                      |
                      ↓
             Weighted Matching Score
             (0.60×text + 0.25×skill + 0.10×exp + 0.05×edu)
                      |
                      ↓
              Job Ranking (sorted descending)
                      |
                      ↓
            Recommendation Engine (Top 5)
                      |
                      ↓
        Explainability + Skill Gap Analysis
                      |
                      ↓
             Streamlit Dashboard
```

---

## 🚀 Installation & Setup

### 1. Clone the repository
```bash
git clone <repository-url>
cd AIResumeScreening
```

### 2. Create a virtual environment
```bash
python -m venv venv
```

**Windows:**
```bash
venv\Scripts\activate
```

**Linux / macOS:**
```bash
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Pre-train the classifier
```bash
python training/train_model.py
```

### 5. Run the application
```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📖 Usage

1. **Navigate** to `Resume Analysis` in the sidebar.
2. **Upload** your resume in PDF or DOCX format.
3. Click **Analyze Resume** — the system processes your resume in seconds.
4. View **extracted skills**, education, and experience summary.
5. Navigate to **Job Recommendations** to see your top 5 matches.
6. **Select any job** to view detailed skill gap, score breakdown, and explanation.
7. Navigate to **Model Evaluation** to train the classifier and see its metrics.

---

## 📊 Matching Algorithm

```
1. Text Preprocessing
   → lowercase, remove stopwords, protect tech terms (C++, Node.js, etc.)

2. TF-IDF Vectorization
   → Convert resume + all job descriptions into numerical vectors
   → ngram_range=(1,2), max_features=8000, sublinear_tf=True

3. Cosine Similarity
   → Measure angle between resume vector and each job vector

4. Multi-factor Weighted Score
   Final Score = 0.60 × Text Similarity
               + 0.25 × Skill Match Percentage
               + 0.10 × Experience Match
               + 0.05 × Education Match

5. Job Ranking
   → Sort all jobs by Final Score (descending)
   → Return Top 5 recommendations
```

---

## 📂 Dataset

`data/jobs.csv` contains **35 sample job records** across categories:
- Software Engineering (Python, Java, Full Stack, Frontend, Backend, Android, iOS)
- Data Science & AI (ML Engineer, Data Scientist, NLP Engineer, AI Research)
- Cloud & DevOps (AWS, Kubernetes, SRE, Terraform)
- Security, Business Analysis, and more

> ⚠️ This is a **sample dataset for demonstration only**. Production systems should use real, verified job postings.

---

## ⚠️ Limitations

- Resume formatting differences may affect text extraction quality
- Skill synonyms may not always be captured (e.g., "sklearn" vs "Scikit-learn")
- Limited to 35 sample job descriptions — real deployment needs thousands
- NLP extraction may miss nested or image-embedded content
- **Match scores do NOT guarantee job interviews or offers**
- Classifier accuracy is limited by small training dataset size
- Possible bias from sample data composition

---

## 🚀 Future Scope

- LLM-based resume analysis (GPT-4o / Gemini)
- AI interview question generation
- LinkedIn job portal API integration
- Multilingual resume support
- Personalized career roadmap recommendations
- Resume improvement suggestions
- Skill gap learning path integration
- Real-time job board scraping

---

## 📜 Disclaimer

> This system provides automated resume analysis and job recommendations **for assistance only**. It should **not** be used as the sole basis for hiring or employment decisions. All match scores reflect text/skill similarity and do not predict hiring outcomes.

---

## 📄 License

MIT License — free for educational and personal use.
