"""
Skill Extractor Module.
Matches resume text against a structured skill dictionary using multi-strategy matching:
  1. Exact phrase matching
  2. Fuzzy partial matching for common abbreviations
"""

import re
import os
import pandas as pd
from typing import Dict, List, Set, Tuple

# Default skill list (used as fallback if skills.csv is not found)
DEFAULT_SKILLS: Dict[str, List[str]] = {
    "Programming Languages": [
        "Python", "Java", "C++", "C#", "C", "JavaScript", "TypeScript",
        "Ruby", "Go", "Rust", "PHP", "Swift", "Kotlin", "R", "Scala", "MATLAB"
    ],
    "Web Development": [
        "HTML", "CSS", "React", "Angular", "Vue.js", "Node.js", "Express.js",
        "Django", "Flask", "FastAPI", "Spring Boot", "ASP.NET", "Next.js",
        "Tailwind CSS", "Bootstrap", "jQuery", "GraphQL", "REST API",
        "WordPress", "Responsive Design"
    ],
    "Databases": [
        "SQL", "MySQL", "PostgreSQL", "MongoDB", "Redis", "Oracle", "SQLite",
        "Cassandra", "DynamoDB", "MariaDB", "Firebase", "Elasticsearch"
    ],
    "Machine Learning & AI": [
        "Machine Learning", "Deep Learning", "NLP", "Natural Language Processing",
        "Computer Vision", "Scikit-learn", "TensorFlow", "PyTorch", "Keras",
        "OpenCV", "Hugging Face", "LLM", "Generative AI", "XGBoost",
        "LightGBM", "Random Forest", "Regression", "Classification", "Clustering"
    ],
    "Data Science & Analytics": [
        "Pandas", "NumPy", "Matplotlib", "Seaborn", "Plotly", "Tableau",
        "Power BI", "Excel", "Data Analysis", "Data Visualization",
        "Data Mining", "Statistics", "Apache Spark", "PySpark", "Hadoop"
    ],
    "Cloud & DevOps": [
        "AWS", "Azure", "Google Cloud", "Docker", "Kubernetes", "CI/CD",
        "Git", "GitHub", "GitLab", "Jenkins", "Linux", "Terraform",
        "Ansible", "Nginx", "Microservices", "Bash", "PowerShell"
    ],
    "Tools & Frameworks": [
        "Postman", "JIRA", "Agile", "Scrum", "VS Code", "Figma",
        "Jupyter", "Selenium", "PyTest", "Unit Testing"
    ],
    "Soft Skills": [
        "Problem Solving", "Communication", "Teamwork", "Leadership",
        "Critical Thinking", "Time Management", "Adaptability", "Collaboration"
    ]
}

# Abbreviation map for common skills
SKILL_ABBREVIATIONS = {
    "ml": "Machine Learning",
    "dl": "Deep Learning",
    "nlp": "Natural Language Processing",
    "cv": "Computer Vision",
    "ai": "Artificial Intelligence",
    "js": "JavaScript",
    "ts": "TypeScript",
    "ts": "TypeScript",
    "py": "Python",
    "sql": "SQL",
    "nosql": "MongoDB",
    "tf": "TensorFlow",
    "k8s": "Kubernetes",
    "gcp": "Google Cloud",
    "bi": "Power BI",
    "vba": "Excel",
    "oop": "Object Oriented Programming",
    "dsa": "Data Structures",
    "api": "REST API"
}


class SkillExtractor:
    """
    Extracts and categorizes skills from resume text.
    """

    def __init__(self, skills_csv_path: str = None):
        self.skill_dict = {}
        self.all_skills_flat: List[str] = []
        self._load_skills(skills_csv_path)

    def _load_skills(self, skills_csv_path: str = None):
        """Load skills from CSV or use built-in defaults."""
        loaded_from_csv = False

        if skills_csv_path and os.path.isfile(skills_csv_path):
            try:
                df = pd.read_csv(skills_csv_path)
                if "skill" in df.columns and "category" in df.columns:
                    for category, group in df.groupby("category"):
                        self.skill_dict[category] = group["skill"].tolist()
                    loaded_from_csv = True
            except Exception:
                pass

        if not loaded_from_csv:
            self.skill_dict = DEFAULT_SKILLS

        # Build flat list preserving case
        for skills in self.skill_dict.values():
            self.all_skills_flat.extend(skills)

        # Deduplicate maintaining order
        seen = set()
        unique = []
        for s in self.all_skills_flat:
            if s.lower() not in seen:
                seen.add(s.lower())
                unique.append(s)
        self.all_skills_flat = unique

    def extract_skills(self, text: str) -> Dict[str, List[str]]:
        """
        Extract matched skills from text and return them by category.
        """
        if not text:
            return {}

        text_lower = text.lower()
        matched_by_category: Dict[str, List[str]] = {}
        matched_set: Set[str] = set()

        for category, skills in self.skill_dict.items():
            matched_in_cat = []
            for skill in skills:
                skill_lower = skill.lower()
                # Use word-boundary safe matching
                pattern = r"\b" + re.escape(skill_lower) + r"\b"
                if re.search(pattern, text_lower):
                    if skill_lower not in matched_set:
                        matched_in_cat.append(skill)
                        matched_set.add(skill_lower)
            if matched_in_cat:
                matched_by_category[category] = matched_in_cat

        # Check abbreviations
        words = re.findall(r"\b\w+\b", text_lower)
        for word in words:
            if word in SKILL_ABBREVIATIONS:
                full_skill = SKILL_ABBREVIATIONS[word]
                if full_skill.lower() not in matched_set:
                    cat = self._find_skill_category(full_skill)
                    if cat:
                        matched_by_category.setdefault(cat, [])
                        if full_skill not in matched_by_category[cat]:
                            matched_by_category[cat].append(full_skill)
                        matched_set.add(full_skill.lower())

        return matched_by_category

    def _find_skill_category(self, skill: str) -> str:
        """Find which category a skill belongs to."""
        skill_lower = skill.lower()
        for category, skills in self.skill_dict.items():
            if any(s.lower() == skill_lower for s in skills):
                return category
        return "Programming Languages"

    def get_flat_skills(self, text: str) -> List[str]:
        """Return flat list of all detected skills."""
        by_cat = self.extract_skills(text)
        result = []
        for skills in by_cat.values():
            result.extend(skills)
        return result

    def get_skill_gap(self, resume_skills: List[str], job_skills_str: str) -> Tuple[List[str], List[str], float]:
        """
        Calculate matched vs missing skills and skill match percentage.
        
        Args:
            resume_skills: Flat list of detected skills from resume
            job_skills_str: Comma-separated required skills from job posting
            
        Returns:
            (matched_skills, missing_skills, match_percentage)
        """
        if not job_skills_str:
            return [], [], 100.0

        # Parse required skills from job description
        required_skills = [s.strip() for s in job_skills_str.split(",") if s.strip()]
        if not required_skills:
            return [], [], 100.0

        resume_skills_lower = {s.lower() for s in resume_skills}

        matched = []
        missing = []
        for req_skill in required_skills:
            req_lower = req_skill.lower()
            # Check exact or partial match
            if req_lower in resume_skills_lower or any(
                req_lower in rs or rs in req_lower
                for rs in resume_skills_lower
            ):
                matched.append(req_skill)
            else:
                missing.append(req_skill)

        total = len(required_skills)
        pct = (len(matched) / total * 100) if total > 0 else 0.0
        return matched, missing, round(pct, 2)

    def get_skill_category_distribution(self, skills: List[str]) -> Dict[str, int]:
        """Returns skill count per category for charting."""
        dist: Dict[str, int] = {}
        for cat, cat_skills in self.skill_dict.items():
            cat_lower = {s.lower() for s in cat_skills}
            count = sum(1 for s in skills if s.lower() in cat_lower)
            if count > 0:
                dist[cat] = count
        return dist
