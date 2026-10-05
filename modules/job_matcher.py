"""
Job Matcher Module.
Implements TF-IDF vectorization + Cosine Similarity for resume-to-job matching,
combined with weighted scoring across text, skill, experience, and education dimensions.
"""

import re
import os
import joblib
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from modules.text_preprocessor import TextPreprocessor

# Weighting configuration for final score
WEIGHTS = {
    "text_similarity": 0.60,
    "skill_match":     0.25,
    "experience_match": 0.10,
    "education_match":  0.05,
}

# Education seniority mapping for scoring
EDU_SENIORITY = {
    "phd": 5, "ph.d": 5,
    "m.tech": 4, "m.e.": 4, "mtech": 4, "me": 4, "m.sc": 4, "msc": 4, "mba": 4, "mca": 4,
    "b.tech": 3, "b.e.": 3, "btech": 3, "be": 3, "bca": 3, "b.sc": 3, "bsc": 3, "bba": 3,
    "diploma": 2, "hsc": 1, "12th": 1
}


class JobMatcher:
    """
    Matches a resume against a job description dataset using TF-IDF + weighted cosine similarity.
    """

    def __init__(
        self,
        preprocessor: TextPreprocessor = None,
        vectorizer_path: str = None
    ):
        self.preprocessor = preprocessor or TextPreprocessor()
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.vectorizer_path = vectorizer_path
        self._load_or_create_vectorizer()

    def _load_or_create_vectorizer(self):
        """Load persisted vectorizer or create a fresh one."""
        if self.vectorizer_path and os.path.isfile(self.vectorizer_path):
            try:
                self.vectorizer = joblib.load(self.vectorizer_path)
                return
            except Exception:
                pass
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=8000,
            sublinear_tf=True,
            min_df=1,
            max_df=0.95,
            analyzer="word"
        )

    def _build_job_corpus(self, jobs_df: pd.DataFrame) -> List[str]:
        """Combine job title, description, and required skills into one text per job."""
        corpus = []
        for _, row in jobs_df.iterrows():
            parts = []
            if "job_title" in row and pd.notna(row["job_title"]):
                parts.append(str(row["job_title"]))
            if "description" in row and pd.notna(row["description"]):
                parts.append(str(row["description"]))
            if "required_skills" in row and pd.notna(row["required_skills"]):
                # Repeat skills to give them higher weight
                skills_text = str(row["required_skills"]).replace(",", " ")
                parts.append(skills_text)
                parts.append(skills_text)
            if "category" in row and pd.notna(row["category"]):
                parts.append(str(row["category"]))
            combined = " ".join(parts)
            corpus.append(self.preprocessor.clean_text(combined))
        return corpus

    def fit_vectorizer(self, jobs_df: pd.DataFrame, resume_text: str = "") -> None:
        """Fit TF-IDF on job corpus + resume text, then optionally save it."""
        corpus = self._build_job_corpus(jobs_df)
        cleaned_resume = self.preprocessor.clean_text(resume_text)
        all_texts = corpus + [cleaned_resume]
        self.vectorizer.fit(all_texts)

        if self.vectorizer_path:
            os.makedirs(os.path.dirname(self.vectorizer_path), exist_ok=True)
            try:
                joblib.dump(self.vectorizer, self.vectorizer_path)
            except Exception:
                pass

    def compute_text_similarity(
        self, resume_text: str, jobs_df: pd.DataFrame
    ) -> np.ndarray:
        """
        Fit vectorizer on the corpus + resume, then compute cosine similarity
        of resume against all job descriptions.
        """
        self.fit_vectorizer(jobs_df, resume_text)
        corpus = self._build_job_corpus(jobs_df)
        cleaned_resume = self.preprocessor.clean_text(resume_text)

        # Transform
        job_vectors = self.vectorizer.transform(corpus)
        resume_vector = self.vectorizer.transform([cleaned_resume])
        sims = cosine_similarity(resume_vector, job_vectors)[0]
        return sims

    def compute_skill_score(
        self, resume_skills: List[str], required_skills_str: str
    ) -> float:
        """Compute skill overlap score as 0–1."""
        if not required_skills_str:
            return 0.5
        required = [s.strip().lower() for s in required_skills_str.split(",") if s.strip()]
        if not required:
            return 0.5
        resume_lower = {s.lower() for s in resume_skills}
        matches = sum(
            1 for req in required
            if req in resume_lower or any(req in rs or rs in req for rs in resume_lower)
        )
        return matches / len(required)

    def compute_experience_score(
        self, candidate_years: float, job_min_years: Any
    ) -> float:
        """Calculate experience match score (0–1)."""
        try:
            required_min = float(job_min_years)
        except (TypeError, ValueError):
            return 0.7  # Assume average match if not specified

        if candidate_years >= required_min:
            # Slight bonus for more experience, capped
            bonus = min((candidate_years - required_min) / 5.0, 0.2)
            return min(1.0 + bonus, 1.0)
        else:
            gap = required_min - candidate_years
            penalty = gap / max(required_min, 1)
            return max(0.0, 1.0 - penalty)

    def compute_education_score(
        self, candidate_degrees: List[str], job_education_req: str
    ) -> float:
        """Calculate education match score (0–1)."""
        if not job_education_req:
            return 0.7

        candidate_max_level = 0
        for deg in candidate_degrees:
            for key, level in EDU_SENIORITY.items():
                if key in deg.lower():
                    candidate_max_level = max(candidate_max_level, level)
                    break

        job_max_level = 0
        for key, level in EDU_SENIORITY.items():
            if key in job_education_req.lower():
                job_max_level = max(job_max_level, level)

        if job_max_level == 0:
            return 0.7  # Unspecified requirement

        if candidate_max_level >= job_max_level:
            return 1.0
        elif candidate_max_level == job_max_level - 1:
            return 0.7
        elif candidate_max_level == job_max_level - 2:
            return 0.4
        else:
            return 0.2

    def compute_weighted_score(
        self,
        text_sim: float,
        skill_score: float,
        experience_score: float,
        education_score: float
    ) -> float:
        """Compute the final weighted match score (0–100)."""
        raw = (
            WEIGHTS["text_similarity"] * text_sim +
            WEIGHTS["skill_match"] * skill_score +
            WEIGHTS["experience_match"] * experience_score +
            WEIGHTS["education_match"] * education_score
        )
        return round(min(raw * 100, 100.0), 2)

    def match_resume_to_jobs(
        self,
        resume_text: str,
        resume_skills: List[str],
        candidate_experience_years: float,
        candidate_degrees: List[str],
        jobs_df: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """
        Core matching pipeline. Returns list of jobs with scores, sorted descending.
        """
        if jobs_df.empty:
            return []

        # Compute text similarities
        text_sims = self.compute_text_similarity(resume_text, jobs_df)

        results = []
        for idx, row in jobs_df.iterrows():
            t_sim = float(text_sims[idx if idx < len(text_sims) else -1])

            skill_score = self.compute_skill_score(
                resume_skills,
                str(row.get("required_skills", ""))
            )
            exp_score = self.compute_experience_score(
                candidate_experience_years,
                row.get("experience_years_min", 0)
            )
            edu_score = self.compute_education_score(
                candidate_degrees,
                str(row.get("education_required", ""))
            )
            final_score = self.compute_weighted_score(t_sim, skill_score, exp_score, edu_score)

            results.append({
                "job_id": row.get("job_id", f"JOB{idx:03d}"),
                "job_title": str(row.get("job_title", "Unknown Role")),
                "company": str(row.get("company", "Company")),
                "location": str(row.get("location", "N/A")),
                "category": str(row.get("category", "N/A")),
                "experience_level": str(row.get("experience_level", "N/A")),
                "education_required": str(row.get("education_required", "N/A")),
                "required_skills": str(row.get("required_skills", "")),
                "description": str(row.get("description", "")),
                "match_score": final_score,
                "text_similarity": round(t_sim * 100, 2),
                "skill_score": round(skill_score * 100, 2),
                "experience_score": round(exp_score * 100, 2),
                "education_score": round(edu_score * 100, 2),
            })

        # Sort by match_score descending
        results.sort(key=lambda x: x["match_score"], reverse=True)
        return results
