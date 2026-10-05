"""
Recommender Module.
Generates explainable job recommendations with matched/missing skills
and human-readable reasoning for each result.
"""

from typing import Any, Dict, List, Tuple


class Recommender:
    """
    Post-processes matched job results to produce ranked recommendations
    with explainability features.
    """

    def get_top_recommendations(
        self,
        matched_jobs: List[Dict[str, Any]],
        top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """Return top N jobs sorted by match score."""
        return matched_jobs[:top_n]

    def generate_explanation(
        self,
        job: Dict[str, Any],
        resume_skills: List[str],
        candidate_experience: float
    ) -> Dict[str, Any]:
        """
        Generate a human-readable explanation of why this job was recommended.
        """
        reasons = []
        improvements = []

        score = job.get("match_score", 0)
        text_sim = job.get("text_similarity", 0)
        skill_score = job.get("skill_score", 0)
        exp_score = job.get("experience_score", 0)
        edu_score = job.get("education_score", 0)

        # --- Text similarity reasoning ---
        if text_sim >= 70:
            reasons.append(f"[Strong Match] Very high content similarity ({text_sim:.0f}%) with job description")
        elif text_sim >= 50:
            reasons.append(f"[Good Match] Good content similarity ({text_sim:.0f}%) with job description")
        elif text_sim >= 30:
            reasons.append(f"[Moderate Match] Moderate content similarity ({text_sim:.0f}%) with job description")
        else:
            improvements.append("Tailor your resume keywords to match this job description more closely")

        # --- Skill reasoning ---
        required_skills_str = job.get("required_skills", "")
        required_skills = [s.strip() for s in required_skills_str.split(",") if s.strip()]
        resume_lower = {s.lower() for s in resume_skills}

        matched_skills = []
        missing_skills = []
        for req_skill in required_skills:
            req_lower = req_skill.lower()
            if req_lower in resume_lower or any(
                req_lower in rs or rs in req_lower for rs in resume_lower
            ):
                matched_skills.append(req_skill)
            else:
                missing_skills.append(req_skill)

        for s in matched_skills[:4]:
            reasons.append(f"[Skill Match] '{s}' requirement satisfied")

        if missing_skills:
            for s in missing_skills[:3]:
                improvements.append(f"Learn '{s}' to strengthen this application")

        # --- Experience reasoning ---
        if exp_score >= 90:
            reasons.append(f"[Experience Match] Your experience ({candidate_experience:.1f} yrs) matches well")
        elif exp_score >= 60:
            reasons.append(f"[Experience Close] Your experience level is close to requirements")
        else:
            improvements.append("Gain more hands-on experience for this role")

        # --- Education reasoning ---
        if edu_score >= 90:
            reasons.append("[Education Match] Educational qualification meets job requirements")
        elif edu_score >= 60:
            reasons.append("[Education Partial] Education qualification partially meets requirements")

        # --- Overall assessment ---
        if score >= 80:
            match_label = "Excellent Match"
            match_color = "success"
        elif score >= 65:
            match_label = "Good Match"
            match_color = "info"
        elif score >= 45:
            match_label = "Moderate Match"
            match_color = "warning"
        else:
            match_label = "Partial Match"
            match_color = "error"

        return {
            "reasons": reasons,
            "improvements": improvements,
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "match_label": match_label,
            "match_color": match_color,
            "skill_coverage": f"{len(matched_skills)}/{len(required_skills)}" if required_skills else "N/A"
        }

    def rank_summary(self, top_jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Produce a rank-formatted summary list for display."""
        summary = []
        for rank, job in enumerate(top_jobs, start=1):
            summary.append({
                "rank": rank,
                "job_title": job.get("job_title", "Unknown"),
                "company": job.get("company", "Company"),
                "location": job.get("location", "N/A"),
                "match_score": job.get("match_score", 0),
                "category": job.get("category", "N/A"),
                "experience_level": job.get("experience_level", "N/A")
            })
        return summary


# Alias for compatibility
JobRecommender = Recommender
