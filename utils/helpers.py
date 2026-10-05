"""
Helper Utilities.
"""

import os
import re
import pandas as pd
from typing import Any, Dict, List, Optional


def load_jobs_dataframe(csv_path: str) -> pd.DataFrame:
    """Load the jobs CSV, handling missing/malformed files gracefully."""
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"Job dataset not found at: '{csv_path}'. "
            "Please ensure 'data/jobs.csv' exists in the project directory."
        )
    try:
        df = pd.read_csv(csv_path)
        required_cols = ["job_id", "job_title", "required_skills", "description"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"jobs.csv is missing required columns: {missing}")
        df = df.dropna(subset=["job_title", "description"])
        df = df.reset_index(drop=True)
        return df
    except Exception as e:
        raise RuntimeError(f"Error loading job dataset: {str(e)}")


def format_match_score(score: float) -> str:
    """Format a match score as a percentage string."""
    return f"{score:.1f}%"


def get_score_color(score: float) -> str:
    """Return a color category string based on score threshold."""
    if score >= 80:
        return "green"
    elif score >= 60:
        return "blue"
    elif score >= 40:
        return "orange"
    else:
        return "red"


def get_score_label(score: float) -> str:
    """Return a human-readable label for a match score."""
    if score >= 80:
        return "Excellent Match"
    elif score >= 65:
        return "Good Match"
    elif score >= 45:
        return "Moderate Match"
    else:
        return "Partial Match"


def safe_get(d: Dict, key: str, default: Any = "N/A") -> Any:
    """Safely get a value from a dict with a default."""
    val = d.get(key, default)
    return val if val is not None else default


def truncate_text(text: str, max_length: int = 300) -> str:
    """Truncate text to max_length characters, adding ellipsis if needed."""
    if not text:
        return ""
    return text[:max_length].rsplit(" ", 1)[0] + "..." if len(text) > max_length else text


def skills_to_tags_html(skills: List[str]) -> str:
    """Convert a list of skills to HTML tag badges (for rendering in st.markdown)."""
    tags = " ".join(
        f'<span style="background:#1E3A5F;color:#7EC8E3;padding:4px 10px;'
        f'border-radius:12px;margin:3px;display:inline-block;font-size:0.85em;">'
        f'{skill}</span>'
        for skill in skills
    )
    return tags
