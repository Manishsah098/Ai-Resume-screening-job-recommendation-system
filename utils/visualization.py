"""
Visualization Module.
Generates Plotly charts for match scores, skill distribution, and skill gap analysis.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Dict, List, Any


BRAND_COLORS = {
    "primary":   "#6366F1",
    "secondary": "#A855F7",
    "success":   "#10B981",
    "warning":   "#F59E0B",
    "danger":    "#EF4444",
    "bg":        "#0F172A",
    "card":      "#1E293B",
    "text":      "#F8FAFC",
    "text_muted":"#94A3B8",
    "grid":      "#334155"
}

CHART_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=BRAND_COLORS["text"], family="Inter, system-ui, sans-serif"),
    margin=dict(l=20, r=20, t=50, b=20)
)


def create_match_score_chart(top_jobs: List[Dict[str, Any]]) -> go.Figure:
    """
    Horizontal bar chart showing match scores for top recommended jobs.
    """
    if not top_jobs:
        return go.Figure()

    titles = [f"{j.get('job_title', 'Unknown')}" for j in top_jobs]
    scores = [j.get("match_score", 0) for j in top_jobs]
    companies = [j.get("company", "") for j in top_jobs]

    colors = []
    for s in scores:
        if s >= 80:
            colors.append(BRAND_COLORS["success"])
        elif s >= 60:
            colors.append(BRAND_COLORS["primary"])
        elif s >= 40:
            colors.append(BRAND_COLORS["warning"])
        else:
            colors.append(BRAND_COLORS["danger"])

    fig = go.Figure(go.Bar(
        x=scores,
        y=[f"{t}<br><sub>{c}</sub>" for t, c in zip(titles, companies)],
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{s:.1f}%" for s in scores],
        textposition="outside",
        textfont=dict(color=BRAND_COLORS["text"], size=12),
        hovertemplate="<b>%{y}</b><br>Match Score: %{x:.1f}%<extra></extra>"
    ))

    fig.update_layout(
        title=dict(text="Job Match Scores", font=dict(size=16, color=BRAND_COLORS["text"], weight="bold")),
        xaxis=dict(
            title="Match Score (%)", range=[0, 115],
            gridcolor=BRAND_COLORS["grid"], color=BRAND_COLORS["text"]
        ),
        yaxis=dict(autorange="reversed", gridcolor=BRAND_COLORS["grid"], color=BRAND_COLORS["text"]),
        height=max(300, len(top_jobs) * 70),
        **CHART_THEME
    )
    return fig


def create_skill_distribution_chart(skill_dist: Dict[str, int]) -> go.Figure:
    """
    Donut/Pie chart showing detected skill distribution by category.
    """
    if not skill_dist:
        return go.Figure()

    labels = list(skill_dist.keys())
    values = list(skill_dist.values())

    palette = [
        BRAND_COLORS["primary"], BRAND_COLORS["secondary"], BRAND_COLORS["success"],
        BRAND_COLORS["warning"], "#FB923C", "#F472B6", "#2DD4BF", "#A78BFA"
    ]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.6,
        marker=dict(colors=palette[:len(labels)], line=dict(color=BRAND_COLORS["card"], width=2)),
        textinfo="label+percent",
        textfont=dict(size=12, color=BRAND_COLORS["text"]),
        hovertemplate="<b>%{label}</b><br>Skills: %{value}<br>%{percent}<extra></extra>"
    ))

    fig.add_annotation(
        text=f"<b style='font-size:20px;'>{sum(values)}</b><br><span style='font-size:12px;color:#94A3B8;'>Skills</span>",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=16, color=BRAND_COLORS["text"])
    )

    fig.update_layout(
        title=dict(text="Skill Distribution by Category", font=dict(size=16, color=BRAND_COLORS["text"], weight="bold")),
        showlegend=True,
        legend=dict(font=dict(color=BRAND_COLORS["text"]), bgcolor="rgba(0,0,0,0)"),
        height=380,
        **CHART_THEME
    )
    return fig


def create_skill_gap_chart(matched_skills: List[str], missing_skills: List[str]) -> go.Figure:
    """
    Grouped bar chart comparing matched vs. missing skills for a job.
    """
    categories = ["Matched Skills", "Missing Skills"]
    values = [len(matched_skills), len(missing_skills)]
    colors = [BRAND_COLORS["success"], BRAND_COLORS["danger"]]

    fig = go.Figure(go.Bar(
        x=categories,
        y=values,
        marker=dict(color=colors, line=dict(width=0), opacity=0.9),
        text=values,
        textposition="outside",
        textfont=dict(size=14, color=BRAND_COLORS["text"], weight="bold"),
        hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>"
    ))

    total = len(matched_skills) + len(missing_skills)
    pct = (len(matched_skills) / total * 100) if total > 0 else 0

    fig.update_layout(
        title=dict(
            text=f"Skill Coverage: {pct:.0f}% ({len(matched_skills)}/{total} required skills)",
            font=dict(size=15, color=BRAND_COLORS["text"], weight="bold")
        ),
        xaxis=dict(gridcolor=BRAND_COLORS["grid"], color=BRAND_COLORS["text"]),
        yaxis=dict(title="Number of Skills", gridcolor=BRAND_COLORS["grid"], color=BRAND_COLORS["text"]),
        height=300,
        **CHART_THEME
    )
    return fig


def create_score_breakdown_chart(job: Dict[str, Any]) -> go.Figure:
    """
    Radar / spider chart showing multi-factor score breakdown.
    """
    categories = ["Text Similarity", "Skill Match", "Experience", "Education"]
    values = [
        job.get("text_similarity", 0),
        job.get("skill_score", 0),
        job.get("experience_score", 0),
        job.get("education_score", 0)
    ]

    fig = go.Figure(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill="toself",
        fillcolor="rgba(99, 102, 241, 0.3)",
        line=dict(color=BRAND_COLORS["primary"], width=2.5),
        marker=dict(color=BRAND_COLORS["primary"], size=8),
        hovertemplate="<b>%{theta}</b><br>Score: %{r:.1f}%<extra></extra>"
    ))

    fig.update_layout(
        polar=dict(
            bgcolor="rgba(30, 41, 59, 0.6)",
            radialaxis=dict(
                visible=True, range=[0, 100],
                gridcolor=BRAND_COLORS["grid"], color=BRAND_COLORS["text_muted"]
            ),
            angularaxis=dict(gridcolor=BRAND_COLORS["grid"], color=BRAND_COLORS["text"])
        ),
        title=dict(text="Score Breakdown (4 Factors)", font=dict(size=15, color=BRAND_COLORS["text"], weight="bold")),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=BRAND_COLORS["text"], family="Inter, system-ui, sans-serif"),
        height=340,
        margin=dict(l=40, r=40, t=50, b=30)
    )
    return fig


def create_classifier_confidence_chart(probabilities: Dict[str, float]) -> go.Figure:
    """Bar chart showing classifier confidence per category."""
    if not probabilities:
        return go.Figure()

    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:6]
    cats = [item[0] for item in sorted_probs]
    probs = [item[1] for item in sorted_probs]

    fig = go.Figure(go.Bar(
        x=probs,
        y=cats,
        orientation="h",
        marker=dict(
            color=probs,
            colorscale=[[0, BRAND_COLORS["danger"]], [0.5, BRAND_COLORS["warning"]], [1, BRAND_COLORS["success"]]],
            showscale=False,
            line=dict(width=0)
        ),
        text=[f"{p:.1f}%" for p in probs],
        textposition="outside",
        textfont=dict(color=BRAND_COLORS["text"], weight="bold"),
        hovertemplate="<b>%{y}</b><br>Confidence: %{x:.1f}%<extra></extra>"
    ))

    fig.update_layout(
        title=dict(text="Category Confidence Scores", font=dict(size=15, color=BRAND_COLORS["text"], weight="bold")),
        xaxis=dict(title="Confidence (%)", range=[0, 120], gridcolor=BRAND_COLORS["grid"], color=BRAND_COLORS["text"]),
        yaxis=dict(autorange="reversed", color=BRAND_COLORS["text"]),
        height=300,
        **CHART_THEME
    )
    return fig
