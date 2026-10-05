"""
Candidate Career & Job Recommendation PDF Report Generator
===========================================================
Generates an executive, professional PDF evaluation report for internship & viva presentation.
Features:
- Candidate profile summary & contact information
- ML-predicted job category & classification confidence
- Categorized skill breakdown
- Ranked job recommendations table
- Skill gap analysis & actionable career tips
"""

import io
from datetime import datetime
from typing import Any, Dict, List

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT


def generate_candidate_pdf_report(
    parse_result: Dict[str, Any],
    skills_by_cat: Dict[str, List[str]],
    matched_jobs: List[Dict[str, Any]],
    prediction_result: Dict[str, Any] = None,
    recommender_obj = None
) -> bytes:
    """
    Generates a multi-page executive PDF report and returns the raw bytes.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Custom Palette
    COLOR_PRIMARY   = colors.HexColor("#312E81") # Deep Indigo
    COLOR_ACCENT    = colors.HexColor("#4F46E5") # Vibrant Indigo
    COLOR_DARK      = colors.HexColor("#0F172A") # Slate 900
    COLOR_MUTED     = colors.HexColor("#64748B") # Slate 500
    COLOR_BG_LIGHT  = colors.HexColor("#F8FAFC") # Slate 50
    COLOR_SUCCESS   = colors.HexColor("#065F46") # Emerald 800
    COLOR_DANGER    = colors.HexColor("#991B1B") # Ruby 800

    # Custom Typography Styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=COLOR_PRIMARY,
        alignment=TA_LEFT
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=13,
        textColor=COLOR_MUTED,
        alignment=TA_LEFT
    )
    section_title = ParagraphStyle(
        "SectionTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=COLOR_ACCENT,
        spaceBefore=8,
        spaceAfter=4
    )
    body_bold = ParagraphStyle(
        "BodyBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=12,
        textColor=COLOR_DARK
    )
    body_style = ParagraphStyle(
        "BodyRegular",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11.5,
        textColor=COLOR_DARK
    )
    table_cell = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        textColor=COLOR_DARK
    )
    table_cell_bold = ParagraphStyle(
        "TableCellBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=COLOR_DARK
    )

    story = []

    # 1. Header Banner
    header_table_data = [
        [
            Paragraph("<b>AI RESUME SCREENING & JOB RECOMMENDATION REPORT</b>", title_style),
            Paragraph(f"<b>Date:</b> {datetime.now().strftime('%d %b %Y')}<br/><b>Report ID:</b> AIML-{int(datetime.now().timestamp()) % 100000}", subtitle_style)
        ],
        [
            Paragraph("Fixton AIML Internship Capstone Project | Automated Talent Intelligence", subtitle_style),
            ""
        ]
    ]
    header_table = Table(header_table_data, colWidths=[380, 160])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
    ]))
    story.append(header_table)
    story.append(HRFlowable(width="100%", thickness=1.5, color=COLOR_ACCENT, spaceBefore=4, spaceAfter=8))

    # 2. Candidate Overview & Contact Grid
    story.append(Paragraph("1. Candidate Overview & Extracted Profile", section_title))
    
    cand_name = parse_result.get("candidate_name", "Candidate")
    cand_email = parse_result.get("email", "N/A")
    cand_phone = parse_result.get("phone", "N/A")
    cand_loc = parse_result.get("location", "N/A")
    edu_deg = parse_result.get("education", {}).get("degree_summary", "N/A")
    edu_branch = parse_result.get("education", {}).get("branch", "N/A")
    edu_year = parse_result.get("education", {}).get("graduation_year", "N/A")
    exp_years = f"{parse_result.get('experience_years', 0.0):.1f} Years"
    proj_cnt = str(parse_result.get("project_count", 0))

    overview_data = [
        [
            Paragraph("<b>Candidate Name:</b>", body_bold), Paragraph(cand_name, body_style),
            Paragraph("<b>Experience:</b>", body_bold), Paragraph(exp_years, body_style)
        ],
        [
            Paragraph("<b>Email:</b>", body_bold), Paragraph(cand_email, body_style),
            Paragraph("<b>Degree:</b>", body_bold), Paragraph(edu_deg, body_style)
        ],
        [
            Paragraph("<b>Phone:</b>", body_bold), Paragraph(cand_phone, body_style),
            Paragraph("<b>Branch:</b>", body_bold), Paragraph(edu_branch, body_style)
        ],
        [
            Paragraph("<b>Location:</b>", body_bold), Paragraph(cand_loc, body_style),
            Paragraph("<b>Grad Year:</b>", body_bold), Paragraph(str(edu_year), body_style)
        ]
    ]

    overview_table = Table(overview_data, colWidths=[95, 175, 85, 185])
    overview_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), COLOR_BG_LIGHT),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(overview_table)
    story.append(Spacer(1, 8))

    # 3. ML Domain Classification Box
    if prediction_result and prediction_result.get("success"):
        pred_cat = prediction_result.get("predicted_category", "N/A")
        conf = prediction_result.get("confidence", 0.0)
        ml_data = [[
            Paragraph("<b>ML Domain Prediction:</b>", body_bold),
            Paragraph(f"<b>{pred_cat}</b> (Classification Confidence: <b>{conf:.1f}%</b>)", body_style),
            Paragraph("<b>Algorithm:</b>", body_bold),
            Paragraph("Logistic Regression + TF-IDF (88.9% Test Acc)", body_style)
        ]]
        ml_table = Table(ml_data, colWidths=[120, 180, 65, 175])
        ml_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#EEF2FF")),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#C7D2FE")),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(ml_table)
        story.append(Spacer(1, 8))

    # 4. Detected Skills Inventory
    story.append(Paragraph("2. Detected Skills Inventory (Extracted via NLP)", section_title))
    skill_rows = []
    if skills_by_cat:
        for cat, s_list in skills_by_cat.items():
            skill_rows.append([
                Paragraph(f"<b>{cat}</b>", body_bold),
                Paragraph(", ".join(s_list), body_style)
            ])
    else:
        skill_rows.append([Paragraph("Skills", body_bold), Paragraph("No explicit skills detected", body_style)])

    skill_table = Table(skill_rows, colWidths=[140, 400])
    skill_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), COLOR_BG_LIGHT),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(skill_table)
    story.append(Spacer(1, 10))

    # 5. Top Job Recommendations Table
    story.append(Paragraph("3. Top Job Recommendations & Multi-Factor Match Scores", section_title))
    top_jobs = matched_jobs[:5] if matched_jobs else []

    jobs_table_data = [[
        Paragraph("<b>Rank</b>", table_cell_bold),
        Paragraph("<b>Role & Company</b>", table_cell_bold),
        Paragraph("<b>Domain</b>", table_cell_bold),
        Paragraph("<b>Text Sim (60%)</b>", table_cell_bold),
        Paragraph("<b>Skill Match (25%)</b>", table_cell_bold),
        Paragraph("<b>Overall Score</b>", table_cell_bold)
    ]]

    for i, job in enumerate(top_jobs, start=1):
        score_val = job.get("match_score", 0.0)
        jobs_table_data.append([
            Paragraph(f"#{i}", table_cell),
            Paragraph(f"<b>{job.get('job_title', 'Unknown')}</b><br/><font color='#64748B'>{job.get('company', '')} · {job.get('location', '')}</font>", table_cell),
            Paragraph(job.get("category", "General"), table_cell),
            Paragraph(f"{job.get('text_similarity', 0):.1f}%", table_cell),
            Paragraph(f"{job.get('skill_score', 0):.1f}%", table_cell),
            Paragraph(f"<b>{score_val:.1f}%</b>", table_cell_bold)
        ])

    jobs_table = Table(jobs_table_data, colWidths=[35, 185, 110, 75, 75, 60])
    jobs_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#E0E7FF")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(jobs_table)
    story.append(Spacer(1, 10))

    # 6. Deep-Dive Skill Gap & Career Advice for Top Match
    if top_jobs and recommender_obj:
        flat_skills = [s for sub in skills_by_cat.values() for s in sub]
        top_1 = top_jobs[0]
        exp = recommender_obj.generate_explanation(
            top_1, flat_skills, parse_result.get("experience_years", 0.0)
        )

        gap_story = []
        gap_story.append(Paragraph(f"4. Actionable Skill Gap & Recommendations: <b>{top_1.get('job_title')}</b>", section_title))

        matched_str = ", ".join(exp.get("matched_skills", [])) or "None identified"
        missing_str = ", ".join(exp.get("missing_skills", [])) or "None! All requirements satisfied"

        gap_table_data = [
            [
                Paragraph("<b>Matched Skills:</b>", body_bold),
                Paragraph(f"<font color='#065F46'>{matched_str}</font>", body_style)
            ],
            [
                Paragraph("<b>Missing Skills (To Learn):</b>", body_bold),
                Paragraph(f"<font color='#991B1B'>{missing_str}</font>", body_style)
            ]
        ]

        if exp.get("improvements"):
            tips_formatted = "<br/>".join(f"• {t}" for t in exp["improvements"])
            gap_table_data.append([
                Paragraph("<b>Action Steps:</b>", body_bold),
                Paragraph(tips_formatted, body_style)
            ])

        gap_table = Table(gap_table_data, colWidths=[140, 400])
        gap_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), COLOR_BG_LIGHT),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ]))
        gap_story.append(gap_table)
        story.append(KeepTogether(gap_story))

    # 7. Footer & Disclaimer
    story.append(Spacer(1, 12))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#94A3B8"), spaceBefore=4, spaceAfter=6))
    footer_text = Paragraph(
        "<b>Disclaimer:</b> This AI Resume Screening & Recommendation Report is generated by machine learning algorithms for evaluation and career guidance. "
        "Scores reflect natural language cosine similarity and heuristic weighting. Developed for Fixton Internship.",
        ParagraphStyle("Footer", parent=styles["Normal"], fontName="Helvetica", fontSize=7.5, leading=9.5, textColor=COLOR_MUTED, alignment=TA_CENTER)
    )
    story.append(footer_text)

    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
