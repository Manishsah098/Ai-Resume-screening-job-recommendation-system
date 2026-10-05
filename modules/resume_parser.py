"""
Resume Parser Module.
Extracts raw text and structured metadata (Personal Info, Education, Experience, Projects, Certifications)
from PDF and DOCX files with robust error handling.
"""

import io
import os
import re
from typing import Any, Dict, List, Optional, Tuple


class ResumeParser:
    """
    Parses PDF and DOCX resumes and extracts structured fields and plain text.
    """

    # Common degrees pattern
    DEGREE_PATTERNS = [
        r"\b(?:b\.?tech|b\.?e\.?|bachelor\s+of\s+(?:technology|engineering|science|arts|commerce|computer\s+applications))\b",
        r"\b(?:m\.?tech|m\.?e\.?|master\s+of\s+(?:technology|engineering|science|arts|business\s+administration|computer\s+applications))\b",
        r"\b(?:bca|mca|b\.?sc|m\.?sc|bba|mba|ph\.?d|diploma)\b",
        r"\b(?:bachelor'?s|master'?s|doctorate)\b"
    ]

    # Major disciplines
    DISCIPLINE_PATTERNS = [
        r"\b(?:computer\s+science(?:\s+and\s+engineering)?|information\s+technology|data\s+science|artificial\s+intelligence)\b",
        r"\b(?:electronics(?:\s+and\s+communication)?|electrical|mechanical|civil|robotics|software\s+engineering)\b",
        r"\b(?:mathematics|statistics|physics|computational\s+biology)\b"
    ]

    # College / University keywords
    INSTITUTE_PATTERNS = [
        r"([A-Z][a-zA-Z\s,.'-]{2,50}(?:University|Institute\s+of\s+Technology|College\s+of\s+Engineering|IIT|NIT|IIIT|BITS|Polytechnic|Academy))"
    ]

    # Certifications keywords
    CERT_KEYWORDS = [
        "certified", "certification", "aws certified", "azure certified", "gcp certified",
        "google cloud certified", "tensorflow developer", "pmp", "scrum master", "cisco",
        "ccna", "comptia", "oracle certified", "coursera", "udemy", "edx", "deeplearning.ai"
    ]

    @staticmethod
    def extract_text_from_pdf(file_bytes_or_path) -> Tuple[str, Optional[str]]:
        """
        Extract raw text from PDF file object or file path using pypdf.
        Returns: (extracted_text, error_message)
        """
        try:
            from pypdf import PdfReader
            if isinstance(file_bytes_or_path, (str, os.PathLike)):
                reader = PdfReader(file_bytes_or_path)
            elif hasattr(file_bytes_or_path, "read"):
                # Rewind pointer if possible
                if hasattr(file_bytes_or_path, "seek"):
                    file_bytes_or_path.seek(0)
                reader = PdfReader(file_bytes_or_path)
            elif isinstance(file_bytes_or_path, (bytes, bytearray)):
                reader = PdfReader(io.BytesIO(file_bytes_or_path))
            else:
                return "", "Invalid PDF input format."

            pages_text = []
            for i, page in enumerate(reader.pages):
                try:
                    text = page.extract_text()
                    if text:
                        pages_text.append(text)
                except Exception as pe:
                    continue

            full_text = "\n".join(pages_text).strip()
            if not full_text:
                return "", "PDF file contains no readable text. It may be scanned or image-only."
            return full_text, None

        except Exception as e:
            return "", f"Error reading PDF file: {str(e)}"

    @staticmethod
    def extract_text_from_docx(file_bytes_or_path) -> Tuple[str, Optional[str]]:
        """
        Extract raw text from DOCX file object or file path using python-docx.
        Returns: (extracted_text, error_message)
        """
        try:
            import docx
            if isinstance(file_bytes_or_path, (str, os.PathLike)):
                doc = docx.Document(file_bytes_or_path)
            elif hasattr(file_bytes_or_path, "read"):
                if hasattr(file_bytes_or_path, "seek"):
                    file_bytes_or_path.seek(0)
                doc = docx.Document(file_bytes_or_path)
            elif isinstance(file_bytes_or_path, (bytes, bytearray)):
                doc = docx.Document(io.BytesIO(file_bytes_or_path))
            else:
                return "", "Invalid DOCX input format."

            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            
            # Also extract text from tables if present
            for table in doc.tables:
                for row in table.rows:
                    row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                    if row_text:
                        paragraphs.append(row_text)

            full_text = "\n".join(paragraphs).strip()
            if not full_text:
                return "", "DOCX file contains no readable text or is empty."
            return full_text, None

        except Exception as e:
            return "", f"Error reading DOCX file: {str(e)}"

    @classmethod
    def extract_text(cls, file_obj, filename: str) -> Tuple[str, Optional[str]]:
        """
        Detect file type and extract plain text.
        """
        if not filename:
            return "", "File name not provided."
        
        ext = os.path.splitext(filename)[1].lower()
        if ext == ".pdf":
            return cls.extract_text_from_pdf(file_obj)
        elif ext in [".docx", ".doc"]:
            return cls.extract_text_from_docx(file_obj)
        elif ext == ".txt":
            try:
                if hasattr(file_obj, "read"):
                    if hasattr(file_obj, "seek"):
                        file_obj.seek(0)
                    content = file_obj.read()
                    if isinstance(content, bytes):
                        return content.decode("utf-8", errors="ignore"), None
                    return str(content), None
            except Exception as e:
                return "", f"Error reading TXT file: {str(e)}"
        else:
            return "", f"Unsupported file format '{ext}'. Please upload a PDF or DOCX resume."

    @classmethod
    def extract_email(cls, text: str) -> Optional[str]:
        """Extract primary email address from text."""
        match = re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", text)
        return match.group(0) if match else None

    @classmethod
    def extract_phone(cls, text: str) -> Optional[str]:
        """Extract phone number from text."""
        # Matches patterns like +91 9876543210, (123) 456-7890, 98765-43210, etc.
        patterns = [
            r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
            r"(?:\+?91[\-\s]?)?[6789]\d{9}",
            r"\b\d{10}\b"
        ]
        for pat in patterns:
            match = re.search(pat, text)
            if match:
                return match.group(0).strip()
        return None

    @classmethod
    def extract_name(cls, text: str) -> Optional[str]:
        """
        Extract likely candidate name from top lines of the resume.
        """
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        for line in lines[:6]:
            # Ignore headers or contact lines
            if any(term in line.lower() for term in ["resume", "curriculum vitae", "cv", "email", "phone", "http", "linkedin", "github"]):
                continue
            # Look for 2 to 4 word capitalized string
            clean_line = re.sub(r"[^a-zA-Z\s]", "", line).strip()
            words = clean_line.split()
            if 2 <= len(words) <= 4 and all(w[0].isupper() for w in words if w):
                return " ".join(words)
        # Fallback to first line if reasonably short
        if lines:
            first_line = re.sub(r"[^a-zA-Z\s]", "", lines[0]).strip()
            if 1 <= len(first_line.split()) <= 4:
                return first_line
        return "Candidate Profile"

    @classmethod
    def extract_location(cls, text: str) -> Optional[str]:
        """Extract candidate city or region."""
        known_cities = [
            "Bengaluru", "Bangalore", "Hyderabad", "Pune", "Mumbai", "Delhi", "Noida",
            "Gurugram", "Gurgaon", "Chennai", "Kolkata", "Ahmedabad", "Kochi", "San Francisco",
            "New York", "London", "Seattle", "Austin", "Remote", "India"
        ]
        for city in known_cities:
            if re.search(r"\b" + re.escape(city) + r"\b", text, re.IGNORECASE):
                return city
        return "Not Specified"

    @classmethod
    def extract_education(cls, text: str) -> Dict[str, Any]:
        """Extract Degree, Branch, Institute, and Graduation Year."""
        degrees_found = []
        for pat in cls.DEGREE_PATTERNS:
            matches = re.findall(pat, text, re.IGNORECASE)
            degrees_found.extend([m.strip().upper() for m in matches])

        disciplines_found = []
        for pat in cls.DISCIPLINE_PATTERNS:
            matches = re.findall(pat, text, re.IGNORECASE)
            disciplines_found.extend([m.strip().title() for m in matches])

        # Graduation year search (e.g., 2018 - 2022 or Graduated: 2024)
        year_match = re.findall(r"\b(19\d\d|20[0-2]\d)\b", text)
        grad_year = year_match[-1] if year_match else None

        # Clean degrees
        unique_degrees = list(dict.fromkeys(degrees_found))
        unique_disciplines = list(dict.fromkeys(disciplines_found))

        degree_str = ", ".join(unique_degrees[:2]) if unique_degrees else "Not Explicitly Mentioned"
        branch_str = ", ".join(unique_disciplines[:2]) if unique_disciplines else "Engineering / Technology"

        return {
            "degrees": unique_degrees,
            "degree_summary": degree_str,
            "branch": branch_str,
            "graduation_year": grad_year or "N/A"
        }

    @classmethod
    def extract_experience_years(cls, text: str) -> float:
        """Estimate years of experience mentioned in resume."""
        # Pattern like "3+ years of experience", "4.5 years", "2 years"
        exp_patterns = [
            r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)(?:\s+of)?\s*(?:experience|exp)?",
            r"(?:experience|exp):\s*(\d+(?:\.\d+)?)\s*(?:years?|yrs?)"
        ]
        for pat in exp_patterns:
            matches = re.findall(pat, text, re.IGNORECASE)
            if matches:
                try:
                    vals = [float(m) for m in matches if float(m) <= 40]
                    if vals:
                        return max(vals)
                except ValueError:
                    pass

        # Check for year ranges e.g. 2020-2023
        ranges = re.findall(r"\b(201\d|202\d)\s*(?:-|to)\s*(201\d|202\d|present|current)\b", text, re.IGNORECASE)
        total_range_years = 0
        for start_yr, end_yr in ranges:
            try:
                start = int(start_yr)
                end = 2025 if end_yr.lower() in ["present", "current"] else int(end_yr)
                if end >= start:
                    total_range_years += (end - start)
            except Exception:
                continue

        if total_range_years > 0:
            return float(min(total_range_years, 30))

        # Check if fresher / student keywords appear
        if re.search(r"\b(fresher|entry[- ]level|student|intern|internship)\b", text, re.IGNORECASE):
            return 0.0

        return 1.0  # default baseline if unspecified

    @classmethod
    def extract_projects_and_certs(cls, text: str) -> Tuple[int, List[str], List[str]]:
        """Identify project count and certification items."""
        # Project section detection
        project_count = 0
        project_matches = re.findall(r"(?:project|key projects|academic projects|personal projects)[\s\S]*?(?=(?:experience|education|skills|certifications|awards|$))", text, re.IGNORECASE)
        if project_matches:
            # Estimate number of bullet points or headers under projects
            bullets = re.findall(r"(?:•|\*|\-|\d+\.)\s+([A-Z][^\n]+)", project_matches[0])
            project_count = max(len(bullets), 1)
        else:
            # General keyword count for projects
            project_mentions = re.findall(r"\b(?:built|developed|created|implemented|designed)\s+(?:a|an|the)?\s+[A-Za-z\s]{3,30}\b", text, re.IGNORECASE)
            project_count = min(len(project_mentions), 5) or 1

        # Certifications
        certs = []
        for line in text.split("\n"):
            line_clean = line.strip()
            if any(kw in line_clean.lower() for kw in cls.CERT_KEYWORDS):
                if len(line_clean) < 120 and len(line_clean) > 5:
                    certs.append(line_clean)

        return project_count, list(dict.fromkeys(certs))[:5]

    @classmethod
    def parse_resume(cls, file_obj, filename: str) -> Dict[str, Any]:
        """
        Master method to parse a resume document and extract all info.
        """
        raw_text, error = cls.extract_text(file_obj, filename)
        if error:
            return {
                "success": False,
                "error": error,
                "raw_text": "",
                "filename": filename
            }

        name = cls.extract_name(raw_text)
        email = cls.extract_email(raw_text)
        phone = cls.extract_phone(raw_text)
        location = cls.extract_location(raw_text)
        education_info = cls.extract_education(raw_text)
        experience_years = cls.extract_experience_years(raw_text)
        project_count, certs = cls.extract_projects_and_certs(raw_text)

        return {
            "success": True,
            "error": None,
            "filename": filename,
            "raw_text": raw_text,
            "candidate_name": name,
            "email": email or "Not Provided",
            "phone": phone or "Not Provided",
            "location": location,
            "education": education_info,
            "experience_years": experience_years,
            "project_count": project_count,
            "certifications": certs,
            "certification_count": len(certs)
        }
