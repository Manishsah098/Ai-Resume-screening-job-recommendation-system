"""
Text Preprocessing Module for Resume and Job Description Processing.
Preserves critical technical terms (e.g. C++, C#, .NET, Node.js) while removing stopwords and noise.
"""

import re
import string
from typing import List, Set

# Fallback English Stopwords list to ensure offline resilience
DEFAULT_STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves"
}

# Protected technical tokens mapping to safe placeholders
PROTECTED_TERMS_MAP = {
    r"\bc\+\+\b": "cpp_token",
    r"\bc\#\b": "csharp_token",
    r"\b\.net\b": "dotnet_token",
    r"\bnode\.js\b": "nodejs_token",
    r"\bnext\.js\b": "nextjs_token",
    r"\bvue\.js\b": "vuejs_token",
    r"\breact\.js\b": "reactjs_token",
    r"\bexpress\.js\b": "expressjs_token",
    r"\bci\/cd\b": "cicd_token",
    r"\btcp\/ip\b": "tcpip_token",
    r"\bpower bi\b": "powerbi_token",
    r"\bscikit-learn\b": "scikitlearn_token",
    r"\bmachine learning\b": "machinelearning_token",
    r"\bdeep learning\b": "deeplearning_token",
    r"\bnatural language processing\b": "nlp_token",
    r"\bcomputer vision\b": "computervision_token",
    r"\bgenerative ai\b": "generativeai_token"
}

REVERSE_PROTECTED_TERMS_MAP = {
    "cpp_token": "c++",
    "csharp_token": "c#",
    "dotnet_token": ".net",
    "nodejs_token": "node.js",
    "nextjs_token": "next.js",
    "vuejs_token": "vue.js",
    "reactjs_token": "react.js",
    "expressjs_token": "express.js",
    "cicd_token": "ci/cd",
    "tcpip_token": "tcp/ip",
    "powerbi_token": "power bi",
    "scikitlearn_token": "scikit-learn",
    "machinelearning_token": "machine learning",
    "deeplearning_token": "deep learning",
    "nlp_token": "nlp",
    "computervision_token": "computer vision",
    "generativeai_token": "generative ai"
}


class TextPreprocessor:
    """
    NLP Text Preprocessor for resume and job description text.
    """

    def __init__(self, custom_stopwords: Set[str] = None):
        self.stopwords = custom_stopwords if custom_stopwords else DEFAULT_STOPWORDS
        # Try importing nltk stopwords if available
        try:
            import nltk
            from nltk.corpus import stopwords
            try:
                nltk_stops = set(stopwords.words('english'))
                self.stopwords = self.stopwords.union(nltk_stops)
            except LookupError:
                nltk.download('stopwords', quiet=True)
                nltk_stops = set(stopwords.words('english'))
                self.stopwords = self.stopwords.union(nltk_stops)
        except Exception:
            pass

    def protect_tech_terms(self, text: str) -> str:
        """Replace special technical terms with safe alpha tokens."""
        lower_text = text.lower()
        for pattern, placeholder in PROTECTED_TERMS_MAP.items():
            lower_text = re.sub(pattern, placeholder, lower_text)
        return lower_text

    def restore_tech_terms(self, text: str) -> str:
        """Restore placeholder tokens back to recognized technical names."""
        for placeholder, original in REVERSE_PROTECTED_TERMS_MAP.items():
            text = re.sub(r"\b" + placeholder + r"\b", original, text)
        return text

    def clean_text(self, text: str, preserve_case: bool = False) -> str:
        """
        Main text cleaning pipeline:
        1. Protect key technical terms
        2. Lowercase (if not preserve_case)
        3. Remove URLs, emails, phone numbers
        4. Remove unwanted punctuation
        5. Tokenize and remove stopwords
        6. Remove extra whitespace
        """
        if not text or not isinstance(text, str):
            return ""

        # Step 1: Protect special technical terms
        processed = self.protect_tech_terms(text)

        # Step 2: Remove URLs
        processed = re.sub(r"https?://\S+|www\.\S+", " ", processed)

        # Step 3: Remove email addresses and phone formats for generic NLP matching
        processed = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", " ", processed)
        processed = re.sub(r"\+?\d[\d -]{8,}\d", " ", processed)

        # Step 4: Remove non-alphanumeric except underscore (used in placeholders)
        processed = re.sub(r"[^a-zA-Z0-9_\s]", " ", processed)

        # Step 5: Tokenize & Remove stopwords
        tokens = processed.split()
        cleaned_tokens = [
            token for token in tokens
            if token not in self.stopwords and len(token) > 1 or token in PROTECTED_TERMS_MAP.values()
        ]

        # Step 6: Restore technical terms
        result = " ".join(cleaned_tokens)
        result = self.restore_tech_terms(result)

        # Step 7: Collapse multiple spaces
        result = re.sub(r"\s+", " ", result).strip()
        return result

    def tokenize(self, text: str) -> List[str]:
        """Tokenize preprocessed text into word list."""
        cleaned = self.clean_text(text)
        return cleaned.split()


# Singleton instance for general usage
preprocessor = TextPreprocessor()
