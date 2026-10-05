"""
Job Role Classifier Module.
Trains a Logistic Regression classifier on job descriptions to predict job category.
Includes proper train/test split and evaluation metrics.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, precision_recall_fscore_support
)
from sklearn.pipeline import Pipeline

from modules.text_preprocessor import TextPreprocessor


class JobClassifier:
    """
    Classifies a resume into a broad job category using a trained Logistic Regression model.
    """

    def __init__(
        self,
        model_path: str = None,
        vectorizer_path: str = None
    ):
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.pipeline: Optional[Pipeline] = None
        self.categories: List[str] = []
        self.is_trained = False
        self.preprocessor = TextPreprocessor()
        self._load_model()

    def _load_model(self):
        """Load pretrained model pipeline if it exists."""
        if self.model_path and os.path.isfile(self.model_path):
            try:
                self.pipeline = joblib.load(self.model_path)
                if hasattr(self.pipeline, "classes_"):
                    self.categories = list(self.pipeline.classes_)
                elif hasattr(self.pipeline, "named_steps"):
                    lr = self.pipeline.named_steps.get("classifier")
                    if lr and hasattr(lr, "classes_"):
                        self.categories = list(lr.classes_)
                self.is_trained = True
            except Exception:
                self.pipeline = None
                self.is_trained = False

    def _build_training_data(self, jobs_df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Prepare corpus and labels from job dataset."""
        texts = []
        labels = []
        for _, row in jobs_df.iterrows():
            if "category" not in row or pd.isna(row["category"]):
                continue
            parts = [
                str(row.get("job_title", "")),
                str(row.get("description", "")),
                str(row.get("required_skills", "")).replace(",", " ")
            ]
            combined = " ".join(parts)
            cleaned = self.preprocessor.clean_text(combined)
            texts.append(cleaned)
            labels.append(str(row["category"]))
        return texts, labels

    def train(self, jobs_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train the classifier on the job dataset.
        Returns evaluation metrics from the held-out test set.
        """
        texts, labels = self._build_training_data(jobs_df)

        if len(texts) < 4:
            return {
                "success": False,
                "error": "Not enough labeled data to train classifier (need at least 4 samples)."
            }

        unique_cats = list(set(labels))
        self.categories = unique_cats

        # Split data
        test_size = 0.25 if len(texts) >= 16 else 0.2
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                texts, labels, test_size=test_size, random_state=42, stratify=labels
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                texts, labels, test_size=test_size, random_state=42
            )

        # Build pipeline
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)),
            ("classifier", LogisticRegression(
                max_iter=1000, C=1.0, solver="lbfgs",
                class_weight="balanced", random_state=42
            ))
        ])

        self.pipeline.fit(X_train, y_train)
        self.is_trained = True

        # Evaluate
        y_pred = self.pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=self.categories)

        # Save model
        if self.model_path:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            try:
                joblib.dump(self.pipeline, self.model_path)
            except Exception:
                pass

        return {
            "success": True,
            "accuracy": round(acc * 100, 2),
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "categories": self.categories,
            "train_samples": len(X_train),
            "test_samples": len(X_test)
        }

    def predict(self, resume_text: str) -> Dict[str, Any]:
        """Predict the most likely job category for a given resume."""
        if not self.is_trained or self.pipeline is None:
            return {
                "success": False,
                "predicted_category": "Model Not Trained",
                "confidence": 0.0,
                "all_probabilities": {}
            }
        cleaned = self.preprocessor.clean_text(resume_text)
        try:
            prediction = self.pipeline.predict([cleaned])[0]
            probabilities = self.pipeline.predict_proba([cleaned])[0]
            cat_probs = {
                cat: round(float(prob) * 100, 2)
                for cat, prob in zip(self.categories, probabilities)
            }
            confidence = cat_probs.get(prediction, 0.0)
            return {
                "success": True,
                "predicted_category": prediction,
                "confidence": confidence,
                "all_probabilities": dict(sorted(cat_probs.items(), key=lambda x: x[1], reverse=True))
            }
        except Exception as e:
            return {
                "success": False,
                "predicted_category": "Prediction Error",
                "confidence": 0.0,
                "all_probabilities": {},
                "error": str(e)
            }
