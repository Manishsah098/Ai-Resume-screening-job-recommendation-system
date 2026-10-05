"""
Model Training Script.
Train and persist the job-role classifier from the jobs.csv dataset.
Run this script once before the application to pre-train the classifier:
    python training/train_model.py
"""

import os
import sys

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from modules.classifier import JobClassifier
from utils.helpers import load_jobs_dataframe


def main():
    print("=" * 60)
    print("  AI Resume Screening - Classifier Training Script")
    print("=" * 60)

    # Paths
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    jobs_csv = os.path.join(base_dir, "data", "jobs.csv")
    model_path = os.path.join(base_dir, "models", "job_classifier.pkl")

    print(f"\n[1] Loading job dataset from: {jobs_csv}")
    try:
        jobs_df = load_jobs_dataframe(jobs_csv)
        print(f"    Loaded {len(jobs_df)} job records.")
    except Exception as e:
        print(f"    ERROR: {e}")
        sys.exit(1)

    print(f"\n[2] Initializing classifier...")
    classifier = JobClassifier(model_path=model_path)

    print(f"\n[3] Training Logistic Regression classifier on job categories...")
    result = classifier.train(jobs_df)

    if not result.get("success"):
        print(f"    ERROR: {result.get('error', 'Unknown error')}")
        sys.exit(1)

    print(f"\n[OK] Training complete!")
    print(f"  Train samples : {result['train_samples']}")
    print(f"  Test samples  : {result['test_samples']}")
    print(f"  Test Accuracy : {result['accuracy']}%")
    print(f"\n  Categories    : {', '.join(result['categories'])}")
    print(f"\n[4] Model saved to: {model_path}")
    print("\n  IMPORTANT: The accuracy above reflects the classifier on the test set.")
    print("  It is NOT the same as a job match score (which measures resume-job similarity).")
    print("=" * 60)


if __name__ == "__main__":
    main()
