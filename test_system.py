import os
from modules.resume_parser import ResumeParser
from modules.skill_extractor import SkillExtractor
from modules.job_matcher import JobMatcher
from modules.recommender import Recommender
from modules.classifier import JobClassifier
from utils.helpers import load_jobs_dataframe

def test_full_pipeline():
    print("=" * 60)
    print("TESTING AI RESUME SCREENING & JOB RECOMMENDATION SYSTEM")
    print("=" * 60)

    # 1. Parse Resume
    resume_path = "sample_resumes/sample_data_scientist.docx"
    print(f"\n[1] Parsing sample resume: {resume_path}")
    parsed = ResumeParser.parse_resume(resume_path, "sample_data_scientist.docx")
    assert parsed["success"], f"Failed to parse resume: {parsed.get('error')}"
    print(f"    Name: {parsed['candidate_name']}")
    print(f"    Email: {parsed['email']}")
    print(f"    Experience: {parsed['experience_years']} years")
    print(f"    Education: {parsed['education']['degree_summary']}")

    # 2. Extract Skills
    print("\n[2] Extracting Skills...")
    extractor = SkillExtractor("data/skills.csv")
    categorized_skills = extractor.extract_skills(parsed["raw_text"])
    flat_skills = extractor.get_flat_skills(parsed["raw_text"])
    parsed["skills"] = flat_skills
    print(f"    Total detected skills: {len(flat_skills)}")
    print(f"    Skills preview: {flat_skills[:8]}")

    # 3. Job Matching
    print("\n[3] Matching against Job Dataset...")
    jobs_df = load_jobs_dataframe("data/jobs.csv")
    matcher = JobMatcher()
    matched_jobs = matcher.match_resume_to_jobs(
        resume_text=parsed["raw_text"],
        resume_skills=flat_skills,
        candidate_experience_years=parsed["experience_years"],
        candidate_degrees=parsed["education"]["degrees"],
        jobs_df=jobs_df
    )
    print(f"    Matched across {len(matched_jobs)} jobs.")

    # 4. Recommendation & Explainability
    print("\n[4] Top 5 Recommended Jobs with Explainability:")
    recommender = Recommender()
    top_5 = recommender.get_top_recommendations(matched_jobs, top_n=5)
    for i, job in enumerate(top_5, start=1):
        explanation = recommender.generate_explanation(job, flat_skills, parsed["experience_years"])
        print(f"\n    #{i} {job['job_title']} at {job['company']}")
        print(f"       Category: {job['category']} | Match Score: {job['match_score']}% ({explanation['match_label']})")
        print(f"       Text Sim: {job['text_similarity']}% | Skill Score: {job['skill_score']}%")
        print(f"       Skill Coverage: {explanation['skill_coverage']}")
        if explanation["reasons"]:
            print(f"       Key Match: {explanation['reasons'][0]}")

    # 5. Job Role Classification
    print("\n[5] Predicting Job Domain / Category via ML Classifier...")
    classifier = JobClassifier("models/job_classifier.pkl")
    if not classifier.is_trained:
        classifier.train(jobs_df)
    pred = classifier.predict(parsed["raw_text"])
    print(f"    Predicted Role/Category: {pred['predicted_category']}")
    print(f"    Confidence: {pred['confidence']}%")
    print(f"    Category Probabilities: {pred['all_probabilities']}")

    print("\n" + "=" * 60)
    print("[ALL TESTS PASSED SUCCESSFULLY!]")
    print("=" * 60)

if __name__ == "__main__":
    test_full_pipeline()
