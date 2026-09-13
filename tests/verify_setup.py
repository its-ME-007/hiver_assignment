"""
Quick verification script for evaluators
Checks all Phase 2 deliverables are in place
"""

import os
import json
import sys

print("=" * 80)
print("PHASE 2 DELIVERABLES VERIFICATION")
print("=" * 80)

checks = {
    "Python Modules": [
        "01_brand_analysis.py",
        "02_intent_taxonomy.py",
        "03_build_golden_set.py",
        "04_semi_auto_label_golden_set.py",
        "05_establish_baselines.py",
        "06_build_faiss_index.py",
        "07_train_intent_classifier.py",
        "08_escalation_rag_pipeline.py",
        "09_evaluation_harness.py",
        "app.py",
    ],
    "Data Files": [
        "data/processed/threads.parquet",
        "data/golden_set_labeled.jsonl",
        "data/golden_set_labeled.csv",
        "data/intent_taxonomy.json",
        "data/brands/uber/index.faiss",
        "data/brands/uber/metadata.jsonl",
    ],
    "Models": [
        "models/intent_classifier.pkl",
        "models/tfidf_vectorizer.pkl",
    ],
    "Configs": [
        "configs/intents.yaml",
        "configs/labeling_rubric.md",
        "configs/pipeline_config.json",
    ],
    "Reports": [
        "reports/baseline_scores.json",
        "reports/intent_scores.json",
        "reports/retrieval_scores.json",
        "reports/final_evaluation.json",
        "reports/pipeline_test_results.jsonl",
    ],
    "Documentation": [
        "README.md",
        "report.md",
        "PHASE_2_COMPLETE.md",
    ],
    "Dependencies": [
        "requirements.txt",
    ]
}

total = 0
passed = 0
failed = 0

for category, files in checks.items():
    print(f"\n{category}")
    print("-" * 80)
    
    for file_path in files:
        total += 1
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            size_str = f"{size / 1024 / 1024:.1f}MB" if size > 1024*1024 else f"{size / 1024:.1f}KB"
            print(f"  ✅ {file_path:<45} ({size_str})")
            passed += 1
        else:
            print(f"  ❌ {file_path:<45} (MISSING)")
            failed += 1

print("\n" + "=" * 80)
print(f"VERIFICATION SUMMARY: {passed}/{total} files present")
print("=" * 80)

if failed == 0:
    print(f"\n✅ ALL DELIVERABLES PRESENT - Ready for evaluation!")
    print(f"\nNext steps:")
    print(f"  1. Read README.md for setup instructions")
    print(f"  2. Run: streamlit run app.py")
    print(f"  3. Read report.md for full analysis")
    sys.exit(0)
else:
    print(f"\n⚠️  {failed} file(s) missing. Please check the file structure.")
    sys.exit(1)
