"""
Recommendation & Explanation Module for Masarak
Generates a single outputs/recommendations.json file containing recommendations
for all students with bilingual (EN/AR) explanations based on templates.

This script re-uses helper functions from src/recommendation_engine.py and src/utils.py.
Run from project root (the repository scripts assume utils paths are configured).
"""

from pathlib import Path
import sys
import json
from typing import List, Dict, Any, Tuple, Optional

# Ensure src is importable
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from utils import ensure_dirs, load_json, save_json, PROC_DIR, OUTPUTS_DIR
import recommendation_engine as rec_engine


# Default templates (used if template files are missing)
DEFAULT_EN = {
  "gpa_strong": "Your GPA ({gpa}%) is well above the requirement, showing strong readiness.",
  "gpa_borderline": "Your GPA ({gpa}%) is just around the requirement — with effort, you can succeed here.",
  "interest_high": "This aligns with your passion in {interest_area} — imagine studying what excites you every day!",
  "weakness_compensation": "Although your {weak_subject} score is a bit lower, your strength in {strong_subject} makes you a great fit.",
  "career_opportunities": "This major opens career paths in {career_paths}. Current job market: unemployment {unemployment}%, avg salary {salary} NIS.",
  "final_encouragement": "We believe you can shine in this path!"
}

DEFAULT_AR = {
  "gpa_strong": "معدلك ({gpa}%) أعلى بكثير من المطلوب، وهذا يثبت جاهزيتك الأكاديمية.",
  "gpa_borderline": "معدلك ({gpa}%) قريب من الحد المطلوب — بقليل من الجهد ستتفوق هنا.",
  "interest_high": "هذا التخصص يتوافق مع شغفك في {interest_area} — تخيل أنك تدرس ما تحبه كل يوم!",
  "weakness_compensation": "رغم أن نتيجتك في {weak_subject} أقل قليلاً، إلا أن قوتك في {strong_subject} تجعل منك مرشحاً مميزاً.",
  "career_opportunities": "هذا التخصص يفتح لك مجالات عمل في {career_paths}. سوق العمل الحالي: بطالة {unemployment}%، ومتوسط الراتب {salary} شيكل.",
  "final_encouragement": "نثق أنك ستتألق في هذا المسار!"
}


def load_or_create_templates(proc_dir: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Try to load templates from proc_dir; if not present, write defaults and return them.
    Expected filenames:
      - reasons_templates_en.json
      - reasons_templates_ar.json
    """
    en_path = proc_dir / "reasons_templates_en.json"
    ar_path = proc_dir / "reasons_templates_ar.json"

    if en_path.exists():
        try:
            en = load_json(en_path)
        except Exception:
            en = DEFAULT_EN
            save_json(en, en_path)
    else:
        en = DEFAULT_EN
        save_json(en, en_path)

    if ar_path.exists():
        try:
            ar = load_json(ar_path)
        except Exception:
            ar = DEFAULT_AR
            save_json(ar, ar_path)
    else:
        ar = DEFAULT_AR
        save_json(ar, ar_path)

    return en, ar


def _find_program_interest_score(student: Any, program_name: str, program_row: Any) -> Tuple[Optional[str], float]:
    """
    Return (interest_area_name, score_percent) best matching an interest_ column for the program.
    If none found, returns (None, 0.0).
    """
    interest_cols = [c for c in student.index if str(c).startswith("interest_")]
    best_area = None
    best_score = 0.0
    for col in interest_cols:
        area = str(col).replace("interest_", "")
        try:
            score = float(student.get(col, 0.0))
        except:
            score = 0.0
        # crude matching: if area appears in program name or faculty, prefer it
        if area.lower() in str(program_name).lower():
            if score > best_score:
                best_area = area
                best_score = score
        elif area.lower() in str(program_row.get("faculty_name", "")).lower():
            if score > best_score:
                best_area = area
                best_score = score
        else:
            # fallback: keep highest interest even if not matching
            if score > best_score:
                best_area = area
                best_score = score
    return best_area, best_score


def _find_weak_and_supporting_subjects(program_row: Any, student: Any) -> Tuple[Optional[str], Optional[str]]:
    """
    Identify if any core subject is weak (<70) and if there's a supporting subject strong (>=80).
    Returns (weak_subject_name, strong_subject_name) or (None, None).
    """
    core_subjects = program_row.get("core_subjects", []) or []
    # normalize core_subjects to list of names
    subject_names = []
    for item in core_subjects:
        if isinstance(item, dict):
            subject_names.append(item.get("subject") or item.get("name"))
        elif isinstance(item, (list, tuple)) and len(item) >= 1:
            subject_names.append(item[0])
        else:
            subject_names.append(item)
    weak = None
    for subj in subject_names:
        if not subj:
            continue
        possible_cols = [subj, f"{subj}_score", f"score_{subj}", f"{subj}_mark"]
        for col in possible_cols:
            if col in student.index:
                try:
                    val = float(student.get(col, 0.0))
                except:
                    val = 0.0
                if val < 70:
                    weak = subj
                    break
        if weak:
            break

    # find supporting strong subject (any subject column with >=80 that's not the weak one)
    strong = None
    for col in student.index:
        # look for subject-like columns (skip interest_, student_id, name, label, general_average, branch)
        name = str(col)
        if name.startswith("interest_") or name in ("student_id", "student_name", "label", "branch", "general_average", "top_interests"):
            continue
        try:
            val = float(student.get(col, 0.0))
        except:
            continue
        if val >= 80 and name != weak:
            strong = name
            break

    return weak, strong


def generate_reasons_for_program(
    student: Any,
    program_row: Any,
    templates_en: Dict[str, str],
    templates_ar: Dict[str, str]
) -> Tuple[List[str], List[str]]:
    """
    Build reasons_en and reasons_ar lists for a given student+program using templates.
    Follows rules specified:
      - GPA check (>= min_gpa + 10 -> gpa_strong, >= min_gpa -> gpa_borderline)
      - Interest alignment (interest >=70 -> interest_high)
      - Weakness compensation (core subject <70 + supporting subject high)
      - Career prospects (use career_opportunities)
      - Final encouragement (always)
    """
    reasons_en: List[str] = []
    reasons_ar: List[str] = []

    student_gpa = float(student.get("general_average", 0.0))
    min_gpa = float(program_row.get("min_gpa_regular", 0.0))

    # GPA rule
    if student_gpa >= min_gpa + 10:
        en_text = templates_en.get("gpa_strong", DEFAULT_EN["gpa_strong"]).format(gpa=round(student_gpa, 1))
        ar_text = templates_ar.get("gpa_strong", DEFAULT_AR["gpa_strong"]).format(gpa=round(student_gpa, 1))
        reasons_en.append(en_text)
        reasons_ar.append(ar_text)
    elif student_gpa >= min_gpa:
        en_text = templates_en.get("gpa_borderline", DEFAULT_EN["gpa_borderline"]).format(gpa=round(student_gpa, 1))
        ar_text = templates_ar.get("gpa_borderline", DEFAULT_AR["gpa_borderline"]).format(gpa=round(student_gpa, 1))
        reasons_en.append(en_text)
        reasons_ar.append(ar_text)
    else:
        # If below requirement, we still include a borderline/encouraging note
        en_text = templates_en.get("gpa_borderline", DEFAULT_EN["gpa_borderline"]).format(gpa=round(student_gpa, 1))
        ar_text = templates_ar.get("gpa_borderline", DEFAULT_AR["gpa_borderline"]).format(gpa=round(student_gpa, 1))
        reasons_en.append(en_text)
        reasons_ar.append(ar_text)

    # Interest alignment
    program_name = program_row.get("program_name", "")
    interest_area, interest_score = _find_program_interest_score(student, program_name, program_row)
    if interest_area and interest_score >= 70:
        en_text = templates_en.get("interest_high", DEFAULT_EN["interest_high"]).format(interest_area=interest_area)
        ar_text = templates_ar.get("interest_high", DEFAULT_AR["interest_high"]).format(interest_area=interest_area)
        reasons_en.append(en_text)
        reasons_ar.append(ar_text)

    # Weakness compensation
    weak_subj, strong_subj = _find_weak_and_supporting_subjects(program_row, student)
    if weak_subj and strong_subj:
        en_text = templates_en.get("weakness_compensation", DEFAULT_EN["weakness_compensation"])\
            .format(weak_subject=weak_subj, strong_subject=strong_subj)
        ar_text = templates_ar.get("weakness_compensation", DEFAULT_AR["weakness_compensation"])\
            .format(weak_subject=weak_subj, strong_subject=strong_subj)
        reasons_en.append(en_text)
        reasons_ar.append(ar_text)

    # Career prospects
    career_paths = program_row.get("future_jobs") or program_row.get("career_paths") or program_row.get("program_name", "")
    unemployment = program_row.get("unemployment_rate", None)
    salary = program_row.get("expected_salary", None)
    if unemployment is None:
        unemployment = 15
    if salary is None or salary == 0:
        salary = 3500
    en_text = templates_en.get("career_opportunities", DEFAULT_EN["career_opportunities"])\
        .format(career_paths=career_paths, unemployment=unemployment, salary=salary)
    ar_text = templates_ar.get("career_opportunities", DEFAULT_AR["career_opportunities"])\
        .format(career_paths=career_paths, unemployment=unemployment, salary=salary)
    reasons_en.append(en_text)
    reasons_ar.append(ar_text)

    # Final encouragement (ensure at least one encouraging sentence at end)
    en_text = templates_en.get("final_encouragement", DEFAULT_EN["final_encouragement"])
    ar_text = templates_ar.get("final_encouragement", DEFAULT_AR["final_encouragement"])
    reasons_en.append(en_text)
    reasons_ar.append(ar_text)

    # Limit to 4-5 reasons but allow more if logic added; for this task we keep top 4 for readability
    return reasons_en[:4], reasons_ar[:4]


def build_recommendations_json(output_path: Path) -> None:
    """
    Main entry: load models/data, generate recommendations + bilingual reasons, save JSON.
    """
    ensure_dirs()

    # load templates
    templates_en, templates_ar = load_or_create_templates(PROC_DIR)

    # load pre-trained models and data via existing module functions
    models = rec_engine.load_models()
    features_df, program_metadata_df, questions_config, university_data = rec_engine.load_data()

    results: List[Dict[str, Any]] = []

    for idx, student in features_df.iterrows():
        student_id = str(student.get("student_id", idx))
        # build processed student row for model prediction and scoring
        student_df_proc = rec_engine.preprocess_student_data(rec_engine.pd.DataFrame([student]), program_metadata_df)
        probabilities = rec_engine.predict_probabilities(models, student_df_proc)
        scores_df = rec_engine.calculate_program_scores(student, program_metadata_df, probabilities, university_data)
        # pick top 3 non-zero final_score
        top_df = scores_df[scores_df["final_score"] > 0].head(3)
        recs = []
        for _, prog in top_df.iterrows():
            major_name = prog.get("program_name")
            score = float(prog.get("final_score", 0.0))  # 0-1
            reasons_en, reasons_ar = generate_reasons_for_program(student, prog, templates_en, templates_ar)
            recs.append({
                "major": major_name,
                "score": round(score, 4),
                "reasons_en": reasons_en,
                "reasons_ar": reasons_ar
            })
        results.append({
            "student_id": student_id,
            "recommendations": recs
        })

    # save single recommendations.json in outputs folder
    output_path.mkdir(parents=True, exist_ok=True)
    save_json(results, output_path / "recommendations.json")
    print(f"Wrote {len(results)} student recommendations to {output_path / 'recommendations.json'}")


if __name__ == "__main__":
    build_recommendations_json(OUTPUTS_DIR)
