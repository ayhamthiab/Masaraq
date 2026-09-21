"""
Ranked Recommendations module
Generates ranked (1st/2nd/3rd) bilingual reasons per recommendation and writes
outputs/ranked_recommendations.json.

This module is intentionally independent and re-uses functions from recommendation_engine.py.
"""
from pathlib import Path
import sys
from typing import List, Dict, Any, Tuple, Optional

# Make sure local src is importable when running this file directly
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent))  # so "utils" and "ml_models" imports work

from ml_models.utils import ensure_dirs, save_json, PROC_DIR, OUTPUTS_DIR
import ml_models.recommendation_engine as rec_engine


# -----------------------
# Templates for each rank
# -----------------------
FIRST_CHOICE_EN = {
    "intro": "🎯 TOP RECOMMENDATION: This is your ideal academic path!",
    "gpa_strong": "Your outstanding GPA of {gpa}% makes you a perfect candidate for this highly competitive program.",
    "gpa_good": "Your solid GPA of {gpa}% positions you well for success in this excellent program.",
    "gpa_borderline": "Your GPA of {gpa}% meets the requirements, and we're confident this program will help you excel.",
    "interest_match": "Your passion for {interest_area} ({interest_score}% interest level) makes this the perfect match for your career goals.",
    "strength_highlight": "Your exceptional performance in {strong_subject} ({score}%) demonstrates you have exactly what it takes to lead in this field.",
    "career_strong": "This field offers the best career prospects with {unemployment}% unemployment and excellent salaries averaging {salary} NIS.",
    "final_strong": "This is where you belong! We strongly believe you'll thrive and become a leader in this field."
}

FIRST_CHOICE_AR = {
    "intro": "🎯 التوصية الأولى: هذا هو مسارك الأكاديمي المثالي!",
    "gpa_strong": "معدلك المتميز {gpa}% يجعلك مرشحاً مثالياً لهذا البرنامج التنافسي.",
    "gpa_good": "معدلك الجيد {gpa}% يضعك في موقع ممتاز للنجاح في هذا البرنامج الرائع.",
    "gpa_borderline": "معدلك {gpa}% يلبي المتطلبات، ونحن واثقون أن هذا البرنامج سيساعدك على التفوق.",
    "interest_match": "شغفك في {interest_area} (مستوى اهتمام {interest_score}%) يجعل هذا التخصص المطابق المثالي لأهدافك المهنية.",
    "strength_highlight": "أداؤك الاستثنائي في {strong_subject} ({score}%) يُظهر أن لديك بالضبط ما يلزم للريادة في هذا المجال.",
    "career_strong": "هذا المجال يوفر أفضل الفرص المهنية مع معدل بطالة {unemployment}% ورواتب ممتازة بمتوسط {salary} شيكل.",
    "final_strong": "هذا هو مكانك! نؤمن بقوة أنك ستزدهر وتصبح رائداً في هذا المجال."
}

SECOND_CHOICE_EN = {
    "intro": "🥈 EXCELLENT ALTERNATIVE: A strong choice that offers great opportunities!",
    "gpa_strong": "Your excellent GPA of {gpa}% gives you a competitive edge in this respected program.",
    "gpa_good": "Your GPA of {gpa}% aligns well with this program's standards and expectations.",
    "gpa_borderline": "Your GPA of {gpa}% meets this program's requirements, offering you a solid path forward.",
    "interest_match": "Your interest in {interest_area} ({interest_score}%) shows good potential for growth in this field.",
    "strength_highlight": "Your strong performance in {strong_subject} ({score}%) indicates you have valuable skills for this area.",
    "career_good": "This field provides solid career opportunities with {unemployment}% unemployment rate and competitive salaries around {salary} NIS.",
    "final_good": "This is an excellent backup choice that could lead to a fulfilling and successful career."
}

SECOND_CHOICE_AR = {
    "intro": "🥈 بديل ممتاز: خيار قوي يوفر فرص رائعة!",
    "gpa_strong": "معدلك الممتاز {gpa}% يمنحك أفضلية تنافسية في هذا البرنامج المحترم.",
    "gpa_good": "معدلك {gpa}% يتماشى جيداً مع معايير وتوقعات هذا البرنامج.",
    "gpa_borderline": "معدلك {gpa}% يلبي متطلبات هذا البرنامج، ويوفر لك مساراً صلباً للأمام.",
    "interest_match": "اهتمامك في {interest_area} ({interest_score}%) يُظهر إمكانات جيدة للنمو في هذا المجال.",
    "strength_highlight": "أداؤك القوي في {strong_subject} ({score}%) يُظهر أن لديك مهارات قيمة لهذا المجال.",
    "career_good": "هذا المجال يوفر فرص مهنية صلبة مع معدل بطالة {unemployment}% ورواتب تنافسية حوالي {salary} شيكل.",
    "final_good": "هذا خيار احتياطي ممتاز يمكن أن يؤدي إلى مهنة مُرضية وناجحة."
}

THIRD_CHOICE_EN = {
    "intro": "🥉 WORTH CONSIDERING: An interesting option to explore and consider.",
    "gpa_strong": "Your high GPA of {gpa}% opens doors to this program, though it may not be your first choice.",
    "gpa_good": "Your GPA of {gpa}% qualifies you for this program, which could surprise you with its opportunities.",
    "gpa_borderline": "Your GPA of {gpa}% meets this program's entry requirements, offering another pathway to consider.",
    "interest_match": "While {interest_area} ({interest_score}%) isn't your strongest interest, this field might spark new passions.",
    "strength_highlight": "Your skills in {strong_subject} ({score}%) could be valuable in unexpected ways in this field.",
    "career_explore": "This field offers {unemployment}% unemployment rate with average salaries of {salary} NIS - worth exploring further.",
    "final_explore": "Keep this option open - sometimes the best opportunities come from unexpected directions!"
}

THIRD_CHOICE_AR = {
    "intro": "🥉 يستحق النظر: خيار مثير للاهتمام للاستكشاف والتفكير.",
    "gpa_strong": "معدلك العالي {gpa}% يفتح لك أبواب هذا البرنامج، رغم أنه قد لا يكون خيارك الأول.",
    "gpa_good": "معدلك {gpa}% يؤهلك لهذا البرنامج، والذي قد يفاجئك بفرصه.",
    "gpa_borderline": "معدلك {gpa}% يلبي متطلبات القبول في هذا البرنامج، ويوفر مساراً آخر للنظر فيه.",
    "interest_match": "رغم أن {interest_area} ({interest_score}%) ليس اهتمامك الأقوى، إلا أن هذا المجال قد يثير شغف جديد.",
    "strength_highlight": "مهاراتك في {strong_subject} ({score}%) يمكن أن تكون قيمة بطرق غير متوقعة في هذا المجال.",
    "career_explore": "هذا المجال يوفر معدل بطالة {unemployment}% مع متوسط رواتب {salary} شيكل - يستحق المزيد من الاستكشاف.",
    "final_explore": "اتركي هذا الخيار مفتوحاً - أحياناً أفضل الفرص تأتي من اتجاهات غير متوقعة!"
}


import json

def load_templates(proc_dir: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load templates from PROC_DIR/reasons_templates_{en,ar}.json if present.
    Returns (templates_en, templates_ar) where each is a dict with keys:
      'first_choice', 'second_choice', 'third_choice'
    Falls back to module constants if files are missing or malformed.
    """
    en_path = proc_dir / "reasons_templates_en.json"
    ar_path = proc_dir / "reasons_templates_ar.json"

    en = None
    ar = None

    try:
        if en_path.exists():
            with open(en_path, "r", encoding="utf-8") as f:
                data_en = json.load(f)
                # If file already contains the grouped keys, use it directly.
                if isinstance(data_en, dict) and "first_choice" in data_en:
                    en = data_en
                elif isinstance(data_en, dict):
                    # assume this is the structure matching our FIRST_* maps
                    en = {
                        "first_choice": data_en.get("first_choice", data_en),
                        "second_choice": data_en.get("second_choice", data_en),
                        "third_choice": data_en.get("third_choice", data_en)
                    }
    except Exception:
        en = None

    try:
        if ar_path.exists():
            with open(ar_path, "r", encoding="utf-8") as f:
                data_ar = json.load(f)
                if isinstance(data_ar, dict) and "first_choice" in data_ar:
                    ar = data_ar
                elif isinstance(data_ar, dict):
                    ar = {
                        "first_choice": data_ar.get("first_choice", data_ar),
                        "second_choice": data_ar.get("second_choice", data_ar),
                        "third_choice": data_ar.get("third_choice", data_ar)
                    }
    except Exception:
        ar = None

    if not en:
        en = {
            "first_choice": FIRST_CHOICE_EN,
            "second_choice": SECOND_CHOICE_EN,
            "third_choice": THIRD_CHOICE_EN
        }
    if not ar:
        ar = {
            "first_choice": FIRST_CHOICE_AR,
            "second_choice": SECOND_CHOICE_AR,
            "third_choice": THIRD_CHOICE_AR
        }

    return en, ar

# -----------------------
# Helper functions
# -----------------------
def _find_strongest_subject(student: Any) -> Optional[str]:
    """Find the strongest subject column for a student (excluding interest_ and meta cols)."""
    best_subject = None
    best_score = -1.0
    for col in student.index:
        if str(col).startswith("interest_") or col in ("student_id", "student_name", "label", "branch", "general_average"):
            continue
        try:
            score = float(student.get(col, 0.0))
            if score > best_score:
                best_score = score
                best_subject = col
        except Exception:
            continue
    return best_subject


def _get_subject_score(student: Any, subject_name: str) -> float:
    """Return numeric score for a subject column if present."""
    try:
        return float(student.get(subject_name, 0.0))
    except Exception:
        return 0.0


def _find_program_interest_score(student: Any, program_name: str, program_row: Any) -> Tuple[Optional[str], float]:
    """
    Find the interest_ column best matching the program_name or faculty; fallback to highest interest.
    Returns (area_name, score_percent)
    """
    interest_cols = [c for c in student.index if str(c).startswith("interest_")]
    best_area = None
    best_score = 0.0
    for col in interest_cols:
        area = str(col).replace("interest_", "")
        try:
            score = float(student.get(col, 0.0))
        except Exception:
            score = 0.0
        if area.lower() in str(program_name).lower():
            if score > best_score:
                best_area = area
                best_score = score
        elif area.lower() in str(program_row.get("faculty_name", "")).lower():
            if score > best_score:
                best_area = area
                best_score = score
        else:
            if score > best_score:
                best_area = area
                best_score = score
    return best_area, best_score


# -----------------------
# Main generator
# -----------------------
def generate_ranked_reasons(
    student: Any,
    program_row: Any,
    rank: int
) -> Tuple[List[str], List[str]]:
    """
    Generate bilingual reasons for a given rank (1,2,3).
    Returns (reasons_en, reasons_ar) -- lists of strings (max 5 each).
    """
    # pick templates by rank
    if rank == 1:
        tpl_en = FIRST_CHOICE_EN
        tpl_ar = FIRST_CHOICE_AR
    elif rank == 2:
        tpl_en = SECOND_CHOICE_EN
        tpl_ar = SECOND_CHOICE_AR
    else:
        tpl_en = THIRD_CHOICE_EN
        tpl_ar = THIRD_CHOICE_AR

    reasons_en: List[str] = []
    reasons_ar: List[str] = []

    # Intro
    reasons_en.append(tpl_en.get("intro", ""))
    reasons_ar.append(tpl_ar.get("intro", ""))

    # GPA analysis
    student_gpa = float(student.get("general_average", 0.0))
    min_gpa = float(program_row.get("min_gpa_regular", 0.0))

    if student_gpa >= min_gpa + 10:
        gpa_key = "gpa_strong"
    elif student_gpa >= min_gpa:
        gpa_key = "gpa_good"
    else:
        gpa_key = "gpa_borderline"

    reasons_en.append(tpl_en.get(gpa_key, "").format(gpa=round(student_gpa, 1)))
    reasons_ar.append(tpl_ar.get(gpa_key, "").format(gpa=round(student_gpa, 1)))

    # Interest match
    interest_area, interest_score = _find_program_interest_score(student, program_row.get("program_name", ""), program_row)
    if interest_area:
        reasons_en.append(tpl_en.get("interest_match", tpl_en.get("interest_match", "")).format(
            interest_area=interest_area, interest_score=round(interest_score, 1)
        ))
        reasons_ar.append(tpl_ar.get("interest_match", tpl_ar.get("interest_match", "")).format(
            interest_area=interest_area, interest_score=round(interest_score, 1)
        ))

    # Strength highlight
    strong_subj = _find_strongest_subject(student)
    if strong_subj:
        score = _get_subject_score(student, strong_subj)
        reasons_en.append(tpl_en.get("strength_highlight", "").format(
            strong_subject=strong_subj, score=round(score, 1)
        ))
        reasons_ar.append(tpl_ar.get("strength_highlight", "").format(
            strong_subject=strong_subj, score=round(score, 1)
        ))

    # Career / market info
    unemployment = program_row.get("unemployment_rate", 15)
    salary = program_row.get("expected_salary", 3500)
    if rank == 1:
        career_key = "career_strong"
    elif rank == 2:
        career_key = "career_good"
    else:
        career_key = "career_explore"

    reasons_en.append(tpl_en.get(career_key, "").format(unemployment=unemployment, salary=salary))
    reasons_ar.append(tpl_ar.get(career_key, "").format(unemployment=unemployment, salary=salary))

    # Final encouragement
    if rank == 1:
        final_key = "final_strong"
    elif rank == 2:
        final_key = "final_good"
    else:
        final_key = "final_explore"

    reasons_en.append(tpl_en.get(final_key, ""))
    reasons_ar.append(tpl_ar.get(final_key, ""))

    # Limit to max 5 reasons (as requested)
    return reasons_en[:5], reasons_ar[:5]


def build_ranked_recommendations_json(output_path: Path) -> None:
    """
    Build ranked recommendations for all students and save JSON to output_path/"ranked_recommendations.json".
    """
    ensure_dirs()

    # load pre-trained models and data via existing module functions
    # Ensure classes used when models were pickled are available on __main__
    try:
        import ml_models.model_training as _mt
        import sys as _sys
        if hasattr(_mt, "SoftVotingEnsemble"):
            _sys.modules.setdefault("__main__", __import__("__main__"))
            setattr(_sys.modules["__main__"], "SoftVotingEnsemble", _mt.SoftVotingEnsemble)
    except Exception:
        pass

    models = rec_engine.load_models()
    features_df, program_metadata_df, questions_config, university_data = rec_engine.load_data()

    results: List[Dict[str, Any]] = []

    for idx, student in features_df.iterrows():
        student_id = str(student.get("student_id", idx))
        # processed student for model input
        student_df_proc = rec_engine.preprocess_student_data(rec_engine.pd.DataFrame([student]), program_metadata_df)
        probabilities = rec_engine.predict_probabilities(models, student_df_proc)
        scores_df = rec_engine.calculate_program_scores(student, program_metadata_df, probabilities, university_data)

        top_df = scores_df[scores_df["final_score"] > 0].head(3)

        recs = []
        for rank, (_, prog) in enumerate(top_df.iterrows(), start=1):
            major_name = prog.get("program_name")
            score = float(prog.get("final_score", 0.0))
            reasons_en, reasons_ar = generate_ranked_reasons(student, prog, rank)

            confidence_level = "High" if rank == 1 else "Medium" if rank == 2 else "Moderate"
            confidence_level_ar = "عالية" if rank == 1 else "متوسطة" if rank == 2 else "معقولة"
            recommendation_type = "Primary" if rank == 1 else "Alternative" if rank == 2 else "Exploratory"

            recs.append({
                "major": major_name,
                "rank": rank,
                "score": round(score, 4),
                "confidence_level": confidence_level,
                "confidence_level_ar": confidence_level_ar,
                "reasons_en": reasons_en,
                "reasons_ar": reasons_ar,
                "recommendation_type": recommendation_type
            })

        results.append({
            "student_id": student_id,
            "recommendations": recs,
            "total_recommendations": len(recs),
            "generation_date": __import__("datetime").datetime.now().date().isoformat()
        })

    output_path.mkdir(parents=True, exist_ok=True)
    save_json(results, output_path / "ranked_recommendations.json")
    print(f"Created {len(results)} ranked recommendations at {output_path / 'ranked_recommendations.json'}")


if __name__ == "__main__":
    build_ranked_recommendations_json(OUTPUTS_DIR)
