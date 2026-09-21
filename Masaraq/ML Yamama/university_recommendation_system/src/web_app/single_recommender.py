# single_recommender.py
# سكربت محلي 100%: Top-3 + أسباب EN/AR لِريكورد واحد جاهز (processed)

from __future__ import annotations
import os, json
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
import joblib

# ========= إعداد مسارات قابلة للتعديل =========
ROOT = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT, "data")
MODELS_DIR = os.path.join(DATA_DIR, "models")
PROC_DIR = os.path.join(DATA_DIR, "processed")   # مكان القوالب والـmetadata

# ========= قِيَم افتراضية للنصوص (للأمان) =========
DEFAULT_EN = {
    "gpa_strong": "Your GPA ({gpa}%) is well above the requirement, which strengthens your application.",
    "gpa_borderline": "Your GPA ({gpa}%) is around the typical requirement. Keep your performance steady.",
    "interest_high": "Your interests strongly align with this area: {interest_area}.",
    "career_opportunities": "Career outlook: {career_paths}, unemployment ~{unemployment}%, expected salary ~{salary}.",
    "final_encouragement": "This program could be a good fit given your profile."
}
DEFAULT_AR = {
    "gpa_strong": "معدلك ({gpa}%) أعلى من المتطلبات المعتادة، وهذا يعزز فرصك.",
    "gpa_borderline": "معدلك ({gpa}%) قريب من الحد المطلوب عادةً. استمر على هذا الأداء.",
    "interest_high": "ميولك تتوافق بقوة مع هذا المجال: {interest_area}.",
    "career_opportunities": "نظرة وظيفية: {career_paths}، بطالة ~{unemployment}%، راتب متوقع ~{salary}.",
    "final_encouragement": "هذا التخصص قد يكون مناسباً لملفك بشكل عام."
}

# ========= تحميل أدوات مساعدة محلية =========
def _load_json(path: str, default: Any = None) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def _load_dataframe(path_csv_or_json: str) -> pd.DataFrame:
    if not os.path.exists(path_csv_or_json):
        return pd.DataFrame()
    if path_csv_or_json.lower().endswith(".csv"):
        return pd.read_csv(path_csv_or_json, encoding="utf-8")
    return pd.DataFrame(_load_json(path_csv_or_json, default=[]))

# ========= تحميل الداتا والنماذج (محلي) =========
def load_data() -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """
    يعيد:
      - program_metadata_df: يحتوي على program_name, program_name_ar, unemployment_rate, expected_salary, future_jobs, (اختياري area)
      - questions_config: dict
      - university_data: dict
    """
    # حاول نلاقي ملفات منطقية
    meta_path_csv = os.path.join(PROC_DIR, "program_metadata.csv")
    meta_path_json = os.path.join(PROC_DIR, "program_metadata.json")
    program_metadata_df = _load_dataframe(meta_path_csv if os.path.exists(meta_path_csv) else meta_path_json)

    questions_config = _load_json(os.path.join(DATA_DIR, "config", "questions_config.json"), default={})
    university_data = _load_json(os.path.join(PROC_DIR, "university_data.json"), default={})
    return program_metadata_df, questions_config, university_data

def load_templates() -> Tuple[Dict[str, str], Dict[str, str]]:
    templates_en = _load_json(os.path.join(PROC_DIR, "templates_en.json"), default={}) or {}
    templates_ar = _load_json(os.path.join(PROC_DIR, "templates_ar.json"), default={}) or {}
    return templates_en, templates_ar

@dataclass
class Models:
    ensemble: Any = None
    lr: Any = None
    rf: Any = None
    gb: Any = None

def load_models() -> Models:
    def _try(path):
        try:
            return joblib.load(path)
        except Exception:
            return None
    return Models(
        ensemble=_try(os.path.join(MODELS_DIR, "ensemble_model.pkl")),
        lr=_try(os.path.join(MODELS_DIR, "logistic_regression_model.pkl")),
        rf=_try(os.path.join(MODELS_DIR, "random_forest_model.pkl")),
        gb=_try(os.path.join(MODELS_DIR, "gradient_boosting_model.pkl")),
    )

# ========= توابع السكور والـprob =========
def scale_score_0_100(x: float) -> float:
    # قصّ إلى 0..100
    try:
        return float(max(0.0, min(100.0, x)))
    except Exception:
        return 0.0

def get_match_strength(score_100: float) -> Tuple[str, str]:
    if score_100 >= 85:  return ("Excellent", "ممتاز")
    if score_100 >= 70:  return ("Very Good", "جيد جداً")
    if score_100 >= 55:  return ("Good", "جيد")
    if score_100 >= 40:  return ("Fair", "مقبول")
    return ("Poor", "ضعيف")

def predict_probabilities(models: Models, X: pd.DataFrame) -> Dict[str, float]:
    """
    يعيد dict: {label -> prob}. يفضّل ensemble ثم LR ثم RF ثم GB.
    يفترض أن model.classes_ موجودة وأن predict_proba متاحة.
    """
    model = models.ensemble or models.lr or models.rf or models.gb
    if model is None or not hasattr(model, "predict_proba") or not hasattr(model, "classes_"):
        return {}
    probs = model.predict_proba(X)
    if isinstance(probs, list) or isinstance(probs, tuple):
        probs = np.array(probs)
    # لو كانت ثنائية/متعددة الأصناف
    if probs.ndim == 2:
        cls = list(map(str, getattr(model, "classes_", [])))
        row = probs[0]
        return {cls[i]: float(row[i]) for i in range(len(cls))}
    # fallback
    return {}

def _find_program_interest_score(student: pd.Series, program_name: str, program_row: pd.Series) -> Tuple[str, float]:
    """
    يحاول يربط برنامج بمنطقة اهتمام عبر:
      1) عمود area في metadata إن وجد
      2) البحث عن interest_{program_name} في المدخلات
    ويعيد (اسم المنطقة/البرنامج، قيمة 0..100).
    """
    # عبر area
    area = str(program_row.get("area", "") or "").strip()
    if area:
        key = f"interest_{area}"
        val = student.get(key, None)
        if val is None:
            # جرّب الإنجليزية/المسافات
            key2 = f"interest_{area.replace(' ', '_')}"
            val = student.get(key2, 0.0)
        try:
            return (area, float(val))
        except Exception:
            pass

    # عبر program name
    key = f"interest_{program_name}"
    val = student.get(key, None)
    if val is None:
        key2 = f"interest_{program_name.replace(' ', '_')}"
        val = student.get(key2, 0.0)
    try:
        return (program_name, float(val))
    except Exception:
        return ("", 0.0)

def calculate_program_scores(
    student: pd.Series,
    program_metadata_df: pd.DataFrame,
    probabilities: Dict[str, float],
) -> pd.DataFrame:
    """
    يبني DataFrame نهائي فيه final_score (0..100).
    الأساس: احتمال النموذج (إن وجد) * 100 + boosts بسيطة.
    """
    rows = []
    student_gpa = float(student.get("general_average", 0.0) or 0.0)

    for _, row in program_metadata_df.iterrows():
        prog_en = str(row.get("program_name", "") or "")
        prog_ar = str(row.get("program_name_ar", row.get("program_name_arabic", prog_en)) or prog_en)

        p = float(probabilities.get(prog_en, 0.0))   # لو مافي احتمال، صفر
        base = p * 100.0

        # boost بسيط حسب الاهتمام
        area, area_score = _find_program_interest_score(student, prog_en, row)
        boost = 0.0
        if area_score >= 80: boost += 8.0
        elif area_score >= 60: boost += 4.0
        elif area_score >= 40: boost += 2.0

        # boost بسيط إذا GPA قوي
        if student_gpa >= 90: boost += 5.0
        elif student_gpa >= 80: boost += 2.0

        final_score = scale_score_0_100(base + boost)

        rows.append({
            "program_name": prog_en,
            "program_name_ar": prog_ar,
            "final_score": final_score,
            "unemployment_rate": row.get("unemployment_rate", 0.0),
            "expected_salary": row.get("expected_salary", 0.0),
            "future_jobs": row.get("future_jobs", "related fields"),
            "area": row.get("area", "")
        })

    df = pd.DataFrame(rows).sort_values("final_score", ascending=False).reset_index(drop=True)
    return df

# ========= توليد الأسباب بالاعتماد على المقتطف اللي أعطيتني إياه =========
def generate_reasons_for_program(
    student: pd.Series,
    program_row: pd.Series,
    templates_en: Dict[str, str],
    templates_ar: Dict[str, str]
) -> Tuple[List[str], List[str]]:

    reasons_en: List[str] = []
    reasons_ar: List[str] = []

    # GPA
    student_gpa = float(student.get("general_average", 0.0) or 0.0)
    if student_gpa >= 85:
        reasons_en.append(templates_en.get("gpa_strong", DEFAULT_EN["gpa_strong"]).format(gpa=round(student_gpa, 1)))
        reasons_ar.append(templates_ar.get("gpa_strong", DEFAULT_AR["gpa_strong"]).format(gpa=round(student_gpa, 1)))
    else:
        reasons_en.append(templates_en.get("gpa_borderline", DEFAULT_EN["gpa_borderline"]).format(gpa=round(student_gpa, 1)))
        reasons_ar.append(templates_ar.get("gpa_borderline", DEFAULT_AR["gpa_borderline"]).format(gpa=round(student_gpa, 1)))

    # Interest alignment
    program_name = str(program_row.get("program_name", ""))
    area, area_score = _find_program_interest_score(student, program_name, program_row)
    if area and area_score >= 70:
        reasons_en.append(templates_en.get("interest_high", DEFAULT_EN["interest_high"]).format(interest_area=area))
        reasons_ar.append(templates_ar.get("interest_high", DEFAULT_AR["interest_high"]).format(interest_area=area))

    # Career prospects (simple surface)
    unemployment = program_row.get("unemployment_rate", 0.0)
    salary = program_row.get("expected_salary", 0.0)
    career_paths = program_row.get("future_jobs", "related fields")
    reasons_en.append(
        templates_en.get("career_opportunities", DEFAULT_EN["career_opportunities"]).format(
            career_paths=career_paths, unemployment=unemployment, salary=salary
        )
    )
    reasons_ar.append(
        templates_ar.get("career_opportunities", DEFAULT_AR["career_opportunities"]).format(
            career_paths=career_paths, unemployment=unemployment, salary=salary
        )
    )

    # Final encouragement
    reasons_en.append(templates_en.get("final_encouragement", DEFAULT_EN["final_encouragement"]))
    reasons_ar.append(templates_ar.get("final_encouragement", DEFAULT_AR["final_encouragement"]))

    return reasons_en[:4], reasons_ar[:4]

# ========= API العامة: ريكورد واحد جاهز =========
def recommend_for_single(student_ready: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(student_ready, dict):
        raise TypeError("student_ready must be a dict of ready (processed) features")

    program_metadata_df, questions_config, university_data = load_data()
    models = load_models()

    # تجهيز DF من الريكورد الجاهز
    student_series = pd.Series(student_ready)
    student_df_for_model = pd.DataFrame([student_series])

    # احتمالات
    probabilities = predict_probabilities(models, student_df_for_model)

    # ترتيب السكور
    scores_df = calculate_program_scores(student_series, program_metadata_df, probabilities)

    # أعلى 3 مؤهّلين
    eligible_top = scores_df[scores_df["final_score"] > 0].head(3).copy()

    # قوالب
    templates_en, templates_ar = load_templates()

    top3: List[Dict[str, Any]] = []
    for _, prog in eligible_top.iterrows():
        major_en = str(prog.get("program_name", ""))
        major_ar = str(prog.get("program_name_ar", major_en))
        score_100 = scale_score_0_100(float(prog.get("final_score", 0.0)))
        match_en, match_ar = get_match_strength(score_100)
        reasons_en, reasons_ar = generate_reasons_for_program(student_series, prog, templates_en, templates_ar)
        top3.append({
            "major_en": major_en,
            "major_ar": major_ar,
            "score_100": score_100,
            "match_strength_en": match_en,
            "match_strength_ar": match_ar,
            "reasons_en": reasons_en,
            "reasons_ar": reasons_ar,
        })

    return {
        "student_id": str(student_ready.get("student_id", "N/A")),
        "top3": top3,
    }

# ========= تحويل لهيئة القالب recommendations.html =========
def to_template_items(top3: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    items = []
    for e in top3:
        items.append({
            "field_name": e["major_en"],
            "field_name_ar": e.get("major_ar", e["major_en"]),
            "score": float(e["score_100"]) / 100.0,  # القالب يضرب *100
            "match_strength": e.get("match_strength_en", ""),
            "match_strength_ar": e.get("match_strength_ar", ""),
            "reasons": e.get("reasons_en", []),   # احتياطي
            "reasons_en": e.get("reasons_en", []),
            "reasons_ar": e.get("reasons_ar", []),
            "subfields": [],
            "subfields_ar": [],
        })
    return items

# ========= تشغيل يدوي (اختياري) =========
if __name__ == "__main__":
    STUDENT_READY: Dict[str, Any] = {}  # ضع ريكوردك الجاهز هنا
    if not STUDENT_READY:
        print("Please populate STUDENT_READY with a ready (processed) record before running.")
    else:
        import pprint
        result = recommend_for_single(STUDENT_READY)
        pprint.pprint(result, width=120, compact=True)
