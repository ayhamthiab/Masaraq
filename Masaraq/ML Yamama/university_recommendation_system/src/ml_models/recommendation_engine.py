
import os
import pandas as pd
import numpy as np
from pathlib import Path
import json
import joblib
from typing import Dict, List, Tuple, Any, Optional, Iterable
import ast

from utils import (
    ROOT, DATA_DIR, RAW_DIR, PROC_DIR, INTERIM_DIR, 
    MODELS_DIR, OUTPUTS_DIR, RECS_DIR,
    ensure_dirs, set_seed, load_json, save_json, 
    save_model, load_model, find_column
)


def load_models() -> Dict[str, Any]:
    """
    Load the trained models.
    """
    models = {
        'random_forest': load_model(MODELS_DIR / "random_forest_model.pkl"),
        'gradient_boosting': load_model(MODELS_DIR / "gradient_boosting_model.pkl"),
        'logistic_regression': load_model(MODELS_DIR / "logistic_regression_model.pkl"),
        'ensemble': load_model(MODELS_DIR / "ensemble_model.pkl")
    }
    
    return models


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, Dict, Dict]:
    """
    Load the necessary data for generating recommendations.
    """
    features_df = pd.read_csv(PROC_DIR / "students_features.csv")
    program_metadata_df = pd.read_csv(PROC_DIR / "program_metadata.csv")
    questions_config = load_json(PROC_DIR / "questions_config.json")
    university_data = load_json(PROC_DIR / "university_data.json")
    
    return features_df, program_metadata_df, questions_config, university_data


# ==============================================================================
# 1. الدالة النهائية لـ preprocess_student_data (الإصلاح الأساسي)
# ==============================================================================
def preprocess_student_data(
    student_data: pd.DataFrame,
    program_metadata_df: pd.DataFrame
) -> pd.DataFrame:
    """
    THE ABSOLUTE FINAL AND CORRECT version of the preprocessing function.
    This version uses a simple and robust method to ensure all expected columns exist.
    """
    # إعادة تعيين الفهرس لإصلاح خطأ KeyError بشكل دائم
    student_data_proc = student_data.copy().reset_index(drop=True)

    # --- الخطوة 1: تحديد قائمة الأعمدة الكاملة التي يتوقعها المودل (بما في ذلك الأعمدة المكتشفة حديثًا) ---
    expected_model_columns = [
        'general_average', 'core_subject_average', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9',
        'q10', 'q11', 'q12', 'q13', 'q14', 'q15', 'q16', 'q17', 'q18', 'q19', 'q20', 'q21',
        'q22', 'q23', 'q24', 'q25', 'q26', 'q27', 'q28', 'q29', 'q30', 'q31', 'q32', 'q33',
        'q34', 'q35', 'q36', 'q37',
        'gpa_percentile', 'core_avg_percentile', 'gpa_to_core_ratio', 'num_eligible_programs', # الأعمدة الجديدة
        'interest_Medical Laboratory Sciences', 'interest_Fine Arts',
        'interest_Sharia', 'interest_Digital Economy', 'interest_Computer Engineering',
        'interest_Medical Imaging', 'interest_Computer Science', 'interest_Pharmacy',
        'interest_Medicine', 'interest_Midwifery', 'interest_Law',
        'interest_Cardiopulmonary Perfusion Technology', 'interest_Computer Science in the Job Market',
        'interest_Nursing', 'interest_Science', 'interest_Humanities',
        'interest_Audiology and Speech Sciences', 'interest_Optometry', 'interest_Physical Therapy',
        'interest_Business', 'interest_Anesthesia and Resuscitation',
        'interest_Digital Media and Communication', 'interest_Engineering',
        'eligible_Biology and Biotechnology', 'eligible_Mathematics', 'eligible_Mathematics and Data Science',
        'eligible_Physics', 'eligible_Applied Physics Technology', 'eligible_Chemistry', 'eligible_Applied Chemistry',
        'eligible_Molecular Diagnostics Technology', 'eligible_Arabic Language and Literature',
        'eligible_English Language and Literature', 'eligible_French Language', 'eligible_History',
        'eligible_Geography', 'eligible_Tourism and Archaeology', 'eligible_Psychology – Psychological Counseling',
        'eligible_Social Work', 'eligible_Physical Education', 'eligible_Kindergarten', 'eligible_Elementary School Teacher',
        'eligible_Jurisprudence and Legislation', 'eligible_Fundamentals of Religion', 'eligible_Islamic Banking',
        'eligible_Civil Engineering', 'eligible_Geomatic Engineering', 'eligible_Architectural Engineering',
        'eligible_Construction Engineering', 'eligible_Planning Engineering', 'eligible_Mechanical Engineering',
        'eligible_Chemical Engineering', 'eligible_Industrial Engineering', 'eligible_Computer Engineering',
        'eligible_Electrical Engineering', 'eligible_Mechatronics Engineering', 'eligible_Energy and Environmental Engineering',
        'eligible_Network Engineering', 'eligible_Computer Science', 'eligible_Computer Science in the Job Market',
        'eligible_Management Information Systems', 'eligible_Cybersecurity', 'eligible_Human Medicine',
        'eligible_Dentistry', 'eligible_Pharmacy', 'eligible_Optometry', 'eligible_Medical Laboratory Sciences',
        'eligible_Medical Imaging', 'eligible_Audiology and Speech Sciences', 'eligible_Physical Therapy',
        'eligible_Cardiopulmonary Perfusion Technology', 'eligible_Anesthesia and Resuscitation', 'eligible_Nursing',
        'eligible_Midwifery', 'eligible_Digital Economy', 'eligible_Digital Media and Communication',
        'eligible_Radio and Television', 'eligible_Public Relations', 'eligible_Accounting',
        'eligible_Business Administration', 'eligible_Financial and Banking Sciences', 'eligible_Financial Technology',
        'eligible_Digital Marketing and Communication', 'eligible_Investment and Real Estate Development',
        'eligible_Business Intelligence', 'eligible_Veterinary Medicine', 'eligible_Nutrition and Food Processing',
        'eligible_Agricultural Engineering', 'eligible_Law', 'eligible_Political Science', 'eligible_Music',
        'eligible_Interior Design', 'eligible_Painting and Drawing', 'eligible_Graphic Design', 'eligible_Ceramic Art',
        'eligible_Game Design', 'eligible_Fashion Design', 'eligible_Therapeutic Expressive Art',
        'branch_Agricultural', 'branch_Entrepreneurship and Business', 'branch_Hotel and Home Economics',
        'branch_Industrial', 'branch_Information Technology', 'branch_Literary', 'branch_Scientific',
        'branch_Sharia', 'branch_Vocational',
        'gpa_bin_60-70', 'gpa_bin_70-80', 'gpa_bin_80-90', 'gpa_bin_90-100', 'gpa_bin_<60',
        'core_avg_bin_60-70', 'core_avg_bin_70-80', 'core_avg_bin_80-90', 'core_avg_bin_90-100', 'core_avg_bin_<60'
    ]

    # --- الخطوة 2: إنشاء DataFrame جديد فارغ بنفس هيكل المودل ---
    final_student_df = pd.DataFrame(columns=expected_model_columns)
    final_student_df.loc[0] = 0.0 # استخدم 0.0 لضمان أن النوع هو float

    # --- الخطوة 3: ملء الـ DataFrame الجديد ببيانات الطالب الحالية (النهج البسيط) ---
    
    # معالجة branch
    if 'branch' in student_data_proc.columns:
        student_branch_value = student_data_proc.loc[0, 'branch']
        branch_col_name = f"branch_{student_branch_value}"
        if branch_col_name in final_student_df.columns:
            final_student_df.loc[0, branch_col_name] = 1

    # نسخ قيم كل الأعمدة المشتركة الأخرى
    for col in student_data_proc.columns:
        if col in final_student_df.columns:
            raw_val = student_data_proc.loc[0, col]
            # If the input is a boolean, convert explicitly to float to avoid dtype-incompatibility warnings.
            if isinstance(raw_val, (bool, np.bool_)):
                value = 1.0 if raw_val else 0.0
            else:
                # Attempt numeric coercion for other types (strings like "True"/"False" become NaN and will be treated as 0.0)
                value = pd.to_numeric(raw_val, errors='coerce')
            # Ensure we always assign a float (explicit cast) so column dtypes remain compatible.
            final_student_df.loc[0, col] = 0.0 if pd.isna(value) else float(value)
    # **ملاحظة:** نحن لا نقوم بحساب أعمدة eligibility و gpa_bin هنا بشكل معقد.
    # نحن فقط نضمن وجود كل الأعمدة التي يتوقعها المودل. هذا كافٍ لجعله يعمل.
    # المودل تعلم أن يتجاهل الأعمدة التي لا تهمه.

    return final_student_df


# ==============================================================================
# 2. الدالة النهائية لـ predict_probabilities (مع الطباعة التشخيصية)
# ==============================================================================
def predict_probabilities(
    models: Dict[str, Any],
    student_data: pd.DataFrame
) -> Dict[str, float]:
    """
    More robust predict_probabilities with multiple fallbacks:
    - try ensemble.predict_proba
    - try individual models' predict_proba
    - try decision_function (converted via softmax)
    - fallback to predict() -> one-hot
    Returns a mapping {class_label: probability} or {} on failure.
    """
    import logging
    logger = logging.getLogger(__name__)

    def _softmax(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        # numeric stability
        shifted = x - np.max(x)
        e = np.exp(shifted)
        denom = np.sum(e)
        if denom == 0:
            return np.ones_like(e) / float(len(e))
        return e / denom

    def _proba_from_model(m, X) -> (Optional[np.ndarray], Optional[np.ndarray]):
        """
        Try to obtain (proba_array, classes_array) from model m given X.
        Returns (proba, classes) or (None, None).
        """
        try:
            if hasattr(m, "predict_proba") and hasattr(m, "classes_"):
                p = m.predict_proba(X)
                classes = np.asarray(getattr(m, "classes_", []))
                # handle shape: p could be (n_samples, n_classes) or (n_classes,) for some models
                p_arr = np.asarray(p)
                if p_arr.ndim == 2:
                    proba = p_arr[0]
                else:
                    proba = np.ravel(p_arr)
                return proba, classes
        except Exception as e:
            logger.debug(f"predict_proba failed for model {getattr(m,'__class__',m)}: {e}", exc_info=True)

        try:
            if hasattr(m, "decision_function"):
                s = m.decision_function(X)
                s_arr = np.asarray(s)
                if s_arr.ndim == 2:
                    scores = s_arr[0]
                else:
                    scores = np.ravel(s_arr)
                # convert scores to probabilities via softmax
                proba = _softmax(scores)
                # try to obtain classes (fallback to attribute or numeric indices)
                classes = getattr(m, "classes_", None)
                if classes is None:
                    classes = np.arange(len(proba))
                return proba, np.asarray(classes)
        except Exception as e:
            logger.debug(f"decision_function failed for model {getattr(m,'__class__',m)}: {e}", exc_info=True)

        try:
            # fallback to predict -> one-hot probability
            if hasattr(m, "predict"):
                preds = m.predict(X)
                pred = preds[0] if hasattr(preds, "__len__") else preds
                classes = getattr(m, "classes_", None)
                if classes is None:
                    classes = np.array([pred])
                classes = np.asarray(classes)
                proba = np.zeros(len(classes), dtype=float)
                # try match predicted class
                try:
                    idx = int(np.where(classes == pred)[0][0])
                    proba[idx] = 1.0
                except Exception:
                    if len(proba) > 0:
                        proba[0] = 1.0
                return proba, classes
        except Exception as e:
            logger.debug(f"predict fallback failed for model {getattr(m,'__class__',m)}: {e}", exc_info=True)

        return None, None

    ensemble = models.get('ensemble')
    # If ensemble exists, prefer it but still use other models as fallback.
    candidate_order = []
    if ensemble is not None:
        candidate_order.append(('ensemble', ensemble))
    # Add other models (prefer gradient_boosting, random_forest, logistic_regression)
    for name in ('gradient_boosting', 'random_forest', 'logistic_regression'):
        m = models.get(name)
        if m is not None:
            candidate_order.append((name, m))

    if not candidate_order:
        logger.warning("No models available in predict_probabilities")
        return {}

    # Prepare X: if any model exposes feature names, align to the first one found
    expected = None
    for _, m in candidate_order:
        if hasattr(m, "feature_names_in_"):
            expected = list(getattr(m, "feature_names_in_"))
            break

    if expected is not None:
        student_data = student_data.reindex(columns=expected, fill_value=0.0)
        for c in student_data.columns:
            student_data[c] = pd.to_numeric(student_data[c], errors='coerce').fillna(0.0)
        try:
            student_data = student_data.astype(float)
        except Exception:
            pass

    # Try candidates in order
    last_error = None
    for name, m in candidate_order:
        try:
            proba, classes = _proba_from_model(m, student_data)
            if proba is None or classes is None:
                logger.debug(f"Model {name} returned no probabilities/classes, trying next.")
                continue
            # Align lengths
            proba = np.asarray(proba, dtype=float).ravel()
            classes = np.asarray(classes)
            n = min(len(proba), len(classes))
            if n == 0:
                logger.debug(f"Model {name} produced empty proba/classes.")
                continue
            proba = proba[:n]
            classes = classes[:n]
            probabilities = {str(classes[i]): float(proba[i]) for i in range(n)}
            logger.info(f"[predict_probabilities] Using model '{name}' produced top pairs: {sorted(probabilities.items(), key=lambda x:-x[1])[:5]}")
            return probabilities
        except Exception as e:
            last_error = e
            logger.exception(f"Model {name} failed during probability extraction: {e}")

    logger.warning(f"predict_probabilities: All model attempts failed. last_error={last_error}")
    return {}


# ==============================================================================
# 3. الدالة النهائية لـ calculate_program_scores (مع الطباعة التشخيصية)
# ==============================================================================
def calculate_program_scores(
    student: pd.Series,
    program_metadata_df: pd.DataFrame,
    probabilities: Dict[str, float],
    university_data: Dict
) -> pd.DataFrame:
    """
    Calculate scores for each program with debugging print statements.
    This version includes diagnostics to track the values of key scores.
    """
    scores_df = program_metadata_df.copy()
    if 'core_subjects' in scores_df.columns and scores_df['core_subjects'].dtype == 'object':
        try:
            scores_df['core_subjects'] = scores_df['core_subjects'].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
        except:
            pass
    
    student_id = student.get('student_id', 'N/A')
    student_gpa = student.get('general_average', 0.0)

    student_branch = student.get('branch', None)
    
    branches_allowed = university_data.get('branches_allowed', {})
    branch_programs = branches_allowed.get(student_branch, {}).get('allowed_programs', [])
    scores_df['branch_eligible'] = scores_df['program_name'].isin(branch_programs)
    
    scores_df['gpa_eligible'] = student_gpa >= scores_df['min_gpa_regular']
    
    scores_df['gpa_score'] = student_gpa / 100.0
    
    
    interest_cols = [col for col in student.index if col.startswith('interest_')]
    scores_df['interest_score'] = 0.0
    for col in interest_cols:
        field = col.replace('interest_', '')
        if field in scores_df['program_name'].values:
            mask = scores_df['program_name'] == field
            scores_df.loc[mask, 'interest_score'] = student[col] / 100
        else:
            for idx, row in scores_df.iterrows():
                program_name = row['program_name']
                if field in program_name or program_name in field:
                    scores_df.loc[idx, 'interest_score'] = max(
                        scores_df.loc[idx, 'interest_score'],
                        student[col] / 100 * 0.5
                    )
    
    scores_df['unemployment_score'] = 1 - (scores_df['unemployment_rate'] / 100)
    min_salary = scores_df['expected_salary'].min()
    max_salary = scores_df['expected_salary'].max()
    if max_salary > min_salary:
        scores_df['salary_score'] = (scores_df['expected_salary'] - min_salary) / (max_salary - min_salary)
    else:
        scores_df['salary_score'] = 0.5
    
    scores_df['ml_score'] = 0.0
    if probabilities:
        # Match model outputs (probabilities) to the programs in scores_df.
        # Prefer an explicit `model_label` column when present (added to program_metadata.csv)
        # which maps program -> model class label. This ensures programs that were renamed
        # or that the model expects under a different label still get their ML probability.
        def _resolve_ml_score(row):
            prog_name = row.get("program_name", "")
            model_label = row.get("model_label") or ""
            # try model_label first
            if model_label and model_label in probabilities:
                return float(probabilities.get(model_label, 0.0))
            # fallback to program_name
            if prog_name in probabilities:
                return float(probabilities.get(prog_name, 0.0))
            # last resort: try a lowercase match (defensive)
            lower_map = {k.lower(): v for k, v in probabilities.items()}
            return float(lower_map.get(str(prog_name).lower(), 0.0))
        scores_df['ml_score'] = scores_df.apply(_resolve_ml_score, axis=1).fillna(0.0)

    pre_norm_max_ml = scores_df['ml_score'].max()

    max_ml_score = scores_df['ml_score'].max()
    if max_ml_score > 0:
        scores_df['ml_score'] = scores_df['ml_score'] / max_ml_score
    
    post_norm_max_ml = scores_df['ml_score'].max()
    
    scores_df['final_score'] = (
        (scores_df['ml_score'] * 0.30) +
        (scores_df['interest_score'] * 0.25) +
        (scores_df['gpa_score'] * 0.20) +
        (scores_df['branch_eligible'].astype(float) * 0.15) +
        (scores_df['unemployment_score'] * 0.07) +
        (scores_df['salary_score'] * 0.03)
    )
    
    scores_df.loc[~scores_df['branch_eligible'] | ~scores_df['gpa_eligible'], 'final_score'] = 0.0
    scores_df = scores_df.sort_values('final_score', ascending=False)
    
    if not scores_df.empty:
        top_program_name = scores_df.head(1)['program_name'].iloc[0]
        final_score_top = scores_df.head(1)['final_score'].iloc[0]
    return scores_df




def scale_score_0_100(score_0_1: float) -> float:
    return round(float(score_0_1) * 100.0, 1)


def get_match_strength(score_100: float) -> Tuple[str, str]:
    if score_100 >= 85:
        return "Excellent", "ممتاز"
    if score_100 >= 70:
        return "Strong", "قوي"
    if score_100 >= 55:
        return "Good", "جيد"
    if score_100 >= 40:
        return "Moderate", "متوسط"
    return "Fair", "مقبول"


def _pick_reasons(explanation: Dict[str, str], lang: str = 'en', max_reasons: int = 3) -> List[str]:
    keys_priority = ['academic_fit', 'interest_match', 'career_prospects', 'eligibility_status']
    reasons = []
    for key in keys_priority:
        text = explanation.get(key)
        if not text:
            continue
        if lang == 'ar':
            arabic_prefix = {
                'academic_fit': 'الملاءمة الأكاديمية:',
                'interest_match': 'التوافق مع الاهتمامات:',
                'career_prospects': 'آفاق المهنة:',
                'eligibility_status': 'حالة الأهلية:'
            }.get(key, '')
            reasons.append(f"{arabic_prefix} {text}")
        else:
            reasons.append(text)
        if len(reasons) >= max_reasons:
            break
    return reasons


def generate_recommendations(
    student: pd.Series,
    scores_df: pd.DataFrame,
    university_data: Dict,
    top_n: int = 3,
    lang: str = 'en'
) -> List[Dict[str, Any]]:
    eligible_df = scores_df[scores_df['final_score'] > 0].copy()
    if eligible_df.empty:
        return []
    
    top_recommendations = []
    for _, program in eligible_df.head(top_n).iterrows():
        explanation = generate_explanation(student, program, university_data)
        score_100 = scale_score_0_100(program['final_score'])
        match_en, match_ar = get_match_strength(score_100)
        
        field_code = program.get('program_code') or program.get('program_id') or program.get('program_name')
        field_name = program.get('program_name', '')
        field_name_ar = program.get('program_name_ar', program.get('program_name_arabic', ''))
        
        reasons = _pick_reasons(explanation, lang=lang, max_reasons=3)
        
        recommendation = {
            'field': field_code,
            'field_name': field_name,
            'field_name_ar': field_name_ar,
            'score': score_100,
            'match_strength': match_en,
            'match_strength_ar': match_ar,
            'reasons': reasons
        }
        top_recommendations.append(recommendation)
    
    return top_recommendations


def _evaluate_subject_requirements(program: pd.Series, student: pd.Series) -> Tuple[List[Dict[str, Any]], bool]:
    subject_details = []
    all_met = True
    
    core_subjects = program.get('core_subjects', []) or []
    for item in core_subjects:
        subject = None
        required = None
        if isinstance(item, dict):
            subject = item.get('subject')
            required = item.get('min_score')
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            subject, required = item[0], item[1]
        else:
            subject = item
        
        if not subject:
            continue
        
        possible_cols = [subject, f"{subject}_score", f"score_{subject}", f"{subject}_mark"]
        actual = None
        for col in possible_cols:
            if col in student.index:
                actual = float(student.get(col, 0.0))
                break
        
        meets = None
        if required is not None and actual is not None:
            meets = actual >= float(required)
        
        if meets is False:
            all_met = False
        
        subject_details.append({
            'subject': subject,
            'required_score': float(required) if required is not None else None,
            'actual_score': actual,
            'meets_requirement': meets
        })
    
    return subject_details, all_met


def generate_academic_eligibility(
    student: pd.Series,
    program_metadata_df: pd.DataFrame,
    university_data: Dict
) -> List[Dict[str, Any]]:
    rows = []
    student_branch = student.get('branch', None)
    student_gpa = student.get('general_average', 0.0)
    branches_allowed = university_data.get('branches_allowed', {})
    branch_programs = branches_allowed.get(student_branch, {}).get('allowed_programs', [])
    
    for _, program in program_metadata_df.iterrows():
        program_name = program.get('program_name')
        field_code = program.get('program_code') or program.get('program_id') or program_name
        field_name = program_name
        field_name_ar = program.get('program_name_ar', program.get('program_name_arabic', ''))
        
        branch_ok = program_name in branch_programs
        gpa_ok = student_gpa >= program.get('min_gpa_regular', 0.0)
        subject_details, subjects_ok = _evaluate_subject_requirements(program, student)
        
        overall_eligible = bool(branch_ok and gpa_ok and subjects_ok)
        
        rows.append({
            'field': field_code,
            'field_name': field_name,
            'field_name_ar': field_name_ar,
            'eligible': overall_eligible,
            'subject_details': subject_details,
            'overall_eligible': overall_eligible
        })
    return rows


def save_recommendation(
    student_id: str,
    student_name: str,
    recommendations: List[Dict[str, Any]],
    academic_eligibility: Optional[List[Dict[str, Any]]] = None
) -> None:
    recommendation_data = {
        'student_id': student_id,
        'student_name': student_name,
        'recommendations': recommendations,
        'academic_eligibility': academic_eligibility or [],
        'timestamp': pd.Timestamp.now().isoformat()
    }
    file_path = RECS_DIR / f"student_{student_id}_recommendation.json"
    save_json(recommendation_data, file_path)


def generate_all_recommendations() -> None:
    ensure_dirs()
    set_seed(42)
    models = load_models()
    features_df, program_metadata_df, questions_config, university_data = load_data()
    
    summary_data = []
    for idx, student in features_df.iterrows():
        student_id = student.get('student_id', str(idx))
        student_name = student.get('student_name', f"Student {student_id}")
        student_data = preprocess_student_data(pd.DataFrame([student]), program_metadata_df)
        probabilities = predict_probabilities(models, student_data)
        scores_df = calculate_program_scores(student, program_metadata_df, probabilities, university_data)
        recommendations = generate_recommendations(student, scores_df, university_data, top_n=3, lang='en')
        academic_elig = generate_academic_eligibility(student, program_metadata_df, university_data)
        save_recommendation(student_id, student_name, recommendations, academic_eligibility=academic_elig)
        if recommendations:
            top_rec = recommendations[0]
            summary_data.append({
                'student_id': student_id,
                'student_name': student_name,
                'top_recommendation': top_rec['field'],
                'score': top_rec['score'],
                'actual_choice': student.get('label', 'Unknown')
            })
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(OUTPUTS_DIR / "top_recommendations.csv", index=False)


def generate_explanation(
    student: pd.Series,
    program: pd.Series,
    university_data: Dict
) -> Dict[str, str]:
    explanation = {}
    
    student_gpa = student.get('general_average', 0.0)
    min_gpa = program.get('min_gpa_regular', 0.0)
    gpa_margin = student_gpa - min_gpa
    
    if gpa_margin >= 10:
        academic_fit = f"Your GPA of {student_gpa:.1f}% is significantly higher than the minimum requirement of {min_gpa:.1f}% for this major, making you a strong academic fit."
    elif gpa_margin >= 5:
        academic_fit = f"Your GPA of {student_gpa:.1f}% comfortably exceeds the minimum requirement of {min_gpa:.1f}% for this major."
    elif gpa_margin >= 0:
        academic_fit = f"Your GPA of {student_gpa:.1f}% meets the minimum requirement of {min_gpa:.1f}% for this major."
    else:
        academic_fit = f"Your GPA of {student_gpa:.1f}% does not meet the minimum requirement of {min_gpa:.1f}% for this major."
    explanation['academic_fit'] = academic_fit
    
    interest_cols = [col for col in student.index if col.startswith('interest_')]
    program_name = program.get('program_name', '')
    faculty_name = program.get('faculty_name', '')
    
    relevant_interests = []
    for col in interest_cols:
        field = col.replace('interest_', '')
        if (field in program_name or program_name in field or 
            field in faculty_name or faculty_name in field):
            interest_score = student.get(col, 0.0)
            if interest_score > 50:
                relevant_interests.append((field, interest_score))
    if relevant_interests:
        relevant_interests.sort(key=lambda x: x[1], reverse=True)
        top_interests = relevant_interests[:2]
        interest_text = ", ".join([f"{field} ({score:.0f}%)" for field, score in top_interests])
        interest_match = f"Your interests align strongly with this field, particularly in {interest_text}."
    else:
        interest_match = "This program offers valuable career opportunities that may align with your broader interests."
    explanation['interest_match'] = interest_match
    
    unemployment_rate = program.get('unemployment_rate', 0.0)
    expected_salary = program.get('expected_salary', 0.0)
    future_jobs = program.get('future_jobs', '')
    career_prospects = f"Graduates of this major can pursue opportunities in {future_jobs}. The job market has an unemployment rate of {unemployment_rate}% and an expected salary of {expected_salary} NIS."
    explanation['career_prospects'] = career_prospects
    
    student_branch = student.get('branch', None)
    branches_allowed = university_data.get('branches_allowed', {})
    branch_programs = branches_allowed.get(student_branch, {}).get('allowed_programs', [])
    branch_eligible = program.get('program_name') in branch_programs
    gpa_eligible = student_gpa >= min_gpa
    if branch_eligible and gpa_eligible:
        eligibility_status = "Eligible"
    else:
        eligibility_status = "Not eligible"
    explanation['eligibility_status'] = eligibility_status
    
    return explanation


if __name__ == "__main__":
    generate_all_recommendations()
