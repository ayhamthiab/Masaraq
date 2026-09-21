"""
Data preparation module for the Masarak University Recommendation System.

This module handles loading, cleaning, and preprocessing the raw data files.
"""
import re
import os
import pandas as pd
import numpy as np
from pathlib import Path
import json
import joblib
from typing import Dict, List, Tuple, Any, Optional, Iterable

from utils import (
    ROOT, DATA_DIR, RAW_DIR, PROC_DIR, INTERIM_DIR, 
    MODELS_DIR, OUTPUTS_DIR, RECS_DIR,
    ensure_dirs, set_seed, load_json, save_json, 
    save_model, load_model, find_column
)


def load_raw_data() -> Tuple[pd.DataFrame, Dict, Dict]:
    """
    Load the raw data files.
    
    Returns:
        Tuple containing:
        - students_df: DataFrame with student data
        - questions_config: Dictionary with question configuration
        - university_data: Dictionary with university and program data
    """
    # Load students data
    students_file = RAW_DIR / "students_data.csv"
    students_df = pd.read_csv(students_file)
    
    # Load questions configuration
    questions_file = RAW_DIR / "questions_config.json"
    questions_config = load_json(questions_file)
    
    # Load university data
    university_file = RAW_DIR / "university_data.json"
    university_data = load_json(university_file)
    
    return students_df, questions_config, university_data

def _normalize_ar_text(s: str) -> str:
    """تطبيع عربي قوي ومتحفّظ لتقليل فوارق الكتابة دون إفساد المعنى."""
    if s is None:
        return ""
    s = str(s)

    # إزالة التشكيل
    s = re.sub(r'[\u0617-\u061A\u064B-\u0652]', '', s)

    # إزالة التطويل
    s = s.replace('ـ', '')

    # توحيد الألفات
    s = s.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')

    # توحيد الياء/الألف المقصورة
    s = s.replace('ى', 'ي')

    # أرقام عربية -> هندية (اختياري) أو العكس؛ هنا نحو أرقام عربية عادية
    s = s.translate(str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789'))

    # توحيد مسافات وعلامات بسيطة
    s = re.sub(r'[^\w\s\u0600-\u06FF]', ' ', s)  # نحذف الرموز غير-حروف/أرقام
    s = re.sub(r'\s+', ' ', s).strip().lower()
    return s

def _build_answer_mapping_for_question(q_obj: dict) -> dict:
    """
    يبني mapping نص->رقم. لليكرت: 1..5 بنفس ترتيب الخيارات في الكونفيج.
    للاختيارات المتعددة: إن لم توجد قيمة رقمية صريحة، نرقّم 1..N (ثابت لكل سؤال).
    """
    # 1) mapping صريح
    if isinstance(q_obj.get('mapping'), dict):
        return { _normalize_ar_text(k): int(v) for k, v in q_obj['mapping'].items() }

    # 2) options/answers بصيغة قائمة
    options = q_obj.get('options') or q_obj.get('answers') or q_obj.get('choices_with_scores')
    if isinstance(options, list) and options:
        out = {}
        for idx, opt in enumerate(options, start=1):
            if isinstance(opt, str):
                out[_normalize_ar_text(opt)] = idx
            else:
                text = opt.get('text') or opt.get('label') or opt.get('name') or opt.get('option') or ''
                numeric = opt.get('value') or opt.get('score') or opt.get('numeric')  # قد تكون معرفة
                out[_normalize_ar_text(text)] = int(numeric if numeric is not None else idx)
        return out

    # 3) choices نصوص بسيطة
    choices = q_obj.get('choices')
    if isinstance(choices, list) and choices:
        return { _normalize_ar_text(t): i for i, t in enumerate(choices, start=1) }

    return {}

def clean_students_data(df: pd.DataFrame, questions_config: Dict) -> pd.DataFrame:
    """
    تنظيف الداتا + تحويل نصوص الأسئلة لأرقام بناءً على نوع السؤال في الكونفيج.
    - likert: 1..5 وبعدين تعويض بالوسيط (ثم cast إلى int)
    - multiple_choice: ترميز ثابت من الكونفيج/الترتيب ثم تعويض بالـ mode
    """
    cleaned_df = df.copy()

    # student_id كنص
    if 'student_id' in cleaned_df.columns:
        cleaned_df['student_id'] = cleaned_df['student_id'].astype(str)

    # numeric basics
    for col in ['general_average', 'core_subject_average']:
        if col in cleaned_df.columns:
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
            med = cleaned_df[col].median()
            cleaned_df[col] = cleaned_df[col].fillna(med)

    # --- بناء ميتاداتا للأسئلة: نوع + mapping ---
    q_meta: Dict[str, Dict[str, Any]] = {}
    for _, cat in (questions_config or {}).items():
        for q in cat.get('questions', []):
            qid = (q.get('id') or q.get('question_id'))
            if not qid:
                continue
            qid_l = str(qid).lower()
            q_type = (q.get('type') or '').strip().lower()  # 'likert' أو 'multiple_choice' ...
            mapping = _build_answer_mapping_for_question(q)
            q_meta[qid_l] = {'type': q_type, 'mapping': mapping}

    # أعمدة الأسئلة
    q_cols = [c for c in cleaned_df.columns if c.startswith('q') and c[1:].isdigit()]

    for col in q_cols:
        key = col.lower()
        meta = q_meta.get(key, {})
        q_type = meta.get('type', '')
        mapping: Dict[str, int] = meta.get('mapping', {}) or {}

        # محاولات تحويل:
        # 1) أرقام مباشرة
        as_num = pd.to_numeric(cleaned_df[col], errors='coerce')

        # 2) من النص باستخدام التطبيع + mapping
        if mapping:
            norm_txt = cleaned_df[col].astype(str).map(_normalize_ar_text)
            mapped = norm_txt.map(mapping)
            # املأ الأرقام المفقودة بما تم تحويله من النص
            col_vals = as_num.where(~as_num.isna(), mapped)
        else:
            col_vals = as_num

        # الآن نعوّض حسب النوع
        if q_type == 'likert':
            # الوسيط منطقي لليكرت
            med = pd.to_numeric(col_vals, errors='coerce').median()
            if pd.isna(med):
                med = 3  # fallback محايد
            # cast إلى 1..5
            filled = pd.to_numeric(col_vals, errors='coerce').fillna(med)
            filled = filled.clip(lower=1, upper=5).round().astype(int)
            cleaned_df[col] = filled

        elif q_type in ('multiple_choice', 'single', 'mcq'):
            # --- التعديل: ترميز دائم عبر mapping المستمد من ترتيب الكونفيج ---
            if mapping:
                # حوّل دائماً عبر الـ mapping (حسب ترتيب الخيارات)،
                # ولو كانت القيمة أصلاً كود (1..N) خلّيها كما هي.
                valid_codes = set(mapping.values())

                def encode_mcq(v):
                    if pd.isna(v):
                        return np.nan
                    s = _normalize_ar_text(str(v))
                    if s.isdigit() and int(s) in valid_codes:
                        # القيمة أصلاً كود جاهز 1..N
                        return int(s)
                    # ترميز من النص إلى الكود حسب ترتيب الكونفيج
                    return mapping.get(s, np.nan)

                enc = cleaned_df[col].apply(encode_mcq)
                mode_vals = enc.mode(dropna=True)
                fillv = int(mode_vals.iloc[0]) if not mode_vals.empty else int(list(mapping.values())[0])
                cleaned_df[col] = enc.fillna(fillv).astype(int)
            else:
                # بدون mapping معروف: حاول تحويل رقمي فقط (وقد تبقى NaN إن لم تكن أرقام)
                cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')

        else:
            # نوع غير مذكور: طبّق سياسة عامة ذكية
            if mapping:  # اعتبره اسمي
                series_num = pd.to_numeric(col_vals, errors='coerce')
                mode_vals = series_num.mode(dropna=True)
                fillv = int(mode_vals.iloc[0]) if not mode_vals.empty else int(list(mapping.values())[0])
                cleaned_df[col] = series_num.fillna(fillv).astype(int)
            else:
                # اتركها رقمية كما وصلت (قد تحتوي NaN ليتم التعامل معها لاحقًا)
                cleaned_df[col] = col_vals

    # ضبط فرع التوجيهي وفق قائمة الفروع في university_data.json
    if 'branch' in cleaned_df.columns:
        university_file = RAW_DIR / "university_data.json"
        university_data = load_json(university_file)
        valid_branches = list(university_data.get('branches_allowed', {}).keys())
        if valid_branches:
            most_common_branch = cleaned_df['branch'].mode(dropna=True)
            if not most_common_branch.empty:
                most_common_branch = most_common_branch.iloc[0]
                cleaned_df.loc[~cleaned_df['branch'].isin(valid_branches), 'branch'] = most_common_branch

    return cleaned_df


def extract_question_metadata(questions_config: Dict) -> pd.DataFrame:
    """
    Extract metadata about questions from the questions configuration.
    
    Args:
        questions_config: Dictionary with question configuration
    
    Returns:
        DataFrame with question metadata
    """
    question_data = []
    
    for category, category_data in questions_config.items():
        category_weight = category_data.get('weight', 0)
        
        for question in category_data.get('questions', []):
            q_id = question.get('id', '')
            q_text = question.get('text', '')
            q_type = question.get('type', '')
            field_weights = question.get('field_weights', {})
            
            question_data.append({
                'question_id': q_id,
                'text': q_text,
                'type': q_type,
                'category': category,
                'category_weight': category_weight,
                'field_weights': field_weights
            })
    
    return pd.DataFrame(question_data)


def extract_program_metadata(university_data: Dict) -> pd.DataFrame:
    """
    Extract metadata about university programs.
    
    Args:
        university_data: Dictionary with university and program data
    
    Returns:
        DataFrame with program metadata
    """
    program_data = []
    
    for faculty in university_data.get('faculties', []):
        faculty_name = faculty.get('name', '')
        faculty_min_gpa_regular = faculty.get('minimum_gpa_regular', 0)
        faculty_min_gpa_parallel = faculty.get('minimum_gpa_parallel', 0)
        
        for program in faculty.get('programs', []):
            program_name = program.get('name', '')
            core_subjects = program.get('core_subjects', [])
            min_gpa_regular = program.get('minimum_gpa_regular', faculty_min_gpa_regular)
            min_gpa_parallel = program.get('minimum_gpa_parallel', faculty_min_gpa_parallel)
            unemployment_rate = program.get('unemployment_rate', 0)
            expected_salary = program.get('expected_salary', 0)
            future_jobs = program.get('future_jobs', '')
            description = program.get('description', '')
            
            program_data.append({
                'program_name': program_name,
                'faculty_name': faculty_name,
                'core_subjects': core_subjects,
                'min_gpa_regular': min_gpa_regular,
                'min_gpa_parallel': min_gpa_parallel,
                'unemployment_rate': unemployment_rate,
                'expected_salary': expected_salary,
                'future_jobs': future_jobs,
                'description': description
            })
    
    return pd.DataFrame(program_data)


def extract_branch_eligibility(university_data: Dict) -> pd.DataFrame:
    """
    Extract information about which branches are eligible for which programs.
    
    Args:
        university_data: Dictionary with university and program data
    
    Returns:
        DataFrame with branch eligibility information
    """
    eligibility_data = []
    
    branches_allowed = university_data.get('branches_allowed', {})
    for branch, branch_data in branches_allowed.items():
        allowed_programs = branch_data.get('allowed_programs', [])
        
        for program in allowed_programs:
            eligibility_data.append({
                'branch': branch,
                'program_name': program,
                'is_eligible': True
            })
    
    return pd.DataFrame(eligibility_data)


def prepare_data() -> None:
    """
    Main function to prepare the data.
    
    This function:
    1. Loads the raw data
    2. Cleans and preprocesses the data
    3. Extracts metadata
    4. Saves the processed data
    """
    # Ensure directories exist
    ensure_dirs()
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Load raw data
    print("Loading raw data...")
    students_df, questions_config, university_data = load_raw_data()
    
    # Clean students data
    print("Cleaning students data...")
    cleaned_students_df = clean_students_data(students_df, questions_config)
    
    # Extract metadata
    print("Extracting question metadata...")
    question_metadata_df = extract_question_metadata(questions_config)
    
    print("Extracting program metadata...")
    program_metadata_df = extract_program_metadata(university_data)
    
    print("Extracting branch eligibility information...")
    branch_eligibility_df = extract_branch_eligibility(university_data)
    
    # Save processed data
    print("Saving processed data...")
    cleaned_students_df.to_csv(PROC_DIR / "cleaned_students.csv", index=False)
    question_metadata_df.to_csv(PROC_DIR / "question_metadata.csv", index=False)
    program_metadata_df.to_csv(PROC_DIR / "program_metadata.csv", index=False)
    branch_eligibility_df.to_csv(PROC_DIR / "branch_eligibility.csv", index=False)
    
    # Save original data in a more accessible format
    save_json(questions_config, PROC_DIR / "questions_config.json")
    save_json(university_data, PROC_DIR / "university_data.json")
    
    print("Data preparation completed successfully!")


if __name__ == "__main__":
    prepare_data()
