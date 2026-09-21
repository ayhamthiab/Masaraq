"""
Feature engineering module for the Masarak University Recommendation System.

This module handles the creation of features for machine learning models.
"""
import re
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


def load_processed_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the processed data files.
    
    Returns:
        Tuple containing:
        - students_df: DataFrame with cleaned student data
        - question_metadata_df: DataFrame with question metadata
        - program_metadata_df: DataFrame with program metadata
        - branch_eligibility_df: DataFrame with branch eligibility information
    """
    students_df = pd.read_csv(PROC_DIR / "cleaned_students.csv")
    question_metadata_df = pd.read_csv(PROC_DIR / "question_metadata.csv")
    program_metadata_df = pd.read_csv(PROC_DIR / "program_metadata.csv")
    branch_eligibility_df = pd.read_csv(PROC_DIR / "branch_eligibility.csv")
    
    return students_df, question_metadata_df, program_metadata_df, branch_eligibility_df


def _slugify_ar(text: str) -> str:
    """Slug آمن للأعمدة من نص عربي/إنجليزي."""
    if text is None:
        return "na"
    s = str(text).strip().lower()
    # أرقام عربية -> عادية
    s = s.translate(str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789'))
    # استبدال أي شيء غير حرف/رقم/مسافة بشرطة سفلية
    s = re.sub(r'[^\w\s\u0600-\u06FF]', '_', s)
    # مسافات -> _
    s = re.sub(r'\s+', '_', s)
    # تقليل التكرارات
    s = re.sub(r'_+', '_', s).strip('_')
    return s or "na"


def _extract_questions_index_and_options(questions_config: Dict) -> Dict[str, List[str]]:
    """
    يرجّع قاموس: qid -> قائمة الخيارات بالنص (بالترتيب) لأسئلة MCQ.
    نفترض أن الترميز العددي في cleaned_students كان 1..N حسب نفس ترتيب هذه القائمة.
    """
    qid_to_options: Dict[str, List[str]] = {}
    for _, cat in (questions_config or {}).items():
        for q in cat.get('questions', []):
            qid = str(q.get('id') or q.get('question_id') or '').strip().lower()
            if not qid:
                continue
            qtype = (q.get('type') or '').strip().lower()
            if qtype in ('multiple_choice', 'single', 'mcq'):
                # جلب الخيارات من أحد الحقول الشائعة
                opts = None
                if isinstance(q.get('options'), list):
                    opts = [ (o if isinstance(o, str) else (o.get('text') or o.get('label') or o.get('name') or o.get('option') or '')) for o in q['options'] ]
                elif isinstance(q.get('answers'), list):
                    opts = [ (o if isinstance(o, str) else (o.get('text') or o.get('label') or o.get('name') or o.get('option') or '')) for o in q['answers'] ]
                elif isinstance(q.get('choices_with_scores'), list):
                    opts = [ (o if isinstance(o, str) else (o.get('text') or o.get('label') or o.get('name') or o.get('option') or '')) for o in q['choices_with_scores'] ]
                elif isinstance(q.get('choices'), list):
                    opts = [ (o if isinstance(o, str) else str(o)) for o in q['choices'] ]
                if opts:
                    qid_to_options[qid] = [str(x).strip() for x in opts]
    return qid_to_options


def generate_academic_features(students_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate academic features from student data.
    
    Args:
        students_df: DataFrame with student data
    
    Returns:
        DataFrame with academic features
    """
    # Make a copy to avoid modifying the original
    academic_df = students_df.copy()
    
    # GPA features
    if 'general_average' in academic_df.columns:
        # Create GPA bins
        academic_df['gpa_bin'] = pd.cut(
            academic_df['general_average'],
            bins=[0, 60, 70, 80, 90, 100],
            labels=['<60', '60-70', '70-80', '80-90', '90-100']
        )
        
        # Create GPA percentile
        academic_df['gpa_percentile'] = academic_df['general_average'].rank(pct=True) * 100
    
    # Core subject average features
    if 'core_subject_average' in academic_df.columns:
        # Create core subject average bins
        academic_df['core_avg_bin'] = pd.cut(
            academic_df['core_subject_average'],
            bins=[0, 60, 70, 80, 90, 100],
            labels=['<60', '60-70', '70-80', '80-90', '90-100']
        )
        
        # Create core subject average percentile
        academic_df['core_avg_percentile'] = academic_df['core_subject_average'].rank(pct=True) * 100
    
    # Calculate GPA to core subject average ratio
    if 'general_average' in academic_df.columns and 'core_subject_average' in academic_df.columns:
        academic_df['gpa_to_core_ratio'] = academic_df['general_average'] / academic_df['core_subject_average']
    
    return academic_df


def generate_interest_features(
    students_df: pd.DataFrame, 
    question_metadata_df: pd.DataFrame
) -> pd.DataFrame:
    """
    توليد ميزات الاهتمام + One-Hot لأسئلة MCQ.
    - Likert: تُستخدم لحساب interest_* (0..100).
    - MCQ: ننشئ أعمدة one-hot لكل خيار.
      وإذا وُجد PROC_DIR/mcq_option_weights.json نضيف مساهمتها إلى interest_* بحسب الأوزان.
      (يدعم مفاتيح أوزان رقمية '1'..'N' أو نصية للخيار)
    """
    interest_df = students_df.copy()

    # --- قراءة questions_config الخام المخزّنة في PROC_DIR لأخذ نصوص الخيارات ---
    qcfg_path = PROC_DIR / "questions_config.json"
    questions_config = load_json(qcfg_path)

    # تحويل field_weights إلى dict إن كانت نص
    if 'field_weights' in question_metadata_df.columns and question_metadata_df['field_weights'].dtype == 'object':
        question_metadata_df = question_metadata_df.copy()
        question_metadata_df['field_weights'] = question_metadata_df['field_weights'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )

    # تجميع قائمة الأهداف (حقول/برامج) من الميتاداتا ومن أوزان MCQ
    all_interest_targets = set()
    for d in question_metadata_df['field_weights']:
        if isinstance(d, dict):
            all_interest_targets.update(d.keys())

    # تحميل أوزان الـ MCQ (إن وُجدت)
    mcq_weights_path = PROC_DIR / "mcq_option_weights.json"
    try:
        mcq_option_weights = load_json(mcq_weights_path) if mcq_weights_path.exists() else {}
    except Exception:
        mcq_option_weights = {}

    # ضمّ أهداف الاهتمام الآتية من أوزان الـ MCQ (عادة أسماء برامج موسّعة)
    for qid, code_map in (mcq_option_weights or {}).items():
        for code, target_dict in (code_map or {}).items():
            if isinstance(target_dict, dict):
                all_interest_targets.update(target_dict.keys())

    # إنشاء أعمدة interest_* مبدئيًا
    for tgt in sorted(all_interest_targets):
        interest_df[f'interest_{tgt}'] = 0.0

    # --- 1) مساهمة أسئلة Likert في interest_* ---
    qmeta_types = question_metadata_df['type'].astype(str).str.lower()
    likert_meta = question_metadata_df[qmeta_types == 'likert']
    for _, row in likert_meta.iterrows():
        qid = str(row['question_id']).strip()
        cat_w = float(row.get('category_weight', 1.0) or 1.0)
        fweights = row['field_weights'] if isinstance(row['field_weights'], dict) else {}
        if qid in interest_df.columns and fweights:
            # 1..5 -> 0..1
            resp = interest_df[qid].clip(1, 5)
            normalized = (resp - 1) / 4.0
            for tgt, w in fweights.items():
                col = f'interest_{tgt}'
                if col in interest_df.columns:
                    interest_df[col] += normalized * float(w) * cat_w

    # --- 2) MCQ: one-hot + مساهمة الأوزان (إن وجدت) ---
    qid_to_options = _extract_questions_index_and_options(questions_config)
    mcq_meta = question_metadata_df[qmeta_types.isin(['multiple_choice', 'single', 'mcq'])]

    for _, row in mcq_meta.iterrows():
        qid = str(row['question_id']).strip().lower()
        if qid not in interest_df.columns:
            continue

        # خيارات هذا السؤال بالترتيب (1..N) لمواءمة القيم
        options = qid_to_options.get(qid, [])
        nopts = len(options)

        # تنظيف قيَم خارج المجال 1..N (احتياطيًا)
        if nopts > 0:
            valid_set = set(range(1, nopts + 1))
            col_series = pd.to_numeric(interest_df[qid], errors='coerce')
            # قيَم غير صالحة -> NaN
            col_series = col_series.where(col_series.isin(valid_set))
            # تعبئة الـ NaN: بالمود إن توفر، وإلا 1
            mode_val = col_series.mode(dropna=True)
            fillv = int(mode_val.iloc[0]) if not mode_val.empty else 1
            interest_df[qid] = col_series.fillna(fillv).astype(int)

        # one-hot لكل خيار (للاستخدام التحليلي والـ scaffolding)
        if options:
            for idx, opt_text in enumerate(options, start=1):
                slug = _slugify_ar(opt_text)
                oh_col = f"mcq_{qid}__{slug}"
                interest_df[oh_col] = (interest_df[qid] == idx).astype(int)

                # مساهمة الأوزان:
                # لو الملف رقمي: نقرأ بالـ idx ("1","2",...).
                # لو الملف نصي: نقرأ بالنص مباشرة.
                w_by_target = None
                if qid in mcq_option_weights:
                    w_by_target = (mcq_option_weights[qid].get(str(idx)) or
                                   mcq_option_weights[qid].get(opt_text))
                if isinstance(w_by_target, dict):
                    cat_w = float(row.get('category_weight', 1.0) or 1.0)
                    for tgt, w in w_by_target.items():
                        col_interest = f'interest_{tgt}'
                        if col_interest in interest_df.columns:
                            interest_df[col_interest] += interest_df[oh_col] * float(w) * cat_w
        else:
            # لو ما عرفنا نصوص الخيارات: one-hot بأسماء عامة opt_k
            vals = sorted(v for v in interest_df[qid].dropna().unique().tolist() if isinstance(v, (int, float)))
            for idx in vals:
                oh_col = f"mcq_{qid}__opt_{int(idx)}"
                interest_df[oh_col] = (interest_df[qid] == idx).astype(int)
                w_by_target = mcq_option_weights.get(qid, {}).get(str(int(idx)))
                if isinstance(w_by_target, dict):
                    cat_w = float(row.get('category_weight', 1.0) or 1.0)
                    for tgt, w in w_by_target.items():
                        col_interest = f'interest_{tgt}'
                        if col_interest in interest_df.columns:
                            interest_df[col_interest] += interest_df[oh_col] * float(w) * cat_w

    # --- 3) تطبيع interest_* إلى 0..100 ---
    interest_cols = [c for c in interest_df.columns if c.startswith('interest_')]
    for col in interest_cols:
        mx = interest_df[col].max()
        if pd.notna(mx) and mx > 0:
            interest_df[col] = (interest_df[col] / mx) * 100.0

    # --- 4) أفضل 3 اهتمامات (مرجعي فقط) ---
    if interest_cols:
        interest_df['top_interests'] = interest_df[interest_cols].apply(
            lambda x: ','.join(x.nlargest(3).index.str.replace('interest_', '')),
            axis=1
        )

    return interest_df


def generate_eligibility_features(
    students_df: pd.DataFrame,
    program_metadata_df: pd.DataFrame,
    branch_eligibility_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate eligibility features based on university requirements (vectorized & fast).

    المنطق:
    - مؤهل إذا كان البرنامج مسموح لفرع التوجيهي + (GPA >= حد القبول العادي أو الموازي).
    - ينشئ أعمدة boolean: eligible_<program_name>
    - يحسب num_eligible_programs = عدد البرامج المؤهَّل لها الطالب.
    """
    eligibility_df = students_df.copy()

    # حضّر قاموس: فرع -> مجموعة البرامج المسموح بها
    branch_to_programs = {
        b: set(g["program_name"]) for b, g in branch_eligibility_df.groupby("branch")
    }

    # سلاسل مساعدة
    branches = eligibility_df["branch"]
    gpa = eligibility_df["general_average"]

    # لكل برنامج: عمود مؤهل/غير مؤهل
    for _, prog in program_metadata_df.iterrows():
        prog_name = prog["program_name"]
        min_reg = float(prog.get("min_gpa_regular", 0))
        min_par = float(prog.get("min_gpa_parallel", 0))
        col = f"eligible_{prog_name}"

        # صلاحية الفرع
        branch_ok = branches.map(lambda b: prog_name in branch_to_programs.get(b, set()))
        # شرط الـ GPA (عادي أو موازي)
        gpa_ok = (gpa >= min_reg) | (gpa >= min_par)

        eligibility_df[col] = branch_ok & gpa_ok

    # عدد البرامج المؤهَّل لها الطالب
    elig_cols = [c for c in eligibility_df.columns if c.startswith("eligible_")]
    if elig_cols:
        eligibility_df["num_eligible_programs"] = eligibility_df[elig_cols].sum(axis=1)

    return eligibility_df


def merge_features(
    academic_df: pd.DataFrame,
    interest_df: pd.DataFrame,
    eligibility_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge all feature DataFrames into a single DataFrame.
    
    Args:
        academic_df: DataFrame with academic features
        interest_df: DataFrame with interest features
        eligibility_df: DataFrame with eligibility features
    
    Returns:
        Merged DataFrame with all features
    """
    # Start with academic features
    merged_df = academic_df.copy()
    
    # Add interest features
    interest_cols = [col for col in interest_df.columns if col.startswith('interest_') or col == 'top_interests']
    merged_df = pd.concat([merged_df, interest_df[interest_cols]], axis=1)
    
    # Add eligibility features
    eligibility_cols = [col for col in eligibility_df.columns if col.startswith('eligible_') or col == 'num_eligible_programs']
    merged_df = pd.concat([merged_df, eligibility_df[eligibility_cols]], axis=1)
    
    return merged_df


def prepare_ml_dataset(
    features_df: pd.DataFrame,
    program_metadata_df: pd.DataFrame
) -> pd.DataFrame:
    ml_df = features_df.copy()

    if 'label' in ml_df.columns:
        valid_programs = set(program_metadata_df['program_name'])
        ml_df.loc[~ml_df['label'].isin(valid_programs), 'label'] = np.nan
        ml_df = ml_df.dropna(subset=['label'])

    # One-hot للأعمدة الفئوية الكبيرة فقط (اللي بعدها مفيدة)
    categorical_cols = ['branch', 'gpa_bin', 'core_avg_bin']
    for col in categorical_cols:
        if col in ml_df.columns:
            dummies = pd.get_dummies(ml_df[col], prefix=col)
            ml_df = pd.concat([ml_df, dummies], axis=1)
            ml_df = ml_df.drop(col, axis=1)

    # أعمدة نصية مرجعية لا تدخل للـ ML مباشرة
    drop_cols = []
    if 'top_interests' in ml_df.columns:
        drop_cols.append('top_interests')

    if drop_cols:
        ml_df = ml_df.drop(columns=drop_cols)

    return ml_df


def engineer_features() -> None:
    """
    Main function to engineer features.
    
    This function:
    1. Loads the processed data
    2. Generates academic features
    3. Generates interest features
    4. Generates eligibility features
    5. Merges all features
    6. Prepares the final dataset for machine learning
    7. Saves the feature-engineered data
    """
    # Ensure directories exist
    ensure_dirs()
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Load processed data
    print("Loading processed data...")
    students_df, question_metadata_df, program_metadata_df, branch_eligibility_df = load_processed_data()
    
    # Generate features
    print("Generating academic features...")
    academic_df = generate_academic_features(students_df)
    
    print("Generating interest features...")
    interest_df = generate_interest_features(students_df, question_metadata_df)
    
    print("Generating eligibility features...")
    eligibility_df = generate_eligibility_features(students_df, program_metadata_df, branch_eligibility_df)
    
    # Merge features
    print("Merging features...")
    features_df = merge_features(academic_df, interest_df, eligibility_df)
    
    # Prepare ML dataset
    print("Preparing ML dataset...")
    ml_df = prepare_ml_dataset(features_df, program_metadata_df)
    
    # Save feature-engineered data
    print("Saving feature-engineered data...")
    features_df.to_csv(PROC_DIR / "students_features.csv", index=False)
    ml_df.to_csv(PROC_DIR / "merged_dataset.csv", index=False)
    
    # Save intermediate data for debugging
    academic_df.to_csv(INTERIM_DIR / "academic_features.csv", index=False)
    interest_df.to_csv(INTERIM_DIR / "interest_features.csv", index=False)
    eligibility_df.to_csv(INTERIM_DIR / "eligibility_features.csv", index=False)
    
    print("Feature engineering completed successfully!")


if __name__ == "__main__":
    engineer_features()
