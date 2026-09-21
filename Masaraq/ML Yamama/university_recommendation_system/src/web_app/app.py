import os
import csv
import json
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for, session
from werkzeug.utils import secure_filename
import pandas as pd
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import joblib
import re
import numpy as np
import uuid
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import Config

try:
    from ..ocr_module.diploma_scanner import DiplomaScanner
    from ..utils.interest_assessment import InterestAssessment
except (ImportError, ValueError):
    from src.ocr_module.diploma_scanner import DiplomaScanner
    from src.utils.interest_assessment import InterestAssessment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app_config = Config()
app.config['SECRET_KEY'] = app_config.SECRET_KEY
app.config['UPLOAD_FOLDER'] = app_config.UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = app_config.MAX_CONTENT_LENGTH
app.config['ALLOWED_EXTENSIONS'] = app_config.ALLOWED_EXTENSIONS
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]  
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


rf_model = joblib.load(os.path.join(MODELS_DIR, "random_forest_model.pkl"))
gb_model = joblib.load(os.path.join(MODELS_DIR, "gradient_boosting_model.pkl"))
lr_model = joblib.load(os.path.join(MODELS_DIR, "logistic_regression_model.pkl"))

try:
    diploma_scanner = DiplomaScanner()
    ia = InterestAssessment()
    logger.info("Modules initialized successfully")
except Exception as e:
    logger.error(f"Error initializing modules: {e}", exc_info=True)
    diploma_scanner = DiplomaScanner()
    ia = InterestAssessment()

CSV_DIR = "data/processed"
CSV_PATH = os.path.join(CSV_DIR, "interest_records.csv")

def ensure_csv_header(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    header = ["branch", "general_average", "core_subject_average"] + [f"q{i}" for i in range(1, 38)]
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)

def append_csv_record(path: str, row: list):
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)

_DEFAULT_EN = {
  "gpa_strong": "Your GPA ({gpa}%) is well above the requirement, showing strong readiness.",
  "gpa_borderline": "Your GPA ({gpa}%) is just around the requirement — with effort, you can succeed here.",
  "interest_high": "This aligns with your passion in {interest_area} — imagine studying what excites you every day!",
  "weakness_compensation": "Although your {weak_subject} score is a bit lower, your strength in {strong_subject} makes you a great fit.",
  "career_opportunities": "This major opens career paths in {career_paths}. Current job market: unemployment {unemployment}%, avg salary {salary} NIS.",
  "final_encouragement": "We believe you can shine in this path!"
}
_DEFAULT_AR = {
  "gpa_strong": "معدلك ({gpa}%) أعلى بكثير من المطلوب، وهذا يثبت جاهزيتك الأكاديمية.",
  "gpa_borderline": "معدلك ({gpa}%) قريب من الحد المطلوب — بقليل من الجهد ستتفوق هنا.",
  "interest_high": "هذا التخصص يتوافق مع شغفك في {interest_area} — تخيل أنك تدرس ما تحبه كل يوم!",
  "weakness_compensation": "رغم أن نتيجتك في {weak_subject} أقل قليلاً، إلا أن قوتك في {strong_subject} تجعل منك مرشحاً مميزاً.",
  "career_opportunities": "هذا التخصص يفتح لك مجالات عمل في {career_paths}. سوق العمل الحالي: بطالة {unemployment}%، ومتوسط الراتب {salary} شيكل.",
  "final_encouragement": "نثق أنك ستتألق في هذا المسار!"
}

def _load_or_create_templates_safe():
    """
    Load templates from the central processed data directory (src/data/processed).
    Falls back to bundled defaults. If the processed files are missing, write
    default templates there so they become the single source of truth.
    """
    try:
        # central processed directory (project root / data / processed)
        proc_dir = Path(__file__).resolve().parents[2] / "data" / "processed"
        en_path = proc_dir / "reasons_templates_en.json"
        ar_path = proc_dir / "reasons_templates_ar.json"

        en = _DEFAULT_EN
        ar = _DEFAULT_AR

        # load if present
        if en_path.exists():
            try:
                en = json.loads(en_path.read_text(encoding="utf-8"))
            except Exception:
                en = _DEFAULT_EN
        if ar_path.exists():
            try:
                ar = json.loads(ar_path.read_text(encoding="utf-8"))
            except Exception:
                ar = _DEFAULT_AR

        # ensure files exist in processed dir (write defaults if missing)
        try:
            proc_dir.mkdir(parents=True, exist_ok=True)
            if not en_path.exists():
                en_path.write_text(json.dumps(_DEFAULT_EN, ensure_ascii=False, indent=2), encoding="utf-8")
            if not ar_path.exists():
                ar_path.write_text(json.dumps(_DEFAULT_AR, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            # ignore write failures, keep defaults in memory
            pass

        return en, ar
    except Exception:
        return _DEFAULT_EN, _DEFAULT_AR


class _SeriesLike:
    """التعامل مع dict كأنه Pandas Series (get/index) لتوافق الدوال."""
    def __init__(self, d): self._d = dict(d or {})
    @property
    def index(self): return self._d.keys()
    def get(self, k, default=None): return self._d.get(k, default)


def _find_program_interest_score(student, program_name: str, program_row: dict):
    """
    تختار أفضل interest_XXX متوافق اسميًا مع اسم البرنامج/الكلية.
    ترجع (اسم مجال الاهتمام, النسبة).
    """
    interest_cols = [c for c in student.index if str(c).startswith("interest_")]
    best_area, best_score = None, 0.0
    for col in interest_cols:
        area = str(col).replace("interest_", "")
        try:
            score = float(student.get(col, 0.0))
        except:
            score = 0.0
        if area.lower() in str(program_name).lower():
            if score > best_score: best_area, best_score = area, score
        elif area.lower() in str(program_row.get("faculty_name", "")).lower():
            if score > best_score: best_area, best_score = area, score
        else:
            if score > best_score: best_area, best_score = area, score
    return best_area, best_score


def _find_weak_and_supporting_subjects(program_row: dict, student) -> tuple:
    """
    تبحث عن مادة أساسية ضعيفة (<70) ومادة داعمة قوية (>=80).
    ترجع (weak_subject, strong_subject) أو (None, None).
    """
    core_subjects = program_row.get("core_subjects", []) or []
    # طبّع إلى أسماء
    names = []
    for item in core_subjects:
        if isinstance(item, dict):
            names.append(item.get("subject") or item.get("name"))
        elif isinstance(item, (list, tuple)) and item:
            names.append(item[0])
        else:
            names.append(item)

    weak = None
    for subj in names:
        if not subj: continue
        for cand in [subj, f"{subj}_score", f"score_{subj}", f"{subj}_mark"]:
            if cand in student.index:
                try:
                    val = float(student.get(cand, 0.0))
                except:
                    val = 0.0
                if val < 70:
                    weak = subj
                    break
        if weak: break

    strong = None
    for col in student.index:
        name = str(col)
        if name.startswith("interest_") or name in ("student_id","student_name","label","branch","general_average","top_interests"):
            continue
        try:
            val = float(student.get(name, 0.0))
        except:
            continue
        if val >= 80 and name != weak:
            strong = name
            break

    return weak, strong


def generate_reasons_for_program(student, program_row: dict, templates_en: dict, templates_ar: dict):
    """
    نفس منطق الملف المرجعي: GPA / Interest / ضعف+تعويض / فرص عمل / تشجيع.
    ترجع (reasons_en, reasons_ar) كل منها قائمة نصوص (حتى 4 عناصر).
    """
    reasons_en, reasons_ar = [], []

    student_gpa = float(student.get("general_average", 0.0))
    min_gpa = float(program_row.get("min_gpa_regular", 0.0))

    if student_gpa >= min_gpa + 10:
        reasons_en.append(templates_en.get("gpa_strong", _DEFAULT_EN["gpa_strong"]).format(gpa=round(student_gpa,1)))
        reasons_ar.append(templates_ar.get("gpa_strong", _DEFAULT_AR["gpa_strong"]).format(gpa=round(student_gpa,1)))
    elif student_gpa >= min_gpa:
        reasons_en.append(templates_en.get("gpa_borderline", _DEFAULT_EN["gpa_borderline"]).format(gpa=round(student_gpa,1)))
        reasons_ar.append(templates_ar.get("gpa_borderline", _DEFAULT_AR["gpa_borderline"]).format(gpa=round(student_gpa,1)))
    else:
        reasons_en.append(templates_en.get("gpa_borderline", _DEFAULT_EN["gpa_borderline"]).format(gpa=round(student_gpa,1)))
        reasons_ar.append(templates_ar.get("gpa_borderline", _DEFAULT_AR["gpa_borderline"]).format(gpa=round(student_gpa,1)))

    pname = program_row.get("program_name", "")
    interest_area, interest_score = _find_program_interest_score(student, pname, program_row)
    if interest_area and interest_score >= 70:
        reasons_en.append(templates_en.get("interest_high", _DEFAULT_EN["interest_high"]).format(interest_area=interest_area))
        reasons_ar.append(templates_ar.get("interest_high", _DEFAULT_AR["interest_high"]).format(interest_area=interest_area))

    weak_subj, strong_subj = _find_weak_and_supporting_subjects(program_row, student)
    if weak_subj and strong_subj:
        reasons_en.append(templates_en.get("weakness_compensation", _DEFAULT_EN["weakness_compensation"]).format(weak_subject=weak_subj, strong_subject=strong_subj))
        reasons_ar.append(templates_ar.get("weakness_compensation", _DEFAULT_AR["weakness_compensation"]).format(weak_subject=weak_subj, strong_subject=strong_subj))

    career_paths = program_row.get("future_jobs") or program_row.get("career_paths") or pname
    unemployment = program_row.get("unemployment_rate", 15)
    salary = program_row.get("expected_salary", 3500) or 3500
    reasons_en.append(templates_en.get("career_opportunities", _DEFAULT_EN["career_opportunities"]).format(career_paths=career_paths, unemployment=unemployment, salary=salary))
    reasons_ar.append(templates_ar.get("career_opportunities", _DEFAULT_AR["career_opportunities"]).format(career_paths=career_paths, unemployment=unemployment, salary=salary))

    reasons_en.append(templates_en.get("final_encouragement", _DEFAULT_EN["final_encouragement"]))
    reasons_ar.append(templates_ar.get("final_encouragement", _DEFAULT_AR["final_encouragement"]))

    return reasons_en[:4], reasons_ar[:4]


def generate_ranked_reasons(student, program_row: dict, templates_en: dict, templates_ar: dict, rank: int = 0):
    """
    Generate a ranked set of reason strings for a program using the provided
    templates. rank: 0 => FIRST_CHOICE, 1 => SECOND_CHOICE, 2 => THIRD_CHOICE
    Returns (reasons_en:list, reasons_ar:list)
    """
    try:
        key_map = {0: "FIRST_CHOICE", 1: "SECOND_CHOICE", 2: "THIRD_CHOICE"}
        key = key_map.get(int(rank), "THIRD_CHOICE")

        tpl_en = templates_en.get(key, {}) or {}
        tpl_ar = templates_ar.get(key, {}) or {}

        reasons_en, reasons_ar = [], []

        # Student & program basic info
        student_gpa = float(student.get("general_average", 0.0))
        min_gpa = float(program_row.get("min_gpa_regular", 0.0))
        # decide which gpa message to use
        if student_gpa >= min_gpa + 10:
            gpa_key = "gpa_strong"
        elif student_gpa >= min_gpa + 5:
            gpa_key = "gpa_good"
        else:
            gpa_key = "gpa_borderline"

        # interest area & score
        pname_local = program_row.get("program_name", "") if isinstance(program_row, dict) else ""
        interest_area, interest_score = _find_program_interest_score(student, pname_local, program_row)
        try:
            interest_score = round(float(interest_score), 1)
        except Exception:
            interest_score = 0.0

        # supporting subject and its score (try several column name patterns)
        weak_subj, strong_subj = _find_weak_and_supporting_subjects(program_row, student)
        strong_score = None
        if strong_subj:
            for cand in (strong_subj, f"{strong_subj}_score", f"score_{strong_subj}", f"{strong_subj}_mark"):
                if cand in student.index:
                    try:
                        strong_score = float(student.get(cand, 0.0))
                        break
                    except Exception:
                        continue

        # career / labour info
        unemployment = program_row.get("unemployment_rate", 15)
        salary = program_row.get("expected_salary", 3500) or 3500

        fmt_ctx = {
            "gpa": round(student_gpa, 1),
            "interest_area": interest_area or "",
            "interest_score": interest_score,
            "strong_subject": strong_subj or "",
            "score": (round(strong_score, 1) if strong_score is not None else ""),
            "unemployment": unemployment,
            "salary": salary,
        }

        # Build English reasons (intro first)
        intro_en = tpl_en.get("intro")
        if intro_en:
            reasons_en.append(intro_en.format(**fmt_ctx))

        gpa_msg_en = tpl_en.get(gpa_key)
        if gpa_msg_en:
            reasons_en.append(gpa_msg_en.format(**fmt_ctx))

        if interest_area and tpl_en.get("interest_match"):
            reasons_en.append(tpl_en.get("interest_match").format(**fmt_ctx))

        if strong_subj and tpl_en.get("strength_highlight"):
            reasons_en.append(tpl_en.get("strength_highlight").format(**fmt_ctx))

        # career message: try multiple possible keys
        for ck in ("career_strong", "career_good", "career_explore"):
            if tpl_en.get(ck):
                reasons_en.append(tpl_en.get(ck).format(**fmt_ctx))
                break

        final_key_en = tpl_en.get("final_strong") or tpl_en.get("final_good") or tpl_en.get("final_explore")
        if final_key_en:
            reasons_en.append(final_key_en.format(**fmt_ctx))

        # Build Arabic reasons
        intro_ar = tpl_ar.get("intro")
        if intro_ar:
            reasons_ar.append(intro_ar.format(**fmt_ctx))

        gpa_msg_ar = tpl_ar.get(gpa_key)
        if gpa_msg_ar:
            reasons_ar.append(gpa_msg_ar.format(**fmt_ctx))

        if interest_area and tpl_ar.get("interest_match"):
            reasons_ar.append(tpl_ar.get("interest_match").format(**fmt_ctx))

        if strong_subj and tpl_ar.get("strength_highlight"):
            reasons_ar.append(tpl_ar.get("strength_highlight").format(**fmt_ctx))

        for ck in ("career_strong", "career_good", "career_explore"):
            if tpl_ar.get(ck):
                reasons_ar.append(tpl_ar.get(ck).format(**fmt_ctx))
                break

        final_key_ar = tpl_ar.get("final_strong") or tpl_ar.get("final_good") or tpl_ar.get("final_explore")
        if final_key_ar:
            reasons_ar.append(final_key_ar.format(**fmt_ctx))

        # Trim to max 4 reasons each (intro + up to 3 details)
        return reasons_en[:4], reasons_ar[:4]
    except Exception as e:
        logger.exception(f"generate_ranked_reasons error: {e}")
        return [], []


def _get_match_strength(score_100: float):
    """نفس سُلّم المطابقة المستخدم بالمشروع."""
    if score_100 >= 85:  return "Excellent", "ممتاز"
    if score_100 >= 70:  return "Strong", "قوي"
    if score_100 >= 55:  return "Good", "جيد"
    if score_100 >= 40:  return "Moderate", "متوسط"
    return "Fair", "مقبول"


def _lookup_program_row_by_name(program_name: str) -> dict:
    """
    يحاول جلب بيانات البرنامج من UNIVERSITY_CFG (إن وُجدت) وإلا يرجع حدّاً أدنى.
    نتوقع أن UNIVERSITY_CFG قد تحتوي programs / programs_by_name / أو حقول مكافئة.
    """
    base = {"program_name": program_name, "min_gpa_regular": 0, "faculty_name": "",
            "core_subjects": [], "future_jobs": "", "unemployment_rate": 15, "expected_salary": 3500}
    try:
        prog = None
        if isinstance(UNIVERSITY_CFG, dict):
            by_name = UNIVERSITY_CFG.get("programs_by_name") or {}
            prog = by_name.get(program_name)
            if not prog:
                for item in UNIVERSITY_CFG.get("programs", []):
                    if item.get("program_name") == program_name or item.get("program_name_en") == program_name:
                        prog = item; break
        if prog:
            base.update({k: v for k, v in prog.items() if v is not None})
    except Exception:
        pass
    return base
STREAM_SLUG_TO_EN = {
    "scientific": "Scientific",
    "literary": "Literary",
    "sharia": "Sharia",
    "agricultural": "Agricultural",
    "industrial": "Industrial",
    "hotel_home_econ": "Hotel and Home Economics",
    "home_economics": "Home Economics",
    "technological": "Technological",
    "entrepreneurship": "Entrepreneurship and Business",
}

CORE_SUBJECT_KEYS_BY_SLUG: Dict[str, List[str]] = {
    "scientific": ["mathematics", "physics", "chemistry", "biology"],
    "literary": ["arabic_language", "english_language", "history", "geography"],
    "sharia": ["quran_sciences", "arabic_language"],
    "agricultural": ["biology", "chemistry"],
    "industrial": ["mathematics", "physics", "technology"],
    "hotel_home_econ": ["management_economics"],
    "home_economics": ["professional_sciences"],
    "technological": ["technology", "mathematics"],
    "entrepreneurship": ["management_economics"],
}

def _parse_float(val: Optional[str]) -> Optional[float]:
    """تحويل مدخل نصي إلى float مع دعم الفاصلة العربية والنسبة المئوية."""
    if val is None:
        return None
    s = str(val).strip()
    if s == "":
        return None
    s = s.replace("%", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None

def _extract_general_average_from_dict(d: dict) -> Optional[float]:
    """يحاول استخراج المعدل العام من عدة مفاتيح محتملة (يشمل score_overall_average من HTML)."""
    if not isinstance(d, dict):
        return None
    candidates = [
        # من HTML
        "score_overall_average",
        # إنجليزي شائع
        "general_average", "overall_average", "total_average", "avg", "average", "gpa",
        "gen_avg", "total_avg", "overall_avg",
        # عربي
        "المعدل العام", "المعدل_العام", "المعدل", "معدل", "المعدلالعام",
    ]
    for k in candidates:
        if k in d:
            f = _parse_float(d.get(k))
            if f is not None:
                return round(f, 2)
    # fallback: أي مفتاح فيه "معدل"
    for k, v in d.items():
        if isinstance(k, str) and "معدل" in k:
            f = _parse_float(v)
            if f is not None:
                return round(f, 2)
    return None


def compute_core_average_by_slug(stream_slug: str, scores: dict) -> float:
    """
    يحسب معدل المواد الأساسية بناءً على الحقول الظاهرة في HTML لذلك الفرع (بالـslug).
    - يجمع قيم subjects المعرّفة في CORE_SUBJECT_KEYS_BY_SLUG[slug] إن كانت موجودة ومعبأة.
    - يقسم على عدد القيم المحتسبة فقط.
    """
    keys = CORE_SUBJECT_KEYS_BY_SLUG.get(stream_slug, [])
    if not keys:
        return 0.0

    total = 0.0
    count = 0
    for key in keys:
        val = scores.get(key)
        f = _parse_float(val) if val is not None else None
        if f is not None:
            total += f
            count += 1

    return round(total / count, 2) if count else 0.0

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']
def _normalize_scores_keys(ocr_scores: dict) -> dict:
    """حوّل مفاتيح الـOCR إلى المفاتيح القياسية اللي التمبلت/الباك إند بيستخدمها."""
    alias_map = {
        # أدبي
        "arabic": "arabic_language",
        "english": "english_language",

        # شرعي
        "quran": "quran_sciences",

        # ريادة الأعمال / الفندقي
        "management_and_economics": "management_economics",
    }
    norm = dict(ocr_scores or {})
    for src, dst in alias_map.items():
        if src in norm and dst not in norm:
            norm[dst] = norm[src]
    return norm


@app.route('/')
def index():
    language = request.args.get('lang', 'en')
    return render_template('index.html', language=language)

@app.route('/upload_diploma', methods=['GET', 'POST'])
def upload_diploma():
    language = request.args.get('lang', 'en')

    if request.method == 'POST':
        if 'diploma_file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['diploma_file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                # 1) OCR
                result = diploma_scanner.run_ocr(filepath)
                if not result:
                    flash('OCR returned no data')
                    return redirect(request.url)

                # 2) تطبيع الفرع لقيم التمبلت
                raw_stream = (result.get("stream") or "").strip().lower()
                stream_map = {
                    # نفس قيم <option value="..."> في التمبلت
                    "scientific": "scientific",
                    "literary": "literary",
                    "sharia": "sharia",
                    "agricultural": "agricultural",
                    "industrial": "industrial",
                    "hotel and home economics": "hotel_home_econ",
                    "home economics": "home_economics",
                    "technological": "technological",
                    "entrepreneurship and business": "entrepreneurship",
                }
                stream = stream_map.get(raw_stream, raw_stream)  # fallback: نفس النص

                # 3) المواد الأساسية المتوقعة في الواجهة لكل فرع
                CORE_KEYS_BY_STREAM = {
                    "scientific": ["mathematics", "physics", "chemistry", "biology"],
                    "literary": ["arabic_language", "english_language", "history", "geography"],
                    "sharia": ["quran_sciences", "arabic_language"],
                    "agricultural": ["biology", "chemistry"],
                    "industrial": ["mathematics", "physics", "technology"],
                    "hotel_home_econ": ["management_economics"],
                    "home_economics": ["professional_sciences"],
                    "technological": ["technology", "mathematics"],
                    "entrepreneurship": ["management_economics"],
                }

                ocr_scores = result.get("scores", {}) or {}
                ocr_scores = _normalize_scores_keys(ocr_scores)  
                expected_keys = CORE_KEYS_BY_STREAM.get(stream, [])

                # 4) جهِّز scores ضمن البنية اللي التمبلت بيستخدمها
                scores_out = {}
                
                # المعدل العام
                if "overall_average" in ocr_scores:
                    scores_out["overall_average"] = ocr_scores.get("overall_average")

                # عبّي المواد الأساسية لهالفرع إن كانت موجودة برد الـOCR
                for key in expected_keys:
                    if key in ocr_scores:
                        scores_out[key] = ocr_scores[key]
                
                std_result = {
                    "stream": stream,
                    "scores": scores_out
                }

                # 5) خزّن بالجلسة وروِّح عالصفحة
                session['diploma_data'] = std_result
                return redirect(url_for('manual_entry', lang=language))

            except Exception as e:
                logger.error(f"Error processing diploma: {str(e)}", exc_info=True)
                flash(f"Error processing diploma: {str(e)}")
                return redirect(request.url)

    return render_template('upload_diploma.html', language=language)

@app.route('/manual_entry', methods=['GET', 'POST'])
def manual_entry():
    language = request.args.get('lang', 'en')

    if request.method == 'POST':
        try:
            form_data = request.form.to_dict()
            academic_data = {
                'stream': form_data.get('stream', '').strip(),  # slug من الواجهة
                'scores': {}
            }
            # 1) قراءة المعدل العام من الحقل الحقيقي في HTML
            ga_from_form = form_data.get('score_overall_average')
            if ga_from_form is not None and str(ga_from_form).strip() != "":
                parsed = _parse_float(ga_from_form)
                if parsed is not None:
                    academic_data["general_average"] = round(parsed, 2)

            # 2) درجات المواد: نضيف كل score_* عدا score_overall_average
            for field, value in form_data.items():
                if field.startswith('score_') and field != 'score_overall_average' and value.strip() != "":
                    subject_key = field.replace('score_', '').strip()  # مثال: mathematics, arabic_language
                    academic_data['scores'][subject_key] = value  # نخزن النص؛ التحويل للـfloat لاحقاً

            # 3) دمج مع OCR (الأولوية للإدخال اليدوي)
            ocr_data = session.get('diploma_data') or {}
            merged_stream = academic_data.get('stream') or ocr_data.get('stream', '')
            merged_scores = dict(ocr_data.get('scores', {}))
            merged_scores.update(academic_data.get('scores', {}))

            merged = {
                "stream": merged_stream,   # slug/عربي/إنجليزي — سنحوّله لاحقًا عند الحفظ
                "scores": merged_scores
            }
            # لو المعدل العام مش بإدخال المستخدم، خذه من OCR إن وُجد
            ga_final = academic_data.get("general_average")
            if ga_final is None:
                ga_final = _extract_general_average_from_dict(ocr_data)
            if ga_final is not None:
                merged["general_average"] = ga_final

            session['academic_data'] = merged
            return redirect(url_for('interest_assessment', lang=language))
        except Exception as e:
            logger.error(f"Error processing manual entry: {str(e)}", exc_info=True)
            flash(f"Error processing form data: {str(e)}")
            return redirect(request.url)

    diploma_data = session.get('diploma_data', {})
    return render_template('manual_entry.html', diploma_data=diploma_data, language=language)
@app.route('/interest_assessment', methods=['GET', 'POST'])
def interest_assessment():
    lang = request.args.get('lang', 'ar')
    questionnaire = ia.get_questionnaire(language=lang)
    ids_in_order = ia.get_question_ids_in_order()  # ['q1'..'q37']

    if request.method == 'POST':
        try:
            answers_in_order = [request.form.get(qid, '').strip() for qid in ids_in_order]

            acad = session.get('academic_data') or session.get('diploma_data') or {}
            stream_raw = (acad.get('stream') or "").strip()  
            scores = acad.get('scores') or {}

            gen_avg = acad.get("general_average")
            gen_avg = _parse_float(gen_avg) if gen_avg is not None else None
            gen_avg = round(gen_avg, 2) if gen_avg is not None else 0.0

            if stream_raw in STREAM_SLUG_TO_EN:
                stream_slug = stream_raw
            else:
                inv_en_to_slug = {v: k for k, v in STREAM_SLUG_TO_EN.items()}
                stream_slug = inv_en_to_slug.get(stream_raw, None)

            core_avg = compute_core_average_by_slug(stream_slug, scores) if stream_slug else 0.0

            stream_en = STREAM_SLUG_TO_EN[stream_slug] if stream_slug else stream_raw

            record = {
                "branch": stream_en,
                "general_average": gen_avg,
                "core_subject_average": core_avg,
            }
            for qid, ans in zip(ids_in_order, answers_in_order):
                record[qid] = ans  # q1..q37

            final_df = process_single_record_exact(record, QUESTIONS_CFG, UNIVERSITY_CFG, MCQ_WEIGHTS)
            # Prepare result row and predictions
            result_row = final_df.to_dict(orient="records")[0]

            # Predict top-3 using the chosen model (gb_model here)
            try:
                top3_labels, top3_probs = _predict_topk_labels(rf_model, final_df, k=3)
            except Exception as pred_err:
                logger.error(f"Prediction failed: {pred_err}", exc_info=True)
                top3_labels, top3_probs = [], []

            # Attach prediction fields and a unique row_id to the saved row
            row_id = str(uuid.uuid4())
            result_row['row_id'] = row_id
            for i in range(3):
                label = top3_labels[i] if i < len(top3_labels) else ''
                prob = top3_probs[i] if i < len(top3_probs) else 0.0
                result_row[f'predicted_top{i+1}_label'] = label
                result_row[f'predicted_top{i+1}_prob'] = prob

            # Compute top_interests (text) from any interest_* columns and add to result_row
            try:
                interest_keys = [k for k in result_row.keys() if str(k).startswith('interest_')]
                top_interests_list = []
                if interest_keys:
                    interest_pairs = []
                    for k in interest_keys:
                        try:
                            v = float(result_row.get(k) or 0.0)
                        except Exception:
                            v = 0.0
                        interest_pairs.append((k.replace('interest_', ''), v))
                    interest_pairs.sort(key=lambda x: x[1], reverse=True)
                    top_interests_list = [name for name, val in interest_pairs[:3] if val > 0]
                result_row['top_interests'] = ','.join(top_interests_list)
            except Exception:
                result_row['top_interests'] = ''

            # Save processed row (with row_id, predictions and top_interests) to Desktop processed_result.csv
            try:
                desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
                os.makedirs(desktop_path, exist_ok=True)
                output_file = os.path.join(desktop_path, "processed_result.csv")
                # ensure fieldnames include the new prediction, row_id and top_interests columns
                fieldnames = list(final_df.columns)
                if 'top_interests' not in fieldnames:
                    fieldnames.append('top_interests')
                fieldnames += ['row_id', 'predicted_top1_label', 'predicted_top1_prob',
                                                       'predicted_top2_label', 'predicted_top2_prob',
                                                       'predicted_top3_label', 'predicted_top3_prob']
                write_header = not os.path.exists(output_file)
                with open(output_file, "a", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    if write_header:
                        writer.writeheader()
                    # ensure all keys exist in result_row for DictWriter
                    for k in fieldnames:
                        if k not in result_row:
                            result_row[k] = ''
                    writer.writerow(result_row)
                logger.info(f"✅ Saved processed row to {output_file} (row_id={row_id})")
            except Exception as save_err:
                logger.warning(f"Could not save result to Desktop: {save_err}")

            MAJOR_NAME_AR = globals().get("MAJOR_NAME_AR", {})  

            templates_en, templates_ar = _load_or_create_templates_safe()
            student_series = _SeriesLike(result_row)

            rec_items = []
            for idx, (lbl, p) in enumerate(zip(top3_labels, top3_probs)):
                program_row = _lookup_program_row_by_name(lbl)

                # Use ranked templates: rank = idx (0:first,1:second,2:third)
                reasons_en, reasons_ar = generate_ranked_reasons(
                    student_series, program_row, templates_en, templates_ar, rank=idx
                )

                score_float = float(p if p is not None else 0.0)
                score_100 = score_float * 100.0
                match_en, match_ar = _get_match_strength(score_100)

                rec_items.append({
                    "field_name": lbl,
                    "field_name_ar": MAJOR_NAME_AR.get(lbl, lbl),
                    "score": score_float,
                    "match_strength": match_en,
                    "match_strength_ar": match_ar,
                    "reasons": (reasons_ar if lang == 'ar' else reasons_en),
                    "reasons_en": reasons_en,
                    "reasons_ar": reasons_ar,
                    "subfields": [],
                    "subfields_ar": [],
                })

            recommendations_payload = {
                "recommendations": rec_items,
                "university_options": []
            }

            return render_template(
                'recommendations.html',
                language=lang,
                recommendations=recommendations_payload,
                row_id=row_id
            )

        except Exception as e:
            logger.error(f"Error processing record: {str(e)}", exc_info=True)
            return render_template(
                'recommendations.html',
                language=lang,
                recommendations={"error": str(e), "recommendations": [], "university_options": []}
            )

    return render_template('interest_assessment.html', language=lang, questionnaire=questionnaire, ordered_answers=None)







BASE_DIR = Path(__file__).resolve().parent  
try:
    with open(BASE_DIR / "questions_config.json", "r", encoding="utf-8") as f:
        QUESTIONS_CFG = json.load(f)
except FileNotFoundError:
    logger.error("questions_config.json غير موجود في نفس المجلد.")
    QUESTIONS_CFG = {}

try:
    with open(BASE_DIR / "university_data.json", "r", encoding="utf-8") as f:
        UNIVERSITY_CFG = json.load(f)
except FileNotFoundError:
    logger.error("university_data.json غير موجود في نفس المجلد.")
    UNIVERSITY_CFG = {}

try:
    with open(BASE_DIR / "mcq_option_weights.json", "r", encoding="utf-8") as f:
        MCQ_WEIGHTS = json.load(f)
except FileNotFoundError:
    MCQ_WEIGHTS = None  

def _normalize_ar_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = re.sub(r'[\u0617-\u061A\u064B-\u0652]', '', s)
    s = s.replace('ـ', '')
    s = s.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')
    s = s.replace('ى', 'ي')
    s = s.translate(str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789'))
    s = re.sub(r'[^\w\s\u0600-\u06FF]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip().lower()
    return s

def _build_answer_mapping_for_question(q_obj: dict) -> dict:
    if isinstance(q_obj.get('mapping'), dict):
        return { _normalize_ar_text(k): int(v) for k, v in q_obj['mapping'].items() }
    options = q_obj.get('options') or q_obj.get('answers') or q_obj.get('choices_with_scores')
    if isinstance(options, list) and options:
        out = {}
        for idx, opt in enumerate(options, start=1):
            if isinstance(opt, str):
                out[_normalize_ar_text(opt)] = idx
            else:
                text = opt.get('text') or opt.get('label') or opt.get('name') or opt.get('option') or ''
                numeric = opt.get('value') or opt.get('score') or opt.get('numeric')
                out[_normalize_ar_text(text)] = int(numeric if numeric is not None else idx)
        return out
    choices = q_obj.get('choices')
    if isinstance(choices, list) and choices:
        return { _normalize_ar_text(t): i for i, t in enumerate(choices, start=1) }
    return {}

def _extract_question_meta_table(questions_config: Dict) -> pd.DataFrame:
    rows = []
    for category, cat in (questions_config or {}).items():
        cat_w = cat.get('weight', 0)
        for q in cat.get('questions', []):
            rows.append({
                'question_id': str(q.get('id', '')).strip().lower(),
                'type': (q.get('type', '') or '').strip().lower(),
                'category': category,
                'category_weight': cat_w,
                'field_weights': q.get('field_weights', {}),
                '_mapping': _build_answer_mapping_for_question(q),
                '_options': (
                    [ (o if isinstance(o, str) else (o.get('text') or o.get('label') or o.get('name') or o.get('option') or ''))
                      for o in (q.get('options') or q.get('answers') or q.get('choices_with_scores') or []) ]
                    or q.get('choices') or []
                )
            })
    return pd.DataFrame(rows)

def _slugify_ar(text: str) -> str:
    s = '' if text is None else str(text).strip().lower()
    s = s.translate(str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789'))
    s = re.sub(r'[^\w\s\u0600-\u06FF]', '_', s)
    s = re.sub(r'\s+', '_', s)
    s = re.sub(r'_+', '_', s).strip('_')
    return s or "na"

def _clean_one_record(record: Dict[str, any], questions_config: Dict, university_data: Dict) -> pd.DataFrame:
    df = pd.DataFrame([record]).copy()
    if 'student_id' in df.columns:
        df['student_id'] = df['student_id'].astype(str)

    for col in ['general_average', 'core_subject_average']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            med = df[col].median()
            df[col] = df[col].fillna(med)

    qmeta = _extract_question_meta_table(questions_config)
    qmeta = qmeta.set_index('question_id', drop=False)
    q_cols = [c for c in df.columns if c.startswith('q') and c[1:].isdigit()]

    for col in q_cols:
        key = col.lower()
        meta = qmeta.loc[key] if key in qmeta.index else None
        q_type = (meta['type'] if meta is not None else '') or ''
        mapping = (meta['_mapping'] if meta is not None else {}) or {}

        as_num = pd.to_numeric(df[col], errors='coerce')
        if mapping:
            norm_txt = df[col].astype(str).map(_normalize_ar_text)
            mapped = norm_txt.map(mapping)
            col_vals = as_num.where(~as_num.isna(), mapped)
        else:
            col_vals = as_num

        if q_type == 'likert':
            med = pd.to_numeric(col_vals, errors='coerce').median()
            if pd.isna(med): med = 3
            filled = pd.to_numeric(col_vals, errors='coerce').fillna(med)
            df[col] = filled.clip(lower=1, upper=5).round().astype(int)
        elif q_type in ('multiple_choice', 'single', 'mcq'):
            if mapping:
                series_num = pd.to_numeric(col_vals, errors='coerce')
                mode_vals = series_num.mode(dropna=True)
                fillv = int(mode_vals.iloc[0]) if not mode_vals.empty else int(list(mapping.values())[0])
                df[col] = series_num.fillna(fillv).astype(int)
            else:
                df[col] = col_vals
        else:
            if mapping:
                series_num = pd.to_numeric(col_vals, errors='coerce')
                mode_vals = series_num.mode(dropna=True)
                fillv = int(mode_vals.iloc[0]) if not mode_vals.empty else int(list(mapping.values())[0])
                df[col] = series_num.fillna(fillv).astype(int)
            else:
                df[col] = col_vals

    if 'branch' in df.columns:
        valid_branches = list((university_data or {}).get('branches_allowed', {}).keys())
        if valid_branches:
            most_common_branch = df['branch'].mode(dropna=True)
            if not most_common_branch.empty:
                most_common_branch = most_common_branch.iloc[0]
                df.loc[~df['branch'].isin(valid_branches), 'branch'] = most_common_branch
    return df

def _academic_features(students_df: pd.DataFrame) -> pd.DataFrame:
    out = students_df.copy()
    if 'general_average' in out.columns:
        out['gpa_bin'] = pd.cut(out['general_average'], bins=[0,60,70,80,90,100],
                                labels=['<60','60-70','70-80','80-90','90-100'])
        out['gpa_percentile'] = out['general_average'].rank(pct=True)*100
    if 'core_subject_average' in out.columns:
        out['core_avg_bin'] = pd.cut(out['core_subject_average'], bins=[0,60,70,80,90,100],
                                     labels=['<60','60-70','70-80','80-90','90-100'])
        out['core_avg_percentile'] = out['core_subject_average'].rank(pct=True)*100
    if 'general_average' in out.columns and 'core_subject_average' in out.columns:
        out['gpa_to_core_ratio'] = out['general_average'] / out['core_subject_average']
    return out

def _extract_qid_to_options(questions_config: Dict) -> Dict[str, List[str]]:
    qid_to_options = {}
    for _, q in [(None, q) for cat in (questions_config or {}).values() for q in cat.get('questions', [])]:
        qid = str(q.get('id') or q.get('question_id') or '').strip().lower()
        if not qid: continue
        qtype = (q.get('type') or '').strip().lower()
        if qtype in ('multiple_choice','single','mcq'):
            opts = None
            if isinstance(q.get('options'), list):
                opts = [ (o if isinstance(o,str) else (o.get('text') or o.get('label') or o.get('name') or o.get('option') or '')) for o in q['options'] ]
            elif isinstance(q.get('answers'), list):
                opts = [ (o if isinstance(o,str) else (o.get('text') or o.get('label') or o.get('name') or o.get('option') or '')) for o in q['answers'] ]
            elif isinstance(q.get('choices_with_scores'), list):
                opts = [ (o if isinstance(o,str) else (o.get('text') or o.get('label') or o.get('name') or o.get('option') or '')) for o in q['choices_with_scores'] ]
            elif isinstance(q.get('choices'), list):
                opts = [ (o if isinstance(o,str) else str(o)) for o in q['choices'] ]
            if opts:
                qid_to_options[qid] = [str(x).strip() for x in opts]
    return qid_to_options

def _interest_features(students_df: pd.DataFrame,
                       question_meta_df: pd.DataFrame,
                       questions_config: Dict,
                       mcq_option_weights: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None) -> pd.DataFrame:
    interest_df = students_df.copy()

    qmeta = question_meta_df.copy()
    if 'field_weights' in qmeta.columns and qmeta['field_weights'].dtype == 'object':
        qmeta['field_weights'] = qmeta['field_weights'].apply(lambda x: x if isinstance(x, dict) else x)

    all_fields = set()
    for fw in qmeta['field_weights']:
        if isinstance(fw, dict):
            all_fields.update(fw.keys())

    for field in all_fields:
        interest_df[f'interest_{field}'] = 0.0

    likert_meta = qmeta[qmeta['type'].str.lower() == 'likert']
    for _, row in likert_meta.iterrows():
        qid = str(row['question_id']).strip()
        cat_w = float(row['category_weight'])
        fweights = row['field_weights']
        if isinstance(fweights, dict) and qid in interest_df.columns:
            resp = interest_df[qid].clip(1,5)
            normalized = (resp - 1)/4.0
            for field, w in fweights.items():
                col = f'interest_{field}'
                if col in interest_df.columns:
                    interest_df[col] += normalized * float(w) * cat_w

    qid_to_options = _extract_qid_to_options(questions_config)
    mcq_meta = qmeta[qmeta['type'].str.lower().isin(['multiple_choice','single','mcq'])]
    mcq_option_weights = mcq_option_weights or {}
    for _, row in mcq_meta.iterrows():
        qid = str(row['question_id']).strip().lower()
        cat_w = float(row['category_weight'])
        if qid not in interest_df.columns:
            continue
        options = qid_to_options.get(qid, [])
        if not options:
            observed_vals = sorted([v for v in interest_df[qid].dropna().unique().tolist() if isinstance(v,(int,float))])
            options = [f"opt_{int(i)}" for i in observed_vals]
        for idx, opt_text in enumerate(options, start=1):
            slug = _slugify_ar(opt_text)
            col = f"mcq_{qid}__{slug}"
            interest_df[col] = (interest_df[qid] == idx).astype(int)
            opt_w_by_field = (mcq_option_weights.get(qid, {}) or {}).get(opt_text, None)
            if isinstance(opt_w_by_field, dict):
                for field, w in opt_w_by_field.items():
                    col_interest = f'interest_{field}'
                    if col_interest in interest_df.columns:
                        interest_df[col_interest] += interest_df[col] * float(w) * cat_w

    interest_cols = [c for c in interest_df.columns if c.startswith('interest_')]
    for col in interest_cols:
        mx = interest_df[col].max()
        if pd.notna(mx) and mx > 0:
            interest_df[col] = (interest_df[col] / mx) * 100.0
    if interest_cols:
        interest_df['top_interests'] = interest_df[interest_cols].apply(
            lambda x: ','.join(x.nlargest(3).index.str.replace('interest_','')), axis=1
        )

    return interest_df

def _branch_eligibility_table(university_data: Dict) -> pd.DataFrame:
    rows = []
    branches_allowed = (university_data or {}).get('branches_allowed', {})
    for branch, branch_data in branches_allowed.items():
        for program in branch_data.get('allowed_programs', []):
            rows.append({'branch': branch, 'program_name': program, 'is_eligible': True})
    return pd.DataFrame(rows)

def _program_metadata_table(university_data: Dict) -> pd.DataFrame:
    rows = []
    for fac in (university_data or {}).get('faculties', []):
        fac_name = fac.get('name','')
        fac_min_r = fac.get('minimum_gpa_regular', 0)
        fac_min_p = fac.get('minimum_gpa_parallel', 0)
        for prog in fac.get('programs', []):
            rows.append({
                'program_name': prog.get('name',''),
                'faculty_name': fac_name,
                'core_subjects': prog.get('core_subjects', []),
                'min_gpa_regular': prog.get('minimum_gpa_regular', fac_min_r),
                'min_gpa_parallel': prog.get('minimum_gpa_parallel', fac_min_p),
                'unemployment_rate': prog.get('unemployment_rate', 0),
                'expected_salary': prog.get('expected_salary', 0),
                'future_jobs': prog.get('future_jobs',''),
                'description': prog.get('description',''),
            })
    return pd.DataFrame(rows)

def _eligibility_features(students_df: pd.DataFrame,
                          program_meta_df: pd.DataFrame,
                          branch_elig_df: pd.DataFrame) -> pd.DataFrame:
    out = students_df.copy()
    branch_to_programs = {b: set(g["program_name"]) for b, g in branch_elig_df.groupby("branch")}
    branches = out["branch"]
    gpa = out["general_average"]
    for _, prog in program_meta_df.iterrows():
        prog_name = prog["program_name"]
        min_reg = float(prog.get("min_gpa_regular", 0))
        min_par = float(prog.get("min_gpa_parallel", 0))
        col = f"eligible_{prog_name}"
        branch_ok = branches.map(lambda b: prog_name in branch_to_programs.get(b, set()))
        gpa_ok = (gpa >= min_reg) | (gpa >= min_par)
        out[col] = branch_ok & gpa_ok
    elig_cols = [c for c in out.columns if c.startswith("eligible_")]
    if elig_cols:
        out["num_eligible_programs"] = out[elig_cols].sum(axis=1)
    return out

def _merge_features(academic_df: pd.DataFrame,
                    interest_df: pd.DataFrame,
                    eligibility_df: pd.DataFrame) -> pd.DataFrame:
    merged = academic_df.copy()
    interest_cols = [c for c in interest_df.columns if c.startswith('interest_') or c == 'top_interests']
    merged = pd.concat([merged, interest_df[interest_cols]], axis=1)
    elig_cols = [c for c in eligibility_df.columns if c.startswith('eligible_') or c == 'num_eligible_programs']
    merged = pd.concat([merged, eligibility_df[elig_cols]], axis=1)
    return merged

def _prepare_ml_dataset(features_df: pd.DataFrame, program_meta_df: pd.DataFrame) -> pd.DataFrame:
    ml = features_df.copy()
    if 'label' in ml.columns:
        valid_programs = set(program_meta_df['program_name'])
        ml.loc[~ml['label'].isin(valid_programs), 'label'] = np.nan
        ml = ml.dropna(subset=['label'])
    for col in ['branch', 'gpa_bin', 'core_avg_bin']:
        if col in ml.columns:
            dummies = pd.get_dummies(ml[col], prefix=col)
            ml = pd.concat([ml, dummies], axis=1)
            ml = ml.drop(columns=[col])
    if 'top_interests' in ml.columns:
        ml = ml.drop(columns=['top_interests'])
    return ml

def process_single_record_exact(
    record: Dict[str, any],
    questions_config: Dict,
    university_data: Dict,
    mcq_option_weights: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
) -> pd.DataFrame:
    cleaned = _clean_one_record(record, questions_config, university_data)
    question_meta_df = _extract_question_meta_table(questions_config)
    program_meta_df  = _program_metadata_table(university_data)
    branch_elig_df   = _branch_eligibility_table(university_data)

    academic_df   = _academic_features(cleaned)
    interest_df   = _interest_features(cleaned, question_meta_df, questions_config, mcq_option_weights)
    eligibility_df= _eligibility_features(cleaned, program_meta_df, branch_elig_df)

    features_df = _merge_features(academic_df, interest_df, eligibility_df)
    ml_df       = _prepare_ml_dataset(features_df, program_meta_df)
    return ml_df.iloc[[0]]  # صف واحد


@app.route('/process_record', methods=['POST'])
def process_record():
    try:
        payload = request.get_json(force=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "Invalid JSON payload"}), 400

        branch = payload.get("branch")
        ga = payload.get("general_average")
        core = payload.get("core_subject_average")
        answers = payload.get("answers") or {}

        record = {"branch": branch, "general_average": ga, "core_subject_average": core}
        for k, v in answers.items():
            record[str(k).strip().lower()] = v  # q1..q37

        final_df = process_single_record_exact(record, QUESTIONS_CFG, UNIVERSITY_CFG, MCQ_WEIGHTS)
        return jsonify({
            "columns": list(final_df.columns),
            "row": final_df.iloc[0].to_dict()
        })
    except Exception as e:
        logger.error(f"/process_record error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
    
def _predict_topk_labels(model, ml_df: pd.DataFrame, k: int = 3):
    """
    Robustly predict top-k labels and probabilities from a fitted model.
    Returns (top_labels:list, top_probs:list). On failure returns ([], []).
    """
    try:
        cols = getattr(model, "feature_names_in_", None)
        if cols is not None:
            X = ml_df.reindex(columns=cols, fill_value=0)
        else:
            X = ml_df

        # Sanitize input: replace inf/-inf with NaN, fill NaNs with 0, and clip extreme values
        try:
            X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            X = X.clip(lower=-1e6, upper=1e6)
        except Exception:
            X = X.copy().fillna(0.0)

        proba = None
        # Try predict_proba
        if hasattr(model, "predict_proba"):
            try:
                p = model.predict_proba(X)
                proba = p[0] if hasattr(p, "__len__") and np.array(p).ndim > 1 else np.ravel(p)
            except Exception as e:
                logger.debug(f"predict_proba failed: {e}", exc_info=True)
                proba = None

        # Try decision_function if predict_proba not available or failed
        if proba is None and hasattr(model, "decision_function"):
            try:
                s = model.decision_function(X)
                proba = s[0] if hasattr(s, "__len__") and np.array(s).ndim > 1 else np.ravel(s)
            except Exception as e:
                logger.debug(f"decision_function failed: {e}", exc_info=True)
                proba = None

        # Fallback to predict (one-hot like probability)
        if proba is None:
            try:
                preds = model.predict(X)
                pred = preds[0] if hasattr(preds, "__len__") else preds
                classes = getattr(model, "classes_", np.array([pred]))
                proba = np.zeros(len(classes), dtype=float)
                # try to find index of predicted class
                try:
                    idx = int(np.where(classes == pred)[0][0])
                    proba[idx] = 1.0
                except Exception:
                    # If class not found, put probability on first class
                    if len(proba) > 0:
                        proba[0] = 1.0
            except Exception as e:
                logger.error(f"Model predict fallback failed: {e}", exc_info=True)
                return [], []

        proba = np.asarray(proba, dtype=float).ravel()

        classes = getattr(model, "classes_", None)
        if classes is None:
            # create synthetic class labels 0..n-1
            classes = np.arange(len(proba))
        else:
            classes = np.asarray(classes)

        # If lengths mismatch, try to align by trimming to the min length
        n = min(len(proba), len(classes))
        if n == 0:
            return [], []

        proba = proba[:n]
        classes = classes[:n]

        top_idx = np.argsort(proba)[::-1][:k]
        top_labels = [str(classes[i]) for i in top_idx]
        top_probs = [float(proba[i]) for i in top_idx]

        logger.info(f"[Predict] Top-{k} predictions: {list(zip(top_labels, [round(p,6) for p in top_probs]))}")
        return top_labels, top_probs

    except Exception as e:
        logger.exception(f"_predict_topk_labels unexpected error: {e}")
        return [], []

@app.route('/submit_choice', methods=['POST'])
def submit_choice():
    """
    Accepts a student's chosen major and writes a labelled record to
    data/processed/labelled_choices.csv. It looks up the processed_result.csv
    (saved to Desktop) by row_id, merges metadata, removes sensitive fields,
    and appends the labelled row.
    """
    try:
        chosen = request.form.get('chosen_major', '').strip()
        source = request.form.get('chosen_major_source', 'manual').strip()
        consent = request.form.get('consent', 'off') == 'on'
        row_id = request.form.get('row_id', '').strip()
        lang = request.form.get('lang', 'en')
        timestamp = datetime.utcnow().isoformat()

        # Try to find the processed row on the user's Desktop processed_result.csv
        processed_row = None
        try:
            desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
            processed_file = os.path.join(desktop_path, "processed_result.csv")
            if os.path.exists(processed_file):
                with open(processed_file, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for r in reader:
                        if r.get("row_id") == row_id:
                            processed_row = r
                            break
        except Exception:
            processed_row = None

        # Prepare labelled choices path inside project data/processed
        labelled_dir = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
        labelled_dir = os.path.abspath(labelled_dir)
        os.makedirs(labelled_dir, exist_ok=True)
        labelled_file = os.path.join(labelled_dir, "labelled_choices.csv")

        if processed_row is None:
            record = {
                "row_id": row_id,
                "chosen_major": chosen,
                "chosen_major_source": source,
                "chosen_at": timestamp,
                "consent": consent
            }
        else:
            # Merge processed row with label metadata
            record = dict(processed_row)
            record.update({
                "chosen_major": chosen,
                "chosen_major_source": source,
                "chosen_at": timestamp,
                "consent": consent
            })

        # Remove sensitive fields if present
        for sensitive in ("student_name", "student_id"):
            if sensitive in record:
                record.pop(sensitive, None)

        # Ensure consistent column order when writing
        write_header = not os.path.exists(labelled_file)
        with open(labelled_file, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(record.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(record)

        flash(("تم حفظ اختيارك، شكرًا لمساهمتك" if lang == "ar" else "Your choice was recorded, thanks for your contribution"))
        return redirect(url_for('index'))
    except Exception as e:
        logger.error(f"submit_choice error: {e}", exc_info=True)
        flash("حدث خطأ أثناء حفظ الاختيار" if request.form.get('lang', 'en') == 'ar' else "Failed to record choice")
        return redirect(url_for('index'))


# -----------------------
# Ranked recommendations: helper + route
# -----------------------
from pathlib import Path as _Path
def _load_ranked_recommendations() -> Optional[Dict[str, Any]]:
    """
    Load outputs/ranked_recommendations.json if present and return parsed JSON.
    """
    try:
        root = Path(__file__).resolve().parents[2]
        ranked_path = root / "src" / "outputs" / "ranked_recommendations.json"
        if ranked_path.exists():
            with open(ranked_path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        logger.exception("Failed to load ranked_recommendations.json")
    return None

@app.route('/ranked_recommendations')
def view_ranked_recommendations():
    """
    Render recommendations.html using the precomputed ranked_recommendations.json.
    Optional query param student_id to view a specific student's recommendations.
    Falls back gracefully if file is missing or malformed.
    """
    lang = request.args.get('lang', 'en')
    student_id = request.args.get('student_id', None)

    ranked = _load_ranked_recommendations()
    if not ranked:
        # fallback: show a friendly message via template
        return render_template('recommendations.html', language=lang, recommendations={"error": "No ranked_recommendations.json found", "recommendations": [], "university_options": []}, row_id="ranked")

    # ranked is expected to be a list of student recommendation entries
    selected = None
    if student_id:
        for entry in ranked:
            if str(entry.get("student_id")) == str(student_id):
                selected = entry
                break
    if selected is None:
        # default to first entry
        selected = ranked[0] if isinstance(ranked, list) and ranked else None

    if not selected:
        return render_template('recommendations.html', language=lang, recommendations={"error": "No recommendations available", "recommendations": [], "university_options": []}, row_id="ranked")

    recs_raw = selected.get("recommendations", [])
    rec_items = []
    for r in recs_raw:
        # Map fields from ranked structure to template structure
        major = r.get("major") or r.get("field") or ""
        score = float(r.get("score", 0.0))
        score_100 = score * 100.0 if score <= 1.0 else score
        match_en, match_ar = _get_match_strength(score_100)
        reasons_en = r.get("reasons_en", []) or []
        reasons_ar = r.get("reasons_ar", []) or []
        rec_items.append({
            "field_name": major,
            "field_name_ar": r.get("field_name_ar", major),
            "score": float(score),
            "match_strength": match_en,
            "match_strength_ar": match_ar,
            "reasons": (reasons_ar if lang == 'ar' else reasons_en),
            "reasons_en": reasons_en,
            "reasons_ar": reasons_ar,
            "subfields": [],
            "subfields_ar": [],
        })

    recommendations_payload = {
        "recommendations": rec_items,
        "university_options": []
    }

    return render_template('recommendations.html', language=lang, recommendations=recommendations_payload, row_id=selected.get("student_id", "ranked"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
