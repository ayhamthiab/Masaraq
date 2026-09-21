import cv2
import easyocr
import re
import json
from typing import Dict, List, Optional
from google import genai

GEMINI_API_KEY = "AIzaSyAi0-7aJM20Jt3mhVPfkmBx91x8gXFAt0E"


class DiplomaScanner:
    CORE_SUBJECTS_PER_BRANCH: Dict[str, List[str]] = {
        "علمي": ["الرياضيات", "الفيزياء", "الكيمياء", "الأحياء"],
        "ادبي": ["اللغة العربية", "اللغة الإنجليزية", "التاريخ", "الجغرافيا"],
        "شرعي": ["القرآن الكريم و علومة", "اللغة العربية"],
        "زراعي": ["الأحياء", "الكيمياء"],
        "صناعي": ["الرياضيات", "الفيزياء", "التكنولوجيا"],
        "فندقي او الاقتصاد منزلي": ["الإدارة والاقتصاد"],
        " الاقتصاد منزلي": [" العلوم المهنية"],
        "تكنولوجي": ["التكنولوجيا", "الرياضيات"],
        "ريادة الأعمال": ["الإدارة والاقتصاد"],
    }
    CORE_SUBJECTS_PER_BRANCH_English: Dict[str, List[str]] = {
        "Scientific": ["Mathematics", "Physics", "Chemistry", "Biology"],
        "Literary": ["Arabic Language", "English Language", "History", "Geography"],
        "Sharia": ["The Holy Quran and its sciences", "Arabic Language"],
        "Agricultural": ["Biology", "Chemistry"],
        "Industrial": ["Mathematics", "Physics", "Technology"],
        "Hotel and Home Economics": ["Management and Economics"],
        "Home Economics": ["Professional sciences"],
        "Technological": ["Technology", "Mathematics"],
        "Entrepreneurship and Business": ["Management and Economics"],
    }

    SUBJECTS_AR_TO_CODE = {
        "الرياضيات": "math",
        "الفيزياء": "physics",
        "الكيمياء": "chemistry",
        "الأحياء": "biology",
        "اللغة العربية": "arabic_lang",
        "اللغة الإنجليزية": "english_lang",
        "التاريخ": "history",
        "العلوم المهنية": "professional_sciences",
        "الجغرافيا": "geography",
        "التكنولوجيا": "technology",
        "تكنولوجيا المعلومات": "it",
        "الإدارة والاقتصاد": "management_economics",
        "القرآن الكريم و علومة": "The Holy Quran and its sciences",
    }
    SUBJECTS_EN_TO_CODE = {
        "Mathematics": "math",
        "Physics": "physics",
        "Chemistry": "chemistry",
        "Biology": "biology",
        "Arabic Language": "arabic_lang",
        "English Language": "english_lang",
        "History": "history",
        "Geography": "geography",
        "Professional sciences": "professional_sciences",
        "Technology": "technology",
        "Information Technology": "it",
        "Management and Economics": "management_economics",
        "The Holy Quran and its sciences": "quran",
    }
    SUBJECT_CODE_TO_PRETTY = {
        "math": "mathematics",
        "physics": "physics",
        "chemistry": "chemistry",
        "biology": "biology",
        "arabic_lang": "arabic",
        "english_lang": "english",
        "history": "history",
        "geography": "geography",
        "technology": "technology",
        "it": "it",
        "management_economics": "management_and_economics",
        "quran": "quran",
        "Professional sciences": "sciences",
    }

    BRANCH_AR_TO_EN_STREAM = {
        "العلمي": "scientific",
        "علمي": "scientific",
        "الأدبي": "literary",
        "ادبي": "literary",
        "الصناعي": "industrial",
        "الريادة والأعمال": "entrepreneurship and business",
        "الشرعي": "sharia",
        "الاقتصاد المنزلي": "hotel and home economics",
        "فندقي او اقتصاد منزلي": "hotel and home economics",
        "اقتصاد منزلي": "  home economics",
        "كفاءة مهنية اقتصاد منزلي": "hotel and home economics",
        "زراعي": "agricultural",
        "تكنولوجي": "technological",
    }
    BRANCH_EN_TO_STREAM = {
        "Scientific": "scientific",
        "Literary": "literary",
        "Industrial": "industrial",
        "Entrepreneurship and Business": "entrepreneurship and business",
        "Sharia": "sharia",
        "Hotel and Home Economics": "hotel and home economics",
        "  Home Economics": " home economics",
        "Agricultural": "agricultural",
        "Technological": "technological",
    }

    AR_TO_EN_SUBJECT = {
        "الرياضيات": "Mathematics",
        "الفيزياء": "Physics",
        "الكيمياء": "Chemistry",
        "الأحياء": "Biology",
        "اللغة العربية": "Arabic Language",
        "اللغة الإنجليزية": "English Language",
        "التاريخ": "History",
        "الجغرافيا": "Geography",
        "التكنولوجيا": "Technology",
        "تكنولوجيا المعلومات": "Information Technology",
        "الإدارة والاقتصاد": "Management and Economics",
        "القرآن الكريم و علومة": "The Holy Quran and its sciences",
        "العلوم المهنية": "Professional sciences",
    }
    EN_TO_AR_SUBJECT = None  

    DESCRIPTION = """مرحباً،

أريد منك أن تقوم بترتيب وتحليل النص المرفق بعناية فائقة جداً، وتستخرج منه المعلومات التالية بدقة عالية جداً بحيث لا يكون هناك مجال للأخطاء:

1. اسم  للطالب.  
2. رقم الجلوس المكون من 8 خانات.  
3. المعدل الكامل (رقم النسبة المئوية مع الرمز % إذا وُجد).  
4. العلامات لكل مادةاساسيه موجودة في الفرع الدراسي الخاص بالطالب.
5.المخرجات لازم يكون ب اللغة الانجليزيه
6.قم بتحويل العلامات التي نهايتها العظمى من 200 او من 150 الى 100 قبل اخراج النتائج 
7.اذا كان في شهادة الطال انه "غير مستكمل " او "لم يستكمل متطلبات النجاح" اخرج ان معدله و علامات مواده =0 
--- الفروع و المواد لكل فرع 
CORE_SUBJECTS_PER_BRANCH: Dict[str, List[str]] = {
    "علمي": ["الرياضيات", "الفيزياء", "الكيمياء", "الأحياء"],
    "ادبي": ["اللغة العربية", "اللغة الإنجليزية", "التاريخ", "الجغرافيا"],
    "شرعي": ["القرآن الكريم و علومة", "اللغة العربية"],
    "زراعي": ["الأحياء", "الكيمياء"],
    "صناعي": ["الرياضيات", "الفيزياء", "التكنولوجيا"],
    "الاقتصاد منزلي": ["الإدارة والاقتصاد"],
    " الاقتصاد منزلي": [" العلوم المهنية"],
    "تكنولوجي": ["التكنولوجيا", "الرياضيات"],
    "ريادة الأعمال": ["الإدارة والاقتصاد"],
} 
CORE_SUBJECTS_PER_BRANCH_English: Dict[str, List[str]] = {
    "Scientific": ["Mathematics", "Physics", "Chemistry", "Biology"],
    "Literary": ["Arabic Language", "English Language", "History", "Geography"],
    "Sharia": ["The Holy Quran and its sciences", "Arabic Language"],
    "Agricultural": ["Biology", "Chemistry"],
    "Industrial": ["Mathematics", "Physics", "Technology"],
    "Hotel and Home Economics": ["Management and Economics"],
    "Home Economics": ["Professional sciences"],
    "Technological": ["Technology", "Mathematics"],
    "Entrepreneurship and Business": ["Management and Economics"],
}
---
-- التزم حرفيا ب الاسماء 
**مهم جداً:**  

- أرجو أن تلتزم بصيغة إخراج بيانات **ثابتة ومنسقة** تسهل عليَّ استخراج المعلومات برمجياً بدون أخطاء.  
-ما يكون في اي اشي قبلها او بعدها او اضافي فقط الصيغة الي رح اعطيك اياها 
```json
{
  "name": "اسم الطالب ",
  "average": "المعدل كاملاً، مثلاً % 95.6  ",
  "branch": "اسم الفرع بدقة",
  "seat_number": "رقم الجلوس 8 خانات",
  "subjects": {
    "اسم المادة 1": "علامتها",
    "اسم المادة 2": "علامتها",
    "...": "..."
  }
}
"""

    def __init__(self, api_key: str = GEMINI_API_KEY, gpu: bool = False):
        
        self.EN_TO_AR_SUBJECT = {v: k for k, v in self.AR_TO_EN_SUBJECT.items()}
        
        self.reader = easyocr.Reader(['ar', 'en'], gpu=gpu)
        self.client = genai.Client(api_key=api_key)

    def run_ocr(self, image_path: str) -> Optional[Dict]:
        """Run OCR + Gemini parsing. Returns {'stream': str, 'scores': {...}} or None."""
        img = self._load_image(image_path)
        if img is None:
            return None

        gray = self._preprocess_image(img)
        extracted_text = self._ocr_extract_text(gray)
        prompt = self._build_prompt(extracted_text)

        reply = self._ask_gemini_once(prompt)
        data = self._extract_json_from_reply(reply)
        if not data:
            return None

        return self._build_scores_from_data(data)

    def _ask_gemini_once(self, text: str) -> str:
        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=text
        )
        return response.text.strip()

    def _load_image(self, path: str):
        return cv2.imread(path)

    def _preprocess_image(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        return gray

    def _ocr_extract_text(self, gray_img) -> str:
        results = self.reader.readtext(gray_img)
        return "\n".join([res[1] for res in results])

    def _build_prompt(self, extracted_text: str) -> str:
        return self.DESCRIPTION + "\n" + extracted_text

    def _extract_json_from_reply(self, reply: str):
        m = re.search(r"```json\s*(\{.*?\})\s*```", reply, re.DOTALL)
        return json.loads(m.group(1)) if m else None

    def _to_number(self, value):
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return int(value) if float(value).is_integer() else float(value)
        m = re.search(r"(\d+(?:\.\d+)?)", str(value))
        if not m:
            return None
        num = m.group(1)
        if "." in num:
            f = float(num)
            return int(f) if f.is_integer() else f
        return int(num)

    def _pick_stream(self, branch_text: str) -> str:
        if not branch_text:
            return ""
        return (
            self.BRANCH_AR_TO_EN_STREAM.get(branch_text)
            or self.BRANCH_EN_TO_STREAM.get(branch_text)
            or branch_text.lower()
        )

    def _core_subjects_for_branch(self, branch_text: str) -> List[str]:
        if branch_text in self.CORE_SUBJECTS_PER_BRANCH:
            return self.CORE_SUBJECTS_PER_BRANCH[branch_text]
        if branch_text in self.CORE_SUBJECTS_PER_BRANCH_English:
            return self.CORE_SUBJECTS_PER_BRANCH_English[branch_text]
        return []

    def _lookup_subject_value(self, subjects_found: Dict[str, str], required_name: str):
        if required_name in subjects_found:
            return subjects_found[required_name]
        if required_name in self.AR_TO_EN_SUBJECT:
            alt = self.AR_TO_EN_SUBJECT[required_name]
            return subjects_found.get(alt)
        if required_name in self.EN_TO_AR_SUBJECT:
            alt = self.EN_TO_AR_SUBJECT[required_name]
            return subjects_found.get(alt)
        return None

    def _pretty_key_from_subject_name(self, name: str) -> str:
        code = (
            self.SUBJECTS_AR_TO_CODE.get(name)
            or self.SUBJECTS_EN_TO_CODE.get(name)
        )
        if not code:
            return re.sub(r"\s+", "_", name.strip().lower())
        return self.SUBJECT_CODE_TO_PRETTY.get(code, code)

    def _build_scores_from_data(self, data: Dict) -> Dict:
        average = data.get("average")
        branch = data.get("branch")
        subjects_found = data.get("subjects", {})

        stream_en = self._pick_stream(branch)
        scores: Dict[str, float] = {}

        overall_avg = self._to_number(average)
        if overall_avg is not None:
            scores["overall_average"] = overall_avg

        required_subjects = self._core_subjects_for_branch(branch)
        for req_name in required_subjects:
            raw_val = self._lookup_subject_value(subjects_found, req_name)
            val = self._to_number(raw_val)
            if val is None:
                continue
            pretty_key = self._pretty_key_from_subject_name(req_name)
            scores[pretty_key] = val

        return {"stream": stream_en, "scores": scores}
