# -*- coding: utf-8 -*-
"""
Interest Assessment (bilingual, no I/O)
- Likert: values '1'..'5'
- Multiple-choice: values always Arabic text
- Order: q1..q29, q30,q31 (timed), q32..q37
"""
from typing import List, Dict, Any
import logging
logger = logging.getLogger(__name__)

SECTION_TO_UI = {
    'social_factors': 'personality',
    'personality': 'subjects',
    'skills_and_preferences': 'career_goals',
    'activities': 'activities',
}
LIKERT_VALUES = ['5','4','3','2','1']

EMBEDDED_CONFIG_DATA = {
    "social_factors": {
        "questions": [
            {"id":"q1","type":"likert",
             "text_ar":"إلى أي درجة يؤثر رأي والديك في اختيار التخصص؟",
             "text_en":"To what extent do your parents' opinions influence your major choice?",
             "options_ar":["أوافق بشدة","أوافق","محايد","لا أوافق","لا أوافق بشدة"],
             "options_en":["Strongly agree","Agree","Neutral","Disagree","Strongly disagree"]},
            {"id":"q2","type":"multiple_choice",
             "text_ar":"هل ستختار تخصصًا مختلفًا لو كان رأي أهلك مخالف لرغبتك؟",
             "text_en":"Would you choose a different major if your parents' opinion conflicted with your preference?",
             "options_ar":["نعم","لا","ربما"],
             "options_en":["Yes","No","Maybe"]},
            {"id":"q3","type":"likert",
             "text_ar":"مدى أهمية أن يكون تخصصك ذو مكانة عالية في المجتمع؟",
             "text_en":"How important is it for your major to have high social status?",
             "options_ar":["أوافق بشدة","أوافق","محايد","لا أوافق","لا أوافق بشدة"],
             "options_en":["Strongly agree","Agree","Neutral","Disagree","Strongly disagree"]},
            {"id":"q4","type":"multiple_choice",
             "text_ar":"لو كان تخصص ذو مكانة اجتماعية عالية لكن لا تحبه، هل ستختاره؟",
             "text_en":"If a major has high social status but you do not like it, would you choose it?",
             "options_ar":["نعم","لا","يعتمد"],
             "options_en":["Yes","No","It depends"]},
            {"id":"q5","type":"likert",
             "text_ar":"ما مدى اهتمامك بفرص العمل المستقبلية عند اختيار التخصص؟",
             "text_en":"How much do you consider future job opportunities when choosing a major?",
             "options_ar":["أوافق بشدة","أوافق","محايد","لا أوافق","لا أوافق بشدة"],
             "options_en":["Strongly agree","Agree","Neutral","Disagree","Strongly disagree"]},
            {"id":"q6","type":"multiple_choice",
             "text_ar":"إذا كان التخصص الذي تحبه قليل الطلب في السوق، ماذا ستفعل؟",
             "text_en":"If the major you like has low market demand, what would you do?",
             "options_ar":["أختاره","أبحث عن بديل","أختار تخصصًا قريبًا منه"],
             "options_en":["Choose it","Look for an alternative","Choose a related major"]},
            {"id":"q7","type":"multiple_choice",
             "text_ar":"ما هو مستوى متوسط دخل الأسرة؟",
             "text_en":"What is your household's average income level?",
             "options_ar":["منخفض","متوسط","مرتفع"],
             "options_en":["Low","Medium","High"]},
        ]
    },
    "personality": {
        "questions": [
            {"id":"q8","type":"likert",
             "text_ar":"أشعر بالارتباك والتوتر عند التحدث أمام مجموعة من الأشخاص",
             "text_en":"I feel nervous and tense when speaking in front of a group of people",
             "options_ar":["أوافق بشدة","أوافق","محايد","لا أوافق","لا أوافق بشدة"],
             "options_en":["Strongly agree","Agree","Neutral","Disagree","Strongly disagree"]},
            {"id":"q9","type":"likert",
             "text_ar":"هل تضطر لكتابة ما تود شرائه قبل الذهاب إلى السوق؟",
             "text_en":"Do you need to write down what you want to buy before going to the market?",
             "options_ar":["أوافق بشدة","أوافق","محايد","لا أوافق","لا أوافق بشدة"],
             "options_en":["Strongly agree","Agree","Neutral","Disagree","Strongly disagree"]},
            {"id":"q10","type":"likert",
             "text_ar":"أشعر بالمسؤولية عندما توكل إلي مهمة ما",
             "text_en":"I feel responsible when I am assigned a task",
             "options_ar":["أوافق بشدة","أوافق","محايد","لا أوافق","لا أوافق بشدة"],
             "options_en":["Strongly agree","Agree","Neutral","Disagree","Strongly disagree"]},
            {"id":"q11","type":"likert",
             "text_ar":"إذا رأيت طفلًا يحاول قطع الشارع، لا أتردد لمساعدته",
             "text_en":"If I see a child trying to cross the street, I do not hesitate to help",
             "options_ar":["أوافق بشدة","أوافق","محايد","لا أوافق","لا أوافق بشدة"],
             "options_en":["Strongly agree","Agree","Neutral","Disagree","Strongly disagree"]},
            {"id":"q12","type":"likert",
             "text_ar":"أفضل العمل لوحدي عن العمل مع مجموعة",
             "text_en":"I prefer working alone rather than in a group",
             "options_ar":["أوافق بشدة","أوافق","محايد","لا أوافق","لا أوافق بشدة"],
             "options_en":["Strongly agree","Agree","Neutral","Disagree","Strongly disagree"]},
            {"id":"q13","type":"likert",
             "text_ar":"أمتنع أحيانًا عن المشاركة في الحصص الصفية بسبب التوتر",
             "text_en":"Sometimes I refrain from participating in class due to stress",
             "options_ar":["أوافق بشدة","أوافق","محايد","لا أوافق","لا أوافق بشدة"],
             "options_en":["Strongly agree","Agree","Neutral","Disagree","Strongly disagree"]},
            {"id":"q14","type":"likert",
             "text_ar":"أنجح بالتأثير على الآخرين وإقناعهم",
             "text_en":"I succeed in influencing and persuading others",
             "options_ar":["أوافق بشدة","أوافق","محايد","لا أوافق","لا أوافق بشدة"],
             "options_en":["Strongly agree","Agree","Neutral","Disagree","Strongly disagree"]},
            {"id":"q15","type":"multiple_choice",
             "text_ar":"طلب منك صديقك إثبات أن الألم شعور حقيقي، ماذا تفعل؟",
             "text_en":"A friend asked you to prove that pain is a real feeling. What do you do?",
             "options_ar":["تقرصه لتثبت","تشرح له عن النواقل","تشرح له حقيقة شعور الألم"],
             "options_en":["Pinch them to prove it","Explain neurotransmitters","Explain the nature of pain"]},
            {"id":"q16","type":"multiple_choice",
             "text_ar":"تم منحك عملًا في شركة ناشئة، ردة فعلك؟",
             "text_en":"You were offered a job at a startup. Your reaction?",
             "options_ar":["سأتحمس لذلك مع بقائي حذرًا","سأكون قلقًا","سأكون متحمسًا","غالبًا لا أجازف"],
             "options_en":["Excited but cautious","I would be anxious","I would be excited","I usually avoid risks"]},
            {"id":"q17","type":"multiple_choice",
             "text_ar":"أي جملة توافق عليها أكثر؟",
             "text_en":"Which statement do you agree with most?",
             "options_ar":["الاختلاف والتنوع هما أساس الحياة","كل شيء له وقته ومكانه","من يصل مبكرًا يحصد نجاحًا","استرح واستمتع"],
             "options_en":["Diversity is the essence of life","Everything has its time and place","The early bird gets success","Rest and enjoy"]},
            {"id":"q18","type":"likert",
             "text_ar":"أفكر بشكل منطقي وأهتم بالقواعد أكثر من المشاعر",
             "text_en":"I think logically and care about rules more than emotions",
             "options_ar":["أوافق بشدة","أوافق","محايد","لا أوافق","لا أوافق بشدة"],
             "options_en":["Strongly agree","Agree","Neutral","Disagree","Strongly disagree"]},
            {"id":"q19","type":"multiple_choice",
             "text_ar":"أستطيع الدراسة أو العمل خلف المكتب لمدة طويلة",
             "text_en":"I can study or work at a desk for a long time",
             "options_ar":["لا مشكلة لدي","فقط إذا كنت أحب ما أدرسه","مع تغيير المكان","لا أحب الأعمال الكتبية"],
             "options_en":["No problem","Only if I like the subject","With a change of setting","I don't like desk work"]},
        ]
    },
    "skills_and_preferences": {
        "questions": [
            {"id":"q20","type":"likert",
             "text_ar":"أهرع عند رؤيتي للدم ولا أستطيع تحمله",
             "text_en":"I rush away at the sight of blood and cannot tolerate it",
             "options_ar":["أوافق بشدة","أوافق","محايد","لا أوافق","لا أوافق بشدة"],
             "options_en":["Strongly agree","Agree","Neutral","Disagree","Strongly disagree"]},
            {"id":"q21","type":"likert",
             "text_ar":"أميل نحو التعليم الحرفي أو التطبيقي أكثر من الأكاديمي",
             "text_en":"I lean towards vocational/applied learning more than academic",
             "options_ar":["أوافق بشدة","أوافق","محايد","لا أوافق","لا أوافق بشدة"],
             "options_en":["Strongly agree","Agree","Neutral","Disagree","Strongly disagree"]},
            {"id":"q22","type":"likert",
             "text_ar":"أشعر بالفضول حيال التحديثات التكنولوجية",
             "text_en":"I am curious about technological updates",
             "options_ar":["أوافق بشدة","أوافق","محايد","لا أوافق","لا أوافق بشدة"],
             "options_en":["Strongly agree","Agree","Neutral","Disagree","Strongly disagree"]},
            {"id":"q23","type":"likert",
             "text_ar":"أنتبه إلى الأخطاء اللغوية وأرغب بتصحيحها",
             "text_en":"I notice language mistakes and want to correct them",
             "options_ar":["أوافق بشدة","أوافق","محايد","لا أوافق","لا أوافق بشدة"],
             "options_en":["Strongly agree","Agree","Neutral","Disagree","Strongly disagree"]},
            {"id":"q24","type":"likert",
             "text_ar":"أحرص على متابعة القضايا السياسية وتحليلها",
             "text_en":"I make sure to follow and analyze political issues",
             "options_ar":["أوافق بشدة","أوافق","محايد","لا أوافق","لا أوافق بشدة"],
             "options_en":["Strongly agree","Agree","Neutral","Disagree","Strongly disagree"]},
            {"id":"q25","type":"likert",
             "text_ar":"هل تعاني من حساسية جلدية أو تنفسية حيال النباتات؟",
             "text_en":"Do you have skin or respiratory allergies to plants?",
             "options_ar":["أوافق بشدة","أوافق","محايد","لا أوافق","لا أوافق بشدة"],
             "options_en":["Strongly agree","Agree","Neutral","Disagree","Strongly disagree"]},
            {"id":"q26","type":"likert",
             "text_ar":"هل تتميز بأسلوب لباس مختلف وملفت؟",
             "text_en":"Do you have a distinct and eye-catching clothing style?",
             "options_ar":["أوافق بشدة","أوافق","محايد","لا أوافق","لا أوافق بشدة"],
             "options_en":["Strongly agree","Agree","Neutral","Disagree","Strongly disagree"]},
            {"id":"q27","type":"multiple_choice",
             "text_ar":"(1,2,5,10,13,26,29,48)ما هو الرقم الذي لا ينتمي لمجموعة الأرقام؟",
             "text_en":"Which number does not belong to the group?",
             "options_ar":["لا أعرف","48","26","1"],
             "options_en":["I don't know","48","26","1"]},
            {"id":"q28","type":"multiple_choice",
             "text_ar":"ما هي الكلمة التي لا تنتمي لبقية الكلمات؟",
             "text_en":"Which word does not belong to the rest?",
             "options_ar":["عادل","منحاز","صائب","منصف"],
             "options_en":["Fair","Biased","Correct","Just"]},
            {"id":"q29","type":"multiple_choice",
             "text_ar":"هل يجوز لأي رجل عربي الزواج من أخت أرملته؟",
             "text_en":"Is it permissible for any Arab man to marry his widow's sister?",
             "options_ar":["يجوز له ذلك بعد الطلاق","لا مشكلة إن لم يكن عربي","لن يستطيع الزواج","لا أعرف"],
             "options_en":["Permissible after divorce","No issue if he's not Arab","He cannot marry","I don't know"]},
        ]
    },
    "activities": {
        "questions": [
            {"id":"q32","type":"multiple_choice",
             "text_ar":"أحب ممارسة:",
             "text_en":"I like practicing:",
             "options_ar":["الرسم والتلوين","العزف على آلة موسيقية","التصوير الفوتوغرافي","الكتابة والتأليف"],
             "options_en":["Drawing and coloring","Playing a musical instrument","Photography","Writing and authoring"]},
            {"id":"q33","type":"multiple_choice",
             "text_ar":"أحب متابعة أفلام:",
             "text_en":"I like watching movies that are:",
             "options_ar":["الغموض والتشويق","التحقيقات البوليسية","الخيال والخدع","الدرامية أو الكوميدية","العلمية أو الخيال العلمي"],
             "options_en":["Mystery and suspense","Detective/crime","Fantasy and tricks","Drama or comedy","Science or sci-fi"]},
            {"id":"q34","type":"multiple_choice",
             "text_ar":"أحب الأنشطة الخارجية:",
             "text_en":"For outdoor activities, I prefer:",
             "options_ar":["ركوب الدراجة","السباحة","التخييم والرحلات","ممارسة الألعاب الرياضية"],
             "options_en":["Cycling","Swimming","Camping and trips","Playing sports"]},
            {"id":"q35","type":"multiple_choice",
             "text_ar":"أستمتع أكثر بـ:",
             "text_en":"I enjoy more:",
             "options_ar":["تصميم وبرمجة الكمبيوتر","تصوير وصناعة الفيديوهات","إجراء التجارب العلمية","تعلم لغات جديدة"],
             "options_en":["Computer design & programming","Filming & video production","Conducting scientific experiments","Learning new languages"]},
            {"id":"q36","type":"multiple_choice",
             "text_ar":"عندما أكون لوحدي أحب:",
             "text_en":"When I'm alone I like:",
             "options_ar":["الاستماع للموسيقى أو البودكاست","قراءة كتاب أو مجلة","كتابة مذكرات أو قصص","مشاهدة فيديوهات تعليمية"],
             "options_en":["Listening to music/podcasts","Reading a book or magazine","Writing journals or stories","Watching educational videos"]},
            {"id":"q37","type":"multiple_choice",
             "text_ar":"أجد نفسي أكثر إبداعًا عندما:",
             "text_en":"I find myself most creative when:",
             "options_ar":["أعمل على مشروع فني","أجرب تجربة علمية","أحل لغز أو تحدي","أبتكر قصة أو مشهد"],
             "options_en":["Working on an art project","Trying a scientific experiment","Solving a puzzle or challenge","Creating a story or scene"]},
        ]
    }
}

class InterestAssessment:
    def __init__(self):
        self.sections = EMBEDDED_CONFIG_DATA
        self.ordered_items = self._flatten_in_order()
        expected = [f"q{i}" for i in range(1, 38)]
        got = self.get_question_ids_in_order()
        if len(got) != 37:
            logger.warning("Questions count != 37. Got: %s", got)

    def get_questionnaire(self, language: str = 'ar') -> List[Dict[str, Any]]:
        lng = 'en' if language == 'en' else 'ar'
        questionnaire: List[Dict[str, Any]] = []
        for item in self.ordered_items:
            q_type = item['type']
            labels_ar = item['options_ar']
            labels_en = item['options_en']
            if q_type == 'likert':
                labels_show = labels_en if lng == 'en' else labels_ar
                labels_show = (labels_show + [''] * 5)[:5]
                options = [{ 'value': LIKERT_VALUES[i], 'label': labels_show[i]} for i in range(5)]
            else:
                labels_show = labels_en if lng == 'en' else labels_ar
                # value دائماً عربي
                options = [{ 'value': labels_ar[i], 'label': labels_show[i]} for i in range(len(labels_ar))]
            questionnaire.append({
                'id': item['id'],
                'category': item['ui_category'],
                'question': item['text_en'] if lng == 'en' else item['text_ar'],
                'options': options,
                'type': q_type,
            })
        return questionnaire

    def get_question_ids_in_order(self) -> List[str]:
        """q1..q29 ثم q30,q31 (timed) ثم q32..q37"""
        first = [f"q{i}" for i in range(1, 30)]
        timed = ["q30", "q31"]
        last  = [f"q{i}" for i in range(32, 38)]
        return first + timed + last

    def get_question_types_map(self) -> Dict[str, str]:
        return {q['id']: q['type'] for q in self.ordered_items}

    def _flatten_in_order(self) -> List[Dict[str, Any]]:
        ordered: List[Dict[str, Any]] = []
        for section_key, section_obj in self.sections.items():
            ui_category = SECTION_TO_UI.get(section_key, 'activities')
            for q in section_obj.get('questions', []):
                q_id = str(q.get('id','')).strip()
                q_type = str(q.get('type','')).strip()
                q_text_ar = str(q.get('text_ar','')).strip()
                q_text_en = str(q.get('text_en','')).strip()
                options_ar = [str(o).strip() for o in q.get('options_ar',[])]
                options_en = [str(o).strip() for o in q.get('options_en',[])]
                if not q_id or not q_type or not q_text_ar or not options_ar:
                    continue
                ordered.append({
                    'id': q_id,
                    'type': q_type,
                    'text_ar': q_text_ar,
                    'text_en': q_text_en or q_text_ar,
                    'options_ar': options_ar,
                    'options_en': options_en or options_ar,
                    'ui_category': ui_category,
                })
        return ordered
