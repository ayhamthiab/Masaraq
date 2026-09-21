# University Recommendation System (Masaraq) | نظام توصية الجامعات (مسارك)

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/Flask-2.3.3-green.svg" alt="Flask 2.3.3">
  <img src="https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg" alt="scikit-learn 1.3.0">
  <img src="https://img.shields.io/badge/OCR-Tesseract-purple.svg" alt="Tesseract OCR">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
</div>

<div align="center">
  <h3>English | <a href="#arabic-section">العربية</a></h3>
</div>

---

## 🎯 Project Overview

The University Recommendation System (Masaraq) is a machine learning-based application designed to help Palestinian high school (Tawjihi) graduates make informed decisions about their university education. The system analyzes academic performance, personal interests, and socioeconomic factors to provide personalized university field recommendations.

### Key Features

- **OCR Transcript Analysis**: Automatically extracts grades from Tawjihi diploma images/PDFs
- **Bilingual Support**: Full Arabic and English language support
- **Interest Assessment**: Comprehensive questionnaire to evaluate student interests and preferences
- **Machine Learning Model**: Uses ensemble ML models to provide personalized field recommendations
- **Academic Eligibility Check**: Verifies student eligibility for different university fields
- **Responsive Web Interface**: User-friendly interface accessible on various devices

## 🏗️ System Architecture

The system follows a modular architecture with the following main components:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  Data Input     │────▶│  Processing     │────▶│  Recommendation │
│  (OCR/Manual)   │     │  Pipeline       │     │  Engine         │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                      │                       │
         │                      │                       │
         ▼                      ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│                      Web Application Layer                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Technology Stack

- **Backend**: Python, Flask
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn (Random Forest, Gradient Boosting, Logistic Regression)
- **OCR**: Tesseract, OpenCV, PyTesseract
- **Frontend**: HTML, CSS, JavaScript, Bootstrap
- **Language Support**: Flask-Babel, PyArabic

## 🔄 Project Flow

The system follows a comprehensive workflow to generate university field recommendations:

### 1. Data Collection

**Option A: OCR Diploma Scanning**
- User uploads their Tawjihi diploma (PDF or image)
- The OCR module (`diploma_scanner.py`) processes the image:
  - Image preprocessing (grayscale conversion, thresholding, denoising)
  - Text extraction using Tesseract OCR with Arabic and English language support
  - Pattern matching to identify grades, subjects, and student information
  - Validation of extracted data

**Option B: Manual Data Entry**
- User manually enters their academic information
- System validates the input data
- Missing fields are identified for completion

### 2. Demographics Collection

- User provides personal and demographic information:
  - Name, gender, age
  - City of residence
  - Family size and income level
  - Parents' education level

### 3. Interest Assessment

- User completes a comprehensive interest assessment questionnaire
- The assessment covers four key areas:
  - Personality traits and preferences
  - Academic subjects of interest
  - Career goals and aspirations
  - Activities and hobbies
- The `interest_assessment.py` module analyzes responses to:
  - Determine personality type (analytical, creative, social, practical)
  - Calculate interest alignment with different university fields
  - Generate interest scores for each field

### 4. Recommendation Generation

- The `recommendation_model.py` module processes all collected data:
  - Academic data is processed and validated
  - Feature engineering creates a comprehensive student profile
  - Ensemble machine learning models (Random Forest, Gradient Boosting, Logistic Regression) generate predictions
  - Model predictions are weighted and combined
  - Academic eligibility rules are applied
  - Interest alignment boosts are calculated
  - Personal factors are considered

### 5. Results Presentation

- The system presents personalized recommendations:
  - Top 3 university fields with match scores
  - Detailed reasons for each recommendation
  - Academic eligibility assessment
  - Potential specializations within each field
  - Admission probability estimates
  - Next steps guidance

## 🧩 Technical Components

### OCR Module (`diploma_scanner.py`)

The OCR module is responsible for extracting academic information from Tawjihi diploma images:

- **DiplomaScanner Class**: Processes diploma images/PDFs
  - Handles both file uploads and byte data
  - Supports multiple image formats and PDF processing
  - Implements image preprocessing techniques for better OCR results
  - Uses pattern matching to extract grades and student information

- **TextExtractor Class**: Validates and processes extracted text
  - Identifies missing fields
  - Validates extraction results
  - Determines student stream based on subjects

### ML Models (`recommendation_model.py`)

The recommendation engine uses ensemble machine learning techniques:

- **UniversityRecommendationModel Class**: Core recommendation engine
  - Feature processing pipeline with numeric and categorical transformers
  - Multiple ML models (Random Forest, Gradient Boosting, Logistic Regression)
  - Model training and evaluation functions
  - Prediction combining with weighted averaging
  - Academic requirements filtering
  - Interest alignment boosting
  - Admission probability calculation

### Interest Assessment (`interest_assessment.py`)

The interest assessment module evaluates student preferences and interests:

- **InterestAssessment Class**: Manages questionnaire and analysis
  - Comprehensive question bank covering multiple categories
  - Response analysis algorithms
  - Personality type determination
  - Field match calculation based on responses
  - Bilingual result formatting

### Web Application (`app.py`)

The Flask web application coordinates the entire system:

- **Route Handlers**: Process user requests and form submissions
- **Session Management**: Maintains user data across multiple steps
- **Template Rendering**: Generates HTML responses with Jinja2 templates
- **API Endpoints**: Provides programmatic access to system functionality

## 🚀 Installation and Setup

### Prerequisites

- Python 3.9 or higher
- Tesseract OCR installed on your system
- Git (for cloning the repository)

### Installation Steps

1. Clone the repository:
   ```bash
   git clone 
   cd Masaraq
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. Install dependencies:
   Note: requirements.txt pins critical versions for reproducibility. easyocr pulls heavy dependencies (torch/torchvision). If you don't need easyocr, remove it from requirements before installing.
   ```bash
   pip install -r requirements.txt
   ```

4. Ensure Tesseract OCR is installed:
   - **Linux**: `sudo apt-get install tesseract-ocr tesseract-ocr-ara`
   - **macOS**: `brew install tesseract tesseract-lang`
   - **Windows**: Download and install from the UB-Mannheim build: https://github.com/UB-Mannheim/tesseract/wiki
   After installing on Windows, ensure the Tesseract binary is in your PATH or configure `pytesseract.pytesseract.tesseract_cmd` in your code, for example:
   ```python
   import pytesseract
   pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
   ```
   If you prefer not to install system Tesseract and want OCR only via easyocr (Python-based), note that easyocr pulls large dependencies (torch/torchvision) and may increase installation time and disk usage.

5. Configure the system (optional):
   - Edit `config.py` to customize settings
   - Adjust OCR settings if needed
   - Modify model parameters if required

## 📋 Usage Guide

### Running the Application

1. Start the web application:
   ```bash
   python run.py
   ```

2. Open your browser and navigate to:
   ```
   http://localhost:5000
   ```
   By default the server listens on port 5000. You can change the port by setting the PORT environment variable before running:
   - Windows (cmd.exe):
     ```
     set PORT=8080 && python run.py
     ```
   - PowerShell:
     ```
     $env:PORT=8080; python run.py
     ```
   - macOS / Linux:
     ```
     PORT=8080 python run.py
     ```

### User Workflow

1. **Start**: Choose language (Arabic or English) and select input method (upload diploma or manual entry)

2. **Academic Data**:
   - Upload Tawjihi diploma OR
   - Manually enter academic information
   - Review and complete any missing fields

3. **Demographics**:
   - Enter personal and demographic information
   - All fields are required for accurate recommendations

4. **Interest Assessment**:
   - Complete the interest questionnaire
   - Answer all questions for best results

5. **Recommendations**:
   - Review personalized field recommendations
   - Check academic eligibility for different fields
   - Explore potential specializations
   - Consider next steps

## 📁 Project Structure

```
university_recommendation_system/
├── config.py                     # Configuration settings
├── requirements.txt              # Dependencies
├── run.py                        # Main entry point
├── run_web_app.py                # Web application runner
├── run_direct.py                 # Direct execution script
├── run_server.py                 # Server deployment script
├── run_test.py                   # Test runner
├── test_model_load.py            # Model loading test
├── data/                         # Data directory
│   ├── uploads/                  # Uploaded diplomas
│   ├── models/                   # Trained ML models
│   ├── generated/                # Generated data
│   ├── raw/                      # Raw data sources
│   └── processed/                # Processed data
└── src/                          # Source code
    ├── data_collection/          # Data collection modules
    │   ├── __init__.py
    │   └── university_scraper.py # University data scraper
    ├── data_processing/          # Data processing modules
    │   ├── __init__.py
    │   └── data_generator.py     # Training data generator
    ├── ml_models/                # ML model implementations
    │   ├── __init__.py
    │   └── recommendation_model.py # Recommendation engine
    ├── ocr_module/               # OCR processing
    │   ├── __init__.py
    │   └── diploma_scanner.py    # Diploma OCR scanner
    ├── utils/                    # Utility functions
    │   ├── __init__.py
    │   └── interest_assessment.py # Interest assessment
    └── web_app/                  # Flask web application
        ├── __init__.py
        ├── app.py                # Flask application
        ├── templates/            # HTML templates
        │   ├── base.html         # Base template
        │   ├── index.html        # Home page
        │   ├── upload_diploma.html # Diploma upload page
        │   ├── manual_entry.html # Manual entry page
        │   ├── demographics_entry.html # Demographics page
        │   ├── interest_assessment.html # Interest assessment
        │   └── recommendations.html # Recommendations page
        └── static/               # Static files (CSS, JS)
```

## 🧪 Development and Testing

### Running Tests

```bash
python run_test.py
```

### Adding New Features

1. **New University Fields**:
   - Add field definitions to `config.py` in the `UNIVERSITY_FIELDS` dictionary
   - Include eligibility criteria and minimum score requirements

2. **Extending the OCR Module**:
   - Add new pattern recognition in `diploma_scanner.py`
   - Update the validation logic in the `TextExtractor` class

3. **Enhancing the ML Model**:
   - Modify feature engineering in `recommendation_model.py`
   - Adjust model weights in the `_combine_predictions` method

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.



---

<a name="arabic-section"></a>

# نظام توصية الجامعات (مسارك) | University Recommendation System (Masaraq)

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/Flask-2.3.3-green.svg" alt="Flask 2.3.3">
  <img src="https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg" alt="scikit-learn 1.3.0">
  <img src="https://img.shields.io/badge/OCR-Tesseract-purple.svg" alt="Tesseract OCR">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
</div>

<div align="center">
  <h3><a href="#university-recommendation-system-masaraq--نظام-توصية-الجامعات-مسارك">English</a> | العربية</h3>
</div>

---

## 🎯 نظرة عامة على المشروع

نظام توصية الجامعات (مسارك) هو تطبيق قائم على التعلم الآلي مصمم لمساعدة خريجي الثانوية العامة (التوجيهي) الفلسطينيين على اتخاذ قرارات مستنيرة بشأن تعليمهم الجامعي. يقوم النظام بتحليل الأداء الأكاديمي والاهتمامات الشخصية والعوامل الاجتماعية والاقتصادية لتقديم توصيات مخصصة للتخصصات الجامعية.

### الميزات الرئيسية

- **تحليل الشهادات باستخدام OCR**: استخراج الدرجات تلقائيًا من صور/ملفات PDF لشهادة التوجيهي
- **دعم ثنائي اللغة**: دعم كامل للغتين العربية والإنجليزية
- **تقييم الاهتمامات**: استبيان شامل لتقييم اهتمامات الطالب وتفضيلاته
- **نموذج التعلم الآلي**: يستخدم نماذج التعلم الآلي المجمعة لتقديم توصيات مخصصة
- **التحقق من الأهلية الأكاديمية**: التحقق من أهلية الطالب للتخصصات الجامعية المختلفة
- **واجهة ويب متجاوبة**: واجهة سهلة الاستخدام يمكن الوصول إليها على مختلف الأجهزة

## 🏗️ بنية النظام

يتبع النظام بنية معيارية مع المكونات الرئيسية التالية:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  إدخال البيانات  │────▶│  معالجة         │────▶│  محرك           │
│  (OCR/يدوي)     │     │  البيانات       │     │  التوصيات       │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                      │                       │
         │                      │                       │
         ▼                      ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│                      طبقة تطبيق الويب                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### التقنيات المستخدمة

- **الخلفية**: Python, Flask
- **معالجة البيانات**: Pandas, NumPy
- **التعلم الآلي**: Scikit-learn (Random Forest, Gradient Boosting, Logistic Regression)
- **OCR**: Tesseract, OpenCV, PyTesseract
- **الواجهة الأمامية**: HTML, CSS, JavaScript, Bootstrap
- **دعم اللغة**: Flask-Babel, PyArabic

## 🔄 تدفق المشروع

يتبع النظام سير عمل شامل لتوليد توصيات التخصصات الجامعية:

### 1. جمع البيانات

**الخيار أ: مسح شهادة التوجيهي باستخدام OCR**
- يقوم المستخدم بتحميل شهادة التوجيهي الخاصة به (PDF أو صورة)
- تقوم وحدة OCR (`diploma_scanner.py`) بمعالجة الصورة:
  - معالجة مسبقة للصورة (تحويل إلى اللون الرمادي، العتبة، إزالة الضوضاء)
  - استخراج النص باستخدام Tesseract OCR مع دعم اللغتين العربية والإنجليزية
  - مطابقة الأنماط لتحديد الدرجات والمواد ومعلومات الطالب
  - التحقق من صحة البيانات المستخرجة

**الخيار ب: إدخال البيانات يدويًا**
- يقوم المستخدم بإدخال معلوماته الأكاديمية يدويًا
- يتحقق النظام من صحة البيانات المدخلة
- يتم تحديد الحقول المفقودة لاستكمالها

### 2. جمع البيانات الديموغرافية

- يقدم المستخدم معلومات شخصية وديموغرافية:
  - الاسم، الجنس، العمر
  - مدينة الإقامة
  - حجم الأسرة ومستوى الدخل
  - المستوى التعليمي للوالدين

### 3. تقييم الاهتمامات

- يكمل المستخدم استبيان تقييم الاهتمامات الشامل
- يغطي التقييم أربعة مجالات رئيسية:
  - سمات الشخصية والتفضيلات
  - المواد الأكاديمية ذات الاهتمام
  - الأهداف المهنية والطموحات
  - الأنشطة والهوايات
- تقوم وحدة `interest_assessment.py` بتحليل الإجابات لـ:
  - تحديد نوع الشخصية (تحليلية، إبداعية، اجتماعية، عملية)
  - حساب مدى التوافق مع مختلف التخصصات الجامعية
  - توليد درجات الاهتمام لكل تخصص

### 4. توليد التوصيات

- تقوم وحدة `recommendation_model.py` بمعالجة جميع البيانات المجمعة:
  - تتم معالجة البيانات الأكاديمية والتحقق من صحتها
  - تقوم هندسة الميزات بإنشاء ملف تعريف شامل للطالب
  - تقوم نماذج التعلم الآلي المجمعة (Random Forest, Gradient Boosting, Logistic Regression) بتوليد التنبؤات
  - يتم ترجيح تنبؤات النموذج ودمجها
  - يتم تطبيق قواعد الأهلية الأكاديمية
  - يتم حساب تعزيزات توافق الاهتمامات
  - يتم النظر في العوامل الشخصية

### 5. عرض النتائج

- يقدم النظام توصيات مخصصة:
  - أفضل 3 تخصصات جامعية مع درجات التطابق
  - أسباب مفصلة لكل توصية
  - تقييم الأهلية الأكاديمية
  - التخصصات المحتملة داخل كل مجال
  - تقديرات احتمالية القبول
  - إرشادات للخطوات التالية

## 🧩 المكونات التقنية

### وحدة OCR (`diploma_scanner.py`)

وحدة OCR مسؤولة عن استخراج المعلومات الأكاديمية من صور شهادة التوجيهي:

- **فئة DiplomaScanner**: تعالج صور/ملفات PDF للشهادات
  - تتعامل مع تحميلات الملفات وبيانات البايت
  - تدعم تنسيقات صور متعددة ومعالجة PDF
  - تنفذ تقنيات المعالجة المسبقة للصور للحصول على نتائج OCR أفضل
  - تستخدم مطابقة الأنماط لاستخراج الدرجات ومعلومات الطالب

- **فئة TextExtractor**: تتحقق من صحة النص المستخرج وتعالجه
  - تحدد الحقول المفقودة
  - تتحقق من نتائج الاستخراج
  - تحدد فرع الطالب بناءً على المواد

### نماذج التعلم الآلي (`recommendation_model.py`)

يستخدم محرك التوصيات تقنيات التعلم الآلي المجمعة:

- **فئة UniversityRecommendationModel**: محرك التوصيات الأساسي
  - خط أنابيب معالجة الميزات مع محولات رقمية وفئوية
  - نماذج تعلم آلي متعددة (Random Forest, Gradient Boosting, Logistic Regression)
  - وظائف تدريب النموذج وتقييمه
  - دمج التنبؤات باستخدام المتوسط المرجح
  - تصفية المتطلبات الأكاديمية
  - تعزيز توافق الاهتمامات
  - حساب احتمالية القبول

### تقييم الاهتمامات (`interest_assessment.py`)

وحدة تقييم الاهتمامات تقيم تفضيلات واهتمامات الطالب:

- **فئة InterestAssessment**: تدير الاستبيان والتحليل
  - بنك أسئلة شامل يغطي فئات متعددة
  - خوارزميات تحليل الإجابات
  - تحديد نوع الشخصية
  - حساب تطابق التخصص بناءً على الإجابات
  - تنسيق النتائج بلغتين

### تطبيق الويب (`app.py`)

تطبيق Flask للويب ينسق النظام بأكمله:

- **معالجات المسارات**: تعالج طلبات المستخدم وإرسال النماذج
- **إدارة الجلسات**: تحافظ على بيانات المستخدم عبر خطوات متعددة
- **عرض القوالب**: ينشئ استجابات HTML باستخدام قوالب Jinja2
- **نقاط نهاية API**: توفر وصولاً برمجيًا إلى وظائف النظام

## 🚀 التثبيت والإعداد

### المتطلبات الأساسية

- Python 3.9 أو أعلى
- Tesseract OCR مثبت على نظامك
- Git (لاستنساخ المستودع)

### خطوات التثبيت

1. استنساخ المستودع:
   ```bash
   git clone 
   cd Masaraq
   ```

2. إنشاء بيئة افتراضية:
   ```bash
   python -m venv venv
   source venv/bin/activate  # على Windows، استخدم: venv\Scripts\activate
   ```

3. تثبيت التبعيات:
   ```bash
   pip install -r requirements.txt
   ```

4. تأكد من تثبيت Tesseract OCR:
   - **Linux**: `sudo apt-get install tesseract-ocr tesseract-ocr-ara`
   - **macOS**: `brew install tesseract tesseract-lang`
   - **Windows**: قم بالتنزيل والتثبيت من [Tesseract GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

5. تكوين النظام (اختياري):
   - قم بتحرير `config.py` لتخصيص الإعدادات
   - ضبط إعدادات OCR إذا لزم الأمر
   - تعديل معلمات النموذج إذا لزم الأمر

## 📋 دليل الاستخدام

### تشغيل التطبيق

1. بدء تشغيل تطبيق الويب:
   ```bash
   python run.py
   ```

2. افتح متصفحك وانتقل إلى:
   ```
   http://localhost:8080
   ```

### سير عمل المستخدم

1. **البداية**: اختر اللغة (العربية أو الإنجليزية) وحدد طريقة الإدخال (تحميل الشهادة أو الإدخال اليدوي)

2. **البيانات الأكاديمية**:
   - قم بتحميل شهادة التوجيهي أو
   - أدخل المعلومات الأكاديمية يدويًا
   - راجع وأكمل أي حقول مفقودة

3. **البيانات الديموغرافية**:
   - أدخل المعلومات الشخصية والديموغرافية
   - جميع الحقول مطلوبة للحصول على توصيات دقيقة

4. **تقييم الاهتمامات**:
   - أكمل استبيان الاهتمامات
   - أجب على جميع الأسئلة للحصول على أفضل النتائج

5. **التوصيات**:
   - راجع توصيات التخصصات المخصصة
   - تحقق من الأهلية الأكاديمية للتخصصات المختلفة
   - استكشف التخصصات المحتملة
   - فكر في الخطوات التالية

## 📁 هيكل المشروع

```
university_recommendation_system/
├── config.py                     # إعدادات التكوين
├── requirements.txt              # التبعيات
├── run.py                        # نقطة الدخول الرئيسية
├── run_web_app.py                # مشغل تطبيق الويب
├── run_direct.py                 # نص التنفيذ المباشر
├── run_server.py                 # نص نشر الخادم
├── run_test.py                   # مشغل الاختبار
├── test_model_load.py            # اختبار تحميل النموذج
├── data/                         # دليل البيانات
│   ├── uploads/                  # الشهادات المحملة
│   ├── models/                   # نماذج التعلم الآلي المدربة
│   ├── generated/                # البيانات المولدة
│   ├── raw/                      # مصادر البيانات الخام
│   └── processed/                # البيانات المعالجة
└── src/                          # الشفرة المصدرية
    ├── data_collection/          # وحدات جمع البيانات
    │   ├── __init__.py
    │   └── university_scraper.py # مستخرج بيانات الجامعات
    ├── data_processing/          # وحدات معالجة البيانات
    │   ├── __init__.py
    │   └── data_generator.py     # مولد بيانات التدريب
    ├── ml_models/                # تنفيذات نماذج التعلم الآلي
    │   ├── __init__.py
    │   └── recommendation_model.py # محرك التوصيات
    ├── ocr_module/               # معالجة OCR
    │   ├── __init__.py
    │   └── diploma_scanner.py    # ماسح شهادات OCR
    ├── utils/                    # وظائف مساعدة
    │   ├── __init__.py
    │   └── interest_assessment.py # تقييم الاهتمامات
    └── web_app/                  # تطبيق Flask للويب
        ├── __init__.py
        ├── app.py                # تطبيق Flask
        ├── templates/            # قوالب HTML
        │   ├── base.html         # القالب الأساسي
        │   ├── index.html        #
