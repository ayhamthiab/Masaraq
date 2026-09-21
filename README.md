# MASARAK — University Major Recommendation System

MASARAK is a machine-learning web application that helps students identify suitable university majors. It combines academic performance, Tawjihi branch eligibility, and answers to academic, personal, and social questions to rank the three majors that best match a student's profile and explain each recommendation.

This project was developed by Group 2 at An-Najah National University, Faculty of Engineering and Information Technology.

## Main features

- Top-three university-major recommendations, ranked by predicted compatibility
- Academic, personal, social, and interest-based student profiling
- Eligibility filtering using Tawjihi branch and minimum-GPA requirements
- Explanations describing why each recommended major may be a good fit
- Manual grade entry or diploma image/PDF input through OCR
- Arabic and English support in the web interface and OCR pipeline
- Student-feedback collection for future improvement

## Dataset

No suitable public dataset was available, so the team created a balanced synthetic dataset. The root [`students_data.csv`](students_data.csv) contains **6,300 student records and 44 columns**, including:

- student ID and name
- Tawjihi branch
- overall GPA and core-subject average
- responses to 37 structured questions
- the target university-major label

Because the data is synthetic, the system is a proof of concept and should not replace advice from an academic counselor. The report identifies collecting and validating against real student outcomes as the most important future step.

## Machine-learning models

The project trains and compares three multiclass classifiers:

- **Logistic Regression** (one-vs-rest baseline)
- **Random Forest**
- **Gradient Boosting**

The training code also builds a probability-based soft-voting ensemble and applies program-eligibility masks. The report highlights Gradient Boosting as the best-performing final classifier, while the application retains all three model artifacts for prediction.

## Basic workflow

1. Collect grades through manual entry or OCR and gather questionnaire responses.
2. Clean numeric and questionnaire values and encode categorical answers.
3. Engineer academic features (GPA bins, percentiles, and ratios), normalized interest scores, one-hot questionnaire features, and per-program eligibility flags.
4. Split the prepared data into stratified training, validation, and test sets (80%/10%/10%).
5. Train and tune the classifiers, then rank majors from their predicted probabilities.
6. Present the top three eligible recommendations with supporting explanations.

## Results

The report presents the following rounded results for the selected Gradient Boosting approach:

| Recommendation output | Accuracy |
| --- | ---: |
| Single major | 73% |
| Top 3 majors (adopted approach) | 87% |
| Top 5 majors | 95% |

The saved evaluation output gives the corresponding Gradient Boosting validation Top-3 and Top-5 accuracies as **87.5%** and **95.03%**. On the held-out test set, Gradient Boosting achieved **73.40% accuracy**, **72.90% weighted F1**, **80.93% Top-3 accuracy**, and **85.90% Top-5 accuracy**.

## Technologies

- Python 3.9+
- Flask and Jinja2
- pandas and NumPy
- scikit-learn and joblib
- EasyOCR, OpenCV, Pillow, and Google Gen AI for diploma processing
- HTML, CSS, JavaScript, and Bootstrap templates

## Installation

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements_Group2.txt
Set-Location "Masaraq\ML Yamama\university_recommendation_system"
```

On macOS or Linux, activate the environment with `source .venv/bin/activate` and change directory with `cd "Masaraq/ML Yamama/university_recommendation_system"`.

The OCR path uses EasyOCR and an external Gen AI service, so it may require additional configuration and network access. Manual entry can be used when OCR is unavailable. Keep service credentials in environment variables; do not commit API keys.

## How to run

The application requires trained model files under `src/models/`. Large `.pkl` artifacts are ignored by Git and should not be added to the repository. If the artifacts are unavailable, generate them from the included data:

```powershell
python src/data_processing/data_preparation.py
python src/data_processing/feature_engineering.py
python src/ml_models/model_training.py
```

Then start the web application:

```powershell
python run.py
```

Open <http://localhost:5000>. To use another port in PowerShell:

```powershell
$env:PORT = 8000
python run.py
```

## Project structure

```text
ML_FinalProject_Group2/
├── report.pdf                     # Main project report
├── students_data.csv              # Synthetic student dataset
├── requirements_Group2.txt        # Python dependencies
└── Masaraq/ML Yamama/university_recommendation_system/
    ├── run.py                      # Flask application entry point
    ├── config.py                   # Application configuration
    ├── data/                       # Runtime uploads and generated data
    └── src/
        ├── data/raw/               # Source data and configuration
        ├── data/processed/         # Cleaned and engineered datasets
        ├── data_processing/        # Cleaning and feature engineering
        ├── ml_models/              # Training and recommendation logic
        ├── models/                 # Local trained model artifacts
        ├── ocr_module/             # Diploma OCR pipeline
        ├── outputs/                # Metrics and generated recommendations
        ├── utils/                  # Interest-assessment utilities
        └── web_app/                # Flask routes, templates, and static files
```

## Team

- Ayham Thiab
- Yamama Sawalmeh
- Hadi Assayra
- Lama Damiri
- Ayham Baarah

Supervised by **Dr. Anas Toma**.
