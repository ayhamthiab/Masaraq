"""
Configuration file for the University Recommendation System
Contains constants, settings, and reference data
"""

import os

class Config:
    """
    Configuration class for the University Recommendation System
    """
    
    # Flask configuration
    SECRET_KEY = 'university-recommendation-system-secret-key'
    
    # File upload settings
    UPLOAD_FOLDER = os.path.join('data', 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max file size
    ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}
    
    # OCR configuration
    OCR_CONFIG = {
        'languages': ['ara', 'eng'],
        'config': '--psm 6',
        'tesseract_cmd': ''  # Empty string means use system default
    }
    
    # Machine learning model configuration
    MODEL_CONFIG = {
        'random_state': 42,
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'test_size': 0.2
    }
    
    # Recommendation weights
    RECOMMENDATION_WEIGHTS = {
        'academic_fit': 0.4,
        'interest_alignment': 0.3,
        'market_viability': 0.2,
        'personal_factors': 0.1
    }
    
    # Interest assessment categories
    INTEREST_CATEGORIES = {
        'personality': 'Personality traits and preferences',
        'subjects': 'Academic subjects of interest',
        'career_goals': 'Career goals and aspirations',
        'activities': 'Activities and hobbies'
    }
    
    # Palestinian cities by region
    PALESTINIAN_CITIES = {
        'north': [
            'Jenin', 'Nablus', 'Tulkarem', 'Tubas', 'Qalqilya'
        ],
        'central': [
            'Ramallah', 'Jerusalem', 'Jericho', 'Salfit'
        ],
        'south': [
            'Bethlehem', 'Hebron'
        ],
        'gaza': [
            'Gaza City', 'Khan Yunis', 'Rafah', 'Deir al-Balah', 'Jabalia'
        ]
    }
    
    # Tawjihi streams and subjects
    TAWJIHI_STREAMS = {
        'scientific': {
            'name_ar': 'العلمي',
            'name_en': 'Scientific',
            'subjects': [
                'mathematics', 'physics', 'chemistry', 'biology', 
                'arabic', 'english'
            ],
            'core_subjects': [
                'mathematics', 'physics', 'chemistry'
            ]
        },
        'literary': {
            'name_ar': 'الأدبي',
            'name_en': 'Literary',
            'subjects': [
                'mathematics', 'history', 'geography', 'arabic', 
                'english'
            ],
            'core_subjects': [
                'arabic', 'history', 'geography'
            ]
        },
        'commercial': {
            'name_ar': 'التجاري',
            'name_en': 'Commercial',
            'subjects': [
                'mathematics', 'accounting', 'economics', 'arabic', 
                'english'
            ],
            'core_subjects': [
                'mathematics', 'accounting', 'economics'
            ]
        },
        'industrial': {
            'name_ar': 'الصناعي',
            'name_en': 'Industrial',
            'subjects': [
                'mathematics', 'physics', 'industrial_subjects', 
                'arabic', 'english'
            ],
            'core_subjects': [
                'mathematics', 'physics', 'industrial_subjects'
            ]
        },
        'agricultural': {
            'name_ar': 'الزراعي',
            'name_en': 'Agricultural',
            'subjects': [
                'mathematics', 'biology', 'agricultural_subjects', 
                'arabic', 'english'
            ],
            'core_subjects': [
                'mathematics', 'biology', 'agricultural_subjects'
            ]
        },
        'home_economics': {
            'name_ar': 'الاقتصاد المنزلي',
            'name_en': 'Home Economics',
            'subjects': [
                'mathematics', 'home_economics_subjects', 'arabic', 
                'english'
            ],
            'core_subjects': [
                'mathematics', 'home_economics_subjects'
            ]
        }
    }
    
    # University fields information
    UNIVERSITY_FIELDS = {
        'engineering': {
            'name_ar': 'الهندسة',
            'name_en': 'Engineering',
            'eligible_streams': ['scientific'],
            'min_scores': {
                'mathematics': 70,
                'physics': 70
            },
            'subfields': [
                'Civil Engineering', 'Mechanical Engineering', 'Electrical Engineering',
                'Computer Engineering', 'Chemical Engineering', 'Architectural Engineering'
            ],
            'subfields_ar': [
                'الهندسة المدنية', 'الهندسة الميكانيكية', 'الهندسة الكهربائية',
                'هندسة الحاسوب', 'الهندسة الكيميائية', 'الهندسة المعمارية'
            ],
            'employment_rate': 0.73
        },
        'medicine': {
            'name_ar': 'الطب',
            'name_en': 'Medicine',
            'eligible_streams': ['scientific'],
            'min_scores': {
                'mathematics': 80,
                'biology': 80,
                'chemistry': 80
            },
            'subfields': [
                'Medicine', 'Dentistry', 'Pharmacy', 'Nursing', 'Physiotherapy'
            ],
            'subfields_ar': [
                'الطب', 'طب الأسنان', 'الصيدلة', 'التمريض', 'العلاج الطبيعي'
            ],
            'employment_rate': 0.84
        },
        'business': {
            'name_ar': 'إدارة الأعمال',
            'name_en': 'Business Administration',
            'eligible_streams': ['scientific', 'literary', 'commercial'],
            'min_scores': {
                'mathematics': 60
            },
            'subfields': [
                'Business Administration', 'Finance', 'Marketing', 'Accounting',
                'Management', 'Economics'
            ],
            'subfields_ar': [
                'إدارة الأعمال', 'التمويل', 'التسويق', 'المحاسبة',
                'الإدارة', 'الاقتصاد'
            ],
            'employment_rate': 0.66
        },
        'law': {
            'name_ar': 'القانون',
            'name_en': 'Law',
            'eligible_streams': ['scientific', 'literary'],
            'min_scores': {
                'arabic': 70
            },
            'subfields': [
                'Law', 'International Law', 'Commercial Law', 'Public Law'
            ],
            'subfields_ar': [
                'القانون', 'القانون الدولي', 'القانون التجاري', 'القانون العام'
            ],
            'employment_rate': 0.58
        },
        'education': {
            'name_ar': 'التعليم',
            'name_en': 'Education',
            'eligible_streams': ['scientific', 'literary', 'commercial', 'industrial', 'agricultural', 'home_economics'],
            'min_scores': {},
            'subfields': [
                'Teaching', 'Special Education', 'Educational Administration',
                'Curriculum Development'
            ],
            'subfields_ar': [
                'التدريس', 'التربية الخاصة', 'الإدارة التربوية',
                'تطوير المناهج'
            ],
            'employment_rate': 0.71
        },
        'it': {
            'name_ar': 'تكنولوجيا المعلومات',
            'name_en': 'Information Technology',
            'eligible_streams': ['scientific'],
            'min_scores': {
                'mathematics': 65
            },
            'subfields': [
                'Computer Science', 'Software Engineering', 'Information Systems',
                'Cybersecurity', 'Data Science'
            ],
            'subfields_ar': [
                'علوم الحاسوب', 'هندسة البرمجيات', 'نظم المعلومات',
                'الأمن السيبراني', 'علم البيانات'
            ],
            'employment_rate': 0.81
        },
        'agriculture': {
            'name_ar': 'الزراعة',
            'name_en': 'Agriculture',
            'eligible_streams': ['scientific', 'agricultural'],
            'min_scores': {
                'biology': 60
            },
            'subfields': [
                'Agricultural Engineering', 'Plant Production', 'Animal Production',
                'Food Technology'
            ],
            'subfields_ar': [
                'الهندسة الزراعية', 'الإنتاج النباتي', 'الإنتاج الحيواني',
                'تكنولوجيا الأغذية'
            ],
            'employment_rate': 0.62
        },
        'arts': {
            'name_ar': 'الفنون',
            'name_en': 'Arts',
            'eligible_streams': ['scientific', 'literary', 'commercial'],
            'min_scores': {},
            'subfields': [
                'Fine Arts', 'Graphic Design', 'Media Studies', 'Performing Arts',
                'Music'
            ],
            'subfields_ar': [
                'الفنون الجميلة', 'التصميم الجرافيكي', 'دراسات الإعلام', 'فنون الأداء',
                'الموسيقى'
            ],
            'employment_rate': 0.52
        }
    }
    
    # Palestinian universities information
    UNIVERSITIES = {
        'An-Najah': {
            'name': 'An-Najah National University',
            'arabic_name': 'جامعة النجاح الوطنية',
            'location': 'Nablus',
            'type': 'Public',
            'url': 'https://www.najah.edu/'
        },
        'Birzeit': {
            'name': 'Birzeit University',
            'arabic_name': 'جامعة بيرزيت',
            'location': 'Birzeit, Ramallah',
            'type': 'Public',
            'url': 'https://www.birzeit.edu/'
        },
        'Al-Quds_Abu_Dis': {
            'name': 'Al-Quds University',
            'arabic_name': 'جامعة القدس',
            'location': 'Abu Dis, Jerusalem',
            'type': 'Public',
            'url': 'https://www.alquds.edu/'
        },
        'Hebron': {
            'name': 'Hebron University',
            'arabic_name': 'جامعة الخليل',
            'location': 'Hebron',
            'type': 'Public',
            'url': 'https://www.hebron.edu/'
        },
        'Arab_American_Jenin': {
            'name': 'Arab American University',
            'arabic_name': 'الجامعة العربية الأمريكية',
            'location': 'Jenin',
            'type': 'Private',
            'url': 'https://www.aaup.edu/'
        }
    }
