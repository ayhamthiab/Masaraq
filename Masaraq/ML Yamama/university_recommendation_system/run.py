#!/usr/bin/env python3
"""
University Recommendation System - Main Entry Point
Run this file to start the web application.
"""

import os
import logging
from src.web_app.app import app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)

# Create necessary directories
os.makedirs('data/uploads', exist_ok=True)
os.makedirs('data/models', exist_ok=True)
os.makedirs('data/generated', exist_ok=True)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    
    # Print startup message
    print("""
    ===============================================
      University Recommendation System Starting
    ===============================================
    
    Open your browser and navigate to:
    http://localhost:{}
    
    Press Ctrl+C to stop the server
    ===============================================
    """.format(port))
    
    # Run the Flask application
    app.run(debug=True, host='0.0.0.0', port=port)
