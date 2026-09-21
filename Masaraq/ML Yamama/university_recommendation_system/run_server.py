#!/usr/bin/env python3
"""
University Recommendation System Server Starter
This script starts the web application with our fixed create_student_profile function
"""

import os
import sys
import argparse
from run_direct import main as run_direct_main

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run the University Recommendation System server')
    parser.add_argument('--port', '-p', type=int, default=8084, 
                        help='Port to run the server on (default: 8084)')
    args = parser.parse_args()
    
    # Run the direct runner with the specified port
    print(f"Starting server on port {args.port}")
    run_direct_main(port=args.port)
