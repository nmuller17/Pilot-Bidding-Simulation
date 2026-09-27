"""
Project directory layout, resolved from this file so scripts work from any cwd.
"""
import os

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
ENV_FILE = os.path.join(PROJECT_ROOT, ".env")
