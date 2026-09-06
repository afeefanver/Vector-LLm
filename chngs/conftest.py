# vector-llm/conftest.py
# Adds the project root to sys.path so all test files can import
# config, auth, llm.* etc without needing sys.path hacks.
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
