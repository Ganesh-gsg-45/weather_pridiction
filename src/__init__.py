"""
src package initialization file.
Automatically ensures src/ is in sys.path so both `from src.weatherprediction...`
and `from weatherprediction...` work seamlessly.
"""
import sys
import os

src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

