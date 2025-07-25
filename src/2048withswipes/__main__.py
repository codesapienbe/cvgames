#!/usr/bin/env python3
"""
Main entry point for 2048 with Swipes game
This allows the module to be run with: python -m src.2048withswipes
"""

import sys
import os

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import and run the game
from __init__ import main

if __name__ == "__main__":
    main() 