#!/usr/bin/env python3
"""
Back Button Integration Helper

This script helps integrate the standardized back button into existing CVGames.
It can automatically add the necessary imports and code to games.

Usage:
    python integrate_back_button.py [game_directory]
    
Example:
    python integrate_back_button.py ../tetris
"""

import os
import sys
import re
from pathlib import Path

def find_main_file(game_dir):
    """Find the main game file (usually __init__.py)"""
    game_path = Path(game_dir)
    
    # Look for __init__.py first
    init_file = game_path / "__init__.py"
    if init_file.exists():
        return init_file
    
    # Look for other common main files
    for file in game_path.glob("*.py"):
        if file.name in ["main.py", "game.py", "app.py"]:
            return file
    
    return None

def add_imports(file_path):
    """Add back button imports to the file"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check if imports already exist
    if "from back_button import BackButton" in content:
        print(f"✅ Back button imports already exist in {file_path}")
        return False
    
    # Find the import section
    lines = content.split('\n')
    
    # Find where to insert the imports
    insert_index = -1
    for i, line in enumerate(lines):
        if line.strip().startswith('import ') or line.strip().startswith('from '):
            insert_index = i
    
    # Add the imports
    new_imports = [
        "",
        "# Import the back button system",
        "sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'cvstore'))",
        "from back_button import BackButton"
    ]
    
    # Insert after the last import
    if insert_index >= 0:
        lines.insert(insert_index + 1, '\n'.join(new_imports))
    else:
        # If no imports found, add at the beginning
        lines = new_imports + [''] + lines
    
    # Write back to file
    with open(file_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"✅ Added back button imports to {file_path}")
    return True

def find_main_function(file_path):
    """Find the main function or game loop"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    lines = content.split('\n')
    
    # Look for common patterns
    patterns = [
        r'def main\(\):',
        r'while True:',
        r'while cap\.isOpened\(\):',
        r'cv2\.VideoCapture\(0\)',
        r'cap = cv2\.VideoCapture'
    ]
    
    for pattern in patterns:
        for i, line in enumerate(lines):
            if re.search(pattern, line):
                return i
    
    return -1

def add_back_button_initialization(file_path):
    """Add back button initialization"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check if already initialized
    if "BackButton(" in content:
        print(f"✅ Back button already initialized in {file_path}")
        return False
    
    lines = content.split('\n')
    
    # Find camera setup
    camera_patterns = [
        r'cap\.set\(3,',
        r'cap\.set\(4,',
        r'cv2\.VideoCapture',
        r'HandDetector'
    ]
    
    insert_index = -1
    for pattern in camera_patterns:
        for i, line in enumerate(lines):
            if re.search(pattern, line):
                insert_index = i
                break
        if insert_index >= 0:
            break
    
    if insert_index >= 0:
        # Add initialization after camera setup
        init_lines = [
            "",
            "    # Initialize back button",
            "    back_button = BackButton(screen_width, screen_height)"
        ]
        
        lines.insert(insert_index + 1, '\n'.join(init_lines))
        
        with open(file_path, 'w') as f:
            f.write('\n'.join(lines))
        
        print(f"✅ Added back button initialization to {file_path}")
        return True
    
    return False

def add_back_button_handling(file_path):
    """Add back button input handling"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check if already handled
    if "back_button.handle_input" in content:
        print(f"✅ Back button handling already exists in {file_path}")
        return False
    
    lines = content.split('\n')
    
    # Find the main game loop
    loop_patterns = [
        r'while True:',
        r'while cap\.isOpened\(\):',
        r'while ret:'
    ]
    
    loop_start = -1
    for pattern in loop_patterns:
        for i, line in enumerate(lines):
            if re.search(pattern, line):
                loop_start = i
                break
        if loop_start >= 0:
            break
    
    if loop_start >= 0:
        # Find where to insert the handling code
        insert_index = loop_start + 1
        
        # Look for hand detection code
        hand_patterns = [
            r'hands,',
            r'results\.multi_hand_landmarks',
            r'HandDetector',
            r'findHands'
        ]
        
        for pattern in hand_patterns:
            for i in range(loop_start, len(lines)):
                if re.search(pattern, lines[i]):
                    insert_index = i + 1
                    break
            if insert_index > loop_start + 1:
                break
        
        # Add handling code
        handling_code = [
            "",
            "        # Handle back button input",
            "        hand_position = None",
            "        hand_landmarks = None",
            "        if hands:",
            "            hand_position = hands[0][\"lmList\"][9][:2]  # Palm center",
            "            # Convert to MediaPipe landmarks format for back button",
            "            hand_landmarks = type('HandLandmarks', (), {",
            "                'landmark': [type('Landmark', (), {",
            "                    'x': lm[0] / screen_width,",
            "                    'y': lm[1] / screen_height",
            "                })() for lm in hands[0][\"lmList\"]]",
            "            })()",
            "",
            "        # Check if user wants to exit",
            "        if back_button.handle_input(key, hand_landmarks, hand_position):",
            "            print(\"User approved exit - returning to app store\")",
            "            cap.release()",
            "            cv2.destroyAllWindows()",
            "            return"
        ]
        
        lines.insert(insert_index, '\n'.join(handling_code))
        
        with open(file_path, 'w') as f:
            f.write('\n'.join(lines))
        
        print(f"✅ Added back button handling to {file_path}")
        return True
    
    return False

def add_back_button_drawing(file_path):
    """Add back button drawing"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check if already drawn
    if "back_button.draw" in content:
        print(f"✅ Back button drawing already exists in {file_path}")
        return False
    
    lines = content.split('\n')
    
    # Find cv2.imshow calls
    imshow_pattern = r'cv2\.imshow'
    
    for i, line in enumerate(lines):
        if re.search(imshow_pattern, line):
            # Add drawing before imshow
            drawing_code = [
                "",
            "        # Draw back button",
            "        back_button.draw(frame, hand_position)"
            ]
            
            lines.insert(i, '\n'.join(drawing_code))
            
            with open(file_path, 'w') as f:
                f.write('\n'.join(lines))
            
            print(f"✅ Added back button drawing to {file_path}")
            return True
    
    return False

def integrate_back_button(game_dir):
    """Integrate back button into a game"""
    print(f"🔧 Integrating back button into {game_dir}")
    
    # Find the main file
    main_file = find_main_file(game_dir)
    if not main_file:
        print(f"❌ Could not find main file in {game_dir}")
        return False
    
    print(f"📁 Found main file: {main_file}")
    
    # Add imports
    add_imports(main_file)
    
    # Add initialization
    add_back_button_initialization(main_file)
    
    # Add handling
    add_back_button_handling(main_file)
    
    # Add drawing
    add_back_button_drawing(main_file)
    
    print(f"✅ Successfully integrated back button into {game_dir}")
    print(f"💡 Test the integration by:")
    print(f"   1. Running the game")
    print(f"   2. Pressing 'B' to open approval dialog")
    print(f"   3. Using pinch gesture over the BACK button")
    
    return True

def main():
    if len(sys.argv) != 2:
        print("Usage: python integrate_back_button.py [game_directory]")
        print("Example: python integrate_back_button.py ../tetris")
        return
    
    game_dir = sys.argv[1]
    
    if not os.path.exists(game_dir):
        print(f"❌ Directory {game_dir} does not exist")
        return
    
    integrate_back_button(game_dir)

if __name__ == "__main__":
    main() 