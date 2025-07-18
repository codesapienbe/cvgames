#!/usr/bin/env python3

import sys
from pathlib import Path

def debug_game_loading():
    """Debug the game loading process"""
    print("🔍 Debugging game loading...")
    
    # Get the src path (same logic as app store)
    current_file = Path(__file__)
    src_path = current_file.parent / 'src'
    print(f"📁 Current file: {current_file}")
    print(f"📁 Src path: {src_path}")
    print(f"📁 Src path exists: {src_path.exists()}")
    
    if not src_path.exists():
        print("❌ Src path does not exist!")
        return
    
    # List all directories
    all_dirs = [d for d in src_path.iterdir() if d.is_dir()]
    print(f"📁 All directories: {[d.name for d in all_dirs]}")
    
    # Filter game directories
    game_dirs = [d for d in src_path.iterdir() if d.is_dir() and d.name not in ['appstore', 'cvstore', '__pycache__']]
    print(f"🎮 Game directories: {[d.name for d in game_dirs]}")
    
    # Check each game directory
    valid_games = []
    for game_dir in game_dirs:
        init_file = game_dir / '__init__.py'
        if init_file.exists():
            print(f"✅ {game_dir.name}: has __init__.py")
            valid_games.append(game_dir.name)
        else:
            print(f"❌ {game_dir.name}: missing __init__.py")
    
    print(f"\n🎯 Valid games found: {len(valid_games)}")
    print(f"🎯 Valid game names: {valid_games}")

if __name__ == "__main__":
    debug_game_loading() 