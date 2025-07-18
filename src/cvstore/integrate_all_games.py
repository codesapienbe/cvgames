#!/usr/bin/env python3
"""
Back Button Integration for All Games

This script automatically integrates the standardized back button into all
CVGames in the collection.

Usage:
    python integrate_all_games.py
"""

import os
import sys
import subprocess
from pathlib import Path

def get_all_game_directories():
    """Get all game directories from src/"""
    src_dir = Path(__file__).parent.parent
    game_dirs = []
    
    # Exclude non-game directories
    exclude_dirs = {
        'cvstore', 'appstore', '__pycache__', 'cv_games.egg-info'
    }
    
    for item in src_dir.iterdir():
        if item.is_dir() and item.name not in exclude_dirs:
            # Check if it has a main file
            main_files = ['__init__.py', 'main.py', 'game.py', 'app.py']
            for main_file in main_files:
                if (item / main_file).exists():
                    game_dirs.append(item.name)
                    break
    
    return sorted(game_dirs)

def integrate_game(game_name):
    """Integrate back button into a specific game"""
    try:
        result = subprocess.run([
            sys.executable, 'integrate_back_button.py', f'../{game_name}'
        ], capture_output=True, text=True, cwd=Path(__file__).parent)
        
        if result.returncode == 0:
            return True, result.stdout
        else:
            return False, result.stderr
    except Exception as e:
        return False, str(e)

def main():
    print("🎮 CVGames Back Button Integration")
    print("=" * 50)
    
    # Get all game directories
    game_dirs = get_all_game_directories()
    
    print(f"📁 Found {len(game_dirs)} games to integrate:")
    for game in game_dirs:
        print(f"   - {game}")
    
    print("\n🔧 Starting integration...")
    print("-" * 50)
    
    # Track results
    successful = []
    failed = []
    
    for i, game in enumerate(game_dirs, 1):
        print(f"[{i}/{len(game_dirs)}] Integrating {game}...", end=" ")
        
        success, output = integrate_game(game)
        
        if success:
            print("✅ SUCCESS")
            successful.append(game)
        else:
            print("❌ FAILED")
            failed.append(game)
            print(f"   Error: {output}")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 INTEGRATION SUMMARY")
    print("=" * 50)
    print(f"✅ Successful: {len(successful)} games")
    print(f"❌ Failed: {len(failed)} games")
    print(f"📈 Success rate: {len(successful)/(len(successful)+len(failed))*100:.1f}%")
    
    if successful:
        print(f"\n✅ Successfully integrated back button into:")
        for game in successful:
            print(f"   - {game}")
    
    if failed:
        print(f"\n❌ Failed to integrate back button into:")
        for game in failed:
            print(f"   - {game}")
    
    print(f"\n💡 Next steps:")
    print(f"   1. Test a few games to ensure integration works")
    print(f"   2. Press 'B' or use pinch gesture over BACK button")
    print(f"   3. Verify approval dialog appears and works correctly")
    
    if failed:
        print(f"\n⚠️  For failed games, you may need to:")
        print(f"   1. Check if the game has a different structure")
        print(f"   2. Manually integrate using the integration guide")
        print(f"   3. Update the integration script for edge cases")

if __name__ == "__main__":
    main() 