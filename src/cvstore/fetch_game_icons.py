#!/usr/bin/env python3
"""
CVGames Icon Fetcher

This script fetches random images for all games in the CVGames collection
and saves them as icon.png files. This is done once to avoid fetching
images on-demand during app store startup.

Usage:
    python fetch_game_icons.py
"""

import os
import sys
import requests
import time
from pathlib import Path
from typing import List, Dict

class GameIconFetcher:
    def __init__(self):
        self.src_dir = Path(__file__).parent.parent
        self.icon_size = 128
        self.api_url = "https://picsum.photos"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'CVGames-IconFetcher/1.0'
        })
        
        # Track results
        self.results = {
            'successful': [],
            'failed': [],
            'skipped': []
        }
    
    def get_all_game_directories(self) -> List[str]:
        """Get all game directories from src/"""
        game_dirs = []
        
        # Exclude non-game directories
        exclude_dirs = {
            'cvstore', 'appstore', '__pycache__', 'cv_games.egg-info'
        }
        
        for item in self.src_dir.iterdir():
            if item.is_dir() and item.name not in exclude_dirs:
                # Check if it has a main file
                main_files = ['__init__.py', 'main.py', 'game.py', 'app.py']
                for main_file in main_files:
                    if (item / main_file).exists():
                        game_dirs.append(item.name)
                        break
        
        return sorted(game_dirs)
    
    def fetch_random_image(self, game_name: str) -> bool:
        """Fetch a random image for a specific game"""
        try:
            # Construct URL for random image
            url = f"{self.api_url}/{self.icon_size}/{self.icon_size}"
            
            print(f"📸 Fetching icon for {game_name}...", end=" ")
            
            # Fetch the image
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                # Create icon file path
                game_dir = self.src_dir / game_name
                icon_path = game_dir / "icon.png"
                
                # Save the image
                with open(icon_path, 'wb') as f:
                    f.write(response.content)
                
                print("✅ SUCCESS")
                self.results['successful'].append(game_name)
                return True
            else:
                print(f"❌ HTTP {response.status_code}")
                self.results['failed'].append(game_name)
                return False
                
        except requests.exceptions.Timeout:
            print("❌ TIMEOUT")
            self.results['failed'].append(game_name)
            return False
        except requests.exceptions.RequestException as e:
            print(f"❌ REQUEST ERROR: {e}")
            self.results['failed'].append(game_name)
            return False
        except Exception as e:
            print(f"❌ ERROR: {e}")
            self.results['failed'].append(game_name)
            return False
    
    def check_existing_icon(self, game_name: str) -> bool:
        """Check if a game already has an icon.png file"""
        game_dir = self.src_dir / game_name
        icon_path = game_dir / "icon.png"
        return icon_path.exists()
    
    def fetch_all_icons(self, force_update: bool = False):
        """Fetch icons for all games"""
        print("🎮 CVGames Icon Fetcher")
        print("=" * 50)
        
        # Get all game directories
        game_dirs = self.get_all_game_directories()
        
        print(f"📁 Found {len(game_dirs)} games:")
        for game in game_dirs:
            status = "✅" if self.check_existing_icon(game) else "❌"
            print(f"   {status} {game}")
        
        print(f"\n🔧 Starting icon fetch...")
        if force_update:
            print("⚠️  Force update mode - will overwrite existing icons")
        else:
            print("💡 Skipping games that already have icons")
        print("-" * 50)
        
        # Process each game
        for i, game in enumerate(game_dirs, 1):
            print(f"[{i}/{len(game_dirs)}] ", end="")
            
            # Check if icon already exists
            if not force_update and self.check_existing_icon(game):
                print(f"Skipping {game} (icon already exists)")
                self.results['skipped'].append(game)
                continue
            
            # Fetch icon
            self.fetch_random_image(game)
            
            # Small delay to be respectful to the API
            time.sleep(0.5)
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print a summary of the results"""
        print("\n" + "=" * 50)
        print("📊 FETCH SUMMARY")
        print("=" * 50)
        print(f"✅ Successful: {len(self.results['successful'])}")
        print(f"❌ Failed: {len(self.results['failed'])}")
        print(f"⏭️  Skipped: {len(self.results['skipped'])}")
        
        total_processed = len(self.results['successful']) + len(self.results['failed'])
        if total_processed > 0:
            success_rate = len(self.results['successful']) / total_processed * 100
            print(f"📈 Success rate: {success_rate:.1f}%")
        
        if self.results['successful']:
            print(f"\n✅ Successfully fetched icons for:")
            for game in self.results['successful']:
                print(f"   - {game}")
        
        if self.results['failed']:
            print(f"\n❌ Failed to fetch icons for:")
            for game in self.results['failed']:
                print(f"   - {game}")
        
        if self.results['skipped']:
            print(f"\n⏭️  Skipped (already have icons):")
            for game in self.results['skipped']:
                print(f"   - {game}")
        
        print(f"\n💡 Next steps:")
        print(f"   1. Icons are saved as icon.png in each game directory")
        print(f"   2. App store will use these local icons instead of fetching")
        print(f"   3. You can manually replace any icons you don't like")
        print(f"   4. Run with --force to update all icons again")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Fetch random icons for all CVGames')
    parser.add_argument('--force', action='store_true', 
                       help='Force update all icons (overwrite existing)')
    parser.add_argument('--size', type=int, default=128,
                       help='Icon size in pixels (default: 128)')
    
    args = parser.parse_args()
    
    # Create fetcher and run
    fetcher = GameIconFetcher()
    fetcher.icon_size = args.size
    fetcher.fetch_all_icons(force_update=args.force)

if __name__ == "__main__":
    main() 