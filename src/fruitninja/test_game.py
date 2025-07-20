#!/usr/bin/env python3
"""
Test script for Fruit Ninja CV Game
Tests the core game components without requiring camera input
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test if all required modules can be imported"""
    try:
        import cv2
        import mediapipe as mp
        import numpy as np
        print("✅ All required modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_fruit_class():
    """Test the Fruit class functionality"""
    try:
        from __init__ import Fruit
        import random
        
        # Test fruit creation
        fruit = Fruit(100, 100, "apple", (0, 0, 255))
        assert fruit.x == 100
        assert fruit.y == 100
        assert fruit.fruit_type == "apple"
        assert fruit.color == (0, 0, 255)
        assert not fruit.sliced
        
        # Test fruit update
        original_y = fruit.y
        fruit.update()
        assert fruit.y != original_y  # Should have moved due to gravity
        
        print("✅ Fruit class working correctly")
        return True
    except Exception as e:
        print(f"❌ Fruit class test failed: {e}")
        return False

def test_bomb_class():
    """Test the Bomb class functionality"""
    try:
        from __init__ import Bomb
        
        # Test bomb creation
        bomb = Bomb(200, 200)
        assert bomb.x == 200
        assert bomb.y == 200
        assert not bomb.exploded
        
        # Test bomb update
        original_y = bomb.y
        bomb.update()
        assert bomb.y != original_y  # Should have moved due to gravity
        
        print("✅ Bomb class working correctly")
        return True
    except Exception as e:
        print(f"❌ Bomb class test failed: {e}")
        return False

def test_game_initialization():
    """Test game initialization (without camera)"""
    try:
        from __init__ import FruitNinja
        
        # Test that game can be created
        game = FruitNinja()
        assert game.score == 0
        assert game.bombs_exploded == 0
        assert not game.game_over
        assert len(game.fruits) == 0
        assert len(game.bombs) == 0
        assert len(game.fruit_types) == 5  # 5 different fruits
        
        print("✅ Game initialization working correctly")
        return True
    except Exception as e:
        print(f"❌ Game initialization test failed: {e}")
        return False

def test_collision_detection():
    """Test collision detection logic"""
    try:
        from __init__ import FruitNinja, Fruit, Bomb
        import math
        
        game = FruitNinja()
        
        # Test fruit collision
        fruit = Fruit(100, 100, "apple", (0, 0, 255))
        game.fruits.append(fruit)
        
        # Hand position at fruit location (should trigger collision)
        game.check_swipe_collision((100, 100))
        assert fruit.sliced
        assert game.score == 10
        
        # Test bomb collision
        bomb = Bomb(200, 200)
        game.bombs.append(bomb)
        
        # Hand position at bomb location (should trigger explosion)
        game.check_swipe_collision((200, 200))
        assert bomb.exploded
        assert game.bombs_exploded == 1
        assert game.score == -10  # 10 from fruit - 20 from bomb
        
        print("✅ Collision detection working correctly")
        return True
    except Exception as e:
        print(f"❌ Collision detection test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Fruit Ninja CV Game Components")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_fruit_class,
        test_bomb_class,
        test_game_initialization,
        test_collision_detection
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Game is ready to play.")
        print("\n🚀 To start the game, run:")
        print("   python __init__.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    main() 