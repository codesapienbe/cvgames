import cv2
import mediapipe as mp
import numpy as np
import random
import math
import time
import sys
import os
import argparse
import json
from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import threading
from collections import deque

# Import the back button system
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'cvstore'))
    from back_button import BackButton
except ImportError:
    import sys
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    sys.path.insert(0, project_root)
    sys.path.append(os.path.join(project_root, 'src', 'cvstore'))
    from back_button import BackButton

class GameState(Enum):
    MAIN_MENU = 1
    PLAYING = 2
    PAUSED = 3
    GAME_OVER = 4
    LEVEL_COMPLETE = 5
    SETTINGS = 6
    ACHIEVEMENTS = 7

class WeaponType(Enum):
    FIST = 0
    PISTOL = 1
    SHOTGUN = 2
    CHAINGUN = 3
    ROCKET_LAUNCHER = 4
    PLASMA_RIFLE = 5
    BFG = 6

class EnemyType(Enum):
    ZOMBIEMAN = 1
    SHOTGUN_GUY = 2
    IMP = 3
    DEMON = 4
    CACODEMON = 5
    BARON_OF_HELL = 6
    CYBERDEMON = 7

class PickupType(Enum):
    HEALTH_SMALL = 1
    HEALTH_LARGE = 2
    ARMOR = 3
    AMMO_CLIP = 4
    AMMO_SHELLS = 5
    AMMO_ROCKETS = 6
    KEYCARD_RED = 7
    KEYCARD_BLUE = 8
    KEYCARD_YELLOW = 9

class DifficultyLevel(Enum):
    EASY = 1
    MEDIUM = 2
    HARD = 3
    NIGHTMARE = 4

@dataclass
class GameConfig:
    """Game configuration settings"""
    difficulty: DifficultyLevel = DifficultyLevel.MEDIUM
    sound_enabled: bool = True
    show_fps: bool = True
    gesture_sensitivity: float = 0.8
    auto_aim: bool = False
    crosshair_color: Tuple[int, int, int] = (255, 0, 0)

class DoomColors:
    """Extended Doom color palette with authentic colors"""
    # Base colors
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    
    # Doom signature colors
    DARK_RED = (139, 0, 0)
    BRIGHT_RED = (255, 0, 0)
    BLOOD_RED = (180, 0, 0)
    
    # Grays and metals
    DARK_GRAY = (64, 64, 64)
    MEDIUM_GRAY = (128, 128, 128)
    LIGHT_GRAY = (192, 192, 192)
    METAL_GRAY = (169, 169, 169)
    
    # Environment
    WALL_BROWN = (139, 69, 19)
    FLOOR_GRAY = (105, 105, 105)
    CEILING_DARK = (47, 79, 79)
    
    # UI Colors
    HUD_GREEN = (0, 255, 0)
    HUD_YELLOW = (255, 255, 0)
    HUD_ORANGE = (255, 165, 0)
    HUD_RED = (255, 69, 0)
    
    # Effects
    MUZZLE_FLASH = (255, 255, 200)
    EXPLOSION_ORANGE = (255, 140, 0)
    PLASMA_BLUE = (0, 191, 255)
    
    # Enemy colors
    ZOMBIE_GREEN = (34, 139, 34)
    IMP_BROWN = (160, 82, 45)
    DEMON_PINK = (255, 20, 147)
    CACODEMON_RED = (220, 20, 60)

class Vector2D:
    """2D Vector class for game physics"""
    def __init__(self, x: float = 0.0, y: float = 0.0):
        self.x = x
        self.y = y
    
    def __add__(self, other):
        return Vector2D(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        return Vector2D(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar):
        return Vector2D(self.x * scalar, self.y * scalar)
    
    def magnitude(self):
        return math.sqrt(self.x * self.x + self.y * self.y)
    
    def normalize(self):
        mag = self.magnitude()
        if mag > 0:
            return Vector2D(self.x / mag, self.y / mag)
        return Vector2D(0, 0)
    
    def distance_to(self, other):
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

class Weapon:
    """Advanced weapon system with authentic Doom mechanics"""
    def __init__(self, weapon_type: WeaponType):
        self.type = weapon_type
        self.name = ""
        self.max_ammo = 0
        self.current_ammo = 0
        self.damage = 0
        self.fire_rate = 0.0
        self.reload_time = 0.0
        self.spread = 0.0
        self.projectile_speed = 0.0
        self.sound_fire = ""
        self.sound_reload = ""
        self.muzzle_flash_size = 20
        self.recoil_strength = 5
        self.auto_fire = False
        
        self.last_shot_time = 0
        self.reloading = False
        self.reload_start_time = 0
        self.selected = False
        
        self._setup_weapon_stats()
    
    def _setup_weapon_stats(self):
        """Setup weapon-specific statistics"""
        weapon_stats = {
            WeaponType.FIST: {
                "name": "FISTS", "max_ammo": -1, "current_ammo": -1, "damage": 20,
                "fire_rate": 0.5, "reload_time": 0, "spread": 0, "projectile_speed": 0,
                "muzzle_flash_size": 0, "recoil_strength": 0, "auto_fire": True
            },
            WeaponType.PISTOL: {
                "name": "PISTOL", "max_ammo": 200, "current_ammo": 50, "damage": 15,
                "fire_rate": 0.4, "reload_time": 1.2, "spread": 0.02, "projectile_speed": 1500,
                "muzzle_flash_size": 15, "recoil_strength": 3, "auto_fire": False
            },
            WeaponType.SHOTGUN: {
                "name": "SHOTGUN", "max_ammo": 50, "current_ammo": 8, "damage": 70,
                "fire_rate": 1.2, "reload_time": 2.0, "spread": 0.12, "projectile_speed": 1200,
                "muzzle_flash_size": 30, "recoil_strength": 12, "auto_fire": False
            },
            WeaponType.CHAINGUN: {
                "name": "CHAINGUN", "max_ammo": 300, "current_ammo": 100, "damage": 12,
                "fire_rate": 0.08, "reload_time": 3.0, "spread": 0.04, "projectile_speed": 1600,
                "muzzle_flash_size": 20, "recoil_strength": 2, "auto_fire": True
            },
            WeaponType.ROCKET_LAUNCHER: {
                "name": "ROCKET LAUNCHER", "max_ammo": 20, "current_ammo": 5, "damage": 150,
                "fire_rate": 1.8, "reload_time": 2.5, "spread": 0, "projectile_speed": 800,
                "muzzle_flash_size": 40, "recoil_strength": 15, "auto_fire": False
            },
            WeaponType.PLASMA_RIFLE: {
                "name": "PLASMA RIFLE", "max_ammo": 100, "current_ammo": 25, "damage": 25,
                "fire_rate": 0.15, "reload_time": 2.2, "spread": 0.01, "projectile_speed": 1400,
                "muzzle_flash_size": 25, "recoil_strength": 4, "auto_fire": True
            },
            WeaponType.BFG: {
                "name": "BFG 9000", "max_ammo": 10, "current_ammo": 2, "damage": 500,
                "fire_rate": 3.0, "reload_time": 4.0, "spread": 0, "projectile_speed": 1000,
                "muzzle_flash_size": 60, "recoil_strength": 20, "auto_fire": False
            }
        }
        
        stats = weapon_stats[self.type]
        for key, value in stats.items():
            setattr(self, key, value)
    
    def can_shoot(self) -> bool:
        """Check if weapon can shoot"""
        current_time = time.time()
        return (self.current_ammo != 0 and 
                not self.reloading and
                current_time - self.last_shot_time >= self.fire_rate)
    
    def shoot(self) -> bool:
        """Fire the weapon"""
        if not self.can_shoot():
            return False
        
        if self.current_ammo > 0:
            self.current_ammo -= 1
        self.last_shot_time = time.time()
        return True
    
    def start_reload(self):
        """Start reloading process"""
        if self.current_ammo < self.max_ammo and not self.reloading and self.max_ammo > 0:
            self.reloading = True
            self.reload_start_time = time.time()
    
    def update(self):
        """Update weapon state"""
        if self.reloading and self.max_ammo > 0:
            if time.time() - self.reload_start_time >= self.reload_time:
                self.current_ammo = self.max_ammo
                self.reloading = False
    
    def get_reload_progress(self) -> float:
        """Get reload progress as percentage"""
        if not self.reloading:
            return 1.0
        return min(1.0, (time.time() - self.reload_start_time) / self.reload_time)
    
    def draw_weapon(self, frame, screen_width, screen_height, recoil_offset=0):
        """Draw weapon sprite on screen"""
        weapon_bottom_y = screen_height - 50 + int(recoil_offset)
        weapon_center_x = screen_width - 200
        
        if self.type == WeaponType.FIST:
            self._draw_fist(frame, weapon_center_x, weapon_bottom_y)
        elif self.type == WeaponType.PISTOL:
            self._draw_pistol(frame, weapon_center_x, weapon_bottom_y)
        elif self.type == WeaponType.SHOTGUN:
            self._draw_shotgun(frame, weapon_center_x, weapon_bottom_y)
        elif self.type == WeaponType.CHAINGUN:
            self._draw_chaingun(frame, weapon_center_x, weapon_bottom_y)
        elif self.type == WeaponType.ROCKET_LAUNCHER:
            self._draw_rocket_launcher(frame, weapon_center_x, weapon_bottom_y)
        elif self.type == WeaponType.PLASMA_RIFLE:
            self._draw_plasma_rifle(frame, weapon_center_x, weapon_bottom_y)
        elif self.type == WeaponType.BFG:
            self._draw_bfg(frame, weapon_center_x, weapon_bottom_y)
    
    def _draw_fist(self, frame, x, y):
        """Draw fist sprite"""
        # Fist shape
        fist_points = np.array([
            [x - 30, y - 40], [x + 30, y - 40],
            [x + 35, y - 20], [x + 30, y],
            [x - 30, y], [x - 35, y - 20]
        ], np.int32)
        cv2.fillPoly(frame, [fist_points], DoomColors.WALL_BROWN)
        
        # Knuckles
        for i in range(4):
            knuckle_x = x - 20 + (i * 10)
            cv2.circle(frame, (knuckle_x, y - 25), 4, DoomColors.LIGHT_GRAY, -1)
    
    def _draw_pistol(self, frame, x, y):
        """Draw pistol sprite"""
        # Barrel
        barrel_points = np.array([
            [x - 12, y - 100], [x + 12, y - 100],
            [x + 15, y - 50], [x - 15, y - 50]
        ], np.int32)
        cv2.fillPoly(frame, [barrel_points], DoomColors.DARK_GRAY)
        
        # Grip
        grip_points = np.array([
            [x - 20, y - 50], [x + 20, y - 50],
            [x + 25, y], [x - 25, y]
        ], np.int32)
        cv2.fillPoly(frame, [grip_points], DoomColors.WALL_BROWN)
        
        # Trigger guard
        cv2.ellipse(frame, (x, y - 30), (10, 18), 0, 0, 360, DoomColors.METAL_GRAY, 3)
        
        # Sight
        cv2.rectangle(frame, (x - 2, y - 105), (x + 2, y - 95), DoomColors.WHITE, -1)
    
    def _draw_shotgun(self, frame, x, y):
        """Draw shotgun sprite"""
        # Double barrel
        cv2.rectangle(frame, (x - 10, y - 120), (x - 3, y - 40), DoomColors.DARK_GRAY, -1)
        cv2.rectangle(frame, (x + 3, y - 120), (x + 10, y - 40), DoomColors.DARK_GRAY, -1)
        
        # Pump action
        cv2.rectangle(frame, (x - 15, y - 80), (x + 15, y - 60), DoomColors.WALL_BROWN, -1)
        cv2.rectangle(frame, (x - 12, y - 77), (x + 12, y - 63), DoomColors.METAL_GRAY, 2)
        
        # Stock
        stock_points = np.array([
            [x - 20, y - 40], [x + 20, y - 40],
            [x + 30, y], [x - 30, y]
        ], np.int32)
        cv2.fillPoly(frame, [stock_points], DoomColors.WALL_BROWN)
    
    def _draw_chaingun(self, frame, x, y):
        """Draw chaingun sprite"""
        # Rotating barrels (6 barrels in circle)
        rotation = time.time() * 5 if self.last_shot_time > time.time() - 0.5 else 0
        for i in range(6):
            angle = (i * math.pi / 3) + rotation
            barrel_x = x + int(20 * math.cos(angle))
            barrel_y = y - 80 + int(20 * math.sin(angle))
            cv2.circle(frame, (barrel_x, barrel_y), 6, DoomColors.DARK_GRAY, -1)
            cv2.circle(frame, (barrel_x, barrel_y), 4, DoomColors.BLACK, -1)
        
        # Main body
        cv2.rectangle(frame, (x - 35, y - 100), (x + 35, y - 30), DoomColors.METAL_GRAY, -1)
        
        # Ammo belt
        belt_points = np.array([
            [x - 40, y - 90], [x - 50, y - 70],
            [x - 45, y - 50], [x - 35, y - 60]
        ], np.int32)
        cv2.fillPoly(frame, [belt_points], DoomColors.HUD_YELLOW)
        
        # Handle
        cv2.rectangle(frame, (x - 25, y - 30), (x + 25, y), DoomColors.WALL_BROWN, -1)
    
    def _draw_rocket_launcher(self, frame, x, y):
        """Draw rocket launcher sprite"""
        # Main tube
        cv2.rectangle(frame, (x - 25, y - 120), (x + 25, y - 40), DoomColors.DARK_GRAY, -1)
        
        # Rocket visible in tube
        cv2.rectangle(frame, (x - 8, y - 115), (x + 8, y - 90), DoomColors.HUD_RED, -1)
        cv2.circle(frame, (x, y - 115), 8, DoomColors.HUD_ORANGE, -1)
        
        # Handle
        cv2.rectangle(frame, (x - 15, y - 60), (x + 15, y - 20), DoomColors.WALL_BROWN, -1)
        
        # Trigger
        cv2.rectangle(frame, (x - 5, y - 50), (x + 5, y - 30), DoomColors.BLACK, -1)
        
        # Scope
        cv2.rectangle(frame, (x - 30, y - 100), (x - 25, y - 80), DoomColors.BLACK, -1)
        cv2.circle(frame, (x - 27, y - 90), 3, DoomColors.WHITE, 1)
    
    def _draw_plasma_rifle(self, frame, x, y):
        """Draw plasma rifle sprite"""
        # Main body (futuristic design)
        body_points = np.array([
            [x - 30, y - 90], [x + 30, y - 90],
            [x + 35, y - 70], [x + 30, y - 50],
            [x - 30, y - 50], [x - 35, y - 70]
        ], np.int32)
        cv2.fillPoly(frame, [body_points], DoomColors.PLASMA_BLUE)
        
        # Energy core (glowing effect)
        core_intensity = 0.5 + 0.5 * math.sin(time.time() * 3)
        core_color = tuple(int(c * core_intensity) for c in DoomColors.PLASMA_BLUE)
        cv2.circle(frame, (x, y - 70), 12, core_color, -1)
        cv2.circle(frame, (x, y - 70), 8, DoomColors.WHITE, -1)
        
        # Barrel with energy glow
        cv2.rectangle(frame, (x - 8, y - 110), (x + 8, y - 90), DoomColors.METAL_GRAY, -1)
        glow_color = tuple(int(c * core_intensity) for c in DoomColors.PLASMA_BLUE)
        cv2.rectangle(frame, (x - 6, y - 108), (x + 6, y - 92), glow_color, -1)
        
        # Handle
        cv2.rectangle(frame, (x - 20, y - 50), (x + 20, y - 20), DoomColors.DARK_GRAY, -1)
    
    def _draw_bfg(self, frame, x, y):
        """Draw BFG 9000 sprite"""
        # Massive main body
        cv2.rectangle(frame, (x - 40, y - 130), (x + 40, y - 30), DoomColors.HUD_GREEN, -1)
        
        # Energy chamber (large and glowing)
        chamber_intensity = 0.3 + 0.7 * math.sin(time.time() * 2)
        chamber_color = tuple(int(c * chamber_intensity) for c in DoomColors.HUD_GREEN)
        cv2.circle(frame, (x, y - 80), 25, chamber_color, -1)
        cv2.circle(frame, (x, y - 80), 20, DoomColors.WHITE, -1)
        cv2.circle(frame, (x, y - 80), 15, chamber_color, -1)
        
        # Massive barrel
        cv2.rectangle(frame, (x - 15, y - 140), (x + 15, y - 130), DoomColors.DARK_GRAY, -1)
        
        # Warning labels
        cv2.putText(frame, "BFG", (x - 15, y - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, DoomColors.BLACK, 1)
        cv2.putText(frame, "9000", (x - 18, y - 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, DoomColors.BLACK, 1)
        
        # Handle (reinforced)
        cv2.rectangle(frame, (x - 25, y - 30), (x + 25, y), DoomColors.METAL_GRAY, -1)

class Projectile:
    """Projectile system for bullets, rockets, plasma, etc."""
    def __init__(self, start_pos: Vector2D, direction: Vector2D, weapon_type: WeaponType):
        self.position = start_pos
        self.direction = direction.normalize()
        self.weapon_type = weapon_type
        self.speed = 0
        self.damage = 0
        self.lifetime = 0
        self.max_lifetime = 5.0
        self.explosive = False
        self.explosion_radius = 0
        self.size = 3
        self.color = DoomColors.MUZZLE_FLASH
        self.trail = deque(maxlen=10)
        
        self._setup_projectile_stats()
        
        self.creation_time = time.time()
        self.alive = True
    
    def _setup_projectile_stats(self):
        """Setup projectile-specific statistics"""
        if self.weapon_type == WeaponType.PISTOL:
            self.speed = 1500
            self.damage = 15
            self.color = DoomColors.MUZZLE_FLASH
            self.size = 2
        elif self.weapon_type == WeaponType.SHOTGUN:
            self.speed = 1200
            self.damage = 70
            self.color = DoomColors.MUZZLE_FLASH
            self.size = 3
        elif self.weapon_type == WeaponType.CHAINGUN:
            self.speed = 1600
            self.damage = 12
            self.color = DoomColors.MUZZLE_FLASH
            self.size = 2
        elif self.weapon_type == WeaponType.ROCKET_LAUNCHER:
            self.speed = 800
            self.damage = 150
            self.explosive = True
            self.explosion_radius = 100
            self.color = DoomColors.HUD_RED
            self.size = 8
        elif self.weapon_type == WeaponType.PLASMA_RIFLE:
            self.speed = 1400
            self.damage = 25
            self.color = DoomColors.PLASMA_BLUE
            self.size = 5
        elif self.weapon_type == WeaponType.BFG:
            self.speed = 1000
            self.damage = 500
            self.explosive = True
            self.explosion_radius = 200
            self.color = DoomColors.HUD_GREEN
            self.size = 12
    
    def update(self, dt: float):
        """Update projectile position and lifetime"""
        if not self.alive:
            return
        
        # Add current position to trail
        self.trail.append((self.position.x, self.position.y))
        
        # Update position
        velocity = self.direction * (self.speed * dt)
        self.position = self.position + velocity
        
        # Update lifetime
        self.lifetime += dt
        if self.lifetime >= self.max_lifetime:
            self.alive = False
    
    def draw(self, frame):
        """Draw projectile with trail effects"""
        if not self.alive:
            return
        
        # Draw trail
        trail_alpha = 1.0
        for i, trail_pos in enumerate(self.trail):
            alpha = trail_alpha * (i / len(self.trail)) * 0.5
            if alpha > 0:
                trail_color = tuple(int(c * alpha) for c in self.color)
                cv2.circle(frame, (int(trail_pos[0]), int(trail_pos[1])), 
                          max(1, self.size // 2), trail_color, -1)
        
        # Draw main projectile
        pos_x, pos_y = int(self.position.x), int(self.position.y)
        
        if self.weapon_type == WeaponType.ROCKET_LAUNCHER:
            # Draw rocket shape
            rocket_length = 15
            end_x = pos_x + int(self.direction.x * rocket_length)
            end_y = pos_y + int(self.direction.y * rocket_length)
            cv2.line(frame, (pos_x, pos_y), (end_x, end_y), DoomColors.HUD_RED, self.size)
            cv2.circle(frame, (end_x, end_y), self.size, DoomColors.HUD_ORANGE, -1)
        elif self.weapon_type == WeaponType.PLASMA_RIFLE:
            # Draw plasma ball with energy effect
            intensity = 0.5 + 0.5 * math.sin(time.time() * 10)
            glow_color = tuple(int(c * intensity) for c in self.color)
            cv2.circle(frame, (pos_x, pos_y), self.size + 2, glow_color, -1)
            cv2.circle(frame, (pos_x, pos_y), self.size, DoomColors.WHITE, -1)
        elif self.weapon_type == WeaponType.BFG:
            # Draw BFG projectile with massive energy effect
            intensity = 0.3 + 0.7 * math.sin(time.time() * 5)
            for radius in range(self.size, self.size + 10, 2):
                alpha = 1.0 - ((radius - self.size) / 10.0)
                glow_color = tuple(int(c * intensity * alpha) for c in self.color)
                cv2.circle(frame, (pos_x, pos_y), radius, glow_color, 2)
        else:
            # Draw standard bullet
            cv2.circle(frame, (pos_x, pos_y), self.size, self.color, -1)

class Enemy:
    """Advanced enemy AI system with multiple types"""
    def __init__(self, x: float, y: float, enemy_type: EnemyType, screen_width: int, screen_height: int):
        self.position = Vector2D(x, y)
        self.enemy_type = enemy_type
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Base stats
        self.max_health = 100
        self.health = 100
        self.speed = 1.5
        self.damage = 20
        self.width = 60
        self.height = 80
        self.detection_range = 400
        self.attack_range = 150
        self.attack_cooldown = 2.0
        self.score_value = 50
        
        # AI state
        self.target_position = Vector2D(x, y)
        self.last_attack_time = 0
        self.last_direction_change = time.time()
        self.state = "patrol"  # patrol, chase, attack, dead
        self.direction = random.uniform(0, 2 * math.pi)
        
        # Visual
        self.animation_frame = 0
        self.animation_speed = 0.1
        self.hit_time = 0
        self.death_time = 0
        self.color = DoomColors.DARK_RED
        
        # Status
        self.alive = True
        self.pain_time = 0
        
        self._setup_enemy_stats()
    
    def _setup_enemy_stats(self):
        """Setup enemy-specific statistics"""
        enemy_stats = {
            EnemyType.ZOMBIEMAN: {
                "max_health": 20, "speed": 1.0, "damage": 10, "width": 50, "height": 70,
                "detection_range": 300, "attack_range": 200, "attack_cooldown": 1.5,
                "score_value": 50, "color": DoomColors.ZOMBIE_GREEN
            },
            EnemyType.SHOTGUN_GUY: {
                "max_health": 30, "speed": 1.2, "damage": 25, "width": 55, "height": 75,
                "detection_range": 350, "attack_range": 180, "attack_cooldown": 2.0,
                "score_value": 75, "color": DoomColors.ZOMBIE_GREEN
            },
            EnemyType.IMP: {
                "max_health": 60, "speed": 2.0, "damage": 20, "width": 60, "height": 80,
                "detection_range": 400, "attack_range": 150, "attack_cooldown": 1.8,
                "score_value": 100, "color": DoomColors.IMP_BROWN
            },
            EnemyType.DEMON: {
                "max_health": 150, "speed": 3.0, "damage": 40, "width": 80, "height": 60,
                "detection_range": 500, "attack_range": 80, "attack_cooldown": 1.0,
                "score_value": 150, "color": DoomColors.DEMON_PINK
            },
            EnemyType.CACODEMON: {
                "max_health": 400, "speed": 1.5, "damage": 30, "width": 100, "height": 100,
                "detection_range": 600, "attack_range": 200, "attack_cooldown": 2.5,
                "score_value": 300, "color": DoomColors.CACODEMON_RED
            },
            EnemyType.BARON_OF_HELL: {
                "max_health": 1000, "speed": 2.5, "damage": 80, "width": 120, "height": 140,
                "detection_range": 700, "attack_range": 180, "attack_cooldown": 2.0,
                "score_value": 500, "color": DoomColors.DARK_RED
            },
            EnemyType.CYBERDEMON: {
                "max_health": 4000, "speed": 2.0, "damage": 200, "width": 150, "height": 180,
                "detection_range": 800, "attack_range": 300, "attack_cooldown": 3.0,
                "score_value": 1000, "color": DoomColors.METAL_GRAY
            }
        }
        
        if self.enemy_type in enemy_stats:
            stats = enemy_stats[self.enemy_type]
            for key, value in stats.items():
                setattr(self, key, value)
            self.health = self.max_health
    
    def update(self, player_pos: Vector2D, dt: float):
        """Update enemy AI and behavior"""
        if not self.alive:
            return
        
        current_time = time.time()
        distance_to_player = self.position.distance_to(player_pos)
        
        # State machine AI
        if self.state == "patrol":
            if distance_to_player < self.detection_range:
                self.state = "chase"
                self.target_position = player_pos
            else:
                # Random patrol movement
                if current_time - self.last_direction_change > 3.0:
                    self.direction = random.uniform(0, 2 * math.pi)
                    self.last_direction_change = current_time
                
                # Move in current direction
                velocity = Vector2D(math.cos(self.direction), math.sin(self.direction)) * (self.speed * dt)
                self.position = self.position + velocity
        
        elif self.state == "chase":
            if distance_to_player > self.detection_range * 1.5:
                self.state = "patrol"
            elif distance_to_player < self.attack_range:
                self.state = "attack"
            else:
                # Move toward player
                direction_to_player = (player_pos - self.position).normalize()
                velocity = direction_to_player * (self.speed * dt)
                self.position = self.position + velocity
        
        elif self.state == "attack":
            if distance_to_player > self.attack_range * 1.2:
                self.state = "chase"
            elif current_time - self.last_attack_time > self.attack_cooldown:
                self.last_attack_time = current_time
                return self.damage  # Return damage dealt to player
        
        # Keep enemy on screen
        self.position.x = max(self.width // 2, min(self.screen_width - self.width // 2, self.position.x))
        self.position.y = max(self.height // 2, min(self.screen_height - self.height // 2, self.position.y))
        
        # Update animation
        self.animation_frame += self.animation_speed
        
        return 0  # No damage dealt
    
    def take_damage(self, damage: int) -> bool:
        """Apply damage to enemy"""
        if not self.alive:
            return False
        
        self.health -= damage
        self.hit_time = time.time()
        self.pain_time = time.time()
        
        # Enter chase state when hit
        if self.state == "patrol":
            self.state = "chase"
        
        if self.health <= 0:
            self.alive = False
            self.death_time = time.time()
            return True  # Enemy killed
        
        return False
    
    def check_collision_with_projectile(self, projectile: Projectile) -> bool:
        """Check if projectile hits this enemy"""
        if not self.alive or not projectile.alive:
            return False
        
        distance = self.position.distance_to(projectile.position)
        return distance < (max(self.width, self.height) // 2 + projectile.size)
    
    def draw(self, frame):
        """Draw enemy with animations and effects"""
        if not self.alive and time.time() - self.death_time > 2.0:
            return
        
        current_time = time.time()
        x, y = int(self.position.x), int(self.position.y)
        
        # Death animation
        if not self.alive:
            death_progress = min(1.0, (current_time - self.death_time) / 2.0)
            self._draw_death_animation(frame, x, y, death_progress)
            return
        
        # Hit flash effect
        hit_flash = current_time - self.hit_time < 0.2
        pain_effect = current_time - self.pain_time < 0.5
        
        base_color = DoomColors.WHITE if hit_flash else self.color
        if pain_effect:
            base_color = tuple(min(255, c + 50) for c in base_color)
        
        # Draw enemy based on type
        if self.enemy_type == EnemyType.ZOMBIEMAN:
            self._draw_zombieman(frame, x, y, base_color)
        elif self.enemy_type == EnemyType.SHOTGUN_GUY:
            self._draw_shotgun_guy(frame, x, y, base_color)
        elif self.enemy_type == EnemyType.IMP:
            self._draw_imp(frame, x, y, base_color)
        elif self.enemy_type == EnemyType.DEMON:
            self._draw_demon(frame, x, y, base_color)
        elif self.enemy_type == EnemyType.CACODEMON:
            self._draw_cacodemon(frame, x, y, base_color)
        elif self.enemy_type == EnemyType.BARON_OF_HELL:
            self._draw_baron(frame, x, y, base_color)
        elif self.enemy_type == EnemyType.CYBERDEMON:
            self._draw_cyberdemon(frame, x, y, base_color)
        
        # Health bar
        self._draw_health_bar(frame, x, y)
    
    def _draw_zombieman(self, frame, x, y, color):
        """Draw zombieman enemy"""
        # Body
        cv2.rectangle(frame, (x - self.width//4, y - self.height//2), 
                     (x + self.width//4, y + self.height//2), color, -1)
        
        # Head
        cv2.circle(frame, (x, y - self.height//2 - 10), self.width//4, color, -1)
        
        # Eyes (red glow)
        cv2.circle(frame, (x - 5, y - self.height//2 - 10), 2, DoomColors.BRIGHT_RED, -1)
        cv2.circle(frame, (x + 5, y - self.height//2 - 10), 2, DoomColors.BRIGHT_RED, -1)
        
        # Weapon (rifle)
        rifle_length = 30
        cv2.line(frame, (x + self.width//4, y), 
                (x + self.width//4 + rifle_length, y - 5), DoomColors.DARK_GRAY, 3)
    
    def _draw_shotgun_guy(self, frame, x, y, color):
        """Draw shotgun guy enemy"""
        # Similar to zombieman but slightly larger and different weapon
        self._draw_zombieman(frame, x, y, color)
        
        # Shotgun instead of rifle
        cv2.rectangle(frame, (x + self.width//4, y - 3), 
                     (x + self.width//4 + 25, y + 3), DoomColors.WALL_BROWN, -1)
    
    def _draw_imp(self, frame, x, y, color):
        """Draw imp enemy"""
        # More demon-like body
        body_points = np.array([
            [x, y - self.height//2],
            [x - self.width//3, y],
            [x - self.width//4, y + self.height//2],
            [x + self.width//4, y + self.height//2],
            [x + self.width//3, y]
        ], np.int32)
        cv2.fillPoly(frame, [body_points], color)
        
        # Demon head with horns
        cv2.circle(frame, (x, y - self.height//2 - 5), self.width//3, color, -1)
        
        # Horns
        horn_left = np.array([[x - 10, y - self.height//2 - 15], 
                             [x - 15, y - self.height//2 - 25], 
                             [x - 5, y - self.height//2 - 20]], np.int32)
        horn_right = np.array([[x + 10, y - self.height//2 - 15], 
                              [x + 15, y - self.height//2 - 25], 
                              [x + 5, y - self.height//2 - 20]], np.int32)
        cv2.fillPoly(frame, [horn_left], DoomColors.DARK_GRAY)
        cv2.fillPoly(frame, [horn_right], DoomColors.DARK_GRAY)
        
        # Glowing eyes
        cv2.circle(frame, (x - 8, y - self.height//2 - 5), 3, DoomColors.BRIGHT_RED, -1)
        cv2.circle(frame, (x + 8, y - self.height//2 - 5), 3, DoomColors.BRIGHT_RED, -1)
        
        # Claws
        claw_offset = int(10 * math.sin(self.animation_frame))
        cv2.line(frame, (x - self.width//3, y), 
                (x - self.width//3 - 15, y + claw_offset), DoomColors.WHITE, 2)
        cv2.line(frame, (x + self.width//3, y), 
                (x + self.width//3 + 15, y + claw_offset), DoomColors.WHITE, 2)
    
    def _draw_demon(self, frame, x, y, color):
        """Draw demon enemy (pink demon)"""
        # Quadruped body
        cv2.ellipse(frame, (x, y), (self.width//2, self.height//3), 0, 0, 360, color, -1)
        
        # Head
        cv2.circle(frame, (x - self.width//3, y - self.height//4), self.width//4, color, -1)
        
        # Massive jaws
        jaw_points = np.array([
            [x - self.width//2, y - self.height//4],
            [x - self.width//2 - 20, y],
            [x - self.width//3, y + 5],
            [x - self.width//3, y - self.height//4]
        ], np.int32)
        cv2.fillPoly(frame, [jaw_points], color)
        
        # Sharp teeth
        for i in range(3):
            tooth_x = x - self.width//2 - 5 + (i * 8)
            cv2.line(frame, (tooth_x, y - 5), (tooth_x, y + 5), DoomColors.WHITE, 2)
        
        # Eyes
        cv2.circle(frame, (x - self.width//3 - 5, y - self.height//4), 2, DoomColors.BRIGHT_RED, -1)
        
        # Legs (animated)
        leg_offset = int(5 * math.sin(self.animation_frame * 2))
        for leg_x in [x - self.width//3, x, x + self.width//3]:
            cv2.line(frame, (leg_x, y + self.height//3), 
                    (leg_x + leg_offset, y + self.height//2), color, 4)
    
    def _draw_cacodemon(self, frame, x, y, color):
        """Draw cacodemon enemy (floating red sphere)"""
        # Main spherical body with floating animation
        float_offset = int(10 * math.sin(time.time() * 2 + self.position.x * 0.01))
        draw_y = y + float_offset
        
        cv2.circle(frame, (x, draw_y), self.width//2, color, -1)
        
        # Eye (large, central)
        eye_size = self.width//4
        cv2.circle(frame, (x, draw_y - 10), eye_size, DoomColors.BLACK, -1)
        cv2.circle(frame, (x, draw_y - 10), eye_size - 3, DoomColors.BRIGHT_RED, -1)
        cv2.circle(frame, (x + 3, draw_y - 13), eye_size // 3, DoomColors.WHITE, -1)
        
        # Mouth with teeth
        mouth_points = np.array([
            [x - self.width//3, draw_y + 10],
            [x + self.width//3, draw_y + 10],
            [x, draw_y + self.width//3]
        ], np.int32)
        cv2.fillPoly(frame, [mouth_points], DoomColors.BLACK)
        
        # Teeth around the mouth
        for i in range(8):
            angle = i * math.pi / 4
            tooth_x = x + int((self.width//3 - 5) * math.cos(angle))
            tooth_y = draw_y + 10 + int(15 * math.sin(angle))
            cv2.line(frame, (tooth_x, tooth_y), 
                    (tooth_x + int(8 * math.cos(angle)), tooth_y + int(8 * math.sin(angle))), 
                    DoomColors.WHITE, 2)
    
    def _draw_baron(self, frame, x, y, color):
        """Draw baron of hell enemy"""
        # Large imposing figure
        cv2.rectangle(frame, (x - self.width//3, y - self.height//2), 
                     (x + self.width//3, y + self.height//2), color, -1)
        
        # Massive head
        cv2.circle(frame, (x, y - self.height//2 - 15), self.width//2, color, -1)
        
        # Large horns
        horn_left = np.array([[x - 20, y - self.height//2 - 25], 
                             [x - 35, y - self.height//2 - 45], 
                             [x - 10, y - self.height//2 - 30]], np.int32)
        horn_right = np.array([[x + 20, y - self.height//2 - 25], 
                              [x + 35, y - self.height//2 - 45], 
                              [x + 10, y - self.height//2 - 30]], np.int32)
        cv2.fillPoly(frame, [horn_left], DoomColors.DARK_GRAY)
        cv2.fillPoly(frame, [horn_right], DoomColors.DARK_GRAY)
        
        # Glowing eyes
        cv2.circle(frame, (x - 12, y - self.height//2 - 15), 5, DoomColors.BRIGHT_RED, -1)
        cv2.circle(frame, (x + 12, y - self.height//2 - 15), 5, DoomColors.BRIGHT_RED, -1)
        
        # Fireballs in hands
        fireball_intensity = 0.5 + 0.5 * math.sin(time.time() * 3)
        fireball_color = tuple(int(c * fireball_intensity) for c in DoomColors.HUD_ORANGE)
        cv2.circle(frame, (x - self.width//2, y), 8, fireball_color, -1)
        cv2.circle(frame, (x + self.width//2, y), 8, fireball_color, -1)
    
    def _draw_cyberdemon(self, frame, x, y, color):
        """Draw cyberdemon enemy (boss)"""
        # Massive mechanical body
        cv2.rectangle(frame, (x - self.width//2, y - self.height//2), 
                     (x + self.width//2, y + self.height//2), DoomColors.METAL_GRAY, -1)
        
        # Demon head
        cv2.circle(frame, (x, y - self.height//2 - 20), self.width//3, color, -1)
        
        # Cybernetic eye
        cv2.circle(frame, (x, y - self.height//2 - 20), 8, DoomColors.BRIGHT_RED, -1)
        cv2.circle(frame, (x, y - self.height//2 - 20), 5, DoomColors.WHITE, -1)
        
        # Rocket launcher arm
        launcher_points = np.array([
            [x + self.width//2, y - 20],
            [x + self.width//2 + 30, y - 30],
            [x + self.width//2 + 35, y - 10],
            [x + self.width//2 + 5, y]
        ], np.int32)
        cv2.fillPoly(frame, [launcher_points], DoomColors.DARK_GRAY)
        
        # Mechanical legs
        cv2.rectangle(frame, (x - 15, y + self.height//2), 
                     (x - 5, y + self.height//2 + 20), DoomColors.METAL_GRAY, -1)
        cv2.rectangle(frame, (x + 5, y + self.height//2), 
                     (x + 15, y + self.height//2 + 20), DoomColors.METAL_GRAY, -1)
        
        # Warning lights
        if int(time.time() * 2) % 2:
            cv2.circle(frame, (x - 20, y - 10), 3, DoomColors.BRIGHT_RED, -1)
            cv2.circle(frame, (x + 20, y - 10), 3, DoomColors.BRIGHT_RED, -1)
    
    def _draw_death_animation(self, frame, x, y, progress):
        """Draw death animation"""
        # Fade out and fall down
        alpha = 1.0 - progress
        fall_offset = int(progress * 30)
        
        # Draw simplified death sprite
        death_color = tuple(int(c * alpha) for c in DoomColors.DARK_RED)
        cv2.ellipse(frame, (x, y + fall_offset), 
                   (int(self.width * alpha), int(self.height//3 * alpha)), 
                   0, 0, 360, death_color, -1)
    
    def _draw_health_bar(self, frame, x, y):
        """Draw enemy health bar"""
        if self.health >= self.max_health:
            return  # Don't show health bar at full health
        
        health_ratio = self.health / self.max_health
        bar_width = self.width
        bar_height = 6
        bar_x = x - bar_width // 2
        bar_y = y - self.height // 2 - 30
        
        # Background
        cv2.rectangle(frame, (bar_x, bar_y), 
                     (bar_x + bar_width, bar_y + bar_height), 
                     DoomColors.DARK_GRAY, -1)
        
        # Health bar
        health_width = int(bar_width * health_ratio)
        health_color = (DoomColors.HUD_GREEN if health_ratio > 0.5 else 
                       DoomColors.HUD_YELLOW if health_ratio > 0.25 else 
                       DoomColors.HUD_RED)
        
        cv2.rectangle(frame, (bar_x, bar_y), 
                     (bar_x + health_width, bar_y + bar_height), 
                     health_color, -1)

class Pickup:
    """Collectible items in the game"""
    def __init__(self, x: float, y: float, pickup_type: PickupType):
        self.position = Vector2D(x, y)
        self.pickup_type = pickup_type
        self.alive = True
        self.creation_time = time.time()
        self.bob_offset = 0
        self.rotation = 0
        
        # Setup pickup properties
        self.value = 0
        self.color = DoomColors.HUD_GREEN
        self.size = 15
        self.name = ""
        
        self._setup_pickup_stats()
    
    def _setup_pickup_stats(self):
        """Setup pickup-specific properties"""
        pickup_stats = {
            PickupType.HEALTH_SMALL: {"value": 10, "color": DoomColors.HUD_GREEN, "size": 12, "name": "STIM PACK"},
            PickupType.HEALTH_LARGE: {"value": 25, "color": DoomColors.HUD_GREEN, "size": 18, "name": "MEDIKIT"},
            PickupType.ARMOR: {"value": 50, "color": DoomColors.PLASMA_BLUE, "size": 16, "name": "ARMOR"},
            PickupType.AMMO_CLIP: {"value": 20, "color": DoomColors.HUD_YELLOW, "size": 10, "name": "CLIP"},
            PickupType.AMMO_SHELLS: {"value": 8, "color": DoomColors.HUD_ORANGE, "size": 12, "name": "SHELLS"},
            PickupType.AMMO_ROCKETS: {"value": 5, "color": DoomColors.HUD_RED, "size": 14, "name": "ROCKETS"},
            PickupType.KEYCARD_RED: {"value": 1, "color": DoomColors.BRIGHT_RED, "size": 16, "name": "RED KEY"},
            PickupType.KEYCARD_BLUE: {"value": 1, "color": DoomColors.PLASMA_BLUE, "size": 16, "name": "BLUE KEY"},
            PickupType.KEYCARD_YELLOW: {"value": 1, "color": DoomColors.HUD_YELLOW, "size": 16, "name": "YELLOW KEY"}
        }
        
        if self.pickup_type in pickup_stats:
            stats = pickup_stats[self.pickup_type]
            for key, value in stats.items():
                setattr(self, key, value)
    
    def update(self, dt: float):
        """Update pickup animations"""
        if not self.alive:
            return
        
        # Bobbing animation
        self.bob_offset = math.sin(time.time() * 3 + self.position.x * 0.1) * 8
        
        # Rotation animation for keycards
        if self.pickup_type in [PickupType.KEYCARD_RED, PickupType.KEYCARD_BLUE, PickupType.KEYCARD_YELLOW]:
            self.rotation += dt * 2
    
    def check_collision(self, player_pos: Vector2D, pickup_radius: float = 30) -> bool:
        """Check if player picked up this item"""
        if not self.alive:
            return False
        
        distance = self.position.distance_to(player_pos)
        return distance < pickup_radius
    
    def draw(self, frame):
        """Draw pickup item"""
        if not self.alive:
            return
        
        x = int(self.position.x)
        y = int(self.position.y + self.bob_offset)
        
        # Draw pickup based on type
        if self.pickup_type == PickupType.HEALTH_SMALL:
            # Small health cross
            cv2.line(frame, (x - 6, y), (x + 6, y), self.color, 3)
            cv2.line(frame, (x, y - 6), (x, y + 6), self.color, 3)
        
        elif self.pickup_type == PickupType.HEALTH_LARGE:
            # Large medikit
            cv2.rectangle(frame, (x - 10, y - 8), (x + 10, y + 8), DoomColors.WHITE, -1)
            cv2.line(frame, (x - 6, y), (x + 6, y), self.color, 3)
            cv2.line(frame, (x, y - 6), (x, y + 6), self.color, 3)
        
        elif self.pickup_type == PickupType.ARMOR:
            # Armor vest
            armor_points = np.array([
                [x, y - 12], [x - 10, y - 8], [x - 12, y + 8],
                [x + 12, y + 8], [x + 10, y - 8]
            ], np.int32)
            cv2.fillPoly(frame, [armor_points], self.color)
        
        elif self.pickup_type == PickupType.AMMO_CLIP:
            # Ammo clip
            cv2.rectangle(frame, (x - 4, y - 8), (x + 4, y + 8), self.color, -1)
            cv2.rectangle(frame, (x - 3, y - 6), (x + 3, y + 6), DoomColors.DARK_GRAY, -1)
        
        elif self.pickup_type == PickupType.AMMO_SHELLS:
            # Shotgun shells
            for i in range(4):
                shell_x = x - 6 + (i * 4)
                cv2.rectangle(frame, (shell_x - 1, y - 6), (shell_x + 1, y + 6), self.color, -1)
        
        elif self.pickup_type == PickupType.AMMO_ROCKETS:
            # Rocket
            cv2.ellipse(frame, (x, y), (12, 4), 0, 0, 360, self.color, -1)
            cv2.circle(frame, (x + 8, y), 4, DoomColors.HUD_ORANGE, -1)
        
        elif self.pickup_type in [PickupType.KEYCARD_RED, PickupType.KEYCARD_BLUE, PickupType.KEYCARD_YELLOW]:
            # Rotating keycard
            card_size = self.size
            # Simple keycard representation
            cv2.rectangle(frame, (x - card_size//2, y - card_size//3), 
                         (x + card_size//2, y + card_size//3), self.color, -1)
            cv2.rectangle(frame, (x - card_size//3, y - card_size//4), 
                         (x + card_size//3, y + card_size//4), DoomColors.WHITE, -1)
        
        # Glow effect
        glow_intensity = 0.3 + 0.3 * math.sin(time.time() * 4)
        glow_color = tuple(int(c * glow_intensity) for c in self.color)
        cv2.circle(frame, (x, y), self.size + 5, glow_color, 2)

class ParticleEffect:
    """Particle system for explosions, blood, etc."""
    def __init__(self, x: float, y: float, effect_type: str, count: int = 20):
        self.position = Vector2D(x, y)
        self.effect_type = effect_type
        self.particles = []
        self.creation_time = time.time()
        self.lifetime = 2.0
        
        # Create particles
        for _ in range(count):
            particle = {
                'pos': Vector2D(x + random.uniform(-10, 10), y + random.uniform(-10, 10)),
                'vel': Vector2D(random.uniform(-100, 100), random.uniform(-100, 100)),
                'color': self._get_particle_color(),
                'size': random.randint(1, 4),
                'life': 1.0
            }
            self.particles.append(particle)
    
    def _get_particle_color(self):
        """Get color based on effect type"""
        if self.effect_type == "blood":
            return DoomColors.BLOOD_RED
        elif self.effect_type == "explosion":
            return random.choice([DoomColors.HUD_ORANGE, DoomColors.HUD_RED, DoomColors.HUD_YELLOW])
        elif self.effect_type == "sparks":
            return DoomColors.HUD_YELLOW
        else:
            return DoomColors.WHITE
    
    def update(self, dt: float):
        """Update particle positions and lifetimes"""
        for particle in self.particles:
            # Update position
            particle['pos'] = particle['pos'] + (particle['vel'] * dt)
            
            # Apply gravity for blood/explosion effects
            if self.effect_type in ["blood", "explosion"]:
                particle['vel'].y += 200 * dt  # Gravity
            
            # Reduce life
            particle['life'] -= dt / self.lifetime
    
    def draw(self, frame):
        """Draw all particles"""
        for particle in self.particles:
            if particle['life'] <= 0:
                continue
            
            x, y = int(particle['pos'].x), int(particle['pos'].y)
            alpha = particle['life']
            
            if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                color = tuple(int(c * alpha) for c in particle['color'])
                size = max(1, int(particle['size'] * alpha))
                cv2.circle(frame, (x, y), size, color, -1)
    
    def is_alive(self) -> bool:
        """Check if effect is still active"""
        return any(p['life'] > 0 for p in self.particles)

class GameMap:
    """Game level/map system"""
    def __init__(self, level_number: int):
        self.level_number = level_number
        self.width = 1280
        self.height = 720
        self.enemies = []
        self.pickups = []
        self.spawn_points = []
        self.walls = []  # For future collision detection
        
        self._generate_level()
    
    def _generate_level(self):
        """Generate level content based on level number"""
        # Increase difficulty with level number
        enemy_count = min(3 + self.level_number, 15)
        pickup_count = max(5, 10 - self.level_number // 2)
        
        # Generate enemies
        enemy_types = list(EnemyType)
        available_types = enemy_types[:min(len(enemy_types), 2 + self.level_number // 2)]
        
        for _ in range(enemy_count):
            x = random.randint(100, self.width - 100)
            y = random.randint(100, self.height - 100)
            enemy_type = random.choice(available_types)
            self.enemies.append(Enemy(x, y, enemy_type, self.width, self.height))
        
        # Generate pickups
        pickup_types = [
            PickupType.HEALTH_SMALL, PickupType.HEALTH_LARGE, PickupType.ARMOR,
            PickupType.AMMO_CLIP, PickupType.AMMO_SHELLS, PickupType.AMMO_ROCKETS
        ]
        
        for _ in range(pickup_count):
            x = random.randint(50, self.width - 50)
            y = random.randint(50, self.height - 50)
            pickup_type = random.choice(pickup_types)
            self.pickups.append(Pickup(x, y, pickup_type))
        
        # Add keycards for later levels
        if self.level_number > 1:
            keycard_types = [PickupType.KEYCARD_RED, PickupType.KEYCARD_BLUE, PickupType.KEYCARD_YELLOW]
            keycard_type = keycard_types[(self.level_number - 2) % 3]
            x = random.randint(200, self.width - 200)
            y = random.randint(200, self.height - 200)
            self.pickups.append(Pickup(x, y, keycard_type))
    
    def get_enemies_alive(self) -> int:
        """Get count of living enemies"""
        return len([e for e in self.enemies if e.alive])
    
    def is_complete(self) -> bool:
        """Check if level is complete"""
        return self.get_enemies_alive() == 0

class Achievement:
    """Achievement system"""
    def __init__(self, name: str, description: str, condition_func):
        self.name = name
        self.description = description
        self.condition_func = condition_func
        self.unlocked = False
        self.unlock_time = 0

class GameStats:
    """Game statistics tracking"""
    def __init__(self):
        self.shots_fired = 0
        self.hits_landed = 0
        self.enemies_killed = 0
        self.damage_dealt = 0
        self.damage_taken = 0
        self.pickups_collected = 0
        self.levels_completed = 0
        self.playtime = 0
        self.start_time = time.time()
    
    def get_accuracy(self) -> float:
        """Calculate shooting accuracy"""
        if self.shots_fired == 0:
            return 0.0
        return (self.hits_landed / self.shots_fired) * 100
    
    def update_playtime(self):
        """Update total playtime"""
        self.playtime = time.time() - self.start_time

class UltimateDoomGame:
    """Main game class - Ultimate Doom experience"""
    def __init__(self, screen_width=1280, screen_height=720):
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.7
        )
        
        # Game configuration
        self.config = GameConfig()
        
        # Game state
        self.state = GameState.MAIN_MENU
        self.current_level = 1
        self.game_map = None
        
        # Player stats
        self.health = 100
        self.max_health = 100
        self.armor = 0
        self.max_armor = 200
        self.score = 0
        self.lives = 3
        
        # Player position and movement
        self.player_pos = Vector2D(screen_width // 2, screen_height // 2)
        
        # Weapon system
        self.weapons = {wtype: Weapon(wtype) for wtype in WeaponType}
        self.current_weapon_type = WeaponType.PISTOL
        self.unlocked_weapons = {WeaponType.FIST, WeaponType.PISTOL}
        
        # Inventory
        self.keycards = set()
        self.ammo = {
            "bullets": 50,
            "shells": 8,
            "rockets": 0,
            "cells": 0
        }
        
        # Hand tracking
        self.left_hand = None
        self.right_hand = None
        self.weapon_hand = None
        
        # Game objects
        self.projectiles = []
        self.particle_effects = []
        
        # Visual effects
        self.screen_shake = 0
        self.screen_flash = 0
        self.recoil_offset = 0
        self.crosshair_pos = Vector2D(screen_width // 2, screen_height // 2)
        
        # UI and menus
        self.back_button = BackButton(screen_width, screen_height)
        self.menu_selection = 0
        self.show_fps = True
        self.fps_counter = 0
        self.last_fps_time = time.time()
        
        # Game statistics and achievements
        self.stats = GameStats()
        self.achievements = self._setup_achievements()
        
        # Timing
        self.last_update_time = time.time()
        self.game_start_time = time.time()
        
        # Sound system (placeholder)
        self.sound_queue = []
        
    def _setup_achievements(self) -> List[Achievement]:
        """Setup game achievements"""
        achievements = [
            Achievement("First Blood", "Kill your first enemy", 
                       lambda: self.stats.enemies_killed >= 1),
            Achievement("Marksman", "Achieve 80% accuracy", 
                       lambda: self.stats.get_accuracy() >= 80 and self.stats.shots_fired >= 50),
            Achievement("Survivor", "Complete level 1", 
                       lambda: self.stats.levels_completed >= 1),
            Achievement("Demon Slayer", "Kill 100 enemies", 
                       lambda: self.stats.enemies_killed >= 100),
            Achievement("Arsenal Master", "Unlock all weapons", 
                       lambda: len(self.unlocked_weapons) >= len(WeaponType)),
            Achievement("Nightmare Warrior", "Complete level 5", 
                       lambda: self.stats.levels_completed >= 5),
            Achievement("Perfectionist", "Complete a level without taking damage", 
                       lambda: hasattr(self, 'perfect_level') and self.perfect_level),
            Achievement("Speed Demon", "Complete level 1 in under 60 seconds", 
                       lambda: hasattr(self, 'level_1_time') and self.level_1_time < 60),
        ]
        return achievements
    
    def detect_enhanced_gesture(self, landmarks, hand_label: str) -> Tuple[str, Optional[Vector2D]]:
        """Enhanced gesture detection with position"""
        if len(landmarks.landmark) < 21:
            return "unknown", None
        
        # Get finger landmarks
        thumb_tip = landmarks.landmark[4]
        index_tip = landmarks.landmark[8]
        middle_tip = landmarks.landmark[12]
        ring_tip = landmarks.landmark[16]
        pinky_tip = landmarks.landmark[20]
        
        thumb_pip = landmarks.landmark[3]
        index_pip = landmarks.landmark[6]
        middle_pip = landmarks.landmark[10]
        ring_pip = landmarks.landmark[14]
        pinky_pip = landmarks.landmark[18]
        
        wrist = landmarks.landmark[0]
        
        # Convert to screen coordinates
        wrist_pos = Vector2D(wrist.x * self.screen_width, wrist.y * self.screen_height)
        index_pos = Vector2D(index_tip.x * self.screen_width, index_tip.y * self.screen_height)
        
        # Check finger states
        index_extended = index_tip.y < index_pip.y
        middle_extended = middle_tip.y < middle_pip.y
        ring_extended = ring_tip.y < ring_pip.y
        pinky_extended = pinky_tip.y < pinky_pip.y
        
        # Thumb state (depends on hand orientation)
        if hand_label == "Right":
            thumb_extended = thumb_tip.x > thumb_pip.x
        else:
            thumb_extended = thumb_tip.x < thumb_pip.x
        
        # Gesture classification
        extended_fingers = sum([index_extended, middle_extended, ring_extended, pinky_extended, thumb_extended])
        
        # Gun gesture (index only)
        if index_extended and not middle_extended and not ring_extended and not pinky_extended:
            return "gun", index_pos
        
        # Open palm (all extended)
        elif extended_fingers >= 4:
            return "open_palm", wrist_pos
        
        # Fist (all closed)
        elif extended_fingers == 0:
            return "fist", wrist_pos
        
        # Thumbs up
        elif thumb_extended and extended_fingers == 1:
            return "thumbs_up", wrist_pos
        
        # Peace/victory sign
        elif index_extended and middle_extended and extended_fingers == 2:
            return "peace", wrist_pos
        
        # Three fingers (weapon select)
        elif index_extended and middle_extended and ring_extended and extended_fingers == 3:
            return "three", wrist_pos
        
        return "unknown", wrist_pos
    
    def check_reload_gesture(self) -> bool:
        """Check for reload gesture (hands together)"""
        if not self.left_hand or not self.right_hand:
            return False
        
        left_wrist = Vector2D(self.left_hand.landmark[0].x, self.left_hand.landmark[0].y)
        right_wrist = Vector2D(self.right_hand.landmark[0].x, self.right_hand.landmark[0].y)
        
        # Check if hands are close together
        distance = left_wrist.distance_to(right_wrist)
        return distance < 0.15  # Threshold for hands being together
    
    def handle_gesture_input(self):
        """Process hand gestures for game input"""
        if not self.left_hand and not self.right_hand:
            return
        
        # Process each hand
        left_gesture, left_pos = ("unknown", None)
        right_gesture, right_pos = ("unknown", None)
        
        if self.left_hand:
            left_gesture, left_pos = self.detect_enhanced_gesture(self.left_hand, "Left")
        
        if self.right_hand:
            right_gesture, right_pos = self.detect_enhanced_gesture(self.right_hand, "Right")
        
        # Weapon handling (prefer right hand for shooting)
        if right_gesture == "gun" and right_pos:
            self.weapon_hand = self.right_hand
            self.crosshair_pos = right_pos
            self.handle_shooting()
        elif left_gesture == "gun" and left_pos:
            self.weapon_hand = self.left_hand
            self.crosshair_pos = left_pos
            self.handle_shooting()
        
        # Melee attacks
        if right_gesture == "fist" or left_gesture == "fist":
            self.handle_melee_attack()
        
        # Weapon switching
        if right_gesture == "thumbs_up" or left_gesture == "thumbs_up":
            self.switch_weapon(1)
        elif right_gesture == "peace":
            self.switch_weapon(-1)
        
        # Reload gesture
        if self.check_reload_gesture():
            self.current_weapon.start_reload()
        
        # Movement (open palm gestures)
        if left_gesture == "open_palm" and left_pos:
            self.handle_movement(left_pos, "left")
        if right_gesture == "open_palm" and right_pos:
            self.handle_movement(right_pos, "right")
    
    @property
    def current_weapon(self) -> Weapon:
        """Get current weapon"""
        return self.weapons[self.current_weapon_type]
    
    def switch_weapon(self, direction: int):
        """Switch to next/previous weapon"""
        weapons_list = list(self.unlocked_weapons)
        if not weapons_list:
            return
        
        weapons_list.sort(key=lambda x: x.value)
        
        try:
            current_index = weapons_list.index(self.current_weapon_type)
            new_index = (current_index + direction) % len(weapons_list)
            self.current_weapon_type = weapons_list[new_index]
        except ValueError:
            self.current_weapon_type = weapons_list[0]
    
    def handle_shooting(self):
        """Handle weapon firing"""
        if self.state != GameState.PLAYING:
            return
        
        weapon = self.current_weapon
        if not weapon.can_shoot():
            return
        
        if weapon.shoot():
            self.stats.shots_fired += 1
            
            # Create projectile(s)
            if weapon.type == WeaponType.FIST:
                self.handle_melee_attack()
                return
            elif weapon.type == WeaponType.SHOTGUN:
                # Shotgun fires multiple pellets
                pellet_count = 7
                for _ in range(pellet_count):
                    spread_angle = random.uniform(-weapon.spread, weapon.spread)
                    direction = self.calculate_shot_direction(spread_angle)
                    projectile = Projectile(self.crosshair_pos, direction, weapon.type)
                    self.projectiles.append(projectile)
            else:
                # Single projectile weapons
                spread_angle = random.uniform(-weapon.spread, weapon.spread)
                direction = self.calculate_shot_direction(spread_angle)
                projectile = Projectile(self.crosshair_pos, direction, weapon.type)
                self.projectiles.append(projectile)
            
            # Visual effects
            self.recoil_offset = weapon.recoil_strength
            if weapon.type in [WeaponType.ROCKET_LAUNCHER, WeaponType.BFG]:
                self.screen_shake = 10
            
            # Sound effect (placeholder)
            self.sound_queue.append(f"weapon_fire_{weapon.type.name.lower()}")
    
    def calculate_shot_direction(self, spread_angle: float = 0) -> Vector2D:
        """Calculate shot direction from crosshair"""
        # For hand gesture aiming, direction is from center to crosshair
        center = Vector2D(self.screen_width // 2, self.screen_height // 2)
        direction = (self.crosshair_pos - center).normalize()
        
        # Apply spread
        if spread_angle != 0:
            cos_angle = math.cos(spread_angle)
            sin_angle = math.sin(spread_angle)
            new_x = direction.x * cos_angle - direction.y * sin_angle
            new_y = direction.x * sin_angle + direction.y * cos_angle
            direction = Vector2D(new_x, new_y)
        
        return direction
    
    def handle_melee_attack(self):
        """Handle melee/fist attacks"""
        if self.state != GameState.PLAYING:
            return
        
        melee_range = 100
        melee_damage = 20 if self.current_weapon_type == WeaponType.FIST else 30
        
        # Check for enemies in melee range
        for enemy in self.game_map.enemies:
            if not enemy.alive:
                continue
            
            distance = self.player_pos.distance_to(enemy.position)
            if distance < melee_range:
                if enemy.take_damage(melee_damage):
                    self.stats.enemies_killed += 1
                    self.score += enemy.score_value
                
                # Add blood effect
                self.add_particle_effect(enemy.position.x, enemy.position.y, "blood")
                break
    
    def handle_movement(self, hand_pos: Vector2D, hand_side: str):
        """Handle player movement based on hand position"""
        if self.state != GameState.PLAYING:
            return
        
        # Movement based on hand position relative to screen
        center = Vector2D(self.screen_width // 2, self.screen_height // 2)
        movement_vector = hand_pos - center
        
        # Normalize and apply movement speed
        if movement_vector.magnitude() > 50:  # Dead zone
            movement_vector = movement_vector.normalize() * 100  # Movement speed
            
            # Apply movement (with boundary checking)
            new_pos = self.player_pos + movement_vector * 0.02  # Movement factor
            
            self.player_pos.x = max(50, min(self.screen_width - 50, new_pos.x))
            self.player_pos.y = max(50, min(self.screen_height - 50, new_pos.y))
    
    def update_projectiles(self, dt: float):
        """Update all projectiles"""
        for projectile in self.projectiles[:]:  # Copy list for safe iteration
            if not projectile.alive:
                self.projectiles.remove(projectile)
                continue
            
            projectile.update(dt)
            
            # Check bounds
            if (projectile.position.x < 0 or projectile.position.x > self.screen_width or
                projectile.position.y < 0 or projectile.position.y > self.screen_height):
                projectile.alive = False
                continue
            
            # Check enemy collisions
            for enemy in self.game_map.enemies:
                if enemy.check_collision_with_projectile(projectile):
                    self.stats.hits_landed += 1
                    self.stats.damage_dealt += projectile.damage
                    
                    if enemy.take_damage(projectile.damage):
                        self.stats.enemies_killed += 1
                        self.score += enemy.score_value
                        
                        # Unlock new weapons based on kills
                        self.check_weapon_unlocks()
                    
                    # Create hit effects
                    self.add_particle_effect(enemy.position.x, enemy.position.y, "blood")
                    
                    # Handle explosive projectiles
                    if projectile.explosive:
                        self.create_explosion(projectile.position, projectile.explosion_radius, projectile.damage)
                    
                    projectile.alive = False
                    break
    
    def create_explosion(self, position: Vector2D, radius: float, damage: int):
        """Create explosion effect and damage"""
        # Visual explosion effect
        self.add_particle_effect(position.x, position.y, "explosion", 30)
        self.screen_shake = 15
        self.screen_flash = 0.5
        
        # Damage enemies in radius
        for enemy in self.game_map.enemies:
            if not enemy.alive:
                continue
            
            distance = position.distance_to(enemy.position)
            if distance < radius:
                # Damage falls off with distance
                explosion_damage = int(damage * (1.0 - distance / radius))
                if enemy.take_damage(explosion_damage):
                    self.stats.enemies_killed += 1
                    self.score += enemy.score_value
    
    def check_weapon_unlocks(self):
        """Check and unlock new weapons based on progress"""
        kills = self.stats.enemies_killed
        
        if kills >= 5 and WeaponType.SHOTGUN not in self.unlocked_weapons:
            self.unlocked_weapons.add(WeaponType.SHOTGUN)
            self.add_message("SHOTGUN UNLOCKED!")
        
        if kills >= 15 and WeaponType.CHAINGUN not in self.unlocked_weapons:
            self.unlocked_weapons.add(WeaponType.CHAINGUN)
            self.add_message("CHAINGUN UNLOCKED!")
        
        if kills >= 25 and WeaponType.ROCKET_LAUNCHER not in self.unlocked_weapons:
            self.unlocked_weapons.add(WeaponType.ROCKET_LAUNCHER)
            self.add_message("ROCKET LAUNCHER UNLOCKED!")
        
        if kills >= 50 and WeaponType.PLASMA_RIFLE not in self.unlocked_weapons:
            self.unlocked_weapons.add(WeaponType.PLASMA_RIFLE)
            self.add_message("PLASMA RIFLE UNLOCKED!")
        
        if kills >= 100 and WeaponType.BFG not in self.unlocked_weapons:
            self.unlocked_weapons.add(WeaponType.BFG)
            self.add_message("BFG 9000 UNLOCKED!")
    
    def add_message(self, message: str):
        """Add message to display queue (placeholder)"""
        # This would add to a message queue for display
        pass
    
    def add_particle_effect(self, x: float, y: float, effect_type: str, count: int = 20):
        """Add particle effect at position"""
        effect = ParticleEffect(x, y, effect_type, count)
        self.particle_effects.append(effect)
    
    def update_enemies(self, dt: float):
        """Update all enemies"""
        if not self.game_map:
            return
        
        for enemy in self.game_map.enemies:
            if enemy.alive:
                damage = enemy.update(self.player_pos, dt)
                if damage > 0:
                    self.take_damage(damage)
    
    def take_damage(self, damage: int):
        """Apply damage to player"""
        # Armor absorbs damage first
        if self.armor > 0:
            armor_absorbed = min(damage // 2, self.armor)
            self.armor -= armor_absorbed
            damage -= armor_absorbed
        
        self.health -= damage
        self.stats.damage_taken += damage
        
        # Screen effects for taking damage
        self.screen_flash = 0.3
        self.screen_shake = 5
        
        if self.health <= 0:
            self.handle_player_death()
    
    def handle_player_death(self):
        """Handle player death"""
        self.lives -= 1
        if self.lives <= 0:
            self.state = GameState.GAME_OVER
        else:
            # Respawn
            self.health = self.max_health
            self.armor = 0
            self.player_pos = Vector2D(self.screen_width // 2, self.screen_height // 2)
    
    def update_pickups(self):
        """Update and check pickup collection"""
        if not self.game_map:
            return
        
        for pickup in self.game_map.pickups[:]:
            pickup.update(0.016)  # Assuming 60 FPS
            
            if pickup.check_collision(self.player_pos):
                self.collect_pickup(pickup)
                self.game_map.pickups.remove(pickup)
    
    def collect_pickup(self, pickup: Pickup):
        """Collect a pickup item"""
        self.stats.pickups_collected += 1
        
        if pickup.pickup_type == PickupType.HEALTH_SMALL:
            self.health = min(self.max_health, self.health + pickup.value)
        elif pickup.pickup_type == PickupType.HEALTH_LARGE:
            self.health = min(self.max_health, self.health + pickup.value)
        elif pickup.pickup_type == PickupType.ARMOR:
            self.armor = min(self.max_armor, self.armor + pickup.value)
        elif pickup.pickup_type == PickupType.AMMO_CLIP:
            self.ammo["bullets"] += pickup.value
        elif pickup.pickup_type == PickupType.AMMO_SHELLS:
            self.ammo["shells"] += pickup.value
        elif pickup.pickup_type == PickupType.AMMO_ROCKETS:
            self.ammo["rockets"] += pickup.value
        elif pickup.pickup_type in [PickupType.KEYCARD_RED, PickupType.KEYCARD_BLUE, PickupType.KEYCARD_YELLOW]:
            self.keycards.add(pickup.pickup_type)
        
        # Update weapon ammo
        self.update_weapon_ammo()
        
        # Sound effect
        self.sound_queue.append("pickup_item")
    
    def update_weapon_ammo(self):
        """Update weapon ammo from inventory"""
        # Update ammo for weapons based on inventory
        if self.current_weapon_type in [WeaponType.PISTOL, WeaponType.CHAINGUN]:
            ammo_type = "bullets"
        elif self.current_weapon_type == WeaponType.SHOTGUN:
            ammo_type = "shells"
        elif self.current_weapon_type == WeaponType.ROCKET_LAUNCHER:
            ammo_type = "rockets"
        elif self.current_weapon_type in [WeaponType.PLASMA_RIFLE, WeaponType.BFG]:
            ammo_type = "cells"
        else:
            return
        
        # Refill weapon if it has ammo available
        weapon = self.current_weapon
        if weapon.current_ammo == 0 and self.ammo[ammo_type] > 0:
            ammo_to_load = min(weapon.max_ammo, self.ammo[ammo_type])
            weapon.current_ammo = ammo_to_load
            self.ammo[ammo_type] -= ammo_to_load
    
    def check_achievements(self):
        """Check and unlock achievements"""
        current_time = time.time()
        
        for achievement in self.achievements:
            if not achievement.unlocked and achievement.condition_func():
                achievement.unlocked = True
                achievement.unlock_time = current_time
                self.add_message(f"ACHIEVEMENT UNLOCKED: {achievement.name}")
    
    def start_new_game(self):
        """Start a new game"""
        self.state = GameState.PLAYING
        self.current_level = 1
        self.health = self.max_health
        self.armor = 0
        self.score = 0
        self.lives = 3
        self.stats = GameStats()
        
        # Reset position
        self.player_pos = Vector2D(self.screen_width // 2, self.screen_height // 2)
        
        # Reset weapons
        self.current_weapon_type = WeaponType.PISTOL
        self.unlocked_weapons = {WeaponType.FIST, WeaponType.PISTOL}
        
        # Load first level
        self.load_level(1)
    
    def load_level(self, level_number: int):
        """Load a specific level"""
        self.current_level = level_number
        self.game_map = GameMap(level_number)
        
        # Clear existing game objects
        self.projectiles.clear()
        self.particle_effects.clear()
        
        # Reset level-specific stats
        self.level_start_time = time.time()
        self.level_damage_taken = 0
    
    def complete_level(self):
        """Handle level completion"""
        self.stats.levels_completed += 1
        
        # Check for perfect level achievement
        if self.level_damage_taken == 0:
            self.perfect_level = True
        
        # Check for speed run achievement
        level_time = time.time() - self.level_start_time
        if self.current_level == 1 and level_time < 60:
            self.level_1_time = level_time
        
        self.state = GameState.LEVEL_COMPLETE
        
        # Bonus points
        time_bonus = max(0, int((120 - level_time) * 10))  # Bonus for completing quickly
        health_bonus = self.health * 2
        self.score += time_bonus + health_bonus
    
    def update_visual_effects(self, dt: float):
        """Update visual effects"""
        # Screen shake
        if self.screen_shake > 0:
            self.screen_shake -= dt * 30
            self.screen_shake = max(0, self.screen_shake)
        
        # Screen flash
        if self.screen_flash > 0:
            self.screen_flash -= dt * 2
            self.screen_flash = max(0, self.screen_flash)
        
        # Recoil
        if self.recoil_offset > 0:
            self.recoil_offset -= dt * 40
            self.recoil_offset = max(0, self.recoil_offset)
        
        # Update particle effects
        for effect in self.particle_effects[:]:
            effect.update(dt)
            if not effect.is_alive():
                self.particle_effects.remove(effect)
    
    def update(self):
        """Main game update loop"""
        current_time = time.time()
        dt = current_time - self.last_update_time
        self.last_update_time = current_time
        
        # Update stats
        self.stats.update_playtime()
        
        # Update based on game state
        if self.state == GameState.PLAYING:
            # Update weapons
            for weapon in self.weapons.values():
                weapon.update()
            
            # Update game objects
            self.update_projectiles(dt)
            self.update_enemies(dt)
            self.update_pickups()
            
            # Check level completion
            if self.game_map and self.game_map.is_complete():
                self.complete_level()
        
        # Update visual effects
        self.update_visual_effects(dt)
        
        # Check achievements
        self.check_achievements()
        
        # Update FPS counter
        self.fps_counter += 1
        if current_time - self.last_fps_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.last_fps_time = current_time
    
    def draw_background(self, frame):
        """Draw game background"""
        if self.state == GameState.PLAYING:
            # Doom-style gradient background
            for y in range(self.screen_height):
                intensity = 1.0 - (y / self.screen_height) * 0.8
                if y < self.screen_height // 2:
                    # Sky area
                    gray_value = int(20 + (60 * intensity))
                    color = (gray_value, gray_value, gray_value)
                else:
                    # Floor area
                    brown_intensity = intensity * 0.7
                    color = (int(42 * brown_intensity), int(42 * brown_intensity), int(165 * brown_intensity))
                
                cv2.line(frame, (0, y), (self.screen_width, y), color, 1)
            
            # Add some texture lines for atmosphere
            for i in range(0, self.screen_width, 100):
                line_intensity = 0.3 + 0.2 * math.sin(time.time() * 0.5 + i * 0.01)
                line_color = tuple(int(30 * line_intensity) for _ in range(3))
                cv2.line(frame, (i, 0), (i, self.screen_height), line_color, 1)
        
        elif self.state == GameState.MAIN_MENU:
            # Dark menu background
            frame[:] = (20, 20, 20)
        
        elif self.state == GameState.GAME_OVER:
            # Red-tinted background
            frame[:] = (30, 0, 0)
        
        else:
            # Default dark background
            frame[:] = (10, 10, 10)
    
    def draw_hud(self, frame):
        """Draw heads-up display"""
        if self.state != GameState.PLAYING:
            return
        
        # Apply screen shake
        shake_x = int(random.uniform(-self.screen_shake, self.screen_shake))
        shake_y = int(random.uniform(-self.screen_shake, self.screen_shake))
        
        # Health display
        health_ratio = self.health / self.max_health
        health_width = 200
        health_height = 25
        health_x = 20 + shake_x
        health_y = self.screen_height - 120 + shake_y
        
        # Health background
        cv2.rectangle(frame, (health_x, health_y), 
                     (health_x + health_width, health_y + health_height), 
                     DoomColors.DARK_GRAY, -1)
        
        # Health bar
        current_health_width = int(health_width * health_ratio)
        if health_ratio > 0.6:
            health_color = DoomColors.HUD_GREEN
        elif health_ratio > 0.3:
            health_color = DoomColors.HUD_YELLOW
        else:
            health_color = DoomColors.HUD_RED
        
        cv2.rectangle(frame, (health_x, health_y), 
                     (health_x + current_health_width, health_y + health_height), 
                     health_color, -1)
        
        # Health text
        cv2.putText(frame, f"HEALTH: {self.health}", 
                   (health_x, health_y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, DoomColors.WHITE, 2)
        
        # Armor display
        if self.armor > 0:
            armor_x = health_x
            armor_y = health_y - 40
            armor_ratio = self.armor / self.max_armor
            armor_width = int(health_width * armor_ratio)
            
            cv2.rectangle(frame, (armor_x, armor_y), 
                         (armor_x + health_width, armor_y + 20), 
                         DoomColors.DARK_GRAY, -1)
            cv2.rectangle(frame, (armor_x, armor_y), 
                         (armor_x + armor_width, armor_y + 20), 
                         DoomColors.PLASMA_BLUE, -1)
            
            cv2.putText(frame, f"ARMOR: {self.armor}", 
                       (armor_x, armor_y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, DoomColors.WHITE, 2)
        
        # Ammo display
        weapon = self.current_weapon
        ammo_x = 20 + shake_x
        ammo_y = health_y - 70
        
        if weapon.reloading:
            reload_progress = weapon.get_reload_progress()
            ammo_text = f"RELOADING... {int(reload_progress * 100)}%"
            color = DoomColors.HUD_YELLOW
        else:
            if weapon.max_ammo > 0:
                ammo_text = f"AMMO: {weapon.current_ammo}/{weapon.max_ammo}"
            else:
                ammo_text = "UNLIMITED"
            color = DoomColors.WHITE if weapon.current_ammo > 0 else DoomColors.HUD_RED
        
        cv2.putText(frame, ammo_text, (ammo_x, ammo_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # Weapon name
        weapon_y = ammo_y - 35
        cv2.putText(frame, weapon.name, (ammo_x, weapon_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, DoomColors.HUD_ORANGE, 2)
        
        # Score display
        score_text = f"SCORE: {self.score:08d}"
        cv2.putText(frame, score_text, (20 + shake_x, 40 + shake_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, DoomColors.HUD_YELLOW, 2)
        
        # Level and enemy count
        if self.game_map:
            level_text = f"LEVEL: {self.current_level}"
            enemy_count = self.game_map.get_enemies_alive()
            enemy_text = f"ENEMIES: {enemy_count}"
            
            cv2.putText(frame, level_text, (20 + shake_x, 80 + shake_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, DoomColors.WHITE, 2)
            cv2.putText(frame, enemy_text, (20 + shake_x, 110 + shake_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, DoomColors.WHITE, 2)
        
        # Lives display
        lives_text = f"LIVES: {self.lives}"
        cv2.putText(frame, lives_text, (self.screen_width - 150 + shake_x, 40 + shake_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, DoomColors.WHITE, 2)
        
        # Keycards
        keycard_y = 80
        for i, keycard_type in enumerate([PickupType.KEYCARD_RED, PickupType.KEYCARD_BLUE, PickupType.KEYCARD_YELLOW]):
            if keycard_type in self.keycards:
                keycard_x = self.screen_width - 50 - (i * 30) + shake_x
                keycard_colors = {
                    PickupType.KEYCARD_RED: DoomColors.BRIGHT_RED,
                    PickupType.KEYCARD_BLUE: DoomColors.PLASMA_BLUE,
                    PickupType.KEYCARD_YELLOW: DoomColors.HUD_YELLOW
                }
                cv2.rectangle(frame, (keycard_x, keycard_y + shake_y), 
                             (keycard_x + 20, keycard_y + 30 + shake_y), 
                             keycard_colors[keycard_type], -1)
        
        # FPS counter
        if self.show_fps and hasattr(self, 'current_fps'):
            fps_text = f"FPS: {self.current_fps}"
            cv2.putText(frame, fps_text, (self.screen_width - 120 + shake_x, self.screen_height - 20 + shake_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, DoomColors.WHITE, 1)
    
    def draw_crosshair(self, frame):
        """Draw targeting crosshair"""
        if self.state != GameState.PLAYING or not self.weapon_hand:
            return
        
        x, y = int(self.crosshair_pos.x), int(self.crosshair_pos.y)
        
        # Apply screen shake
        shake_x = int(random.uniform(-self.screen_shake, self.screen_shake))
        shake_y = int(random.uniform(-self.screen_shake, self.screen_shake))
        x += shake_x
        y += shake_y
        
        # Crosshair size based on weapon accuracy
        weapon = self.current_weapon
        crosshair_size = int(15 + (weapon.spread * 500))
        
        # Dynamic crosshair color
        if weapon.current_ammo == 0:
            color = DoomColors.HUD_RED
        elif weapon.reloading:
            color = DoomColors.HUD_YELLOW
        else:
            color = self.config.crosshair_color
        
        # Draw crosshair
        thickness = 3
        gap = 5
        
        # Horizontal line
        cv2.line(frame, (x - crosshair_size, y), (x - gap, y), color, thickness)
        cv2.line(frame, (x + gap, y), (x + crosshair_size, y), color, thickness)
        
        # Vertical line
        cv2.line(frame, (x, y - crosshair_size), (x, y - gap), color, thickness)
        cv2.line(frame, (x, y + gap), (x, y + crosshair_size), color, thickness)
        
        # Center dot
        cv2.circle(frame, (x, y), 2, color, -1)
        
        # Weapon-specific crosshair modifications
        if weapon.type == WeaponType.SHOTGUN:
            # Shotgun spread indicator
            spread_size = crosshair_size + 10
            cv2.circle(frame, (x, y), spread_size, color, 2)
        elif weapon.type == WeaponType.ROCKET_LAUNCHER:
            # Rocket launcher danger zone
            cv2.circle(frame, (x, y), 50, DoomColors.HUD_RED, 1)
    
    def draw_main_menu(self, frame):
        """Draw main menu"""
        # Title
        title_text = "ULTIMATE DOOM"
        title_size = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_SIMPLEX, 3, 4)[0]
        title_x = (self.screen_width - title_size[0]) // 2
        title_y = 150
        
        # Title glow effect
        glow_intensity = 0.5 + 0.5 * math.sin(time.time() * 2)
        glow_color = tuple(int(c * glow_intensity) for c in DoomColors.BRIGHT_RED)
        
        cv2.putText(frame, title_text, (title_x + 4, title_y + 4), 
                   cv2.FONT_HERSHEY_SIMPLEX, 3, glow_color, 6)
        cv2.putText(frame, title_text, (title_x, title_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 3, DoomColors.WHITE, 4)
        
        # Menu options
        menu_options = [
            "NEW GAME",
            "SETTINGS",
            "ACHIEVEMENTS", 
            "EXIT"
        ]
        
        for i, option in enumerate(menu_options):
            option_y = 300 + (i * 60)
            color = DoomColors.HUD_YELLOW if i == self.menu_selection else DoomColors.WHITE
            
            option_size = cv2.getTextSize(option, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)[0]
            option_x = (self.screen_width - option_size[0]) // 2
            
            cv2.putText(frame, option, (option_x, option_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2)
        
        # Instructions
        instructions = [
            "Use hand gestures to play:",
            "Point finger to aim and shoot",
            "Make fist for melee attacks",
            "Hands together to reload",
            "Thumbs up/down to switch weapons"
        ]
        
        for i, instruction in enumerate(instructions):
            inst_y = 550 + (i * 25)
            cv2.putText(frame, instruction, (50, inst_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, DoomColors.LIGHT_GRAY, 1)
    
    def draw_game_over(self, frame):
        """Draw game over screen"""
        # Semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.screen_width, self.screen_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Game over text
        game_over_text = "GAME OVER"
        text_size = cv2.getTextSize(game_over_text, cv2.FONT_HERSHEY_SIMPLEX, 4, 4)[0]
        text_x = (self.screen_width - text_size[0]) // 2
        text_y = self.screen_height // 2 - 100
        
        cv2.putText(frame, game_over_text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 4, DoomColors.BRIGHT_RED, 4)
        
        # Final statistics
        stats_y = text_y + 100
        stats = [
            f"FINAL SCORE: {self.score:08d}",
            f"ENEMIES KILLED: {self.stats.enemies_killed}",
            f"ACCURACY: {self.stats.get_accuracy():.1f}%",
            f"LEVELS COMPLETED: {self.stats.levels_completed}",
            f"TIME PLAYED: {int(self.stats.playtime)}s"
        ]
        
        for i, stat in enumerate(stats):
            stat_size = cv2.getTextSize(stat, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
            stat_x = (self.screen_width - stat_size[0]) // 2
            stat_y = stats_y + (i * 40)
            
            cv2.putText(frame, stat, (stat_x, stat_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, DoomColors.WHITE, 2)
        
        # Continue prompt
        continue_text = "Press any key to return to menu"
        continue_size = cv2.getTextSize(continue_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        continue_x = (self.screen_width - continue_size[0]) // 2
        continue_y = stats_y + len(stats) * 40 + 50
        
        cv2.putText(frame, continue_text, (continue_x, continue_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, DoomColors.LIGHT_GRAY, 2)
    
    def draw_level_complete(self, frame):
        """Draw level complete screen"""
        # Semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.screen_width, self.screen_height), (0, 50, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Level complete text
        complete_text = f"LEVEL {self.current_level} COMPLETE!"
        text_size = cv2.getTextSize(complete_text, cv2.FONT_HERSHEY_SIMPLEX, 2.5, 3)[0]
        text_x = (self.screen_width - text_size[0]) // 2
        text_y = self.screen_height // 2 - 50
        
        cv2.putText(frame, complete_text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2.5, DoomColors.HUD_GREEN, 3)
        
        # Level statistics
        level_time = time.time() - self.level_start_time
        stats = [
            f"TIME: {int(level_time)}s",
            f"ENEMIES KILLED: {self.stats.enemies_killed}",
            f"SCORE BONUS: {self.health * 2}",
            f"TOTAL SCORE: {self.score}"
        ]
        
        stats_y = text_y + 80
        for i, stat in enumerate(stats):
            stat_size = cv2.getTextSize(stat, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)[0]
            stat_x = (self.screen_width - stat_size[0]) // 2
            cv2.putText(frame, stat, (stat_x, stats_y + i * 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, DoomColors.WHITE, 2)
        
        # Continue prompt
        continue_text = "Press any key to continue"
        continue_size = cv2.getTextSize(continue_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        continue_x = (self.screen_width - continue_size[0]) // 2
        cv2.putText(frame, continue_text, (continue_x, stats_y + len(stats) * 50 + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, DoomColors.LIGHT_GRAY, 2)
    
    def draw_gesture_status(self, frame):
        """Draw current gesture recognition status"""
        if self.state != GameState.PLAYING:
            return
        
        # Gesture indicators
        status_x = self.screen_width - 300
        status_y = 120
        
        cv2.putText(frame, "GESTURE STATUS:", (status_x, status_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, DoomColors.WHITE, 2)
        
        # Hand detection status
        left_status = "LEFT HAND: " + ("DETECTED" if self.left_hand else "NOT FOUND")
        right_status = "RIGHT HAND: " + ("DETECTED" if self.right_hand else "NOT FOUND")
        
        left_color = DoomColors.HUD_GREEN if self.left_hand else DoomColors.HUD_RED
        right_color = DoomColors.HUD_GREEN if self.right_hand else DoomColors.HUD_RED
        
        cv2.putText(frame, left_status, (status_x, status_y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, left_color, 1)
        cv2.putText(frame, right_status, (status_x, status_y + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, right_color, 1)
        
        # Weapon status
        if self.weapon_hand:
            weapon_status = "WEAPON: READY"
            weapon_color = DoomColors.HUD_GREEN
        else:
            weapon_status = "WEAPON: POINT FINGER TO AIM"
            weapon_color = DoomColors.HUD_YELLOW
        
        cv2.putText(frame, weapon_status, (status_x, status_y + 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, weapon_color, 1)
        
        # Reload status
        if self.check_reload_gesture():
            reload_status = "RELOAD: ACTIVE"
            reload_color = DoomColors.HUD_YELLOW
        else:
            reload_status = "RELOAD: BRING HANDS TOGETHER"
            reload_color = DoomColors.WHITE
        
        cv2.putText(frame, reload_status, (status_x, status_y + 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, reload_color, 1)
    
    def draw_screen_effects(self, frame):
        """Draw screen effects like flash and shake"""
        # Screen flash effect
        if self.screen_flash > 0:
            flash_overlay = frame.copy()
            flash_intensity = int(255 * self.screen_flash)
            flash_overlay[:] = (flash_intensity, flash_intensity // 4, flash_intensity // 4)
            cv2.addWeighted(frame, 1.0 - self.screen_flash, flash_overlay, self.screen_flash, 0, frame)
    
    def draw(self, frame):
        """Main drawing function"""
        # Draw background
        self.draw_background(frame)
        
        # Draw screen effects first
        self.draw_screen_effects(frame)
        
        # Draw based on game state
        if self.state == GameState.MAIN_MENU:
            self.draw_main_menu(frame)
        
        elif self.state == GameState.PLAYING:
            # Draw game objects
            if self.game_map:
                # Draw enemies
                for enemy in self.game_map.enemies:
                    enemy.draw(frame)
                
                # Draw pickups
                for pickup in self.game_map.pickups:
                    pickup.draw(frame)
            
            # Draw projectiles
            for projectile in self.projectiles:
                projectile.draw(frame)
            
            # Draw particle effects
            for effect in self.particle_effects:
                effect.draw(frame)
            
            # Draw weapon
            self.current_weapon.draw_weapon(frame, self.screen_width, self.screen_height, int(self.recoil_offset))
            
            # Draw crosshair
            self.draw_crosshair(frame)
            
            # Draw HUD
            self.draw_hud(frame)
            
            # Draw gesture status
            self.draw_gesture_status(frame)
        
        elif self.state == GameState.GAME_OVER:
            self.draw_game_over(frame)
        
        elif self.state == GameState.LEVEL_COMPLETE:
            self.draw_level_complete(frame)
        
        # Draw back button
        hand_pos = None
        if self.left_hand:
            left_wrist = self.left_hand.landmark[0]
            hand_pos = (int(left_wrist.x * self.screen_width), int(left_wrist.y * self.screen_height))
        elif self.right_hand:
            right_wrist = self.right_hand.landmark[0]
            hand_pos = (int(right_wrist.x * self.screen_width), int(right_wrist.y * self.screen_height))
        
        self.back_button.draw(frame, hand_pos)

def main():
    """Main function to run the Ultimate Doom Game"""
    parser = argparse.ArgumentParser(description='Ultimate Hand-Gesture Doom Game')
    parser.add_argument('--camera', type=int, default=0, help='Camera index to use (default: 0)')
    parser.add_argument('--difficulty', type=str, default='medium', 
                       choices=['easy', 'medium', 'hard', 'nightmare'],
                       help='Game difficulty level')
    parser.add_argument('--no-fps', action='store_true', help='Hide FPS counter')
    args = parser.parse_args()

    # Initialize camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: Could not open camera with index {args.camera}")
        exit(-1)

    # Set camera properties for optimal performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Initialize game with configuration
    game = UltimateDoomGame(1280, 720)
    game.config.difficulty = DifficultyLevel[args.difficulty.upper()]
    game.show_fps = not args.no_fps
    
    # Create window
    cv2.namedWindow('Ultimate Doom - Hand Gesture Edition', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Ultimate Doom - Hand Gesture Edition', 1280, 720)

    print("🔥 ULTIMATE DOOM - HAND GESTURE EDITION 🔥")
    print("=" * 60)
    print("👹 RIP AND TEAR WITH YOUR HANDS! 👹")
    print()
    print("🎮 GESTURE CONTROLS:")
    print("   🎯 POINT INDEX FINGER → Aim and shoot")
    print("   👊 MAKE FIST → Melee attacks")
    print("   🙏 HANDS TOGETHER → Reload weapon")
    print("   👍 THUMBS UP → Next weapon")
    print("   👎 THUMBS DOWN → Previous weapon") 
    print("   ✌️  PEACE SIGN → Switch weapons")
    print("   ✋ OPEN PALM → Movement (future)")
    print("   🔙 BACK BUTTON → Exit to app store")
    print()
    print("🎯 WEAPONS TO UNLOCK:")
    print("   • FIST & PISTOL → Available from start")
    print("   • SHOTGUN → Kill 5 enemies")
    print("   • CHAINGUN → Kill 15 enemies") 
    print("   • ROCKET LAUNCHER → Kill 25 enemies")
    print("   • PLASMA RIFLE → Kill 50 enemies")
    print("   • BFG 9000 → Kill 100 enemies")
    print()
    print("🏆 ACHIEVEMENTS AVAILABLE:")
    print("   • First Blood, Marksman, Survivor")
    print("   • Demon Slayer, Arsenal Master, and more!")
    print("=" * 60)
    
    # Game loop
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from camera")
                break

            # Mirror frame for natural interaction
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (1280, 720))

            # Process hand tracking
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = game.hands.process(rgb_frame)

            # Reset hand tracking
            game.left_hand = None
            game.right_hand = None
            game.weapon_hand = None

            # Process detected hands
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    hand_label = handedness.classification[0].label
                    
                    if hand_label == "Left":
                        game.left_hand = hand_landmarks
                    else:
                        game.right_hand = hand_landmarks

            # Handle gesture input
            game.handle_gesture_input()

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            # Handle back button
            hand_pos = None
            if results.multi_hand_landmarks:
                wrist = results.multi_hand_landmarks[0].landmark[0]
                hand_pos = (int(wrist.x * 1280), int(wrist.y * 720))
            
            if game.back_button.handle_input(key, 
                                           results.multi_hand_landmarks[0] if results.multi_hand_landmarks else None, 
                                           hand_pos):
                print("🚪 Exiting Ultimate Doom... Thanks for playing!")
                break

            # Handle game state changes
            if game.state == GameState.MAIN_MENU:
                if key == ord(' ') or key == 13:  # Space or Enter
                    if game.menu_selection == 0:  # New Game
                        game.start_new_game()
                    elif game.menu_selection == 3:  # Exit
                        break
                elif key == ord('w') or key == ord('W'):  # Up
                    game.menu_selection = (game.menu_selection - 1) % 4
                elif key == ord('s') or key == ord('S'):  # Down
                    game.menu_selection = (game.menu_selection + 1) % 4
            
            elif game.state == GameState.LEVEL_COMPLETE:
                if key != 255:  # Any key pressed
                    game.load_level(game.current_level + 1)
                    game.state = GameState.PLAYING
            
            elif game.state == GameState.GAME_OVER:
                if key != 255:  # Any key pressed
                    game.state = GameState.MAIN_MENU

            # Handle general controls
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            elif key == ord('p'):  # Pause
                if game.state == GameState.PLAYING:
                    game.state = GameState.PAUSED
                elif game.state == GameState.PAUSED:
                    game.state = GameState.PLAYING
            elif key == ord('r') and game.state != GameState.PLAYING:  # Restart
                game.start_new_game()

            # Update game
            game.update()

            # Draw everything
            game.draw(frame)

            # Show frame
            cv2.imshow('Ultimate Doom - Hand Gesture Edition', frame)

            # Performance monitoring
            if hasattr(game, 'current_fps') and game.current_fps < 20:
                print(f"⚠️  Performance warning: FPS dropped to {game.current_fps}")

    except KeyboardInterrupt:
        print("\n🛑 Game interrupted by user")
    except Exception as e:
        print(f"💥 Game crashed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Print final statistics
        if hasattr(game, 'stats'):
            print("\n📊 FINAL GAME STATISTICS:")
            print(f"   💀 Enemies Killed: {game.stats.enemies_killed}")
            print(f"   🎯 Accuracy: {game.stats.get_accuracy():.1f}%") 
            print(f"   📈 Final Score: {game.score:08d}")
            print(f"   🏁 Levels Completed: {game.stats.levels_completed}")
            print(f"   ⏰ Time Played: {int(game.stats.playtime)}s")
        
        print("\n👹 RIP AND TEAR COMPLETE! 👹")
        print("Thanks for playing Ultimate Doom - Hand Gesture Edition!")

if __name__ == "__main__":
    main()

