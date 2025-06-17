import cv2
import mediapipe as mp
import numpy as np
import random
import math
import ctypes

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.5)
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

user32 = ctypes.windll.user32
screen_width = user32.GetSystemMetrics(0)
screen_height = user32.GetSystemMetrics(1)

ice_projectiles = []
ice_spikes = []
frozen_ground = []
ice_walls = []
sparkles = []
body_shield = []
prev_hands = {}
shield_active = False

def create_ice_projectile(pos, velocity):
    ice_projectiles.append({
        'pos': list(pos),
        'vel': velocity,
        'life': 60,
        'size': random.randint(8, 15),
        'trail': []
    })

def create_ice_spike_burst(pos):
    for angle in range(0, 360, 45):
        x = pos[0] + 80 * math.cos(math.radians(angle))
        y = pos[1] + 80 * math.sin(math.radians(angle))
        ice_spikes.append({
            'start': pos,
            'end': [x, y],
            'growth': 0,
            'life': 40,
            'thickness': random.randint(3, 8)
        })

def create_foot_freeze(foot_pos):
    for i in range(25):
        angle = random.uniform(0, 2 * math.pi)
        radius = random.randint(20, 80)
        x = foot_pos[0] + radius * math.cos(angle)
        y = foot_pos[1] + radius * math.sin(angle)
        frozen_ground.append({
            'pos': [x, y],
            'size': random.randint(15, 35),
            'life': 100,
            'growth': 0
        })

def create_chest_ice_beam(chest_pos, hand_pos):
    steps = int(math.hypot(hand_pos[0] - chest_pos[0], hand_pos[1] - chest_pos[1]) / 10)
    for i in range(steps):
        t = i / max(steps, 1)
        x = int(chest_pos[0] + t * (hand_pos[0] - chest_pos[0]))
        y = int(chest_pos[1] + t * (hand_pos[1] - chest_pos[1]))
        ice_walls.append({
            'pos': [x, y],
            'height': 0,
            'life': 80,
            'max_height': random.randint(30, 60)
        })

def create_body_shield(body_points):
    global shield_active
    shield_active = True
    for point in body_points:
        for angle in range(0, 360, 20):
            radius = random.randint(40, 100)
            x = point[0] + radius * math.cos(math.radians(angle))
            y = point[1] + radius * math.sin(math.radians(angle))
            body_shield.append({
                'pos': [x, y],
                'life': 60,
                'size': random.randint(8, 20),
                'orbit_angle': angle,
                'center': point
            })

def create_shoulder_blizzard(shoulder_pos):
    for _ in range(40):
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(4, 10)
        ice_projectiles.append({
            'pos': [shoulder_pos[0], shoulder_pos[1]],
            'vel': [speed * math.cos(angle), speed * math.sin(angle)],
            'life': 70,
            'size': random.randint(6, 14),
            'trail': []
        })

def add_sparkles(pos, count=10):
    for _ in range(count):
        sparkles.append({
            'pos': [pos[0] + random.randint(-40, 40), pos[1] + random.randint(-40, 40)],
            'vel': [random.uniform(-3, 3), random.uniform(-4, -1)],
            'life': 40,
            'size': random.randint(2, 6)
        })

def update_effects(frame):
    global shield_active
    h, w = frame.shape[:2]
    
    for proj in ice_projectiles[:]:
        proj['trail'].append(list(proj['pos']))
        if len(proj['trail']) > 10:
            proj['trail'].pop(0)
        
        proj['pos'][0] += proj['vel'][0]
        proj['pos'][1] += proj['vel'][1]
        proj['vel'][1] += 0.15
        proj['life'] -= 1
        
        for i, trail_pos in enumerate(proj['trail']):
            alpha = (i + 1) / len(proj['trail'])
            size = int(proj['size'] * alpha * (proj['life'] / 60))
            cv2.circle(frame, (int(trail_pos[0]), int(trail_pos[1])), 
                      size, (255, 255, 255), -1)
            cv2.circle(frame, (int(trail_pos[0]), int(trail_pos[1])), 
                      size + 3, (255, 220, 180), 1)
        
        if proj['life'] <= 0 or proj['pos'][1] > h:
            create_ice_spike_burst(proj['pos'])
            add_sparkles(proj['pos'], 20)
            ice_projectiles.remove(proj)
    
    for spike in ice_spikes[:]:
        spike['growth'] = min(spike['growth'] + 4, 80)
        spike['life'] -= 1
        
        progress = spike['growth'] / 80
        end_x = int(spike['start'][0] + progress * (spike['end'][0] - spike['start'][0]))
        end_y = int(spike['start'][1] + progress * (spike['end'][1] - spike['start'][1]))
        
        cv2.line(frame, tuple(map(int, spike['start'])), (end_x, end_y), 
                (255, 230, 200), spike['thickness'])
        cv2.line(frame, tuple(map(int, spike['start'])), (end_x, end_y), 
                (255, 255, 255), max(1, spike['thickness'] - 2))
        
        if spike['life'] <= 0:
            ice_spikes.remove(spike)
    
    for ground in frozen_ground[:]:
        ground['growth'] = min(ground['growth'] + 3, ground['size'])
        ground['life'] -= 1
        
        cv2.circle(frame, (int(ground['pos'][0]), int(ground['pos'][1])), 
                  ground['growth'], (255, 240, 220), -1)
        cv2.circle(frame, (int(ground['pos'][0]), int(ground['pos'][1])), 
                  ground['growth'], (255, 210, 180), 2)
        
        if ground['life'] <= 0:
            frozen_ground.remove(ground)
    
    for wall in ice_walls[:]:
        wall['height'] = min(wall['height'] + 4, wall['max_height'])
        wall['life'] -= 1
        
        cv2.rectangle(frame, 
                     (int(wall['pos'][0] - 6), int(wall['pos'][1])),
                     (int(wall['pos'][0] + 6), int(wall['pos'][1] - wall['height'])),
                     (255, 230, 200), -1)
        cv2.rectangle(frame, 
                     (int(wall['pos'][0] - 6), int(wall['pos'][1])),
                     (int(wall['pos'][0] + 6), int(wall['pos'][1] - wall['height'])),
                     (255, 255, 255), 1)
        
        if wall['life'] <= 0:
            ice_walls.remove(wall)
    
    for shield in body_shield[:]:
        shield['orbit_angle'] += 5
        shield['life'] -= 1
        
        orbit_x = shield['center'][0] + 60 * math.cos(math.radians(shield['orbit_angle']))
        orbit_y = shield['center'][1] + 60 * math.sin(math.radians(shield['orbit_angle']))
        
        alpha = shield['life'] / 60
        size = int(shield['size'] * alpha)
        cv2.circle(frame, (int(orbit_x), int(orbit_y)), size, (255, 220, 180), -1)
        cv2.circle(frame, (int(orbit_x), int(orbit_y)), size + 2, (255, 255, 255), 1)
        
        if shield['life'] <= 0:
            body_shield.remove(shield)
    
    if len(body_shield) == 0:
        shield_active = False
    
    for sparkle in sparkles[:]:
        sparkle['pos'][0] += sparkle['vel'][0]
        sparkle['pos'][1] += sparkle['vel'][1]
        sparkle['life'] -= 1
        
        alpha = sparkle['life'] / 40
        size = int(sparkle['size'] * alpha)
        cv2.circle(frame, (int(sparkle['pos'][0]), int(sparkle['pos'][1])), 
                  size, (255, 255, 255), -1)
        
        if sparkle['life'] <= 0:
            sparkles.remove(sparkle)

def detect_hand_gesture(landmarks):
    tips = [mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.INDEX_FINGER_TIP,
            mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_TIP,
            mp_hands.HandLandmark.PINKY_TIP]
    fingers_up = []
    for tip in tips:
        if tip == mp_hands.HandLandmark.THUMB_TIP:
            fingers_up.append(landmarks.landmark[tip].x > landmarks.landmark[tip - 1].x)
        else:
            fingers_up.append(landmarks.landmark[tip].y < landmarks.landmark[tip - 2].y)
    
    if sum(fingers_up) == 0:
        return "ice_ball"
    elif fingers_up == [False, True, False, False, False]:
        return "ice_beam"
    elif sum(fingers_up) == 5:
        return "ground_freeze"
    elif fingers_up == [True, False, False, False, True]:
        return "blizzard"
    return None

cap = cv2.VideoCapture(0)
# set resolution to match screen and FPS to 30 for end-user
cap.set(cv2.CAP_PROP_FRAME_WIDTH, screen_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, screen_height)
cap.set(cv2.CAP_PROP_FPS, 30)
cv2.namedWindow('❄️ Elsa Full Body Magic ❄️', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('❄️ Elsa Full Body Magic ❄️', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue
    
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    pose_results = pose.process(rgb)
    hand_results = hands.process(rgb)
    
    # apply cooler tone: boost blue and green, reduce red for icy palette
    frame[:,:,0] = cv2.add(frame[:,:,0], 50)
    frame[:,:,1] = cv2.add(frame[:,:,1], 30)
    frame[:,:,2] = cv2.subtract(frame[:,:,2], 20)
    
    body_points = []
    foot_positions = []
    hand_positions = []
    
    if pose_results.pose_landmarks:
        landmarks = pose_results.pose_landmarks.landmark
        
        left_foot = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
        right_foot = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
        foot_positions = [
            (int(left_foot.x * w), int(left_foot.y * h)),
            (int(right_foot.x * w), int(right_foot.y * h))
        ]
        
        chest = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
        body_points = [
            (int(chest.x * w), int(chest.y * h)),
            (int(left_shoulder.x * w), int(left_shoulder.y * h)),
            (int(right_shoulder.x * w), int(right_shoulder.y * h))
        ]
    
    current_hands = []
    hands_above_head = 0
    
    if hand_results.multi_hand_landmarks:
        for idx, hand in enumerate(hand_results.multi_hand_landmarks):
            wrist = hand.landmark[mp_hands.HandLandmark.WRIST]
            cx, cy = int(wrist.x * w), int(wrist.y * h)
            current_hands.append((cx, cy))
            hand_positions.append((cx, cy))
            
            if cy < h * 0.2:
                hands_above_head += 1
            
            gesture = detect_hand_gesture(hand)
            
            if gesture == "ice_ball":
                velocity = [0, -8]
                if idx in prev_hands:
                    velocity = [(cx - prev_hands[idx][0]) * 0.8, (cy - prev_hands[idx][1]) * 0.8]
                create_ice_projectile((cx, cy), velocity)
                add_sparkles((cx, cy), 8)
            
            elif gesture == "ice_beam" and body_points:
                create_chest_ice_beam(body_points[0], (cx, cy))
            
            elif gesture == "ground_freeze" and foot_positions:
                for foot_pos in foot_positions:
                    create_foot_freeze(foot_pos)
                    add_sparkles(foot_pos, 15)
            
            elif gesture == "blizzard" and body_points:
                for shoulder in body_points[1:]:
                    create_shoulder_blizzard(shoulder)
    
    if hands_above_head >= 2 and body_points and not shield_active:
        create_body_shield(body_points)
        add_sparkles(body_points[0], 25)
    
    prev_hands = {idx: pos for idx, pos in enumerate(current_hands)}
    
    update_effects(frame)
    
    for _ in range(12):
        cv2.circle(frame, (random.randint(0, w), random.randint(0, h)), 
                  random.randint(1, 3), (255, 255, 255), -1)
    
    cv2.imshow('❄️ Elsa Full Body Magic ❄️', frame)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
