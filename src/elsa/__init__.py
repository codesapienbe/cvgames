import cv2
import mediapipe as mp
import numpy as np
import random
import math

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

particles = []
ice_crystals = []
frozen_objects = []
magic_trails = []
snowmen = []
prev_hands = {}

def create_snowflake(size):
    img = np.zeros((size, size, 4), dtype=np.uint8)
    center = size // 2
    for angle in range(0, 360, 60):
        x1 = int(center + (size//3) * math.cos(math.radians(angle)))
        y1 = int(center + (size//3) * math.sin(math.radians(angle)))
        cv2.line(img, (center, center), (x1, y1), (255, 255, 255, 255), 1)
        x2 = int(center + (size//4) * math.cos(math.radians(angle + 30)))
        y2 = int(center + (size//4) * math.sin(math.radians(angle + 30)))
        cv2.line(img, (center, center), (x2, y2), (200, 230, 255, 200), 1)
    return img

def add_snow_throw(pos, velocity):
    for _ in range(6):
        particles.append({
            'pos': [pos[0] + random.randint(-10, 10), pos[1] + random.randint(-10, 10)],
            'vel': [velocity[0] + random.uniform(-2, 2), velocity[1] + random.uniform(-4, -1)],
            'life': 40,
            'size': random.randint(10, 16),
            'type': 'snow'
        })

def add_ice_beam(start, end):
    steps = int(math.hypot(end[0] - start[0], end[1] - start[1]) / 10)
    for i in range(steps):
        t = i / max(steps, 1)
        x = int(start[0] + t * (end[0] - start[0]))
        y = int(start[1] + t * (end[1] - start[1]))
        ice_crystals.append({
            'pos': [x + random.randint(-2, 2), y + random.randint(-2, 2)],
            'life': 20,
            'size': random.randint(4, 8),
            'type': 'crystal'
        })

def add_freeze_area(pos):
    for angle in range(0, 360, 30):
        radius = random.randint(15, 30)
        x = pos[0] + radius * math.cos(math.radians(angle))
        y = pos[1] + radius * math.sin(math.radians(angle))
        frozen_objects.append({
            'pos': [x, y],
            'life': 25,
            'size': random.randint(5, 10),
            'type': 'freeze'
        })

def add_magic_trail(pos):
    magic_trails.append({
        'pos': list(pos),
        'life': 12,
        'size': random.randint(4, 8),
        'color': (random.randint(150, 255), random.randint(200, 255), 255)
    })

def add_snowman(pos):
    snowmen.append({
        'pos': list(pos),
        'life': 30,
        'growth': 0
    })

def draw_transparent(target, img, position):
    x, y = int(position[0]), int(position[1])
    h, w = img.shape[:2]
    if x < 0 or y < 0 or x + w >= target.shape[1] or y + h >= target.shape[0]:
        return
    alpha = img[:, :, 3] / 255.0
    for c in range(3):
        target[y:y+h, x:x+w, c] = (1.0 - alpha) * target[y:y+h, x:x+w, c] + alpha * img[:, :, c]

def update_effects(frame):
    h, w = frame.shape[:2]
    for p in particles[:]:
        p['pos'][0] += p['vel'][0]
        p['pos'][1] += p['vel'][1]
        p['vel'][1] += 0.18
        p['life'] -= 1
        if p['type'] == 'snow':
            size = int(p['size'] * (p['life'] / 40))
            if size > 0:
                snowflake = create_snowflake(size)
                draw_transparent(frame, snowflake, (p['pos'][0] - size//2, p['pos'][1] - size//2))
        if p['life'] <= 0 or p['pos'][1] > h + 30:
            particles.remove(p)
    for crystal in ice_crystals[:]:
        crystal['life'] -= 1
        alpha = crystal['life'] / 20
        size = int(crystal['size'] * alpha)
        cv2.circle(frame, (int(crystal['pos'][0]), int(crystal['pos'][1])), 
                  size, (255, 255, 255), -1)
        cv2.circle(frame, (int(crystal['pos'][0]), int(crystal['pos'][1])), 
                  size + 2, (200, 230, 255), 1)
        if crystal['life'] <= 0:
            ice_crystals.remove(crystal)
    for obj in frozen_objects[:]:
        obj['life'] -= 1
        alpha = obj['life'] / 25
        size = int(obj['size'] * alpha)
        cv2.circle(frame, (int(obj['pos'][0]), int(obj['pos'][1])), 
                  size, (180, 220, 255), -1)
        if obj['life'] <= 0:
            frozen_objects.remove(obj)
    for trail in magic_trails[:]:
        trail['life'] -= 1
        alpha = trail['life'] / 12
        size = int(trail['size'] * alpha)
        cv2.circle(frame, (int(trail['pos'][0]), int(trail['pos'][1])), 
                  size, trail['color'], -1)
        if trail['life'] <= 0:
            magic_trails.remove(trail)
    for snowman in snowmen[:]:
        snowman['growth'] += 1
        snowman['life'] -= 1
        base_size = min(7, snowman['growth'])
        mid_size = min(5, max(0, snowman['growth'] - 5))
        head_size = min(3, max(0, snowman['growth'] - 10))
        cv2.circle(frame, (int(snowman['pos'][0]), int(snowman['pos'][1])), 
                  base_size, (240, 250, 255), -1)
        if mid_size > 0:
            cv2.circle(frame, (int(snowman['pos'][0]), int(snowman['pos'][1] - 8)), 
                      mid_size, (240, 250, 255), -1)
        if head_size > 0:
            cv2.circle(frame, (int(snowman['pos'][0]), int(snowman['pos'][1] - 14)), 
                      head_size, (240, 250, 255), -1)
            cv2.circle(frame, (int(snowman['pos'][0] - 1), int(snowman['pos'][1] - 15)), 
                      1, (0, 0, 0), -1)
            cv2.circle(frame, (int(snowman['pos'][0] + 1), int(snowman['pos'][1] - 15)), 
                      1, (0, 0, 0), -1)
        if snowman['life'] <= 0:
            snowmen.remove(snowman)

def detect_gesture(landmarks):
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
        return "throw"
    elif fingers_up == [False, True, False, False, False]:
        return "beam"
    elif sum(fingers_up) == 5:
        return "freeze"
    elif fingers_up == [False, True, True, False, False]:
        return "trail"
    elif fingers_up == [True, False, False, False, True]:
        return "snowman"
    else:
        return None

cap = cv2.VideoCapture(0)
cv2.namedWindow('❄️ Elsa Magic Kingdom ❄️', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('❄️ Elsa Magic Kingdom ❄️', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    frame[:,:,0] = cv2.add(frame[:,:,0], 20)
    frame[:,:,1] = cv2.add(frame[:,:,1], 10)
    current_hands = []
    if results.multi_hand_landmarks:
        for idx, hand in enumerate(results.multi_hand_landmarks):
            wrist = hand.landmark[mp_hands.HandLandmark.WRIST]
            cx, cy = int(wrist.x * w), int(wrist.y * h)
            current_hands.append((cx, cy))
            gesture = detect_gesture(hand)
            if gesture == "throw":
                velocity = (0, -2)
                if idx in prev_hands:
                    velocity = ((cx - prev_hands[idx][0]) * 0.5, (cy - prev_hands[idx][1]) * 0.5)
                add_snow_throw((cx, cy), velocity)
            elif gesture == "beam":
                index_tip = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                beam_end = (int(index_tip.x * w + 30), int(index_tip.y * h))
                add_ice_beam((cx, cy), beam_end)
            elif gesture == "freeze":
                add_freeze_area((cx, cy))
            elif gesture == "trail":
                add_magic_trail((cx, cy))
            elif gesture == "snowman":
                if random.random() < 0.1:
                    add_snowman((cx, cy + 10))
            mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(100, 200, 255), thickness=2),
                                    mp_drawing.DrawingSpec(color=(200, 230, 255), thickness=1))
    prev_hands = {idx: pos for idx, pos in enumerate(current_hands)}
    update_effects(frame)
    for _ in range(2):
        cv2.circle(frame, (random.randint(0, w), random.randint(0, h)), 
                  random.randint(1, 2), (255, 255, 255), -1)
    cv2.imshow('❄️ Elsa Magic Kingdom ❄️', frame)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
