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
gesture_feedback = ""
feedback_timer = 0
tutorial_mode = True
tutorial_timer = 0
paused = False
prev_hands = {}

def create_snowflake(size):
    img = np.zeros((size, size, 4), dtype=np.uint8)
    center = size // 2
    for angle in range(0, 360, 60):
        x1 = int(center + (size//3) * math.cos(math.radians(angle)))
        y1 = int(center + (size//3) * math.sin(math.radians(angle)))
        cv2.line(img, (center, center), (x1, y1), (255, 255, 255, 255), 2)
        x2 = int(center + (size//4) * math.cos(math.radians(angle + 30)))
        y2 = int(center + (size//4) * math.sin(math.radians(angle + 30)))
        cv2.line(img, (center, center), (x2, y2), (200, 230, 255, 200), 1)
    return img

def add_snow_throw(pos, velocity):
    for _ in range(12):
        particles.append({
            'pos': [pos[0] + random.randint(-20, 20), pos[1] + random.randint(-20, 20)],
            'vel': [velocity[0] + random.uniform(-3, 3), velocity[1] + random.uniform(-6, -1)],
            'life': 80,
            'size': random.randint(20, 50),
            'type': 'snow',
            'rotation': random.uniform(0, 360)
        })

def add_ice_beam(start, end):
    steps = int(math.hypot(end[0] - start[0], end[1] - start[1]) / 10)
    for i in range(steps):
        t = i / max(steps, 1)
        x = int(start[0] + t * (end[0] - start[0]))
        y = int(start[1] + t * (end[1] - start[1]))
        ice_crystals.append({
            'pos': [x + random.randint(-5, 5), y + random.randint(-5, 5)],
            'life': 60,
            'size': random.randint(8, 25),
            'type': 'crystal'
        })

def add_freeze_area(pos):
    for angle in range(0, 360, 15):
        radius = random.randint(50, 150)
        x = pos[0] + radius * math.cos(math.radians(angle))
        y = pos[1] + radius * math.sin(math.radians(angle))
        frozen_objects.append({
            'pos': [x, y],
            'life': 120,
            'size': random.randint(15, 40),
            'type': 'freeze'
        })

def add_magic_trail(pos):
    magic_trails.append({
        'pos': list(pos),
        'life': 40,
        'size': random.randint(10, 30),
        'color': (random.randint(150, 255), random.randint(200, 255), 255)
    })

def add_snowman(pos):
    snowmen.append({
        'pos': list(pos),
        'life': 200,
        'growth': 0,
        'complete': False
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
        p['vel'][1] += 0.2
        p['life'] -= 1
        p['rotation'] += 5
        if p['type'] == 'snow':
            size = int(p['size'] * (p['life'] / 80))
            if size > 0:
                snowflake = create_snowflake(size)
                draw_transparent(frame, snowflake, (p['pos'][0] - size//2, p['pos'][1] - size//2))
        if p['life'] <= 0 or p['pos'][1] > h + 100:
            particles.remove(p)
    for crystal in ice_crystals[:]:
        crystal['life'] -= 1
        alpha = crystal['life'] / 60
        size = int(crystal['size'] * alpha)
        cv2.circle(frame, (int(crystal['pos'][0]), int(crystal['pos'][1])), 
                  size, (255, 255, 255), -1)
        cv2.circle(frame, (int(crystal['pos'][0]), int(crystal['pos'][1])), 
                  size + 3, (200, 230, 255), 1)
        if crystal['life'] <= 0:
            ice_crystals.remove(crystal)
    for obj in frozen_objects[:]:
        obj['life'] -= 1
        alpha = obj['life'] / 120
        size = int(obj['size'] * alpha)
        cv2.circle(frame, (int(obj['pos'][0]), int(obj['pos'][1])), 
                  size, (180, 220, 255), -1)
        if obj['life'] <= 0:
            frozen_objects.remove(obj)
    for trail in magic_trails[:]:
        trail['life'] -= 1
        alpha = trail['life'] / 40
        size = int(trail['size'] * alpha)
        cv2.circle(frame, (int(trail['pos'][0]), int(trail['pos'][1])), 
                  size, trail['color'], -1)
        if trail['life'] <= 0:
            magic_trails.remove(trail)
    for snowman in snowmen[:]:
        snowman['growth'] += 1
        snowman['life'] -= 1
        base_size = min(30, snowman['growth'])
        mid_size = min(20, max(0, snowman['growth'] - 20))
        head_size = min(15, max(0, snowman['growth'] - 40))
        cv2.circle(frame, (int(snowman['pos'][0]), int(snowman['pos'][1])), 
                  base_size, (240, 250, 255), -1)
        if mid_size > 0:
            cv2.circle(frame, (int(snowman['pos'][0]), int(snowman['pos'][1] - 25)), 
                      mid_size, (240, 250, 255), -1)
        if head_size > 0:
            cv2.circle(frame, (int(snowman['pos'][0]), int(snowman['pos'][1] - 45)), 
                      head_size, (240, 250, 255), -1)
            cv2.circle(frame, (int(snowman['pos'][0] - 5), int(snowman['pos'][1] - 50)), 
                      2, (0, 0, 0), -1)
            cv2.circle(frame, (int(snowman['pos'][0] + 5), int(snowman['pos'][1] - 50)), 
                      2, (0, 0, 0), -1)
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
    wrist = landmarks.landmark[mp_hands.HandLandmark.WRIST]
    index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    if sum(fingers_up) == 0:
        return "throw"
    elif fingers_up == [False, True, False, False, False]:
        return "beam"
    elif sum(fingers_up) == 5:
        return "freeze"
    elif fingers_up == [False, True, True, False, False]:
        return "peace"
    elif fingers_up == [True, False, False, False, True]:
        return "snowman"
    else:
        return "trail"

def draw_ui(frame, paused):
    global gesture_feedback, feedback_timer, tutorial_mode, tutorial_timer
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (10, 10), (w-10, 100), (0, 0, 0, 100), -1)
    gestures = [
        "✊ Fist = Throw Snow", "👆 Point = Ice Beam", "✋ Open = Freeze Area",
        "✌️ Peace = Magic Trail", "🤘 Rock = Build Snowman"
    ]
    if tutorial_mode:
        tutorial_timer += 1
        gesture_idx = (tutorial_timer // 120) % len(gestures)
        cv2.putText(frame, f"Try: {gestures[gesture_idx]}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        if tutorial_timer > 1200:
            tutorial_mode = False
    if gesture_feedback and feedback_timer > 0:
        cv2.putText(frame, gesture_feedback, (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
        feedback_timer -= 1
    effect_count = len(particles) + len(ice_crystals) + len(frozen_objects) + len(snowmen)
    cv2.putText(frame, f"Magic Level: {effect_count}", (w-200, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    if paused:
        cv2.rectangle(frame, (70, 150), (w-70, h-100), (0, 0, 50), -1)
        cv2.putText(frame, "Game Paused", (w//2-130, h//2-40), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255,255,255), 3)
        cv2.putText(frame, "Press 'r' to Resume", (w//2-140, h//2+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(frame, "Press 'n' to Play Again", (w//2-160, h//2+50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(frame, "Press 'q' to Quit", (w//2-110, h//2+90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    if not paused:
        success, frame = cap.read()
        if not success:
            continue
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        frame[:,:,0] = cv2.add(frame[:,:,0], 30)
        frame[:,:,1] = cv2.add(frame[:,:,1], 15)
        current_hands = []
        if results.multi_hand_landmarks:
            for idx, hand in enumerate(results.multi_hand_landmarks):
                wrist = hand.landmark[mp_hands.HandLandmark.WRIST]
                cx, cy = int(wrist.x * w), int(wrist.y * h)
                current_hands.append((cx, cy))
                gesture = detect_gesture(hand)
                if gesture == "throw":
                    velocity = (0, -5)
                    if idx in prev_hands:
                        velocity = ((cx - prev_hands[idx][0]) * 0.5, (cy - prev_hands[idx][1]) * 0.5)
                    add_snow_throw((cx, cy), velocity)
                    gesture_feedback = "❄️ Snow Throw!"
                    feedback_timer = 30
                elif gesture == "beam":
                    index_tip = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    beam_end = (int(index_tip.x * w + 100), int(index_tip.y * h))
                    add_ice_beam((cx, cy), beam_end)
                    gesture_feedback = "⚡ Ice Beam!"
                    feedback_timer = 30
                elif gesture == "freeze":
                    add_freeze_area((cx, cy))
                    gesture_feedback = "🧊 Freeze Zone!"
                    feedback_timer = 30
                elif gesture == "peace":
                    add_magic_trail((cx, cy))
                    gesture_feedback = "✨ Magic Trail!"
                    feedback_timer = 30
                elif gesture == "snowman":
                    if random.random() < 0.1:
                        add_snowman((cx, cy + 50))
                        gesture_feedback = "⛄ Building Snowman!"
                        feedback_timer = 30
                else:
                    add_magic_trail((cx, cy))
                mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(100, 200, 255), thickness=3),
                                        mp_drawing.DrawingSpec(color=(200, 230, 255), thickness=2))
        prev_hands = {idx: pos for idx, pos in enumerate(current_hands)}
        update_effects(frame)
        draw_ui(frame, paused)
        for _ in range(3):
            cv2.circle(frame, (random.randint(0, w), random.randint(0, h)), 
                      random.randint(1, 2), (255, 255, 255), -1)
    else:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        draw_ui(frame, paused)
    cv2.imshow('❄️ Elsa Magic Kingdom ❄️', frame)
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        paused = True
    elif key == ord('r'):
        paused = False
    elif key == ord('n'):
        particles.clear()
        ice_crystals.clear()
        frozen_objects.clear()
        magic_trails.clear()
        snowmen.clear()
        gesture_feedback = ""
        feedback_timer = 0
        tutorial_mode = True
        tutorial_timer = 0
        paused = False

cap.release()
cv2.destroyAllWindows()
