import cv2
import mediapipe as mp
import time
import random
import numpy as np

# Initialize MediaPipe Face Mesh for blink and nose tracking
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit(1)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Game entities
spaceship_y = height - 50
spaceship_x = width // 2
ship_color = (255, 255, 0)
ship_radius = 20

aliens = []  # list of dicts: {x, y}
alien_radius = 15
alien_speed = 200  # px/s
spawn_interval = 1.0  # s
last_spawn = time.time()

bullet = {'x': 0, 'y': 0, 'active': False}
bullet_speed = 600  # px/s
bullet_color = (0, 255, 255)
bullet_radius = 5

score = 0

# Blink detection state
BLINK_THRESHOLD = 0.02
prev_blink = False
blink_cooldown = False

prev_time = time.time()

# Main loop runs on import
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    now = time.time()
    dt = now - prev_time
    prev_time = now

    # Face mesh processing
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    blink = False
    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark
        # Nose tip (landmark 1)
        nose = lm[1]
        spaceship_x = int(nose.x * width)
        # Blink: left eye landmarks 159, 145
        u = lm[159]
        l = lm[145]
        if abs(u.y - l.y) < BLINK_THRESHOLD:
            blink = True

    # Spawn aliens
    if now - last_spawn > spawn_interval:
        aliens.append({'x': random.randint(alien_radius, width - alien_radius), 'y': -alien_radius})
        last_spawn = now

    # Update aliens
    for a in aliens[:]:
        a['y'] += alien_speed * dt
        # Remove if passed bottom
        if a['y'] > height + alien_radius:
            aliens.remove(a)

    # Handle blinking to fire bullet
    if blink and not prev_blink and not blink_cooldown:
        if not bullet['active']:
            bullet['x'] = spaceship_x
            bullet['y'] = spaceship_y - ship_radius
            bullet['active'] = True
        blink_cooldown = True
    if not blink:
        blink_cooldown = False
    prev_blink = blink

    # Update bullet
    if bullet['active']:
        bullet['y'] -= bullet_speed * dt
        if bullet['y'] < -bullet_radius:
            bullet['active'] = False

    # Check collisions
    if bullet['active']:
        for a in aliens[:]:
            if np.hypot(bullet['x'] - a['x'], bullet['y'] - a['y']) < (bullet_radius + alien_radius):
                aliens.remove(a)
                bullet['active'] = False
                score += 1
                break

    # Draw spaceship
    cv2.circle(frame, (spaceship_x, spaceship_y), ship_radius, ship_color, -1)
    # Draw aliens
    for a in aliens:
        cv2.circle(frame, (int(a['x']), int(a['y'])), alien_radius, (0,0,255), -1)
    # Draw bullet
    if bullet['active']:
        cv2.circle(frame, (int(bullet['x']), int(bullet['y'])), bullet_radius, bullet_color, -1)

    # UI overlays
    cv2.putText(frame, f"Score: {score}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(frame, "Blink to shoot, move head to aim", (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

    cv2.imshow("Eye Shooter", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
