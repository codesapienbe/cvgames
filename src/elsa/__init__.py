import cv2
import mediapipe as mp
import numpy as np
import random

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Frost effect parameters
frost_particles = []
MAX_PARTICLES = 100
PARTICLE_LIFESPAN = 50

def create_frost_particle(position):
    return {
        'pos': position,
        'life': PARTICLE_LIFESPAN,
        'size': random.randint(5, 15),
        'speed': (random.uniform(-2, 2), random.uniform(-5, 0))
    }

def update_frost_effects(frame):
    for particle in frost_particles[:]:
        particle['pos'] = (int(particle['pos'][0] + particle['speed'][0]),
                          int(particle['pos'][1] + particle['speed'][1]))
        particle['life'] -= 1
        
        alpha = particle['life'] / PARTICLE_LIFESPAN
        cv2.circle(frame, particle['pos'], int(particle['size'] * alpha), 
                  (255, 255, 255), -1)
        
        if particle['life'] <= 0:
            frost_particles.remove(particle)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue
        
    # Flip and convert color space
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process hands
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get index finger tip position
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            h, w, _ = frame.shape
            x, y = int(index_tip.x * w), int(index_tip.y * h)
            
            # Add frost particles when moving
            if len(frost_particles) < MAX_PARTICLES:
                frost_particles.append(create_frost_particle((x, y)))
            
            # Draw hand connections
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 110, 65), thickness=2),
                mp_drawing.DrawingSpec(color=(245, 230, 230), thickness=2))
    
    # Add magical blue tint
    frame[:,:,0] = cv2.add(frame[:,:,0], 50)  # Boost blue channel
    
    # Update and draw frost effects
    update_frost_effects(frame)
    
    # Add sparkle effect
    if random.random() < 0.3:
        h, w, _ = frame.shape
        cv2.circle(frame, (random.randint(0,w), random.randint(0,h)), 2, (255, 255, 255), -1)
    
    # Display output
    cv2.imshow('Elsa Ice Magic', frame)
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
