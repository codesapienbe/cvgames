import cv2
import mediapipe as mp
import numpy as np
import time

class EyeTicTacToe:
    def __init__(self):
        self.board = [['' for _ in range(3)] for _ in range(3)]
        self.current_player = 'X'
        self.game_over = False
        self.winner = None
        self.selection_time = 0
        self.selection_threshold = 2.0  # seconds to hold gaze
        self.last_selection = None
        
    def check_winner(self):
        # Check rows
        for row in self.board:
            if row[0] == row[1] == row[2] != '':
                return row[0]
        
        # Check columns
        for col in range(3):
            if self.board[0][col] == self.board[1][col] == self.board[2][col] != '':
                return self.board[0][col]
        
        # Check diagonals
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != '':
            return self.board[0][0]
        if self.board[0][2] == self.board[1][1] == self.board[2][0] != '':
            return self.board[0][2]
        
        # Check for draw
        if all(self.board[i][j] != '' for i in range(3) for j in range(3)):
            return 'Draw'
        
        return None
    
    def make_move(self, row, col):
        if self.board[row][col] == '' and not self.game_over:
            self.board[row][col] = self.current_player
            self.current_player = 'O' if self.current_player == 'X' else 'X'
            
            # Check for winner
            winner = self.check_winner()
            if winner:
                self.game_over = True
                self.winner = winner
    
    def reset_game(self):
        self.board = [['' for _ in range(3)] for _ in range(3)]
        self.current_player = 'X'
        self.game_over = False
        self.winner = None
        self.selection_time = 0
        self.last_selection = None

def detect_eye_gaze(face_landmarks, frame_width, frame_height):
    """Detect eye gaze direction and convert to board position"""
    if not face_landmarks:
        return None
    
    # Get eye landmarks
    left_eye = face_landmarks.landmark[159]  # Left eye center
    right_eye = face_landmarks.landmark[386]  # Right eye center
    
    # Calculate average eye position
    eye_x = (left_eye.x + right_eye.x) / 2
    eye_y = (left_eye.y + right_eye.y) / 2
    
    # Convert to pixel coordinates
    pixel_x = int(eye_x * frame_width)
    pixel_y = int(eye_y * frame_height)
    
    # Define board area (center of screen)
    board_width = frame_width * 0.6
    board_height = frame_height * 0.6
    board_x = (frame_width - board_width) / 2
    board_y = (frame_height - board_height) / 2
    
    # Check if gaze is within board area
    if (board_x <= pixel_x <= board_x + board_width and 
        board_y <= pixel_y <= board_y + board_height):
        
        # Convert to board coordinates
        cell_width = board_width / 3
        cell_height = board_height / 3
        
        col = int((pixel_x - board_x) / cell_width)
        row = int((pixel_y - board_y) / cell_height)
        
        # Ensure coordinates are within bounds
        col = max(0, min(2, col))
        row = max(0, min(2, row))
        
        return (row, col)
    
    return None

def main():
    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    mp_draw = mp.solutions.drawing_utils
    
    # Initialize game
    game = EyeTicTacToe()
    
    # Setup camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Create fullscreen window
    cv2.namedWindow("Eye Tic Tac Toe", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Eye Tic Tac Toe", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    # Get frame dimensions
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        
        current_time = time.time()
        
        # Detect eye gaze
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            
            # Draw face mesh
            mp_draw.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)
            
            # Get gaze position
            gaze_pos = detect_eye_gaze(face_landmarks, frame_width, frame_height)
            
            if gaze_pos:
                row, col = gaze_pos
                
                # Check if this is a new selection
                if gaze_pos != game.last_selection:
                    game.selection_time = current_time
                    game.last_selection = gaze_pos
                
                # Check if selection has been held long enough
                if current_time - game.selection_time >= game.selection_threshold:
                    if game.board[row][col] == '':  # Cell is empty
                        game.make_move(row, col)
                        game.selection_time = current_time  # Reset timer
                
                # Draw selection indicator
                cell_width = frame_width * 0.6 / 3
                cell_height = frame_height * 0.6 / 3
                board_x = (frame_width - frame_width * 0.6) / 2
                board_y = (frame_height - frame_height * 0.6) / 2
                
                cell_x = int(board_x + col * cell_width)
                cell_y = int(board_y + row * cell_height)
                
                # Draw selection rectangle
                progress = min(1.0, (current_time - game.selection_time) / game.selection_threshold)
                color = (0, int(255 * progress), int(255 * (1 - progress)))
                cv2.rectangle(frame, (cell_x, cell_y), 
                             (int(cell_x + cell_width), int(cell_y + cell_height)), 
                             color, 3)
        
        # Draw game board
        board_width = frame_width * 0.6
        board_height = frame_height * 0.6
        board_x = int((frame_width - board_width) / 2)
        board_y = int((frame_height - board_height) / 2)
        
        # Draw board background
        cv2.rectangle(frame, (board_x, board_y), 
                     (int(board_x + board_width), int(board_y + board_height)), 
                     (255, 255, 255), -1)
        cv2.rectangle(frame, (board_x, board_y), 
                     (int(board_x + board_width), int(board_y + board_height)), 
                     (0, 0, 0), 3)
        
        # Draw grid lines
        cell_width = board_width / 3
        cell_height = board_height / 3
        
        for i in range(1, 3):
            # Vertical lines
            x = int(board_x + i * cell_width)
            cv2.line(frame, (x, board_y), (x, int(board_y + board_height)), (0, 0, 0), 2)
            # Horizontal lines
            y = int(board_y + i * cell_height)
            cv2.line(frame, (board_x, y), (int(board_x + board_width), y), (0, 0, 0), 2)
        
        # Draw X's and O's
        for row in range(3):
            for col in range(3):
                if game.board[row][col]:
                    cell_x = int(board_x + col * cell_width + cell_width / 2)
                    cell_y = int(board_y + row * cell_height + cell_height / 2)
                    
                    if game.board[row][col] == 'X':
                        # Draw X
                        size = int(min(cell_width, cell_height) * 0.3)
                        cv2.line(frame, (cell_x - size, cell_y - size), 
                                (cell_x + size, cell_y + size), (255, 0, 0), 5)
                        cv2.line(frame, (cell_x - size, cell_y + size), 
                                (cell_x + size, cell_y - size), (255, 0, 0), 5)
                    else:
                        # Draw O
                        radius = int(min(cell_width, cell_height) * 0.3)
                        cv2.circle(frame, (cell_x, cell_y), radius, (0, 0, 255), 5)
        
        # Draw game status
        status_y = 50
        cv2.putText(frame, f"Current Player: {game.current_player}", (50, status_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        if game.game_over:
            if game.winner == 'Draw':
                cv2.putText(frame, "It's a Draw!", (frame_width//2 - 100, status_y + 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            else:
                cv2.putText(frame, f"{game.winner} Wins!", (frame_width//2 - 100, status_y + 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            
            cv2.putText(frame, "Press 'r' to restart or 'q' to quit", 
                       (frame_width//2 - 200, status_y + 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        else:
            cv2.putText(frame, "Look at a cell for 2 seconds to place your mark", 
                       (50, frame_height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        cv2.imshow("Eye Tic Tac Toe", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            game.reset_game()
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 