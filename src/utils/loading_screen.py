import cv2
import numpy as np
import time

def show_loading_screen(width, height, duration=3):
    loading_screen = np.zeros((height, width, 3), dtype=np.uint8)
    bar_width = int(width * 0.8)
    bar_height = 30
    bar_x = (width - bar_width) // 2
    bar_y = height // 2
    start_time = time.time()
    while time.time() - start_time < duration:
        progress = (time.time() - start_time) / duration
        loading_screen.fill(0)
        cv2.putText(loading_screen, "Loading...", (width//2 - 100, bar_y - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.rectangle(loading_screen, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
        progress_width = int(bar_width * progress)
        cv2.rectangle(loading_screen, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), (0, 255, 0), -1)
        percentage = int(progress * 100)
        cv2.putText(loading_screen, f"{percentage}%", (bar_x + bar_width//2 - 20, bar_y + bar_height//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("Loading Screen", loading_screen)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    show_loading_screen(1280, 720)