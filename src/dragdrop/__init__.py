import cv2
from cvzone.HandTrackingModule import HandDetector
import cvzone
import os
import pygame
import sys
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

# Set up OpenTelemetry tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)
span_processor = SimpleSpanProcessor(ConsoleSpanExporter())
trace.get_tracer_provider().add_span_processor(span_processor)

WIDTH, HEIGHT = 1280, 720

cap = cv2.VideoCapture(0)
cap.set(3, WIDTH)
cap.set(4, HEIGHT)
cap.set(5, 60)

detector = HandDetector(detectionCon=0.8)

class DragImg():
    def __init__(self, path, posOrigin, imgType):
        self.posOrigin = posOrigin
        self.imgType = imgType
        self.path = path
        if self.imgType == 'png':
            self.img = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)
        else:
            self.img = cv2.imread(self.path)
        self.size = self.img.shape[:2]
        self.selected = False
    def update(self, cursor):
        ox, oy = self.posOrigin
        h, w = self.size
        if ox < cursor[0] < ox + w and oy < cursor[1] < oy + h:
            self.posOrigin = cursor[0] - w // 2, cursor[1] - h // 2
            self.selected = True
        else:
            self.selected = False

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Drag & Drop (CV+Pygame)")
    font = pygame.font.SysFont("Arial", 32)
    clock = pygame.time.Clock()
    path = os.path.join(os.path.dirname(__file__), "Resources/ImagesPNG")
    myList = os.listdir(path)
    listImg = []
    for x, pathImg in enumerate(myList):
        if 'png' in pathImg:
            imgType = 'png'
        else:
            imgType = 'jpg'
        listImg.append(DragImg(f'{path}/{pathImg}', [50 + x * 300, 50], imgType))
    dragging = None
    with tracer.start_as_current_span("dragdrop_session"):
        running = True
        while running:
            success, img = cap.read()
            img = cv2.flip(img, 1)
            hands, img = detector.findHands(img, flipType=False)
            cursor = None
            drag_event = False
            if hands:
                lmList = hands[0]['lmList']
                length, info, img = detector.findDistance(lmList[8][:2], lmList[12][:2], img)
                if length < 60:
                    cursor = lmList[8][:2]
                    for imgObject in listImg:
                        if imgObject.selected is False and imgObject.update(cursor):
                            dragging = imgObject
                            drag_event = True
                            with tracer.start_as_current_span("drag_start"):
                                pass
                else:
                    if dragging:
                        with tracer.start_as_current_span("drag_drop"):
                            pass
                        dragging = None
            # Draw images on OpenCV frame
            try:
                for imgObject in listImg:
                    h, w = imgObject.size
                    ox, oy = imgObject.posOrigin
                    if imgObject.imgType == "png":
                        img = cvzone.overlayPNG(img, imgObject.img, [ox, oy])
                    else:
                        img[oy:oy + h, ox:ox + w] = imgObject.img
            except:
                pass
            # --- Pygame Event Handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            # --- Pygame Rendering ---
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            surf = pygame.surfarray.make_surface(img_rgb.swapaxes(0, 1))
            screen.blit(surf, (0, 0))
            instr = font.render("Pinch to drag and drop images! Press 'q' to quit.", True, (0,0,0))
            screen.blit(instr, (30, HEIGHT - 50))
            pygame.display.flip()
            # --- Keyboard quit ---
            keys = pygame.key.get_pressed()
            if keys[pygame.K_q]:
                running = False
            clock.tick(30)
    cap.release()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
