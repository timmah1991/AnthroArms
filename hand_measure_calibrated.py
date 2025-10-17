import cv2
import mediapipe as mp
import math
import numpy as np

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

# Checkerboard size and real-world spacing (1 inch = 2.54 cm)
CHECKERBOARD = (7, 7)  # number of inner corners (squares - 1)
SQUARE_SIZE_CM = 2.54  # 1 inch per square

def euclidean(a, b):
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)

def pixel_distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

cap = cv2.VideoCapture(0)
scale_cm_per_px = None
last_valid_scale = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w, _ = frame.shape

    # Try to find checkerboard corners for calibration
    found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if found:
        # Draw the detected corners
        cv2.drawChessboardCorners(frame, CHECKERBOARD, corners, found)
        # Compute average distance between adjacent corners
        distances = []
        for i in range(len(corners) - 1):
            if (i + 1) % CHECKERBOARD[0] != 0:  # skip end of row
                dist = pixel_distance(corners[i][0], corners[i+1][0])
                distances.append(dist)
        if distances:
            avg_dist_px = np.mean(distances)
            scale_cm_per_px = SQUARE_SIZE_CM / avg_dist_px
            last_valid_scale = scale_cm_per_px
    else:
        # If checkerboard not visible, reuse last valid calibration
        scale_cm_per_px = last_valid_scale

    # Process the hand
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks and scale_cm_per_px:
        hand = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        def px(a, b):
            ax, ay = int(a.x * w), int(a.y * h)
            bx, by = int(b.x * w), int(b.y * h)
            return (ax, ay), (bx, by)

        # Draw and label finger segments
        for finger in [[5,6,7,8],[9,10,11,12],[13,14,15,16],[17,18,19,20]]:
            for i in range(3):
                a = hand.landmark[finger[i]]
                b = hand.landmark[finger[i+1]]
                p1, p2 = px(a,b)
                dist_cm = euclidean(a,b) * w * scale_cm_per_px
                mid_x = int((p1[0] + p2[0]) / 2)
                mid_y = int((p1[1] + p2[1]) / 2)
                cv2.line(frame, p1, p2, (0,255,0), 2)
                cv2.putText(frame, f"{dist_cm:.1f}cm", (mid_x, mid_y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1, cv2.LINE_AA)

        # Wrist-to-finger base lines
        wrist = hand.landmark[0]
        for base in [5,9,13,17]:
            p1, p2 = px(wrist, hand.landmark[base])
            dist_cm = euclidean(wrist, hand.landmark[base]) * w * scale_cm_per_px
            mid_x = int((p1[0] + p2[0]) / 2)
            mid_y = int((p1[1] + p2[1]) / 2)
            cv2.line(frame, p1, p2, (255,0,0), 2)
            cv2.putText(frame, f"{dist_cm:.1f}cm", (mid_x, mid_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)

    if scale_cm_per_px:
        cv2.putText(frame, f"Scale: {scale_cm_per_px:.4f} cm/px",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    else:
        cv2.putText(frame, "Waiting for checkerboard calibration...",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    cv2.imshow("Hand Wireframe with Grid Calibration", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()