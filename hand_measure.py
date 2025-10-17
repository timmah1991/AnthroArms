import cv2
import mediapipe as mp
import math

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

# Set known calibration distance in cm (e.g., distance between thumb tip and index tip)
KNOWN_DISTANCE_CM = 8.0

def euclidean(a, b):
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)

cap = cv2.VideoCapture(0)
scale = None

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        # Calibration reference: thumb tip (4) to index tip (8)
        thumb_tip = hand.landmark[4]
        index_tip = hand.landmark[8]
        pixel_dist = euclidean(thumb_tip, index_tip) * w
        if scale is None:
            scale = KNOWN_DISTANCE_CM / pixel_dist  # cm per pixel

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
                dist_cm = euclidean(a,b) * w * scale
                mid_x = int((p1[0] + p2[0]) / 2)
                mid_y = int((p1[1] + p2[1]) / 2)
                cv2.line(frame, p1, p2, (0,255,0), 2)
                cv2.putText(frame, f"{dist_cm:.1f}cm", (mid_x, mid_y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1, cv2.LINE_AA)

        # Draw wrist-to-finger base measurements
        wrist = hand.landmark[0]
        for base in [5,9,13,17]:
            p1, p2 = px(wrist, hand.landmark[base])
            dist_cm = euclidean(wrist, hand.landmark[base]) * w * scale
            mid_x = int((p1[0] + p2[0]) / 2)
            mid_y = int((p1[1] + p2[1]) / 2)
            cv2.line(frame, p1, p2, (255,0,0), 2)
            cv2.putText(frame, f"{dist_cm:.1f}cm", (mid_x, mid_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)

        cv2.putText(frame, f"Scale: {scale:.4f} cm/pixel",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

    cv2.imshow("Hand Wireframe Measurements", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()