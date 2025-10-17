from flask import Flask, Response, render_template_string
import cv2
import mediapipe as mp
import math
import numpy as np

app = Flask(__name__)

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Checkerboard setup
CHECKERBOARD = (7, 7)  # inner corners
SQUARE_SIZE_CM = 2.54  # 1 inch

cap = cv2.VideoCapture(0)
scale_cm_per_px = None
last_valid_scale = None
frame_count = 0

def euclidean(a, b):
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)

def pixel_distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def generate_frames():
    global scale_cm_per_px, last_valid_scale, frame_count
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_count += 1
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Checkerboard detection every 10 frames on downscaled image
        if frame_count % 10 == 0:
            gray_small = cv2.resize(gray, (0,0), fx=0.5, fy=0.5)
            found, corners = cv2.findChessboardCorners(gray_small, CHECKERBOARD, None)
            if found:
                corners = corners * 2.0
                cv2.drawChessboardCorners(frame, CHECKERBOARD, corners, found)
                distances = []
                for i in range(len(corners)-1):
                    if (i+1) % CHECKERBOARD[0] != 0:
                        dist = pixel_distance(corners[i][0], corners[i+1][0])
                        distances.append(dist)
                if distances:
                    avg_dist_px = np.mean(distances)
                    scale_cm_per_px = SQUARE_SIZE_CM / avg_dist_px
                    last_valid_scale = scale_cm_per_px
            else:
                scale_cm_per_px = last_valid_scale
        else:
            scale_cm_per_px = last_valid_scale

        # Hand tracking
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks and scale_cm_per_px:
            hand = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            def px(a, b):
                ax, ay = int(a.x * w), int(a.y * h)
                bx, by = int(b.x * w), int(b.y * h)
                return (ax, ay), (bx, by)

            # Draw finger segments
            for finger in [[5,6,7,8],[9,10,11,12],[13,14,15,16],[17,18,19,20]]:
                for i in range(3):
                    a = hand.landmark[finger[i]]
                    b = hand.landmark[finger[i+1]]
                    p1, p2 = px(a,b)
                    dist_cm = euclidean(a,b) * w * scale_cm_per_px
                    mid_x = int((p1[0]+p2[0])/2)
                    mid_y = int((p1[1]+p2[1])/2)
                    cv2.line(frame, p1, p2, (0,255,0), 2)
                    cv2.putText(frame, f"{dist_cm:.1f}cm", (mid_x, mid_y-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1, cv2.LINE_AA)

            # Wrist-to-finger base lines
            wrist = hand.landmark[0]
            for base in [5,9,13,17]:
                p1, p2 = px(wrist, hand.landmark[base])
                dist_cm = euclidean(wrist, hand.landmark[base]) * w * scale_cm_per_px
                mid_x = int((p1[0]+p2[0])/2)
                mid_y = int((p1[1]+p2[1])/2)
                cv2.line(frame, p1, p2, (255,0,0), 2)
                cv2.putText(frame, f"{dist_cm:.1f}cm", (mid_x, mid_y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)

        if scale_cm_per_px:
            cv2.putText(frame, f"Scale: {scale_cm_per_px:.4f} cm/px",
                        (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        else:
            cv2.putText(frame, "Waiting for checkerboard calibration...",
                        (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Flask routes
@app.route('/')
def index():
    # Simple HTML template
    return render_template_string('''
        <html>
            <head><title>Hand Wireframe</title></head>
            <body>
                <h2>Hand Wireframe with Grid Calibration</h2>
                <img src="{{ url_for('video_feed') }}" width="640" height="480">
            </body>
        </html>
    ''')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)