from flask import Flask, Response, render_template_string
import cv2
import mediapipe as mp
import math
import numpy as np
import time

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

# Measurement thresholds
MEASUREMENT_CHART = {
    "height": (3, 5),
    "width":  (6, 8),
    "depth":  (22, 28)
}

def euclidean(a, b):
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)

def pixel_distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def get_size_label(value, key):
    low, high = MEASUREMENT_CHART[key]
    if value <= low:
        return 'S'
    elif value <= high:
        return 'M'
    else:
        return 'L'

def generate_frames():
    global scale_cm_per_px, last_valid_scale, frame_count
    grip_recommendation = "N/A"
    last_grip = None
    last_change_time = time.time()
    freeze_frame = None
    popup_duration = 5  # seconds

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_count += 1
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- Checkerboard detection every 10 frames ---
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

        measured = {}
        if results.multi_hand_landmarks and scale_cm_per_px:
            hand = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            # --- HEIGHT: average spacing between index → middle → ring → pinky fingertips ---
            fingertips = [hand.landmark[8], hand.landmark[12], hand.landmark[16], hand.landmark[20]]
            distances = []
            for i in range(len(fingertips)-1):
                dist_cm = euclidean(fingertips[i], fingertips[i+1]) * w * scale_cm_per_px
                distances.append(dist_cm)
            if distances:
                measured['height'] = np.mean(distances)

            # --- WIDTH: thumb length ---
            thumb_tip = hand.landmark[4]
            thumb_base = hand.landmark[2]
            measured['width'] = euclidean(thumb_tip, thumb_base) * w * scale_cm_per_px

            # --- DEPTH: index finger length from knuckle to tip ---
            index_knuckle = hand.landmark[5]
            index_tip = hand.landmark[8]
            measured['depth'] = euclidean(index_knuckle, index_tip) * w * scale_cm_per_px

            # Draw measurement chart on frame
            cv2.rectangle(frame, (10,10), (200,130), (50,50,50), -1)
            y = 30
            for k in ['height','width','depth']:
                if k in measured:
                    cv2.putText(frame, f"{k.capitalize()}: {measured[k]:.2f} cm",
                                (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                    y += 30

            # Compute grip recommendation
            if len(measured) == 3:
                labels = [get_size_label(measured[k], k) for k in ['height','width','depth']]
                grip_recommendation = '-'.join(labels)
                cv2.putText(frame, f"Grip: {grip_recommendation}",
                            (10, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                print(f"Grip recommendation: {grip_recommendation}")

                # --- Stability detection for freeze ---
                if grip_recommendation == last_grip:
                    if time.time() - last_change_time > popup_duration:
                        if freeze_frame is None:
                            freeze_frame = frame.copy()
                else:
                    last_change_time = time.time()
                    last_grip = grip_recommendation
                    freeze_frame = None  # reset

        # Draw frame or frozen frame
        output_frame = freeze_frame if freeze_frame is not None else frame

        # If frozen, overlay large pop-up
        if freeze_frame is not None:
            overlay = output_frame.copy()
            cv2.rectangle(overlay, (50, h//2 - 100), (w-50, h//2 + 100), (0,0,0), -1)
            cv2.putText(overlay, f"Your recommended grip:", (100, h//2 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 4)
            cv2.putText(overlay, f'"{grip_recommendation}"', (100, h//2 + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,255), 6)
            alpha = 0.8
            cv2.addWeighted(overlay, alpha, output_frame, 1 - alpha, 0, output_frame)

        if scale_cm_per_px:
            cv2.putText(output_frame, f"Scale: {scale_cm_per_px:.4f} cm/px",
                        (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        else:
            cv2.putText(output_frame, "Waiting for checkerboard calibration...",
                        (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        ret, buffer = cv2.imencode('.jpg', output_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template_string('''
        <html>
            <head>
                <title>Grip Measurement</title>
                <style>
                    body { 
                        background: linear-gradient(to bottom, #1e3c72, #2a5298); 
                        color: #fff; 
                        font-family: Arial, sans-serif;
                        text-align: center;
                    }
                    h2 { margin-top: 20px; }
                    .container { display: flex; justify-content: center; align-items: flex-start; gap: 20px; margin-top: 20px; }
                    .video-container img, .diagram-container img {
                        border: 5px solid #fff; 
                        border-radius: 10px;
                    }
                    .diagram-container { width: 300px; }
                </style>
            </head>
            <body>
                <h2>Hand Wireframe with Grip Recommendation</h2>
                <div class="container">
                    <div class="diagram-container">
                        <img src="{{ url_for('static', filename='hand_diagram.png') }}" width="300">
                        <p>Place fingers together, square to grid, thumb tight against index finger.</p>
                    </div>
                    <div class="video-container">
                        <img src="{{ url_for('video_feed') }}" width="640" height="480">
                    </div>
                </div>
                <p>Grip recommendation updates live above</p>
            </body>
        </html>
    ''')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
