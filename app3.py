import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, Response, render_template_string, jsonify
import threading
import math
import collections
import os

# ---------------- CONFIG ----------------
SMOOTHING_FRAMES = 12

# Grip thresholds in inches (calibrated after checkerboard)
MEASUREMENT_CHART = {
    "height": (2.0, 3.0),  # small <=2", medium <=3", else large
    "width":  (1.5, 2.5),
    "depth":  (2.0, 3.5),
}

# Path to static hand guide image
STATIC_IMAGE_PATH = "static/hand_guide.png"

# ----------------------------------------

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Moving average helper
class MovingAverage:
    def __init__(self, maxlen):
        self.buf = collections.deque(maxlen=maxlen)
    def add(self, val):
        self.buf.append(val)
    def avg(self):
        return sum(self.buf)/len(self.buf) if self.buf else 0.0

# Global moving averages
mv_height = MovingAverage(SMOOTHING_FRAMES)
mv_width = MovingAverage(SMOOTHING_FRAMES)
mv_depth = MovingAverage(SMOOTHING_FRAMES)

# Camera and frame
cap = cv2.VideoCapture(0)
frame_lock = threading.Lock()
current_frame = None

# Global output
current_recommendation = "--"

# ---------------- Calibration Logic ----------------
# Use checkerboard to determine pixel -> inch scale
# Example: measure distance between 1" squares along x and y
checkerboard_pixels_per_inch = 1.0  # default, replace with actual calibration

def calibrate_checkerboard(px_distance, inch_distance=1.0):
    """Returns scaling factor: inches per pixel"""
    global checkerboard_pixels_per_inch
    checkerboard_pixels_per_inch = inch_distance / px_distance

def px_to_inches(px):
    return px * checkerboard_pixels_per_inch

# ---------------- Measurement Logic ----------------
def dist_xy(a,b,img_w,img_h):
    ax, ay = a.x*img_w, a.y*img_h
    bx, by = b.x*img_w, b.y*img_h
    return math.hypot(ax-bx, ay-by)

def compute_hand_measurements(landmarks, img_w, img_h):
    # Height: MCP joints center-to-center
    mcp_indices = [5,9,13,17]
    pair_dists = [dist_xy(landmarks[mcp_indices[i]], landmarks[mcp_indices[i+1]], img_w,img_h)
                  for i in range(len(mcp_indices)-1)]
    height_px = sum(pair_dists)/len(pair_dists)

    # Width: thumb length (CMC -> tip)
    width_px = dist_xy(landmarks[1], landmarks[4], img_w,img_h)

    # Depth: midpoint between thumb MCP & index MCP -> index fingertip
    thumb_mcp = landmarks[2]
    index_mcp = landmarks[5]
    mid_x = (thumb_mcp.x + index_mcp.x)/2
    mid_y = (thumb_mcp.y + index_mcp.y)/2
    class _P: pass
    mid = _P(); mid.x, mid.y = mid_x, mid_y
    depth_px = dist_xy(mid, landmarks[8], img_w,img_h)

    # Convert to inches
    return px_to_inches(height_px), px_to_inches(width_px), px_to_inches(depth_px)

def classify_parameter(value, param_name):
    small_max, med_max = MEASUREMENT_CHART[param_name]
    if value <= small_max: return 'A'
    elif value <= med_max: return 'B'
    else: return 'C'

def grip_string_from_classes(h,w,d):
    return f"{h}-{w}-{d}"

# ---------------- Camera Thread ----------------
def camera_thread():
    global current_frame, current_recommendation
    with mp_hands.Hands(static_image_mode=False,max_num_hands=1,
                        min_detection_confidence=0.6,
                        min_tracking_confidence=0.6) as hands:
        while True:
            ok, frame = cap.read()
            if not ok: continue
            img_h, img_w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            recommendation = "--"
            if results.multi_hand_landmarks:
                lm = results.multi_hand_landmarks[0].landmark
                h_in, w_in, d_in = compute_hand_measurements(lm, img_w, img_h)

                mv_height.add(h_in)
                mv_width.add(w_in)
                mv_depth.add(d_in)

                h_avg = mv_height.avg()
                w_avg = mv_width.avg()
                d_avg = mv_depth.avg()

                h_class = classify_parameter(h_avg, "height")
                w_class = classify_parameter(w_avg, "width")
                d_class = classify_parameter(d_avg, "depth")
                recommendation = grip_string_from_classes(h_class,w_class,d_class)

                mp_drawing.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

            with frame_lock:
                ret, jpeg = cv2.imencode('.jpg', frame)
                if ret:
                    current_frame = jpeg.tobytes()
                    current_recommendation = recommendation

# ---------------- Flask App ----------------
app = Flask(__name__)
current_recommendation = "--"

@app.route('/')
def index():
    html = """
<!DOCTYPE html>
<html>
<head>
<title>Grip Kiosk</title>
<style>
body {background:#222; color:#eee; font-family:sans-serif; text-align:center;}
h1 {color:#0f0;}
img#video_feed {border:2px solid #0f0; margin-top:20px;}
img#hand_guide {margin-top:20px;}
</style>
</head>
<body>
<h1>Grip Recommendation: <span id="recommendation">--</span></h1>
<img id="video_feed" src="/video_feed" width="640">
<br>
<img id="hand_guide" src="/static/hand_guide.png" width="300">
<script>
function updateRecommendation(){
    fetch('/recommendation').then(r=>r.json()).then(data=>{
        document.getElementById('recommendation').innerText = data.recommendation;
    });
}
setInterval(updateRecommendation,200);
</script>
</body>
</html>
"""
    return render_template_string(html)

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            with frame_lock:
                if current_frame is None: continue
                frame = current_frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n'+frame+b'\r\n')
    return Response(generate,mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/recommendation')
def recommendation():
    return jsonify({"recommendation": current_recommendation})

# ---------------- Static Image Export ----------------
def export_placeholder_hand_guide():
    if not os.path.exists("static"):
        os.makedirs("static")
    import numpy as np
    import cv2
    # Create a simple black & white checkerboard background with a hand shape
    img = np.ones((400,400,3),dtype=np.uint8)*255
    # Draw checkerboard squares
    for y in range(0,400,50):
        for x in range(0,400,50):
            if (x//50 + y//50)%2==0:
                cv2.rectangle(img,(x,y),(x+50,y+50),(0,0,0),-1)
    # Draw a simple hand outline (placeholder)
    pts = np.array([[150,300],[180,250],[190,180],[200,150],[210,180],[220,250],[250,300]],np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.polylines(img,[pts],True,(0,0,255),3)
    cv2.imwrite(STATIC_IMAGE_PATH,img)

# ---------------- Main ----------------
if __name__=="__main__":
    export_placeholder_hand_guide()
    threading.Thread(target=camera_thread,daemon=True).start()
    app.run(host='127.0.0.1', port=5000, debug=False)