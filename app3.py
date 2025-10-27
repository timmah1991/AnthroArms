
import math
import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, Response, render_template_string, jsonify
import collections
import threading

# ---------------- CONFIG ----------------
SMOOTHING_FRAMES = 12

# Thresholds (normalized by image diagonal)
MEASUREMENT_CHART = {
    "height": (0.06, 0.11),
    "width":  (0.05, 0.095),
    "depth":  (0.08, 0.14),
}
# ----------------------------------------

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def image_diagonal(width, height):
    return math.hypot(width, height)

# moving average helper
class MovingAverage:
    def __init__(self, maxlen):
        self.buf = collections.deque(maxlen=maxlen)
    def add(self, val):
        self.buf.append(val)
    def avg(self):
        return sum(self.buf)/len(self.buf) if self.buf else 0.0

# global moving averages
mv_height = MovingAverage(SMOOTHING_FRAMES)
mv_width = MovingAverage(SMOOTHING_FRAMES)
mv_depth = MovingAverage(SMOOTHING_FRAMES)

# camera thread
cap = cv2.VideoCapture(0)
frame_lock = threading.Lock()
current_frame = None

def dist_xy(a,b,img_w,img_h):
    ax, ay = a.x*img_w, a.y*img_h
    bx, by = b.x*img_w, b.y*img_h
    return math.hypot(ax-bx, ay-by)

def compute_hand_measurements(landmarks,img_w,img_h):
    mcp_indices = [5,9,13,17]
    pair_dists = [dist_xy(landmarks[mcp_indices[i]], landmarks[mcp_indices[i+1]], img_w,img_h)
                  for i in range(len(mcp_indices)-1)]
    height_px = sum(pair_dists)/len(pair_dists)

    width_px = dist_xy(landmarks[1], landmarks[4], img_w,img_h)

    thumb_mcp = landmarks[2]
    index_mcp = landmarks[5]
    mid_x = (thumb_mcp.x + index_mcp.x)/2
    mid_y = (thumb_mcp.y + index_mcp.y)/2
    class _P: pass
    mid = _P(); mid.x, mid.y = mid_x, mid_y
    depth_px = dist_xy(mid, landmarks[8], img_w,img_h)

    return height_px, width_px, depth_px

def normalized_triplet(px_triplet,img_diag):
    return tuple(px/img_diag for px in px_triplet)

def classify_parameter(norm_value,param_name):
    small_max, med_max = MEASUREMENT_CHART[param_name]
    if norm_value<=small_max: return 'A'
    elif norm_value<=med_max: return 'B'
    else: return 'C'

def grip_string_from_classes(h,w,d):
    return f"{h}-{w}-{d}"

# Thread to continuously read camera frames
def camera_thread():
    global current_frame
    with mp_hands.Hands(static_image_mode=False,max_num_hands=1,
                        min_detection_confidence=0.6,
                        min_tracking_confidence=0.6) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            img_h, img_w = frame.shape[:2]
            img_diag = image_diagonal(img_w,img_h)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            recommendation = "--"
            chart_classes = {"height":"--","width":"--","depth":"--"}
            if results.multi_hand_landmarks:
                lm = results.multi_hand_landmarks[0].landmark
                h_px,w_px,d_px = compute_hand_measurements(lm,img_w,img_h)
                h_n, w_n, d_n = normalized_triplet((h_px,w_px,d_px),img_diag)

                mv_height.add(h_n)
                mv_width.add(w_n)
                mv_depth.add(d_n)

                h_avg, w_avg, d_avg = mv_height.avg(), mv_width.avg(), mv_depth.avg()
                h_class = classify_parameter(h_avg,"height")
                w_class = classify_parameter(w_avg,"width")
                d_class = classify_parameter(d_avg,"depth")
                recommendation = grip_string_from_classes(h_class,w_class,d_class)
                chart_classes = {"height":h_class,"width":w_class,"depth":d_class}

                mp_drawing.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

            ret, jpeg = cv2.imencode('.jpg', frame)
            if ret:
                with frame_lock:
                    current_frame = jpeg.tobytes()
            global chart_data
            chart_data = chart_classes
            global current_recommendation
            current_recommendation = recommendation

# Flask app
app = Flask(__name__)
chart_data = {"height":"--","width":"--","depth":"--"}
current_recommendation = "--"

@app.route('/')
def index():
    html = """
<!DOCTYPE html>
<html>
<head>
<title>Grip Kiosk</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
body {background:#222; color:#eee; font-family:sans-serif; text-align:center;}
h1 {color:#0f0;}
canvas {background:#333; margin-top:20px;}
</style>
</head>
<body>
<h1>Grip Recommendation: <span id="recommendation">--</span></h1>
<img id="video_feed" src="/video_feed" width="640">
<canvas id="chart" width="600" height="400"></canvas>
<script>
const letterToValue = {'A':1,'B':2,'C':3,'--':0};
function getStackedColors(letter){
    const options = ['A','B','C'];
    return options.map(opt=>{
        if(opt==letter) return '#0f0';  // exact = green
        else return '#ff0';             // next closest = yellow
    });
}

let ctx = document.getElementById('chart').getContext('2d');
let chart = new Chart(ctx,{
    type:'bar',
    data:{
        labels:['Height','Width','Depth'],
        datasets:[
            {label:'A',data:[0,0,0],backgroundColor:'#555'},
            {label:'B',data:[0,0,0],backgroundColor:'#555'},
            {label:'C',data:[0,0,0],backgroundColor:'#555'}
        ]
    },
    options:{
        responsive:false,
        scales:{x:{stacked:true}, y:{stacked:true, beginAtZero:true, max:3, ticks:{stepSize:1}}}
    }
});

function updateData(){
    fetch('/chart_data').then(r=>r.json()).then(data=>{
        const labels = ['height','width','depth'];
        const options = ['A','B','C'];
        options.forEach((opt,i)=>{
            chart.data.datasets[i].data = labels.map(l=>{
                return letterToValue[data[l]]==letterToValue[opt]?1:1; // all bars same height 1 for stacked visual
            });
            chart.data.datasets[i].backgroundColor = labels.map(l=>{
                const colors = getStackedColors(data[l]);
                return colors[i];
            });
        });
        chart.update();
        document.getElementById('recommendation').innerText = data.recommendation;
    });
}
setInterval(updateData,200);
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
    return Response(generate(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/chart_data')
def chart_json():
    return jsonify({**chart_data,"recommendation":current_recommendation})

if __name__=="__main__":
    threading.Thread(target=camera_thread,daemon=True).start()
    app.run(host='127.0.0.1',port=5000,debug=False)