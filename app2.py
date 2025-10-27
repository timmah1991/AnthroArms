import collections
import math
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ------------ CONFIG: tweak these -------------
# Number of frames to average for smoothing
SMOOTHING_FRAMES = 12

# Measurement normalization: we normalize distances by the image diagonal so
# thresholds are resolution-independent.
def image_diagonal(width, height):
    return math.hypot(width, height)

# Measurement chart (normalized distance thresholds).
# For each parameter, thresholds are (small_max, medium_max) in normalized units.
# Value <= small_max -> 'A' (small)
# small_max < value <= medium_max -> 'B' (medium)
# value > medium_max -> 'C' (large)
#
# Tweak these numbers to calibrate for your rifle/grip interface.
MEASUREMENT_CHART = {
    "height": (0.06, 0.11),  # example: small <=0.06, medium <=0.11, else large
    "width":  (0.05, 0.095),
    "depth":  (0.08, 0.14),
}
# ------------------------------------------------

# Utility: Euclidean distance between two normalized landmarks (x,y)
def dist_xy(a, b, img_w, img_h):
    ax, ay = a.x * img_w, a.y * img_h
    bx, by = b.x * img_w, b.y * img_h
    return math.hypot(ax - bx, ay - by)

def classify_parameter(norm_value, param_name):
    small_max, med_max = MEASUREMENT_CHART[param_name]
    if norm_value <= small_max:
        return "A"
    elif norm_value <= med_max:
        return "B"
    else:
        return "C"

class MovingAverage:
    def __init__(self, maxlen):
        self.buf = collections.deque(maxlen=maxlen)
    def add(self, val):
        self.buf.append(val)
    def avg(self):
        if not self.buf:
            return 0.0
        return sum(self.buf) / len(self.buf)
    def clear(self):
        self.buf.clear()

def compute_hand_measurements(landmarks, img_w, img_h):
    """
    landmarks: list of 21 mp.landmark objects
    Returns raw distances (pixels): height_px, width_px, depth_px
    Normalized versions will be computed later using image diagonal.
    Landmark indices (MediaPipe):
      0: wrist
      1: thumb_cmc, 2: thumb_mcp, 3: thumb_ip, 4: thumb_tip
      5: index_mcp, 6: index_pip, 7: index_dip, 8: index_tip
      9: middle_mcp, 10: middle_pip, 11: middle_dip, 12: middle_tip
      13: ring_mcp, 14: ring_pip, 15: ring_dip, 16: ring_tip
      17: pinky_mcp, 18: pinky_pip, 19: pinky_dip, 20: pinky_tip
    """
    # Height: average center-to-center between adjacent MCPs for index->pinky
    mcp_indices = [5, 9, 13, 17]  # index, middle, ring, pinky MCP
    pair_dists = []
    for i in range(len(mcp_indices) - 1):
        a = landmarks[mcp_indices[i]]
        b = landmarks[mcp_indices[i + 1]]
        pair_dists.append(dist_xy(a, b, img_w, img_h))
    height_px = sum(pair_dists) / len(pair_dists) if pair_dists else 0.0

    # Width: thumb length (thumb CMC/MCP/choose 2 -> tip)
    # We'll use thumb CMC/1 to tip/4 as thumb length across many hand poses.
    thumb_base = landmarks[1]  # thumb_cmc
    thumb_tip = landmarks[4]
    width_px = dist_xy(thumb_base, thumb_tip, img_w, img_h)

    # Depth: distance from web-space pad (approx midpoint between thumb_mcp (2) and index_mcp (5))
    # to index fingertip (8).
    thumb_mcp = landmarks[2]
    index_mcp = landmarks[5]
    web_x = (thumb_mcp.x + index_mcp.x) / 2.0
    web_y = (thumb_mcp.y + index_mcp.y) / 2.0
    # Create a pseudo-landmark object for midpoint
    class _P: pass
    mid = _P(); mid.x, mid.y = web_x, web_y
    index_tip = landmarks[8]
    depth_px = dist_xy(mid, index_tip, img_w, img_h)

    # Return pixels
    return height_px, width_px, depth_px

def normalized_triplet(px_triplet, img_diag):
    return tuple(px / img_diag for px in px_triplet)

def grip_string_from_classes(hc, wc, dc):
    return f"{hc}-{wc}-{dc}"

def run_kiosk_camera():
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6) as hands:

        # moving averages
        mv_height = MovingAverage(SMOOTHING_FRAMES)
        mv_width = MovingAverage(SMOOTHING_FRAMES)
        mv_depth = MovingAverage(SMOOTHING_FRAMES)

        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed")
                break
            img_h, img_w = frame.shape[:2]
            img_diag = image_diagonal(img_w, img_h)

            # Convert to RGB for MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            recommendation = "--"
            if results.multi_hand_landmarks:
                # Choose the first hand
                hand_landmarks = results.multi_hand_landmarks[0].landmark

                # Compute raw pixel distances
                height_px, width_px, depth_px = compute_hand_measurements(hand_landmarks, img_w, img_h)

                # Normalize
                height_n, width_n, depth_n = normalized_triplet((height_px, width_px, depth_px), img_diag)

                # Add to moving averages
                mv_height.add(height_n)
                mv_width.add(width_n)
                mv_depth.add(depth_n)

                height_avg = mv_height.avg()
                width_avg = mv_width.avg()
                depth_avg = mv_depth.avg()

                # Classify
                h_class = classify_parameter(height_avg, "height")
                w_class = classify_parameter(width_avg, "width")
                d_class = classify_parameter(depth_avg, "depth")

                recommendation = grip_string_from_classes(h_class, w_class, d_class)

                # Draw landmarks
                mp_drawing.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

                # Overlay measurement values and classes
                txt_y = 30
                line_h = 24
                cv2.putText(frame, f"Height (norm avg): {height_avg:.3f} -> {h_class}", (10, txt_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                cv2.putText(frame, f"Width  (norm avg): {width_avg:.3f} -> {w_class}", (10, txt_y + line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                cv2.putText(frame, f"Depth  (norm avg): {depth_avg:.3f} -> {d_class}", (10, txt_y + 2*line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                cv2.putText(frame, f"Grip Recommendation: {recommendation}", (10, txt_y + 3*line_h + 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0,220,0), 2)

            else:
                # No hand: optionally clear moving averages
                # mv_height.clear(); mv_width.clear(); mv_depth.clear()
                cv2.putText(frame, "No hand detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

            cv2.imshow("Kiosk Grip Recommender", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):  # ESC or q to quit
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_kiosk_camera()