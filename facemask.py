import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, 
    max_num_faces=1, 
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Landmark IDs for Metrics
NOSE_TIP = 1
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
LEFT_BROW_TOP = 70
RIGHT_BROW_TOP = 300

def get_landmark_pt(lm_list, index, w, h):
    lm = lm_list.landmark[index]
    return np.array([lm.x * w, lm.y * h, lm.z * w]) # include Z for 3D Euclidean space

# Assessment State Configuration
# Stages: 0=Baseline Calibration, 1=Smile Test, 2=Surprise Test, 3=Final Evaluation
stage = 0
stage_duration = 5.0 # seconds per test
stage_start_time = time.time()

# Baselines and Maxima Storage
baseline_mouth_width = 0.0
baseline_brow_height = 0.0

max_smile_mobility = 0.0
max_brow_mobility = 0.0

cap = cv2.VideoCapture(0)

print("--- Step 1: Please maintain a neutral, relaxed face for baseline calculation ---")

while cap.isOpened():
    success, frame = cap.read()
    if not success: 
        break
    
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    elapsed_time = time.time() - stage_start_time
    time_left = max(0.0, stage_duration - elapsed_time)
    
    if results.multi_face_landmarks:
        lms = results.multi_face_landmarks[0]
        
        # 1. Calculate Normalization Factor (Scale Invariant)
        # Interpupillary/outer-eye distance isolates facial metric from camera distance
        p_left_eye = get_landmark_pt(lms, LEFT_EYE_OUTER, w, h)
        p_right_eye = get_landmark_pt(lms, RIGHT_EYE_OUTER, w, h)
        normalization_dist = np.linalg.norm(p_left_eye - p_right_eye)
        
        if normalization_dist == 0: 
            continue
            
        # 2. Extract Key Feature Points
        p_mouth_l = get_landmark_pt(lms, MOUTH_LEFT, w, h)
        p_mouth_r = get_landmark_pt(lms, MOUTH_RIGHT, w, h)
        p_brow_l = get_landmark_pt(lms, LEFT_BROW_TOP, w, h)
        p_brow_r = get_landmark_pt(lms, RIGHT_BROW_TOP, w, h)
        p_nose = get_landmark_pt(lms, NOSE_TIP, w, h)
        
        # 3. Calculate Normalized Metrics
        # Current Mouth Width
        raw_mouth_width = np.linalg.norm(p_mouth_l - p_mouth_r)
        current_mouth_width = raw_mouth_width / normalization_dist
        
        # Current Eyebrow Height (average distance from brows to nose tip)
        dist_brow_l = np.linalg.norm(p_brow_l - p_nose)
        dist_brow_r = np.linalg.norm(p_brow_r - p_nose)
        current_brow_height = ((dist_brow_l + dist_brow_r) / 2.0) / normalization_dist

        # 4. State Machine for Clinical Prompting
        if stage == 0:
            # Calibrate Neutral Face
            cv2.putText(frame, f"CALIBRATING NEUTRAL FACE... {time_left:.1f}s", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            # Accumulate average baseline values
            baseline_mouth_width = current_mouth_width if baseline_mouth_width == 0 else (baseline_mouth_width * 0.9 + current_mouth_width * 0.1)
            baseline_brow_height = current_brow_height if baseline_brow_height == 0 else (baseline_brow_height * 0.9 + current_brow_height * 0.1)
            
            if elapsed_time >= stage_duration:
                stage = 1
                stage_start_time = time.time()
                print("--- Step 2: Smile as wide as you can! ---")
                
        elif stage == 1:
            # Test Smile (Mouth widening)
            cv2.putText(frame, f"PROMPT: SMILE AS WIDE AS POSSIBLE! {time_left:.1f}s", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            smile_delta = current_mouth_width - baseline_mouth_width
            if smile_delta > max_smile_mobility:
                max_smile_mobility = smile_delta
                
            if elapsed_time >= stage_duration:
                stage = 2
                stage_start_time = time.time()
                print("--- Step 3: Raise your eyebrows in total surprise! ---")
                
        elif stage == 2:
            # Test Surprise (Eyebrow raising)
            cv2.putText(frame, f"PROMPT: SHOW TOTAL SURPRISE! {time_left:.1f}s", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            brow_delta = current_brow_height - baseline_brow_height
            if brow_delta > max_brow_mobility:
                max_brow_mobility = brow_delta
                
            if elapsed_time >= stage_duration:
                stage = 3
                
        elif stage == 3:
            # Final Analysis & Display Thresholds
            # Thresholds derived from clinical computer-vision limits for hypomimia
            # (Typically >10-15% expansion from baseline represents healthy mobile expression)
            smile_score = min(100, max(0, (max_smile_mobility / 0.15) * 100))
            brow_score = min(100, max(0, (max_brow_mobility / 0.12) * 100))
            overall_expressiveness = (smile_score + brow_score) / 2
            
            status = "Healthy Expression Mobility" if overall_expressiveness > 50 else "Potential Masked Facies (Hypomimia)"
            color = (0, 255, 0) if overall_expressiveness > 50 else (0, 0, 255)
            
            cv2.putText(frame, f"Result: {status}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"Overall Expressiveness: {int(overall_expressiveness)}%", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Smile Mobility: {int(smile_score)}% | Brow Mobility: {int(brow_score)}%", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(frame, "Press ESC to Exit", (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.imshow('Quantitative Facial Masking Assessor', frame)
    if cv2.waitKey(5) & 0xFF == 27: 
        break

cap.release()
cv2.destroyAllWindows()
