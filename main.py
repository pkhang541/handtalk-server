import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import sys
import json
import time
import numpy as np
import cv2
from scipy.interpolate import interp1d

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import tensorflow as tf
import keras
import mediapipe as mp
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

print("[INFO] Loading photienanh VSL TensorFlow Keras model (final_model.keras)...")
MODEL_PATH = os.path.join('Models', 'checkpoints', 'final_model.keras')
LABEL_MAP_PATH = os.path.join('Logs', 'label_map.json')

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

model = keras.models.load_model(MODEL_PATH)
print("[OK] Keras model loaded successfully!")

with open(LABEL_MAP_PATH, 'r', encoding='utf-8') as f:
    label_map = json.load(f)
inv_label_map = {v: k for k, v in label_map.items()}
print(f"[OK] Loaded label map with {len(inv_label_map)} classes.")

try:
    import mediapipe.python.solutions.holistic as mp_holistic
except Exception:
    try:
        mp_holistic = mp.solutions.holistic
    except Exception:
        import mediapipe.solutions.holistic as mp_holistic

holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

N_UPPER_BODY_POSE_LANDMARKS = 25
N_HAND_LANDMARKS = 21

def mediapipe_detection(image, holistic_model):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = holistic_model.process(image_rgb)
    return results

def extract_keypoints(results):
    pose_kps = np.zeros((N_UPPER_BODY_POSE_LANDMARKS, 3))
    left_hand_kps = np.zeros((N_HAND_LANDMARKS, 3))
    right_hand_kps = np.zeros((N_HAND_LANDMARKS, 3))
    
    has_hand = False
    has_pose = False
    
    if results and results.pose_landmarks:
        has_pose = True
        for i in range(N_UPPER_BODY_POSE_LANDMARKS):
            if i < len(results.pose_landmarks.landmark):
                res = results.pose_landmarks.landmark[i]
                pose_kps[i] = [res.x, res.y, res.z]
                
    if results and results.left_hand_landmarks:
        left_hand_kps = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark])
        has_hand = True
        
    if results and results.right_hand_landmarks:
        right_hand_kps = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark])
        has_hand = True
        
    keypoints = np.concatenate([pose_kps, left_hand_kps, right_hand_kps])
    return keypoints.flatten(), has_hand, has_pose

def interpolate_keypoints(keypoints_sequence, target_len=60):
    if len(keypoints_sequence) == 0:
        return None
    if len(keypoints_sequence) == 1:
        return np.repeat(keypoints_sequence, target_len, axis=0)

    original_times = np.linspace(0, 1, len(keypoints_sequence))
    target_times = np.linspace(0, 1, target_len)
    num_features = keypoints_sequence[0].shape[0]
    interpolated_sequence = np.zeros((target_len, num_features))

    for feature_idx in range(num_features):
        feature_values = [frame[feature_idx] for frame in keypoints_sequence]
        try:
            interpolator = interp1d(
                original_times, feature_values,
                kind='cubic',
                bounds_error=False,
                fill_value="extrapolate"
            )
            interpolated_sequence[:, feature_idx] = interpolator(target_times)
        except Exception:
            interpolator = interp1d(
                original_times, feature_values,
                kind='linear',
                bounds_error=False,
                fill_value="extrapolate"
            )
            interpolated_sequence[:, feature_idx] = interpolator(target_times)

    return interpolated_sequence

def check_motion(curr_kps, prev_kps, threshold=0.005):
    if prev_kps is None or curr_kps is None:
        return False
    diff = np.abs(curr_kps - prev_kps)
    max_diff = float(np.max(diff))
    non_zero_diffs = diff[diff > 0]
    mean_diff = float(np.mean(non_zero_diffs)) if len(non_zero_diffs) > 0 else 0.0
    return max_diff > threshold or mean_diff > 0.002

app = FastAPI(title="photienanh VSL Recognition API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {
        "status": "online",
        "model": "photienanh VSL Keras Model",
        "classes": len(inv_label_map)
    }

async def handle_websocket(websocket: WebSocket):
    await websocket.accept()
    print("📲 Android Client connected to photienanh VSL WebSocket Server")
    
    accumulated_kps = []
    prev_kps = None
    is_signing = False
    still_count = 0
    frame_counter = 0
    MIN_GESTURE_FRAMES = 8

    try:
        while True:
            data = await websocket.receive_bytes()
            frame_counter += 1
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            results = mediapipe_detection(frame, holistic)
            keypoints, has_hand, has_pose = extract_keypoints(results)
            person_present = has_hand or has_pose

            is_moving = check_motion(keypoints, prev_kps, threshold=0.005)
            prev_kps = keypoints.copy()

            if person_present:
                if is_moving or has_hand or is_signing:
                    accumulated_kps.append(keypoints)
                    if is_moving or has_hand:
                        if not is_signing:
                            print(f"🎬 Gesture STARTED (frame #{frame_counter})")
                        is_signing = True
                        still_count = 0
                    else:
                        still_count += 1
            else:
                if is_signing:
                    still_count += 2

            if frame_counter % 30 == 0:
                print(f"🎥 Live Frame #{frame_counter} | person={person_present}, hand={has_hand}, moving={is_moving}, signing={is_signing}, buffer={len(accumulated_kps)}")

            if is_signing and (still_count >= 3 or len(accumulated_kps) >= 60):
                if len(accumulated_kps) >= MIN_GESTURE_FRAMES:
                    start_t = time.time()
                    kp_interpolated = interpolate_keypoints(accumulated_kps, target_len=60)
                    if kp_interpolated is not None:
                        input_tensor = np.expand_dims(kp_interpolated, axis=0)
                        preds = model.predict(input_tensor, verbose=0)
                        pred_idx = int(np.argmax(preds, axis=1)[0])
                        confidence = float(preds[0][pred_idx])
                        pred_label = inv_label_map.get(pred_idx, "Không xác định")
                        total_t = round(time.time() - start_t, 4)

                        print(f"✅ VSL Prediction Result: \"{pred_label}\" (Conf: {confidence:.2f}, {len(accumulated_kps)} frames, time: {total_t}s)")

                        await websocket.send_json({
                            "type": "recognition_result",
                            "dataset": "photienanh-VSL",
                            "glosses": [pred_label],
                            "gloss_string": pred_label,
                            "sentence": pred_label,
                            "label": pred_label,
                            "class_index": pred_idx,
                            "confidence": confidence,
                            "inference_time": total_t,
                            "total_time": total_t
                        })
                
                accumulated_kps = []
                is_signing = False
                still_count = 0

    except WebSocketDisconnect:
        print("📲 Android Client disconnected from photienanh VSL Server")
    except Exception as e:
        print(f"❌ Error in photienanh VSL WebSocket: {e}")
        try:
            await websocket.close()
        except Exception:
            pass

@app.websocket("/ws/sign-language")
async def websocket_sign_language(websocket: WebSocket):
    await handle_websocket(websocket)

@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    await handle_websocket(websocket)

if __name__ == "__main__":
    import uvicorn
    print("[OK] Starting Uvicorn Server on 0.0.0.0:8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
