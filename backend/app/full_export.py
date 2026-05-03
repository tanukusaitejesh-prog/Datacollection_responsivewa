import numpy as np
from datetime import datetime, timezone
from .landmarks25 import CUSTOM_25_LANDMARK_SPECS

FACE_LANDMARK_COUNT = 478
HAND_LANDMARK_COUNT = 42
BLENDSHAPE_COUNT = 52

# Standard ARKit/MediaPipe Blendshape Names in exact order
BLENDSHAPE_NAMES = [
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft",
    "browOuterUpRight", "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen",
    "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight",
    "mouthFrownLeft", "mouthFrownRight", "mouthFunnel", "mouthLeft",
    "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft",
    "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower",
    "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft",
    "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight",
    "tongueOut"
]

FEATURE_COLUMNS = [
    "HESHL", "HESHR", "SPELL", "SPELR", "SHWRL", "SHWRR", "ELHAL", "ELHAR",
    "THHAL", "THHAR", "THHTIL", "THHTIR", "SPKNL", "SPKNR", "HIANL", "HIANR",
    "KNFOL", "KNFOR", "DFRToFL", "MinDBFAC", "MaxDBFE", "MinDBFE", "Threshold"
]

def generate_full_csv_headers():
    """Generates the ~1700 column headers for the full capture CSV."""
    headers = [
        "H:M:S:MS)", "SubjectID", "Name", "Date", "SessionID", "Action", "Age", "Gender"
    ]
    
    # Pose Joints (25 joints in requested alphabetical order)
    for joint_name, _ in CUSTOM_25_LANDMARK_SPECS:
        headers.extend([f"{joint_name}-x", f"{joint_name}-y", f"{joint_name}-z"])
        
    # Face Landmarks (478)
    for i in range(FACE_LANDMARK_COUNT):
        headers.extend([f"Face-{i}-x", f"Face-{i}-y", f"Face-{i}-z"])
        
    # Blendshapes (52)
    for name in BLENDSHAPE_NAMES:
        headers.append(name)
        
    # Hand Landmarks (42)
    for i in range(HAND_LANDMARK_COUNT):
        headers.extend([f"Hand-{i}-x", f"Hand-{i}-y", f"Hand-{i}-z"])
        
    # Calculated Features (23)
    headers.extend(FEATURE_COLUMNS)
        
    return headers

def format_timestamp(ms):
    """Converts ms to H:M:S:MS) format."""
    total_seconds = int(ms // 1000)
    milliseconds = int(ms % 1000)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02}:{minutes:02}:{seconds:02}:{milliseconds:03})"

def calculate_features(pose_33):
    """Calculates geometric ASD features from 33-landmark pose."""
    T = pose_33.shape[0]
    features = np.zeros((T, len(FEATURE_COLUMNS)))
    
    def dist(a, b):
        return np.linalg.norm(a - b, axis=1)

    # 0: Head, 11: L_Sho, 12: R_Sho, 13: L_Elb, 14: R_Elb, 15: L_Wri, 16: R_Wri
    # 19: L_HTip, 20: R_HTip, 21: L_Thu, 22: R_Thu, 23: L_Hip, 24: R_Hip
    # 25: L_Kne, 26: R_Kne, 27: L_Ank, 28: R_Ank, 31: L_Foo, 32: R_Foo
    
    # HE-SH
    features[:, 0] = dist(pose_33[:, 0, :3], pose_33[:, 11, :3])
    features[:, 1] = dist(pose_33[:, 0, :3], pose_33[:, 12, :3])
    # SP-EL
    spine_mid = (pose_33[:, 23, :3] + pose_33[:, 24, :3]) / 2.0
    features[:, 2] = dist(spine_mid, pose_33[:, 13, :3])
    features[:, 3] = dist(spine_mid, pose_33[:, 14, :3])
    # SH-WR
    features[:, 4] = dist(pose_33[:, 11, :3], pose_33[:, 15, :3])
    features[:, 5] = dist(pose_33[:, 12, :3], pose_33[:, 16, :3])
    # EL-HA
    features[:, 6] = dist(pose_33[:, 13, :3], pose_33[:, 15, :3])
    features[:, 7] = dist(pose_33[:, 14, :3], pose_33[:, 16, :3])
    # TH-HA
    features[:, 8] = dist(pose_33[:, 21, :3], pose_33[:, 15, :3])
    features[:, 9] = dist(pose_33[:, 22, :3], pose_33[:, 16, :3])
    # TH-HTI
    features[:, 10] = dist(pose_33[:, 21, :3], pose_33[:, 19, :3])
    features[:, 11] = dist(pose_33[:, 22, :3], pose_33[:, 20, :3])
    # SP-KN
    features[:, 12] = dist(spine_mid, pose_33[:, 25, :3])
    features[:, 13] = dist(spine_mid, pose_33[:, 26, :3])
    # HI-AN
    features[:, 14] = dist(pose_33[:, 23, :3], pose_33[:, 27, :3])
    features[:, 15] = dist(pose_33[:, 24, :3], pose_33[:, 28, :3])
    # KN-FO
    features[:, 16] = dist(pose_33[:, 25, :3], pose_33[:, 31, :3])
    features[:, 17] = dist(pose_33[:, 26, :3], pose_33[:, 32, :3])
    # DFRToFL (Spine to max foot height)
    feet_y = np.maximum(pose_33[:, 31, 1], pose_33[:, 32, 1])
    features[:, 18] = np.abs(spine_mid[:, 1] - feet_y)
    # MinDBFAC (Min Face Z depth)
    features[:, 19] = np.min(np.abs(pose_33[:, 0:11, 2]), axis=1)
    # MaxDBFE (Feet distance)
    features[:, 20] = dist(pose_33[:, 31, :3], pose_33[:, 32, :3])
    # MinDBFE (Ankle distance)
    features[:, 21] = dist(pose_33[:, 27, :3], pose_33[:, 28, :3])
    # Threshold
    features[:, 22] = 0.5
    
    return features

def flatten_full_capture(payload, pose_25, timestamps):
    """
    Flattens holistic capture data.
    pose_25: [T, 25, 3]
    timestamps: [T] in ms
    """
    T = pose_25.shape[0]
    meta = payload.meta
    
    # Date formatting
    now_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    date_val = now_date
    if meta.session_id and "T" in meta.session_id:
        try:
            parts = meta.session_id.split("-")
            for p in parts:
                if len(p) >= 8 and p[:8].isdigit():
                    date_val = f"{p[:4]}-{p[4:6]}-{p[6:8]}"
                    break
        except: pass

    # Preparation
    pose_33 = np.asarray(payload.keypoints)
    face_keypoints = np.asarray(payload.face_keypoints) if payload.face_keypoints else np.zeros((T, FACE_LANDMARK_COUNT, 3))
    hand_keypoints = np.asarray(payload.hand_keypoints) if payload.hand_keypoints else np.zeros((T, HAND_LANDMARK_COUNT, 3))
    
    blendshapes_data = np.zeros((T, BLENDSHAPE_COUNT))
    if payload.face_blendshapes:
        for t, frame_shapes in enumerate(payload.face_blendshapes):
            if frame_shapes:
                shape_map = {s.get('categoryName'): s.get('score', 0.0) for s in frame_shapes if isinstance(s, dict)}
                for i, name in enumerate(BLENDSHAPE_NAMES):
                    blendshapes_data[t, i] = shape_map.get(name, 0.0)
                    
    features = calculate_features(pose_33)

    final_data = []
    for t in range(T):
        row = [
            format_timestamp(timestamps[t]),
            meta.subject_id or "",
            meta.subject_name or "",
            date_val,
            meta.session_id or "",
            meta.action_type or "",
            meta.age if meta.age is not None else "",
            meta.gender or ""
        ]
        row.extend(pose_25[t].flatten().tolist())
        row.extend(face_keypoints[t].flatten().tolist())
        row.extend(blendshapes_data[t].tolist())
        row.extend(hand_keypoints[t].flatten().tolist())
        row.extend(features[t].tolist())
        final_data.append(row)
        
    return np.array(final_data, dtype=object)
