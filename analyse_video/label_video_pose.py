# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "ultralytics>=8.1.0",
#     "opencv-python>=4.8.0",
#     "pandas>=2.0.0",
#     "numpy>=1.24.0",
#     "lap>=0.5.13",
#       "tqdm"
# ]
# ///

import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from pathlib import Path
from tqdm import tqdm


# 1. Load your frame-timestamp CSV & Dynamically name the output
base_folder="person10"
output_folder="person10_out"
# Ensure the paths are evaluated absolute relative to your current script directory
script_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
dir_path = script_dir / base_folder
out_dir = script_dir / output_folder

# Create output folder if it doesn't exist
out_dir.mkdir(parents=True, exist_ok=True)

print(f"Searching for tracking pairs in: {dir_path.resolve()}")

# 1. Gather all target CSV files containing "camera" and "sync"
csv_files = [
    file for file in dir_path.iterdir()
    if file.is_file() 
    and "sync" in file.name.lower() 
    and file.suffix.lower() == '.csv'
]


# 2. Scan the output directory to find completed files
if out_dir.exists():
   
    for out_file in out_dir.iterdir():
       
        if out_file.is_file() and out_file.name.lower().endswith("_labels.csv"):
            # Reconstruct the original filename: 'walking_05_sync_label.csv' -> 'walking_05_sync.csv'
            original_csv_name = out_file.name.lower().replace("_labels.csv", ".csv")
            
            # Find and remove it from our target file list if it exists there
            csv_files = [f for f in csv_files if f.name.lower() != original_csv_name]
            print(csv_files)

# 3. Exit early if everything in the folder is already finished
if not csv_files:
    print("All files in this directory have already been processed!")
    exit()

for file_path in tqdm(csv_files, desc="Labeling all video files:"):

    csv_path = f"{file_path}"  
    time_df = pd.read_csv(csv_path)
    timestamp_map = dict(zip(time_df['frame'], time_df['unix_ts']))

    # Generate output name based on input name: e.g., "frame_timestamps_labels.csv"
    output_csv_name = f"{file_path.stem}_labels.csv"

    csv_name = file_path.stem 
    # Replace '_sync' suffix with '_camera' to get the base video filename
    # This preserves the activity label ('walking') and numeral ('05') perfectly
    video_name = csv_name.lower().replace("_sync", "_camera") + ".mp4"
    video_path = dir_path / video_name

    # Safety check: Verify the paired video file actually exists in the directory
    if not video_path.exists():
        print(f"Skipping target pair: Found CSV '{file_path.name}' but expected video '{video_name}' is missing!")
        continue

    print(f"Processing Pair -> CSV: {file_path.name} | Video: {video_path.name}")
   

    # 2. Initialize YOLOv8-Pose
    model = YOLO('yolov8n-pose.pt')
    cap = cv2.VideoCapture(video_path)

    # Tracking variables
    prev_hip_y = None
    prev_state = "standing"
    state_history = []
    frame_buffer_size = 5 
    frame_count = 0
    timeline_log = []

    # List of valid labels for quick validation selection mapping
    VALID_LABELS = [
        "walking", "standing", "sit_tv", "reading", "sit_table", "lying_sofa",
        "stand_to_sit", "sit_to_stand", "sit_to_lay", "stand_chair", "bend_down",
        "stumble", "fall", "fall_behind_chair", "fainting", "forward_trip", "absent"
    ]

    def calculate_angle(p1, p2, p3):
        a, b, c = np.array(p1[:2]), np.array(p2[:2]), np.array(p3[:2])
        ba, bc = a - b, c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

    def classify_adl(kp, bbox, velocity, prev_state):
        try:
            # 1. TRACKING QUALITY CHECK
            nose = kp[0]
            l_shoulder, r_shoulder = kp[5], kp[6]
            l_hip, r_hip = kp[11], kp[12]
            l_knee, r_knee = kp[13], kp[14]
            l_ankle, r_ankle = kp[15], kp[16]
            l_wrist, r_wrist = kp[9], kp[10]

            if nose[2] < 0.3 and l_shoulder[2] < 0.3 and r_shoulder[2] < 0.3:
                return "tracking_lost", 0.0

            left_conf = (l_hip[2] + l_knee[2] + l_ankle[2]) / 3
            right_conf = (r_hip[2] + r_knee[2] + r_ankle[2]) / 3
            
            shoulder = r_shoulder if right_conf > left_conf else l_shoulder
            hip = r_hip if right_conf > left_conf else l_hip
            knee = r_knee if right_conf > left_conf else l_knee
            ankle = r_ankle if right_conf > left_conf else l_ankle
            wrist = r_wrist if right_conf > left_conf else l_wrist

            joint_confidence = (hip[2] + knee[2] + shoulder[2]) / 3.0

            x1, y1, x2, y2 = bbox
            bbox_w, bbox_h = x2 - x1, y2 - y1
            aspect_ratio = bbox_w / (bbox_h + 1e-6)

            knee_angle = calculate_angle(hip, knee, ankle)
            hip_angle = calculate_angle(shoulder, hip, knee)
            hand_to_face_dist = np.linalg.norm(wrist[:2] - nose[:2])
            hand_to_foot_dist = np.linalg.norm(wrist[:2] - ankle[:2])
            shoulder_hip_horizontal = abs(shoulder[0] - hip[0])

            if not hasattr(classify_adl, "prev_nose_x"):
                classify_adl.prev_nose_x = nose[0]
            horizontal_head_speed = abs(nose[0] - classify_adl.prev_nose_x)
            classify_adl.prev_nose_x = nose[0]

            head_to_foot_horizontal = abs(nose[0] - ankle[0])
            head_hip_vertical_delta = abs(nose[1] - hip[1])

            # ----------------------------------------------------
            # LEVEL 1 PRIORITY: CRITICAL ACCIDENTS (DYNAMIC MOMENTS)
            # ----------------------------------------------------
            # Catch active falls mid-air using downward velocity thresholds
            if velocity > 12:
                if wrist[1] < shoulder[1] or abs(wrist[0] - nose[0]) > (bbox_w * 0.3):
                    return "forward_trip", 0.95
                return "fall", 0.98

            # STUMBLE: Sharp forward or sideways lurch while staying mostly upright
            if horizontal_head_speed > 25 or (head_to_foot_horizontal > bbox_w * 0.6 and velocity < 10):
                if aspect_ratio < 1.1 and hip_angle < 105:
                    return "stumble", 0.85

            # FAINTING / GRADUAL COLLAPSE
            if velocity > 8 and aspect_ratio > 1.1:
                return "fainting", 0.80

            # ----------------------------------------------------
            # LEVEL 2 PRIORITY: ACTIVE TRANSITIONS & GAIT
            # ----------------------------------------------------
            # ACTIVE TRANSITIONS
            if velocity < -4 and 90 < knee_angle < 165: 
                return "sit_to_stand", joint_confidence
                
            if 4 < velocity <= 12 and 90 < knee_angle < 135:
                return ("sit_to_lay" if "sit" in prev_state else "stand_to_sit"), joint_confidence

            # HEIGHT-AWARE WALKING / STANDING CHECK
            hip_relative_height = (hip[1] - y1) / (bbox_h + 1e-6)

            if knee_angle >= 115 and hip_angle >= 115 and hip_relative_height < 0.55:
                bbox_center_x = (x1 + x2) / 2
                if not hasattr(classify_adl, "prev_box_x"):
                    classify_adl.prev_box_x = bbox_center_x
                    classify_adl.walk_accumulation = 0

                horizontal_displacement = abs(bbox_center_x - classify_adl.prev_box_x)
                classify_adl.prev_box_x = bbox_center_x

                if horizontal_displacement > 0.5 or abs(velocity) > 2:
                    classify_adl.walk_accumulation = min(classify_adl.walk_accumulation + 1, 10)
                else:
                    classify_adl.walk_accumulation = max(classify_adl.walk_accumulation - 1, 0)

                if classify_adl.walk_accumulation > 3:
                    return "walking", joint_confidence
                else:
                    return "standing", joint_confidence

            # ----------------------------------------------------
            # LEVEL 3 PRIORITY: STATIC POSTURES & HORIZONTAL STATES
            # ----------------------------------------------------
            # STATIC HORIZONTAL STATE ANALYSIS (Safely downstream from active motion rules)
            if aspect_ratio > 1.25 or (shoulder_hip_horizontal > bbox_h * 0.55):
                # Global setup height check: If upper edge of bbox is high up, it's a normal ADL
                # (Assumes standard camera layout where furniture is off the absolute floor)
                if y1 < 250 and abs(velocity) < 3:  # Adjust 250 pixel height based on camera perspective if needed
                    return "lying_sofa", joint_confidence
                else:
                    # If flat, stationary, but low on the floor, it's a concluded fall state
                    return "fall", 0.95
                
            # BENDING DOWN
            if hip_angle < 75 and hand_to_foot_dist < (bbox_h * 0.4):
                return "bend_down", joint_confidence
                
            # TRUE SITTING POSTURES
            if knee_angle < 115 and hip_angle < 115:
                if hand_to_face_dist < (bbox_h * 0.25): 
                    return "reading", joint_confidence
                return "sit_tv", joint_confidence

            # THE OCCLUSION FALLBACK (Table Sitting / Leg Shadows)
            if knee[2] < 0.4 and ankle[2] < 0.4:
                is_low_in_frame = (hip[1] > y1 + (bbox_h * 0.3))
                upper_body_stable = (nose[2] > 0.6 and shoulder[2] > 0.6) and abs(velocity) < 2
                
                if is_low_in_frame and upper_body_stable:
                    return "sit_table", 0.85
                else:
                    return "walking" if abs(velocity) > 1 else "standing", 0.70

            return prev_state, joint_confidence
        except Exception:
            return prev_state, 0.0

    # 4. Core Processing Loop
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        # -----------------------------------------------------------------
        # PREPROCESSING FOR LOW LIGHT: CLAHE Enhancement
        # -----------------------------------------------------------------
        # 1. Convert frame to LAB color space (L = Lightness, A/B = Color channels)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # 2. Apply CLAHE to the L (Lightness) channel only
        # clipLimit: higher values mean more contrast. tileGridSize: local neighborhood size
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l_channel)

        # 3. Merge the enhanced L channel back and convert back to BGR
        enhanced_lab = cv2.merge((cl, a_channel, b_channel))
        preprocessed_frame = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        # -----------------------------------------------------------------

        frame_count += 1
        current_timestamp = timestamp_map.get(frame_count, "Unknown Timestamp")

        results = model.track(frame, persist=True, verbose=False)

        # Default values if no person is detected in this frame
        final_output_label = "no_person_detected"
        calculated_confidence = 0.0

        for r in results:
            if r.keypoints is not None and len(r.keypoints.data) > 0 and r.boxes.xyxy is not None:
                kp = r.keypoints.data[0].cpu().numpy()
                bbox = r.boxes.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = bbox
                
                current_hip_y = kp[12][1] if kp[12][2] > 0.5 else kp[11][1]
                velocity = current_hip_y - prev_hip_y if prev_hip_y is not None else 0
                prev_hip_y = current_hip_y

                raw_state, calculated_confidence = classify_adl(kp, bbox, velocity, prev_state)
                
                # Instant pass-through for critical safety-first labels
                # Instant pass-through for critical actions to prevent smoothing erasure
                critical_events = [
                    "fall", "forward_trip", "stumble", "fainting", "tracking_lost",
                    "sit_to_stand", "stand_to_sit", "sit_to_lay"
                ]
                if raw_state in critical_events:
                    smoothed_state = raw_state
                    state_history.clear() # Clear memory so transitions register cleanly frame-by-frame
                else:
                    state_history.append(raw_state)
                    if len(state_history) > frame_buffer_size: state_history.pop(0)
                    smoothed_state = max(set(state_history), key=state_history.count)

                final_output_label = smoothed_state

                # --- LIVE OVERWRITE ENGINE ---
                if calculated_confidence < 0.65:
                    annotated_frame = frame.copy()
                    cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
                    cv2.putText(annotated_frame, "LOW CONFIDENCE - PAUSED. CHECK TERMINAL", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(annotated_frame, f"AI Guess: {smoothed_state} ({calculated_confidence:.2f})", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.imshow("ADL Live Editor", annotated_frame)
                    cv2.waitKey(1)

                    print(f"\n⚠️  [VERIFICATION REQUIRED] Frame: {frame_count} | Timestamp: {current_timestamp}")
                    print(f"   Model Estimated: '{smoothed_state}' (Conf: {calculated_confidence:.2f})")
                    print("   --------------------------------------------------------")
                    print("   [y] Confirm Guess  |  [m] Type Text Manual Override")
                    print("   Or type a number index directly to pick an ADL:")
                    for idx, lbl in enumerate(VALID_LABELS):
                        print(f"    {idx+1}: {lbl:<18}", end='\t' if (idx+1)%3!=0 else '\n')
                    print("\n   --------------------------------------------------------")
                    
                    choice = input("Select Action / Index Number (or 'q' to stop process): ").strip().lower()

                    if choice in ['q', 'exit']:
                        print("\n🛑 Process stopped by user. Saving current logs and exiting gracefully...")
                        cap.release()
                        cv2.destroyAllWindows()
                        
                        # Save the current state of this specific file up to this exact frame before quitting
                        output_df = pd.DataFrame(timeline_log)
                        output_df.to_csv(output_csv_name, index=False)
                        print(f"✅ Partial data saved safely to {output_csv_name}")
                        exit() # Safely breaks the python runtime completely
                        
                    elif choice == 'y' or choice == '':
                        final_output_label = smoothed_state
                    elif choice == 'm':
                        final_output_label = input("Enter clean label text: ").strip()
                    else:
                        try:
                            chosen_idx = int(choice) - 1
                            if 0 <= chosen_idx < len(VALID_LABELS):
                                final_output_label = VALID_LABELS[chosen_idx]
                                print(f">> Label manually set to: {final_output_label}")
                        except ValueError:
                            print("Invalid selection. Reverting to automated guess.")
                            final_output_label = smoothed_state

                prev_state = final_output_label

                # Standard playback visual displays
                cv2.putText(frame, f"Time: {current_timestamp}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"Current Label: {final_output_label}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

        # --- PER-FRAME MANDATORY DATA LOGGING ---
        # Appends EVERY frame into the array block, capture logs even if static
        timeline_log.append({
            "frame": frame_count, 
            "timestamp": current_timestamp, 
            "label": final_output_label,
            "confidence": round(calculated_confidence, 2)
        })

        # Print live state terminal confirmation feedback 
        if frame_count % 30 == 0:  # Print updates every 30 frames to limit terminal spam
            print(f"🎬 Processing Frame {frame_count} | Current Active State: {final_output_label} ({calculated_confidence:.2f})")

        cv2.imshow("ADL Live Editor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

    # Final dataset output formatting save steps
    os.makedirs(output_folder,exist_ok=True)
    output_df = pd.DataFrame(timeline_log)
    output_df.to_csv(f"{output_folder}/{output_csv_name}", index=False)
    print(f"\n Complete! Every frame has been processed and recorded down into your custom output: '{output_csv_name}'")