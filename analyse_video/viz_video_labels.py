# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "opencv-python>=4.8.0",
#     "pandas>=2.0.0",
#     "numpy>=1.24.0",
#     "lap>=0.5.13",
#     "tqdm"
# ]
# ///

import cv2
import pandas as pd
from pathlib import Path
from collections import deque

# Config paths
base_folder = "person3"
output_folder = "person3_out"

# Example file targets
video_name = "stumble_05_camera.mp4"
label_csv_name = "stumble_05_sync_labels.csv"

VALID_LABELS = [
    "standing", "walking", "sitting", "sit_table", "sit_tv", 
    "reading", "lying_sofa", "bend_down", "stand_to_sit", 
    "sit_to_stand", "sit_to_lay", "stumble", "fainting", 
    "forward_trip", "fall", "tracking_lost", "absent"
]

script_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
video_path = script_dir / base_folder / video_name
csv_path = script_dir / output_folder / label_csv_name

if not video_path.exists() or not csv_path.exists():
    print(f"❌ Error: Could not find files!\nVideo: {video_path}\nCSV: {csv_path}")
    exit()

print(f"📖 Loading dataset for verification: {csv_path.name}")
df = pd.read_csv(csv_path)

# Build a mutable map of frame indices to their dataframe positions
frame_to_row_idx = {row['frame']: idx for idx, row in df.iterrows()}

cap = cv2.VideoCapture(str(video_path))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_count = 0
has_modifications = False

# --- NEW: Playback Speed & Frame Buffer ---
playback_delay = 50  # 50ms is ~20fps (slower than standard 30fps)
frame_buffer = deque(maxlen=90) # Stores the last 90 frames (~3 seconds) in memory

print("\n========================================================")
print("📺 SCRUB & EDIT MODE ACTIVATED")
print("  - Playback is slightly slowed down by default.")
print("  - Press [ ] ] to speed up, [ [ ] to slow down.")
print("  - Press [SPACE] to PAUSE.")
print("  - While paused: [A] step back, [D] step forward, [E] Edit label.")
print("  - Press [Q] to save progress and QUIT early.")
print("========================================================\n")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
        
    frame_count += 1
    
    # Store frame and its ID in the buffer for backward scrubbing
    frame_buffer.append((frame_count, frame.copy()))
    
    row_idx = frame_to_row_idx.get(frame_count)
    current_label = df.loc[row_idx, 'label'] if row_idx is not None else "Unlabeled"
    
    text_color = (0, 0, 255) if current_label in ["fall", "forward_trip", "stumble", "fainting"] else (0, 255, 0)
    
    cv2.putText(frame, f"Frame: {frame_count}/{total_frames} | Speed Delay: {playback_delay}ms", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"STATE: {str(current_label).upper()}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
    cv2.putText(frame, "[SPACE] Pause | [ ] ] Faster | [ [ ] Slower", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    cv2.imshow("Dataset Verification Engine", frame)
    key = cv2.waitKey(playback_delay) & 0xFF
    
    # Speed Adjustments
    if key == ord(']'):
        playback_delay = max(1, playback_delay - 10)
    elif key == ord('['):
        playback_delay = min(200, playback_delay + 10)
        
    # --- INTERACTIVE SCRUB MODE ---
    elif key == ord(' '):
        buf_idx = len(frame_buffer) - 1 # Start at the most recent frame
        
        while True:
            # Pull the historical frame from memory
            scrub_frame_num, scrub_img = frame_buffer[buf_idx]
            display_img = scrub_img.copy()
            
            # Get the current label for this specific historical frame
            s_row = frame_to_row_idx.get(scrub_frame_num)
            s_label = df.loc[s_row, 'label'] if s_row is not None else "Unlabeled"
            s_color = (0, 0, 255) if s_label in ["fall", "forward_trip", "stumble", "fainting"] else (0, 255, 255)
            
            # Draw Pause HUD
            cv2.rectangle(display_img, (20, 10), (600, 140), (0, 0, 0), -1)
            cv2.putText(display_img, f"PAUSED - FRAME {scrub_frame_num}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(display_img, f"LABEL: {str(s_label).upper()}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, s_color, 2)
            cv2.putText(display_img, "[A] Back | [D] Forward | [E] Edit | [SPACE] Resume", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow("Dataset Verification Engine", display_img)
            
            pause_key = cv2.waitKey(0) & 0xFF # Wait indefinitely for input
            
            if pause_key == ord(' '):
                print("▶️ Resuming video stream...")
                break # Exit scrub mode, resume playback
            
            elif pause_key == ord('a') and buf_idx > 0:
                buf_idx -= 1 # Move backward in time
                
            elif pause_key == ord('d') and buf_idx < len(frame_buffer) - 1:
                buf_idx += 1 # Move forward in time
                
            elif pause_key == ord('e'):
                # Edit the currently viewed frame
                print(f"\n✏️  EDITING FRAME {scrub_frame_num} | Current: '{s_label}'")
                print("--------------------------------------------------------")
                for idx, lbl in enumerate(VALID_LABELS):
                    print(f"  {idx+1}: {lbl:<15}", end='\t' if (idx+1)%4!=0 else '\n')
                print("\n--------------------------------------------------------")
                
                choice = input("Enter Index to replace label (or ENTER to cancel): ").strip()
                
                if choice:
                    try:
                        chosen_idx = int(choice) - 1
                        if 0 <= chosen_idx < len(VALID_LABELS):
                            new_label = VALID_LABELS[chosen_idx]
                            if s_row is not None:
                                df.loc[s_row, 'label'] = new_label # Update memory dataframe
                                has_modifications = True
                                print(f"✅ Frame {scrub_frame_num} updated to '{new_label}'")
                            else:
                                print("⚠️ Cannot modify: Row missing.")
                        else:
                            print("❌ Invalid selection.")
                    except ValueError:
                        print("❌ Invalid entry.")

    # --- QUIT ---
    elif key == ord('q'):
        print("\n🛑 Review process interrupted by user.")
        break

cap.release()
cv2.destroyAllWindows()

if has_modifications:
    print(f"\n💾 Saving manual overrides directly to: {csv_path.name}...")
    df.to_csv(csv_path, index=False)
    print("✨ File updating completed successfully.")
else:
    print("\n🏁 Playback finished. No labels were altered.")