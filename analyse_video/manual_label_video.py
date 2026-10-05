# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "ultralytics>=8.1.0",
#     "opencv-python>=4.8.0",
#     "pandas>=2.0.0",
#     "numpy>=1.24.0",
#     "tqdm",
#     "lap>=0.5.13"
# ]
# ///

"""
Manual Video Labeling Tool — Hybrid YOLO + Manual Refinement

Phase 1: YOLO runs on the video and assigns coarse labels
         (standing, walking, sit_tv, lying_sofa)
Phase 2: You refine interactively — accept, split, or override
         segments with specific ADL labels.

Controls:
  Mouse slider         Drag to scrub to any frame
  Space                Play / pause
  A / D                Step ±1 frame
  Left / Right Arrow   Jump ±10 frames
  S                    Set START frame marker
  E                    Set END frame, apply selected label to range
  1–9, 0               Select label by number
  N / P                Next / previous label page
  R                    Accept YOLO suggestion for current segment
  U                    Undo last labeling action
  W                    Save and move to next video
  Q                    Save and quit
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO


# ─── CONFIG ─────────────────────────────────────────────────────────────────
base_folder = "person13"
output_folder = "person13_out"

VALID_LABELS = [
    "walking", "standing", "sit_tv", "reading", "sit_table", "lying_sofa",
    "stand_to_sit", "sit_to_stand", "sit_to_lay", "stand_chair", "bend_down",
    "stumble", "fall", "fall_behind_chair", "fainting", "forward_trip", "absent"
]

# BGR colors for timeline and overlay (OpenCV uses BGR)
LABEL_COLORS = {
    "walking":           (0, 200, 0),
    "standing":          (200, 200, 0),
    "sit_tv":            (0, 140, 255),
    "reading":           (255, 0, 200),
    "sit_table":         (180, 0, 180),
    "lying_sofa":        (50, 50, 255),
    "stand_to_sit":      (255, 200, 50),
    "sit_to_stand":      (50, 255, 200),
    "sit_to_lay":        (0, 180, 255),
    "stand_chair":       (200, 200, 50),
    "bend_down":         (0, 255, 255),
    "stumble":           (0, 100, 255),
    "fall":              (0, 0, 255),
    "fall_behind_chair": (30, 30, 200),
    "fainting":          (100, 50, 200),
    "forward_trip":      (0, 80, 230),
    "absent":            (128, 128, 128),
    "unlabeled":         (80, 80, 80),
}


# ─── GEOMETRY HELPER ────────────────────────────────────────────────────────

def calculate_angle(p1, p2, p3):
    """Calculate the angle at p2 formed by p1-p2-p3."""
    a = np.array(p1[:2], dtype=float)
    b = np.array(p2[:2], dtype=float)
    c = np.array(p3[:2], dtype=float)
    ba, bc = a - b, c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


# ─── YOLO COARSE CLASSIFIER ────────────────────────────────────────────────

def coarse_classify(kp, bbox, displacement_buffer):
    """
    Classify a single frame into one of 4 coarse labels:
      standing, walking, sit_tv, lying_sofa

    Uses only bbox geometry, knee angle, and horizontal displacement.
    Much more reliable than the 16-way classifier.
    """
    x1, y1, x2, y2 = bbox
    bbox_w = x2 - x1
    bbox_h = y2 - y1
    aspect_ratio = bbox_w / (bbox_h + 1e-6)

    # ── Lying: wide bounding box ──
    if aspect_ratio > 1.3:
        return "lying_sofa"

    # ── Get key joints (use the side with better confidence) ──
    l_shoulder, r_shoulder = kp[5], kp[6]
    l_hip, r_hip = kp[11], kp[12]
    l_knee, r_knee = kp[13], kp[14]
    l_ankle, r_ankle = kp[15], kp[16]

    left_conf = (l_hip[2] + l_knee[2] + l_ankle[2]) / 3
    right_conf = (r_hip[2] + r_knee[2] + r_ankle[2]) / 3

    if right_conf > left_conf:
        shoulder, hip, knee, ankle = r_shoulder, r_hip, r_knee, r_ankle
    else:
        shoulder, hip, knee, ankle = l_shoulder, l_hip, l_knee, l_ankle

    # Torso nearly horizontal → lying even if bbox is narrow
    shoulder_hip_horizontal = abs(float(shoulder[0]) - float(hip[0]))
    if shoulder_hip_horizontal > bbox_h * 0.55:
        return "lying_sofa"

    # ── Sitting: bent knees ──
    knee_angle = calculate_angle(hip, knee, ankle)
    if knee_angle < 120:
        return "sit_tv"

    # ── Walking vs Standing: horizontal displacement over a sliding window ──
    center_x = (x1 + x2) / 2.0
    displacement_buffer.append(float(center_x))
    if len(displacement_buffer) > 10:
        displacement_buffer.pop(0)

    if len(displacement_buffer) >= 3:
        total_disp = sum(
            abs(displacement_buffer[i] - displacement_buffer[i - 1])
            for i in range(1, len(displacement_buffer))
        )
        avg_disp = total_disp / len(displacement_buffer)
        if avg_disp > 1.5:
            return "walking"

    return "standing"


def yolo_prepass(frames, model_path):
    """
    Run YOLO on all video frames and return a list of coarse labels
    (one per frame). Falls back to 'standing' when no person is detected.
    """
    model = YOLO(model_path)
    labels = []
    displacement_buffer = []

    for frame in tqdm(frames, desc="  YOLO pre-pass"):
        results = model.track(frame, persist=True, verbose=False)

        label = "absent"  # default when no person detected
        for r in results:
            if (r.keypoints is not None
                    and len(r.keypoints.data) > 0
                    and r.boxes.xyxy is not None
                    and len(r.boxes.xyxy) > 0):
                kp = r.keypoints.data[0].cpu().numpy()
                bbox = r.boxes.xyxy[0].cpu().numpy()
                label = coarse_classify(kp, bbox, displacement_buffer)
                break

        labels.append(label)

    return labels


# ─── SEGMENT HELPER ─────────────────────────────────────────────────────────

def find_coarse_segment(coarse_labels, frame_idx):
    """
    Find the contiguous run of the same coarse label around frame_idx.
    Returns (start, end, label) — both inclusive.
    """
    label = coarse_labels[frame_idx]
    start = frame_idx
    while start > 0 and coarse_labels[start - 1] == label:
        start -= 1
    end = frame_idx
    while end < len(coarse_labels) - 1 and coarse_labels[end + 1] == label:
        end += 1
    return start, end, label


# ─── DRAWING ────────────────────────────────────────────────────────────────

def draw_timeline(display, frame_idx, total_frames, labels_p1, labels_p2, start_marker,
                  bar_height=14, margin=10):
    """Draw a color-coded dual timeline bar at the bottom of the frame."""
    h, w = display.shape[:2]
    bar_y1 = h - (2 * bar_height) - margin - 4
    bar_y2 = h - bar_height - margin
    bar_w = w - 2 * margin

    # Background
    cv2.rectangle(display, (margin, bar_y1), (margin + bar_w, bar_y1 + bar_height), (30, 30, 30), -1)
    cv2.rectangle(display, (margin, bar_y2), (margin + bar_w, bar_y2 + bar_height), (30, 30, 30), -1)

    # Colored segments — one vertical line per pixel column
    for px in range(bar_w):
        fi = int(px * total_frames / bar_w)
        fi = min(fi, total_frames - 1)
        
        color1 = LABEL_COLORS.get(labels_p1[fi], (80, 80, 80))
        cv2.line(display, (margin + px, bar_y1), (margin + px, bar_y1 + bar_height), color1, 1)
        
        color2 = LABEL_COLORS.get(labels_p2[fi], (80, 80, 80))
        cv2.line(display, (margin + px, bar_y2), (margin + px, bar_y2 + bar_height), color2, 1)

    # Border
    cv2.rectangle(display, (margin, bar_y1), (margin + bar_w, bar_y1 + bar_height), (180, 180, 180), 1)
    cv2.rectangle(display, (margin, bar_y2), (margin + bar_w, bar_y2 + bar_height), (180, 180, 180), 1)

    # Current position — white cursor
    cx = margin + int(frame_idx * bar_w / max(total_frames - 1, 1))
    cv2.line(display, (cx, bar_y1 - 4), (cx, bar_y2 + bar_height + 4), (255, 255, 255), 2)

    # Start marker — yellow
    if start_marker is not None:
        mx = margin + int(start_marker * bar_w / max(total_frames - 1, 1))
        cv2.line(display, (mx, bar_y1 - 6), (mx, bar_y2 + bar_height + 6), (0, 255, 255), 2)
        # Bracket connecting start marker to cursor
        if frame_idx != start_marker:
            bracket_y = bar_y1 - 6
            cv2.line(display, (mx, bracket_y), (cx, bracket_y), (0, 255, 255), 1)


def draw_overlay(display, frame_idx, total_frames, labels_p1, confidences_p1,
                 labels_p2, confidences_p2, selected_label_idx, start_marker,
                 label_page, is_playing, coarse_labels, person_ids=None, person_modes=None, selected_person_mode_idx=-1):
    """Draw the full HUD overlay: info panel, label palette, controls hint."""
    h, w = display.shape[:2]
    cur_lbl_p1 = labels_p1[frame_idx]
    cur_conf_p1 = confidences_p1[frame_idx]
    cur_lbl_p2 = labels_p2[frame_idx]
    cur_conf_p2 = confidences_p2[frame_idx]
    coarse_label = coarse_labels[frame_idx]

    # ── Semi-transparent top panel ──
    panel_h = 145
    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, display, 0.35, 0, display)

    # Row 1 — Frame counter + play state
    status = "PLAYING" if is_playing else "PAUSED"
    status_col = (0, 255, 0) if is_playing else (0, 200, 255)
    cv2.putText(display, f"Frame: {frame_idx + 1}/{total_frames}",
                (15, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(display, status,
                (250, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_col, 2)

    # Row 2 — Current label P1 + YOLO hint
    lbl_color_p1 = LABEL_COLORS.get(cur_lbl_p1, (200, 200, 200))
    cv2.putText(display, f"P1: {cur_lbl_p1}", (15, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.65, lbl_color_p1, 2)
    conf_col_p1 = (0, 255, 0) if cur_conf_p1 > 0.5 else (0, 0, 200)
    cv2.putText(display, f"Conf: {cur_conf_p1:.2f}", (240, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, conf_col_p1, 1)

    yolo_col = LABEL_COLORS.get(coarse_label, (150, 150, 150))
    cv2.putText(display, f"YOLO: {coarse_label}", (380, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, yolo_col, 1)

    # Row 2.5 — Current label P2
    lbl_color_p2 = LABEL_COLORS.get(cur_lbl_p2, (200, 200, 200))
    cv2.putText(display, f"P2: {cur_lbl_p2}", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, lbl_color_p2, 2)
    conf_col_p2 = (0, 255, 0) if cur_conf_p2 > 0.5 else (0, 0, 200)
    cv2.putText(display, f"Conf: {cur_conf_p2:.2f}", (240, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, conf_col_p2, 1)

    # Row 3 — Start marker + selected label
    row3_y = 96
    if start_marker is not None:
        cv2.putText(display, f"START: frame {start_marker + 1}",
                    (15, row3_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    sel_name = VALID_LABELS[selected_label_idx]
    sel_col = LABEL_COLORS.get(sel_name, (200, 200, 200))
    sel_x = 220 if start_marker is not None else 15
    cv2.putText(display, f"Selected: {sel_name}", (sel_x, row3_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, sel_col, 2)
                
    if person_modes and selected_person_mode_idx >= 0:
        sel_person = person_modes[selected_person_mode_idx]
        if not sel_person:
            sel_person = "None"
        cv2.putText(display, f"Mode: {sel_person}", (sel_x + 220, row3_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)

    # Row 4 — Label palette (current page)
    page_start = label_page * 10
    page_end = min(page_start + 10, len(VALID_LABELS))
    x_cursor = 15
    for i in range(page_start, page_end):
        num = ((i - page_start) + 1) % 10  # 1‑9, then 0
        lbl = VALID_LABELS[i]
        col = LABEL_COLORS.get(lbl, (150, 150, 150))
        marker = ">" if i == selected_label_idx else " "
        text = f"{marker}{num}:{lbl}"
        cv2.putText(display, text,
                    (x_cursor, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)
        text_w = len(text) * 7 + 8
        x_cursor += text_w
        if x_cursor > w - 120:
            break

    total_pages = (len(VALID_LABELS) + 9) // 10
    cv2.putText(display, f"[N/P] pg {label_page + 1}/{total_pages}",
                (w - 110, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (160, 160, 160), 1)

    # ── Controls hint — just above timeline ──
    hint = ("[Space]Play  [A/D]Step  [Arrows]Jump  "
            "[Z]Person  [S]Start  [E]End  [R]Accept  [U]Undo  [C]Reset  [W]Next  [Q]Quit")
    cv2.putText(display, hint,
                (10, h - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (140, 140, 140), 1)


# ─── SAVE ────────────────────────────────────────────────────────────────────

def save_labels(output_path, labels_p1, confidences_p1, labels_p2, confidences_p2, timestamp_map, total_frames, person_ids=None):
    """Save the label data to CSV in the dual timeline format."""
    rows = []
    for i in range(total_frames):
        # Match original convention: output frames are 1-indexed,
        # timestamp looked up from sync CSV using the 1-indexed frame number.
        frame_num = i + 1
        ts = timestamp_map.get(frame_num, timestamp_map.get(i, 0))
        row = {
            "frame": frame_num,
            "timestamp": ts,
            "label_p1": labels_p1[i],
            "confidence_p1": round(confidences_p1[i], 2),
            "label_p2": labels_p2[i],
            "confidence_p2": round(confidences_p2[i], 2)
        }
        if person_ids is not None:
            row["person_id"] = person_ids[i]
        rows.append(row)

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n   💾 Saved: {output_path.name} ({total_frames} frames)")


# ─── INTERACTIVE LABELER ────────────────────────────────────────────────────

def label_video(video_path, sync_csv_path, output_path, model_path, person1=None, person2=None):
    """
    Main interactive labeling loop for a single video.

    Returns True to continue to the next video, False to quit entirely.
    """

    # ── Load all frames into memory (videos are short — ~150 frames) ──
    print(f"\n📹 Loading: {video_path.name}")
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    total_frames = len(frames)
    print(f"   {total_frames} frames loaded")

    if total_frames == 0:
        print("   ⚠️  Empty video, skipping.")
        return True

    # ── Load timestamp mapping ──
    time_df = pd.read_csv(sync_csv_path)
    timestamp_map = dict(
        zip(time_df['frame'].astype(int), time_df['unix_ts'])
    )

    # ── YOLO pre-pass ──
    coarse_labels = yolo_prepass(frames, model_path)

    # Print detected coarse segments
    print("   Coarse segments:")
    idx = 0
    while idx < total_frames:
        seg_s, seg_e, seg_lbl = find_coarse_segment(coarse_labels, idx)
        print(f"     frames {seg_s + 1}–{seg_e + 1}: {seg_lbl}")
        idx = seg_e + 1

    # ── Label arrays — start as a copy of YOLO suggestions ──
    labels_p1 = list(coarse_labels)
    confidences_p1 = [0.0] * total_frames
    labels_p2 = list(coarse_labels)
    confidences_p2 = [0.0] * total_frames
    person_ids = [""] * total_frames

    # ── Person selection modes ──
    person_modes = [""]
    if person1 and person2:
        person_modes = ["", person1, person2, "both"]

    # ── UI state ──
    current_frame = 0
    selected_label_idx = 0
    selected_person_mode_idx = 1 if len(person_modes) > 1 else -1
    label_page = 0
    start_marker = None
    is_playing = False
    undo_stack = []  # [(start, end, l_p1, c_p1, l_p2, c_p2, p_ids)]

    # ── OpenCV window ──
    win = f"Manual Labeler  —  {video_path.name}"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar("Frame", win, 0, total_frames - 1, lambda _: None)

    print("   Interactive editor opened. See controls in the overlay.")
    print("   " + "─" * 55)

    # Arrow-key codes returned by cv2.waitKeyEx on macOS / Linux / Windows
    ARROW_LEFT = (63234, 2424832, 2)
    ARROW_RIGHT = (63235, 2555904, 3)

    while True:
        # ── Sync with trackbar (user may have dragged the slider) ──
        tb = cv2.getTrackbarPos("Frame", win)
        if tb != current_frame:
            current_frame = tb

        # ── Render ──
        display = frames[current_frame].copy()
        draw_overlay(display, current_frame, total_frames, labels_p1, confidences_p1,
                     labels_p2, confidences_p2, selected_label_idx, start_marker, label_page, is_playing,
                     coarse_labels, person_ids, person_modes, selected_person_mode_idx)
        draw_timeline(display, current_frame, total_frames, labels_p1, labels_p2, start_marker)
        cv2.imshow(win, display)

        # ── Input ──
        key = cv2.waitKeyEx(30)

        # Detect window closed via the X button
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            save_labels(output_path, labels_p1, confidences_p1, labels_p2, confidences_p2, timestamp_map, total_frames, person_ids)
            return False

        # No key — advance if playing
        if key == -1:
            if is_playing:
                if current_frame < total_frames - 1:
                    current_frame += 1
                    cv2.setTrackbarPos("Frame", win, current_frame)
                else:
                    is_playing = False
            continue

        key_ascii = key & 0xFF

        # ── SPACE — play / pause ──
        if key_ascii == 32:
            is_playing = not is_playing

        # ── A — step backward 1 frame ──
        elif key_ascii == ord('a'):
            is_playing = False
            current_frame = max(0, current_frame - 1)
            cv2.setTrackbarPos("Frame", win, current_frame)

        # ── D — step forward 1 frame ──
        elif key_ascii == ord('d'):
            is_playing = False
            current_frame = min(total_frames - 1, current_frame + 1)
            cv2.setTrackbarPos("Frame", win, current_frame)

        # ── Left Arrow — jump back 10 frames ──
        elif key in ARROW_LEFT:
            is_playing = False
            current_frame = max(0, current_frame - 10)
            cv2.setTrackbarPos("Frame", win, current_frame)

        # ── Right Arrow — jump forward 10 frames ──
        elif key in ARROW_RIGHT:
            is_playing = False
            current_frame = min(total_frames - 1, current_frame + 10)
            cv2.setTrackbarPos("Frame", win, current_frame)

        # ── Z — toggle person mode ──
        elif key_ascii == ord('z'):
            if len(person_modes) > 1:
                selected_person_mode_idx = (selected_person_mode_idx + 1) % len(person_modes)
                print(f"   👤 Person Mode: '{person_modes[selected_person_mode_idx]}'")
            else:
                print("   ⚠️  Person labeling not enabled for this video.")

        # ── S — set START marker ──
        elif key_ascii == ord('s'):
            start_marker = current_frame
            print(f"   📌 START at frame {current_frame + 1}")

        # ── E — set END and apply label to [start … current] ──
        elif key_ascii == ord('e'):
            if start_marker is None:
                print("   ⚠️  Press S to set a START marker first.")
            elif selected_person_mode_idx == 0:
                print("   ⚠️  Please select a Person Mode (Press Z) before applying a label.")
            else:
                s = min(start_marker, current_frame)
                e = max(start_marker, current_frame)
                label_name = VALID_LABELS[selected_label_idx]
                person_name = person_modes[selected_person_mode_idx] if selected_person_mode_idx >= 0 else ""

                # Save for undo
                undo_stack.append((
                    s, e,
                    list(labels_p1[s:e + 1]),
                    list(confidences_p1[s:e + 1]),
                    list(labels_p2[s:e + 1]),
                    list(confidences_p2[s:e + 1]),
                    list(person_ids[s:e + 1])
                ))

                for i in range(s, e + 1):
                    if selected_person_mode_idx in (1, 3):  # P1 or Both
                        labels_p1[i] = label_name
                        confidences_p1[i] = 1.0
                    if selected_person_mode_idx in (2, 3):  # P2 or Both
                        labels_p2[i] = label_name
                        confidences_p2[i] = 1.0
                    person_ids[i] = "both"  # Always set to "both" as per request

                print(f"   ✅ '{label_name}' + Person '{person_name}' → frames {s + 1}–{e + 1} "
                      f"({e - s + 1} frames)")
                # start_marker = None  # Removed so you can easily label the other person in the same range

        # ── R — accept YOLO suggestion for current coarse segment ──
        elif key_ascii == ord('r'):
            if selected_person_mode_idx == 0:
                print("   ⚠️  Please select a Person Mode (Press Z) before applying a label.")
            else:
                seg_s, seg_e, seg_lbl = find_coarse_segment(coarse_labels, current_frame)
                person_name = person_modes[selected_person_mode_idx] if selected_person_mode_idx >= 0 else ""
                
                undo_stack.append((
                    seg_s, seg_e,
                    list(labels_p1[seg_s:seg_e + 1]),
                    list(confidences_p1[seg_s:seg_e + 1]),
                    list(labels_p2[seg_s:seg_e + 1]),
                    list(confidences_p2[seg_s:seg_e + 1]),
                    list(person_ids[seg_s:seg_e + 1])
                ))
                for i in range(seg_s, seg_e + 1):
                    if selected_person_mode_idx in (1, 3):
                        labels_p1[i] = seg_lbl
                        confidences_p1[i] = 0.95
                    if selected_person_mode_idx in (2, 3):
                        labels_p2[i] = seg_lbl
                        confidences_p2[i] = 0.95
                    person_ids[i] = "both"

                print(f"   ✅ Accepted YOLO '{seg_lbl}' + Person '{person_name}' for frames "
                      f"{seg_s + 1}–{seg_e + 1}")

        # ── U — undo last action ──
        elif key_ascii == ord('u'):
            if undo_stack:
                s, e, o_l_p1, o_c_p1, o_l_p2, o_c_p2, o_pids = undo_stack.pop()
                for i in range(e - s + 1):
                    labels_p1[s + i] = o_l_p1[i]
                    confidences_p1[s + i] = o_c_p1[i]
                    labels_p2[s + i] = o_l_p2[i]
                    confidences_p2[s + i] = o_c_p2[i]
                    person_ids[s + i] = o_pids[i]
                print(f"   ↩️  Undid frames {s + 1}–{e + 1}")
            else:
                print("   ⚠️  Nothing to undo.")

        # ── C — clear / reset all to YOLO defaults ──
        elif key_ascii == ord('c'):
            # Save entire video state to undo stack
            undo_stack.append((
                0, total_frames - 1,
                list(labels_p1),
                list(confidences_p1),
                list(labels_p2),
                list(confidences_p2),
                list(person_ids)
            ))
            
            # Reset
            labels_p1 = list(coarse_labels)
            confidences_p1 = [0.0] * total_frames
            labels_p2 = list(coarse_labels)
            confidences_p2 = [0.0] * total_frames
            person_ids = [""] * total_frames
            print("   ♻️  Reset entire video back to initial YOLO guesses. (Press 'U' to undo)")

        # ── Number keys — select label ──
        elif ord('0') <= key_ascii <= ord('9'):
            if key_ascii == ord('0'):
                page_idx = 9
            else:
                page_idx = key_ascii - ord('1')

            abs_idx = label_page * 10 + page_idx
            if 0 <= abs_idx < len(VALID_LABELS):
                selected_label_idx = abs_idx
                print(f"   🏷️  Selected: {VALID_LABELS[abs_idx]}")

        # ── N — next label page ──
        elif key_ascii == ord('n'):
            total_pages = (len(VALID_LABELS) + 9) // 10
            label_page = (label_page + 1) % total_pages
            print(f"   📄 Page {label_page + 1}/{total_pages}")

        # ── P — previous label page ──
        elif key_ascii == ord('p'):
            total_pages = (len(VALID_LABELS) + 9) // 10
            label_page = (label_page - 1) % total_pages
            print(f"   📄 Page {label_page + 1}/{total_pages}")

        # ── W — save and move to next video ──
        elif key_ascii == ord('w'):
            save_labels(output_path, labels_p1, confidences_p1, labels_p2, confidences_p2,
                        timestamp_map, total_frames, person_ids)
            cv2.destroyWindow(win)
            return True

        # ── Q — save and quit ──
        elif key_ascii == ord('q'):
            save_labels(output_path, labels_p1, confidences_p1, labels_p2, confidences_p2,
                        timestamp_map, total_frames, person_ids)
            cv2.destroyWindow(win)
            return False


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    script_dir = (Path(__file__).parent
                  if "__file__" in locals() else Path.cwd())
    dir_path = script_dir / base_folder
    out_dir = script_dir / output_folder
    out_dir.mkdir(parents=True, exist_ok=True)

    # Locate YOLO model — check local dir first, then parent
    model_path = script_dir / "yolov8n-pose.pt"
    if not model_path.exists():
        model_path = script_dir.parent / "yolov8n-pose.pt"
    if not model_path.exists():
        print("❌ Could not find yolov8n-pose.pt — place it in the "
              "analyse_video/ directory.")
        return

    print(f"📂 Source:  {dir_path.resolve()}")
    print(f"📤 Output:  {out_dir.resolve()}")
    print(f"🤖 Model:   {model_path.name}\n")

    # ── Gather sync CSVs ──
    csv_files = sorted(
        f for f in dir_path.iterdir()
        if f.is_file()
        and "sync" in f.name.lower()
        and f.suffix.lower() == ".csv"
    )

    # ── Skip already-labelled videos ──
    if out_dir.exists():
        done_names = {
            f.name.lower().replace("_labels.csv", ".csv")
            for f in out_dir.iterdir()
            if f.is_file() and f.name.lower().endswith("_labels.csv")
        }
        csv_files = [f for f in csv_files if f.name.lower() not in done_names]

    if not csv_files:
        print("✅ All videos in this directory have already been labelled!")
        return

    print(f"📋 {len(csv_files)} video(s) to label\n")

    for file_path in csv_files:
        # Derive paired video filename
        video_name = (file_path.stem.lower()
                      .replace("_sync", "_camera") + ".mp4")
        video_path = dir_path / video_name

        if not video_path.exists():
            print(f"⚠️  Skipping: CSV '{file_path.name}' but video "
                  f"'{video_name}' not found")
            continue

        output_name = f"{file_path.stem}_labels.csv"
        output_path = out_dir / output_name

        # Safety net — never overwrite an existing label file
        if output_path.exists():
            print(f"⏭️  Skipping: {output_path.name} already exists")
            continue

        print("=" * 60)
        print(f"Processing: {video_path.name}")

        person1, person2 = None, None
        resp = input("Do you need to label persons for this video? (y/n) [n]: ").strip().lower()
        if resp == 'y':
            person1 = input("Enter Person 1 identifier (e.g., person_02): ").strip()
            person2 = input("Enter Person 2 identifier (e.g., person_05): ").strip()

        should_continue = label_video(video_path, file_path,
                                      output_path, str(model_path), person1, person2)
        if not should_continue:
            print("\n🛑 Quit requested. Exiting.")
            break

    cv2.destroyAllWindows()
    print("\n🏁 Done!")


if __name__ == "__main__":
    main()
