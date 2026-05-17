# /// script
# dependencies = [
#   "opencv-python",
#   "pyserial",
#   "pandas",
#   "numpy",
#   "json-fix"
# ]
# ///

import os
import cv2
import json
import sys
import time
import serial
import threading
import datetime
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__)) # inside read_data
parent_dir = os.path.dirname(current_dir)                # inside read_radar_data_ti
common_dir_path = os.path.join(parent_dir, 'common')     # path to common/

# 2. Add 'common' to the Python search path
if common_dir_path not in sys.path:
    sys.path.append(common_dir_path)

# --- RADAR PARSER FALLBACKS ---
#try:

from parseFrame import *
from gui_parser import *
# except ImportError:
#     class UARTParser:
#         def __init__(self, type): self.dataCom = None
#         def readAndParseUartDoubleCOMPort(self): return {"pointCloud": np.array([]), "numDetectedTracks": 0}

class RadarDataCollectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Radar & Video Sync Dataset Collector")
        self.root.geometry("560x780")
        
        self.stop_event = threading.Event()
        self.radar_data = []
        self.video_sync = []
        self.is_recording = False
        
        # Hardware tracking variables
        self.cap = None
        self.video_writer = None
        self.radar_thread = None
        self.recording_start_time = 0
        self.frame_idx = 0
        self.max_duration = 0
        self.activity_slug = ""
        self.person_id = ""
        
        # Paths for current run
        self.video_output_path = ""
        self.radar_output_path = ""
        self.sync_output_path = ""
        
        self.activity_map = {
            "Walking": "walking",
            "Sitting down watching TV": "sit_tv",
            "Reading a book": "reading",
            "Standing still": "standing",
            "Transition: Standing to Sitting": "stand_to_sit",
            "Transition: Sitting to Laying": "sit_to_lay",
            "Transition: Sitting to Standing": "sit_to_stand",
            "Standing via holding a chair": "stand_chair",
            "Tying shoes / Bending down": "bend_down",
            "Lying down on a sofa": "lying_sofa",
            "Sitting behind a table": "sit_table",
            "Stumbling": "stumble",
            "Falling": "fall",
            "Falling behind a chair": "fall_behind_chair",
            "Syncope / Fainting (Slow slide)": "fainting",
            "Forward trip / Hands break": "forward_trip"
        }
        
        self.setup_ui()
        
    def setup_ui(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        # 1. Storage Location Configuration
        lbl_frame_io = ttk.LabelFrame(self.root, text=" 1. Storage Location Configuration ", padding=10)
        lbl_frame_io.pack(fill="x", padx=15, pady=5)
        
        self.entry_root_dir = ttk.Entry(lbl_frame_io, width=42)
        self.entry_root_dir.insert(0, os.getcwd())
        self.entry_root_dir.pack(side="left", padx=5)
        
        btn_browse = ttk.Button(lbl_frame_io, text="Browse", command=self.browse_directory)
        btn_browse.pack(side="left", padx=5)

        # 2. Hardware Interfaces
        lbl_frame_hw = ttk.LabelFrame(self.root, text=" 2. Hardware Interfaces ", padding=10)
        lbl_frame_hw.pack(fill="x", padx=15, pady=5)
        
        ttk.Label(lbl_frame_hw, text="Camera Index:").grid(row=0, column=0, sticky="w", pady=5)
        self.combo_cam = ttk.Combobox(lbl_frame_hw, values=self.detect_cameras(), state="readonly", width=12)
        self.combo_cam.grid(row=0, column=1, sticky="w", padx=5)
        if self.combo_cam['values']: self.combo_cam.current(0)
        
        ttk.Label(lbl_frame_hw, text="Radar COM Port:").grid(row=1, column=0, sticky="w", pady=5)
        self.entry_com = ttk.Entry(lbl_frame_hw, width=15)
        self.entry_com.insert(0, "COM5")
        self.entry_com.grid(row=1, column=1, sticky="w", padx=5)

        self.var_sim_mode = tk.BooleanVar(value=False)
        self.chk_sim = ttk.Checkbutton(lbl_frame_hw, text="Enable Hardware Simulation Mode", variable=self.var_sim_mode)
        self.chk_sim.grid(row=2, column=0, columnspan=2, sticky="w", pady=5)

        # 3. Participant Demographics
        lbl_frame_sub = ttk.LabelFrame(self.root, text=" 3. Participant Demographics ", padding=10)
        lbl_frame_sub.pack(fill="x", padx=15, pady=5)
        
        ttk.Label(lbl_frame_sub, text="Age (years):").grid(row=0, column=0, sticky="w", pady=3)
        self.spin_age = ttk.Spinbox(lbl_frame_sub, from_=1, to=120, width=10)
        self.spin_age.set(25)
        self.spin_age.grid(row=0, column=1, padx=5, sticky="w")
        
        ttk.Label(lbl_frame_sub, text="Height (cm):").grid(row=1, column=0, sticky="w", pady=3)
        self.spin_height = ttk.Spinbox(lbl_frame_sub, from_=50, to=250, width=10)
        self.spin_height.set(175)
        self.spin_height.grid(row=1, column=1, padx=5, sticky="w")
        
        ttk.Label(lbl_frame_sub, text="Weight (kg):").grid(row=2, column=0, sticky="w", pady=3)
        self.spin_weight = ttk.Spinbox(lbl_frame_sub, from_=10, to=200, width=10)
        self.spin_weight.set(70)
        self.spin_weight.grid(row=2, column=1, padx=5, sticky="w")

        # 4. Task Parameters
        lbl_frame_task = ttk.LabelFrame(self.root, text=" 4. Task Parameters ", padding=10)
        lbl_frame_task.pack(fill="x", padx=15, pady=5)
        
        ttk.Label(lbl_frame_task, text="Target Activity:").grid(row=0, column=0, sticky="w", pady=5)
        self.combo_activity = ttk.Combobox(lbl_frame_task, values=list(self.activity_map.keys()), state="readonly", width=38)
        self.combo_activity.grid(row=0, column=1, padx=5, columnspan=2)
        self.combo_activity.current(0)
        
        ttk.Label(lbl_frame_task, text="Initial Delay (sec):").grid(row=1, column=0, sticky="w", pady=5)
        self.combo_delay = ttk.Combobox(lbl_frame_task, values=["5", "6", "10", "15"], state="readonly", width=10)
        self.combo_delay.current(0)
        self.combo_delay.grid(row=1, column=1, padx=5, sticky="w")
        
        ttk.Label(lbl_frame_task, text="Duration (sec):").grid(row=2, column=0, sticky="w", pady=5)
        self.spin_duration = ttk.Spinbox(lbl_frame_task, from_=5, to=300, width=10)
        self.spin_duration.set(15)
        self.spin_duration.grid(row=2, column=1, padx=5, sticky="w")

        # Action Execution Control Center
        self.lbl_status = ttk.Label(self.root, text="System Ready", font=("Helvetica", 11, "bold"), foreground="blue")
        self.lbl_status.pack(pady=10)
        
        self.btn_start = tk.Button(self.root, text="🚀 START EXPERIMENT PIPELINE", bg="#4CAF50", fg="white", font=("Helvetica", 11, "bold"), command=self.start_pipeline_countdown)
        self.btn_start.pack(fill="x", padx=25, pady=3, ipady=6)
        
        self.btn_stop = tk.Button(self.root, text="🛑 STOP EARLY & SAVE", bg="#F44336", fg="white", font=("Helvetica", 11, "bold"), state="disabled", command=self.stop_early)
        self.btn_stop.pack(fill="x", padx=25, pady=3, ipady=6)

    def detect_cameras(self):
        valid_indices = []
        for i in range(3):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY)
            if cap.isOpened():
                valid_indices.append(str(i))
                cap.release()
        return valid_indices if valid_indices else ["0"]

    def browse_directory(self):
        selected = filedialog.askdirectory()
        if selected:
            self.entry_root_dir.delete(0, tk.END)
            self.entry_root_dir.insert(0, selected)

    def get_next_person_dir(self, root_dir):
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        i = 1
        while True:
            candidate_path = os.path.join(root_dir, f"person{i}")
            if not os.path.exists(candidate_path):
                os.makedirs(candidate_path)
                return candidate_path, f"person{i}"
            elif len(os.listdir(candidate_path)) == 0:
                return candidate_path, f"person{i}"
            i += 1

    def calculate_next_run_idx(self, person_dir, activity_slug):
        existing_files = os.listdir(person_dir)
        run_idx = 1
        while True:
            target_filename = f"{activity_slug}_{run_idx:02d}_camera.mp4"
            if target_filename not in existing_files:
                return run_idx
            run_idx += 1

    def record_radar_loop(self, port, is_simulated):
        """Radar read thread remains asynchronous because it does not touch the window GUI system."""
        try:
            if is_simulated:
                while not self.stop_event.is_set():
                    now_time = time.time()
                    now_iso = datetime.datetime.utcnow().isoformat()
                    mock_packet = {
                        "time": now_iso,
                        "unix_ts": now_time,
                        "num_detected_pts": 3,
                        "num_detected_tracks": 1,
                        "track_data": [[1.0, 0.0, 1.2, 2.1]],
                        "height_data": [1.72],
                        "track_indexes": [0],
                        "pc_x": [0.4, 0.5, 0.6],
                        "pc_y": [1.1, 1.2, 1.3],
                        "pc_z": [1.0, 1.0, 1.0],
                        "pc_doppler": [0.0, 0.1, -0.1],
                        "pc_snr": [15, 20, 18]
                    }
                    self.radar_data.append(mock_packet)
                    time.sleep(0.05)
            else:
                ti_parser = UARTParser(type='DoubleCOMPort')
                ti_parser.dataCom = serial.Serial(port, 921600, timeout=0.6)
                
                while not self.stop_event.is_set():
                    ti_output = ti_parser.readAndParseUartDoubleCOMPort()
                    if ti_output:
                        num_tracks = ti_output.get('numDetectedTracks', 0)
                        now_time = time.time()
                        now_iso = datetime.datetime.utcnow().isoformat()

                        pc_raw = ti_output.get('pointCloud', np.array([]))
                        if isinstance(pc_raw, np.ndarray) and pc_raw.size > 0:
                            pc_x, pc_y, pc_z, pc_doppler, pc_snr = pc_raw[:, 0:5].T.tolist()
                        else:
                            pc_x = pc_y = pc_z = pc_doppler = pc_snr = []

                        def clean_nested_structures(val):
                            if isinstance(val, np.ndarray):
                                return val.tolist()
                            return val if val is not None else []
                        
                        reconstructed_pc = np.column_stack((pc_x, pc_y, pc_z, pc_doppler, pc_snr))

                        formatted_packet = {
                            "time": now_iso,
                            "unix_ts": now_time,  
                            "numDetectedPoints": int(ti_output.get('numDetectedPoints', 0)),
                            "numDetectedTracks": int(num_tracks),
                            "trackData": clean_nested_structures(ti_output.get('trackData')),
                            "heightData": clean_nested_structures(ti_output.get('heightData')),
                            "trackIndexes": clean_nested_structures(ti_output.get('trackIndexes')),
                            "pointCloud": reconstructed_pc,
                         
                        }
                        self.radar_data.append(formatted_packet)
        except Exception as e:
            print(f"Radar Thread Error: {e}")

    def stop_early(self):
        if self.is_recording:
            self.finalize_and_save_data()

    def start_pipeline_countdown(self):
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="disabled")
        
        initial_delay = int(self.combo_delay.get())
        self.run_countdown_step(initial_delay)

    def run_countdown_step(self, count):
        if count > 0:
            self.lbl_status.config(text=f"⚠️ GET READY! Recording starts in {count}s...", foreground="orange")
            self.root.after(1000, lambda: self.run_countdown_step(count - 1))
        else:
            self.initialize_hardware_recording()

    def initialize_hardware_recording(self):
        base_dir = self.entry_root_dir.get()
        com_port = self.entry_com.get()
        is_sim = self.var_sim_mode.get()
        self.max_duration = int(self.spin_duration.get())
        
        try:
            cam_idx = int(self.combo_cam.get().split()[0])
        except ValueError:
            cam_idx = 0
            
        age, height, weight = self.spin_age.get(), self.spin_height.get(), self.spin_weight.get()
        self.activity_slug = self.activity_map[self.combo_activity.get()]
        
        person_dir, self.person_id = self.get_next_person_dir(base_dir)
        
        json_meta_path = os.path.join(person_dir, f"{self.person_id}.json")
        if not os.path.exists(json_meta_path):
            meta_payload = {"subject_id": self.person_id, "age_years": int(age), "height_cm": int(height), "weight_kg": int(weight)}
            with open(json_meta_path, 'w') as f:
                json.dump(meta_payload, f, indent=4)

        run_idx = self.calculate_next_run_idx(person_dir, self.activity_slug)
        
        self.video_output_path = os.path.join(person_dir, f"{self.activity_slug}_{run_idx:02d}_camera.mp4")
        self.radar_output_path = os.path.join(person_dir, f"{self.activity_slug}_{run_idx:02d}_radar.json")
        self.sync_output_path = os.path.join(person_dir, f"{self.activity_slug}_{run_idx:02d}_sync.csv")

        if not is_sim:
            self.cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY)
            if not self.cap.isOpened():
                self.lbl_status.config(text="❌ Error: Camera Initialisation Failed", foreground="red")
                self.btn_start.config(state="normal")
                return
            width, height_frame = int(self.cap.get(3)), int(self.cap.get(4))
        else:
            self.cap = None
            width, height_frame = 640, 480

        self.video_writer = cv2.VideoWriter(self.video_output_path, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (width, height_frame))
        
        self.radar_data.clear()
        self.video_sync.clear()
        self.stop_event.clear()
        
        self.radar_thread = threading.Thread(target=self.record_radar_loop, args=(com_port, is_sim), daemon=True)
        self.radar_thread.start()
        
        self.lbl_status.config(text="🔴 RECORDING ACTIVE... CLICK STOP BUTTON ON THE GUI TO END", foreground="red")
        self.btn_stop.config(state="normal")
        
        self.is_recording = True
        self.recording_start_time = time.time()
        self.frame_idx = 0
        
        # Start the non-blocking execution cycle loop on the main thread
        self.record_frame_cycle()

    def record_frame_cycle(self):
        if not self.is_recording:
            return

        # Check for time-based completion bounds
        if (time.time() - self.recording_start_time) >= self.max_duration:
            self.finalize_and_save_data()
            return

        is_sim = self.var_sim_mode.get()
        if is_sim:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            ret = True
        else:
            ret, frame = self.cap.read()

        if not ret:
            self.finalize_and_save_data()
            return

        now_time = time.time()
        ts = datetime.datetime.fromtimestamp(now_time).isoformat(timespec='microseconds')
        
        self.video_writer.write(frame)
        self.video_sync.append({"frame": self.frame_idx, "sys_ts": ts, "unix_ts": now_time})
        
        if self.radar_data:
            last_pkt = self.radar_data[-1]
            pts = last_pkt.get("num_detected_pts", 0)
            trks = last_pkt.get("num_detected_tracks", 0)
            cv2.putText(frame, f"Pts: {pts} | Tracks: {trks}", (15, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        mode_prefix = "[SIMULATION]" if is_sim else "[LIVE]"
        cv2.putText(frame, f"{mode_prefix} Frame: {self.frame_idx} | {ts.split('T')[1]}", (15, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, f"Activity: {self.activity_slug}", (15, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Safe on Mac because this function executes entirely on the Main Thread loop
        cv2.imshow("Live Telemetry Sync Frame Target", frame)
        cv2.waitKey(1)
        
        self.frame_idx += 1
        
        # Request next execution frame step in ~45-50ms (Targets roughly 20-22 FPS cadence bounds)
        self.root.after(45, self.record_frame_cycle)

    def finalize_and_save_data(self):
        self.is_recording = False
        self.btn_stop.config(state="disabled")
        
        self.stop_event.set()
        if self.radar_thread:
            self.radar_thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
        if self.video_writer:
            self.video_writer.release()
            
        cv2.destroyAllWindows()
        
        with open(self.radar_output_path, 'w') as f:
            json.dump(self.radar_data, f)
            
        pd.DataFrame(self.video_sync).to_csv(self.sync_output_path, index=False)
        
        self.lbl_status.config(text=f"✅ Saved output run files successfully inside {self.person_id} structure!", foreground="green")
        messagebox.showinfo("Pipeline Complete", f"Data records safely committed to directory slot:\n{self.person_id}")
        self.btn_start.config(state="normal")

if __name__ == "__main__":
    root = tk.Tk()
    app = RadarDataCollectorGUI(root)
    root.mainloop()