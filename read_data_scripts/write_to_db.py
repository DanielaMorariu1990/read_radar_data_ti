from datetime import datetime
import numpy as np
def save_raw_ti_data(db_conn, ti_output):
    # --- WRITE GUARD: Only if tracks exist ---
    num_tracks = ti_output.get('numDetectedTracks', 0)
    if num_tracks == 0:
        return

    cursor = db_conn.cursor()
    now = datetime.utcnow()

    try:
        # Prepare Point Cloud arrays
        # If no points are detected, we use empty lists
        pc_raw = ti_output.get('pointCloud', np.array([]))
        if pc_raw.size > 0:
            pc_x, pc_y, pc_z, pc_doppler, pc_snr = pc_raw[:, 0:5].T.tolist()
        else:
            pc_x = pc_y = pc_z = pc_doppler = pc_snr = []

        cursor.execute("""
            INSERT INTO public.radar_raw_data (
                time, num_detected_pts, num_detected_tracks,
                track_data, height_data, track_indexes,
                pc_x, pc_y, pc_z, pc_doppler, pc_snr
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            now,
            int(ti_output.get('numDetectedPoints', 0)),
            int(num_tracks),
            ti_output['trackData'].tolist(),
            ti_output['heightData'].tolist(),
            ti_output['trackIndexes'].tolist(),
            pc_x, pc_y, pc_z, pc_doppler, pc_snr
        ))

        db_conn.commit()
    except Exception as e:
        db_conn.rollback()
        print(f"HA Database Error: {e}")
    finally:
        cursor.close()
