import time
import threading
from modules.scene_builder import setup_scene
from modules.puck_tracker import PuckTracker
from modules.strike_planner import StrikePlanner

# Flags to enable/disable components
ENABLE_PUCK_TRACKER = True
ENABLE_STRIKE_PLANNER = False  # Enable both strikers

# Event to signal threads to stop
stop_event = threading.Event()

# Shared puck position buffer and lock
puck_position = {'pos': None}
puck_lock = threading.Lock()

def striker_thread(striker, prev_pos, side):
    while not stop_event.is_set():
        try:
            with puck_lock:
                pos = puck_position['pos']  # Read shared puck position
            if pos is not None and prev_pos[side] is not None:
                strike_point, strike_vel = striker.plan_strike(prev_pos[side], pos)
                if strike_point:
                    striker.execute_strike(strike_point, strike_vel)
            prev_pos[side] = pos
        except Exception as e:
            print(f"[ERROR] {side.capitalize()} striker failed: {e}")
        time.sleep(0.1)

def main():
    sim, handles = setup_scene()
    
    tracker = None
    strikerLeft = None
    strikerRight = None

    if ENABLE_PUCK_TRACKER:
        tracker = PuckTracker(sim, handles['camera'])
        if ENABLE_STRIKE_PLANNER:
            strikerLeft = StrikePlanner(sim, handles['robot_left'])
            strikerRight = StrikePlanner(sim, handles['robot_right'])

    prev_pos = {'left': None, 'right': None}
    print("[INFO] Starting puck tracking and interception...")

    try:
        if ENABLE_STRIKE_PLANNER:
            left_thread = threading.Thread(target=striker_thread, args=(strikerLeft, prev_pos, 'left'))
            right_thread = threading.Thread(target=striker_thread, args=(strikerRight, prev_pos, 'right'))
            left_thread.start()
            right_thread.start()

        while not stop_event.is_set():
            if ENABLE_PUCK_TRACKER:
                try:
                    with puck_lock:
                        puck_position['pos'] = tracker.get_puck_position()  # Update shared puck position
                except Exception as e:
                    print(f"[ERROR] Puck tracking failed: {e}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[INFO] Stopping simulation...")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
    finally:
        stop_event.set()  # Signal threads to stop
        if ENABLE_STRIKE_PLANNER and 'left_thread' in locals():
            left_thread.join()
        if ENABLE_STRIKE_PLANNER and 'right_thread' in locals():
            right_thread.join()
        if tracker:
            del tracker  # Ensure __del__ cleanup runs
        print("\n[INFO] Stopping...")
        sim.stopSimulation()

if __name__ == "__main__":
    main()