from modules.scene_builder import setup_scene
from modules.puck_tracker import PuckTracker
from modules.strike_planner import StrikePlanner
import time

# Flags to enable/disable components
ENABLE_PUCK_TRACKER = True
ENABLE_STRIKE_PLANNER = False  # Only relevant if ENABLE_PUCK_TRACKER is True

def main():
    sim, handles = setup_scene()
    
    tracker = None
    striker = None

    if ENABLE_PUCK_TRACKER:
        tracker = PuckTracker(sim, handles['camera'])
        if ENABLE_STRIKE_PLANNER:
            striker = StrikePlanner(sim, handles['ur5_effector'])

    prev_pos = None
    print("[INFO] Starting puck tracking and interception...")

    try:
        while True:
            if ENABLE_PUCK_TRACKER:
                try:
                    pos = tracker.get_puck_position()
                    if pos is not None and prev_pos is not None and ENABLE_STRIKE_PLANNER:
                        strike_point, strike_vel = striker.plan_strike(prev_pos, pos)
                        if strike_point:
                            striker.execute_strike(strike_point, strike_vel)
                    prev_pos = pos
                except Exception as e:
                    print(f"[ERROR] Puck tracking failed: {e}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[INFO] Stopping simulation...")
        sim.stopSimulation()
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        sim.stopSimulation()

if __name__ == "__main__":
    main()