from modules.scene_builder import setup_scene
from modules.puck_tracker import PuckTracker
from modules.strike_planner import StrikePlanner
import time

def main():
    sim, handles = setup_scene()
    tracker = PuckTracker(sim, handles['camera'])
    striker = StrikePlanner(sim, handles['ur5_effector'])

    prev_pos = None
    print("[INFO] Starting puck tracking and interception...")

    try:
        while True:
            pos = tracker.get_puck_position()
            if pos and prev_pos:
                strike_point, strike_vel = striker.plan_strike(prev_pos, pos)
                if strike_point:
                    striker.execute_strike(strike_point, strike_vel)
            prev_pos = pos
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[INFO] Stopping simulation...")
        sim.stopSimulation()

if __name__ == "__main__":
    main()