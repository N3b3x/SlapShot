import time
import threading
import modules.scene_builder as scene_builder
from modules.scene_builder import setup_scene
from modules.puck_tracker import PuckTracker
from modules.hockey_agent import HockeyAgent

# Flags to enable/disable components
ENABLE_PUCK_TRACKER = True

# Event to signal threads to stop
stop_event = threading.Event()

def main():
    sim, handles = setup_scene()

    goal_width = 0.2  # Example goal width

    # Initialize the PuckTracker
    tracker = PuckTracker(sim, handles['camera'], goal_width)

    # Initialize agents for both robots
    agents = [
        HockeyAgent(sim, sim.require('simIK'), handles['ik_environment'], handles['left_ik_group'], handles['effector_left'], [-0.6, 0, 0.1], tracker),
        HockeyAgent(sim, sim.require('simIK'), handles['ik_environment'], handles['right_ik_group'], handles['effector_right'], [0.6, 0, 0.1], tracker)
    ]

    print("[INFO] Starting puck tracking and agent control...")

    try:
        while not stop_event.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[INFO] Stopping simulation...")
    finally:
        stop_event.set()  # Signal threads to stop
        for agent in agents:
            agent.shutdown()
        if tracker:
            del tracker  # Ensure __del__ cleanup runs
        print("\n[INFO] Stopping...")
        sim.stopSimulation()

if __name__ == "__main__":
    main()