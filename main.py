from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import threading
import os, time
import queue
import numpy as np
import modules.scene_builder as scene_builder
from modules.puck_tracker import PuckTracker
from modules.hockey_agent import HockeyAgent
from modules.game_referee import GameReferee

# Flags to enable/disable components
ENABLE_PUCK_TRACKER = True

# Event to signal threads to stop
stop_event = threading.Event()

# Shared queues for thread-safe communication
puck_data_queue = queue.Queue()  # For puck data from PuckTracker
end_effector_poses = {"left": None, "right": None}  # Shared structure for end-effector poses
poses_lock = threading.Lock()  # Lock for accessing end-effector poses


def process_puck_data(agent, side):
    """
    Continuously process puck data to compute the desired end-effector pose.

    Args:
        agent (HockeyAgent): The hockey agent for the robot.
        side (str): "left" or "right" to identify the robot.
    """
    while not stop_event.is_set():
        try:
            puck_data = puck_data_queue.get(timeout=0.1)  # Get puck data from the queue
            if puck_data:
                # Compute the desired end-effector pose
                pose = agent.compute_end_effector_pose(puck_data)

                # Store the computed pose in the shared structure
                with poses_lock:
                    end_effector_poses[side] = pose
        except queue.Empty:
            continue


def initialize_puck_for_side(sim, puck_handle, side):
    """
    Initialize the puck in the alley of the given side ("left" or "right").
    """
    import random
    from modules.scene_builder import table_length, goal_width, table_top_z, puck_height

    if side == "left":
        x_range = (-table_length / 2 + 0.1, -table_length / 2 + 0.3)
    else:
        x_range = (table_length / 2 - 0.3, table_length / 2 - 0.1)
    y_range = (-goal_width / 2 + 0.05, goal_width / 2 - 0.05)
    x_pos = random.uniform(*x_range)
    y_pos = random.uniform(*y_range)
    sim.setObjectPosition(puck_handle, sim.handle_parent, [x_pos, y_pos, table_top_z + puck_height / 2])
    print(f"[INFO] Puck initialized for {side} at ({x_pos:.2f}, {y_pos:.2f})")

def puck_is_stalled(tracker, threshold_time=5.0, velocity_thresh=1e-3, min_dist=0.01):
    """
    Returns True if the puck hasn't moved significantly for threshold_time seconds.
    """
    # Use the last N positions over the threshold_time window
    positions = tracker.prev_positions
    if len(positions) < 2:
        return False
    # If velocity is low
    velocity = tracker.get_puck_velocity()
    if velocity is None or np.linalg.norm(velocity) > velocity_thresh:
        return False
    # Check if all positions are within min_dist of the latest
    latest = np.array(positions[-1])
    dists = [np.linalg.norm(np.array(p) - latest) for p in positions]
    if max(dists) < min_dist:
        # If the positions cover at least threshold_time (assuming 20Hz update)
        if len(positions) >= int(threshold_time / 0.05):
            return True
    return False

def main(sim_file=None):
    client = RemoteAPIClient()
    sim = client.require('sim')
    
    # Set up the simulation scene
    _, _, handles = scene_builder.setup_scene()

    # Pass table and rail properties to PuckTracker
    tracker = PuckTracker(
        handles,
        scene_builder.goal_width,
        scene_builder.puck_radius,
        0.0,
        scene_builder.camera_fov,
        scene_builder.camera_position[2],
        scene_builder.camera_resolution,
        table_length=scene_builder.table_length,
        table_width=scene_builder.table_width,
        rail_thickness=scene_builder.rail_thickness,
        rail_offsets=(18, 18, 20, 20),  # [top, bottom, left, right]
    )
    tracker.extract_court_bounds()
    tracker.update_scores({"left": 0, "right": 0})  # Ensure scores always initialized

    # Initialize the GameReferee
    referee = GameReferee(
        handles['left_goal_sensor'],
        handles['right_goal_sensor'],
        handles['puck'],
        scene_builder.goal_width
    )

    # Define strike depth (how far into the board the agent can go)
    STRIKE_DEPTH = 0.25

    # Initialize agents for both robots, passing strike_depth
    agents = {
        "left": HockeyAgent(handles['ik_environment'], handles['left_ik_group'], handles['effector_left'], handles['robot_left'], tracker, strike_depth=STRIKE_DEPTH),
        "right": HockeyAgent(handles['ik_environment'], handles['right_ik_group'], handles['effector_right'], handles['robot_right'], tracker, strike_depth=STRIKE_DEPTH)
    }

    # Enable agents (this will set and update the strike zones in the tracker for overlay)
    for agent in agents.values():
        agent.enable()

    print("[INFO] Starting puck tracking and agent control...")

    # For puck missing detection
    last_puck_seen_time = time.time()
    puck_missing_timeout = 5.0  # seconds

    try:
        while not stop_event.is_set():
            # For each agent, get their recommended position and apply it to the robot arm
            for side, agent in agents.items():
                recommended_position = agent.get_recommended_position()
                if recommended_position is not None:
                    scene_builder.move_effector_to_2(
                        handles['ik_environment'],
                        handles[f"{side}_ik_group"],
                        handles[f"{side}_target_dummy"],
                        recommended_position
                    )

            # Check for goals using the referee
            goal_side = referee.check_goal()
            if goal_side:
                print(f"[INFO] Goal scored on the {goal_side} side!")
                tracker.update_scores(referee.scores)
                initialize_puck_for_side(sim, handles['puck'], goal_side)
                tracker.last_puck_detected_time = time.time()

            # Check for puck missing for too long
            if time.time() - tracker.last_puck_detected_time > puck_missing_timeout:
                print("[WARN] Puck not detected for 5s, resetting puck randomly.")
                import random
                side = random.choice(["left", "right"])
                initialize_puck_for_side(sim, handles['puck'], side)
                tracker.last_puck_detected_time = time.time()

            # Check for stalled puck (not moving for threshold_time)
            if puck_is_stalled(tracker, threshold_time=5.0):
                print("[WARN] Puck stalled for 5s, resetting puck randomly.")
                import random
                side = random.choice(["left", "right"])
                initialize_puck_for_side(sim, handles['puck'], side)
                tracker.last_puck_detected_time = time.time()

            #time.sleep(0.01)
    except KeyboardInterrupt:
        print("\n[INFO] Stopping simulation...")
    finally:
        stop_event.set()  # Signal threads to stop
        for agent in agents.values():
            agent.shutdown()
        if tracker:
            del tracker  # Ensure __del__ cleanup runs
        print("\n[INFO] Stopping...")
        if sim_file:
            print(f'Saving sim to {sim_file}')
            sim.saveScene(sim_file)
        sim.stopSimulation()
        

if __name__ == "__main__":
    sim_file = os.path.join(os.path.dirname(__file__), 'air_hockey_play.ttt')  # Example default file path
    main(sim_file=sim_file)