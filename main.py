from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import threading
import queue
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


def main():
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
        rail_offsets=(15, 15, 15, 15),  # [top, bottom, left, right]
    )
    tracker.extract_court_bounds()

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
                scene_builder.initialize_puck_randomly(sim, handles['puck'])

            #time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n[INFO] Stopping simulation...")
    finally:
        stop_event.set()  # Signal threads to stop
        for agent in agents.values():
            agent.shutdown()
        if tracker:
            del tracker  # Ensure __del__ cleanup runs
        print("\n[INFO] Stopping...")
        sim.stopSimulation()


if __name__ == "__main__":
    main()