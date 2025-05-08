import time
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
    # Set up the simulation scene
    sim, simIK, handles = scene_builder.setup_scene()

    # Initialize the PuckTracker
    tracker = PuckTracker(
        sim,
        handles,
        scene_builder.goal_width,
        scene_builder.puck_radius,
        0.0,
        scene_builder.camera_fov,
        scene_builder.camera_position[2],
        scene_builder.camera_resolution
    )
    tracker.extract_court_bounds()

    # Initialize the GameReferee
    referee = GameReferee(
        sim,
        handles['left_goal_sensor'],
        handles['right_goal_sensor'],
        handles['puck'],
        scene_builder.goal_width
    )

    # Initialize agents for both robots
    agents = {
        "left": HockeyAgent(sim, simIK, handles['ik_environment'], handles['left_ik_group'], handles['effector_left'], handles['robot_left'], tracker),
        "right": HockeyAgent(sim, simIK, handles['ik_environment'], handles['right_ik_group'], handles['effector_right'], handles['robot_right'], tracker)
    }

    # Enable agents
    for agent in agents.values():
        agent.enable()

    print("[INFO] Starting puck tracking and agent control...")

    try:
        while not stop_event.is_set():
            # Get puck data from the tracker
            puck_data = tracker.get_puck_data()

            # Feed puck data to each agent and apply their recommended position
            for side, agent in agents.items():
                recommended_position = agent.compute_target_position(puck_data)
                if recommended_position is not None:
                    scene_builder.move_effector_to(
                        sim,
                        simIK,
                        handles['ik_environment'],
                        handles[f"{side}_ik_group"],
                        handles[f"{side}_target_dummy"],
                        recommended_position
                    )

            # Check for goals using the referee
            goal_side = referee.check_goal()
            if goal_side:
                print(f"[INFO] Goal scored on the {goal_side} side!")
                tracker.update_scores(referee.scores)  # Update scores in the tracker display
                # Reset puck position after a goal
                scene_builder.initialize_puck_randomly(sim, handles['puck'])

            #time.sleep(0.05)  # Main loop delay
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