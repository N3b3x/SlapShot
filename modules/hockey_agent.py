import numpy as np
import threading
import time
from modules.scene_builder import move_effector_to, get_robot_position  # Import get_robot_position

class HockeyAgent:
    def __init__(self, sim, simIK, ik_environment, ik_group, effector_handle, robot_handle, puck_tracker):
        """
        Initialize the HockeyAgent.

        Args:
            sim: The simulation object for interacting with the environment.
            simIK: The simIK module.
            ik_environment: The IK environment handle.
            ik_group: The IK group handle.
            effector_handle: The handle to the robot's end-effector.
            robot_handle: The handle to the robot's base.
            puck_tracker: The PuckTracker instance to dynamically acquire court bounds and other parameters.
        """
        self.sim = sim
        self.simIK = simIK
        self.ik_environment = ik_environment
        self.ik_group = ik_group
        self.effector = effector_handle
        # if not self.sim.getSimulationState() == self.sim.simulation_advancing_running:
        #     raise RuntimeError("[ERROR] Simulation is not running. Cannot initialize HockeyAgent.")
        # if robot_handle == -1:
        #     raise ValueError("[ERROR] Invalid robot handle provided.")
        self.puck_tracker = puck_tracker
        self.robot_handle = robot_handle
        self.running = True
        self.enabled = False  # Add an enabled flag to control the agent
        self.thread = threading.Thread(target=self._update_agent, daemon=True)
        self.thread.start()

    def enable(self):
        """
        Enable the agent's operations and initialize the base position and strike zone.
        """
        self.base_position = np.array(self.sim.getObjectPosition(self.robot_handle, self.sim.handle_world))  # Extract base position dynamically
        court_bounds = self.puck_tracker.get_court_bounds()
        inner_bounds = court_bounds["inner"]

        # Calculate the strike zone based on inner bounds with an offset
        offset = 0.1  # Offset for forward motion
        x_min = inner_bounds[0] + offset
        x_max = inner_bounds[0] + inner_bounds[2] - offset
        y_min = inner_bounds[1] + offset
        y_max = inner_bounds[1] + inner_bounds[3] - offset
        self.strike_zone = {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}

        self.enabled = True
        print(f"[INFO] HockeyAgent enabled. Strike zone: {self.strike_zone}")

    def disable(self):
        """
        Disable the agent's operations.
        """
        self.enabled = False
        print("[INFO] HockeyAgent disabled.")

    def update_puck_data(self, position, velocity, z_height):
        """
        Update the puck's position, velocity, and z-height.

        Args:
            position: The puck's position in world coordinates (x, y).
            velocity: The puck's velocity in world coordinates (vx, vy).
            z_height: The puck's z-height.
        """
        with self.lock:
            self.puck_data["position"] = np.array(position)
            self.puck_data["velocity"] = np.array(velocity)
            self.puck_data["z_height"] = z_height

    def get_puck_data_relative_to_base(self):
        """
        Get the puck's position and velocity relative to the robot's base.

        Returns:
            dict: A dictionary containing the puck's relative position, velocity, and z-height.
        """
        with self.lock:
            if self.puck_data["position"] is None or self.puck_data["velocity"] is None:
                return None
            relative_position = self.puck_data["position"] - self.base_position[:2]
            relative_velocity = self.puck_data["velocity"]
            return {
                "position": relative_position,
                "velocity": relative_velocity,
                "z_height": self.puck_data["z_height"]
            }

    def _update_agent(self):
        """
        Continuously monitor the puck's position and plan strikes.
        """
        while self.running:
            if not self.enabled:
                time.sleep(0.1)  # Sleep briefly if the agent is disabled
                continue

            # Check if the simulation is running
            # if not self.sim.getSimulationState() == self.sim.simulation_advancing_running:
            #     print("[WARN] Simulation is not running. Skipping agent update.")
            #     time.sleep(0.1)
            #     continue

            puck_data = self.puck_tracker.get_puck_data()
            if (
                puck_data["position"] is not None
                and puck_data["velocity"] is not None
                and puck_data["trajectory"]
            ):
                # Determine the intersection point with the strike line
                strike_line_x = self.base_position[0]  # Strike line is at the robot's x-coordinate
                for point in puck_data["trajectory"]:
                    if abs(point[0] - strike_line_x) < 0.01:  # Check if the puck is near the strike line
                        strike_position = np.array([strike_line_x, point[1], self.base_position[2]])
                        self._move_to_strike_position(strike_position)
                        break

            time.sleep(0.05)

    def _is_within_strike_zone(self, position):
        """
        Check if a position is within the robot's strike zone.

        Args:
            position (list): The [x, y] position to check.

        Returns:
            bool: True if the position is within the strike zone, False otherwise.
        """
        x, y = position
        return (
            self.strike_zone["x_min"] <= x <= self.strike_zone["x_max"]
            and self.strike_zone["y_min"] <= y <= self.strike_zone["y_max"]
        )

    def move_effector_to(self, target_position):
        """
        Moves the robot's end-effector to a target position using IK,
        while keeping the orientation facing downward.

        Args:
            target_position: The 3D [x, y, z] world position to move to.
        """
        self.sim.setObjectPosition(self.effector, self.sim.handle_world, target_position)
        self.apply_ik_to_sim()

    def apply_ik_to_sim(self):
        """
        Applies inverse kinematics by syncing from simulation, handling the IK group, and syncing back to simulation.
        """
        self.simIK.syncFromSim(self.ik_environment, [self.ik_group])
        self.simIK.handleGroup(self.ik_environment, self.ik_group)
        self.simIK.syncToSim(self.ik_environment, [self.ik_group])

    def _move_to_strike_position(self, position):
        """
        Move the robot's end-effector to the strike position using IK.

        Args:
            position (list): The target (x, y, z) position for the strike.
        """
        self.move_effector_to(position)

    def compute_end_effector_pose(self, puck_data):
        """
        Compute the desired end-effector pose based on puck data.

        Args:
            puck_data (dict): A dictionary containing the puck's position, velocity, and trajectory.

        Returns:
            list: The desired [x, y, z] position for the end-effector.
        """
        # Example logic: Move to the predicted puck position
        if puck_data["trajectory"]:
            target_position = puck_data["trajectory"][0]  # First point in the trajectory
            return [target_position[0], target_position[1], self.base_position[2]]  # Keep the same z-height
        return None

    def compute_target_position(self, puck_data):
        """
        Compute the desired target position for the end-effector based on puck data.

        Args:
            puck_data (dict): A dictionary containing the puck's position, velocity, and trajectory.

        Returns:
            list: The desired [x, y, z] position for the end-effector, or None if no valid position is found.
        """
        if puck_data["position"] is None or puck_data["velocity"] is None:
            return None

        # Predict the puck's future position
        prediction_time = 0.5
        future_position = np.array(puck_data["position"]) + np.array(puck_data["velocity"]) * prediction_time

        # If puck is moving, use normal logic
        if np.linalg.norm(puck_data["velocity"]) > 1e-3:
            if self._is_within_strike_zone(future_position[:2]):
                return [future_position[0], future_position[1], self.base_position[2]]

        # Fallback: If puck is stopped and on our side, go to it and "strike"
        puck_x = puck_data["position"][0]
        my_x = self.base_position[0]
        bounds = self.strike_zone
        # Check if puck is on our half
        if (my_x < 0 and puck_x < 0) or (my_x > 0 and puck_x > 0):
            if self._is_within_strike_zone(puck_data["position"]):
                # Move to puck and prepare to strike toward opponent
                return [puck_data["position"][0], puck_data["position"][1], self.base_position[2]]

        return None

    def shutdown(self):
        """
        Stop the agent's thread and clean up resources.
        """
        self.running = False
        self.thread.join()

    def __del__(self):
        """
        Ensure the agent's thread is stopped when the object is destroyed.
        """
        if hasattr(self, 'thread'):  # Check if thread exists before calling shutdown
            self.shutdown()
