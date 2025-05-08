from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
import threading
import time

class HockeyAgent:
    def __init__(self, ik_environment, ik_group, effector_handle, robot_handle, puck_tracker, strike_depth=0.25):
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
        self.client = RemoteAPIClient()
        self.sim = self.client.require('sim')
        self.simIK = self.client.require('simIK')
        self.ik_environment = ik_environment
        self.ik_group = ik_group
        self.effector = effector_handle
        self.puck_tracker = puck_tracker
        self.robot_handle = robot_handle
        self.running = True
        self.enabled = False  # Add an enabled flag to control the agent
        self.recommended_position = None
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._update_agent, daemon=True)
        self.thread.start()
        self.strike_depth = strike_depth

    def enable(self):
        """
        Enable the agent's operations and initialize the base position and strike zone.
        The strike zone is limited to the agent's own side, within the inner bounds, and only up to strike_depth into the board.
        """
        self.base_position = np.array(self.sim.getObjectPosition(self.robot_handle, self.sim.handle_world))
        court_bounds = self.puck_tracker.get_court_bounds()
        inner_bounds = court_bounds["inner"]

        # inner_bounds: (x_min, y_min, width, height) in board-centered meters
        x_min = inner_bounds[0]
        x_max = inner_bounds[0] + inner_bounds[2]
        y_min = inner_bounds[1]
        y_max = inner_bounds[1] + inner_bounds[3]

        # Determine which side this agent is on
        is_left_agent = self.base_position[0] < 0

        # For left agent, strike zone is from x_min to (x_min + strike_depth)
        # For right agent, strike zone is from (x_max - strike_depth) to x_max
        if is_left_agent:
            sx_min = x_min
            sx_max = min(x_min + self.strike_depth, 0)  # Don't cross center
        else:
            sx_max = x_max
            sx_min = max(x_max - self.strike_depth, 0)  # Don't cross center

        self.strike_zone = {
            "x_min": sx_min,
            "x_max": sx_max,
            "y_min": y_min,
            "y_max": y_max
        }

        # Communicate strike zone to puck tracker for overlay
        if is_left_agent:
            self.puck_tracker.set_strike_zones(left_strike_zone=self.strike_zone)
        else:
            self.puck_tracker.set_strike_zones(right_strike_zone=self.strike_zone)

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
        Only computes and stores the recommended position, does not touch sim API.
        """
        while self.running:
            if not self.enabled:
                time.sleep(0.1)
                continue

            puck_data = self.puck_tracker.get_puck_data()
            recommended = self._compute_recommended_position(puck_data)
            with self.lock:
                self.recommended_position = recommended
            time.sleep(0.05)

    def _compute_recommended_position(self, puck_data):
        """
        Compute the recommended target position for the end-effector based on puck data.
        Uses board-centered coordinates.
        Only plans to hit or position on the agent's own court side (x < 0 for left, x > 0 for right).
        """
        if puck_data["position"] is None or puck_data["velocity"] is None:
            return None
        pos_board = puck_data["position"]["board"]
        vel = puck_data["velocity"]
        prediction_time = 0.5
        future_position = np.array(pos_board) + np.array(vel) * prediction_time

        # Determine which side this agent is on
        my_x = self.base_position[0]
        is_left_agent = my_x < 0

        # Only plan to hit or position on own side
        def is_on_own_side(x):
            return (is_left_agent and x < 0) or (not is_left_agent and x > 0)

        # Only plan if the predicted position is on own side and in strike zone
        if np.linalg.norm(vel) > 1e-3:
            if is_on_own_side(future_position[0]) and self._is_within_strike_zone(future_position):
                return [future_position[0], future_position[1], self.base_position[2]]

        puck_x = pos_board[0]
        if is_on_own_side(puck_x) and self._is_within_strike_zone(pos_board):
            return [pos_board[0], pos_board[1], self.base_position[2]]

        return None

    def _is_within_strike_zone(self, position):
        x, y = position
        return (
            self.strike_zone["x_min"] <= x <= self.strike_zone["x_max"]
            and self.strike_zone["y_min"] <= y <= self.strike_zone["y_max"]
        )

    def get_recommended_position(self):
        """
        Thread-safe getter for the recommended position.
        """
        with self.lock:
            return self.recommended_position

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
