from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
import threading
import time
import queue

class HockeyAgent:
    def __init__(self, ik_environment, ik_group, effector_handle, robot_handle, puck_tracker, target_dummy_handle, joint_handles, strike_depth=0.25):
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
            target_dummy_handle: The handle to the IK target dummy for this agent.
            joint_handles: List of joint handles for the robot.
        """
        self.client = RemoteAPIClient()
        self.sim = self.client.require('sim')
        self.simIK = self.client.require('simIK')
        self.ik_environment = ik_environment
        self.ik_group = ik_group
        self.effector = effector_handle
        self.puck_tracker = puck_tracker
        self.robot_handle = robot_handle
        self.target_dummy = target_dummy_handle
        self.running = True
        self.enabled = False  # Add an enabled flag to control the agent
        self.recommended_position = None
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._update_agent, daemon=True)
        self.thread.start()
        self.strike_depth = strike_depth
        self.motion_queue = queue.Queue()
        self.motion_thread = threading.Thread(target=self._follow_motion_plan, daemon=True)
        self.motion_thread.start()
        self.last_target = None  # Track last target to avoid redundant moves
        self.puck_data = {"position": None, "velocity": None, "z_height": None}
        self.joint_handles = joint_handles
        print(f"[INFO] Joint handles for robot {robot_handle}: {self.joint_handles}")

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

    def _compute_recommended_position(self, puck_data):
        """
        Compute the recommended target position for the end-effector based on puck data.
        Uses board-centered coordinates.
        If puck is not in strike zone but within STRIKE_DEPTH, recommend a strike along the edge to kick it out.
        """
        if puck_data["position"] is None or puck_data["velocity"] is None:
            return None
        pos_board = np.array(puck_data["position"]["board"])
        vel = np.array(puck_data["velocity"])
        strike_zone = self.strike_zone

        my_x = self.base_position[0]
        is_left_agent = my_x < 0

        # Helper: is position in strike zone
        def in_strike_zone(pos):
            x, y = pos
            return (strike_zone["x_min"] <= x <= strike_zone["x_max"] and
                    strike_zone["y_min"] <= y <= strike_zone["y_max"])

        # Helper: is position in STRIKE_DEPTH band (x only)
        def in_strike_depth_band(pos):
            x, _ = pos
            if is_left_agent:
                return strike_zone["x_min"] <= x <= strike_zone["x_max"]
            else:
                return strike_zone["x_min"] <= x <= strike_zone["x_max"]

        # If puck is in strike zone, go to it
        if in_strike_zone(pos_board):
            # Add a small offset behind the puck so we can push forward
            if is_left_agent:
                offset = -0.02  # place behind for left side
            else:
                offset = 0.02   # place behind for right side
            return [pos_board[0] + offset, pos_board[1], self.base_position[2]]

        # If puck is in STRIKE_DEPTH band but outside strike zone (i.e., y out of bounds), try to "kick" it out
        x, y = pos_board
        if in_strike_depth_band(pos_board):
            # Clamp y to nearest edge of strike zone
            if y < strike_zone["y_min"]:
                y_edge = strike_zone["y_min"]
            elif y > strike_zone["y_max"]:
                y_edge = strike_zone["y_max"]
            else:
                y_edge = y  # Shouldn't happen, but for completeness

            # Try to hit behind the puck (toward own goal) to propel it out
            # For left agent, behind is more negative x; for right agent, more positive x
            if is_left_agent:
                x_strike = max(x - 0.04, strike_zone["x_min"])
            else:
                x_strike = min(x + 0.04, strike_zone["x_max"])

            # Stay within strike zone
            x_strike = np.clip(x_strike, strike_zone["x_min"], strike_zone["x_max"])
            y_strike = np.clip(y_edge, strike_zone["y_min"], strike_zone["y_max"])
            return [x_strike, y_strike, self.base_position[2]]

        # If puck is moving, try to predict where it will enter strike zone (ray tracing)
        if np.linalg.norm(vel) > 1e-3:
            bounds = self.puck_tracker.get_court_bounds()["outer"]
            x_min, y_min, w, h = bounds
            x_max, y_max = x_min + w, y_min + h

            position = pos_board.copy()
            velocity = vel.copy()
            max_steps = 300
            time_step = 0.02

            for _ in range(max_steps):
                position = position + velocity * time_step
                if position[0] <= x_min or position[0] >= x_max:
                    velocity[0] = -velocity[0]
                    position[0] = np.clip(position[0], x_min, x_max)
                if position[1] <= y_min or position[1] >= y_max:
                    velocity[1] = -velocity[1]
                    position[1] = np.clip(position[1], y_min, y_max)
                if in_strike_zone(position):
                    return [position[0], position[1], self.base_position[2]]

            # If no intersection, go to center of strike zone edge closest to own goal
            if is_left_agent:
                x_edge = strike_zone["x_min"]
            else:
                x_edge = strike_zone["x_max"]
            y_center = (strike_zone["y_min"] + strike_zone["y_max"]) / 2
            return [x_edge, y_center, self.base_position[2]]

        # If puck is not moving, just stay at center of strike zone edge closest to own goal
        if is_left_agent:
            x_edge = strike_zone["x_min"] + 0.1
        else:
            x_edge = strike_zone["x_max"] - 0.1
        y_center = (strike_zone["y_min"] + strike_zone["y_max"]) / 2
        return [x_edge, y_center, self.base_position[2]]

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

    def _generate_path(self, target_xyz):
        """
        Move the target dummy to the desired target_xyz, then use simIK.generatePath
        to find a joint-space path for driving the IK tip (self.effector) to that target.

        Returns:
            list: A flattened list of joint configurations in row-major order, or None if it fails.
        """
        # Place our IK target dummy at the desired world coordinate
        self.sim.setObjectPosition(self.target_dummy, self.sim.handle_world, target_xyz.tolist())

        try:
            path_point_count = 20  # Increase or decrease for finer or coarser joint stepping
            config_list = self.simIK.generatePath(
                self.ik_environment,
                self.ik_group,
                self.joint_handles,  # Adapt to match your actual joint handles
                self.effector,       # Tip handle
                path_point_count
            )
            if not config_list:
                print(f"[ERROR] Failed to generate path to target: {target_xyz}")
                return None
            return config_list
        except Exception as e:
            print(f"[ERROR] Exception during path generation: {e}")
            return None

    def _follow_motion_plan(self):
        while self.running:
            if not self.enabled:
                time.sleep(0.05)
                continue
            try:
                # Always pick the most recent target
                while True:
                    target = self.motion_queue.get(timeout=0.1)
                    while not self.motion_queue.empty():
                        target = self.motion_queue.get_nowait()
                    break
            except queue.Empty:
                continue

            self.last_target = target
            # Generate a path to the new target
            path = self._generate_path(np.array(target))
            if not path:
                time.sleep(0.05)
                continue

            # Each path point has len(self.joint_handles) entries
            n_joints = len(self.joint_handles)
            total_steps = len(path) // n_joints

            for i in range(total_steps):
                # Check if a new target arrived
                if not self.motion_queue.empty():
                    print("[INFO] New target detected, replanning...")
                    break

                # Apply the i-th configuration to each joint
                for j, joint in enumerate(self.joint_handles):
                    self.sim.setJointPosition(joint, path[i * n_joints + j])

                # Sync IK to sim
                from modules.scene_builder import apply_ik_to_sim
                apply_ik_to_sim(self.simIK, self.ik_environment, self.ik_group)

                time.sleep(0.01)

    def queue_motion(self, target):
        # Only queue if target is different from last
        if self.last_target is None or not np.allclose(self.last_target, target, atol=1e-4):
            # Clear the queue before putting new target
            while not self.motion_queue.empty():
                try:
                    self.motion_queue.get_nowait()
                except queue.Empty:
                    break
            self.motion_queue.put(target)
            print(f"[INFO] New motion target queued: {target}")

    def shutdown(self):
        """
        Stop the agent's thread and clean up resources.
        """
        self.running = False
        self.thread.join()
        self.motion_thread.join()

    def __del__(self):
        """
        Ensure the agent's thread is stopped when the object is destroyed.
        """
        if hasattr(self, 'thread'):
            self.shutdown()
