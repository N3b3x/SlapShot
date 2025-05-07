import numpy as np
import threading
import time
from modules.strike_planner import StrikePlanner

class HockeyAgent:
    def __init__(self, sim, simIK, ik_environment, ik_group, effector_handle, base_position, puck_tracker):
        """
        Initialize the HockeyAgent.

        Args:
            sim: The simulation object for interacting with the environment.
            simIK: The simIK module.
            ik_environment: The IK environment handle.
            ik_group: The IK group handle.
            effector_handle: The handle to the robot's end-effector.
            base_position: The base position of the robot in world coordinates (x, y, z).
            puck_tracker: The PuckTracker instance to dynamically acquire court bounds and other parameters.
        """
        self.sim = sim
        self.simIK = simIK
        self.ik_environment = ik_environment
        self.ik_group = ik_group
        self.effector = effector_handle
        self.base_position = np.array(base_position)
        self.puck_tracker = puck_tracker
        self.strike_planner = StrikePlanner(sim, simIK, ik_environment, ik_group, effector_handle)
        self.running = True
        self.thread = threading.Thread(target=self._update_agent, daemon=True)
        self.thread.start()

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
            puck_position = self.puck_tracker.get_puck_position()
            puck_velocity = self.puck_tracker.get_puck_velocity()

            if puck_position and puck_velocity:
                # Plan and execute a strike
                waypoints = self.strike_planner.plan_strike(puck_position, puck_velocity)
                if waypoints:
                    self.strike_planner.execute_strike(waypoints)

            time.sleep(0.1)

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
        self.shutdown()
