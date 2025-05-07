import numpy as np
import time

class StrikePlanner:
    def __init__(self, sim, simIK, ik_environment, ik_group, effector_handle):
        """
        Initialize the StrikePlanner.

        Args:
            sim: The simulation object.
            simIK: The simIK module.
            ik_environment: The IK environment handle.
            ik_group: The IK group handle.
            effector_handle: The handle to the robot's end-effector.
        """
        self.sim = sim
        self.simIK = simIK
        self.ik_environment = ik_environment
        self.ik_group = ik_group
        self.effector = effector_handle

    def plan_strike(self, puck_position, puck_velocity):
        """
        Plan a strike trajectory to hit the puck.

        Args:
            puck_position: The current position of the puck (x, y).
            puck_velocity: The current velocity of the puck (vx, vy).

        Returns:
            list: A list of waypoints for the paddle to follow.
        """
        if puck_position is None or puck_velocity is None:
            return None

        # Predict the puck's future position
        prediction_time = 0.5  # Predict 0.5 seconds into the future
        future_position = np.array(puck_position) + np.array(puck_velocity) * prediction_time

        # Define the strike position slightly behind the puck's future position
        strike_position = [
            future_position[0] - 0.1 * np.sign(puck_velocity[0]),  # Offset along x-axis
            future_position[1],  # Same y-coordinate
            0.02  # Paddle height above the table
        ]

        # Generate a straight-line trajectory to the strike position
        current_position = self.sim.getObjectPosition(self.effector, -1)
        waypoints = self._generate_straight_path(current_position, strike_position, num_points=10)

        return waypoints

    def execute_strike(self, waypoints):
        """
        Execute a strike by following a trajectory.

        Args:
            waypoints: A list of [x, y, z] positions for the paddle to follow.
        """
        for waypoint in waypoints:
            self.sim.setObjectPosition(self.effector, -1, waypoint)
            self.simIK.handleGroup(self.ik_environment, self.ik_group)
            self.simIK.syncToSim(self.ik_environment, [self.ik_group])
            time.sleep(0.05)  # Small delay for smooth motion

    def _generate_straight_path(self, start, end, num_points):
        """
        Generate a straight-line path between two points.

        Args:
            start: Starting [x, y, z] position.
            end: Ending [x, y, z] position.
            num_points: Number of points along the path.

        Returns:
            list: A list of [x, y, z] positions along the path.
        """
        return np.linspace(start, end, num_points).tolist()