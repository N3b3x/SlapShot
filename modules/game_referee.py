import time
import cv2
import numpy as np

class GameReferee:
    def __init__(self, sim, left_goal_sensor, right_goal_sensor, puck_handle, goal_width):
        """
        Initialize the GameReferee with actual proximity sensor handles.

        Args:
            sim: The simulation object for checking sensor states.
            left_goal_sensor (int): Handle for the left goal's proximity sensor.
            right_goal_sensor (int): Handle for the right goal's proximity sensor.
            puck_handle (int): Handle for the puck object.
            goal_width (float): Width of the goal opening.
        """
        self.sim = sim
        self.left_sensor = left_goal_sensor
        self.right_sensor = right_goal_sensor
        self.puck_handle = puck_handle
        self.goal_width = goal_width
        self.scores = {"left": 0, "right": 0}
        self.last_goal_time = 0
        self.goal_cooldown = 2  # seconds

    def check_goal(self):
        """
        Check proximity sensors to determine if a goal was scored.

        Returns:
            str: "left" or "right" if goal detected, else None.
        """
        now = time.time()
        if now - self.last_goal_time < self.goal_cooldown:
            return None

        try:
            # Use sim.checkProximitySensor for silent detection
            result_left, _, _, detected_obj_left, _ = self.sim.checkProximitySensor(self.left_sensor, self.sim.handle_all)
            result_right, _, _, detected_obj_right, _ = self.sim.checkProximitySensor(self.right_sensor, self.sim.handle_all)

            # Check if the detected object is the puck
            if result_left == 1 and detected_obj_left == self.puck_handle:
                self.scores["right"] += 1
                self.last_goal_time = now
                return "right"
            elif result_right == 1 and detected_obj_right == self.puck_handle:
                self.scores["left"] += 1
                self.last_goal_time = now
                return "left"
        except Exception as e:
            print(f"[ERROR] Failed to check proximity sensor: {e}")
        return None

    def check_stalled_puck(self, puck_position, puck_velocity, last_positions, threshold_time=3.0, velocity_thresh=1e-3):
        """
        Check if the puck has been stalled for too long.

        Returns:
            bool: True if the puck is stalled, False otherwise.
        """
        if np.linalg.norm(puck_velocity) < velocity_thresh:
            # Check if puck hasn't moved for threshold_time
            if len(last_positions) >= int(threshold_time / 0.05):
                dists = [np.linalg.norm(np.array(puck_position) - np.array(pos)) for pos in last_positions]
                if max(dists) < 0.01:
                    return True
        return False

    def display_game_state(self, img, position_bit=0):
        """
        Overlay the game state (scores) as titles on the image.

        Args:
            img: The image to overlay the game state on (numpy array).
            position_bit (int): 0 for bottom, 1 for top to specify where to display the scores.
        """
        if position_bit == 1:
            # Calculate the positions for the scores at the top
            left_score_position = (50, 50)
            right_score_position = (img.shape[1] - 200, 50)
            game_title_position = (img.shape[1] // 2 - 150, 100)
        else:
            # Calculate the positions for the scores at the bottom
            left_score_position = (50, img.shape[0] - 50)
            right_score_position = (img.shape[1] - 200, img.shape[0] - 50)
            game_title_position = (img.shape[1] // 2 - 150, img.shape[0] - 100)

        # Display scores
        cv2.putText(img, f"Left: {self.scores['left']}", left_score_position, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(img, f"Right: {self.scores['right']}", right_score_position, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # Display game title
        cv2.putText(img, "Air Hockey Game", game_title_position, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
