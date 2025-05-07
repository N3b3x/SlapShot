import time
import cv2

class GameReferee:
    def __init__(self, court_bounds, goal_width):
        """
        Initialize the GameReferee.

        Args:
            court_bounds: A dictionary containing the outer and inner bounds of the court.
                          Example: {"outer": (x, y, w, h), "inner": (x, y, w, h)}.
            goal_width: The width of the goal opening.
        """
        self.court_bounds = court_bounds
        self.goal_width = goal_width
        self.scores = {"left": 0, "right": 0}
        self.last_goal_time = 0
        self.goal_cooldown = 2  # Cooldown in seconds to prevent double-counting goals

    def check_goal(self, puck_position):
        """
        Check if the puck has entered a goal and update the score.

        Args:
            puck_position: The puck's position in world coordinates (x, y).

        Returns:
            str: "left" if the left goal was scored, "right" if the right goal was scored, None otherwise.
        """
        if puck_position is None:
            return None

        x, y = puck_position
        outer_x, outer_y, outer_w, outer_h = self.court_bounds["outer"]

        # Check if the puck is outside the table bounds
        if x < outer_x or x > outer_x + outer_w:
            now = time.time()
            if now - self.last_goal_time < self.goal_cooldown:
                return None  # Prevent double-counting goals

            self.last_goal_time = now
            if x < outer_x:  # Left goal
                self.scores["right"] += 1
                return "right"
            elif x > outer_x + outer_w:  # Right goal
                self.scores["left"] += 1
                return "left"

        return None

    def display_game_state(self, img):
        """
        Overlay the game state (scores) on the puck tracker display canvas.

        Args:
            img: The image to overlay the game state on (numpy array).
        """
        # Display scores
        cv2.putText(img, f"Left: {self.scores['left']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(img, f"Right: {self.scores['right']}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Display game status
        cv2.putText(img, "Air Hockey Game", (img.shape[1] // 2 - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
