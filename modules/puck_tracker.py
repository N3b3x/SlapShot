import numpy as np
import cv2
import threading
import time
import random
from modules.game_referee import GameReferee

class PuckTracker:
    """
    A class to track the puck in a hockey simulation using vision data from a camera.

    Features:
    - Detects the puck's position and velocity.
    - Tracks the puck's trail over time.
    - Displays the play area with bounding rectangles and goals.
    - Visualizes masks for different colors (black, green, blue, red).
    """

    # ==========================
    # Initialization
    # ==========================
    def __init__(self, sim, handles, goal_width, puck_radius, wall_buffer, camera_fov, camera_elevation, resolution):
        """
        Initialize the PuckTracker.

        Args:
            sim: The simulation object for interacting with the environment.
            camera_handle: The handle to the vision sensor in the simulation.
            goal_width: The width of the goal opening.
            puck_radius: The radius of the puck in meters.
            camera_fov: The field of view of the camera in degrees.
            camera_elevation: The elevation of the camera from the ground in meters.
            wall_buffer: Additional buffer distance from the walls in meters.
            resolution: The resolution of the camera as [width, height].
        """
        self.sim = sim
        self.handles = handles
        self.camera = handles['camera']  # Vision sensor handle
        self.camera_fov = np.radians(camera_fov)  # Convert FOV to radians
        self.camera_elevation = camera_elevation
        
        # Get the resolution of the vision sensor
        play_area_width_pixels = resolution[0]  # Get the width in pixels

        # Calculate the play area width in meters using trigonometry
        play_area_width_meters = 2 * camera_elevation * np.tan(self.camera_fov / 2)

        # Calculate the pixel-to-meter conversion factor
        self.pixel_to_meter = play_area_width_meters / play_area_width_pixels
        self.prev_positions = []  # Store previous puck positions for velocity calculation
        self.trail = []  # Store puck trail points with timestamps
        self.trail_duration = 2.0  # Duration (in seconds) to keep trail points
        self.running = True  # Control flag for the display thread
        self.lock = threading.Lock()  # Lock for thread-safe access to the camera
        self.thread = threading.Thread(target=self._display_canvas, daemon=True)
        self.thread.start()  # Start the display thread
        self.referee = None  # GameReferee will be initialized after bounds are extracted
        self.goal_width = goal_width
        self.puck_radius = puck_radius
        self.wall_buffer = wall_buffer
        self.referee_initialized = threading.Event()  # Event to signal referee initialization

        # Initialize bounds as None; they will be dynamically extracted
        self.outer_bounds = None
        self.inner_bounds = None
        self.velocity_history = []  # Store recent velocity values for smoothing
        self.velocity_history_size = 5  # Number of recent velocities to average

    def extract_court_bounds(self):
        """
        Dynamically extract the court bounds from the camera feed and initialize the referee.
        """
        img = self._get_image()
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        # Black mask for detecting the play area
        black_lower = (0, 0, 0)
        black_upper = (5, 5, 5)
        black_mask = cv2.inRange(img_hsv, black_lower, black_upper)

        # Extract bounds
        self.outer_bounds, self.inner_bounds = self._extract_play_area_bounds(black_mask)

        # Convert bounds to meters
        if self.outer_bounds and self.inner_bounds:
            self.outer_bounds = tuple(coord * self.pixel_to_meter for coord in self.outer_bounds)
            self.inner_bounds = tuple(coord * self.pixel_to_meter for coord in self.inner_bounds)

            # Initialize the referee with the extracted bounds, goal sensors, and puck handle
            self.referee = GameReferee(
                self.sim,
                self.handles['left_goal_sensor'],
                self.handles['right_goal_sensor'],
                self.handles['puck'],  # Pass the puck handle
                self.goal_width
            )
            self.referee_initialized.set()  # Signal that the referee is initialized

    def get_court_bounds(self):
        """
        Get the court bounds.

        Returns:
            dict: A dictionary containing the outer and inner bounds of the court.
        """
        if self.outer_bounds is None or self.inner_bounds is None:
            self.extract_court_bounds()
        return {"outer": self.outer_bounds, "inner": self.inner_bounds}

    def get_goal_width(self):
        """
        Get the goal width.

        Returns:
            float: The width of the goal opening.
        """
        return self.goal_width

    # ==========================
    # Image Retrieval
    # ==========================
    def _get_image(self):
        """
        Retrieve the current RGB image from the vision sensor and flip it vertically.

        Returns:
            np.ndarray: The flipped RGB image from the vision sensor, or None if unavailable.
        """
        try:
            with self.lock:  # Ensure thread-safe access
                raw_bytes, resolution = self.sim.getVisionSensorImg(self.camera)
                flat_img = self.sim.unpackUInt8Table(raw_bytes)  # Unpack raw image data
                w, h = resolution
                img = np.array(flat_img, dtype=np.uint8).reshape((h, w, 3))  # Reshape into RGB image
                return cv2.flip(img, 0)  # Flip the image vertically
        except Exception as e:
            print(f"[ERROR] Failed to get image: {e}")
            return None

    # ==========================
    # Puck Detection and Tracking
    # ==========================
    def get_puck_position(self):
        """
        Detect the puck's position in Cartesian coordinates.

        Returns:
            tuple: The puck's position in meters (x, y), or None if not detected.
        """
        img = self._get_image()
        if img is None:
            print("[DEBUG] No valid image retrieved. Skipping puck detection.")
            return None

        # Convert to HSV to detect the red puck
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        lower_red1, upper_red1 = (0, 100, 100), (10, 255, 255)
        lower_red2, upper_red2 = (170, 100, 100), (180, 255, 255)

        # Combine masks for both red ranges
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        # Find the center of the puck using image moments
        moments = cv2.moments(mask)
        if moments["m00"] > 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            pos = (cx * self.pixel_to_meter, cy * self.pixel_to_meter)

            # Debugging information
            #print(f"[DEBUG] Detected puck position (pixels): ({cx}, {cy})")
            #print(f"[DEBUG] Detected puck position (meters): {pos}")

            # Add to trail with current timestamp
            self.trail.append((pos[0], pos[1], time.time()))

            # Apply a moving average filter
            self.prev_positions.append(pos)
            if len(self.prev_positions) > 5:  # Keep the last 5 positions
                self.prev_positions.pop(0)
            avg_pos = np.mean(self.prev_positions, axis=0)

            # Debugging information
            #print(f"[DEBUG] Filtered puck position (meters): {avg_pos}")
            return avg_pos

        # Debugging information
        print("[DEBUG] No puck detected")
        return None

    def get_puck_velocity(self):
        """
        Calculate the puck's velocity based on its previous positions.

        Returns:
            tuple: The puck's velocity (dx, dy) in meters per second, or None if insufficient data.
        """
        if len(self.prev_positions) < 2:
            return None

        # Calculate the instantaneous velocity
        dx = self.prev_positions[-1][0] - self.prev_positions[-2][0]
        dy = self.prev_positions[-1][1] - self.prev_positions[-2][1]
        velocity = (dx, dy)

        # Add the velocity to the history
        self.velocity_history.append(velocity)
        if len(self.velocity_history) > self.velocity_history_size:
            self.velocity_history.pop(0)  # Maintain a fixed history size

        # Calculate the averaged velocity
        avg_dx = sum(v[0] for v in self.velocity_history) / len(self.velocity_history)
        avg_dy = sum(v[1] for v in self.velocity_history) / len(self.velocity_history)

        return avg_dx, avg_dy

    def predict_puck_trajectory(self, restitution=0.9, randomness=0.05, agent_x=None):
        """
        Predict the puck's trajectory using ray tracing and bounce logic.

        Args:
            restitution (float): Coefficient of restitution for puck reflections.
            randomness (float): Randomness factor to simulate imperfect predictions.
            agent_x (float): X position of the agent to check if puck is moving toward it.

        Returns:
            list: A list of predicted positions (x, y) along the puck's trajectory.
        """
        if self.prev_positions and len(self.prev_positions) >= 2:
            position = np.array(self.prev_positions[-1])
            velocity = np.array(self.get_puck_velocity())
            if velocity is None or np.linalg.norm(velocity) < 1e-3:
                return []

            # Only predict if puck is moving toward the agent (if agent_x is given)
            if agent_x is not None:
                if (velocity[0] > 0 and position[0] > agent_x) or (velocity[0] < 0 and position[0] < agent_x):
                    return []

            bounds = self.get_court_bounds()["outer"]
            if not bounds:
                return []

            x_min, y_min, w, h = bounds
            x_max, y_max = x_min + w, y_min + h
            trajectory = [position.copy()]
            time_step = 0.02  # Finer simulation time step

            for _ in range(200):  # Predict up to 200 steps
                position = position + velocity * time_step

                # Bounce off left/right walls
                if position[0] <= x_min or position[0] >= x_max:
                    velocity[0] = -velocity[0] * restitution * (1 + randomness * (random.random() - 0.5))
                    position[0] = np.clip(position[0], x_min, x_max)
                # Bounce off top/bottom walls
                if position[1] <= y_min or position[1] >= y_max:
                    velocity[1] = -velocity[1] * restitution * (1 + randomness * (random.random() - 0.5))
                    position[1] = np.clip(position[1], y_min, y_max)

                trajectory.append(position.copy())

                # If agent_x is given, stop if we cross the agent's strike plane
                if agent_x is not None:
                    if (velocity[0] > 0 and position[0] >= agent_x) or (velocity[0] < 0 and position[0] <= agent_x):
                        break

            return trajectory
        return []

    def get_puck_data(self):
        """
        Get the puck's current position, velocity, and predicted trajectory.

        Returns:
            dict: A dictionary containing the puck's position, velocity, and trajectory.
        """
        position = self.get_puck_position()
        velocity = self.get_puck_velocity()
        # No agent_x here; agents can call predict_puck_trajectory with their own strike plane if needed
        trajectory = self.predict_puck_trajectory()
        return {"position": position, "velocity": velocity, "trajectory": trajectory}

    # ==========================
    # Play Area Detection
    # ==========================
    def _extract_play_area_bounds(self, mask: np.ndarray, tolerance=100):
        """
        Extract outer and inner play area bounds by scanning from the image center.

        Outer bounds are visually detected using rail-to-playfield transitions.
        Inner bounds are computed by shrinking the outer bounds based on puck radius + wall buffer.

        Args:
            mask (np.ndarray): Binary mask of the play area (rails in white, playfield in black).
            tolerance (int): Difference threshold to skip recomputation if mask hasn't changed.

        Returns:
            tuple: (outer_rect, inner_rect), each as (x, y, w, h).
        """
        h, w = mask.shape
        cx, cy = w // 2, h // 2

        # Use cached version if mask hasn't changed significantly
        if hasattr(self, "_last_mask") and hasattr(self, "_cached_bounds"):
            diff = np.sum(cv2.absdiff(self._last_mask, mask))
            #print(f"[DEBUG] Mask difference: {diff}")
            if diff < tolerance:
                #print("[DEBUG] Using cached bounds")
                return self._cached_bounds
            else:
                print("[DEBUG] Mask changed significantly, recalculating bounds")

        print(f"[DEBUG] Mask dimensions: (h={h}, w={w}), Center: (cx={cx}, cy={cy})")
        
        self._last_mask = mask.copy()

        # ----------------------------------------
        # Top bound: scan upward from center
        top = None
        for y in range(cy, 0, -1):
            if mask[y, cx] == 0 and mask[y - 1, cx] > 0:
                top = y
                print(f"[DEBUG] Top bound found at y={top}")
                break

        # Bottom bound: scan downward from center
        bottom = None
        for y in range(cy, h - 1):
            if mask[y, cx] == 0 and mask[y + 1, cx] > 0:
                bottom = y
                print(f"[DEBUG] Bottom bound found at y={bottom}")
                break

        if top is None or bottom is None:
            print("[DEBUG] Failed to find top or bottom bounds")
            return None, None

        # ----------------------------------------
        # Horizontal scan line: just below top bound
        scan_y = top + 10 if top + 10 < h else top
        print(f"[DEBUG] Scanning horizontal bounds at y={scan_y}")

        # Left bound: scan left from center
        left = None
        for x in range(cx, 0, -1):
            if mask[scan_y, x] == 0 and mask[scan_y, x - 1] > 0:
                left = x
                #print(f"[DEBUG] Left bound found at x={left}")
                break

        # Right bound: scan right from center
        right = None
        for x in range(cx, w - 1):
            if mask[scan_y, x] == 0 and mask[scan_y, x + 1] > 0:
                right = x
                #print(f"[DEBUG] Right bound found at x={right}")
                break

        if left is None or right is None:
            print("[DEBUG] Failed to find left or right bounds")
            return None, None


        # ----------------------------------------
        # Construct outer bounding box (pixel coords)
        outer_rect = (left, top, right - left, bottom - top)
        print(f"[DEBUG] Outer bounds: {outer_rect}")

        # Define rail offsets for inner bounds (in pixels)
        rail_offsets = [18, 18, 18, 18]  # [top, bottom, left, right]

        # Apply offsets to calculate inner bounding box
        ix = left + rail_offsets[2]
        iy = top + rail_offsets[0]
        iw = max((right - left) - (rail_offsets[2] + rail_offsets[3]), 1)
        ih = max((bottom - top) - (rail_offsets[0] + rail_offsets[1]), 1)
        inner_rect = (ix, iy, iw, ih)
        print(f"[DEBUG] Inner bounds: {inner_rect}")

        # Cache result
        self._cached_bounds = (outer_rect, inner_rect)
        return outer_rect, inner_rect

    # ==========================
    # Visualization
    # ==========================
    def _display_canvas(self):
        """
        Continuously display the vision sensor feed with overlays for puck position, velocity, and game state.
        """
        print("[DEBUG] Starting display thread...")

        while self.running:
            img = self._get_image()
            if img is None or img.size == 0:
                print("[ERROR] No image data retrieved. Skipping frame.")
                time.sleep(0.1)
                continue

            img_rgb = img
            img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

            # Create masks for black, green, blue, and red items
            black_lower = (0, 0, 0)
            black_upper = (5, 5, 5)
            black_mask = cv2.inRange(img_hsv, black_lower, black_upper)
            black_colored = cv2.merge([black_mask, black_mask, black_mask])  # Black as grayscale

            green_lower = (40, 50, 50)
            green_upper = (80, 255, 255)
            green_mask = cv2.inRange(img_hsv, green_lower, green_upper)
            green_colored = cv2.merge([np.zeros_like(green_mask), green_mask, np.zeros_like(green_mask)])  # Green

            blue_lower = np.array([100, 150, 100])
            blue_upper = np.array([130, 255, 255])
            blue_mask = cv2.inRange(img_hsv, blue_lower, blue_upper)
            blue_colored = cv2.merge([np.zeros_like(blue_mask), np.zeros_like(blue_mask), blue_mask])  # Blue

            red_lower1 = np.array([0, 100, 100])
            red_upper1 = np.array([10, 255, 255])
            red_lower2 = np.array([170, 100, 100])
            red_upper2 = np.array([180, 255, 255])
            red_mask1 = cv2.inRange(img_hsv, red_lower1, red_upper1)
            red_mask2 = cv2.inRange(img_hsv, red_lower2, red_upper2)
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)
            red_colored = cv2.merge([red_mask, np.zeros_like(red_mask), np.zeros_like(red_mask)])  # Red

            # Create a new black mask with bounding rectangles
            bounding_mask = np.zeros_like(black_mask)
            outer_bounds, inner_bounds = self._extract_play_area_bounds(black_mask)

            if outer_bounds:
                x, y, w, h = outer_bounds
                cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (255, 255, 255), 3)  # white, thicker border
                cv2.rectangle(bounding_mask, (x, y), (x + w, y + h), 255, 3)

            if inner_bounds:
                ix, iy, iw, ih = inner_bounds
                cv2.rectangle(img_rgb, (ix, iy), (ix + iw, iy + ih), (255, 255, 0), 3)  # yellow, thicker border
                cv2.rectangle(bounding_mask, (ix, iy), (ix + iw, iy + ih), 255, 3)

            bounding_mask_colored = cv2.merge([bounding_mask, bounding_mask, bounding_mask])

            # Combine colored masks
            combined_colored_mask = cv2.addWeighted(black_colored, 0.5, bounding_mask_colored, 1.0, 0)
            combined_colored_mask = cv2.addWeighted(combined_colored_mask, 1.0, green_colored, 1.0, 0)
            combined_colored_mask = cv2.addWeighted(combined_colored_mask, 1.0, blue_colored, 1.0, 0)
            combined_colored_mask = cv2.addWeighted(combined_colored_mask, 1.0, red_colored, 1.0, 0)

            # Overlay the mask on the original image
            overlay = cv2.addWeighted(img_rgb, 0.7, combined_colored_mask, 0.3, 0)

            # Overlay the center of the board
            board_center_x = img_rgb.shape[1] // 2
            board_center_y = img_rgb.shape[0] // 2
            cv2.circle(img_rgb, (board_center_x, board_center_y), 10, (0, 0, 255), -1)
            cv2.putText(img_rgb, "C", (board_center_x - 15, board_center_y - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Get the puck position and velocity
            puck_pos = self.get_puck_position()
            puck_velocity = self.get_puck_velocity()
            
            # Check if self.referee is initialized
            if self.referee is not None and puck_pos is not None:
                # Check for goals and update scores
                goal_side = self.referee.check_goal()
                if goal_side:
                    print(f"[INFO] Goal scored on the {goal_side} side!")

            # Check for goals using proximity sensors
            if self.referee:
                goal_side = self.referee.check_goal()
                if goal_side:
                    print(f"[INFO] Goal scored on the {goal_side} side!")

            # Overlay game state
            if self.referee:
                self.referee.display_game_state(img_rgb)

            # Overlay puck position
            if puck_pos is not None:
                cx = int(puck_pos[0] / self.pixel_to_meter)
                cy = int(puck_pos[1] / self.pixel_to_meter)
                cv2.circle(img_rgb, (cx, cy), 10, (0, 255, 0), -1)
                cv2.putText(img_rgb, f"Pos: ({puck_pos[0]:.2f}, {puck_pos[1]:.2f})",
                            (cx + 15, cy - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (139, 0, 0), 2)

            # Overlay puck velocity as an arrow
            if puck_velocity:
                vx, vy = puck_velocity
                end_x = int(cx + vx * 50 / self.pixel_to_meter)
                end_y = int(cy + vy * 50 / self.pixel_to_meter)
                cv2.arrowedLine(img_rgb, (cx, cy), (end_x, end_y), (255, 0, 0), 2, tipLength=0.3)
                cv2.putText(img_rgb, f"Vel: ({vx:.2f}, {vy:.2f})",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (139, 0, 0), 2)

            # Stack the original image and the mask image vertically
            combined_view = cv2.vconcat([img_rgb, combined_colored_mask])

            # Show the combined view in a window
            try:
                cv2.imshow("Puck Tracker and Mask View", cv2.cvtColor(combined_view, cv2.COLOR_RGB2BGR))
            except Exception as e:
                print(f"[ERROR] Failed to display image: {e}")
                break

            # Break if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[DEBUG] 'q' pressed. Exiting display thread.")
                break

        print("[DEBUG] Display thread exiting.")
        cv2.destroyAllWindows()

    def update_scores(self, scores):
        """
        Update the scores displayed on the canvas.

        Args:
            scores (dict): A dictionary containing the scores for "left" and "right".
        """
        self.scores = scores

    # ==========================
    # Cleanup
    # ==========================
    def shutdown(self):
        """
        Stop the display thread and clean up resources.
        """
        self.running = False
        self.thread.join()
        cv2.destroyAllWindows()

    def __del__(self):
        """
        Stop the display thread and clean up when the object is destroyed.
        """
        self.shutdown()