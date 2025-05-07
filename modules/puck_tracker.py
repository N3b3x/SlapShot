import numpy as np
import cv2
import threading
import time
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
    def __init__(self, sim, camera_handle, goal_width):
        """
        Initialize the PuckTracker.

        Args:
            sim: The simulation object for interacting with the environment.
            camera_handle: The handle to the vision sensor in the simulation.
            goal_width: The width of the goal opening.
        """
        self.sim = sim
        self.camera = camera_handle
        self.pixel_to_meter = 1.2 / 640  # Conversion factor for pixels to meters
        self.prev_positions = []  # Store previous puck positions for velocity calculation
        self.trail = []  # Store puck trail points with timestamps
        self.trail_duration = 2.0  # Duration (in seconds) to keep trail points
        self.running = True  # Control flag for the display thread
        self.lock = threading.Lock()  # Lock for thread-safe access to the camera
        self.thread = threading.Thread(target=self._display_canvas, daemon=True)
        self.thread.start()  # Start the display thread
        self.referee = None  # GameReferee will be initialized after bounds are extracted
        self.goal_width = goal_width

        # Initialize bounds as None; they will be dynamically extracted
        self.outer_bounds = None
        self.inner_bounds = None

    def extract_court_bounds(self):
        """
        Dynamically extract the court bounds from the camera feed.

        Returns:
            tuple: (outer_bounds, inner_bounds) in pixel coordinates.
        """
        img = self._get_image()
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        # Black mask for detecting the play area
        black_lower = (0, 0, 0)
        black_upper = (5, 5, 5)
        black_mask = cv2.inRange(img_hsv, black_lower, black_upper)

        # Extract bounds
        self.outer_bounds, self.inner_bounds = self._extract_play_area_bounds(black_mask, rail_offset=(18, 18, 25, 25))

        # Convert bounds to meters
        if self.outer_bounds and self.inner_bounds:
            self.outer_bounds = tuple(coord * self.pixel_to_meter for coord in self.outer_bounds)
            self.inner_bounds = tuple(coord * self.pixel_to_meter for coord in self.inner_bounds)

            # Initialize the referee with the extracted bounds
            self.referee = GameReferee({"outer": self.outer_bounds, "inner": self.inner_bounds}, self.goal_width)

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
        Retrieve the current RGB image from the vision sensor.

        Returns:
            np.ndarray: The RGB image from the vision sensor.
        """
        try:
            with self.lock:  # Ensure thread-safe access
                raw_bytes, resolution = self.sim.getVisionSensorImg(self.camera)
                flat_img = self.sim.unpackUInt8Table(raw_bytes)  # Unpack raw image data
                w, h = resolution
                img = np.array(flat_img, dtype=np.uint8).reshape((h, w, 3))  # Reshape into RGB image
                return img
        except Exception as e:
            print(f"[ERROR] Failed to get image: {e}")
            return np.zeros((480, 640, 3), dtype=np.uint8)  # Dummy fallback

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
            print(f"[DEBUG] Detected puck position (pixels): ({cx}, {cy})")
            print(f"[DEBUG] Detected puck position (meters): {pos}")

            # Add to trail with current timestamp
            self.trail.append((pos[0], pos[1], time.time()))

            # Apply a moving average filter
            self.prev_positions.append(pos)
            if len(self.prev_positions) > 5:  # Keep the last 5 positions
                self.prev_positions.pop(0)
            avg_pos = np.mean(self.prev_positions, axis=0)

            # Debugging information
            print(f"[DEBUG] Filtered puck position (meters): {avg_pos}")
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
        dx = self.prev_positions[-1][0] - self.prev_positions[-2][0]
        dy = self.prev_positions[-1][1] - self.prev_positions[-2][1]
        return dx, dy

    # ==========================
    # Play Area Detection
    # ==========================
    def _extract_play_area_bounds(self, mask: np.ndarray, rail_offset=(20, 20, 20, 20)):
        """
        Extract the outer and inner bounding rectangles of the play area.

        Args:
            mask (np.ndarray): Binary mask of the play area (rails in black).
            rail_offset (tuple): Offsets for shrinking the outer rectangle inward as 
                                 (top, bottom, left, right).

        Returns:
            tuple: (outer_rect, inner_rect), each as (x, y, w, h).
        """
        # Close gaps in the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        cleaned = cv2.dilate(cleaned, kernel, iterations=1)

        # Combine all contours into one
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None

        # Combine all contours into a single mask
        combined_mask = np.zeros_like(mask)
        cv2.drawContours(combined_mask, contours, -1, 255, thickness=cv2.FILLED)

        # Get the bounding rectangle of the combined mask
        x, y, w, h = cv2.boundingRect(cv2.findNonZero(combined_mask))
        outer_rect = (x, y, w, h)

        # Shrink inward by rail_offset to get the inner playable area
        top_offset, bottom_offset, left_offset, right_offset = rail_offset
        ix = x + left_offset
        iy = y + top_offset
        iw = max(w - (left_offset + right_offset), 1)
        ih = max(h - (top_offset + bottom_offset), 1)
        inner_rect = (ix, iy, iw, ih)

        return outer_rect, inner_rect

    # ==========================
    # Visualization
    # ==========================
    def _display_canvas(self):
        """
        Continuously display the vision sensor feed with overlays for puck position, velocity, and game state.
        """
        while self.running:
            img = self._get_image()
            img_rgb = img
            img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

            # Create masks for black, green, and blue items
            # Black mask (very dark black range)
            black_lower = (0, 0, 0)
            black_upper = (5, 5, 5)
            black_mask = cv2.inRange(img_hsv, black_lower, black_upper)
            black_colored = cv2.merge([black_mask, black_mask, black_mask])  # Black as grayscale

            # Create a new black mask with bounding rectangles
            bounding_mask = np.zeros_like(black_mask)
            outer_bounds, inner_bounds = self._extract_play_area_bounds(black_mask, rail_offset=(18, 18, 25, 25))
            
            if outer_bounds:
                x, y, w, h = outer_bounds
                cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (255, 255, 255), 3)  # white, thicker border
                cv2.rectangle(bounding_mask, (x, y), (x + w, y + h), 255, 3)

            if inner_bounds:
                ix, iy, iw, ih = inner_bounds
                cv2.rectangle(img_rgb, (ix, iy), (ix + iw, iy + ih), (255, 255, 0), 3)  # yellow, thicker border
                cv2.rectangle(bounding_mask, (ix, iy), (ix + iw, iy + ih), 255, 3)

            # Ensure bounding_mask matches the dimensions and channels of black_colored
            bounding_mask_colored = cv2.merge([bounding_mask, bounding_mask, bounding_mask])

            # Green mask
            green_lower = (40, 50, 50)
            green_upper = (80, 255, 255)
            green_mask = cv2.inRange(img_hsv, green_lower, green_upper)
            green_colored = cv2.merge([np.zeros_like(green_mask), green_mask, np.zeros_like(green_mask)])  # Green

            # Blue mask
            blue_lower = np.array([100, 150, 100])  # Adjusted for a brighter blue
            blue_upper = np.array([130, 255, 255])
            blue_mask = cv2.inRange(img_hsv, blue_lower, blue_upper)
            blue_colored = cv2.merge([np.zeros_like(blue_mask), np.zeros_like(blue_mask), blue_mask])  # Blue

            # Red mask (accounting for HSV wraparound)
            red_lower1 = np.array([0, 100, 100])
            red_upper1 = np.array([10, 255, 255])
            red_lower2 = np.array([170, 100, 100])
            red_upper2 = np.array([180, 255, 255])
            red_mask1 = cv2.inRange(img_hsv, red_lower1, red_upper1)
            red_mask2 = cv2.inRange(img_hsv, red_lower2, red_upper2)
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)
            red_colored = cv2.merge([red_mask, np.zeros_like(red_mask), np.zeros_like(red_mask)])  # Red

            # Combine colored masks
            combined_colored_mask = cv2.addWeighted(black_colored, 0.2, bounding_mask_colored, 1.0, 0)
            combined_colored_mask = cv2.addWeighted(combined_colored_mask, 1.0, green_colored, 1.0, 0)
            combined_colored_mask = cv2.addWeighted(combined_colored_mask, 1.0, blue_colored, 1.0, 0)
            combined_colored_mask = cv2.addWeighted(combined_colored_mask, 1.0, red_colored, 1.0, 0)

            # Overlay the mask on the original image
            overlay = cv2.addWeighted(img_rgb, 0.7, combined_colored_mask, 0.3, 0)

            # Overlay the center of the board
            board_center_x = img_rgb.shape[1] // 2  # Middle of the image width
            board_center_y = img_rgb.shape[0] // 2  # Middle of the image height
            cv2.circle(img_rgb, (board_center_x, board_center_y), 10, (0, 0, 255), -1)  # Draw the board center
            cv2.putText(img_rgb, "C",
                        (board_center_x - 15, board_center_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Get the puck position and velocity
            puck_pos = self.get_puck_position()
            puck_velocity = self.get_puck_velocity()

            # Check for goals and update scores
            goal_side = self.referee.check_goal(puck_pos)
            if goal_side:
                print(f"[INFO] Goal scored on the {goal_side} side!")

            # Overlay game state
            self.referee.display_game_state(img_rgb)

            # Overlay puck position
            if puck_pos is not None:
                cx = int(puck_pos[0] / self.pixel_to_meter)  # Convert back to pixels
                cy = int(puck_pos[1] / self.pixel_to_meter)
                cv2.circle(img_rgb, (cx, cy), 10, (0, 255, 0), -1)  # Draw the puck

                # Display puck position as text
                cv2.putText(img_rgb, f"Pos: ({puck_pos[0]:.2f}, {puck_pos[1]:.2f})",
                            (cx + 15, cy - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (139, 0, 0), 2)

                # Prune trail: keep only points newer than trail_duration
                now = time.time()
                self.trail = [(x, y, t) for x, y, t in self.trail if now - t <= self.trail_duration]

                # Convert to pixel coordinates
                trail_points = [
                    (int(x / self.pixel_to_meter), int(y / self.pixel_to_meter), t)
                    for x, y, t in self.trail
                ]

                # Draw fading polyline segment-by-segment
                for i in range(1, len(trail_points)):
                    (x1, y1, t1) = trail_points[i - 1]
                    (x2, y2, t2) = trail_points[i]
                    
                    # Use average age between two points
                    age = (now - t1 + now - t2) / 2
                    alpha = 1.0 - min(age / self.trail_duration, 1.0)  # fade 1 → 0

                    # Color gradient: blue (old) to green (new)
                    r = 0
                    g = int(255 * alpha)
                    b = int(255 * (1 - alpha))
                    color = (b, g, r)

                    cv2.line(img_rgb, (x1, y1), (x2, y2), color, thickness=2)

            # Overlay puck velocity as an arrow
            if puck_velocity:
                vx, vy = puck_velocity
                end_x = int(cx + vx * 50 / self.pixel_to_meter)  # Scale velocity for visualization
                end_y = int(cy + vy * 50 / self.pixel_to_meter)
                cv2.arrowedLine(img_rgb, (cx, cy), (end_x, end_y), (255, 0, 0), 2, tipLength=0.3)  # Draw arrow
                cv2.putText(img_rgb, f"Vel: ({vx:.2f}, {vy:.2f})",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (139, 0, 0), 2)

            # Stack the original image and the mask image vertically
            combined_view = cv2.vconcat([img_rgb, combined_colored_mask])

            # Show the combined view in a window
            cv2.imshow("Puck Tracker and Mask View", cv2.cvtColor(combined_view, cv2.COLOR_RGB2BGR))

            # Break if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

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