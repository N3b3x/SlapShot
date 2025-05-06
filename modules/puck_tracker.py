import numpy as np
import cv2
import threading


class PuckTracker:
    def __init__(self, sim, camera_handle):
        self.sim = sim
        self.camera = camera_handle
        self.pixel_to_meter = 1.2 / 640  # Conversion factor
        self.prev_positions = []  # Store previous positions for filtering
        self.running = True  # To control the display thread
        self.lock = threading.Lock()  # Lock for thread safety
        self.thread = threading.Thread(target=self._display_canvas, daemon=True)
        self.thread.start()  # Start the display thread

    def _get_image(self):
        """
        Retrieve the current image from the vision sensor.
        """
        try:
            with self.lock:  # Ensure thread-safe access
                image, resolution = self.sim.getVisionSensorImg(self.camera)
                img = np.frombuffer(image, dtype=np.uint8).reshape((resolution[1], resolution[0], 3))
                img = img.copy()  # Create a writable copy of the array
                return img
        except Exception as e:
            print(f"[ERROR] Failed to get image: {e}")
            return np.zeros((480, 640, 3), dtype=np.uint8)  # Dummy fallback

    def get_puck_position(self):
        """
        Get the position of the puck (in Cartesian coordinates).
        """
        img = self._get_image()

        # Convert to HSV to detect the red puck
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))  # Red color range

        # Find the center of the puck using image moments
        moments = cv2.moments(mask)
        if moments["m00"] > 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            pos = (cx * self.pixel_to_meter, cy * self.pixel_to_meter)

            # Apply a moving average filter
            self.prev_positions.append(pos)
            if len(self.prev_positions) > 5:  # Keep the last 5 positions
                self.prev_positions.pop(0)
            avg_pos = np.mean(self.prev_positions, axis=0)
            return avg_pos
        return None

    def get_puck_velocity(self):
        """
        Calculate the puck's velocity based on its previous positions.
        """
        if len(self.prev_positions) < 2:
            return None
        dx = self.prev_positions[-1][0] - self.prev_positions[-2][0]
        dy = self.prev_positions[-1][1] - self.prev_positions[-2][1]
        return dx, dy

    def _display_canvas(self):
        """
        Continuously display the vision sensor feed with overlays for puck position and velocity.
        """
        while self.running:
            img = self._get_image()

            # Get the puck position and velocity
            puck_pos = self.get_puck_position()
            puck_velocity = self.get_puck_velocity()

            # Overlay the center of the board
            board_center_x = img.shape[1] // 2  # Middle of the image width
            board_center_y = img.shape[0] // 2  # Middle of the image height
            cv2.circle(img, (board_center_x, board_center_y), 10, (0, 0, 255), -1)  # Draw the board center
            cv2.putText(img, "C",
                        (board_center_x - 15, board_center_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Overlay puck position
            if puck_pos is not None:
                cx = int(puck_pos[0] / self.pixel_to_meter)  # Convert back to pixels
                cy = int(puck_pos[1] / self.pixel_to_meter)
                cv2.circle(img, (cx, cy), 10, (0, 255, 0), -1)  # Draw the puck

                # Display puck position as text
                cv2.putText(img, f"Pos: ({puck_pos[0]:.2f}, {puck_pos[1]:.2f})",
                            (cx + 15, cy - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (139, 0, 0), 2)

            # Overlay puck velocity as an arrow
            if puck_velocity:
                vx, vy = puck_velocity
                end_x = int(cx + vx * 50 / self.pixel_to_meter)  # Scale velocity for visualization
                end_y = int(cy + vy * 50 / self.pixel_to_meter)
                cv2.arrowedLine(img, (cx, cy), (end_x, end_y), (255, 0, 0), 2, tipLength=0.3)  # Draw arrow
                cv2.putText(img, f"Vel: ({vx:.2f}, {vy:.2f})",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (139, 0, 0), 2)

            # Show the image in a window
            cv2.imshow("Puck Tracker", img)

            # Break if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

    def __del__(self):
        """
        Stop the display thread and clean up when the object is destroyed.
        """
        self.running = False
        self.thread.join()