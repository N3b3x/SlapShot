import numpy as np
import cv2

class PuckTracker:
    def __init__(self, sim, camera_handle):
        self.sim = sim
        self.camera = camera_handle
        self.pixel_to_meter = 1.2 / 640  # Conversion factor
        self.prev_positions = []  # Store previous positions for filtering

    def get_puck_position(self):
        image, res_x, res_y = self.sim.getVisionSensorCharImage(self.camera)
        img = np.frombuffer(image, dtype=np.uint8).reshape((res_y, res_x, 3))
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))  # Red color range
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
        if len(self.prev_positions) < 2:
            return None
        dx = self.prev_positions[-1][0] - self.prev_positions[-2][0]
        dy = self.prev_positions[-1][1] - self.prev_positions[-2][1]
        return dx, dy