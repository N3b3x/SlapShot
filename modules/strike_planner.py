import numpy as np
import time

class StrikePlanner:
    def __init__(self, sim, effector_handle):
        self.sim = sim
        self.effector = effector_handle

    def plan_strike(self, prev_pos, curr_pos):
        dx = curr_pos[0] - prev_pos[0]
        dy = curr_pos[1] - prev_pos[1]
        speed = np.hypot(dx, dy)

        if speed < 0.01:
            return None, None

        strike_x = curr_pos[0] + dx * 30
        strike_y = curr_pos[1] + dy * 30
        strike_pos = [strike_x - 0.5, strike_y, 0.22]
        velocity = [dx * 5, dy * 5, 0]
        return strike_pos, velocity

    def execute_strike(self, position, velocity):
        self.sim.setObjectPosition(self.effector, -1, position)
        time.sleep(0.3)
        self.sim.setObjectVelocity(self.effector, velocity, [0, 0, 0])
        print(f"[ACTION] Strike at {position} with {velocity}")