import numpy as np
import time
from modules.kinematics import Ur3Solver
from scipy.spatial.transform import Rotation

class StrikePlanner:
    def __init__(self, sim, effector_handle, num_joints=6):
        self.sim = sim
        self.effector = effector_handle

        self.table_width = 0.7
        self.table_length = 1.2 # x direction
        self.puck_radius = 0.075
        self.defense_line = 0.2
        self.num_joints = num_joints

        self.ik_solver = Ur3Solver()

    def predict_intercept(
            self, 
            puck_pos, 
            puck_vel, 
            table_width, 
            defense_line):
        
        dx = defense_line - puck_pos[0]
        if puck_vel[0] == 0:
            # just wait for the puck 
            return np.array([defense_line, puck_pos[1]])

        t = dx / puck_vel[0]

        y_raw = puck_pos[1] + puck_vel[1] * t

        period = 2 * table_width
        y_mod = y_raw % period

        if y_mod <= table_width:
            y_intercept = y_mod
        else:
            y_intercept = period - y_mod

        # always guards across defense line
        return np.array([defense_line, y_intercept])


    def extract_data_from_sim(self, puck_handle):
        puck_pos = self.sim.getObjectPosition(puck_handle, self.effector)
        puck_pos = np.array(puck_pos[:2])
        puck_vel, _ = self.sim.getObjectVelocity(puck_handle)
        return puck_pos, puck_vel


    def plan_strike(self, puck_pos, puck_vel):
        # dx = curr_pos[0] - prev_pos[0]
        # dy = curr_pos[1] - prev_pos[1]
        # speed = np.hypot(dx, dy)

        # if speed < 0.01:
        #     return None, None

        # strike_x = curr_pos[0] + dx * 30
        # strike_y = curr_pos[1] + dy * 30
        # strike_pos = [strike_x - 0.5, strike_y, 0.22]
        # velocity = [dx * 5, dy * 5, 0]
        # return strike_pos, velocity

        target_x, target_y = self.predict_intercept(
            puck_pos, puck_vel, self.table_width, self.defense_line)
        
        target_x = self.defense_line
        target_y = 0
        
        target_orientation = Rotation.from_euler(
            'XYZ', [0, 0, 0], degrees=True).as_matrix()

        target_pos = [target_x, target_y, 0]
        joint_angles = self.ik_solver.perform_ik( target_pos, target_orientation)

        if joint_angles is None:
            print(f'No solution found for target pos:\n{target_pos}\n{target_orientation}')
        
        return joint_angles
        

    def execute_strike(self, joint_angles):
        # self.sim.setObjectPosition(self.effector, -1, position)
        # time.sleep(0.3)
        # self.sim.setObjectVelocity(self.effector, velocity, [0, 0, 0])
        # print(f"[ACTION] Strike at {position} with {velocity}")
        max_ang_vel = np.deg2rad(180)
        max_ang_accel = np.deg2rad(40)
        max_ang_jerk = np.deg2rad(80)

        alias = self.sim.getObjectAlias(self.effector, 2)
        self.sim.moveToConfig({
            'joints': [self.sim.getObject(f'{alias}/joint', {'index': idx}) 
                    for idx in range(self.num_joints)],
            'maxVel': self.num_joints * [max_ang_vel],
            'maxAccel': self.num_joints * [max_ang_accel],
            'maxJerk': self.num_joints * [max_ang_jerk],
            'targetPos': joint_angles
        })


    