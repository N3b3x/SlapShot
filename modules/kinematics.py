import numpy as np
import sympy as sp
from sympy import symbols
import modules.dh_tables
from scipy.spatial.transform import Rotation

def get_transform(a, alpha, d, theta):
    return sp.Matrix([
        [sp.cos(theta), -sp.sin(theta) * sp.cos(alpha), sp.sin(theta) * sp.sin(alpha), a * sp.cos(theta)],
        [sp.sin(theta),  sp.cos(theta) * sp.cos(alpha), -sp.cos(theta) * sp.sin(alpha), a * sp.sin(theta)],
        [0,              sp.sin(alpha), sp.cos(alpha), d],
        [0,              0, 0, 1]
    ])

def get_forward_kinematics(dh_table):

    T = sp.eye(4)
    transforms = [T]
    for link in dh_table:
        T = T * get_transform(link['a'], link['alpha'], link['d'], link['theta'])
        transforms.append(T)
    
    return transforms

def build_symbolic_com_jacobians(origins, z_vectors, com_positions):
    """
    Constructs the Jacobian matrices (linear + angular) for the centers of mass of each link.

    Parameters:
    - origins: list of sympy Matrix, position of each joint frame (length N+1)
    - z_vectors: list of sympy Matrix, z-axis of each joint frame (length N)
    - com_positions: list of sympy Matrix, center of mass positions for each link (length N)

    Returns:
    - jacobians: list of sympy Matrix, each 6xN Jacobian for a link's CoM
    """
    num_joints = len(z_vectors)
    jacobians = []

    for link_idx in range(num_joints):
        J = sp.zeros(3, num_joints)
        o_c = com_positions[link_idx]
        for j in range(link_idx + 1):
            o_n = origins[j]
            z_n = z_vectors[j]
            J[:3, j] = sp.simplify(z_n.cross(o_c - o_n) ) # Linear velocity Jacobian
            # J[3:, j] = z_n                   # Angular velocity Jacobian
        jacobians.append(J)

    return jacobians


class Ur3Solver:
    def __init__(self):
        
        self.dh_table = dh_tables.ur3_dh_table

    def perform_position_ik(self, X_ee, R_ee):
        
        d6 = self.dh_table[5]['d']
        X_wrist = X_ee - d6 * R_ee[:, 2]
        x, y = X_wrist[0], X_wrist[1]
        
        q1 = np.atan2(y, x)

        a1 = self.dh_table[0]['d']
        a2 = self.dh_table[1]['a']
        a3 = self.dh_table[2]['a']

        # project in 2d based on q1

        r = np.sqrt(x**2 + y**2)
        z = X_wrist[2] - a1
        # law of cosines
        D = (r**2 + z**2 - a2**2 - a3**2)/ (2 * a2 * a3)
        if abs(D) > 1: 
            return None

        q3 = np.atan2(np.sqrt(1 - D**2), D) # elbow up
        q2 = np.atan2(z, r) - np.atan2(a3*np.sin(q3), a2 + a3 * np.cos(q3))
        
        sol = (q1, q2, q3)
        return sol
        

    def perform_orientation_ik(self, X_ee, R_ee, q1):
        
        R_30 = Rotation.from_euler('Z', q1).as_matrix()
        R_63 = R_30.T @ R_ee
        R_63 = Rotation.from_matrix(R_63)
        [q4, q5, q6] = R_63.as_euler('ZYZ')
        return q4, q5, q6
        

    def perform_ik(self, X_ee, R_ee):
        
        sol = self.perform_position_ik(X_ee, R_ee)
        if sol is None:
            return None
        q1, q2, q3 = self.perform_position_ik(X_ee, R_ee)
        q4, q5, q6 = self.perform_orientation_ik(X_ee, R_ee, q1)
        
        return q1, q2, q3, q4, q5, q6
        

# # Create hockey table
# # Define table dimensions and z height as variables
# table_length = 1.2
# table_width = 0.7
# table_height = 0.02
# z_height = 0.01  # Table's base height from the ground
# table_top_z = z_height + table_height  # Top of the table

# robot_offset = 0.2  # Distance behind the goal
# robot_z = table_top_z  # On table


# ur3_ik_solver = Ur3Solver()
# left_paddle_start_pos = [-table_length / 4, 0, robot_z]
# left_paddle_orientation = Rotation.from_euler('Y', 90, degrees=True).as_matrix()
# sol = ur3_ik_solver.perform_ik(left_paddle_start_pos, left_paddle_orientation)
# print(list(sol))