from modules.kinematics import *
import numpy as np
import sympy as sp

from modules.dh_tables import ur3_dh_table, q_syms

paddle_start_pos = np.array([0.5023243616767452, 0.10890051446054394, 0.08219026213745537])

paddle_goal_orientation = np.array([7.04317383600834e-05, 0.0009407205501931991, 1.570929925127943])

target_orientation = Rotation.from_euler(
    'XYZ', paddle_goal_orientation, degrees=False).as_matrix()

print(target_orientation)


transforms = get_forward_kinematics(ur3_dh_table)

T_60_sym = transforms[-1]
T_60_func = lambdify(q_syms, T_60_sym, modules='numpy')

q_expected = np.deg2rad([-90, 60, 45, -15, -90, 90])

T_60 = T_60_func(*q_expected)
print(T_60)

print(Rotation.from_matrix(T_60[:3, :3]).as_euler('XYZ'))
# target_pos = paddle_start_pos


# ik_solver = Ur3Solver()
# joint_angles = ik_solver.perform_ik( target_pos, target_orientation)

# print(joint_angles)