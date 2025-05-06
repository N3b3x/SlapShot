import sympy as sp

# https://www.universal-robots.com/articles/ur/application-installation/dh-parameters-for-calculations-of-kinematics-and-dynamics/

q_syms = sp.symbols('q1:7')


ur5_dh_table = [
    {'theta':q_syms[0], 'a':0, 'd':0.1625, 'alpha':sp.pi/2},
    {'theta':q_syms[1], 'a':-0.425, 'd':0, 'alpha':0},
    {'theta':q_syms[2], 'a':-0.3922, 'd':0, 'alpha':0},
    {'theta':q_syms[3], 'a':0, 'd':0.1333, 'alpha':sp.pi/2},
    {'theta':q_syms[4], 'a':0, 'd':0.0997, 'alpha':-sp.pi/2},
    {'theta':q_syms[5], 'a':0, 'd':0.0996, 'alpha':0},
]

ur3_dh_table = [
    {'theta':q_syms[0], 'a':0, 'd':0.1519, 'alpha':sp.pi/2},
    {'theta':q_syms[1], 'a':-0.24365, 'd':0, 'alpha':0},
    {'theta':q_syms[2], 'a':-0.21325, 'd':0, 'alpha':0},
    {'theta':q_syms[3], 'a':0, 'd':0.11235, 'alpha':sp.pi/2},
    {'theta':q_syms[4], 'a':0, 'd':0.08535, 'alpha':-sp.pi/2},
    {'theta':q_syms[5], 'a':0, 'd':0.0819, 'alpha':0},
]