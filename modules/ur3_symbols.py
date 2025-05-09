import sympy as sp

q_syms = sp.symbols('q1:7')
a_syms = sp.symbols('a1:7')
d_syms = sp.symbols('d1:7')


dq1 = sp.Symbol(r'\dot{q_1}')
dq2 = sp.Symbol(r'\dot{q_2}')
dq3 = sp.Symbol(r'\dot{q_3}')
dq4 = sp.Symbol(r'\dot{q_4}')
dq5 = sp.Symbol(r'\dot{q_5}')
dq6 = sp.Symbol(r'\dot{q_6}')
dq_syms = (dq1, dq2, dq3, dq4, dq5, dq6)

theta_syms = sp.symbols('theta1:7')

m_syms = sp.symbols('m1:7')

r_c0 = sp.zeros(1, 3)

r_c1x = sp.Symbol(r'r_{c_1,x}')
r_c1y = sp.Symbol(r'r_{c_1,y}')
r_c1z = sp.Symbol(r'r_{c_1,z}')
r_c1 = sp.Matrix([r_c1x, r_c1y, r_c1z])

r_c2x = sp.Symbol(r'r_{c_2,x}')
r_c2y = sp.Symbol(r'r_{c_2,y}')
r_c2z = sp.Symbol(r'r_{c_2,z}')
r_c2 = sp.Matrix([r_c2x, r_c2y, r_c2z])

r_c3x = sp.Symbol(r'r_{c_3,x}')
r_c3y = sp.Symbol(r'r_{c_3,y}')
r_c3z = sp.Symbol(r'r_{c_3,z}')
r_c3 = sp.Matrix([r_c3x, r_c3y, r_c3z])

r_c4x = sp.Symbol(r'r_{c_4,x}')
r_c4y = sp.Symbol(r'r_{c_4,y}')
r_c4z = sp.Symbol(r'r_{c_4,z}')
r_c4 = sp.Matrix([r_c4x, r_c4y, r_c4z])

r_c5x = sp.Symbol(r'r_{c_5,x}')
r_c5y = sp.Symbol(r'r_{c_5,y}')
r_c5z = sp.Symbol(r'r_{c_5,z}')
r_c5 = sp.Matrix([r_c5x, r_c5y, r_c5z])

r_c6x = sp.Symbol(r'r_{c_6,x}')
r_c6y = sp.Symbol(r'r_{c_6,y}')
r_c6z = sp.Symbol(r'r_{c_6,z}')
r_c6 = sp.Matrix([r_c6x, r_c6y, r_c6z])
