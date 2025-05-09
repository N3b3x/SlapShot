import sympy as sp


def compute_inertia_matrices(jacobians, masses):
    inertia_matrices = []

    for J, m in zip(jacobians, masses):
        D = m * J.T * J
        # D = sp.simplify(D)
        inertia_matrices.append(D)
    
    return inertia_matrices


def compute_coriolis_matrix(D, q_syms:list, dq_syms:list):
    '''
    Args:
        D: Inertia Matrix
        q_syms: list of symbolic joint variables
        dq_syms: list of symbolic time derivative of joint variables
    '''
    n = len(q_syms)
    coriolis = sp.Matrix.zeros(n)

    for k in range(n):
        for j in range(n):
            c_kj = 0
            for i in range(n):
                c_ijk = (sp.diff(D[k, j], q_syms[j]) 
                         + sp.diff(D[k, i], q_syms[j])
                         - sp.diff(D[i, j], q_syms[k]))/2
                c_kj += c_ijk * dq_syms[i]
            # c_kj = sp.simplify(c_kj)
            # takes too long to simplify
            coriolis[k, j] = c_kj
    
    return coriolis

def compute_com_wrt_base(center_of_masses, transforms):
    com_wrt_base_list = []
    for com, T in zip(center_of_masses, transforms):
        temp_com = sp.ones(4, 1)
        # make the shape have a one at the bottom to use the transform matrix
        temp_com[:3, :] = com

        com_wrt_base = T * temp_com
        com_wrt_base_list.append(com_wrt_base[:3, :])
    return com_wrt_base_list


def compute_potential_energy(g_vec, m_syms, center_of_masses_wrt_base):
    U_q = sp.zeros(1, 1)
    for com, m in zip(center_of_masses_wrt_base, m_syms):
        U_q += g_vec.T * com * m
    return U_q
    

def compute_gravity_torque_vector(U_q, q_syms):
    n = len(q_syms)
    g_q = sp.ones(n, 1)
    for i in range(n):
        g_q[i] = sp.diff(U_q, q_syms[i])
    return g_q