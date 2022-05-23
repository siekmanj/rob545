import numpy as np
import sympy

sympy.var('t r_1 r_2 L_1 L_2 I_1 I_2 g x_1 y_1 x_2 y_2 m_1 m_2')

θ_1 = sympy.Function('θ_1')('t')
θ_2 = sympy.Function('θ_2')('t')

x_1 = sympy.sin(θ_1) * r_1
y_1 = -sympy.cos(θ_2) * r_1

x_2 = sympy.sin(θ_1) * L_1 + sympy.sin(θ_2) * r_2
y_2 = -sympy.cos(θ_1) * L_1 - sympy.cos(θ_2) * r_2

K = 0.5 * m_1 * (sympy.diff(x_1, t)**2 + sympy.diff(y_1, t)**2) \
    + 0.5 * m_2 * (sympy.diff(x_2, t)**2 + sympy.diff(y_2, t)**2) \
    + 0.5 * I_1 * sympy.diff(θ_1, t) ** 2 \
    + 0.5 * I_2 * sympy.diff(θ_2, t) ** 2

P = g * (m_1 * y_1 + m_2 * y_2)

Lagrange = K - P

dL_ddotθ_1 = sympy.diff(Lagrange, sympy.diff(θ_1, t))
dL_ddotθ_2 = sympy.diff(Lagrange, sympy.diff(θ_2, t))

tau_1 = sympy.diff(dL_ddotθ_1, t) - sympy.diff(Lagrange, θ_1)
tau_2 = sympy.diff(dL_ddotθ_2, t) - sympy.diff(Lagrange, θ_2)

sympy.pprint(tau_1)

#tau_1 = sympy.diff(d) - sympy.diff(Lagrange, θ_1)
#tau_2 = sympy.diff(sympy.diff(Lagrange, sympy.diff(θ_2, t), t)) - sympy.diff(Lagrange, θ_2)

#sympy.pprint(tau_1)

def generate_quintic_traj(q_initial, qdot_initial, qddot_initial, q_final, qdot_final, qddot_final, T):
    # Matrix inverse to get the a_i coefficients
    a = np.linalg.inv([
        [0, 0, 0, 0, 0, 1],                          # S(0) = 0
        [T**5, T**4, T**3, T**2, T, 1],              # S(T) = 1
        [0, 0, 0, 0, 1, 0],                          # dS(0) = 0
        [5 * T**4, 4 * T**3, 3 * T**2, 2 * T, 1, 0], # dS(T) = 0
        [0, 0, 0, 2, 0, 0],                          # ddS(0) = 0
        [20 * T**3, 12 * T**2, 6*T, 2, 0, 0],        # ddS(T) = 0
    ]) @ np.array([0, 1, 0, 0, 0, 0])

    def _fn(t):
        s_t = np.sum([a[i] * t**(i+1) for i in range(5)])
        ds_t = np.sum([(i+1) * a[i] * t**(i) for i in range(5)])
        dds_t = np.sum([(i+1)*(i+2) * a[i] * t**(i) for i in range(5)])
        return 1

generate_quintic_traj(None,None,None,None,None,None, 2)

