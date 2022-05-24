import numpy as np
import sympy

def pendulum_dynamics(
            _L_1: float = 1,
            _L_2: float = 1,
            _m_1: float = 3,
            _m_2: float = 2,
            _I_1: float = 2,
            _I_2: float = 1,
            _r_1: float = 0.5,
            _r_2: float = 0.5,
            _g: float = 9.81,
):
    sympy.var('t r_1 r_2 L_1 L_2 I_1 I_2 m_1 m_2 g tau_1 tau_2')

    θ_1 = sympy.Function('θ_1')('t')
    θ_2 = sympy.Function('θ_2')('t')

    x_1 = sympy.sin(θ_1) * r_1
    y_1 = -sympy.cos(θ_1) * r_1

    x_2 = sympy.sin(θ_1) * L_1 + sympy.sin(θ_1 + θ_2) * r_2
    y_2 = -sympy.cos(θ_1) * L_1 - sympy.cos(θ_1 + θ_2) * r_2

    # Our kinetic energy
    K = 0.5 * m_1 * (sympy.diff(x_1, t)**2 + sympy.diff(y_1, t)**2) \
        + 0.5 * m_2 * (sympy.diff(x_2, t)**2 + sympy.diff(y_2, t)**2) \
        + 0.5 * I_1 * m_1 * sympy.diff(θ_1, t) ** 2 \
        + 0.5 * I_2 * m_2 * sympy.diff(θ_2, t) ** 2

    # Potential Energy
    P = g * (m_1 * y_1 + m_2 * y_2)

    Lagrange = K - P

    from sympy.physics.mechanics import LagrangesMethod
    LM = LagrangesMethod(Lagrange, [θ_1, θ_2])
    
    full = LM.form_lagranges_equations()
    
    tau = sympy.Matrix([tau_1, tau_2])
    qdd = sympy.Matrix([sympy.diff(sympy.diff(θ_1, t), t), sympy.diff(sympy.diff(θ_2, t), t)])
    q_ddot = LM.mass_matrix.inv() * (LM.forcing + tau)

    # Substitute all the model parameters here for efficiency
    subs = [
        (L_1, _L_1),
        (L_2, _L_2),
        (m_1, _m_1),
        (m_2, _m_2),
        (I_1, _I_1),
        (I_2, _I_2),
        (r_1, _r_1),
        (r_2, _r_2),
        (g, _g),
    ]
    q_ddot = q_ddot.subs(subs)

    from sympy.utilities.lambdify import lambdify
    q_ddot_fast  = lambdify(
        [θ_1, θ_2, sympy.diff(θ_1, t), sympy.diff(θ_2, t), tau_1, tau_2],
        q_ddot,
    )

    def compute_accelerations(
            torques,
            q,
            q_dot,
    ):
        """
        Helper function to calculate joint accelerations given concrete
        values for all the variables in the equations of motion
        """
        assert len(torques) == 2
        assert len(q) == 2
        assert len(q_dot) == 2

        ret = q_ddot_fast(*q, *q_dot, *torques)
        return np.array(ret).flatten()

    return compute_accelerations

def animate_pendulum(q, L1: float = 1, L2: float = 1):
   import matplotlib.animation as animation
   import matplotlib.pyplot as plt
   assert len(q.shape) == 2
   assert q.shape[-1] == 2
   
   plt.rcParams["figure.figsize"] = 8,6

   fig, ax = plt.subplots()

   ax.axis([-3,3,-2,2])
   
   base, = ax.plot(0, 0, marker="o")
   p1, = ax.plot(0, 1, marker="o")
   p2, = ax.plot(0, 1, marker="o")
   link1, = ax.plot([], [], color="crimson", zorder=4)
   link2, = ax.plot([], [], color="crimson", zorder=4)

   def update(t):
       t = int(t)
       theta_1, theta_2 = q[t]

       x_1 = np.sin(theta_1) * L1
       y_1 = -np.cos(theta_1) * L1

       x_2 = x_1 + np.sin(theta_1 + theta_2) * L2
       y_2 = y_1 - np.cos(theta_1 + theta_2) * L2
       
       p1.set_data([x_1], [y_1])
       p2.set_data([x_2], [y_2])
       link1.set_data([0, x_1], [0, y_1])
       link2.set_data([x_1, x_2], [y_1, y_2])

       return p1, p2, link1, link2

   ani = animation.FuncAnimation(fig, update, interval=10, blit=True, repeat=True,
                       frames=np.linspace(0, len(q), num=len(q), endpoint=False))
   #writer = animation.writers['ffmpeg'](fps=15, metadata=dict(artist='Me'), bitrate=1800)
   #ani.save('traj.mp4', writer=writer)
   #return HTML(ani.to_jshtml()) 
   plt.show()

def generate_quintic_traj(
        q_initial,
        q_dot_initial,
        q_ddot_initial,
        q_final,
        q_dot_final,
        q_ddot_final,
        T,
):
    # Matrix inverse to get the a_i coefficients
    a = np.linalg.inv([
        [1, 0, 0, 0, 0, 0],                          # S(0) = 0
        [1, T**1, T**2, T**3, T**4, T**5],           # S(T) = 1
        [0, 1, 0, 0, 0, 0],                          # dS(0) = 0
        [0, 1, 2 * T, 3 * T**2, 4 * T**3, 5 * T**4], # dS(T) = 0
        [0, 0, 2, 0, 0, 0],                          # ddS(0) = 0
        [0, 0, 2, 6 * T, 12 * T**2, 20 * T**3],      # ddS(T) = 0
    ]) @ np.array([0, 1, 0, 0, 0, 0])

    def traj(t):
        s_t = a[0] + a[1] * t + a[2] * t**2 + a[3] * t**3 + a[4] * t**4 + a[5] * t**5
        ds_t = a[1] + 2 * a[2] * t + 3 * a[3] * t**2 + 4 * a[4] * t**3 + 5 * a[5] * t**4
        dds_t = a[2] + 6 * a[3] * t + 12 * a[4] * t**2 + 20 * a[5] * t**3

        q_t = q_initial + s_t * (q_final - q_initial)
        dq_t = (q_dot_final - q_dot_initial) * ds_t
        ddq_t = (q_ddot_final - q_ddot_initial) * dds_t
        return q_t, dq_t, ddq_t
    return traj

    #import matplotlib.pyplot as plt
    #xs = np.linspace(0,T,num=50)
    #ys = np.array([_s(x_t) for x_t in xs])
    #y_1 = ys[:,0]
    #y_2 = ys[:,1]
    #y_3 = ys[:,2]
    #plt.plot(xs, y_1, label='s(t)')
    #plt.plot(xs, y_2, label='s\'(t)')
    #plt.plot(xs, y_3, label='s\'\'(t)')
    #plt.legend()
    #plt.show()
    #_s(1)

class DoublePendulumSimulator:
    def __init__(self, equations_of_motion, initial_q, initial_q_dot):
        self.equations_of_motion = equations_of_motion
        self.q = initial_q
        self.q_dot = initial_q_dot
        self.t = 0

    def step(self, torques, dt=0.005):
        # Solve equations of motion for joint acceleration
        q_ddot = self.equations_of_motion(torques, self.q, self.q_dot)

        # Euler integration to find position and velocity
        self.q_dot = self.q_dot + dt * q_ddot
        self.q = self.q + dt * self.q_dot
        self.t += dt
        
        return self.q, self.q_dot

q_i = np.array([np.pi, np.pi/4])
q_f = np.array([np.pi/4, np.pi/2])
dq = np.zeros(2)
ddq = np.zeros(2)
T = 2
traj = generate_quintic_traj(q_i,dq,ddq,q_f,dq,ddq, 2)

qs = np.array([x for x,_,_ in [traj(t) for t in np.linspace(0, T, num=50)]])

#animate_pendulum(qs)

sim = DoublePendulumSimulator(pendulum_dynamics(), q_i, dq)
animate_pendulum(np.array([x for x, _ in [sim.step(np.zeros(2)) for _ in range(5000)]]))
