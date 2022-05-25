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
        + 0.5 * I_2 * m_2 * (sympy.diff(θ_1,t) + sympy.diff(θ_2, t)) ** 2

    # Potential Energy
    P = g * (m_1 * y_1 + m_2 * y_2)

    Lagrange = K - P

    from sympy.physics.mechanics import LagrangesMethod
    LM = LagrangesMethod(Lagrange, [θ_1, θ_2])
    
    full = LM.form_lagranges_equations()
    
    tau = sympy.Matrix([tau_1, tau_2])
    qdd = sympy.Matrix([sympy.diff(sympy.diff(θ_1, t), t), sympy.diff(sympy.diff(θ_2, t), t)])
    q_ddot = LM.mass_matrix.inv() * (LM.forcing + tau)
    ff_tau = LM.mass_matrix * qdd - LM.forcing

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
    ff_tau = ff_tau.subs(subs)

    from sympy.utilities.lambdify import lambdify
    q_ddot_fast  = lambdify(
        [θ_1, θ_2, sympy.diff(θ_1, t), sympy.diff(θ_2, t), tau_1, tau_2],
        q_ddot,
    )

    tau_ff_fast = lambdify(
        [θ_1, θ_2, sympy.diff(θ_1, t), sympy.diff(θ_2, t), sympy.diff(sympy.diff(θ_1, t), t), sympy.diff(sympy.diff(θ_2, t), t)],
        ff_tau,
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

    def compute_ff_torques(
            q,
            q_dot,
            q_ddot,
    ):
        """
        Helper function to calculate feedforward torques given concrete
        values for all the variables in the equations of motion
        """
        assert len(q) == 2
        assert len(q_dot) == 2
        assert len(q_ddot) == 2

        ret = tau_ff_fast(*q, *q_dot, *q_ddot)
        return np.array(ret).flatten()
        #return np.zeros(2)

    return compute_accelerations, compute_ff_torques

def animate_pendulum(q, L1: float = 1, L2: float = 1, ref=None):
    """
    Code for animating a trajectory of pendulum configurations. Can plot a
    second configuration trajectory if passed into 'ref'
    """
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

    if ref is not None:
        assert len(ref) == len(q)
        ref_base, = ax.plot(0, 0, marker="x")
        ref_p1, = ax.plot(0, 1, marker="x")
        ref_p2, = ax.plot(0, 1, marker="x")
        ref_link1, = ax.plot([], [], color="blue", zorder=4)
        ref_link2, = ax.plot([], [], color="blue", zorder=4)

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

        if ref is not None:
            ref_theta_1, ref_theta_2 = ref[t]

            ref_x_1 = np.sin(ref_theta_1) * L1
            ref_y_1 = -np.cos(ref_theta_1) * L1

            ref_x_2 = ref_x_1 + np.sin(ref_theta_1 + ref_theta_2) * L2
            ref_y_2 = ref_y_1 - np.cos(ref_theta_1 + ref_theta_2) * L2
            ref_p1.set_data([ref_x_1], [ref_y_1])
            ref_p2.set_data([ref_x_2], [ref_y_2])
            ref_link1.set_data([0, ref_x_1], [0, ref_y_1])
            ref_link2.set_data([ref_x_1, ref_x_2], [ref_y_1, ref_y_2])
            return p1, p2, link1, link2, ref_p1, ref_p2, ref_link1, ref_link2

        else:
            return p1, p2, link1, link2

    ani = animation.FuncAnimation(fig, update, interval=10, blit=True, repeat=True,
                        frames=np.linspace(0, len(q), num=len(q), endpoint=False))
    #writer = animation.writers['ffmpeg'](fps=100, metadata=dict(artist='Me'), bitrate=1800)
    #ani.save('traj.mp4', writer=writer)
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
        dq_t = (q_final - q_initial) * ds_t
        ddq_t = (q_final - q_initial) * dds_t
        return q_t, dq_t, ddq_t

    #import matplotlib.pyplot as plt
    #xs = np.linspace(0,T,num=50)
    #ys = np.array([traj(x_t) for x_t in xs])
    #y_1 = ys[:,0,0]
    #y_2 = ys[:,1,0]
    #y_3 = ys[:,2,0]
    #plt.plot(xs, y_1, label='s(t) θ_1')
    #plt.plot(xs, y_2, label='s\'(t) θ_1')
    #plt.plot(xs, y_3, label='s\'\'(t) θ_1')
    #plt.legend()
    #plt.show()

    #y_1 = ys[:,0,1]
    #y_2 = ys[:,1,1]
    #y_3 = ys[:,2,1]
    #plt.plot(xs, y_1, label='s(t) θ_2')
    #plt.plot(xs, y_2, label='s\'(t) θ_2')
    #plt.plot(xs, y_3, label='s\'\'(t) θ_2')
    #plt.legend()
    #plt.show()

    return traj

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
        self.q_ddot = q_ddot
        self.q_dot = self.q_dot + dt * q_ddot
        self.q = self.q + dt * self.q_dot
        self.t += dt
        
        return self.q, self.q_dot

class PID:
    """
    No-frills PID controller
    """
    def __init__(self, Kp, Ki, Kd):
        assert len(Kp) == len(Ki)
        assert len(Kp) == len(Kd)

        self.Kp = np.array(Kp)
        self.Ki = np.array(Ki)
        self.Kd = np.array(Kd)
        self.integral = np.zeros(self.Kp.shape)

    def step(self, error, velocity_error, dt):
        self.integral += error * dt
        return self.Kp * error + self.Ki * self.integral + self.Kd * velocity_error

# The starting configuration
q_i = np.array([-np.pi/4, 0])

# The ending configuration
q_f = np.array([np.pi/4, np.pi/2])

# The starting and ending joint velocities
dq = np.zeros(2)

# The starting and ending joint accelerations
ddq = np.zeros(2)

# The length of our trajectory in seconds
T = 2

# The function which gives us the pendulum configuration at time t \in [0, T]
traj = generate_quintic_traj(q_i, dq, ddq, q_f, dq, ddq, T)

# Create the pendulum dynamics simulator starting at our initial state
acceleration_dynamics, _ = pendulum_dynamics()
_, torque_ff = pendulum_dynamics(_m_1=1.2, _m_2=1.2)

sim = DoublePendulumSimulator(acceleration_dynamics, q_i, dq)

# Create a PID controller with fairly stiff proportional gains
pid = PID([200, 200], [0, 0], [60, 60])

dt = 0.005
real_traj = []
real_qacc = []
us = []
q_actual, q_dot_actual = q_i, dq
t = 0
for q_target, q_dot_target, q_ddot_target in [traj(t) for t in np.linspace(0, T, num=int(T/dt))]:
    err = q_target - q_actual
    vel_err = q_dot_target - q_dot_actual

    u = pid.step(err, vel_err, dt) + torque_ff(q_target, q_dot_target, q_ddot_target)

    q_actual, q_dot_actual = sim.step(u, dt=dt)

    real_qacc += [sim.q_ddot]
    real_traj += [q_actual]
    us += [u]
    t+=1

# The reference trajectory
qs = np.array([x for x,_,_ in [traj(t) for t in np.linspace(0, T, num=int(T/dt))]])
qdds = np.array([x for _,_,x in [traj(t) for t in np.linspace(0, T, num=int(T/dt))]])

import matplotlib.pyplot as plt
#plt.plot(np.array(real_qacc)[:,0], label='actual theta_ddot')
#plt.plot(qdds[:,0], label='reference theta_ddot')
#plt.legend()
#plt.show()

plt.plot(np.array(real_traj)[:,0], label='actual theta_1')
plt.plot(qs[:,0], label='reference theta_1')
plt.legend()
plt.show()

plt.plot(np.array(real_traj)[:,1], label='actual theta_2')
plt.plot(qs[:,1], label='reference theta_2')
plt.legend()
plt.show()

plt.plot(np.array(np.array(us)[:,0]), label='torque (joint 1)')
plt.plot(np.array(np.array(us)[:,1]), label='torque (joint 2)')
plt.legend()
plt.show()

# Plot the actual dynamic trajectory and the reference trajectory for comparison
animate_pendulum(np.array(real_traj), ref=qs)
