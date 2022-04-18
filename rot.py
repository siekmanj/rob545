import numpy as np

# Rotate about x (roll)
def R_x(o):
    R = np.array([
        [1, 0, 0],
        [0, np.cos(o), -np.sin(o)],
        [0, np.sin(o), np.cos(o)],
    ])
    return np.around(R, decimals=3)

# Rotate about y (pitch)
def R_y(o):
    R = np.array([
        [np.cos(o), 0, np.sin(o)],
        [0, 1, 0],
        [-np.sin(o), 0, np.cos(o)]
    ])
    return np.around(R, decimals=3)

# Rotate about z (yaw)
def R_z(o):
    R = np.array([
        [np.cos(o), -np.sin(o), 0],
        [np.sin(o), np.cos(o), 0],
        [0, 0, 1],
    ])
    return np.around(R, decimals=3)

# Create skew symmetric (so3) matrix from 3d vector v
def so3(v):
    assert len(v) == 3
    w = np.zeros((3,3))
    w[0][1] = -v[2]
    w[0][2] = v[1]
    w[1][2] = -v[0]
    return w - w.T

# Create homogenous transformation matrix from SO3 and translation vector
def T(R, p):
    t = np.zeros((4,4))
    t[:3,:3] = R
    t[:3,3] = np.array(p).T
    t[3] = [0, 0, 0, 1]
    return np.around(t, decimals=3)

# Create SE3 from a twist and angular displacement
def SE3(w, v, theta):
    w_skew = so3(w)

    e_twist_t = np.zeros((4,4))

    e_twist_t[:3, :3] = np.eye(3) + np.sin(theta) * w_skew + \
                        (1 - np.cos(theta)) * (w_skew @ w_skew)

    e_twist_t[:3, 3] = (np.eye(3) * theta + (1 - np.cos(theta)) * w_skew + \
                       (theta - np.sin(theta)) * (w_skew @ w_skew)) @ v
    e_twist_t[3, 3] = 1
    #return np.around(e_twist_t, decimals=3)
    return e_twist_t

# Create a 6x6 Adj representation of a 4x4 SE3 matrix for transforming
# twists
def adjoint(T):
    R = T[:3,:3]
    p = T[:3,-1]
    adj = np.zeros((6,6))
    adj[:3,:3] = R # Upper left is R
    adj[3:,3:] = R # Lower right is also R
    adj[3:,:3] = so3(p) @ R # Lower left is so3(p) mmul R
    return adj

# Create a 6d screw vector from a rotation axis `s`, the vector to any point
# along the rotation axis `q`, a rotation displacement `theta`, and velocity
# along the axis `pitch`
def screw(s, q, pitch, rotation: bool = False):
    pitch = float(pitch)
    q = np.array(q, dtype=np.float64)
    w = np.array(s, dtype=np.float64)

    if rotation:
        v = np.cross(-np.array(s), q) + pitch * np.array(s)
        v /= np.linalg.norm(w)
        w /= np.linalg.norm(w)
    else:
        v = np.array(w) * pitch
        v /= np.linalg.norm(v)
        w = np.zeros(3)
    return w, v

if True:
    # Problem 1 b
    M = T(np.eye(3), [0, 2, 1])
    print('Problem 1 initial transform:')
    print(M)

    theta1 = 0
    theta2 = np.pi/2
    theta3 = -np.pi/2
    theta4 = 1

    eS1 = SE3([0, 0, 1], [0, 0, 0], theta1)
    eS2 = SE3([0, 0, 1], [1, 0, 0], theta2)
    eS3 = SE3([0, 0, 1], [2, 0, 0], theta3)
    eS4 = SE3([0, 0, 0], [0, 0, 1], theta4)

    print('Problem 1 space screw result:')
    S = eS1 @ eS2 @ eS3 @ eS4 @ M
    print(S)

    eB1 = SE3([0, 0, 1], [-2, 0, 0,], theta1)
    eB2 = SE3([0, 0, 1], [-1, 0, 0,], theta2)
    eB3 = SE3([0, 0, 1], [0, 0, 0,], theta3)
    eB4 = SE3([0, 0, 0], [0, 0, 1,], theta4)

    print('Problem 1 body screw result:')
    B = M @ eB1 @ eB2 @ eB3 @ eB4
    print(B)

if True:
    # Problem 2
    M = T(R_x(np.pi/2) @ R_y(np.pi), [0, 3, 2])
    print('Problem 2 initial transform M:')
    print(M)

    theta1 = np.pi/2
    theta2 = np.pi/2
    theta3 = 1
    S1 = screw([0, 0, 1], [0, 0, 0], 0, True)
    S2 = screw([1, 0, 0], [0, 0, 2], 0, True)
    S3 = screw([0, 1, 0], [0, 0, 2], 1, False)

    eS1 = SE3(*S1, theta1)
    eS2 = SE3(*S2, theta2)
    eS3 = SE3(*S3, theta3)

    print('Problem 2 space screw result:')
    S = eS1 @ eS2 @ eS3 @ M
    print(np.around(S))

    B1 = screw([0, 1, 0], [0, 0, -3], 0, True)
    B2 = screw([-1, 0, 0], [0, 0, -3], 0, True)
    B3 = screw([0, 0, 1], [0, 0, 1], 1, False)
    eB1 = SE3(*B1, theta1)
    eB2 = SE3(*B2, theta2)
    eB3 = SE3(*B3, theta3)

    print('Problem 2 body screw result:')
    B = M @ eB1 @ eB2 @ eB3
    print(np.around(B))

    print("Space frame Jacobian:")
    JS1 = np.hstack(S1)
    JS2 = adjoint(eS1) @ np.hstack(S2)
    JS3 = adjoint(eS1 @ eS2) @ np.hstack(S3)
    JS = np.vstack([JS1, JS2, JS3]).T
    print(np.around(JS))

    print("Body frame Jacobian:")
    JB3 = np.hstack(B3)
    JB2 = adjoint(np.linalg.inv(eB3)) @ np.hstack(B2)
    JB1 = adjoint(np.linalg.inv(eB2 @ eB3)) @ np.hstack(B1)
    JB = np.vstack([JB1, JB2, JB3]).T
    print(np.around(JB))
