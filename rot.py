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

if False:
    # Problem 1 b
    M = T(np.eye(3), [0, 2, 1])

    theta1 = 0
    theta2 = np.pi/2
    theta3 = -np.pi/2
    theta4 = 1

    eS1 = SE3([0, 0, 1], [0, 0, 0], theta1)
    eS2 = SE3([0, 0, 1], [1, 0, 0], theta2)
    eS3 = SE3([0, 0, 1], [2, 0, 0], theta3)
    eS4 = SE3([0, 0, 0], [0, 0, 1], theta4)

    print('Space screw result:')
    S = eS1 @ eS2 @ eS3 @ eS4 @ M
    print(S)

    eB1 = SE3([0, 0, 1], [-2, 0, 0,], theta1)
    eB2 = SE3([0, 0, 1], [-1, 0, 0,], theta2)
    eB3 = SE3([0, 0, 1], [0, 0, 0,], theta3)
    eB4 = SE3([0, 0, 0], [0, 0, 1,], theta4)

    print('Body screw result:')
    B = M @ eB1 @ eB2 @ eB3 @ eB4
    print(B)

if True:
    # Problem 2
    M = T(R_x(np.pi/2) @ R_y(np.pi), [0, 3, 2])
    print(M)

    theta1 = np.pi/2
    theta2 = np.pi/2
    theta3 = 1

    eS1 = SE3([0, 0, 1], [0, 0, 0], theta1)
    eS2 = SE3([1, 0, 0], [0, 2, 0], theta2)
    eS3 = SE3([0, 0, 0], [0, 1, 0], theta3)

    print('Space screw result:')
    S = eS1 @ eS2 @ eS3 @ M
    print(np.around(S))

    eB1 = SE3([0, 1, 0], [3, 0, 0,], theta1)
    eB2 = SE3([-1, 0, 0], [0, 3, 0,], theta2)
    eB3 = SE3([0, 0, 0], [0, 0, 1], theta3)

    print('Body screw result:')
    B = M @ eB1 @ eB2 @ eB3
    print(np.around(B))


exit(0)

def screw(q, s, theta, pitch):
    w = np.array(s) * theta
    v = np.cross(-np.array(s) * theta, q) + pitch * np.array(s) * theta

    if np.sum(np.abs(v[:3])) > 1e-6:
        v /= np.linalg.norm(w[:3])
        w /= np.linalg.norm(w[:3])
    else:
        raise NotImplementedError(q,s,theta,pitch)
    return w, v

# Problem 3.24
t = 4

q1 = np.zeros(3)
s1 = [0, 0, 1] # Rotate about z
theta1 = np.pi/4
S1_w, S1_v = screw(q1, s1, theta1, 2)
e_S1 = SE3(S1_w, S1_v, t*theta1)
print(e_S1)

q2 = np.array([0, 0, 10])
s2 = [0, 1, 0]
theta2 = np.pi/8
S2_w, S2_v = screw(q2, s2, theta2, 0)
e_S2 = SE3(S2_2, S2_v, t*theta2)
print(e_S2)

q3 = np.array([0, 5, 5])
s3 = [1, 0, 0]
theta3 = np.pi/4
S3_w, S3_v = screw(q3, s3, theta3, 0)
e_S3 = SE3(S3_w, S3_v, t*theta3)
print(e_S3)

T_sb = T(R_z(np.pi/2), [0, 8, 5])
print(T_sb)

T_sb_prime = e_S1 @ e_S2 @ e_S3 @ T_sb
print('final:', T_sb_prime)
